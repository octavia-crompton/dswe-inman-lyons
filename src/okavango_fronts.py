"""
Okavango flood-front analysis utilities.

Canonical, deduplicated versions of the helpers used by the
CYGNSS front-normal velocity notebook.
"""
from __future__ import annotations

import calendar
import re
from pathlib import Path
from typing import Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import TwoSlopeNorm
from pyproj import Transformer
from shapely.geometry import LineString, MultiLineString, Point, box
from shapely.ops import linemerge

# These module-level sentinels mirror the notebook's config constants.
# They are set at import time but can be overridden by callers.
WATER_CLASSES: set = {1, 2, 3, 4}
WATER_THRESHOLD: float | None = None
STRICT_TIME_FROM_FILENAME: bool = True
TIME_REGEX: str = r"(\d{4})[\-_](\d{2})[\-_](\d{2})"
ENGINE_TRY: tuple = ("h5netcdf", "netcdf4", "scipy")


def _choose_dims(da):
    # Identify time + 2 spatial dims (robust to names: time/lat/lon vs y/x)
    time_dim = next((d for d in da.dims if d.lower() == "time"), None)
    if time_dim is None:
        raise ValueError("Expected a 'time' dimension in the DataArray.")
    spatial_dims = [d for d in da.dims if d != time_dim]
    if len(spatial_dims) != 2:
        raise ValueError(f"Expected exactly 2 spatial dims, found {spatial_dims}.")
    # Prefer lat/lon naming if present
    y_dim = next((d for d in spatial_dims if "lat" in d.lower()), spatial_dims[0])
    x_dim = next((d for d in spatial_dims if "lon" in d.lower()), spatial_dims[1])
    return time_dim, y_dim, x_dim


def _choose_spatial_dims(obj):
    dims = list(obj.dims)
    # try common names first
    y_dim = next((d for d in dims if "lat" in d.lower() or d.lower() in ("y","row")), None)
    x_dim = next((d for d in dims if "lon" in d.lower() or d.lower() in ("x","col")), None)
    if y_dim is None or x_dim is None:
        # fallback: take the last two dims in order
        y_dim, x_dim = dims[-2], dims[-1]
    return y_dim, x_dim


def _collect_channel_samples(gdf_m: gpd.GeoDataFrame, step_m: float, apex_xy: np.ndarray | None):
    """
    Densify all lines and return arrays of sample point coords and unit tangents.
    Returns: xs, ys, txs, tys  (all 1D np.ndarrays)
    """
    xs, ys, txs, tys = [], [], [], []

    for geom in gdf_m.geometry:
        if geom is None:
            continue
        # Iterate over LineStrings
        if isinstance(geom, MultiLineString):
            lines = list(geom.geoms)
        else:
            lines = [geom]

        for line in lines:
            if line.length == 0:
                continue

            xy, s = _densify_line(line, step_m)
            # Orient outward if apex given
            if apex_xy is not None:
                xy = _orient_line_outward(xy, apex_xy)

            t = _tangent_vectors(xy)  # shape (n,2)
            valid = np.isfinite(t[:,0]) & np.isfinite(t[:,1])
            if not np.any(valid):
                continue

            xs.extend(xy[valid, 0])
            ys.extend(xy[valid, 1])
            txs.extend(t[valid, 0])
            tys.extend(t[valid, 1])

    if len(xs) == 0:
        raise ValueError("No valid line samples found in channel GeoDataFrame.")
    return np.asarray(xs), np.asarray(ys), np.asarray(txs), np.asarray(tys)


def _densify_line(line: LineString, step_m: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return equally spaced points (x,y) and chainage s (m) along a projected LineString."""
    L = line.length
    if L == 0:
        xy = np.array(line.coords[:1])
        s = np.array([0.0])
        return xy, s
    n = max(1, int(np.ceil(L/step_m)))
    s = np.linspace(0.0, L, n+1)
    pts = [line.interpolate(dist) for dist in s]
    xy = np.array([(p.x, p.y) for p in pts])
    return xy, s


def _ensure_time_coord(da: xr.DataArray, path: Path):
    """Guarantee a 'time' dimension. If missing and STRICT_TIME_FROM_FILENAME, infer 1-step time from filename."""
    if "time" in da.dims:
        return da
    if STRICT_TIME_FROM_FILENAME:
        m = re.search(TIME_REGEX, path.name)
        if not m:
            raise RuntimeError(f"No time in dims and couldn't parse time from filename: {path.name}")
        y, mth, d = map(int, m.groups())
        # create a singleton time dimension
        da = da.expand_dims(time=[np.datetime64(f"{y:04d}-{mth:02d}-{d:02d}")])
        return da
    # else: add a dummy index to allow concat; you can later replace with real times
    da = da.expand_dims(time=[np.datetime64("NaT")])
    return da


def _find_lat_lon_names(ds: xr.Dataset):
    lat_candidates = [n for n in ds.coords if n.lower() in ("lat", "latitude", "y")]
    lon_candidates = [n for n in ds.coords if n.lower() in ("lon", "longitude", "x")]
    if not lat_candidates or not lon_candidates:
        # try variables if not in coords
        lat_candidates = lat_candidates or [n for n in ds.variables if n.lower() in ("lat", "latitude", "y")]
        lon_candidates = lon_candidates or [n for n in ds.variables if n.lower() in ("lon", "longitude", "x")]
    if not lat_candidates or not lon_candidates:
        raise RuntimeError("Could not infer lat/lon names.")
    return lat_candidates[0], lon_candidates[0]


def _infer_latlon_names(ds: xr.Dataset) -> Tuple[str, str]:
    for lat_name in ("lat","latitude","y"):
        if lat_name in ds.coords:
            break
    else:
        raise ValueError("Could not find a latitude coordinate in ds_perp (looked for lat/latitude/y).")
    for lon_name in ("lon","longitude","x"):
        if lon_name in ds.coords:
            break
    else:
        raise ValueError("Could not find a longitude coordinate in ds_perp (looked for lon/longitude/x).")
    return lat_name, lon_name


def _infer_latlon_names_in_ds(ds: xr.Dataset):
    for lat_name in ("lat","latitude","y"):
        if lat_name in ds.coords:
            break
    else:
        raise ValueError("Could not find latitude coord in ds (looked for lat/latitude/y).")
    for lon_name in ("lon","longitude","x"):
        if lon_name in ds.coords:
            break
    else:
        raise ValueError("Could not find longitude coord in ds (looked for lon/longitude/x).")
    return lat_name, lon_name


def _infer_mask(da0: xr.DataArray):
    """Return a boolean DataArray mask of 'wet' pixels based on data heuristics."""
    # Compute quick stats on the first valid slice to infer type
    # (use .max() over a small slice to avoid large computations)
    sample = da0.isel(time=0)
    # if it's all-NaN, find a non-NaN slice
    for i in range(da0.sizes["time"]):
        tmp = da0.isel(time=i)
        if np.isfinite(tmp.values).any():
            sample = tmp
            break

    # Heuristic 1: DSWE-like integer classes
    try:
        is_int = np.issubdtype(sample.dtype, np.integer)
    except Exception:
        is_int = False
    vmax = np.nanmax(sample.values)
    vmin = np.nanmin(sample.values)

    if is_int and vmax <= 5:
        # Treat DSWE 1–4 as water
        return da0.isin(list(WATER_CLASSES))

    # Heuristic 2: 0–1 floats (probability/fraction)
    if np.issubdtype(sample.dtype, np.floating) and (0 <= vmin <= 1) and (0 <= vmax <= 1.5):
        thr = 0.5 if WATER_THRESHOLD is None else WATER_THRESHOLD
        return da0 > thr

    # Fallback: any finite pixel counts toward "extent"
    return xr.ufuncs.isfinite(da0)


def _latlon_1d(da2d, y_dim, x_dim):
    """
    Return 1D lat (along y) and lon (along x) coordinate arrays.
    Handles common cases where 'lat'/'lon' are 1D or 2D coords.
    """
    if "lat" in da2d.coords:
        latc = da2d["lat"]
    elif y_dim in da2d.coords:
        latc = da2d[y_dim]
    else:
        raise ValueError("Could not find latitude coordinates ('lat' or y-dim).")

    if "lon" in da2d.coords:
        lonc = da2d["lon"]
    elif x_dim in da2d.coords:
        lonc = da2d[x_dim]
    else:
        raise ValueError("Could not find longitude coordinates ('lon' or x-dim).")

    # Make them 1D along each axis
    if latc.ndim == 2:
        lat_1d = latc.mean(x_dim).values
    else:
        lat_1d = latc.values

    if lonc.ndim == 2:
        lon_1d = lonc.mean(y_dim).values
    else:
        lon_1d = lonc.values

    return np.asarray(lat_1d), np.asarray(lon_1d)


def _latlon_mesh(ds, y_dim, x_dim):
    # Retrieve lat/lon coordinates as 2D arrays
    if "lat" in ds.coords:
        latc = ds["lat"]
    else:
        latc = ds.coords.get(y_dim)
    if "lon" in ds.coords:
        lonc = ds["lon"]
    else:
        lonc = ds.coords.get(x_dim)
    if latc is None or lonc is None:
        raise ValueError("Could not find latitude/longitude coordinates in dataset.")

    if latc.ndim == 1 and lonc.ndim == 1:
        lat2d, lon2d = np.meshgrid(latc.values, lonc.values, indexing="ij")
    else:
        lat2d, lon2d = np.asarray(latc), np.asarray(lonc)
    return lat2d, lon2d


def _local_utm_epsg(mean_lat: float, mean_lon: float) -> int:
    zone = int(np.floor((mean_lon + 180.0) / 6.0) + 1)
    return (32600 if mean_lat >= 0 else 32700) + zone


def _normalize_dims(da: xr.DataArray, time_name: str, lat_name: str, lon_name: str):
    # rename to canonical
    rename_map = {}
    if time_name != "time" and time_name in da.dims:
        rename_map[time_name] = "time"
    if lat_name != "lat" and lat_name in da.dims:
        rename_map[lat_name] = "lat"
    if lon_name != "lon" and lon_name in da.dims:
        rename_map[lon_name] = "lon"
    if rename_map:
        da = da.rename(rename_map)
    # transpose to (time, lat, lon)
    want = ("time", "lat", "lon")
    missing = [d for d in want if d not in da.dims]
    if missing:
        raise RuntimeError(f"Missing expected dims after rename: {missing}")
    return da.transpose(*want)


def _open_dataset_robust(p: Path):
    last_err = None
    for eng in ENGINE_TRY:
        try:
            return xr.open_dataset(p, engine=eng, decode_times=True, mask_and_scale=True)
        except Exception as e:
            last_err = e
    raise last_err


def _orient_line_outward(xy: np.ndarray, apex_xy: np.ndarray) -> np.ndarray:
    """Flip line so its median tangent points away from apex."""
    if len(xy) < 2:
        return xy
    t = _tangent_vectors(xy)
    i = len(xy)//2
    t_mid = t[i]
    v_out = xy[i] - apex_xy     # vector from apex to midpoint
    if np.dot(t_mid, v_out) < 0:
        return xy[::-1]          # flip direction
    return xy


def _pick_var_with_latlon(ds: xr.Dataset, lat_name: str, lon_name: str):
    # choose the first data var that has both lat & lon dims
    for v in ds.data_vars:
        dims = tuple(ds[v].dims)
        if lat_name in dims and lon_name in dims:
            return v
    # fallback: largest 2D/3D var
    candidates = sorted(ds.data_vars, key=lambda v: ds[v].size, reverse=True)
    if not candidates:
        raise RuntimeError("Dataset contains no data variables.")
    return candidates[0]


def _sample_fields_at_points(ds_perp: xr.Dataset, lat: np.ndarray, lon: np.ndarray, vars=("u_perp","v_perp","mask_front")) -> xr.Dataset:
    lat_name, lon_name = _infer_latlon_names(ds_perp)
    pts = xr.Dataset({lat_name: (("points",), lat), lon_name: (("points",), lon)})
    return ds_perp[list(vars)].interp({lat_name: pts[lat_name], lon_name: pts[lon_name]})


def _sample_fields_at_points_pairwise(
    ds_perp: xr.Dataset,
    lat: np.ndarray,
    lon: np.ndarray,
    vars=("u_perp","v_perp","mask_front"),
    eps: float = 1e-12
) -> xr.Dataset:
    """
    Pairwise sampling of (lat_i, lon_i) with bilinear interpolation on a rectilinear grid.
    Returns a Dataset with variables shaped (points,). Boolean masks use nearest-neighbor.
    """
    lat = np.asarray(lat).astype(float)
    lon = np.asarray(lon).astype(float)
    if lat.shape != lon.shape:
        raise ValueError(f"lat and lon must have the same shape; got {lat.shape} vs {lon.shape}")

    lat_name, lon_name = _infer_latlon_names(ds_perp)
    y_axis = np.asarray(ds_perp[lat_name].values)
    x_axis = np.asarray(ds_perp[lon_name].values)
    if y_axis.ndim != 1 or x_axis.ndim != 1:
        raise ValueError("This sampler expects 1D lat/lon coordinates (rectilinear grid).")

    # Ensure ascending axes for searchsorted; remember if we flipped
    y_asc = y_axis[0] <= y_axis[-1]
    x_asc = x_axis[0] <= x_axis[-1]
    y_use = y_axis if y_asc else y_axis[::-1]
    x_use = x_axis if x_asc else x_axis[::-1]

    Ny, Nx = len(y_use), len(x_use)

    # Bracketing indices
    iy = np.searchsorted(y_use, lat) - 1
    ix = np.searchsorted(x_use, lon) - 1
    oob = (iy < 0) | (ix < 0) | (iy >= Ny-1) | (ix >= Nx-1)

    iy = np.clip(iy, 0, Ny-2)
    ix = np.clip(ix, 0, Nx-2)

    y0 = y_use[iy]; y1 = y_use[iy+1]
    x0 = x_use[ix]; x1 = x_use[ix+1]
    wy = (lat - y0) / np.maximum(y1 - y0, eps)
    wx = (lon - x0) / np.maximum(x1 - x0, eps)
    wy = np.clip(wy, 0.0, 1.0)
    wx = np.clip(wx, 0.0, 1.0)

    data = {}
    for name in vars:
        A = ds_perp[name].transpose(lat_name, lon_name).values  # ensure (lat, lon) order
        if not y_asc:
            A = A[::-1, :]
        if not x_asc:
            A = A[:, ::-1]

        if name == "mask_front" or A.dtype == bool:
            # nearest neighbor for boolean mask
            iy0 = iy; iy1 = iy + 1
            ix0 = ix; ix1 = ix + 1
            choose_iy = np.where(np.abs(lat - y0) <= np.abs(y1 - lat), iy0, iy1)
            choose_ix = np.where(np.abs(lon - x0) <= np.abs(x1 - lon), ix0, ix1)
            val = A[choose_iy, choose_ix].astype(bool)
            val = np.where(oob, False, val)
        else:
            # bilinear for continuous vars
            q11 = A[iy,   ix  ]
            q21 = A[iy+1, ix  ]
            q12 = A[iy,   ix+1]
            q22 = A[iy+1, ix+1]
            val = (q11*(1-wy)*(1-wx) +
                   q21*(   wy)*(1-wx) +
                   q12*(1-wy)*(   wx) +
                   q22*(   wy)*(   wx))
            val = np.where(oob, np.nan, val)

        data[name] = val

    return xr.Dataset(
        {k: (("points",), v) for k, v in data.items()},
        coords={"points": np.arange(lat.size)}
    )


def _spatial_gradients_meters(Fmid, y_dim, x_dim):
    """
    Compute ∂F/∂x and ∂F/∂y in *per meter* using geographic coords.
    Fmid is a 2D DataArray (no time dim).
    """
    lat_1d, lon_1d = _latlon_1d(Fmid, y_dim, x_dim)

    # Gradients w.r.t. degrees
    dF_dlat, dF_dlon = np.gradient(
        Fmid.values, lat_1d, lon_1d, edge_order=2
    )  # axis0=y(lat), axis1=x(lon)

    # Convert degrees -> meters (Okavango ~ -19°, but do it properly per row)
    # Mean meters/degree (sufficient for ~regional scales)
    M_PER_DEG_LAT = 111_132.0  # meters per degree latitude
    # Lon meters per degree varies with latitude:
    lat2d = np.repeat(lat_1d[:, None], len(lon_1d), axis=1)
    M_PER_DEG_LON = 111_320.0 * np.cos(np.deg2rad(lat2d))
    M_PER_DEG_LON = np.maximum(M_PER_DEG_LON, 1e-9)  # guard near poles

    dF_dy_m = dF_dlat / M_PER_DEG_LAT        # north–south derivative per meter
    dF_dx_m = dF_dlon / M_PER_DEG_LON        # east–west derivative per meter

    return dF_dx_m, dF_dy_m


def _tangent_vectors(xy: np.ndarray) -> np.ndarray:
    """Centered differences for unit tangents at each point (projected meters)."""
    if len(xy) == 1:
        return np.array([[np.nan, np.nan]])
    d = np.empty_like(xy, dtype=float)
    d[1:-1] = xy[2:] - xy[:-2]
    d[0]  = xy[1] - xy[0]
    d[-1] = xy[-1] - xy[-2]
    norm = np.hypot(d[:,0], d[:,1])
    norm = np.where(norm == 0, np.nan, norm)
    t = d / norm[:,None]
    return t


def _to_local_meters(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError("Channel GeoDataFrame has no CRS. Set gdf.crs to EPSG:4326 or a projected CRS.")
    if gdf.crs.to_string().lower() in ("epsg:4326","wgs84") or gdf.crs.is_geographic:
        # choose a local UTM from dataset extent
        bb = gdf.total_bounds  # [minx, miny, maxx, maxy] in degrees
        mean_lon = 0.5*(bb[0]+bb[2])
        mean_lat = 0.5*(bb[1]+bb[3])
        epsg = _local_utm_epsg(mean_lat, mean_lon)
        return gdf.to_crs(epsg=epsg)
    return gdf  # already metric


def _to_monthly_mean(da: xr.DataArray) -> xr.DataArray:
    """Monthly mean (time='MS'); keeps coords and attrs."""
    if "time" not in da.dims:
        raise ValueError("This DataArray has no 'time' dimension.")
    dam = da.resample(time="MS").mean(keep_attrs=True)
    dam.name = (da.name or "var") + "_monthly"
    return dam


def area_weighted_mean_by_mask(da: xr.DataArray, mask2d: xr.DataArray) -> xr.DataArray:
    """
    Weighted mean over a boolean mask for each timestep.
    da:   (time, lat, lon)
    mask: (lat, lon) -> True inside region
    """
    time_name, lat_name, lon_name = da.dims
    w1d = coslat_weights(da[lat_name])
    w2d = w1d.broadcast_like(da.isel({time_name: 0}))
    num = (da.where(mask2d) * w2d).sum(dim=(lat_name, lon_name), skipna=True)
    den = (w2d.where(mask2d)).sum(dim=(lat_name, lon_name), skipna=True)
    return num / den


def by_region_timeseries(da: xr.DataArray, mask, names: list[str]) -> pd.DataFrame:
    """Compute area-weighted time series for each region name using the integer mask."""
    time_name, lat_name, lon_name = da.dims
    w1d = coslat_weights(da[lat_name])
    w2d = w1d.broadcast_like(da.isel({time_name: 0}))

    out = {}
    for i, name in enumerate(names):
        mask2d = (mask == i)
        num = (da.where(mask2d) * w2d).sum(dim=(lat_name, lon_name), skipna=True)
        den = (w2d.where(mask2d)).sum(dim=(lat_name, lon_name), skipna=True)
        out[name] = (num / den).to_pandas()
    df = pd.DataFrame(out)
    df.index.name = "time"
    return df


def classify_front_direction(
    ds_perp: xr.Dataset,
    *,
    angle_tol_deg: float = 30.0,   # within ±30° counts as "directed that way"
    min_speed_m_per_day: float = 0.0,  # ignore tiny/unstable vectors
    apex: tuple | None = None      # e.g., apex=(-19.0, 22.4)  (lat, lon)
) -> xr.Dataset:
    """
    Decompose the perpendicular motion into:
      (a) a fixed south-east (SE) direction, and
      (b) optionally, the radial direction 'outwards from apex'.

    Inputs
    ------
    ds_perp : Dataset with variables u_perp, v_perp (m/day), mask_front (bool).
    angle_tol_deg : angular tolerance to call a pixel "aligned".
    min_speed_m_per_day : speed floor to avoid noisy angles.
    apex : (lat, lon) of delta apex; if provided, adds 'outwards from apex' metrics.

    Returns
    -------
    ds_out : Dataset with v_SE, angle_to_SE, aligned_SE, and (if apex) v_outward_apex,
             angle_to_outward, aligned_outward. Carries through mask_front.
    """
    y_dim, x_dim = _choose_spatial_dims(ds_perp["u_perp"])
    u = ds_perp["u_perp"].values
    v = ds_perp["v_perp"].values
    speed = np.hypot(u, v)
    front_mask = np.asarray(ds_perp["mask_front"].values, dtype=bool)

    tiny = 1e-12

    # --- Fixed SE direction (east+, north-) ---
    se_unit = np.array([1.0/np.sqrt(2.0), -1.0/np.sqrt(2.0)])  # (east, north)
    dot_se = u*se_unit[0] + v*se_unit[1]           # m/day, signed component along SE
    cos_se = dot_se / np.maximum(speed, tiny)
    angle_se = np.degrees(np.arccos(np.clip(cos_se, -1.0, 1.0)))  # 0° = exactly SE
    aligned_se = (
        front_mask
        & (speed >= min_speed_m_per_day)
        & np.isfinite(angle_se)
        & (angle_se <= angle_tol_deg)
    )

    data_vars = {
        "v_SE":       ((y_dim, x_dim), np.where(front_mask, dot_se, np.nan)),
        "angle_to_SE":((y_dim, x_dim), np.where(front_mask, angle_se, np.nan)),
        "aligned_SE": ((y_dim, x_dim), aligned_se),
        "mask_front": ((y_dim, x_dim), front_mask),
    }

    # Optional: radial "outwards from apex"
    if apex is not None:
        lat0, lon0 = apex
        lat2d, lon2d = _latlon_mesh(ds_perp, y_dim, x_dim)

        # Convert delta lon/lat to meters (east/north). Use row-wise meters/deg for lon.
        M_PER_DEG_LAT = 111_132.0
        M_PER_DEG_LON = 111_320.0 * np.cos(np.deg2rad(lat2d))
        dx_m = (lon2d - lon0) * M_PER_DEG_LON   # east offset from apex
        dy_m = (lat2d - lat0) * M_PER_DEG_LAT   # north offset from apex
        rmag = np.hypot(dx_m, dy_m)
        rx = dx_m / np.maximum(rmag, tiny)
        ry = dy_m / np.maximum(rmag, tiny)

        dot_out = u*rx + v*ry     # m/day, positive = moving away from apex
        cos_out = dot_out / np.maximum(speed, tiny)
        ang_out = np.degrees(np.arccos(np.clip(cos_out, -1.0, 1.0)))  # 0° = exactly outward
        aligned_out = (
            front_mask
            & (speed >= min_speed_m_per_day)
            & np.isfinite(ang_out)
            & (ang_out <= angle_tol_deg)
        )

        data_vars.update({
            "v_outward_apex":   ((y_dim, x_dim), np.where(front_mask, dot_out, np.nan)),
            "angle_to_outward": ((y_dim, x_dim), np.where(front_mask, ang_out, np.nan)),
            "aligned_outward":  ((y_dim, x_dim), aligned_out),
        })

    ds_out = xr.Dataset(
        data_vars=data_vars,
        coords={y_dim: ds_perp[y_dim], x_dim: ds_perp[x_dim]},
        attrs={
            "angle_tol_deg": angle_tol_deg,
            "min_speed_m_per_day": min_speed_m_per_day,
            "note": "Angles compare the perpendicular motion vector (u_perp, v_perp) to SE or outward-from-apex directions."
        }
    )

    # Quick summary stats as attributes
    total_front = np.sum(front_mask)
    def _pct(mask): 
        n = np.sum(mask) if mask is not None else 0
        return float(100.0 * n / max(total_front, 1))

    summary = {
        "front_pixels": int(total_front),
        "pct_aligned_SE": _pct(ds_out["aligned_SE"].values),
        "median_v_SE_m_per_day": float(np.nanmedian(ds_out["v_SE"].values)),
    }
    if "aligned_outward" in ds_out:
        summary.update({
            "pct_aligned_outward": _pct(ds_out["aligned_outward"].values),
            "median_v_outward_m_per_day": float(np.nanmedian(ds_out["v_outward_apex"].values)),
        })
    ds_out.attrs["summary"] = summary
    return ds_out


def coslat_weights(lat: xr.DataArray) -> xr.DataArray:
    """Return 1D cos(latitude) weights as an xarray DataArray."""
    return xr.DataArray(np.cos(np.deg2rad(lat.values)), coords={lat.dims[0]: lat}, dims=lat.dims)


def find_lat_lon_names(ds: xr.Dataset) -> Tuple[str, str]:
    lat = next(n for n in ds.dims if n.lower() in ("lat", "latitude", "y"))
    lon = next(n for n in ds.dims if n.lower() in ("lon", "longitude", "x"))
    return lat, lon


def front_normal_velocity(
    da: xr.DataArray,
    t1,
    t2,
    *,
    front_value: float | None = None,   # e.g., 0.5 if F is 0–1 wetness
    bandwidth: float = 0.05,            # +/- band around front_value
    grad_quantile: float = 0.90,        # keep top q gradient as "front" if front_value is None
    smooth_px: int = 0,                 # optional rolling mean window (pixels) to stabilize grads
    grad_eps: float = 1e-12
) -> xr.Dataset:
    """
    Compute front-normal velocity field between two dates for a (time, y, x) DataArray.

    Returns a Dataset with:
    - v_normal (m/day): speed perpendicular to the front (sign indicates direction)
    - u_perp, v_perp (m/day): eastward/northward components of the perpendicular motion
    - n_x, n_y: unit normal components
    - grad_mag, dF_dt: helpers
    - mask_front: boolean mask where the estimate is reliable/meaningful (front zone)
    """

    time_dim, y_dim, x_dim = _choose_dims(da)

    F1 = da.sel({time_dim: pd.to_datetime(t1)}, method="nearest")
    F2 = da.sel({time_dim: pd.to_datetime(t2)}, method="nearest")
    t1_real = pd.to_datetime(F1[time_dim].item())
    t2_real = pd.to_datetime(F2[time_dim].item())
    if t1_real == t2_real:
        raise ValueError("The two dates resolve to the same snapshot; choose different dates.")

    dt_days = (t2_real - t1_real) / pd.Timedelta(days=1)

    # Optional smoothing to stabilize spatial gradients (box/rolling; avoids SciPy dependency)
    def _smooth2(F):
        if smooth_px and smooth_px > 1:
            return (
                F.rolling({y_dim: smooth_px, x_dim: smooth_px}, center=True)
                 .mean()
                 .astype(F.dtype)
            )
        return F

    F1s = _smooth2(F1)
    F2s = _smooth2(F2)
    Fmid = 0.5 * (F1s + F2s)

    # ∂F/∂t (per day)
    dF_dt = (F2s - F1s) / dt_days

    # ∇F in meters^-1 from mid-snapshot
    dF_dx_m, dF_dy_m = _spatial_gradients_meters(Fmid, y_dim, x_dim)
    grad_mag = np.hypot(dF_dx_m, dF_dy_m)

    # Unit normal (points toward increasing F)
    denom = np.maximum(grad_mag, grad_eps)
    n_x = dF_dx_m / denom
    n_y = dF_dy_m / denom

    # Front-normal speed v_n (m/day). Negative means the front moves *toward lower F*
    # (typical for a wetting front encroaching into a drier region if F increases behind the front).
    v_normal = -(dF_dt.values) / denom

    # Perpendicular motion vector components (east/north), m/day
    u_perp = v_normal * n_x
    v_perp = v_normal * n_y

    # FRONT MASK: either a level-set band around 'front_value', or "high-gradient" band
    if front_value is not None:
        band = np.abs(Fmid.values - front_value) <= bandwidth
    else:
        # keep pixels with strong gradients (likely the front zone)
        thresh = np.nanquantile(grad_mag, grad_quantile)
        band = grad_mag >= thresh

    # Also require non-tiny gradients where division is well-conditioned
    band &= grad_mag > (np.nanmedian(grad_mag) * 1e-3)

    # Pack into Dataset with coords/attrs
    ds = xr.Dataset(
        {
            "v_normal": ( (y_dim, x_dim), np.where(band, v_normal, np.nan) ),
            "u_perp":   ( (y_dim, x_dim), np.where(band, u_perp,   np.nan) ),
            "v_perp":   ( (y_dim, x_dim), np.where(band, v_perp,   np.nan) ),
            "n_x":      ( (y_dim, x_dim), np.where(band, n_x,      np.nan) ),
            "n_y":      ( (y_dim, x_dim), np.where(band, n_y,      np.nan) ),
            "grad_mag": ( (y_dim, x_dim), grad_mag ),
            "dF_dt":    ( (y_dim, x_dim), dF_dt.values ),
            "mask_front": ( (y_dim, x_dim), band.astype(bool) ),
        },
        coords={y_dim: Fmid[y_dim], x_dim: Fmid[x_dim]},
        attrs={
            "t1": str(t1_real),
            "t2": str(t2_real),
            "delta_days": float(dt_days),
            "method": "level-set normal velocity v_n = -(dF/dt)/|∇F| using mid-snapshot gradients"
        }
    )

    # Units metadata
    var_units = da.attrs.get("units", "")
    ds["dF_dt"].attrs.update({
        "long_name": f"time derivative {t1_real}→{t2_real}",
        "units": f"{var_units}/day" if var_units else "per_day",
    })
    ds["grad_mag"].attrs.update({
        "long_name": "spatial gradient magnitude",
        "units": f"{var_units}/m" if var_units else "per_meter",
    })
    ds["v_normal"].attrs.update({
        "long_name": f"front-normal speed between {t1_real} and {t2_real}",
        "units": "m/day",
        "note": "negative means moving toward lower F; positive toward higher F"
    })
    ds["u_perp"].attrs.update({"units": "m/day", "long_name": "eastward perpendicular component"})
    ds["v_perp"].attrs.update({"units": "m/day", "long_name": "northward perpendicular component"})
    ds["n_x"].attrs.update({"units": "1", "long_name": "unit normal (east)"})
    ds["n_y"].attrs.update({"units": "1", "long_name": "unit normal (north)"})
    ds["mask_front"].attrs.update({"long_name": "front zone mask (threshold/gradient-based)"})

    return ds


def front_normal_velocity_monthly_climatology(
    da: xr.DataArray,
    *,
    front_value: float | None = None,
    bandwidth: float = 0.05,
    grad_quantile: float = 0.90,
    smooth_px: int = 0,
    grad_eps: float = 1e-12,
    weight_by_dt: bool = False,        # if True, weight each pair's contribution by its Δt (days)
    max_gap_days: float | None = None, # ignore pairs with Δt > this (e.g., large gaps); None = keep all
) -> xr.Dataset:
    """
    Build a monthly (1..12) multi-year composite of front-normal velocity from a time-lat-lon DataArray.

    Method: for each adjacent pair of snapshots (t_i, t_{i+1}), compute front-normal fields using
    your 'front_normal_velocity', tag the result with the midpoint month, and aggregate across years.

    Returns
    -------
    xr.Dataset with dims ('month', y, x) and variables:
      - v_normal_mean, u_perp_mean, v_perp_mean (m/day)
      - pairs_count (count of contributing pairs per pixel)
    """
    # discover dims/coords from your helpers
    time_dim, y_dim, x_dim = _choose_dims(da)
    times = pd.to_datetime(da[time_dim].values)
    if len(times) < 2:
        raise ValueError("Need at least two time steps to compute velocities.")

    # Pre-allocate accumulators
    ny, nx = da.sizes[y_dim], da.sizes[x_dim]
    months = np.arange(1, 13, dtype=int)

    sum_vn = np.zeros((12, ny, nx), dtype=np.float64)
    sum_up = np.zeros((12, ny, nx), dtype=np.float64)
    sum_vp = np.zeros((12, ny, nx), dtype=np.float64)
    w_acc   = np.zeros((12, ny, nx), dtype=np.float64)  # weights (Δt if weighted, else counts)
    n_pairs = np.zeros((12, ny, nx), dtype=np.int32)    # number of *valid* contributing pairs

    for i in range(len(times) - 1):
        t1 = times[i]
        t2 = times[i + 1]
        dt_days = (t2 - t1) / pd.Timedelta(days=1)
        if dt_days <= 0:
            continue
        if (max_gap_days is not None) and (dt_days > max_gap_days):
            # Skip long gaps (e.g., sensor outages)
            continue

        # Compute velocity for this pair
        ds = front_normal_velocity(
            da, t1, t2,
            front_value=front_value,
            bandwidth=bandwidth,
            grad_quantile=grad_quantile,
            smooth_px=smooth_px,
            grad_eps=grad_eps
        )
        # Pull arrays
        vn = ds["v_normal"].values     # (y,x), NaN where not front/invalid
        up = ds["u_perp"].values
        vp = ds["v_perp"].values
        valid = np.isfinite(vn)

        if not np.any(valid):
            continue

        mon_idx = ((t1 + (t2 - t1) / 2).month) - 1  # 0..11

        # Weight = Δt_days (if requested), else 1 per valid pixel
        w = dt_days if weight_by_dt else 1.0

        # Accumulate only where valid
        sum_vn[mon_idx][valid] += vn[valid] * w
        sum_up[mon_idx][valid] += up[valid] * w
        sum_vp[mon_idx][valid] += vp[valid] * w
        w_acc[mon_idx][valid]  += w
        n_pairs[mon_idx][valid] += 1

    # Finalize means (avoid divide-by-zero)
    with np.errstate(invalid="ignore", divide="ignore"):
        vn_mean = np.divide(sum_vn, w_acc, out=np.full_like(sum_vn, np.nan), where=w_acc > 0)
        up_mean = np.divide(sum_up, w_acc, out=np.full_like(sum_up, np.nan), where=w_acc > 0)
        vp_mean = np.divide(sum_vp, w_acc, out=np.full_like(sum_vp, np.nan), where=w_acc > 0)

    # Build Dataset
    ds_out = xr.Dataset(
        {
            "v_normal_mean": (("month", y_dim, x_dim), vn_mean),
            "u_perp_mean":   (("month", y_dim, x_dim), up_mean),
            "v_perp_mean":   (("month", y_dim, x_dim), vp_mean),
            "pairs_count":   (("month", y_dim, x_dim), n_pairs),
        },
        coords={
            "month": ("month", months),
            y_dim: da[y_dim],
            x_dim: da[x_dim],
        },
        attrs={
            "note": "Monthly multi-year mean of front-normal velocity computed from adjacent time pairs. "
                    f"{'Weighted by Δt (days).' if weight_by_dt else 'Unweighted mean over pairs.'}"
        }
    )
    ds_out["v_normal_mean"].attrs.update({"units": "m/day", "long_name": "monthly mean front-normal speed"})
    ds_out["u_perp_mean"].attrs.update({"units": "m/day", "long_name": "monthly mean perpendicular east component"})
    ds_out["v_perp_mean"].attrs.update({"units": "m/day", "long_name": "monthly mean perpendicular north component"})
    ds_out["pairs_count"].attrs.update({"long_name": "number of valid time-pairs contributing"})
    return ds_out


def front_speed_along_channels(
    ds_perp: xr.Dataset,
    gdf_channels: gpd.GeoDataFrame,
    *,
    sample_spacing_m: float = 500.0,
    apex_latlon: Tuple[float,float] | None = None,
    require_front_mask: bool = True,
    min_speed_m_per_day: float = 0.0
) -> gpd.GeoDataFrame:
    """
    Project the perpendicular front motion onto channel tangents to get an along-channel speed (m/day).

    Returns a GeoDataFrame of POINT samples with:
      - line_id, s_m (chainage), lon, lat
      - u_perp, v_perp (m/day), speed_perp (m/day)
      - v_along_m_per_day (signed; positive in the oriented downstream/outward direction)
      - mask_front (bool)
    """
    lat_name, lon_name = _infer_latlon_names(ds_perp)

    # Work in a metric CRS for geometry; also prepare apex in same CRS if given
    gdf_m = _to_local_meters(gdf_channels.copy())
    if apex_latlon is not None:
        apex_gdf = gpd.GeoDataFrame(geometry=[Point(apex_latlon[1], apex_latlon[0])], crs="EPSG:4326").to_crs(gdf_m.crs)
        apex_xy = np.array([apex_gdf.geometry.iloc[0].x, apex_gdf.geometry.iloc[0].y])
    else:
        apex_xy = None

    rows = []
    line_id = 0
    for geom in gdf_m.geometry:
        if geom is None:
            continue
        # Merge multilines into a single line for traversal
        if geom.geom_type == "MultiLineString":
            geom = linemerge(geom)
        if geom.length == 0:
            continue

        xy, s = _densify_line(geom, sample_spacing_m)

        # Orient outward/downstream if apex supplied
        if apex_xy is not None:
            xy = _orient_line_outward(xy, apex_xy)

        # Unit tangent at each sample
        t = _tangent_vectors(xy)  # shape (n,2)

        # Convert samples back to lon/lat for raster sampling
        pts_ll = gpd.GeoSeries([Point(x, y) for x,y in xy], crs=gdf_m.crs).to_crs(epsg=4326)
        lat = np.array([p.y for p in pts_ll])
        lon = np.array([p.x for p in pts_ll])

        # Sample perpendicular motion fields
        # smp = _sample_fields_at_points(ds_perp, lat, lon, vars=("u_perp","v_perp","mask_front"))
        smp = _sample_fields_at_points_pairwise(ds_perp, lat, lon, vars=("u_perp","v_perp","mask_front"))

        u = smp["u_perp"].values
        v = smp["v_perp"].values
        front_mask = np.array(smp["mask_front"].values, dtype=bool)

        speed_perp = np.hypot(u, v)
        # Along-channel projection (east/north in meters/day dotted with tangent in meters)
        v_along = u*t[:,0] + v*t[:,1]

        # optional filtering
        if require_front_mask:
            v_along = np.where(front_mask, v_along, np.nan)
            speed_perp = np.where(front_mask, speed_perp, np.nan)
        if min_speed_m_per_day > 0:
            keep = (np.nan_to_num(np.abs(v_along)) >= min_speed_m_per_day)
            v_along = np.where(keep, v_along, np.nan)

        # Pack rows
        for i in range(len(s)):
            rows.append({
                "line_id": line_id,
                "s_m": float(s[i]),
                "lon": float(lon[i]),
                "lat": float(lat[i]),
                "u_perp": float(u[i]) if np.isfinite(u[i]) else np.nan,
                "v_perp": float(v[i]) if np.isfinite(v[i]) else np.nan,
                "speed_perp_m_per_day": float(speed_perp[i]) if np.isfinite(speed_perp[i]) else np.nan,
                "v_along_m_per_day": float(v_along[i]) if np.isfinite(v_along[i]) else np.nan,
                "mask_front": bool(front_mask[i]),
                "geometry": Point(xy[i,0], xy[i,1])  # in metric CRS for easy plotting with channels
            })
        line_id += 1

    out = gpd.GeoDataFrame(rows, crs=gdf_m.crs)
    # Also provide lon/lat geometry for web maps if you prefer
    out["geometry_wgs84"] = out.to_crs(epsg=4326).geometry
    return out


def make_region_masks(lat: xr.DataArray, lon: xr.DataArray, polygons: list) -> tuple:
    """Build a regionmask.Regions object and mask over (lat, lon)."""
    regs = regionmask.Regions(outlines=polygons)
    mask = regs.mask(geodataframe=None, lon_or_obj=lon, lat=lat)  # (lat, lon) integer labels per region
    return regs, mask


def monthly_composite_fixed_pair(
    da: xr.DataArray, month1: int, month2: int, *,
    reducer: str = "mean"
) -> xr.DataArray:
    """
    Multi-year composite of 'month2 - month1' for each year where both months exist.

    Returns a 2D DataArray (y,x) with the composite difference.
    """
    dam = _to_monthly_mean(da)
    # attach helpers
    dam = dam.assign_coords(year=("time", pd.to_datetime(dam.time.values).year),
                            month=("time", pd.to_datetime(dam.time.values).month))

    m1 = dam.where(dam["month"] == month1, drop=True)
    m2 = dam.where(dam["month"] == month2, drop=True)

    # swap to year coord (one value per year per month after resample)
    m1y = m1.swap_dims({"time": "year"})
    m2y = m2.swap_dims({"time": "year"})

    # align to common years
    common_years = np.intersect1d(m1y["year"].values, m2y["year"].values)
    if common_years.size == 0:
        raise ValueError("No overlapping years found for the two months.")

    d = (m2y.sel(year=common_years) - m1y.sel(year=common_years))
    if reducer == "mean":
        comp = d.mean(dim="year", skipna=True, keep_attrs=True)
    elif reducer == "median":
        comp = d.median(dim="year", skipna=True, keep_attrs=True)
    else:
        raise ValueError("reducer must be 'mean' or 'median'")

    comp.name = f"diff_{calendar.month_abbr[month2]}_minus_{calendar.month_abbr[month1]}"
    comp.attrs["note"] = f"Composite of ({calendar.month_name[month2]} - {calendar.month_name[month1]}) across years"
    comp.attrs["years_used"] = common_years.tolist()
    return comp


def monthly_composite_lagged(
    da: xr.DataArray, lag_months: int = 1, *,
    weight_by_dt: bool = False,
    reducer: str = "mean",
    max_gap_days: float | None = None) -> xr.Dataset:
    """
    For each calendar month m (1..12), compute the composite of
    value(m) - value(m - lag) across all years where both exist.

    Returns a Dataset with:
      - diff_mean(month,y,x)
      - pairs_count(month,y,x)
      - (optional) diff_median if reducer='median'
    """
    dam = _to_monthly_mean(da)

    # month index for the "later" month in the pair
    month_later = xr.DataArray(pd.to_datetime(dam.time.values).month, coords={"time": dam.time}, dims=("time",))

    # differences of consecutive monthly means with the specified lag
    # Build shifted series so we get pairs (t, t - lag)
    da_later = dam.copy()
    da_earlier = dam.shift(time=lag_months)
    # Δt in days (for optional weighting)
    dt_days = (da_later["time"] - da_earlier["time"]).astype("timedelta64[D]").astype(float)

    # If requested, drop pairs with very large gaps
    if max_gap_days is not None:
        good_gap = dt_days <= max_gap_days
        da_later = da_later.where(good_gap)
        da_earlier = da_earlier.where(good_gap)
        dt_days = xr.where(good_gap, dt_days, np.nan)

    diff = da_later - da_earlier  # (time,y,x)
    diff = diff.isel(time=slice(lag_months, None))  # first 'lag' entries have no earlier pair
    month = month_later.isel(time=slice(lag_months, None))

    # group by month of the LATER snapshot
    if weight_by_dt:
        w = dt_days.isel(time=slice(lag_months, None))
        num = (diff * w).groupby(month).sum(dim="time", skipna=True)
        den = w.groupby(month).sum(dim="time", skipna=True)
        diff_mean = xr.where(den > 0, num / den, np.nan)
    else:
        diff_mean = diff.groupby(month).mean(dim="time", skipna=True)

    pairs_count = diff.groupby(month).count(dim="time")  # counts non-NaN per pixel

    ds = xr.Dataset(
        {
            "diff_mean": diff_mean.rename({"group": "month"}),
            "pairs_count": pairs_count.rename({"group": "month"})
        }
    )
    if reducer == "median":
        diff_median = diff.groupby(month).median(dim="time", skipna=True)
        ds["diff_median"] = diff_median.rename({"group": "month"})

    # add month coordinate 1..12 explicitly if some are missing
    full_months = xr.DataArray(np.arange(1,13), dims=("month",), name="month")
    ds = ds.reindex(month=full_months, fill_value=np.nan)
    ds.attrs["note"] = f"Monthly composite of value(m) - value(m-{lag_months}) across years"
    ds.attrs["weighted_by_dt_days"] = bool(weight_by_dt)
    if max_gap_days is not None:
        ds.attrs["max_gap_days"] = float(max_gap_days)
    return ds


def pick_var_with_latlon(ds: xr.Dataset, lat: str, lon: str) -> str:
    """Pick a data variable that has (lat, lon) dims; prefer names with 'mask'/'water'."""
    preferred = [v for v in ds.data_vars if ("mask" in v.lower()) or ("water" in v.lower())]
    for v in preferred:
        if {lat, lon}.issubset(ds[v].dims):
            return v
    for v in ds.data_vars:
        if {lat, lon}.issubset(ds[v].dims):
            return v
    return list(ds.data_vars)[0]


def plot_month_grid(da_monthly_diff: xr.DataArray, title: str = "Monthly composite of differences"):
    """
    If 'month' in dims -> 12-panel grid.
    Else -> single panel (for a single composite like comp_pair).
    Uses a robust, safe diverging norm centered at 0.
    """
    # Build a safe norm from all values that will be plotted
    norm, vmax = two_slope_norm_safe(da_monthly_diff.values, quantile=98, fallback=1.0)

    if "month" in da_monthly_diff.dims:
        fig, axes = plt.subplots(3, 4, figsize=(14, 9), constrained_layout=True)
        last_im = None
        for m, ax in enumerate(axes.ravel(), start=1):
            arr = da_monthly_diff.sel(month=m)
            last_im = arr.plot(ax=ax, cmap="RdBu_r", norm=norm, add_colorbar=False)
            ax.set_title(calendar.month_abbr[m])
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.01)
        cbar.set_label("Difference (same units as input)")
        fig.suptitle(title, y=1.02, fontsize=12)
        plt.show()
    else:
        # Single 2D composite (e.g., fixed month pair)
        plt.figure(figsize=(7, 6))
        im = da_monthly_diff.plot(cmap="RdBu_r", norm=norm)
        plt.title(title + ("" if vmax > 0 else " (constant ~0; scaled with fallback)"))
        plt.xlabel("Longitude"); plt.ylabel("Latitude")
        plt.show()


def plot_parallel_velocity_with_channels(px, gdf_channels, *,
                                         apex=None,
                                         max_dist_m=3000.0,
                                         linewidth_by_order=True,
                                         title_dates=None):
    """
    One plot with:
      • Raster: px['v_parallel'] (m/day), optionally masked to ≤ max_dist_m from channels.
      • Overlay: channel network lines.
      • Optional: apex marker (lat, lon).

    Parameters
    ----------
    px : xr.Dataset (output of velocity_parallel_to_nearest_channel_field)
    gdf_channels : GeoDataFrame of channel polylines
    apex : (lat, lon) or None
    max_dist_m : float or None, mask raster to pixels within this distance to a channel
    linewidth_by_order : bool, vary line width if 'RIV_ORD' column exists
    title_dates : str or None, e.g., f"{ds_perp.attrs['t1']} → {ds_perp.attrs['t2']}"
    """
    # --- coord names & extent ---
    lat_name = next(n for n in ("lat","latitude","y") if n in px.coords)
    lon_name = next(n for n in ("lon","longitude","x") if n in px.coords)
    lat_min, lat_max = float(px[lat_name].min()), float(px[lat_name].max())
    lon_min, lon_max = float(px[lon_name].min()), float(px[lon_name].max())

    # Raster to plot: signed parallel speed; optionally restrict to near channels
    v = px["v_parallel"]
    if (max_dist_m is not None) and ("dist_to_channel_m" in px):
        v = v.where(px["dist_to_channel_m"] <= float(max_dist_m))

    # Symmetric, robust color scale around zero
    vabs = np.abs(v.values)
    vmax = float(np.nanpercentile(vabs, 98)) if np.isfinite(vabs).any() else 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    # Channels in WGS84 and clipped to raster bbox
    channels_wgs84 = gdf_channels.to_crs(epsg=4326)
    bbox_poly = box(lon_min, lat_min, lon_max, lat_max)
    bbox_gdf  = gpd.GeoDataFrame(geometry=[bbox_poly], crs="EPSG:4326")
    try:
        channels_clip = gpd.clip(channels_wgs84, bbox_gdf)
    except Exception:
        sel = channels_wgs84.intersects(bbox_poly)
        channels_clip = channels_wgs84.loc[sel].copy()
        channels_clip["geometry"] = channels_clip.geometry.intersection(bbox_poly)

    # --- plot ---
    fig, ax = plt.subplots(figsize=(10, 9))

    # Raster (no colorbar yet, we add it explicitly for consistent labeling)
    mappable = v.plot(ax=ax, cmap="RdBu_r", norm=norm, add_colorbar=False)
    cbar = plt.colorbar(mappable, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Parallel front speed (m/day)")

    # Channel network on top
    if "RIV_ORD" in channels_clip.columns and linewidth_by_order:
        lw = 0.4 + 0.25 * channels_clip["RIV_ORD"].clip(upper=8)
        channels_clip.plot(ax=ax, linewidth=lw, color="black", alpha=0.9)
    else:
        channels_clip.plot(ax=ax, linewidth=0.9, color="black", alpha=0.9)

    # Apex marker (optional)
    if apex is not None:
        ax.plot(apex[1], apex[0], marker="^", markersize=8, color="k",
                mec="white", mew=0.8, label="Apex")
        ax.legend(loc="lower right")

    ax.set_xlim(lon_min, lon_max); ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    if title_dates is not None:
        ax.set_title(f"Front motion parallel to nearest channel (m/day)\n{title_dates}")
    else:
        ax.set_title("Front motion parallel to nearest channel (m/day)")
    plt.show()


def plot_three_panel(da1, da2, diff, t1, t2, *, aspect_equal=True):
    CBAR_KW = {"shrink": 0.75, "aspect": 30, "pad": 0.02}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    da1.plot.pcolormesh(ax=axes[0], shading="auto", robust=True, cbar_kwargs=CBAR_KW)
    axes[0].set_title(t1); 
    if aspect_equal: axes[0].set_aspect("equal")

    da2.plot.pcolormesh(ax=axes[1], shading="auto", robust=True, cbar_kwargs=CBAR_KW)
    axes[1].set_title(t2);
    if aspect_equal: axes[1].set_aspect("equal")

    diff.plot.pcolormesh(ax=axes[2], shading="auto", cmap="RdBu_r", center=0, robust=True, cbar_kwargs=CBAR_KW)
    axes[2].set_title(f"Δ ({t2} − {t1})");
    if aspect_equal: axes[2].set_aspect("equal")
    plt.show()


def stats(label, arr):
    v = np.asarray(arr.values)
    print(f"{label}: min={np.nanmin(v):.3g}, max={np.nanmax(v):.3g}, mean={np.nanmean(v):.3g}")


def two_slope_norm_safe(arr, quantile=98, fallback=1.0):
    """
    Return a TwoSlopeNorm centered at 0 with symmetric limits.
    If the data are all NaN or all ~0, fall back to 'fallback'.
    """
    a = np.asarray(arr)
    finite = np.isfinite(a)
    if not finite.any():
        vmax = float(fallback)
    else:
        vabs = np.abs(a[finite])
        # robust cap
        vmax = float(np.nanpercentile(vabs, quantile))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = float(np.nanmax(vabs)) if vabs.size else fallback
            if not np.isfinite(vmax) or vmax <= 0:
                vmax = float(fallback)
    return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax), vmax


def velocity_parallel_to_nearest_channel_field(
    ds_perp: xr.Dataset,
    gdf_channels: gpd.GeoDataFrame,
    *,
    sample_spacing_m: float = 500.0,
    apex_latlon: tuple[float,float] | None = None,  # (lat, lon)
    require_front_mask: bool = True,
    max_distance_m: float | None = None
) -> xr.Dataset:
    """
    For each (front) pixel, compute the component of the perpendicular motion (u_perp,v_perp)
    that is PARALLEL to the nearest channel's tangent. If 'apex_latlon' is provided,
    channel tangents are oriented outward, so v_parallel>0 means downstream/outward.

    Returns an xr.Dataset with variables:
      - v_parallel (m/day): signed parallel component (NaN where not computed)
      - v_parallel_abs (m/day): magnitude of the parallel component
      - dist_to_channel_m (m): distance to nearest sampled channel point
    """

    # --- coords and arrays ---
    lat_name, lon_name = _infer_latlon_names_in_ds(ds_perp)
    lat1d = np.asarray(ds_perp[lat_name].values)
    lon1d = np.asarray(ds_perp[lon_name].values)
    Ny, Nx = lat1d.size, lon1d.size

    # 2D lat/lon
    lat2d, lon2d = np.meshgrid(lat1d, lon1d, indexing="ij")

    # Front mask & vectors
    mask_front = np.asarray(ds_perp["mask_front"].values, dtype=bool) if "mask_front" in ds_perp else np.ones((Ny, Nx), dtype=bool)
    U = np.asarray(ds_perp["u_perp"].values, dtype=float)
    V = np.asarray(ds_perp["v_perp"].values, dtype=float)

    # where we'll compute
    if require_front_mask:
        compute_mask = mask_front & np.isfinite(U) & np.isfinite(V)
    else:
        compute_mask = np.isfinite(U) & np.isfinite(V)
    idxs = np.where(compute_mask.ravel())[0]
    if idxs.size == 0:
        raise ValueError("No pixels to compute (check mask_front or data availability).")

    # --- common metric CRS (local UTM) ---
    # Use ds extent to choose UTM zone
    lat_c = float(np.nanmean(lat2d))
    lon_c = float(np.nanmean(lon2d))
    epsg_local = _local_utm_epsg(lat_c, lon_c)
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_local}", always_xy=True)

    # Project raster pixel centers to meters
    X2d, Y2d = transformer.transform(lon2d, lat2d)  # lon, lat -> X, Y (m)
    Xq = X2d.ravel()[idxs]
    Yq = Y2d.ravel()[idxs]
    Uq = U.ravel()[idxs]
    Vq = V.ravel()[idxs]

    # Project channels to same CRS
    gdf_m = gdf_channels.to_crs(epsg=epsg_local)
    apex_xy = None
    if apex_latlon is not None:
        ax, ay = transformer.transform(apex_latlon[1], apex_latlon[0])
        apex_xy = np.array([ax, ay], dtype=float)

    # Densify channels and collect sample tangents
    xs, ys, txs, tys = _collect_channel_samples(gdf_m, step_m=sample_spacing_m, apex_xy=apex_xy)

    # --- nearest neighbor search (KD-tree) ---
    pts = np.column_stack([xs, ys])
    try:
        from scipy.spatial import cKDTree  # fast if SciPy is available
        tree = cKDTree(pts)
        dists, nidx = tree.query(np.column_stack([Xq, Yq]), k=1)
    except Exception:
        try:
            from sklearn.neighbors import KDTree as SKKDTree
            tree = SKKDTree(pts, leaf_size=40)
            dists, nidx = tree.query(np.column_stack([Xq, Yq]), k=1, return_distance=True)
            dists = dists[:,0]; nidx = nidx[:,0]
        except Exception as e:
            raise ImportError("Need scipy.spatial.cKDTree or sklearn.neighbors.KDTree for nearest-channel lookup.") from e

    # Optional: reject pixels far from any channel sample
    if max_distance_m is not None:
        keep = dists <= float(max_distance_m)
    else:
        keep = np.ones_like(dists, dtype=bool)

    # Unit tangent at the nearest channel sample
    tx = txs[nidx]
    ty = tys[nidx]

    # Project front motion onto tangent
    vpar = Uq * tx + Vq * ty
    vpar_abs = np.abs(vpar)

    # Write back into full-size arrays
    v_parallel = np.full((Ny, Nx), np.nan, dtype=float)
    v_parallel_abs = np.full((Ny, Nx), np.nan, dtype=float)
    dist_out = np.full((Ny, Nx), np.nan, dtype=float)

    # only assign where 'compute_mask' and 'keep' are true
    sub_keep = keep
    flat_pos = idxs[sub_keep]
    v_parallel.ravel()[flat_pos] = vpar[sub_keep]
    v_parallel_abs.ravel()[flat_pos] = vpar_abs[sub_keep]
    dist_out.ravel()[flat_pos] = dists[sub_keep]

    ds_out = xr.Dataset(
        {
            "v_parallel":       ((lat_name, lon_name), v_parallel),
            "v_parallel_abs":   ((lat_name, lon_name), v_parallel_abs),
            "dist_to_channel_m":((lat_name, lon_name), dist_out),
        },
        coords={lat_name: ds_perp[lat_name], lon_name: ds_perp[lon_name]},
        attrs={
            "epsg_local": epsg_local,
            "note": "v_parallel is the component of (u_perp,v_perp) along the nearest channel tangent. "
                    "If apex_latlon was provided, tangents were oriented outward so v_parallel>0 is downstream/outward."
        }
    )
    ds_out["v_parallel"].attrs.update({"units": "m/day", "long_name": "front motion parallel to nearest channel"})
    ds_out["v_parallel_abs"].attrs.update({"units": "m/day", "long_name": "abs(parallel front motion)"})
    ds_out["dist_to_channel_m"].attrs.update({"units": "m", "long_name": "distance to nearest channel sample"})
    return ds_out
