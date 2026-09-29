"""grace_blocks.py – discover JPL mascon blocks and extract block TWS from the local netCDF.

The JPL RL06.3M mascon product is delivered on a 0.5° grid, but every pixel
inside one mascon carries an identical time series.  Hashing each pixel's
series therefore recovers the mascon "blocks".  Everything here works from
the local netCDF (``GRCTellus.JPL.*.nc``), which carries ``time_bounds`` and
so gives unambiguous month labels.  This replaces the Earth Engine
``NASA/GRACE/MASS_GRIDS_V04/MASCON_CRI`` reduction used previously, whose
``system:index`` dates were the day *before* each solution period and which
therefore labelled ~90 % of months one month early.

Typical use
-----------
    from src.grace_blocks import (discover_blocks, block_area_m2, block_tws,
                                  domain_tws, neighbour_block, ee_geometry)

    blocks = discover_blocks(GRACE_NC, ref_geom=delta_union, pad_deg=8)
    delta_blocks = [b for b in blocks if b["intersects_ref"]]
    ne = [b for b in delta_blocks if b["quadrant"] == "NE"][0]
    tws_ne = block_tws(GRACE_NC, ne)          # cm, indexed by month start
    south = neighbour_block(ne, blocks, "S")
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Geod
from shapely.geometry import box as shapely_box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.prepared import prep

_GEOD = Geod(ellps="WGS84")
TWS_VAR = "lwe_thickness"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def open_grace(nc_path: str | Path) -> xr.Dataset:
    """Open the JPL mascon netCDF with longitudes wrapped to −180…180 and sorted."""
    ds = xr.open_dataset(nc_path)
    lon = ds["lon"].values
    if lon.max() > 180:
        ds = ds.assign_coords(lon=np.where(lon > 180, lon - 360, lon)).sortby("lon")
    return ds


def solution_months(ds: xr.Dataset) -> pd.DatetimeIndex:
    """Month-start label for every GRACE solution, from the midpoint of ``time_bounds``.

    Falls back to the ``time`` coordinate when ``time_bounds`` is missing.
    """
    if "time_bounds" in ds:
        tb = ds["time_bounds"].values
        t0 = pd.to_datetime(tb[:, 0])
        t1 = pd.to_datetime(tb[:, 1])
        mid = t0 + (t1 - t0) / 2
    else:
        mid = pd.to_datetime(ds["time"].values)
    return pd.DatetimeIndex(mid).to_period("M").to_timestamp()


def _dedupe_months(values: np.ndarray, ds: xr.Dataset) -> pd.Series:
    """Series indexed by month start; if two solutions map to one month keep the
    one whose midpoint is closest to the 16th of that month."""
    months = solution_months(ds)
    if "time_bounds" in ds:
        tb = ds["time_bounds"].values
        mid = pd.to_datetime(tb[:, 0]) + (pd.to_datetime(tb[:, 1]) - pd.to_datetime(tb[:, 0])) / 2
    else:
        mid = pd.to_datetime(ds["time"].values)
    dist = np.abs((pd.DatetimeIndex(mid) - (months + pd.Timedelta(days=15))).days)
    df = pd.DataFrame({"month": months, "val": values, "dist": dist})
    df = df.sort_values(["month", "dist"]).drop_duplicates("month", keep="first")
    s = pd.Series(df["val"].values, index=pd.DatetimeIndex(df["month"]), name="tws_cm")
    s.index.name = "date"
    return s.sort_index()


# ---------------------------------------------------------------------------
# Block discovery
# ---------------------------------------------------------------------------
def discover_blocks(
    nc_path: str | Path,
    ref_geom: BaseGeometry,
    pad_deg: float = 8.0,
    ref_point=None,
) -> list[dict]:
    """Group 0.5° pixels with identical time series into mascon blocks.

    Parameters
    ----------
    nc_path : path to the JPL mascon netCDF.
    ref_geom : shapely geometry used (a) to define the search window
        (``ref_geom.bounds`` ± ``pad_deg``) and (b) to flag blocks that
        intersect it (``intersects_ref``) and their overlap fraction
        (``frac_ref`` = intersection area / ref area, in degrees²).
    pad_deg : half-width of the search window beyond ``ref_geom.bounds``.
    ref_point : (lon, lat) used for the NE/NW/SE/SW ``quadrant`` label.
        Defaults to ``ref_geom.centroid``.

    Returns
    -------
    list of dicts with keys ``block_id, quadrant, lat0, lat1, lon0, lon1,
    n_pixels, intersects_ref, frac_ref, geometry, clat, clon``.
    Blocks are sorted north→south then west→east, and numbered in that order.
    """
    ds = open_grace(nc_path)
    da = ds[TWS_VAR]
    lats_all = da["lat"].values
    lons_all = da["lon"].values
    dlat = float(np.median(np.abs(np.diff(lats_all))))
    dlon = float(np.median(np.abs(np.diff(lons_all))))

    minx, miny, maxx, maxy = ref_geom.bounds
    da_sub = da.sel(
        lon=lons_all[(lons_all >= minx - pad_deg) & (lons_all <= maxx + pad_deg)],
        lat=lats_all[(lats_all >= miny - pad_deg) & (lats_all <= maxy + pad_deg)],
    )
    ts = da_sub.stack(cell=("lat", "lon")).transpose("time", "cell").dropna("cell", how="all")
    cell_index = ts["cell"].to_index()
    arr = ts.values
    ds.close()

    groups: dict[str, list[int]] = defaultdict(list)
    for i in range(arr.shape[1]):
        v = np.round(arr[:, i].astype("float64"), 6)
        v = np.where(np.isnan(v), -9999.0, v)
        groups[hashlib.md5(v.tobytes()).hexdigest()].append(i)

    if ref_point is None:
        c = ref_geom.centroid
        ref_point = (c.x, c.y)
    ref_prep = prep(ref_geom)
    ref_area = ref_geom.area

    raw = []
    for idxs in groups.values():
        lc = np.array([cell_index[j][0] for j in idxs], dtype=float)
        lo = np.array([cell_index[j][1] for j in idxs], dtype=float)
        lat0, lat1 = float(lc.min() - dlat / 2), float(lc.max() + dlat / 2)
        lon0, lon1 = float(lo.min() - dlon / 2), float(lo.max() + dlon / 2)
        poly = shapely_box(lon0, lat0, lon1, lat1)
        clat, clon = (lat0 + lat1) / 2, (lon0 + lon1) / 2
        ns = "N" if clat > ref_point[1] else "S"
        ew = "W" if clon < ref_point[0] else "E"
        raw.append(dict(
            quadrant=f"{ns}{ew}", lat0=lat0, lat1=lat1, lon0=lon0, lon1=lon1,
            clat=clat, clon=clon, n_pixels=len(idxs), geometry=poly,
            intersects_ref=bool(ref_prep.intersects(poly)),
            frac_ref=float(ref_geom.intersection(poly).area / ref_area) if ref_area > 0 else 0.0,
        ))
    raw.sort(key=lambda b: (-b["lat1"], b["lon0"]))
    for k, b in enumerate(raw, start=1):
        b["block_id"] = f"B{k:03d}"
    return raw


def block_area_m2(block: dict | BaseGeometry) -> float:
    """Geodesic (WGS84) area of a block or any lon/lat polygon, in m²."""
    geom = block["geometry"] if isinstance(block, dict) else block
    area, _ = _GEOD.geometry_area_perimeter(geom)
    return abs(float(area))


def blocks_union(blocks: Sequence[dict]):
    return unary_union([b["geometry"] for b in blocks])


def gdf_union(gdf):
    """Dissolve every geometry of a GeoDataFrame / GeoSeries into one shapely geometry.

    Prefers ``union_all()``, which geopandas added in 1.0, and falls back to the
    ``unary_union`` property on older versions.  geopandas 1.x deprecates
    ``unary_union`` while geopandas 0.14 has no ``union_all``, so calling either
    one directly breaks on one of this project's two environments.
    """
    geoms = getattr(gdf, "geometry", gdf)
    if hasattr(geoms, "union_all"):
        return geoms.union_all()
    return geoms.unary_union


def describe(block: dict) -> str:
    return (f"{block['block_id']} ({block['quadrant']})  "
            f"lon {block['lon0']:.2f}–{block['lon1']:.2f}  "
            f"lat {block['lat0']:.2f}–{block['lat1']:.2f}  "
            f"{block['n_pixels']} px")


def neighbour_block(block: dict, blocks: Iterable[dict], direction: str) -> dict | None:
    """Block adjacent to ``block`` in direction ``"N" | "S" | "E" | "W"``.

    The target is the box one block-height (or width) away; among blocks that
    intersect it and lie strictly on that side, the one with the largest
    overlap is returned (``None`` if nothing qualifies).
    """
    h = block["lat1"] - block["lat0"]
    w = block["lon1"] - block["lon0"]
    if direction == "S":
        target = shapely_box(block["lon0"], block["lat0"] - h, block["lon1"], block["lat0"])
        side = lambda b: b["lat1"] <= block["lat0"] + 1e-6
    elif direction == "N":
        target = shapely_box(block["lon0"], block["lat1"], block["lon1"], block["lat1"] + h)
        side = lambda b: b["lat0"] >= block["lat1"] - 1e-6
    elif direction == "W":
        target = shapely_box(block["lon0"] - w, block["lat0"], block["lon0"], block["lat1"])
        side = lambda b: b["lon1"] <= block["lon0"] + 1e-6
    elif direction == "E":
        target = shapely_box(block["lon1"], block["lat0"], block["lon1"] + w, block["lat1"])
        side = lambda b: b["lon0"] >= block["lon1"] - 1e-6
    else:
        raise ValueError("direction must be one of N, S, E, W")
    cands = [b for b in blocks
             if b["block_id"] != block["block_id"] and side(b)
             and b["geometry"].intersects(target)]
    if not cands:
        return None
    return max(cands, key=lambda b: b["geometry"].intersection(target).area)


# ---------------------------------------------------------------------------
# TWS extraction
# ---------------------------------------------------------------------------
def block_tws(nc_path: str | Path, block: dict) -> pd.Series:
    """Block TWS anomaly (cm LWE) from the centre pixel, indexed by month start."""
    ds = open_grace(nc_path)
    vals = ds[TWS_VAR].sel(lat=block["clat"], lon=block["clon"], method="nearest").values
    s = _dedupe_months(vals, ds)
    ds.close()
    return s.rename(f"TWS_{block['block_id']}_cm")


def domain_tws(nc_path: str | Path, blocks: Sequence[dict]) -> pd.DataFrame:
    """Area-weighted TWS of a union of blocks.

    Returns a DataFrame indexed by month start with columns
    ``TWS_cm`` (area-weighted mean LWE, cm) and ``TWS_km3`` (stored volume
    anomaly, km³), plus the total geodesic area as ``.attrs['area_m2']``.
    """
    areas = np.array([block_area_m2(b) for b in blocks])
    series = [block_tws(nc_path, b) for b in blocks]
    df = pd.concat(series, axis=1)
    vals = df.values
    km3 = np.nansum(vals / 100.0 * areas, axis=1) / 1e9
    all_nan = np.isnan(vals).all(axis=1)
    km3[all_nan] = np.nan
    cm = km3 * 1e9 / areas.sum() * 100.0
    out = pd.DataFrame({"TWS_cm": cm, "TWS_km3": km3}, index=df.index)
    out.index.name = "date"
    out.attrs["area_m2"] = float(areas.sum())
    return out


def tws_difference(nc_path: str | Path, a: dict, b: dict, name: str) -> pd.DataFrame:
    """DataFrame with TWS of blocks ``a`` and ``b`` and their difference ``a − b``.

    Columns: ``TWS_<a>_cm, TWS_<b>_cm, <name>`` — e.g. name ``"NW_NE_diff_cm"``.
    """
    sa = block_tws(nc_path, a)
    sb = block_tws(nc_path, b)
    df = pd.concat([sa, sb], axis=1)
    df[name] = df.iloc[:, 0] - df.iloc[:, 1]
    return df


# ---------------------------------------------------------------------------
# Earth Engine geometry
# ---------------------------------------------------------------------------
def ee_geometry(blocks_or_geom, geodesic: bool = True):
    """Earth Engine MultiPolygon for a list of blocks or a shapely (Multi)Polygon.

    Coordinates are nested as ``[polygon][ring][vertex]`` so each block is a
    separate polygon (earlier notebooks passed ``[ring][vertex]`` and got one
    polygon with several rings).
    """
    import ee  # imported lazily so the rest of the module works without EE

    if isinstance(blocks_or_geom, BaseGeometry):
        geom = blocks_or_geom
        polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    else:
        polys = [b["geometry"] for b in blocks_or_geom]
    coords = [[[list(c) for c in p.exterior.coords]] for p in polys]
    return ee.Geometry.MultiPolygon(coords, proj="EPSG:4326", geodesic=geodesic)


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------
def block_map(blocks: Sequence[dict], highlight: dict[str, Sequence[dict]] | None = None,
              ref_gdf=None, center=None, zoom_start: int = 7, ref_name: str = "Reference polygon"):
    """Folium map of all blocks with optional highlighted groups.

    ``highlight`` maps a colour (hex) to a list of blocks, e.g.
    ``{"#1b9e77": USE_BLOCKS, "#d62728": [SOUTH_BLOCK]}``.
    """
    import folium

    if center is None:
        lats = [b["clat"] for b in blocks]
        lons = [b["clon"] for b in blocks]
        center = (float(np.mean(lats)), float(np.mean(lons)))
    m = folium.Map(location=list(center), zoom_start=zoom_start, tiles=None)
    folium.TileLayer("CartoDB positron", name="Carto Positron").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Esri Imagery", overlay=False, control=True,
    ).add_to(m)
    if ref_gdf is not None:
        folium.GeoJson(
            ref_gdf.__geo_interface__, name=ref_name,
            style_function=lambda f: {"color": "#8c2d04", "weight": 3, "fillOpacity": 0.15},
        ).add_to(m)

    colour_of: dict[str, str] = {}
    for colour, group in (highlight or {}).items():
        for b in group:
            if b is not None:
                colour_of[b["block_id"]] = colour

    fg = folium.FeatureGroup(name="Mascon blocks", show=True)
    for b in blocks:
        colour = colour_of.get(b["block_id"], "#bbbbbb")
        sel = b["block_id"] in colour_of
        folium.Rectangle(
            bounds=[[b["lat0"], b["lon0"]], [b["lat1"], b["lon1"]]],
            color=colour, weight=3 if sel else 1, fill=sel,
            fill_opacity=0.2 if sel else 0.0,
            tooltip=describe(b),
        ).add_to(fg)
    fg.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m
