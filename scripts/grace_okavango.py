# pip install earthaccess xarray rioxarray geopandas shapely pandas numpy
import pathlib, warnings
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import box
import rioxarray  # noqa: F401
import earthaccess as ea

# ========================== USER CONFIG ==========================
DATA_DIR = pathlib.Path("./grace_okavango_out"); DATA_DIR.mkdir(parents=True, exist_ok=True)
START, END = "2002-04-01", "2025-10-01"
AOI_POLYFILE = None  # put "okavango_basin.geojson" for precise basin; else bbox is used
OKAVANGO_BBOX = (19.0, -21.8, 24.8, -17.0)  # (minLon, minLat, maxLon, maxLat)

PREFERRED_SHORTNAMES = [
    "TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4",
    "TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V03",
    "TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.1_V2",
    "TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06_V2",
]

# ========================== HELPERS ==========================
def load_aoi():
    if AOI_POLYFILE:
        aoi = gpd.read_file(AOI_POLYFILE).to_crs(4326)
        geom = aoi.geometry.unary_union
        return gpd.GeoSeries([geom], crs="EPSG:4326")
    minx, miny, maxx, maxy = OKAVANGO_BBOX
    return gpd.GeoSeries([box(minx, miny, maxx, maxy)], crs="EPSG:4326")

def pick_lwe_var(ds: xr.Dataset) -> xr.DataArray:
    for v in ["lwe_thickness","LWE_thickness","lwe_thickness_jpl","lwe_thickness_csr","water_equivalent_thickness"]:
        if v in ds: return ds[v]
    for v in ds.data_vars:
        ln = str(ds[v].attrs.get("long_name","")).lower()
        if "water equivalent" in ln or "lwe" in ln: return ds[v]
    raise KeyError("Could not find LWE thickness variable.")

def _infer_lon_lat_names(da: xr.DataArray):
    # dims first
    dims_lower = {d.lower(): d for d in da.dims}
    lon_name = next((dims_lower[k] for k in ("lon","longitude","x") if k in dims_lower), None)
    lat_name = next((dims_lower[k] for k in ("lat","latitude","y") if k in dims_lower), None)
    # coords fallback
    if lon_name is None or lat_name is None:
        coords_lower = {c.lower(): c for c in da.coords}
        if lon_name is None:
            lon_name = next((coords_lower[k] for k in ("lon","longitude","x") if k in coords_lower), None)
        if lat_name is None:
            lat_name = next((coords_lower[k] for k in ("lat","latitude","y") if k in coords_lower), None)
    if lon_name is None or lat_name is None:
        raise KeyError(f"Cannot infer lon/lat names from dims={list(da.dims)} coords={list(da.coords)}")
    return lon_name, lat_name

def ensure_spatial(da: xr.DataArray) -> xr.DataArray:
    """Make sure lon/lat dims exist, are sorted, and set for rioxarray, each time."""
    lon_name, lat_name = _infer_lon_lat_names(da)
    # wrap 0..360 → -180..180 if needed
    if float(da[lon_name].max()) > 180:
        da = da.assign_coords({lon_name: (((da[lon_name] + 180) % 360) - 180)})
    # sort lon asc, lat desc
    da = da.sortby(lon_name).sortby(lat_name, ascending=False)
    # set spatial dims and CRS
    da = da.rio.set_spatial_dims(x_dim=lon_name, y_dim=lat_name, inplace=False)
    da = da.rio.write_crs("EPSG:4326", inplace=False)
    return da

def clip_and_write(da2d: xr.DataArray, aoi_gs: gpd.GeoSeries, ts_str: str, out_dir: pathlib.Path):
    """Re-assert spatial dims on the *slice*, clip, write GeoTIFF, return mean/std.
    plain (unweighted) mean over all clipped grid cells inside your AOI, for each time step.
    collapses over all remaining dimensions (lat and lon) → simple arithmetic mean of LWE across all included cells.
    It does not weight cells by cos(latitude) or actual cell area.
    Every retained GRACE cell counts equally, even though they differ slightly in physical area with latitude.
    """
    da2d = ensure_spatial(da2d)  # ←  do this on every slice
    clipped = da2d.rio.clip(aoi_gs.to_crs(4326).geometry, all_touched=True, drop=True)
    tif_path = out_dir / f"grace_lwe_okavango_{ts_str}.tif"
    clipped.rio.to_raster(tif_path)
    return float(clipped.mean(skipna=True).values), float(clipped.std(skipna=True).values), str(tif_path)

def find_dataset_and_granules():
    cats = []
    for sn in PREFERRED_SHORTNAMES:
        try:
            cats.extend(ea.search_datasets(short_name=sn) or [])
        except Exception:
            pass
    if not cats:
        cats = ea.search_datasets(keyword="GRACE GRACE-FO Mascon RL06 JPL CRI grid") or []

    def score(meta):
        sname = meta["umm"].get("ShortName","")
        s = sum(k in sname for k in ("GRAC","GRFO","MASCON","CRI","RL06"))
        if "RL06.3" in sname: s += 2
        if sname.endswith("_V4"): s += 2
        return s

    cats = sorted(cats, key=score, reverse=True)
    bbox = OKAVANGO_BBOX

    for d in cats:
        cid = d["meta"]["concept-id"]
        sname = d["umm"].get("ShortName","")
        try:
            gs = ea.search_data(concept_id=cid, temporal=(START, END), bounding_box=bbox, count=5000)
            if gs:
                print(f"Using dataset: {sname} (concept-id={cid}) → {len(gs)} granule(s)")
                return d, gs
        except Exception as e:
            warnings.warn(f"Skipping {sname}: {e}")
    raise RuntimeError("No suitable GRACE/FO JPL Mascon dataset returned granules for your time/bbox.")

def open_cloud(granules):
    opened = ea.open(granules)
    if not opened: raise RuntimeError("earthaccess.open returned no file-like objects.")
    with opened[0] as fo:
        return xr.open_dataset(fo, decode_times=True)

# ========================== MAIN ==========================
def main():
    aoi = load_aoi()
    ea.login()  # will cache a token

    dataset_meta, granules = find_dataset_and_granules()

    local_files = ea.download(granules, local_path=str(DATA_DIR))
    print(f"Downloaded: {len(local_files)} local file(s)")

    rows = []

    if local_files:
        # Local NetCDF path
        for i, f in enumerate(sorted(map(str, local_files))):
            with xr.open_dataset(f, decode_times=True) as ds:
                da = pick_lwe_var(ds)
                # set spatial on full array once (safe), but we will also re-assert per-slice
                da = ensure_spatial(da)
                if "time" in da.dims:
                    da = da.sel(time=slice(START, END))
                    for t in pd.to_datetime(da["time"].values):
                        # IMPORTANT: drop keeps a 2-D slice; we'll re-ensure spatial in clip_and_write
                        da_t = da.sel(time=t, drop=True)
                        ts = pd.to_datetime(t).strftime("%Y-%m")
                        m, s, tif = clip_and_write(da_t, aoi, ts, DATA_DIR)
                        rows.append({"date": ts, "lwe_mean_cm": m, "lwe_std_cm": s, "tif": tif})
                else:
                    ts = "no-time"
                    m, s, tif = clip_and_write(da, aoi, ts, DATA_DIR)
                    rows.append({"date": ts, "lwe_mean_cm": m, "lwe_std_cm": s, "tif": tif})
    else:
        # Cloud-hosted single granule with full time axis
        ds = open_cloud(granules)
        da = pick_lwe_var(ds)
        da = ensure_spatial(da)
        if "time" in da.dims:
            da = da.sel(time=slice(START, END))
            for t in pd.to_datetime(da["time"].values):
                da_t = da.sel(time=t, drop=True)
                ts = pd.to_datetime(t).strftime("%Y-%m")
                m, s, tif = clip_and_write(da_t, aoi, ts, DATA_DIR)
                rows.append({"date": ts, "lwe_mean_cm": m, "lwe_std_cm": s, "tif": tif})
        else:
            ts = "no-time"
            m, s, tif = clip_and_write(da, aoi, ts, DATA_DIR)
            rows.append({"date": ts, "lwe_mean_cm": m, "lwe_std_cm": s, "tif": tif})

    pd.DataFrame(rows).sort_values("date").to_csv(DATA_DIR / "grace_okavango_timeseries.csv", index=False)
    print("Wrote:", DATA_DIR / "grace_okavango_timeseries.csv")

if __name__ == "__main__":
    main()

# All of the PREFERRED_SHORTNAMES below point to JPL GRACE/GRACE-FO mascon L3 products.
# They contain monthly global water storage / height anomalies relative to a time mean,
# in equivalent water thickness (cm), derived from the GRACE + GRACE-FO missions and
# processed with the JPL mascon solution (RL06M).
#
# Name components:
#   GRAC-GRFO  = GRACE + GRACE-FO combined record
#   MASCON     = mascon inversion (3° equal-area spherical caps)
#   CRI        = "Coastal Resolution Improvement" filter to reduce land–ocean leakage
#   GRID       = gridded fields
#   RL06, RL06.1, RL06.3 = successive RL06 reprocessings (updated background models, etc.)
#   V2, V03, V4 = product version numbers within that release
#
# Of these, TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4 is the current recommended JPL
# dataset for most land-hydrology applications.


# The product TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4 has a spatial resolution of 0.5° × 0.5° (lat × lon) on a global grid. 
# NASA Earthdata
# So each grid cell is about:
# ~55 km in latitude, and
# ~55 km × cos(latitude) in longitude (e.g., ~51 km at 25°S near the Okavango).