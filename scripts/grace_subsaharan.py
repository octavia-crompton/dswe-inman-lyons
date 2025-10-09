# pip install earthaccess xarray rioxarray geopandas shapely pandas numpy
import pathlib, warnings
import pandas as pd
import xarray as xr
import rioxarray  # noqa: F401
import earthaccess as ea

# ========================== CONFIG ==========================
OUT_DIR = pathlib.Path("../date/grace_subsaharan"); OUT_DIR.mkdir(parents=True, exist_ok=True)
START, END = "2002-04-01", "2025-10-01"

# Sub-Saharan Africa bbox: (minLon, minLat, maxLon, maxLat)
REGION_BBOX = (-20.0, -35.0, 52.0, 15.0)

WRITE_MONTHLY_TIFS = True   # set False if you only want the NetCDF
TIF_PREFIX = "grace_lwe_subsaharan_"  # per-month tifs: prefix + YYYY-MM + .tif

PREFERRED_SHORTNAMES = [
    "TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4",
    "TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V03",
    "TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.1_V2",
    "TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06_V2",
]

# ========================== HELPERS ==========================
def pick_lwe_var(ds: xr.Dataset) -> xr.DataArray:
    for v in ["lwe_thickness","LWE_thickness","lwe_thickness_jpl","lwe_thickness_csr","water_equivalent_thickness"]:
        if v in ds: return ds[v]
    for v in ds.data_vars:
        ln = str(ds[v].attrs.get("long_name","")).lower()
        if "water equivalent" in ln or "lwe" in ln: return ds[v]
    raise KeyError("LWE variable not found")

def _infer_lon_lat_names(da: xr.DataArray):
    def find(name_opts, pool):
        for k in name_opts:
            if k in pool: return k
        return None
    dims = {d.lower(): d for d in da.dims}
    coords = {c.lower(): c for c in da.coords}
    lon = dims.get("lon") or dims.get("longitude") or dims.get("x") or coords.get("lon") or coords.get("longitude") or coords.get("x")
    lat = dims.get("lat") or dims.get("latitude") or dims.get("y") or coords.get("lat") or coords.get("latitude") or coords.get("y")
    if lon is None or lat is None:
        raise KeyError(f"Cannot infer lon/lat from dims={list(da.dims)} coords={list(da.coords)}")
    # map back to original case
    lon = next(n for n in da.coords if n.lower()==lon) if lon in coords else next(n for n in da.dims if n.lower()==lon)
    lat = next(n for n in da.coords if n.lower()==lat) if lat in coords else next(n for n in da.dims if n.lower()==lat)
    return lon, lat

def ensure_spatial(da: xr.DataArray) -> xr.DataArray:
    lon_name, lat_name = _infer_lon_lat_names(da)
    # wrap 0..360 → -180..180 if needed
    if float(da[lon_name].max()) > 180:
        da = da.assign_coords({lon_name: (((da[lon_name] + 180) % 360) - 180)})
    # sort lon asc, lat desc (north→south)
    da = da.sortby(lon_name).sortby(lat_name, ascending=False)
    # set spatial dims + CRS
    da = da.rio.set_spatial_dims(x_dim=lon_name, y_dim=lat_name, inplace=False)
    da = da.rio.write_crs("EPSG:4326", inplace=False)
    return da

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
    for d in cats:
        cid = d["meta"]["concept-id"]
        sname = d["umm"].get("ShortName","")
        try:
            # broad search; we’ll subset by bbox ourselves after open
            gs = ea.search_data(concept_id=cid, temporal=(START, END), count=2)
            if gs:
                print(f"Using dataset: {sname} (concept-id={cid})")
                return d, gs
        except Exception as e:
            warnings.warn(f"Skipping {sname}: {e}")
    raise RuntimeError("No suitable GRACE/FO JPL Mascon dataset found.")

def open_any(granules, out_dir: pathlib.Path):
    # Try download first; if cloud-only, open via fsspec
    files = ea.download(granules, local_path=str(out_dir))
    if files:
        print(f"Downloaded: {len(files)} local file(s)")
        return xr.open_dataset(str(files[0]), decode_times=True)
    # cloud-hosted
    opened = ea.open(granules)
    if not opened: raise RuntimeError("earthaccess.open returned no file-like objects.")
    with opened[0] as fo:
        return xr.open_dataset(fo, decode_times=True)

# ========================== MAIN ==========================
def main():
    ea.login()

    _, granules = find_dataset_and_granules()
    ds = open_any(granules, OUT_DIR)

    # pick variable & normalize spatial coords
    da = pick_lwe_var(ds)
    da = ensure_spatial(da)

    # subset time & region (fast, avoids vector clipping)
    lon_name, lat_name = _infer_lon_lat_names(da)
    minx, miny, maxx, maxy = REGION_BBOX
    da = da.sel(time=slice(START, END))
    da_reg = da.sel({lon_name: slice(minx, maxx), lat_name: slice(maxy, miny)})  # lat is descending

    # ---- Write a single compressed NetCDF cube (time × lat × lon) ----
    nc_path = OUT_DIR / "grace_lwe_subsaharan_cm.nc"
    comp = dict(zlib=True, complevel=4, _FillValue=-9999.0)
    encoding = {da_reg.name: comp}
    da_reg.astype("float32").to_netcdf(nc_path, encoding=encoding)
    print("Wrote:", nc_path)

    # ---- (Optional) Write monthly GeoTIFFs ----
    if WRITE_MONTHLY_TIFS:
        for t in pd.to_datetime(da_reg["time"].values):
            da_t = da_reg.sel(time=t, drop=True)  # 2D slice
            ts = pd.to_datetime(t).strftime("%Y-%m")
            da_t = ensure_spatial(da_t)  # assert per-slice
            tif_path = OUT_DIR / f"{TIF_PREFIX}{ts}.tif"
            da_t.rio.to_raster(
                tif_path,
                dtype="float32",
                nodata=-9999.0,
                compress="DEFLATE",
                zlevel=4,
            )
        print(f"Wrote monthly GeoTIFFs to {OUT_DIR}")

if __name__ == "__main__":
    main()
