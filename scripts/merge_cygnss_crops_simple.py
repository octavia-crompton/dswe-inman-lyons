#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge cropped daily CYGNSS watermask files into one time-stacked NetCDF.
Tolerates variable-name differences like 'water_mask' vs 'watermask'.
"""

import re
import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import xarray as xr


# ---------- helpers ----------
def parse_day_from_name(path: Path) -> Optional[pd.Timestamp]:
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", path.name)  # okav_YYYY-MM-DD.nc
    if m:
        return pd.Timestamp(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
    m2 = re.search(r"(\d{4})(\d{2})(\d{2})", path.name)   # okav_YYYYMMDD.nc
    if m2:
        return pd.Timestamp(f"{m2.group(1)}-{m2.group(2)}-{m2.group(3)}")
    return None

def _canon(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())

def resolve_var_name(ds: xr.Dataset, requested: Optional[str]) -> str:
    """Return a variable name present in ds. Try exact, then canonical (strip _/-, lower)."""
    names = list(ds.data_vars)
    if requested is None:
        # auto-pick: prefer names with 'water'/'mask' that are 2D or time+2D
        prefs = [v for v in names if ("water" in v.lower()) or ("mask" in v.lower())]
        for pool in (prefs, names):
            for v in pool:
                nd = ds[v].ndim
                if nd in (2, 3):
                    return v
        return names[0]
    if requested in names:
        return requested
    rc = _canon(requested)
    for v in names:
        if _canon(v) == rc:
            return v
    # last fallback to auto-pick
    return resolve_var_name(ds, None)

def load_one_as_da(p: Path, var_name: Optional[str], engine: str) -> xr.DataArray:
    ds = xr.open_dataset(p, engine=engine, decode_times=False, mask_and_scale=True)

    var = resolve_var_name(ds, var_name)
    da = ds[var]

    # Drop singleton time if present
    if ("time" in da.dims) and (da.sizes.get("time", 1) == 1):
        da = da.squeeze("time", drop=True)

    # Coerce to 2D spatial and rename to lat/lon
    if da.ndim != 2:
        spatial_dims = [d for d in da.dims if d != "time"][-2:]
        da = da.transpose(*spatial_dims)
    d1, d2 = da.dims[-2], da.dims[-1]
    da = da.rename({d1: "lat", d2: "lon"})

    # Attach coords if present
    for cand in ("lat", "latitude", "y"):
        if cand in ds and ds[cand].ndim == 1 and ds[cand].size == da.sizes["lat"]:
            da = da.assign_coords(lat=np.asarray(ds[cand].values))
            break
    for cand in ("lon", "longitude", "x"):
        if cand in ds and ds[cand].ndim == 1 and ds[cand].size == da.sizes["lon"]:
            da = da.assign_coords(lon=np.asarray(ds[cand].values))
            break

    # Add time from filename (fallback to decode_cf)
    t = parse_day_from_name(p)
    if t is None:
        try:
            t = xr.decode_cf(ds)["time"].to_index()[0]
        except Exception:
            t = pd.NaT
    ds.close()

    da = da.expand_dims(time=[t])
    da.name = var
    return da


def merge_crops_simple(crops: List[Path], out_nc: Path, var_name: Optional[str], engine: str) -> Path:
    assert crops, "No crop files passed to merge."

    arrays, times = [], []
    for p in crops:
        da = load_one_as_da(p, var_name, engine)
        arrays.append(da)
        times.append(da.coords["time"].values[0])

    # Drop NaT dates
    valid = [i for i, t in enumerate(times) if not (pd.isna(t) or (isinstance(t, np.datetime64) and str(t) == "NaT"))]
    if not valid:
        raise RuntimeError("Could not infer any valid dates from filenames/metadata.")
    arrays = [arrays[i] for i in valid]

    series = xr.concat(arrays, dim="time").sortby("time")

    # De-duplicate dates by keeping first
    if series.indexes.get("time", None) is not None and series.indexes["time"].duplicated().any():
        series = series.sel(time=~series.indexes["time"].duplicated(keep="first"))

    comp = dict(zlib=True, complevel=4)
    series.to_netcdf(out_nc, encoding={series.name: comp})
    return out_nc


# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Merge cropped daily CYGNSS NetCDFs into a single time-stacked file.")
    ap.add_argument("--crops-dir", type=Path, required=True, help="Directory containing okav_*.nc crops.")
    ap.add_argument("--out", type=Path, required=True, help="Output NetCDF path.")
    ap.add_argument("--pattern", default="okav_*.nc", help="Glob pattern within crops-dir (default: okav_*.nc)")
    ap.add_argument("--var", default=None, help="Variable name (e.g., watermask or water_mask).")
    ap.add_argument("--engine", choices=["netcdf4", "h5netcdf"], default="netcdf4", help="Backend engine.")
    return ap.parse_args()


def main():
    args = parse_args()
    crops = sorted(args.crops_dir.glob(args.pattern))
    if not crops:
        raise FileNotFoundError(f"No files matched {args.crops_dir / args.pattern}")
    print(f"Found {len(crops)} crop files. Merging…")
    out_path = merge_crops_simple(crops=crops, out_nc=args.out, var_name=args.var, engine=args.engine)
    print("Merged file:", out_path)

if __name__ == "__main__":
    main()
