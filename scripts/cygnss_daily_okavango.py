#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CYGNSS UC Berkeley Watermask (Daily v3.2) — Okavango subset + optional merge

Usage:
  python cygnss_okavango_download_daily.py \
    --out data/cygnss_okavango_daily \
    --merge

Notes:
- Requires: earthaccess, xarray, netCDF4 (or h5netcdf), numpy, pandas
- You will be prompted to authenticate (earthaccess).
"""

import os
import re
import argparse
from pathlib import Path
from typing import Tuple, List, Optional

import earthaccess as ea
import xarray as xr
import numpy as np
import pandas as pd

# ------------------ Defaults (can be overridden via CLI) ------------------
SHORT_NAME = "CYGNSS_L3_UC_BERKELEY_WATERMASK_DAILY_V3.2"
VERSION    = "3.2"
TEMPORAL   = ("2016-01-01", "2030-12-31")  # wide cap; filter via CLI as needed
OKAV_BBOX  = (21.696, -20.223, 24.103, -18.221)  # (min_lon, min_lat, max_lon, max_lat)

# ------------------ Helpers ------------------
def find_lat_lon_names(ds: xr.Dataset) -> Tuple[str, str]:
    lat = next(n for n in ds.dims if n.lower() in ("lat", "latitude", "y"))
    lon = next(n for n in ds.dims if n.lower() in ("lon", "longitude", "x"))
    return lat, lon

def pick_var_with_latlon(ds: xr.Dataset, lat: str, lon: str) -> str:
    preferred = [v for v in ds.data_vars if ("mask" in v.lower()) or ("water" in v.lower())]
    for v in preferred:
        if {lat, lon}.issubset(ds[v].dims):
            return v
    for v in ds.data_vars:
        if {lat, lon}.issubset(ds[v].dims):
            return v
    return list(ds.data_vars)[0]

def aoi_lat_slice(ds: xr.Dataset, lat_name: str, bbox) -> slice:
    lat_vals = ds[lat_name].values
    south, north = bbox[1], bbox[3]
    return slice(north, south) if (lat_vals[0] > lat_vals[-1]) else slice(south, north)

def day_from_granule(gran) -> Optional[pd.Timestamp]:
    """
    Parse a 'day stamp' from granule metadata/URL. Returns pandas Timestamp (UTC, date only).
    Accepts YYYY-MM-DD, YYYY_MM_DD, or YYYYMMDD anywhere in the name.
    """
    name = getattr(gran, "producer_granule_id", None)
    if not isinstance(name, str):
        umm = getattr(gran, "umm", None)
        if isinstance(umm, dict):
            name = umm.get("DataGranule", {}).get("ProducerGranuleId")
    if not isinstance(name, str):
        try:
            links = gran.data_links(access="external") or gran.data_links(access="direct") or []
            if links:
                name = str(links[0])
        except Exception:
            name = None
    if isinstance(name, str):
        # Try flexible patterns
        m = re.search(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})", name)
        if m:
            y, mo, d = map(int, m.groups())
            try:
                return pd.Timestamp(year=y, month=mo, day=d)
            except ValueError:
                return None
    return None

def crop_and_save_nc(
    fobj,
    granule,
    idx: int,
    out_dir: Path,
    bbox=OKAV_BBOX,
) -> Path:
    # Open WITHOUT time decoding
    last_err = None
    for engine in ("netcdf4", "h5netcdf"):
        try:
            ds = xr.open_dataset(fobj, engine=engine, decode_times=False, mask_and_scale=True)
            break
        except Exception as e:
            last_err = e
            ds = None
    if ds is None:
        raise last_err if last_err else RuntimeError("Failed to open dataset.")

    tstamp = day_from_granule(granule)
    stamp = (tstamp.strftime("%Y-%m-%d") if isinstance(tstamp, pd.Timestamp) else f"{idx:04d}")

    lat, lon = find_lat_lon_names(ds)
    lat_slice = aoi_lat_slice(ds, lat, bbox)
    sub = ds.sel({lon: slice(bbox[0], bbox[2]), lat: lat_slice})

    out_nc = out_dir / f"okav_{stamp}.nc"
    if out_nc.exists():  # avoid overwrite
        k = 1
        while (out_dir / f"okav_{stamp}_{k}.nc").exists():
            k += 1
        out_nc = out_dir / f"okav_{stamp}_{k}.nc"

    enc = {v: {"zlib": True, "complevel": 4} for v in sub.data_vars}
    sub.to_netcdf(out_nc, encoding=enc)
    ds.close()
    return out_nc

def merge_crops_to_nc(
    crops: List[Path],
    out_nc: Path,
) -> Path:
    """Merge daily crops into one NetCDF with a proper daily time axis."""
    if not crops:
        raise ValueError("No crop files to merge.")
    datasets = []
    times = []
    main_var = None
    lat_name = lon_name = None

    for p in crops:
        ds = xr.open_dataset(p, decode_times=False, mask_and_scale=True)
        if lat_name is None or lon_name is None:
            lat_name, lon_name = find_lat_lon_names(ds)
        v = pick_var_with_latlon(ds, lat_name, lon_name)
        if main_var is None:
            main_var = v

        # infer date from filename: okav_YYYY-MM-DD*.nc or okav_YYYYMMDD*.nc
        m = re.search(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})", p.name)
        if m:
            tstamp = pd.Timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        else:
            tstamp = pd.NaT

        da = ds[v].expand_dims(time=[tstamp])  # add time dim
        # standardize coordinate names for concat safety
        da = da.rename({lat_name: "lat", lon_name: "lon"})
        datasets.append(da)
        times.append(tstamp)
        ds.close()

    valid = [i for i, t in enumerate(times) if pd.notna(t)]
    if not valid:
        raise RuntimeError("Could not infer any valid YYYY-MM-DD timestamps from crop filenames.")
    series = xr.concat([datasets[i] for i in valid], dim="time").sortby("time")
    series.name = main_var

    # Write compressed NetCDF
    comp = dict(zlib=True, complevel=4)
    encoding = {series.name: comp}
    series.to_netcdf(out_nc, encoding=encoding)
    return out_nc

# ------------------ Main ------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Download CYGNSS daily watermask files, crop to AOI, and optionally merge.")
    ap.add_argument("--short-name", default=SHORT_NAME, help="Earthdata short name")
    ap.add_argument("--version", default=VERSION, help="Collection version (string)")
    ap.add_argument("--t0", default=TEMPORAL[0], help="Start date YYYY-MM-DD")
    ap.add_argument("--t1", default=TEMPORAL[1], help="End date YYYY-MM-DD (inclusive upper bound in search)")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("MINLON","MINLAT","MAXLON","MAXLAT"),
                    default=OKAV_BBOX, help="Bounding box")
    ap.add_argument("--out", type=Path, default=Path("data/cygnss_okavango_daily"), help="Base output directory")
    ap.add_argument("--merge", action="store_true", help="Merge cropped daily files into a single NetCDF")
    return ap.parse_args()

def main():
    args = parse_args()

    base_out: Path = args.out
    crops_dir: Path = base_out / "crops_nc"
    merged_nc: Path = base_out / "cygnss_okavango_daily_merged.nc"
    base_out.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    print("Logging in to Earthdata via earthaccess...")
    ea.login()

    print("Searching collection:", args.short_name, "version:", args.version)
    results = ea.search_data(
        short_name=args.short_name,
        version=args.version,
        temporal=(args.t0, args.t1),
        # bbox omitted at search; global files are cropped locally
    )
    print(f"Found {len(results)} daily files (server may paginate).")

    print("Opening authenticated file objects...")
    files = ea.open(results)

    written_nc: List[Path] = []
    for i, (fobj, gran) in enumerate(zip(files, results)):
        try:
            p = crop_and_save_nc(fobj, gran, i, crops_dir, bbox=tuple(args.bbox))
            print("Wrote:", p.name)
            written_nc.append(p)
        except Exception as e:
            print("Skipped one ->", e)

    written_nc = sorted(written_nc, key=lambda p: p.name)
    print(f"Per-day crops written: {len(written_nc)} -> {crops_dir}")

    if args.merge and written_nc:
        try:
            out_path = merge_crops_to_nc(written_nc, merged_nc)
            print("Merged file:", out_path)
        except Exception as e:
            print("Merge failed ->", e)

if __name__ == "__main__":
    main()
