#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CYGNSS UC Berkeley Watermask (v3.1) — Okavango subset + optional merge

Usage:
  python cygnss_okavango_download.py \
    --out cygnss_okavango_full \
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
SHORT_NAME = "CYGNSS_L3_UC_BERKELEY_WATERMASK_V3.1"
VERSION    = "3.1"
TEMPORAL   = ("2016-01-01", "2030-12-31")  # end is just a cap
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
    # last resort
    return list(ds.data_vars)[0]


def aoi_lat_slice(ds: xr.Dataset, lat_name: str, bbox) -> slice:
    lat_vals = ds[lat_name].values
    south, north = bbox[1], bbox[3]
    return slice(north, south) if (lat_vals[0] > lat_vals[-1]) else slice(south, north)


def decode_months_since(ds: xr.Dataset, time_name: str = "time") -> pd.DatetimeIndex:
    """Convert 'months since YYYY-MM-DD ...' to pandas datetime (no cftime)."""
    if time_name not in ds:
        raise ValueError("No time coordinate found to decode.")
    t = ds[time_name]
    units = t.attrs.get("units", "")
    if "months since" not in units:
        try:
            return xr.decode_cf(ds)[time_name].to_index()
        except Exception:
            return pd.RangeIndex(len(t))  # unlabeled fallback
    base_str = units.split("months since", 1)[1].strip().split()[0]  # 'YYYY-MM-DD'
    base = pd.Timestamp(base_str)
    offs = pd.to_numeric(t.values)
    return pd.DatetimeIndex([base + pd.DateOffset(months=int(m)) for m in offs])


def month_from_granule(gran) -> Optional[str]:
    """
    Return 'YYYY-MM' by parsing granule name/id/URL.
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
        m = re.search(r"(\d{4})[-_](\d{2})", name)
        if m:
            y, mo = int(m.group(1)), int(m.group(2))
            return f"{y:04d}-{mo:02d}"
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

    stamp = month_from_granule(granule) or f"{idx:04d}"

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
    """Merge monthly crops into one NetCDF with a proper time axis."""
    if not crops:
        raise ValueError("No crop files to merge.")
    # Open all, align on lat/lon, pick a main var, add time coord from filenames
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
        # infer time from filename: okav_YYYY-MM*.nc
        m = re.search(r"(\d{4})-(\d{2})", p.name)
        if m:
            tstamp = pd.Timestamp(f"{m.group(1)}-{m.group(2)}-01")
        else:
            tstamp = pd.NaT  # will drop later if NaT persists
        da = ds[v].expand_dims(time=[tstamp])  # add time dim
        # keep coords consistent
        da = da.rename({lat_name: "lat", lon_name: "lon"})  # standardized names
        datasets.append(da)
        times.append(tstamp)
        ds.close()

    # Drop entries with NaT time (if any)
    valid = [i for i, t in enumerate(times) if pd.notna(t)]
    if not valid:
        raise RuntimeError("Could not infer any valid YYYY-MM timestamps from crop filenames.")
    series = xr.concat([datasets[i] for i in valid], dim="time").sortby("time")
    series.name = main_var

    # Write compressed NetCDF
    comp = dict(zlib=True, complevel=4)
    encoding = {series.name: comp}
    series.to_netcdf(out_nc, encoding=encoding)
    return out_nc


# ------------------ Main ------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Download CYGNSS watermask monthly files, crop to AOI, and optionally merge.")
    ap.add_argument("--short-name", default=SHORT_NAME, help="Earthdata short name")
    ap.add_argument("--version", default=VERSION, help="Collection version (string)")
    ap.add_argument("--t0", default=TEMPORAL[0], help="Start date YYYY-MM-DD")
    ap.add_argument("--t1", default=TEMPORAL[1], help="End date YYYY-MM-DD (inclusive upper bound in search)")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("MINLON","MINLAT","MAXLON","MAXLAT"),
                    default=OKAV_BBOX, help="Bounding box")
    ap.add_argument("--out", type=Path, default=Path("cygnss_okavango_full"), help="Base output directory")
    ap.add_argument("--merge", action="store_true", help="Merge cropped monthly files into a single NetCDF")
    return ap.parse_args()


def main():
    args = parse_args()

    base_out: Path = args.out
    crops_dir: Path = base_out / "crops_nc"
    merged_nc: Path = base_out / "cygnss_okavango_full_merged.nc"
    base_out.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    print("Logging in to Earthdata via earthaccess...")
    ea.login()

    print("Searching collection:", args.short_name, "version:", args.version)
    results = ea.search_data(
        short_name=args.short_name,
        version=args.version,
        temporal=(args.t0, args.t1),
        # Files are global; bbox filter optional (kept off here).
    )
    print(f"Found {len(results)} monthly files (server may paginate).")

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
    print(f"Per-month crops written: {len(written_nc)} -> {crops_dir}")

    if args.merge and written_nc:
        try:
            out_path = merge_crops_to_nc(written_nc, merged_nc)
            print("Merged file:", out_path)
        except Exception as e:
            print("Merge failed ->", e)


if __name__ == "__main__":
    main()
