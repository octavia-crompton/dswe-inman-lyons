#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge cropped daily CYGNSS watermask files into one time-stacked NetCDF.
Handles: var-name differences, files with groups, and files with no top-level data_vars.
"""

import re
import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Iterable

import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset as NCDS  # for group/variable discovery


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

def _open_dataset_with_fallback(p: Path, engine: str, group: Optional[str] = None) -> Tuple[xr.Dataset, str]:
    tried = []
    try:
        ds = xr.open_dataset(p, engine=engine, group=group, decode_times=False, mask_and_scale=True)
        return ds, engine
    except Exception as e1:
        tried.append((engine, str(e1)))
    fallback = "h5netcdf" if engine == "netcdf4" else "netcdf4"
    try:
        ds = xr.open_dataset(p, engine=fallback, group=group, decode_times=False, mask_and_scale=True)
        return ds, fallback
    except Exception as e2:
        tried.append((fallback, str(e2)))
        msg = [f"Failed to open {p} (group='{group or '/'}') with both engines:"]
        for eng, err in tried:
            msg.append(f"  - {eng}: {err}")
        raise RuntimeError("\n".join(msg))

def _iter_groups(nc: NCDS, base: str = "") -> Iterable[Tuple[str, "netCDF4.Group"]]:
    yield base or "/", nc
    for name, grp in nc.groups.items():
        sub = f"{base}/{name}" if base else f"/{name}"
        yield from _iter_groups(grp, sub)

def discover_group_and_var(p: Path) -> Optional[Tuple[str, str, Tuple[str, ...]]]:
    """Return (group_path, var_name, dims) for the best candidate variable, or None if none found."""
    with NCDS(p, "r") as nc:
        candidates = []
        for gpath, grp in _iter_groups(nc):
            for vname, var in grp.variables.items():
                dims = tuple(var.dimensions)
                # Skip scalars or pure coord-like names
                if len(dims) == 0:
                    continue
                name_c = _canon(vname)
                score = 0
                if "water" in name_c: score += 3
                if "mask" in name_c:  score += 2
                # prefer 2D or time+2D
                if len(dims) == 2: score += 3
                if len(dims) == 3 and any(d.lower() == "time" for d in dims): score += 2
                # downrank obvious coords
                if vname.lower() in {"lat","latitude","y","lon","longitude","x","time"}: score -= 4
                # simple dtype preference: numeric arrays
                try:
                    if np.issubdtype(var.dtype, np.number):
                        score += 1
                except Exception:
                    pass
                candidates.append((score, gpath, vname, dims))
        if not candidates:
            return None
        # pick highest score; stable sort to prefer shallower groups
        candidates.sort(key=lambda t: (t[0], -len(t[1])), reverse=True)
        best = candidates[0]
        if best[0] < 0:
            return None
        return best[1], best[2], best[3]

def resolve_var_name(ds: xr.Dataset, requested: Optional[str]) -> Optional[str]:
    names = list(ds.data_vars)
    if not names:
        return None
    if requested is None:
        prefs = [v for v in names if ("water" in v.lower()) or ("mask" in v.lower())]
        for pool in (prefs, names):
            for v in pool:
                if ds[v].ndim in (2, 3):
                    return v
        return None
    if requested in names:
        return requested
    rc = _canon(requested)
    for v in names:
        if _canon(v) == rc:
            return v
    # fallback: any 2D/3D
    for v in names:
        if ds[v].ndim in (2, 3):
            return v
    return None

def load_one_as_da(p: Path, var_name: Optional[str], engine: str) -> Optional[xr.DataArray]:
    """Return a 2D DataArray with lat/lon plus a single 'time' coord; return None to skip file."""
    # 1) Try top-level
    try:
        ds, used_engine = _open_dataset_with_fallback(p, engine, group=None)
    except Exception as e:
        print(f"[warn] {p}: cannot open top-level dataset: {e}")
        ds, used_engine = None, None

    vname = None
    if ds is not None:
        vname = resolve_var_name(ds, var_name)
        if vname is not None:
            da = ds[vname]
        else:
            ds.close()
            ds = None  # force group discovery path

    # 2) If no top-level var, discover in groups
    group_used = None
    if ds is None:
        found = discover_group_and_var(p)
        if not found:
            print(f"[warn] {p}: no variables found in any group; skipping.")
            return None
        group_used, vname, _dims = found
        # strip leading "/"
        group_arg = None if group_used in ("/", "") else group_used.lstrip("/")
        ds, used_engine = _open_dataset_with_fallback(p, engine, group=group_arg)
        if vname not in ds:
            print(f"[warn] {p}: discovered var '{vname}' not present when opened (group={group_arg}); skipping.")
            ds.close()
            return None
        da = ds[vname]

    # Drop singleton time if present
    if ("time" in da.dims) and (da.sizes.get("time", 1) == 1):
        da = da.squeeze("time", drop=True)

    # Coerce to 2D spatial
    if da.ndim == 3:
        spatial_dims = [d for d in da.dims if d != "time"][-2:]
        da = da.transpose(*spatial_dims)
    elif da.ndim != 2:
        spatial_dims = [d for d in da.dims if d != "time"]
        if len(spatial_dims) >= 2:
            da = da.transpose(*spatial_dims[-2:])
        else:
            print(f"[warn] {p}: expected a 2D field (optionally with time), got dims {da.dims}; skipping.")
            ds.close()
            return None

    # Rename to lat/lon
    d1, d2 = da.dims[-2], da.dims[-1]
    da = da.rename({d1: "lat", d2: "lon"})

    # Attach coords if present
    for cand in ("lat", "latitude", "y"):
        if (cand in ds) and (ds[cand].ndim == 1) and (ds[cand].size == da.sizes["lat"]):
            da = da.assign_coords(lat=np.asarray(ds[cand].values))
            break
    for cand in ("lon", "longitude", "x"):
        if (cand in ds) and (ds[cand].ndim == 1) and (ds[cand].size == da.sizes["lon"]):
            da = da.assign_coords(lon=np.asarray(ds[cand].values))
            break

    # Time from filename (fallback to decode_cf)
    t = parse_day_from_name(p)
    if t is None:
        try:
            if "time" in ds:
                t = xr.decode_cf(ds)["time"].to_index()[0]
        except Exception:
            t = pd.NaT

    ds.close()

    # Finalize
    da = da.expand_dims(time=[t])
    if da.name is None or da.name == "":
        da.name = var_name if var_name else "watermask"
    return da


def merge_crops_simple(crops: List[Path], out_nc: Path, var_name: Optional[str], engine: str) -> Path:
    assert crops, "No crop files passed to merge."

    arrays, times = [], []
    skipped = 0
    for p in crops:
        da = load_one_as_da(p, var_name, engine)
        if da is None:
            skipped += 1
            continue
        arrays.append(da)
        times.append(da.coords["time"].values[0])

    if not arrays:
        raise RuntimeError("No usable variables found in any files (all empty or unsupported layout).")

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

    if skipped:
        print(f"[info] Skipped {skipped} file(s) with no usable data variable.")
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
