"""et_products.py – monthly ET from Earth Engine products and local GLEAM netCDFs.

    from src.et_products import DATASETS, compute_et_products, et_wide

    df_et = compute_et_products(ee_region, area_m2, START, END,
                                csv_path=FIG_DIR / "et_monthly.csv", run=RUN_EE,
                                gleam_geom=block_union)
    et_mm_wide = et_wide(df_et, "et_mm_mean")

Output columns: ``dataset, date (month start), et_mm_mean, et_km3_total, coverage``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from src.ee_monthly import make_monthly_ic, reduce_monthly_chunked, totals_to_df

R_EARTH = 6.371e6
GLEAM_DIR = Path(__file__).resolve().parent.parent / "data" / "gleam"
GLEAM_NAME = "GLEAM_v42a"


# ---------------------------------------------------------------------------
# Converters (native time step → mm for that image)
# ---------------------------------------------------------------------------
def mod16_mm(img):
    return img.select("ET").multiply(0.1)              # kg m-2 per 8 days, scale 0.1


def terraclimate_mm(img):
    return img.select("aet").multiply(0.1)


def fldas_mm(img):
    import ee
    d0 = ee.Date(img.get("system:time_start"))
    secs = d0.advance(1, "month").difference(d0, "second")
    return img.select("Evap_tavg").multiply(secs)      # kg m-2 s-1 → mm / month


def era5_land_mm(img):
    return img.select("total_evaporation_sum").multiply(-1000)   # m (negative) → mm


def pml_mm(img):
    return img.select("ET").multiply(0.01).multiply(8)  # mm/day ×0.01, 8-day composite


def wapor_mm(img):
    import ee
    d0 = ee.Date(img.get("system:time_start"))
    d1_cand = d0.advance(10, "day")
    m_end = ee.Date.fromYMD(d0.get("year"), d0.get("month"), 1).advance(1, "month")
    d1 = ee.Date(ee.Algorithms.If(d1_cand.millis().lte(m_end.millis()), d1_cand, m_end))
    return img.select("L1-AETI-D").multiply(0.1).multiply(d1.difference(d0, "day"))


def ssebop_mm(img):
    return img.select("et")


DATASETS: list[dict] = [
    dict(name="MOD16A2GF_v61",   id="MODIS/061/MOD16A2GF", to_mm=mod16_mm, scale=500, start="2000-01-01"),
    dict(name="PML_v2_landET",    id="projects/pml_evapotranspiration/PML/OUTPUT/PML_V22a",
         to_mm=pml_mm, scale=500, start="2000-01-01"),
    dict(name="TerraClimate_aet", id="IDAHO_EPSCOR/TERRACLIMATE", to_mm=terraclimate_mm, scale=4638, start="1958-01-01"),
    dict(name="FLDAS_Evap",       id="NASA/FLDAS/NOAH01/C/GL/M/V001", to_mm=fldas_mm, scale=11132, start="1982-01-01"),
    dict(name="ERA5Land_totalET", id="ECMWF/ERA5_LAND/MONTHLY_AGGR", to_mm=era5_land_mm, scale=11132, start="1950-02-01"),
    dict(name="USGS_SSEBop",
         id="projects/earthengine-legacy/assets/projects/usgs-ssebop/modis_et_v5_monthly",
         to_mm=ssebop_mm, scale=1000, start="2003-01-01"),
    dict(name="WaPORv3_AETI",     id="FAO/WAPOR/3/L1_AETI_D", to_mm=wapor_mm, scale=248, start="2018-01-01",
         africa_only=True),
]


# ---------------------------------------------------------------------------
# Earth Engine products
# ---------------------------------------------------------------------------
def ee_et_products(region, area_m2: float, start: str, end: str,
                   datasets: Sequence[dict] = DATASETS, min_coverage: float = 0.5,
                   verbose: bool = True) -> pd.DataFrame:
    import ee
    dfs = []
    for ds in datasets:
        ds_start = max(pd.to_datetime(start), pd.to_datetime(ds["start"])).strftime("%Y-%m-%d")
        if pd.to_datetime(ds_start) >= pd.to_datetime(end):
            continue
        ic = ee.ImageCollection(ds["id"]).filterBounds(region).filterDate(ds_start, end)
        mic = make_monthly_ic(ic, ds["to_mm"], ds_start, end)
        totals = reduce_monthly_chunked(mic, region, scale_m=ds["scale"])
        df = totals_to_df(totals, ds["name"], area_m2, prefix="et", min_coverage=min_coverage)
        if verbose:
            ok = df["et_mm_mean"].notna()
            print(f"{ds['name']:18s}: {ok.sum():3d} valid months "
                  f"({df.loc[ok, 'date'].min().date() if ok.any() else '—'} → "
                  f"{df.loc[ok, 'date'].max().date() if ok.any() else '—'})")
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# GLEAM (local netCDFs)
# ---------------------------------------------------------------------------
def gleam_files(gleam_dir: str | Path = GLEAM_DIR) -> list[Path]:
    return sorted(Path(gleam_dir).glob("E_*_GLEAM_v4.2a_MO.nc"))


def gleam_et_over_geometry(files: Sequence[Path], geom, start: str, end: str,
                           name: str = GLEAM_NAME, min_coverage: float = 0.5) -> pd.DataFrame:
    """Monthly GLEAM ET over a shapely geometry (pixel-centre-in-polygon mask).

    Dates are snapped to **month start** (GLEAM stamps months at month end).
    ``et_mm_mean`` is the mean over valid (non-NaN) pixels in the mask,
    ``et_km3_total`` scales it to the full mask area, ``coverage`` is the
    valid-area fraction.
    """
    from shapely import contains_xy

    if not files:
        return pd.DataFrame(columns=["dataset", "date", "et_mm_mean", "et_km3_total", "coverage"])
    blon0, blat0, blon1, blat1 = geom.bounds
    dsets = []
    for f in files:
        ds = xr.open_dataset(f)
        lat_desc = ds["lat"].values[0] > ds["lat"].values[-1]
        lat_slice = slice(blat1 + 0.5, blat0 - 0.5) if lat_desc else slice(blat0 - 0.5, blat1 + 0.5)
        dsets.append(ds.sel(lon=slice(blon0 - 0.5, blon1 + 0.5), lat=lat_slice))
    da = xr.concat(dsets, dim="time").sortby("time")["E"]
    for d in dsets:
        d.close()

    lats, lons = da["lat"].values, da["lon"].values
    dlat = abs(float(np.median(np.diff(lats))))
    dlon = abs(float(np.median(np.diff(lons))))
    pixel_area = (R_EARTH ** 2 * np.deg2rad(dlon)
                  * np.abs(np.sin(np.deg2rad(lats + dlat / 2)) - np.sin(np.deg2rad(lats - dlat / 2))))
    area_2d = np.broadcast_to(pixel_area[:, None], (len(lats), len(lons)))
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    mask = contains_xy(geom, lon_grid, lat_grid)
    total_area = float(area_2d[mask].sum())

    vals_all = da.values
    times = pd.to_datetime(da["time"].values).to_period("M").to_timestamp()
    rows = []
    for t in range(vals_all.shape[0]):
        vals = vals_all[t]
        valid = mask & np.isfinite(vals)
        va = float(area_2d[valid].sum())
        if va <= 0 or total_area <= 0:
            mm, cov = np.nan, 0.0
        else:
            m3 = float(np.sum(vals[valid] * area_2d[valid]) / 1000.0)
            mm, cov = m3 / va * 1000.0, va / total_area
            if cov < min_coverage:
                mm = np.nan
        rows.append({"dataset": name, "date": times[t], "et_mm_mean": mm,
                     "et_km3_total": mm / 1000.0 * total_area / 1e9, "coverage": cov})
    da.close()
    df = pd.DataFrame(rows)
    df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] < pd.Timestamp(end))]
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def compute_et_products(region, area_m2: float, start: str, end: str,
                        csv_path: str | Path, run: bool = True,
                        gleam_geom=None, datasets: Sequence[dict] = DATASETS,
                        africa: bool = True, min_coverage: float = 0.5) -> pd.DataFrame:
    """All ET products (EE + GLEAM) over one geometry, cached to ``csv_path``.

    ``region`` is an ``ee.Geometry``; ``gleam_geom`` a shapely geometry for the
    local GLEAM files (skipped if ``None`` or no files present).  ``africa=False``
    drops Africa-only products (WaPOR).
    """
    from src.cache import load_or_compute

    dsets = [d for d in datasets if africa or not d.get("africa_only")]

    def _compute():
        df = ee_et_products(region, area_m2, start, end, dsets, min_coverage)
        if gleam_geom is not None:
            files = gleam_files()
            if files:
                dg = gleam_et_over_geometry(files, gleam_geom, start, end, min_coverage=min_coverage)
                print(f"{GLEAM_NAME:18s}: {dg['et_mm_mean'].notna().sum():3d} valid months")
                df = pd.concat([df, dg], ignore_index=True)
            else:
                print("No GLEAM files found — run: python scripts/download_gleam.py")
        df = clean_trailing_zeros(df)
        return df.sort_values(["dataset", "date"]).reset_index(drop=True)

    return load_or_compute(csv_path, _compute, run=run)


def clean_trailing_zeros(df: pd.DataFrame, col: str = "et_mm_mean", tiny: float = 1e-3) -> pd.DataFrame:
    """NaN-out leading/trailing zero rows per dataset (EE pads months outside coverage).

    With masked empty months this is mostly a no-op, kept as a safety net.
    """
    df = df.copy()
    for name, sub in df.groupby("dataset"):
        real = sub[sub[col] > tiny]
        if real.empty:
            continue
        outside = (df["dataset"] == name) & ((df["date"] < real["date"].min()) | (df["date"] > real["date"].max()))
        df.loc[outside, [c for c in df.columns if c.startswith("et_")]] = np.nan
    return df


def et_wide(df: pd.DataFrame, value: str = "et_mm_mean") -> pd.DataFrame:
    """Pivot to date × dataset (dates snapped to month start, duplicates averaged)."""
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.to_period("M").dt.to_timestamp()
    return d.pivot_table(index="date", columns="dataset", values=value, aggfunc="mean").sort_index()


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["et_mm_mean"].notna()]
    return (ok.groupby("dataset")
              .agg(first=("date", "min"), last=("date", "max"), n=("date", "size"),
                   mean_mm=("et_mm_mean", "mean"), coverage=("coverage", "mean"))
              .round({"mean_mm": 1, "coverage": 2}))
