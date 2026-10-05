"""precip.py – monthly CHIRPS precipitation over an Earth Engine geometry.

    df_p = compute_chirps(ee_region, area_m2, START, END,
                          csv_path=FIG_DIR / "chirps_monthly.csv", run=RUN_EE)

Columns: ``dataset, date (month start), ppt_mm_mean, ppt_km3_total, coverage``.
Months with no CHIRPS imagery (beyond the latest release) are NaN, not 0.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ee_monthly import make_monthly_ic, reduce_monthly_chunked, totals_to_df

CHIRPS_ID = "UCSB-CHG/CHIRPS/DAILY"


def ee_chirps_monthly(region, area_m2: float, start: str, end: str,
                      scale_m: float = 5566, min_coverage: float = 0.5) -> pd.DataFrame:
    import ee
    daily = ee.ImageCollection(CHIRPS_ID).filterBounds(region).filterDate(start, end)
    mic = make_monthly_ic(daily, lambda img: img.select("precipitation"), start, end,
                          period="day", is_rate=True)       # mm/day → pro-rated month total
    totals = reduce_monthly_chunked(mic, region, scale_m=scale_m, chunk_months=60)
    df = totals_to_df(totals, "CHIRPS_monthly", area_m2, prefix="ppt", min_coverage=min_coverage)
    ok = df["ppt_mm_mean"].notna()
    print(f"CHIRPS: {ok.sum()} valid months "
          f"({df.loc[ok, 'date'].min().date()} → {df.loc[ok, 'date'].max().date()})")
    return df


def compute_chirps(region, area_m2: float, start: str, end: str,
                   csv_path: str | Path, run: bool = True) -> pd.DataFrame:
    from src.cache import load_or_compute
    return load_or_compute(csv_path, lambda: ee_chirps_monthly(region, area_m2, start, end), run=run)
