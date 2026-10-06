"""ee_monthly.py – generic Earth Engine helpers: pro-rated monthly composites + chunked area sums.

Monthly values are built as the **mean daily rate of the images overlapping the
calendar month, scaled to the month's length**:

    mm(month) = Σ_i rate_i · overlap_i / Σ_i overlap_i × days_in_month

where ``overlap_i`` is the number of days of image *i*'s period that fall in the
month (per pixel, only where the image is valid).  This handles three things
the earlier ``filterDate(m0, m1).sum()`` did not:

* **8-day composites (MOD16, PML) and dekads (WaPOR) that straddle month
  boundaries** are split between the months they cover; previously a month got
  3 or 4 whole composites (24 or 32 days), alternating ±20 % month to month.
* **WaPOR's third dekad** runs to the end of the month (11 days in 31-day
  months); previously it was counted as 10.
* A **masked composite** scales the remaining ones up rather than silently
  lowering the month; a month with fewer than ``min_days_frac`` of its days
  covered (e.g. the partial month at the end of a record) is masked.

Two other fixes relative to the first notebook copies are kept: a month with no
imagery is fully masked (→ NaN, not 0), and masked pixels are excluded from area
means by also summing the valid pixel area (``reduce_monthly_chunked``).
"""
from __future__ import annotations

import calendar
import datetime as _dt

import ee
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Period rules – client-side (unit-testable) and server-side (ee.Date) twins
# ---------------------------------------------------------------------------
def period_end_8day(t0: _dt.datetime) -> _dt.datetime:
    """End of a MODIS-style 8-day composite starting at ``t0``.

    Composites restart on 1 January, so the last one of the year (DOY 361)
    covers only 5 days (6 in leap years).
    """
    return min(t0 + _dt.timedelta(days=8), _dt.datetime(t0.year + 1, 1, 1))


def period_end_dekad(t0: _dt.datetime) -> _dt.datetime:
    """End of a WaPOR dekad starting at ``t0``: 1–10, 11–20, 21–end of month."""
    if t0.day >= 21:
        return _dt.datetime(t0.year + (t0.month == 12), t0.month % 12 + 1, 1)
    return t0 + _dt.timedelta(days=10)


def period_end_month(t0: _dt.datetime) -> _dt.datetime:
    return _dt.datetime(t0.year + (t0.month == 12), t0.month % 12 + 1, 1)


def period_end_day(t0: _dt.datetime) -> _dt.datetime:
    return t0 + _dt.timedelta(days=1)


def overlap_days(t0: _dt.datetime, t1: _dt.datetime, m0: _dt.datetime, m1: _dt.datetime) -> float:
    """Days of [t0, t1) that fall inside [m0, m1)."""
    return max(0.0, (min(t1, m1) - max(t0, m0)).total_seconds() / 86400.0)


def prorate_month(rates, overlaps, days_in_month: int, min_days_frac: float = 0.75) -> float:
    """Client-side twin of the server-side month formula (for tests).

    ``rates`` in mm/day (NaN = masked), ``overlaps`` in days.  Returns NaN when
    the valid overlaps cover less than ``min_days_frac`` of the month.
    """
    rates, overlaps = np.asarray(rates, float), np.asarray(overlaps, float)
    ok = np.isfinite(rates) & (overlaps > 0)
    den = overlaps[ok].sum()
    if den < min_days_frac * days_in_month:
        return np.nan
    return float((rates[ok] * overlaps[ok]).sum() / den * days_in_month)


def _ee_period_end(period: str):
    """Server-side period-end function matching the client-side rules above."""
    def end_8day(t0):
        return ee.Date(ee.Algorithms.If(
            t0.advance(8, "day").millis().lte(ee.Date.fromYMD(t0.get("year").add(1), 1, 1).millis()),
            t0.advance(8, "day"), ee.Date.fromYMD(t0.get("year").add(1), 1, 1)))

    def end_dekad(t0):
        month_end = ee.Date.fromYMD(t0.get("year"), t0.get("month"), 1).advance(1, "month")
        return ee.Date(ee.Algorithms.If(ee.Number(t0.get("day")).gte(21), month_end, t0.advance(10, "day")))

    return {
        "8day": end_8day,
        "dekad": end_dekad,
        "month": lambda t0: ee.Date.fromYMD(t0.get("year"), t0.get("month"), 1).advance(1, "month"),
        "day": lambda t0: t0.advance(1, "day"),
    }[period]


PERIOD_MAX_DAYS = {"8day": 8, "dekad": 11, "month": 31, "day": 1}


# ---------------------------------------------------------------------------
# Monthly composites
# ---------------------------------------------------------------------------
def n_months(start_date, end_date) -> int:
    """Number of calendar months in [start, end), counted client-side.

    ``ee.Date.difference(..., "month")`` is fractional (it uses a mean month
    length), so truncating it drops the last month whenever the span falls just
    under a whole number, e.g. 2019-01-01 → 2020-01-01 gave 11.
    """
    return (pd.Period(pd.Timestamp(end_date), "M") - pd.Period(pd.Timestamp(start_date), "M")).n


def monthly_sequence(start_date, end_date):
    return ee.List.sequence(0, n_months(start_date, end_date) - 1)


def make_monthly_ic(ic, to_mm_fn, start_date, end_date, period: str = "month",
                    is_rate: bool = False, min_days_frac: float = 0.75,
                    band_out: str = "mm"):
    """Pro-rated monthly totals (mm) for each calendar month in [start, end).

    ``to_mm_fn(img)`` returns a single-band image: the total over the image's
    own period (``is_rate=False``) or a daily rate in mm/day (``is_rate=True``).
    ``period`` selects the period rule: ``"month"``, ``"8day"``, ``"dekad"`` or
    ``"day"``.  Months with no imagery, or with valid coverage below
    ``min_days_frac`` of their days, are masked.
    """
    start = ee.Date(start_date)
    period_end = _ee_period_end(period)
    look_back = PERIOD_MAX_DAYS[period]

    def month_img(m):
        m = ee.Number(m)
        m0 = start.advance(m, "month")
        m1 = m0.advance(1, "month")
        days_in_month = m1.difference(m0, "day")

        def per_image(img):
            t0 = ee.Date(img.get("system:time_start"))
            t1 = period_end(t0)
            period_days = t1.difference(t0, "day")
            lo = ee.Date(ee.Algorithms.If(t0.millis().gt(m0.millis()), t0, m0))
            hi = ee.Date(ee.Algorithms.If(t1.millis().lt(m1.millis()), t1, m1))
            overlap = hi.difference(lo, "day").max(0)
            val = ee.Image(to_mm_fn(img))
            rate = val if is_rate else val.divide(period_days)
            # explicit float casts: images with zero overlap would otherwise get a
            # constant-typed band and break the collection's type homogeneity
            contrib = rate.multiply(overlap).toFloat().rename("contrib")
            days = ee.Image.constant(overlap).toFloat().updateMask(rate.mask()).rename("days")
            return contrib.addBands(days)

        sub = ic.filterDate(m0.advance(-look_back, "day"), m1).map(per_image)

        def prorated():
            num = sub.select("contrib").sum()
            den = sub.select("days").sum()
            mm = num.divide(den).multiply(days_in_month)
            return mm.updateMask(den.gte(days_in_month.multiply(min_days_frac)))

        mm = ee.Image(ee.Algorithms.If(sub.size().gt(0), prorated(),
                                        ee.Image.constant(0).selfMask()))
        return (mm.rename(band_out)
                .set({"system:time_start": m0.millis(),
                      "system:index": m0.format("YYYYMM"),
                      "ym": m0.format("YYYY-MM")}))

    if n_months(start_date, end_date) <= 0:
        return ee.ImageCollection([])
    return ee.ImageCollection.fromImages(monthly_sequence(start_date, end_date).map(month_img))


# ---------------------------------------------------------------------------
# Reduction
# ---------------------------------------------------------------------------
def reduce_monthly_chunked(monthly_ic, region, scale_m, band: str = "mm",
                           chunk_months: int = 24, tile_scale: int = 8) -> dict:
    """Sum ``band`` (mm) × pixel area over ``region`` for every image.

    Returns ``{YYYYMM: {"m3": float|None, "area_m2": float|None}}``; ``None``
    means the month was fully masked over the region.
    """
    n = int(monthly_ic.size().getInfo())
    if n == 0:
        return {}
    imgs = monthly_ic.toList(n)
    out: dict[str, dict] = {}

    def to_vol(img):
        img = ee.Image(img)
        idx = ee.String(img.get("system:index"))
        val = img.select(band)
        m3 = val.divide(1000).multiply(ee.Image.pixelArea()).rename("m3")
        area = ee.Image.pixelArea().updateMask(val.mask()).rename("area")
        return m3.addBands(area).set("system:index", idx)

    for i in range(0, n, chunk_months):
        sub = ee.ImageCollection.fromImages(imgs.slice(i, min(i + chunk_months, n)))
        d = sub.map(to_vol).toBands().reduceRegion(
            reducer=ee.Reducer.sum(), geometry=region,
            scale=scale_m, maxPixels=1e13, tileScale=tile_scale).getInfo()
        for k, v in d.items():
            ym, _, kind = k.partition("_")
            out.setdefault(ym, {})[kind] = (None if v is None else float(v))
    return out


def totals_to_df(totals: dict, dataset: str, area_m2: float, prefix: str,
                 min_coverage: float = 0.5) -> pd.DataFrame:
    """Turn ``reduce_monthly_chunked`` output into a tidy frame.

    Columns: ``dataset, date, <prefix>_mm_mean, <prefix>_km3_total, coverage``.
    ``<prefix>_mm_mean`` is the mean over *valid* pixels; ``<prefix>_km3_total``
    scales that mean to the full ``area_m2`` (i.e. assumes masked pixels behave
    like the valid ones).  Months with ``coverage < min_coverage`` are NaN.
    """
    rows = []
    for ym, d in totals.items():
        date = pd.to_datetime(ym + "01", format="%Y%m%d")
        m3, va = d.get("m3"), d.get("area")
        if m3 is None or va is None or va <= 0:
            mm, cov = np.nan, 0.0
        else:
            mm, cov = m3 / va * 1000.0, va / area_m2
            if cov < min_coverage:
                mm = np.nan
        rows.append({"dataset": dataset, "date": date,
                     f"{prefix}_mm_mean": mm,
                     f"{prefix}_km3_total": mm / 1000.0 * area_m2 / 1e9,
                     "coverage": cov})
    if not rows:
        return pd.DataFrame(columns=["dataset", "date", f"{prefix}_mm_mean", f"{prefix}_km3_total", "coverage"])
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
