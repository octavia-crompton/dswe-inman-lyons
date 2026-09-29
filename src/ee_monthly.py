"""ee_monthly.py – generic Earth Engine helpers: monthly composites + chunked area sums.

Two fixes relative to the earlier notebook copies:

* A month with **no imagery** becomes a fully-masked image, so it reduces to
  ``null`` → ``NaN`` (previously ``constant(0)`` → a fake 0 mm month).
* Masked pixels are **not** zero-filled.  Alongside the volume we also sum the
  *valid* pixel area, so area means are ``volume / valid_area`` and a
  ``coverage`` fraction is reported (previously ``unmask(0)`` counted masked
  water / barren pixels as 0 mm while keeping them in the denominator, which
  halved MOD16 over the delta).
"""
from __future__ import annotations

import ee
import numpy as np
import pandas as pd


def monthly_sequence(start_date, end_date):
    s, e = ee.Date(start_date), ee.Date(end_date)
    return ee.List.sequence(0, e.difference(s, "month").toInt().subtract(1))


def make_monthly_ic(ic, to_mm_fn, start_date, end_date, band_out: str = "mm"):
    """Sum ``to_mm_fn(img)`` over each calendar month in [start, end).

    ``to_mm_fn`` must return a single-band image in mm for that image's own
    time step.  Months with no images are fully masked.
    """
    start = ee.Date(start_date)

    def month_img(m):
        m = ee.Number(m)
        m0 = start.advance(m, "month")
        m1 = m0.advance(1, "month")
        sub = ic.filterDate(m0, m1).map(to_mm_fn)
        mm = ee.Image(ee.Algorithms.If(sub.size().gt(0), sub.sum(),
                                        ee.Image.constant(0).selfMask()))
        return (mm.rename(band_out)
                .set({"system:time_start": m0.millis(),
                      "system:index": m0.format("YYYYMM"),
                      "ym": m0.format("YYYY-MM")}))

    months = ee.List(ee.Algorithms.If(
        ee.Date(end_date).difference(start, "month").toInt().gt(0),
        monthly_sequence(start_date, end_date), ee.List([])))
    return ee.ImageCollection.fromImages(months.map(month_img))


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
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
