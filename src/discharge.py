"""discharge.py – Mohembo monthly inflow as volume / depth over a domain area."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MOHEMBO_MONTHLY = Path(__file__).resolve().parent.parent / "data" / "mohembo_1357100_Q_monthly_mean.csv"


def quadratic_gap_fill(s: pd.Series, max_gap: int = 2, n_side: int = 3) -> tuple[pd.Series, int]:
    """Fill short runs of NaN with a local 2nd-order polynomial.

    For each run of at most ``max_gap`` consecutive NaNs, a quadratic is fitted
    through up to ``n_side`` originally-valid points on each side and evaluated
    inside the gap.  Anchors are required on **both** sides, so nothing is
    extrapolated past the ends of the record, and longer gaps are left as NaN.
    With fewer than three anchors the fit drops to linear.

    Implemented with numpy rather than ``Series.interpolate(method="polynomial")``
    so it does not depend on SciPy: pandas routes polynomial interpolation
    through SciPy and enforces a minimum SciPy version, which differs between
    this project's environments.  Fitting locally also avoids the global
    quadratic spline that pandas would otherwise build over the whole series.

    Returns the filled series and the number of values filled.
    """
    y = s.to_numpy(dtype=float).copy()
    n = len(y)
    is_nan = np.isnan(y)
    if not is_nan.any():
        return pd.Series(y, index=s.index, name=s.name), 0

    valid_idx = np.flatnonzero(~is_nan)          # positions valid *before* any filling
    n_filled = 0
    i = 0
    while i < n:
        if not is_nan[i]:
            i += 1
            continue
        j = i
        while j < n and is_nan[j]:
            j += 1
        if (j - i) <= max_gap:
            left = valid_idx[valid_idx < i][-n_side:]
            right = valid_idx[valid_idx >= j][:n_side]
            if len(left) and len(right):
                anchors = np.concatenate([left, right])
                order = min(2, len(anchors) - 1)
                coef = np.polyfit(anchors.astype(float), y[anchors], order)
                y[i:j] = np.polyval(coef, np.arange(i, j, dtype=float))
                n_filled += j - i
        i = j
    return pd.Series(y, index=s.index, name=s.name), n_filled


def load_mohembo_monthly(area_m2: float, path: str | Path = MOHEMBO_MONTHLY,
                         max_gap_months: int = 2) -> pd.DataFrame:
    """Monthly mean Mohembo discharge → ``Qin_m3s, Qin_km3, Qin_mm``.

    Gaps of at most ``max_gap_months`` are filled by a local 2nd-order
    polynomial; longer gaps stay NaN.  ``Qin_mm`` is the inflow spread over
    ``area_m2``.
    """
    m = pd.read_csv(path, parse_dates=["month"]).rename(columns={"month": "date"})
    m["date"] = m["date"].dt.to_period("M").dt.to_timestamp()
    m = m.groupby("date", as_index=False)["Q_m3s_monthly_mean"].mean().sort_values("date")
    full = pd.date_range(m["date"].min(), m["date"].max(), freq="MS")
    m = m.set_index("date").reindex(full)
    m.index.name = "date"

    filled, n_filled = quadratic_gap_fill(m["Q_m3s_monthly_mean"], max_gap=max_gap_months)

    out = pd.DataFrame(index=m.index)
    out["Qin_m3s"] = filled
    days = out.index.to_series().dt.days_in_month.values
    out["Qin_km3"] = out["Qin_m3s"] * days * 86400 / 1e9
    out["Qin_mm"] = out["Qin_km3"] * 1e9 / area_m2 * 1000.0
    print(f"Mohembo: {out['Qin_m3s'].notna().sum()} months "
          f"({out.index.min().date()} → {out.index.max().date()}), gap-filled {n_filled}")
    return out.reset_index()
