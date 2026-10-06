"""balance.py – assemble a monthly water balance on a continuous month grid.

    bal = assemble_balance(df_et, df_chirps, df_tws, qin=df_q,
                           extra={"NW_NE_diff_cm": df_nw_ne["NW_NE_diff_cm"]},
                           start=START, ds_scheme="centered")
    bal.df            # one row per month, NaN where a term is missing
    bal.et_km3_wide   # per-product ET (km³) on the same grid

Conventions
-----------
* Every input is snapped to **month start**; duplicate months are averaged.
* ``dS`` (storage change assigned to month *t*) uses a **centred** difference
  ``(S[t+1] − S[t−1]) / 2`` by default, because a GRACE solution is the mean
  state over its month while P, ET, Q are month totals.  ``ds_scheme="backward"``
  gives ``S[t] − S[t−1]``.  Either way the difference is taken on the
  continuous grid.
* Before differencing, storage gaps of at most ``fill_tws_gap_months``
  consecutive months (default 1) are linearly interpolated in time; longer
  gaps, such as the 11-month GRACE/GRACE-FO gap, are never bridged.  Filled
  months are flagged in ``TWS_filled``, and ``dS_uses_filled_tws`` marks every
  storage change that depends on one.
* ET is the median of the products available each month, but only when at
  least ``min_et_products`` contribute (default 4); with fewer, ET and the
  residual are NaN and the month is not a closure month.  Product coverage
  thins late in the record (7–8 products to 2024, 4 in 2025, 3 in 2026).
* Residual: ``resid = Qin + P − ET − dS`` (Qin = 0 when no gauge is given).
  For the delta this is the unmeasured net outflow + storage-model error;
  for a closed dry cell it should be ~0.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.et_products import et_wide


def to_month_start(s) -> pd.Series | pd.DatetimeIndex:
    """Snap datetimes to the first of the month (works for Series and Index)."""
    if isinstance(s, (pd.DatetimeIndex, pd.Index)):
        return pd.DatetimeIndex(pd.to_datetime(s)).to_period("M").to_timestamp()
    return pd.to_datetime(s).dt.to_period("M").dt.to_timestamp()


def _monthly(obj, name: str | None = None) -> pd.Series | pd.DataFrame:
    """Series/DataFrame (or frame with a 'date' column) → month-start index, deduped."""
    if isinstance(obj, pd.DataFrame) and "date" in obj.columns:
        obj = obj.set_index("date")
    obj = obj.copy()
    obj.index = to_month_start(obj.index)
    obj = obj.groupby(level=0).mean(numeric_only=True)
    if name is not None and isinstance(obj, pd.Series):
        obj = obj.rename(name)
    return obj.sort_index()


@dataclass
class Balance:
    df: pd.DataFrame
    et_km3_wide: pd.DataFrame
    et_mm_wide: pd.DataFrame
    area_m2: float
    ds_scheme: str
    has_qin: bool
    extra_cols: list[str] = field(default_factory=list)
    tws_fill_months: int = 0
    min_et_products: int = 1

    @property
    def flux_cols(self) -> list[str]:
        return (["Qin_km3"] if self.has_qin else []) + ["P_km3"]

    def closure(self) -> pd.DataFrame:
        """Rows with every balance term present."""
        return self.df[self.df["has_closure"]]

    def summary(self) -> str:
        d = self.closure()
        lines = [f"Balance grid: {len(self.df)} months ({self.df.index.min().date()} → "
                 f"{self.df.index.max().date()}), {len(d)} with full closure",
                 f"ΔS scheme: {self.ds_scheme};  area = {self.area_m2 / 1e6:,.0f} km²",
                 f"TWS gaps ≤ {self.tws_fill_months} month interpolated: "
                 f"{int(self.df['TWS_filled'].sum())} months "
                 f"({int(self.df['dS_uses_filled_tws'].sum())} ΔS months depend on them)",
                 f"ET median requires ≥ {self.min_et_products} products: "
                 f"{int(self.df['et_too_few_products'].sum())} months with ET data dropped",
                 "Mean over closure months (mm/month):"]
        for c, lab in [("P_mm", "P"), ("ET_mm", "ET"), ("dS_mm", "ΔS"), ("Qin_mm", "Qin"), ("resid_mm", "resid")]:
            if c in d:
                lines.append(f"  {lab:6s} {d[c].mean():7.1f}")
        return "\n".join(lines)


def assemble_balance(
    et_df: pd.DataFrame,
    chirps_df: pd.DataFrame,
    tws_df: pd.DataFrame,
    area_m2: float,
    qin: pd.DataFrame | None = None,
    extra: dict[str, pd.Series] | None = None,
    start: str = "2002-04-01",
    end: str | None = None,
    ds_scheme: str = "centered",
    fill_tws_gap_months: int = 1,
    min_et_products: int = 4,
) -> Balance:
    """Build the monthly balance table.

    Parameters
    ----------
    et_df : tidy ET frame (``dataset, date, et_mm_mean, et_km3_total``).
    chirps_df : ``date, ppt_mm_mean, ppt_km3_total``.
    tws_df : ``TWS_cm, TWS_km3`` indexed by month start (from ``grace_blocks.domain_tws``).
    area_m2 : domain area, used for cm ↔ mm ↔ km³ consistency checks only.
    qin : optional ``date, Qin_m3s, Qin_km3, Qin_mm``.
    extra : optional ``{column_name: Series}`` (e.g. TWS head differences) to
        carry along on the same grid.
    fill_tws_gap_months : interpolate storage gaps of at most this many
        consecutive months before differencing (0 disables).
    min_et_products : the ET median is used only in months where at least this
        many products contribute; otherwise ET (and the residual) are NaN.
    """
    if ds_scheme not in ("centered", "backward"):
        raise ValueError("ds_scheme must be 'centered' or 'backward'")

    et_km3 = et_wide(et_df, "et_km3_total")
    et_mm = et_wide(et_df, "et_mm_mean")
    p = _monthly(chirps_df)
    s = _monthly(tws_df)
    parts = {
        "P_mm": p["ppt_mm_mean"], "P_km3": p["ppt_km3_total"],
        "ET_mm": et_mm.median(axis=1, skipna=True), "ET_km3": et_km3.median(axis=1, skipna=True),
        "n_et_products": et_km3.notna().sum(axis=1),
        "TWS_cm": s["TWS_cm"], "TWS_km3": s["TWS_km3"],
    }
    has_qin = qin is not None
    if has_qin:
        q = _monthly(qin)
        for c in ("Qin_m3s", "Qin_km3", "Qin_mm"):
            parts[c] = q[c]
    extra_cols = []
    for name, ser in (extra or {}).items():
        parts[name] = _monthly(ser, name)
        extra_cols.append(name)

    df = pd.concat(parts, axis=1).sort_index()
    last = pd.Timestamp(end) if end else df.index.max()
    grid = pd.date_range(pd.Timestamp(start), last, freq="MS")
    df = df.reindex(grid)
    df.index.name = "date"

    too_few = df["ET_km3"].notna() & (df["n_et_products"] < min_et_products)
    df.loc[too_few, ["ET_mm", "ET_km3"]] = np.nan
    df["et_too_few_products"] = too_few

    df["TWS_filled"] = False
    for col in ("TWS_cm", "TWS_km3"):
        df[col], filled = _fill_short_gaps(df[col], fill_tws_gap_months)
        df["TWS_filled"] |= filled

    f = df["TWS_filled"]
    if ds_scheme == "centered":
        df["dS_cm"] = (df["TWS_cm"].shift(-1) - df["TWS_cm"].shift(1)) / 2
        df["dS_km3"] = (df["TWS_km3"].shift(-1) - df["TWS_km3"].shift(1)) / 2
        uses = f.shift(-1, fill_value=False) | f.shift(1, fill_value=False)
    else:
        df["dS_cm"] = df["TWS_cm"] - df["TWS_cm"].shift(1)
        df["dS_km3"] = df["TWS_km3"] - df["TWS_km3"].shift(1)
        uses = f | f.shift(1, fill_value=False)
    df["dS_uses_filled_tws"] = uses & df["dS_cm"].notna()
    df["dS_mm"] = df["dS_cm"] * 10.0

    qin_km3 = df["Qin_km3"] if has_qin else 0.0
    qin_mm = df["Qin_mm"] if has_qin else 0.0
    df["resid_km3"] = qin_km3 + df["P_km3"] - df["ET_km3"] - df["dS_km3"]
    df["resid_mm"] = qin_mm + df["P_mm"] - df["ET_mm"] - df["dS_mm"]
    need = ["P_km3", "ET_km3", "dS_km3"] + (["Qin_km3"] if has_qin else [])
    df["has_closure"] = df[need].notna().all(axis=1)

    return Balance(df=df, et_km3_wide=et_km3.reindex(grid), et_mm_wide=et_mm.reindex(grid),
                   area_m2=area_m2, ds_scheme=ds_scheme, has_qin=has_qin, extra_cols=extra_cols,
                   tws_fill_months=fill_tws_gap_months, min_et_products=min_et_products)


def _fill_short_gaps(s: pd.Series, max_gap: int) -> tuple[pd.Series, pd.Series]:
    """Linearly interpolate (in time) interior runs of at most `max_gap` NaNs.

    Returns the filled series and a boolean mask of the values that were filled.
    Leading/trailing gaps and longer runs stay NaN.
    """
    is_nan = s.isna()
    if max_gap <= 0 or not is_nan.any():
        return s, pd.Series(False, index=s.index)
    run_len = is_nan.groupby((~is_nan).cumsum()).transform("sum")
    filled = s.interpolate(method="time", limit_area="inside")
    filled[is_nan & (run_len > max_gap)] = np.nan
    return filled, is_nan & filled.notna()


def monthly_climatology(df: pd.DataFrame, cols: dict[str, str]) -> pd.DataFrame:
    """Mean and std by calendar month for ``{column: label}``."""
    g = df.groupby(df.index.month)
    out = {}
    for col, lab in cols.items():
        out[f"{lab}_mean"] = g[col].mean()
        out[f"{lab}_std"] = g[col].std()
    clim = pd.DataFrame(out)
    clim.index.name = "month"
    return clim


def write_outputs(bal: Balance, fig_dir, geom_tag: str, blocks, qin: pd.DataFrame | None = None) -> None:
    """Save mass_balance.csv, per-product ET (km³) and geometry_info.txt."""
    from pathlib import Path
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    bal.df.reset_index().to_csv(fig_dir / "mass_balance.csv", index=False)
    bal.et_km3_wide.reset_index().to_csv(fig_dir / "et_products_km3.csv", index=False)
    if qin is not None:
        qin.to_csv(fig_dir / "mohembo_monthly.csv", index=False)
    with open(fig_dir / "geometry_info.txt", "w") as fh:
        fh.write(f"GEOM = {geom_tag}\n")
        fh.write(f"area_km2 = {bal.area_m2 / 1e6:.1f}\n")
        fh.write(f"dS_scheme = {bal.ds_scheme}\n")
        fh.write(f"tws_fill_gap_months = {bal.tws_fill_months}\n")
        fh.write(f"min_et_products = {bal.min_et_products}\n")
        fh.write(f"n_blocks = {len(blocks)}\n")
        for b in blocks:
            fh.write(f"  {b['block_id']} ({b['quadrant']}): lon {b['lon0']:.2f}–{b['lon1']:.2f}, "
                     f"lat {b['lat0']:.2f}–{b['lat1']:.2f}\n")
    print(f"Outputs written to {fig_dir.resolve()}")
