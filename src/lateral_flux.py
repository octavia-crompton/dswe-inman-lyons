"""lateral_flux.py – tunable lateral-flux terms driven by GRACE head differences.

Model (per ET product)::

    Qin + P − ET + Σ_k α_k · x_k + c ≈ ΔS

where ``x_k`` are monthly TWS differences between adjacent mascon blocks (cm)
and ``α_k`` (km³ month⁻¹ cm⁻¹) are fitted.  The fit is done on **monthly**
residuals, with an intercept ``c`` that absorbs any constant bias of the ET
product.  The reference ("null") model is the intercept alone, so the reported
RMSE improvement measures what the head-difference term explains beyond a
constant bias.

Earlier notebook versions regressed the *cumulative* residual on the
*cumulative* head difference.  Because the head differences have a non-zero
mean, their cumulative sum is essentially a straight line, so that α mostly
fitted the drift of each ET product and flipped sign from product to product.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from src.balance import Balance


def _rmse(a: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean(a ** 2)))


def fit_lateral(resid: np.ndarray, X: np.ndarray, bounds=None, fit_intercept: bool = True) -> dict:
    """Least-squares fit of ``resid + X·α + c ≈ 0``.

    ``bounds`` is ``(lo, hi)`` per column of ``X`` (``None`` → unbounded), e.g.
    ``([-np.inf, -np.inf], [np.inf, 0])`` to force β ≤ 0.
    """
    X = np.atleast_2d(np.asarray(X, float))
    if X.shape[0] != len(resid):
        X = X.T
    ok = np.isfinite(resid) & np.all(np.isfinite(X), axis=1)
    r, A = np.asarray(resid, float)[ok], X[ok]
    if fit_intercept:
        A = np.column_stack([A, np.ones(len(r))])
    lo = [-np.inf] * X.shape[1] if bounds is None else list(bounds[0])
    hi = [np.inf] * X.shape[1] if bounds is None else list(bounds[1])
    if fit_intercept:
        lo, hi = lo + [-np.inf], hi + [np.inf]
    sol = lsq_linear(A, -r, bounds=(lo, hi))
    alpha = sol.x[: X.shape[1]]
    c = float(sol.x[-1]) if fit_intercept else 0.0
    rmse_null = _rmse(r - r.mean()) if fit_intercept else _rmse(r)
    rmse_fit = _rmse(r + A @ sol.x)
    return dict(alpha=alpha, intercept=c, rmse_null=rmse_null, rmse_fit=rmse_fit, n=int(ok.sum()),
                improvement_pct=(1 - rmse_fit / rmse_null) * 100 if rmse_null > 0 else 0.0)


def alpha_sensitivity(resid: np.ndarray, x: np.ndarray, alphas: np.ndarray,
                      fit_intercept: bool = True) -> np.ndarray:
    """RMSE of ``resid + α·x + c(α)`` for each α (c chosen optimally)."""
    ok = np.isfinite(resid) & np.isfinite(x)
    r, xv = np.asarray(resid, float)[ok], np.asarray(x, float)[ok]
    out = []
    for a in alphas:
        e = r + a * xv
        if fit_intercept:
            e = e - e.mean()
        out.append(_rmse(e))
    return np.asarray(out)


def rmse_surface(resid, x1, x2, agrid, bgrid, fit_intercept: bool = True) -> np.ndarray:
    """RMSE over an (β × α) grid for the two-term model."""
    ok = np.isfinite(resid) & np.isfinite(x1) & np.isfinite(x2)
    r, a1, a2 = (np.asarray(v, float)[ok] for v in (resid, x1, x2))
    grid = np.empty((len(bgrid), len(agrid)))
    for bi, b in enumerate(bgrid):
        e0 = r + b * a2
        for ai, a in enumerate(agrid):
            e = e0 + a * a1
            if fit_intercept:
                e = e - e.mean()
            grid[bi, ai] = _rmse(e)
    return grid


def per_model_fits(bal: Balance, x_cols: list[str], bounds=None, fit_intercept: bool = True,
                   min_months: int = 24) -> tuple[pd.DataFrame, dict]:
    """Fit the lateral-flux model separately for every ET product.

    Returns ``(results, plot_data)``: ``results`` has one row per product with
    ``alpha_<x>`` columns, ``intercept``, ``RMSE_null_km3``, ``RMSE_fit_km3``,
    ``improvement_pct``, ``n``; ``plot_data[product]`` holds cumulative series
    (``cum_ds, cum_base, cum_tuned``) for plotting, computed over the months
    used in the fit.
    """
    df = bal.df
    results, plot_data = [], {}
    flux_base = df[bal.flux_cols].sum(axis=1, min_count=len(bal.flux_cols))
    for et_name in sorted(bal.et_km3_wide.columns):
        et = bal.et_km3_wide[et_name]
        flux = flux_base - et
        resid = flux - df["dS_km3"]
        X = df[x_cols]
        ok = resid.notna() & X.notna().all(axis=1)
        if ok.sum() < min_months:
            continue
        fit = fit_lateral(resid[ok].values, X[ok].values, bounds=bounds, fit_intercept=fit_intercept)
        row = {"ET_model": et_name}
        for k, xc in enumerate(x_cols):
            row[f"alpha_{xc}"] = fit["alpha"][k]
        row.update(intercept=fit["intercept"], RMSE_null_km3=fit["rmse_null"],
                   RMSE_fit_km3=fit["rmse_fit"], improvement_pct=fit["improvement_pct"], n=fit["n"])
        results.append(row)

        d = df.loc[ok]
        corr = sum(fit["alpha"][k] * d[xc] for k, xc in enumerate(x_cols))
        plot_data[et_name] = dict(
            cum_ds=d["dS_km3"].cumsum(),
            cum_base=flux[ok].cumsum(),
            cum_tuned=(flux[ok] + corr).cumsum(),
            cum_tuned_bias=(flux[ok] + corr + fit["intercept"]).cumsum(),
            alpha=fit["alpha"], intercept=fit["intercept"],
            rmse_null=fit["rmse_null"], rmse_fit=fit["rmse_fit"],
        )
    res = pd.DataFrame(results)
    if not res.empty:
        res = res.sort_values("RMSE_fit_km3").reset_index(drop=True)
    return res, plot_data


def summarize_fits(results: pd.DataFrame, label: str) -> None:
    if results.empty:
        print(f"{label}: no ET product had enough months.")
        return
    a_cols = [c for c in results.columns if c.startswith("alpha_")]
    print(f"── {label} ──")
    print(results.to_string(index=False, float_format="%.4f"))
    med = results["improvement_pct"].median()
    print(f"\nMedian RMSE improvement over bias-only null: {med:.1f}%")
    for c in a_cols:
        sgn = np.sign(results[c])
        agree = (sgn == sgn.iloc[0]).all()
        print(f"  {c}: median {results[c].median():+.4f} km³/mo/cm, "
              f"{'consistent sign' if agree else 'SIGN FLIPS between ET products'}")
    if med < 5:
        print("  → the head-difference term explains almost nothing beyond a constant bias.")
    elif not all((np.sign(results[c]) == np.sign(results[c]).iloc[0]).all() for c in a_cols):
        print("  → coefficients change sign between ET products: no consistent lateral-flux signal.")
    else:
        print("  → a consistent lateral-flux signal is present across ET products.")
