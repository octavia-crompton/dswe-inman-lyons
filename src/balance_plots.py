"""balance_plots.py – standard figures for the mass-balance notebooks."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.balance import Balance
from src.lateral_flux import alpha_sensitivity, rmse_surface

MONTH_LABELS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
C_P, C_ET, C_DS, C_Q, C_RES = "#31a354", "#e6550d", "#c51b7d", "#0570b0", "grey"


def _save(fig, save):
    if save is not None:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=160, bbox_inches="tight")


def _grid(n, ncols=2, h=3.5, sharex=True, sharey=False):
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, h * nrows), sharex=sharex, sharey=sharey)
    axes = np.atleast_1d(axes).flatten()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def plot_et_products(et_mm_wide: pd.DataFrame, title: str, save=None):
    fig, ax = plt.subplots(figsize=(13, 5))
    for c in et_mm_wide.columns:
        ax.plot(et_mm_wide.index, et_mm_wide[c].values, lw=1, label=c)
    ax.set_title(title)
    ax.set_ylabel("ET (mm / month)")
    ax.legend(title="Dataset", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(True, lw=0.2)
    plt.tight_layout()
    _save(fig, save)
    return fig


def plot_et_two_domains(wide_a: pd.DataFrame, wide_b: pd.DataFrame, label_a: str, label_b: str,
                        title: str, save=None):
    """Per-product panels comparing ET over two geometries (e.g. block vs delta polygon)."""
    models = sorted(set(wide_a.columns) | set(wide_b.columns))
    fig, axes = _grid(len(models), sharey=True)
    for ax, m in zip(axes, models):
        if m in wide_a:
            ax.plot(wide_a.index, wide_a[m], lw=0.8, color=C_Q, label=label_a)
        if m in wide_b:
            ax.plot(wide_b.index, wide_b[m], lw=0.8, color="#d62728", ls="--", label=label_b)
        ma = wide_a[m].mean() if m in wide_a else np.nan
        mb = wide_b[m].mean() if m in wide_b else np.nan
        ax.text(0.02, 0.95, f"{label_a}: {ma:.0f} | {label_b}: {mb:.0f} mm/mo", transform=ax.transAxes,
                fontsize=8, va="top", bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
        ax.set_title(m, fontsize=10)
        ax.grid(True, lw=0.2)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].set_ylabel("ET (mm / month)")
    fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    _save(fig, save)
    return fig


def plot_tws_pair(df: pd.DataFrame, col_a: str, col_b: str, diff_col: str,
                  label_a: str, label_b: str, title: str, save=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    ax1.plot(df.index, df[col_a], lw=0.9, label=label_a)
    ax1.plot(df.index, df[col_b], lw=0.9, label=label_b)
    ax1.set_ylabel("LWE thickness (cm)")
    ax1.set_title(title)
    ax1.legend(ncol=2)
    ax1.grid(True, lw=0.2)
    ax2.plot(df.index, df[diff_col], lw=1.2, color="k")
    ax2.axhline(0, lw=0.5, color="0.5", ls="--")
    ax2.set_ylabel(f"{label_a} − {label_b} (cm)")
    ax2.grid(True, lw=0.2)
    plt.tight_layout()
    _save(fig, save)
    print(f"{label_a} − {label_b}: mean {df[diff_col].mean():+.2f} cm, std {df[diff_col].std():.2f} cm")
    return fig


def plot_tws_series(series: dict[str, pd.Series], title: str, save=None):
    fig, ax = plt.subplots(figsize=(13, 4.5))
    for lab, s in series.items():
        ax.plot(s.index, s.values, lw=1, label=lab)
    ax.set_ylabel("TWS anomaly (cm)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, lw=0.2)
    plt.tight_layout()
    _save(fig, save)
    return fig


# ---------------------------------------------------------------------------
# Balance
# ---------------------------------------------------------------------------
def plot_balance_panels(bal: Balance, title_prefix: str, save=None):
    """P, ET, ΔS + residual (mm / month) on the continuous grid."""
    df = bal.df
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    ax = axes[0]
    ax.bar(df.index, df["P_mm"], width=25, color="steelblue", alpha=0.7, label="P (CHIRPS)")
    if bal.has_qin:
        ax.plot(df.index, df["Qin_mm"], lw=1, color=C_Q, label="Qin (Mohembo, mm over domain)")
    ax.set_ylabel("mm / month")
    ax.set_title(f"{title_prefix} — precipitation" + (" and inflow" if bal.has_qin else ""))
    ax.legend(fontsize=9)
    ax.grid(True, lw=0.2)
    ax = axes[1]
    ax.plot(df.index, df["ET_mm"], lw=1, color=C_ET, label="ET (median of products)")
    ax.fill_between(df.index, 0, df["ET_mm"].fillna(0), where=df["ET_mm"].notna(), alpha=0.15, color=C_ET)
    ax.set_ylabel("mm / month")
    ax.set_title(f"{title_prefix} — ET")
    ax.legend(fontsize=9)
    ax.grid(True, lw=0.2)
    ax = axes[2]
    # markers so months without a valid neighbour (isolated by GRACE gaps) still show
    ax.plot(df.index, df["dS_mm"], lw=1.2, color="k", marker="o", ms=2.5,
            label=f"GRACE ΔS ({bal.ds_scheme})")
    if "dS_uses_filled_tws" in df and df["dS_uses_filled_tws"].any():
        f = df["dS_uses_filled_tws"]
        ax.plot(df.index[f], df.loc[f, "dS_mm"], "o", ms=4.5, mfc="white", mec="k", mew=0.9,
                ls="none", label="ΔS using an interpolated GRACE month")
    ax.bar(df.index, df["resid_mm"], width=25, alpha=0.4, color=C_RES,
           label="Residual (" + ("Qin + " if bal.has_qin else "") + "P − ET − ΔS)")
    ax.axhline(0, lw=0.5, color="0.5", ls="--")
    ax.set_ylabel("mm / month")
    ax.set_title(f"{title_prefix} — GRACE ΔS and residual")
    ax.legend(fontsize=9)
    ax.grid(True, lw=0.2)
    plt.tight_layout()
    _save(fig, save)
    return fig


def plot_terms(bal: Balance, title: str, save=None, window: int = 3):
    """Monthly balance terms (km³), rolling-mean smoothed on the continuous grid."""
    df = bal.df
    terms = {}
    if bal.has_qin:
        terms["Qin (Mohembo)"] = (df["Qin_km3"], C_Q, "-")
    terms["P (CHIRPS)"] = (df["P_km3"], C_P, "--")
    terms["−ET (median)"] = (-df["ET_km3"], "#8856a7", "-.")
    terms[f"ΔS (GRACE, {bal.ds_scheme})"] = (df["dS_km3"], C_DS, ":")
    ds_label = f"ΔS (GRACE, {bal.ds_scheme})"
    fig, ax = plt.subplots(figsize=(13, 5))
    for lab, (s, c, ls) in terms.items():
        sm = s.rolling(window, center=True, min_periods=window).mean()
        mk = dict(marker="o", ms=2.5) if lab == ds_label else {}    # ΔS has GRACE gaps
        ax.plot(sm.index, sm.values, color=c, ls=ls, lw=1.3, label=lab, **mk)
    ax.axhline(0, lw=1, c="0.4")
    ax.set_title(f"{title} ({window}-month smoothed)")
    ax.set_ylabel("km³ / month")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, lw=0.2)
    plt.tight_layout()
    _save(fig, save)
    return fig


def plot_cumulative(bal: Balance, title: str, save=None):
    """Σ ΔS vs Σ(Qin + P − ET) over closure months, plus the cumulative residual."""
    d = bal.closure()
    flux = d[bal.flux_cols].sum(axis=1) - d["ET_km3"]
    lab = "∑(" + ("Qin + " if bal.has_qin else "") + "P − ET)"
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(d.index, d["dS_km3"].cumsum(), color=C_DS, ls=":", lw=2, label="∑ΔS (GRACE)")
    ax.plot(d.index, flux.cumsum(), color=C_Q, lw=2, label=lab)
    ax.plot(d.index, d["resid_km3"].cumsum(), color=C_RES, ls="--", lw=1.5, label="∑ residual")
    ax.axhline(0, lw=1, c="0.4")
    ax.set_title(title)
    ax.set_ylabel("km³ (cumulative)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, lw=0.2)
    plt.tight_layout()
    _save(fig, save)
    print(f"Cumulative residual over {len(d)} closure months: {d['resid_km3'].sum():+.2f} km³ "
          f"(mean {d['resid_km3'].mean():+.3f} km³/month)")
    return fig


def plot_per_model_cumulative(bal: Balance, title_prefix: str, fig_dir=None):
    """One cumulative panel per ET product (saved under fig_dir/single-model/<product>/)."""
    base = bal.df[bal.flux_cols + ["dS_km3"]].dropna()
    lab = "∑(" + ("Qin + " if bal.has_qin else "") + "P − ET)"
    models = [m for m in bal.et_km3_wide.columns if bal.et_km3_wide[m].reindex(base.index).notna().sum() >= 6]
    fig, axes = _grid(len(models), h=3.2)
    for ax, m in zip(axes, models):
        et = bal.et_km3_wide[m].reindex(base.index)
        d = base[et.notna()]
        flux = d[bal.flux_cols].sum(axis=1) - et[et.notna()]
        ax.plot(d.index, d["dS_km3"].cumsum(), color=C_DS, ls=":", lw=2, label="∑ΔS (GRACE)")
        ax.plot(d.index, flux.cumsum(), color=C_Q, lw=1.8, label=lab)
        ax.axhline(0, lw=0.8, c="0.4")
        ax.set_title(f"ET = {m}  ({len(d)} months)", fontsize=10)
        ax.set_ylabel("km³")
        ax.grid(True, lw=0.2)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle(f"{title_prefix} — cumulative balance per ET product", fontsize=13, y=1.01)
    plt.tight_layout()
    if fig_dir is not None:
        _save(fig, Path(fig_dir) / "cumulative_per_model.png")
    return fig


def plot_monthly_divergence(bal: Balance, title: str, save=None):
    """Box plots of the monthly residual by calendar month, per ET product."""
    df = bal.df
    flux_base = df[bal.flux_cols].sum(axis=1, min_count=len(bal.flux_cols))
    models = list(bal.et_km3_wide.columns)
    fig, axes = _grid(len(models), sharex=False, sharey=True)
    for ax, m in zip(axes, models):
        r = (flux_base - bal.et_km3_wide[m] - df["dS_km3"]).dropna()
        if len(r) < 12:
            ax.set_visible(False)
            continue
        data = [r[r.index.month == k].values for k in range(1, 13)]
        bp = ax.boxplot(data, positions=range(1, 13), widths=0.6, patch_artist=True, showfliers=False,
                        medianprops=dict(color="k", lw=1.2))
        for p in bp["boxes"]:
            p.set_facecolor("#a6bddb")
            p.set_alpha(0.7)
        mm = r.groupby(r.index.month).mean()
        ax.plot(mm.index, mm.values, "o-", color=C_ET, lw=1.5, ms=4, zorder=5, label="mean")
        ax.axhline(0, lw=0.8, ls="--", color="0.4")
        ax.set_title(f"{m}  (n={len(r)})", fontsize=10)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(MONTH_LABELS, fontsize=8)
        ax.grid(True, axis="y", lw=0.2)
        ax.set_ylabel("Residual (km³/mo)")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    _save(fig, save)
    return fig


def plot_climatology(clim: pd.DataFrame, labels: list[str], colors: list[str], title: str, save=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(1, 13)
    w = 0.8 / len(labels)
    for i, (lab, c) in enumerate(zip(labels, colors)):
        ax.bar(x + (i - (len(labels) - 1) / 2) * w, clim[f"{lab}_mean"], width=w, color=c,
               yerr=clim[f"{lab}_std"], capsize=2, label=lab)
    ax.axhline(0, lw=0.5, color="0.5", ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(MONTH_LABELS)
    ax.set_ylabel("mm / month")
    ax.set_title(title)
    ax.legend(fontsize=10)
    ax.grid(True, lw=0.2, axis="y")
    plt.tight_layout()
    _save(fig, save)
    return fig


# ---------------------------------------------------------------------------
# Lateral-flux fits
# ---------------------------------------------------------------------------
def plot_tuned_cumulative(results: pd.DataFrame, plot_data: dict, title: str, save=None, show_bias=True):
    if results.empty:
        print("Nothing to plot.")
        return None
    models = results["ET_model"].tolist()
    a_cols = [c for c in results.columns if c.startswith("alpha_")]
    fig, axes = _grid(len(models), h=4)
    for ax, m in zip(axes, models):
        d = plot_data[m]
        ax.plot(d["cum_ds"].index, d["cum_ds"].values, color=C_DS, ls=":", lw=2, label="∑ΔS (GRACE)")
        ax.plot(d["cum_base"].index, d["cum_base"].values, color=C_Q, lw=1.2, alpha=0.45, label="∑(flux − ET)")
        alab = ", ".join(f"{c[6:]}={v:.3f}" for c, v in zip(a_cols, d["alpha"]))
        ax.plot(d["cum_tuned"].index, d["cum_tuned"].values, color="#2ca02c", lw=2, label=f"∑(… + α·Δ)   {alab}")
        if show_bias:
            ax.plot(d["cum_tuned_bias"].index, d["cum_tuned_bias"].values, color="#2ca02c", lw=1, ls="--",
                    label=f"∑(… + α·Δ + c)   c={d['intercept']:+.3f} km³/mo")
        ax.axhline(0, lw=0.8, c="0.4")
        ax.set_title(f"{m}   monthly RMSE {d['rmse_null']:.3f} → {d['rmse_fit']:.3f} km³", fontsize=10)
        ax.set_ylabel("km³ (cumulative)")
        ax.grid(True, lw=0.2)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    _save(fig, save)
    return fig


def plot_alpha_sensitivity(bal: Balance, results: pd.DataFrame, x_col: str,
                           alphas=np.linspace(-1.5, 1.5, 301), title=None, save=None):
    if results.empty:
        return None
    df = bal.df
    flux_base = df[bal.flux_cols].sum(axis=1, min_count=len(bal.flux_cols))
    models = results["ET_model"].tolist()
    fig, axes = _grid(len(models))
    for ax, m in zip(axes, models):
        resid = (flux_base - bal.et_km3_wide[m] - df["dS_km3"]).values
        curve = alpha_sensitivity(resid, df[x_col].values, alphas)
        row = results.set_index("ET_model").loc[m]
        a_opt = row[f"alpha_{x_col}"]
        ax.plot(alphas, curve, lw=1.5, color=C_Q)
        ax.axvline(a_opt, ls="--", lw=1, color="#2ca02c", label=f"α* = {a_opt:.4f}")
        ax.axvline(0, ls=":", lw=0.8, color="0.5")
        ax.axhline(row["RMSE_null_km3"], ls=":", lw=0.8, color=C_DS, label=f"null RMSE = {row['RMSE_null_km3']:.3f}")
        ax.set_title(m, fontsize=10)
        ax.set_ylabel("monthly RMSE (km³)")
        ax.set_xlabel("α (km³ / month / cm)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, lw=0.2)
    fig.suptitle(title or f"Monthly RMSE vs α for {x_col}", fontsize=13, y=1.01)
    plt.tight_layout()
    _save(fig, save)
    return fig


def plot_rmse_heatmap(bal: Balance, results: pd.DataFrame, x1: str, x2: str,
                      agrid=None, bgrid=None, title=None, save=None):
    if results.empty:
        return None
    df = bal.df
    flux_base = df[bal.flux_cols].sum(axis=1, min_count=len(bal.flux_cols))
    a_all, b_all = results[f"alpha_{x1}"].values, results[f"alpha_{x2}"].values
    if agrid is None:
        agrid = np.linspace(min(-1.0, a_all.min() - 0.3), max(1.0, a_all.max() + 0.3), 121)
    if bgrid is None:
        bgrid = np.linspace(min(-1.0, b_all.min() - 0.3), max(0.5, b_all.max() + 0.3), 121)
    models = results["ET_model"].tolist()
    fig, axes = _grid(len(models), h=5, sharex=False)
    for ax, m in zip(axes, models):
        resid = (flux_base - bal.et_km3_wide[m] - df["dS_km3"]).values
        grid = rmse_surface(resid, df[x1].values, df[x2].values, agrid, bgrid)
        row = results.set_index("ET_model").loc[m]
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis_r",
                       extent=[agrid[0], agrid[-1], bgrid[0], bgrid[-1]])
        ax.plot(row[f"alpha_{x1}"], row[f"alpha_{x2}"], "r*", ms=14, zorder=5)
        ax.set_xlabel(f"α ({x1})")
        ax.set_ylabel(f"β ({x2})")
        ax.set_title(f"{m}  (RMSE* = {row['RMSE_fit_km3']:.3f} km³)", fontsize=10)
        plt.colorbar(im, ax=ax, label="monthly RMSE (km³)", shrink=0.85)
    fig.suptitle(title or f"RMSE surface: {x1} vs {x2}", fontsize=13, y=1.01)
    plt.tight_layout()
    _save(fig, save)
    return fig
