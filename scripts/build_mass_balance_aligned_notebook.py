"""Generate notebooks/mass_balance_grace_aligned.ipynb (3-block domain) on top of src/ modules.

Shared sections reuse the cell templates in build_notebooks.py; the unique analysis
sections of the pre-refactor notebook (archive/mass_balance_grace_aligned_pre_refactor.ipynb,
old cells 24-39 and 49-57) are kept, adapted to the new objects and bug-fixed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_mass_balance_notebooks import (md, code, setup_cell, DELTA_CELL, DISCOVER_CELL, EE_GEOM_CELL, ET_CELL,
                             ET_DIAG_MD, ET_DIAG_CELL, CHIRPS_CELL, GRACE_MD, GRACE_CELL, MOHEMBO_CELL,
                             BALANCE_MD, plots_cell, PER_MODEL_CELL, TUNING_MD, tuning_cells, save_cell,
                             write)

# ---------------------------------------------------------------------------
# Shared-section cells specific to the 3-block domain
# ---------------------------------------------------------------------------
TITLE_MD = md(r'''
# Mass Balance — GRACE-aligned (3 mascon blocks)

Okavango Delta water budget with **P**, **ET**, **Q** and **GRACE ΔS** all
computed over the same three JPL mascon blocks (NE + SE + SW) that overlap the
delta polygon.  Dropping the NW block (1.6 % of the delta) eliminates a large
dry-Kalahari footprint while retaining 98.4 % of the delta polygon coverage.
The NW block (and the block NW of it) are used only as **lateral-flux proxies**.

$$Q_{in} + P - ET - \Delta S = Q_{out} + G \;(\text{residual})$$

The pipeline lives in `src/` (`grace_blocks`, `et_products`, `precip`, `discharge`,
`balance`, `lateral_flux`, `balance_plots`); this notebook chooses the geometry,
calls it, and then keeps the analyses that are unique to this domain: what drives
the NW−NE and SW−NW TWS head gradients (§12), regime-adaptive ensemble ET (§13)
and a Darcy upper bound on lateral subsurface flux (§14).
''')

SELECT_CELL = code(r'''
from shapely.geometry import Point

# Drop the NW block (< 2 % of delta) from the domain
USE_BLOCKS = [b for b in delta_blocks if b["quadrant"] != "NW"]
NE_BLOCK = [b for b in USE_BLOCKS if b["quadrant"] == "NE"][0]
SE_BLOCK = [b for b in USE_BLOCKS if b["quadrant"] == "SE"][0]
SW_BLOCK = [b for b in USE_BLOCKS if b["quadrant"] == "SW"][0]
DOMAIN_LABEL = "3-block domain"

# Delta-adjacent NW block = the block immediately west of the NE block.
# (The pre-refactor notebook took the first "NW"-quadrant block from *all* blocks and
#  got B009 at lon 18–19°E, one block too far west, instead of the block at 19–22°E.)
NW_BLOCK = neighbour_block(NE_BLOCK, blocks, "W")

# "NW-of-NW" cell: the block containing the point one block-width west and one
# block-height north of the NW block's centre (None → that trace is skipped).
_pt = Point(NW_BLOCK["clon"] - (NW_BLOCK["lon1"] - NW_BLOCK["lon0"]),
            NW_BLOCK["clat"] + (NW_BLOCK["lat1"] - NW_BLOCK["lat0"]))
NWNW_BLOCK = next((b for b in blocks if b["geometry"].contains(_pt)), None)

for lab, b in [("NE", NE_BLOCK), ("SE", SE_BLOCK), ("SW", SW_BLOCK), ("NW (proxy)", NW_BLOCK),
               ("NW-of-NW (proxy)", NWNW_BLOCK)]:
    print(f"{lab:17s}: {describe(b) if b is not None else '— not found —'}")
print(f"Delta polygon covered by the domain: {sum(b['frac_ref'] for b in USE_BLOCKS):.1%}")
m = block_map(blocks, {"#1b9e77": USE_BLOCKS, "#d95f02": [NW_BLOCK], "#7570b3": [NWNW_BLOCK]}, ref_gdf=gdf,
              center=(centroid.y, centroid.x), zoom_start=7, ref_name="Delta polygon")
map_path = FIG_DIR / "study_area_map.html"; m.save(str(map_path))
IFrame(src=str(map_path), width=800, height=500)
''')

PROXY_MD = md(r'''
### Adjacent-block TWS differences (lateral-flux proxies)

Block TWS series from the local GRACE netCDF form two lateral-flux proxies:

| Proxy | Gradient | Physical interpretation |
|-------|----------|------------------------|
| $\Delta_1$ | TWS$_{\text{NW}}$ − TWS$_{\text{NE}}$ | West→east flow across the NE block boundary |
| $\Delta_2$ | TWS$_{\text{SW}}$ − TWS$_{\text{NW}}$ | South→north gradient driving flow into the domain from the SW |
''')

PROXY_CELL = code(r'''
df_nw_ne = tws_difference(GRACE_NC, NW_BLOCK, NE_BLOCK, "NW_NE_diff_cm")
df_sw_nw = tws_difference(GRACE_NC, SW_BLOCK, NW_BLOCK, "SW_NW_diff_cm")
bp.plot_tws_pair(df_nw_ne, df_nw_ne.columns[0], df_nw_ne.columns[1], "NW_NE_diff_cm", "NW", "NE",
                 "GRACE TWS: NW and NE blocks"); plt.show()
bp.plot_tws_pair(df_sw_nw, df_sw_nw.columns[0], df_sw_nw.columns[1], "SW_NW_diff_cm", "SW", "NW",
                 "GRACE TWS: SW and NW blocks"); plt.show()

# Convenience frames for the analysis sections below: a `date` column and short names
df_nw_ne = (df_nw_ne.rename(columns={df_nw_ne.columns[0]: "TWS_NW_cm", df_nw_ne.columns[1]: "TWS_NE_cm"})
            .reset_index())
df_sw_nw = (df_sw_nw.rename(columns={df_sw_nw.columns[0]: "TWS_SW_cm", df_sw_nw.columns[1]: "TWS_NW_cm"})
            .reset_index())
df_nwnw = (block_tws(GRACE_NC, NWNW_BLOCK).rename("TWS_NWNW_cm").reset_index()
           if NWNW_BLOCK is not None else None)
''')

BALANCE_CELL = code(r'''
bal = assemble_balance(df_et, df_chirps, df_tws, block_area_m2_,
                       qin=q_mohembo,
                       extra={"NW_NE_diff_cm": df_nw_ne.set_index("date")["NW_NE_diff_cm"],
                              "SW_NW_diff_cm": df_sw_nw.set_index("date")["SW_NW_diff_cm"]},
                       start=START, ds_scheme="centered")
df_balance = bal.df
xlim = (pd.Timestamp("2002-01-01"), bal.df.index.max())     # common x-range for time-series figures
print(bal.summary())
print("\nResidual = Qin + P − ET − ΔS  (= Qout + G)")
df_balance.tail(6)
''')

REGIME_MD = md(r'''
### Residual by hydrological regime

Water years (Oct–Sep) are classified as dry / normal / wet by their total input
$Q_{in} + P$ to the domain, and the monthly residual of each ET product is
compared across regimes.
''')

REGIME_CELL = code(r'''
# ── Residual diagnostics: dry / normal / wet years (based on inflow + local P) ──
# Water years are labelled by their *ending* year (Oct 2023–Sep 2024 → 2024);
# note src/time_utils.water_year uses the *starting* year, so lists here are in ending-year convention.
_bal = df_balance[["Qin_km3", "P_km3"]].dropna()
_bal_wy = np.where(_bal.index.month >= 10, _bal.index.year + 1, _bal.index.year)
wy_input = pd.DataFrame({"wy": _bal_wy, "input_km3": _bal["Qin_km3"] + _bal["P_km3"]})
wy_input = wy_input.groupby("wy").agg(total_input=("input_km3", "sum"),
                                        n_months=("input_km3", "count")).reset_index()
wy_input = wy_input[wy_input["n_months"] == 12].copy()

dry_thresh_inp = wy_input["total_input"].quantile(0.25)
wet_thresh_inp = wy_input["total_input"].quantile(0.75)
wy_input["regime"] = np.where(wy_input["total_input"] < dry_thresh_inp, "Dry",
                     np.where(wy_input["total_input"] > wet_thresh_inp, "Wet", "Normal"))

dry_wy_set    = set(wy_input.loc[wy_input["regime"] == "Dry",    "wy"])
normal_wy_set = set(wy_input.loc[wy_input["regime"] == "Normal", "wy"])
wet_wy_set    = set(wy_input.loc[wy_input["regime"] == "Wet",    "wy"])

print(f"Classification: annual Qin + P into the domain (km³), ending-year water years")
print(f"Dry   (<25th pctl, <{dry_thresh_inp:.1f} km³): {sorted(dry_wy_set)}")
print(f"Normal (25–75th pctl):                  {sorted(normal_wy_set)}")
print(f"Wet   (>75th pctl, >{wet_thresh_inp:.1f} km³): {sorted(wet_wy_set)}")

PERIOD_COLORS = {"Dry": "#d73027", "Normal": "#fee08b", "Wet": "#4575b4"}

flux_base = df_balance[bal.flux_cols].sum(axis=1, min_count=len(bal.flux_cols))
rows_diag = []
for et_name in sorted(bal.et_km3_wide.columns):
    resid = (flux_base - bal.et_km3_wide[et_name] - df_balance["dS_km3"]).dropna()
    if len(resid) < 24:
        continue
    wy_arr = np.where(resid.index.month >= 10, resid.index.year + 1, resid.index.year)   # ending-year WY
    for label, wy_set in [("Dry", dry_wy_set), ("Normal", normal_wy_set), ("Wet", wet_wy_set)]:
        mask = np.isin(wy_arr, list(wy_set))
        if mask.sum() == 0:
            continue
        rows_diag.append({"ET_model": et_name, "period": label,
                          "resid_mean": resid[mask].mean(), "resid_std": resid[mask].std(),
                          "n": int(mask.sum())})
df_diag = pd.DataFrame(rows_diag)
all_et = sorted(df_diag["ET_model"].unique())

# ── Plot: grouped bar chart (mean ± std) ──
periods = ["Dry", "Normal", "Wet"]
n_et = len(all_et)
bar_width = 0.25
x = np.arange(n_et)
fig, ax = plt.subplots(figsize=(max(10, n_et * 1.8), 5))
for k, period in enumerate(periods):
    sub = df_diag[df_diag["period"] == period].set_index("ET_model").reindex(all_et)
    ax.bar(x + (k - 1) * bar_width, sub["resid_mean"].values, bar_width, yerr=sub["resid_std"].values,
           capsize=3, color=PERIOD_COLORS[period], alpha=0.8, edgecolor="0.3", lw=0.5, label=period)
ax.axhline(0, lw=0.8, c="0.4", ls=":")
ax.set_xticks(x)
ax.set_xticklabels(all_et, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Mean residual ± 1 std (km³ month⁻¹)")
ax.set_title("Budget residual (Qin + P − ET − ΔS) by hydrological regime\n"
             "(classified by annual Qin + local P; n per model in the table below)", fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, lw=0.2, axis="y")
plt.tight_layout()
plt.show()

# ── Summary table (n differs between ET products because their records differ) ──
print("\n── Mean residual ± std (km³/month), n months ──")
tbl = df_diag.assign(txt=lambda d: d.apply(lambda r: f"{r.resid_mean:+.3f} ± {r.resid_std:.3f} (n={r.n})", axis=1))
print(tbl.pivot(index="ET_model", columns="period", values="txt")[periods].to_string())
''')

# ---------------------------------------------------------------------------
# §12 — drivers of the head gradients (old cells 24 (2nd half), 25, 29–37)
# ---------------------------------------------------------------------------
DRIVERS_MD = md(r'''
## 12 — What drives the TWS head gradients?

The NW−NE and SW−NW head differences are the regressors of the lateral-flux
model above.  This section asks what controls them: regional rainfall
(Angolan highlands vs the delta), Mohembo inflow, and flood extent.
''')

RAIN_PLOTS_CELL = code(r'''
# ── Regional CHIRPS rainfall (Angolan highlands vs Delta) ──
CHIRPS_REGIONS = Path("../data/chirps_monthly_by_regions.csv")
df_chirps_reg = pd.read_csv(CHIRPS_REGIONS)
df_chirps_reg["date"] = pd.to_datetime(df_chirps_reg["ym"] + "-01")

# Angolan highlands = mean of highland_cuito + highland_cubango
angola_hl = (df_chirps_reg[df_chirps_reg["region_id"].isin(
                 ["highland_cuito", "highland_cubango"])]
             .groupby("date")["precip_mm"].mean().reset_index()
             .rename(columns={"precip_mm": "P_angola_mm"}))

# Delta = mean of east_lower_okavango + west_lower_okavango
delta_rain = (df_chirps_reg[df_chirps_reg["region_id"].isin(
                  ["east_lower_okavango", "west_lower_okavango"])]
              .groupby("date")["precip_mm"].mean().reset_index()
              .rename(columns={"precip_mm": "P_delta_mm"}))

df_rain = angola_hl.merge(delta_rain, on="date")
# Make sure the rainfall table is a continuous monthly series (rolling windows below rely on it)
df_rain = (df_rain.set_index("date")
           .reindex(pd.date_range(df_rain["date"].min(), df_rain["date"].max(), freq="MS"))
           .rename_axis("date").reset_index())
df_rain["P_gradient_mm"] = df_rain["P_angola_mm"] - df_rain["P_delta_mm"]

# Monthly anomalies (subtract long-term monthly climatology)
for col in ["P_angola_mm", "P_delta_mm"]:
    clim = df_rain.groupby(df_rain["date"].dt.month)[col].transform("mean")
    df_rain[col.replace("_mm", "_anom")] = df_rain[col] - clim

# ── Plot NW−NE (3 panels) with NW-of-NW cell ──
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

ax1.plot(df_nw_ne["date"], df_nw_ne["TWS_NW_cm"], lw=0.9, label="NW block")
ax1.plot(df_nw_ne["date"], df_nw_ne["TWS_NE_cm"], lw=0.9, label="NE block")
if df_nwnw is not None:
    ax1.plot(df_nwnw["date"], df_nwnw["TWS_NWNW_cm"], lw=1.1, ls="--",
             color="green", label="NW-of-NW cell")
ax1.set_ylabel("LWE thickness (cm)")
ax1.set_title("GRACE TWS anomaly: NW block, NE block & NW-of-NW cell")
ax1.legend(ncol=3, fontsize=8)
ax1.grid(True, lw=0.2)

ax2.plot(df_nw_ne["date"], df_nw_ne["NW_NE_diff_cm"], lw=1.2, color="k")
ax2.axhline(0, lw=0.5, color="0.5", ls="--")
ax2.set_ylabel("NW − NE (cm)")
ax2.set_title("TWS difference (NW − NE)")
ax2.grid(True, lw=0.2)

ax3.plot(df_rain["date"], df_rain["P_angola_anom"], lw=0.9,
         label="Angolan highlands", color="steelblue")
ax3.plot(df_rain["date"], df_rain["P_delta_anom"], lw=0.9,
         label="Delta", color="darkorange")
ax3.axhline(0, lw=0.5, color="0.5", ls="--")
ax3.set_ylabel("Rainfall anomaly (mm)")
ax3.set_title("Monthly rainfall anomaly: Angolan highlands vs Delta")
ax3.legend(ncol=2)
ax3.grid(True, lw=0.2)

ax1.set_xlim(xlim)
plt.tight_layout()
plt.show()

# ── Plot SW−NW (3 panels) ──
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

ax1.plot(df_sw_nw["date"], df_sw_nw["TWS_SW_cm"], lw=0.9, label="SW block")
ax1.plot(df_sw_nw["date"], df_sw_nw["TWS_NW_cm"], lw=0.9, label="NW block")
ax1.set_ylabel("LWE thickness (cm)")
ax1.set_title("GRACE TWS anomaly: SW block vs NW block")
ax1.legend(ncol=2)
ax1.grid(True, lw=0.2)

ax2.plot(df_sw_nw["date"], df_sw_nw["SW_NW_diff_cm"], lw=1.2, color="k")
ax2.axhline(0, lw=0.5, color="0.5", ls="--")
ax2.set_ylabel("SW − NW (cm)")

ax3.plot(df_rain["date"], df_rain["P_angola_anom"], lw=0.9,
         label="Angolan highlands", color="steelblue")
ax3.plot(df_rain["date"], df_rain["P_delta_anom"], lw=0.9,
         label="Delta", color="darkorange")
ax3.axhline(0, lw=0.5, color="0.5", ls="--")
ax3.set_ylabel("Rainfall anomaly (mm)")
ax3.set_title("Monthly rainfall anomaly: Angolan highlands vs Delta")
ax3.legend(ncol=2)
ax3.grid(True, lw=0.2)

ax1.set_xlim(xlim)
plt.tight_layout()
plt.show()
''')

CLIM_CELL = code(r'''
# ── Monthly mean DSWE flood extent (Landsat + Sentinel-2) ──
df_landsat = pd.read_csv(Path("../data/monthly_landsat_dswe.csv"))
df_landsat["date"] = pd.to_datetime(df_landsat["date"])

df_sentinel = pd.read_csv(Path("../data/monthly_sentinel2_dswe.csv"))
df_sentinel["date"] = pd.to_datetime(df_sentinel["date"])

# Monthly climatology
landsat_clim = df_landsat.groupby("month")["km2"].agg(["mean", "std"]).rename(
    columns={"mean": "L_mean", "std": "L_std"})
sentinel_clim = df_sentinel.groupby("month")["km2"].agg(["mean", "std"]).rename(
    columns={"mean": "S_mean", "std": "S_std"})

month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()

ln1 = ax1.errorbar(landsat_clim.index, landsat_clim["L_mean"],
                   yerr=landsat_clim["L_std"], fmt="o-", capsize=3,
                   label="Landsat DSWE", color="steelblue")
ln2 = ax2.errorbar(sentinel_clim.index, sentinel_clim["S_mean"],
                   yerr=sentinel_clim["S_std"], fmt="s-", capsize=3,
                   label="Sentinel-2 DSWE", color="darkorange")

ax1.set_xticks(range(1, 13))
ax1.set_xticklabels(month_labels)
ax1.set_xlabel("Month")
ax1.set_ylabel("Landsat flood extent (km²)", color="steelblue")
ax2.set_ylabel("Sentinel-2 flood extent (km²)", color="darkorange")
ax1.tick_params(axis="y", labelcolor="steelblue")
ax2.tick_params(axis="y", labelcolor="darkorange")
ax1.set_title("Monthly mean DSWE flood extent")
lns = [ln1, ln2]
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs)
ax1.grid(True, lw=0.2)
plt.tight_layout()
plt.show()

# ── Monthly-mean head gradient climatology ──
df_nw_ne["month"] = df_nw_ne["date"].dt.month
df_sw_nw["month"] = df_sw_nw["date"].dt.month

clim_nw_ne = df_nw_ne.groupby("month")["NW_NE_diff_cm"].agg(["mean", "std"]).rename(
    columns={"mean": "NW_NE_mean", "std": "NW_NE_std"})
clim_sw_nw = df_sw_nw.groupby("month")["SW_NW_diff_cm"].agg(["mean", "std"]).rename(
    columns={"mean": "SW_NW_mean", "std": "SW_NW_std"})

months = clim_nw_ne.index

fig, ax = plt.subplots(figsize=(10, 5))
ax.errorbar(months, clim_nw_ne["NW_NE_mean"], yerr=clim_nw_ne["NW_NE_std"],
            fmt="o-", capsize=3, label="NW − NE", color="steelblue")
ax.errorbar(clim_sw_nw.index, clim_sw_nw["SW_NW_mean"], yerr=clim_sw_nw["SW_NW_std"],
            fmt="s-", capsize=3, label="SW − NW", color="darkorange")
ax.axhline(0, lw=0.5, color="0.5", ls="--")
ax.set_xticks(months)
ax.set_xticklabels(month_labels)
ax.set_xlabel("Month")
ax.set_ylabel("Head gradient (cm)")
ax.set_title("Monthly mean GRACE TWS head gradients")
ax.legend()
ax.grid(True, lw=0.2)
plt.tight_layout()
plt.show()

# ── Monthly-mean CHIRPS rainfall climatology: Highlands vs Okavango ──
rain_clim = df_rain.groupby(df_rain["date"].dt.month).agg(
    P_angola_mean=("P_angola_mm", "mean"),
    P_angola_std=("P_angola_mm", "std"),
    P_delta_mean=("P_delta_mm", "mean"),
    P_delta_std=("P_delta_mm", "std"),
    P_gradient_mean=("P_gradient_mm", "mean"),
    P_gradient_std=("P_gradient_mm", "std"),
)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

# Top: Highlands and Delta monthly mean rainfall
ax1.errorbar(rain_clim.index - 0.15, rain_clim["P_angola_mean"],
             yerr=rain_clim["P_angola_std"], fmt="o-", capsize=3,
             label="Angolan highlands", color="steelblue")
ax1.errorbar(rain_clim.index + 0.15, rain_clim["P_delta_mean"],
             yerr=rain_clim["P_delta_std"], fmt="s-", capsize=3,
             label="Okavango Delta", color="darkorange")
ax1.set_ylabel("Rainfall (mm/month)")
ax1.set_title("Monthly mean CHIRPS rainfall")
ax1.legend()
ax1.grid(True, lw=0.2)

# Bottom: difference (Highlands − Delta)
ax2.errorbar(rain_clim.index, rain_clim["P_gradient_mean"],
             yerr=rain_clim["P_gradient_std"], fmt="D-", capsize=3,
             color="k", label="Highlands − Delta")
ax2.axhline(0, lw=0.5, color="0.5", ls="--")
ax2.set_xticks(rain_clim.index)
ax2.set_xticklabels(month_labels)
ax2.set_xlabel("Month")
ax2.set_ylabel("Rainfall difference (mm/month)")
ax2.set_title("Monthly mean rainfall gradient (Highlands − Delta)")
ax2.legend()
ax2.grid(True, lw=0.2)

plt.tight_layout()
plt.show()
''')

DROUGHT_MD = md(r'''
### Drought-year rainfall gradient hypothesis

Test whether the **Angola → Delta rainfall gradient** is stronger during
drought years in the Angolan highlands.  Drought years are defined as
water-years (Oct–Sep) where total Angolan-highlands rainfall falls below
the 25th percentile.
''')

DROUGHT_CELL = code(r'''
from scipy import stats

# ── Water-year aggregation (Oct–Sep) ──
# Water years are labelled by their *ending* year (Oct 2023–Sep 2024 → 2024);
# note src/time_utils.water_year uses the *starting* year, so the drought-year list below is in ending-year convention.
df_rain["wy"] = np.where(df_rain["date"].dt.month >= 10,
                          df_rain["date"].dt.year + 1,
                          df_rain["date"].dt.year)

wy_rain = df_rain.groupby("wy").agg(
    P_angola_total=("P_angola_mm", "sum"),
    P_delta_total=("P_delta_mm", "sum"),
    P_gradient_total=("P_gradient_mm", "sum"),
    n_months=("P_angola_mm", "count"),
).reset_index()

# Keep only complete water-years (12 months)
wy_rain = wy_rain[wy_rain["n_months"] == 12].copy()

# Drought threshold: 25th percentile of Angolan highlands annual rainfall
drought_thresh = wy_rain["P_angola_total"].quantile(0.25)
wy_rain["drought"] = wy_rain["P_angola_total"] < drought_thresh

drought_yrs = wy_rain[wy_rain["drought"]]
normal_yrs  = wy_rain[~wy_rain["drought"]]

print(f"Drought threshold (25th pctl): {drought_thresh:.0f} mm/yr")
print(f"Drought years ({len(drought_yrs)}, ending-year WY): {sorted(drought_yrs['wy'].tolist())}")
print(f"Normal years  ({len(normal_yrs)}): {sorted(normal_yrs['wy'].tolist())}")

# ── Compare gradient during drought vs normal years ──
grad_drought = drought_yrs["P_gradient_total"]
grad_normal  = normal_yrs["P_gradient_total"]

print(f"\n--- Annual rainfall gradient (Angola − Delta) ---")
print(f"Drought years:  mean = {grad_drought.mean():.0f} mm,  "
      f"std = {grad_drought.std():.0f} mm")
print(f"Normal  years:  mean = {grad_normal.mean():.0f} mm,  "
      f"std = {grad_normal.std():.0f} mm")

# Two-sample t-test (unequal variance)
t_stat, p_val = stats.ttest_ind(grad_drought, grad_normal, equal_var=False)
print(f"\nWelch's t-test:  t = {t_stat:.2f},  p = {p_val:.4f}")
if p_val < 0.05:
    print("→ Significant difference in gradient at α = 0.05")
else:
    print("→ No significant difference at α = 0.05")

# ── Also compare Angola and Delta rainfall separately ──
print(f"\n--- Angolan highlands annual rainfall ---")
print(f"Drought: {drought_yrs['P_angola_total'].mean():.0f} mm  |  "
      f"Normal: {normal_yrs['P_angola_total'].mean():.0f} mm")
print(f"\n--- Delta annual rainfall ---")
print(f"Drought: {drought_yrs['P_delta_total'].mean():.0f} mm  |  "
      f"Normal: {normal_yrs['P_delta_total'].mean():.0f} mm")

# Fractional reduction: which region loses more rainfall during droughts?
angola_frac = (1 - drought_yrs["P_angola_total"].mean() /
               normal_yrs["P_angola_total"].mean()) * 100
delta_frac  = (1 - drought_yrs["P_delta_total"].mean() /
               normal_yrs["P_delta_total"].mean()) * 100
print(f"\nFractional reduction during drought:")
print(f"  Angola:  {angola_frac:.1f}%")
print(f"  Delta:   {delta_frac:.1f}%")
if angola_frac > delta_frac:
    print("→ Angola rainfall drops proportionally MORE during droughts "
          "→ gradient weakens")
else:
    print("→ Delta rainfall drops proportionally MORE during droughts "
          "→ gradient strengthens")

# ── Visualise ──
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# (a) Annual rainfall time series
ax = axes[0]
ax.bar(wy_rain["wy"] - 0.15, wy_rain["P_angola_total"], width=0.3,
       label="Angola highlands", color="steelblue")
ax.bar(wy_rain["wy"] + 0.15, wy_rain["P_delta_total"], width=0.3,
       label="Delta", color="darkorange")
for _, row in drought_yrs.iterrows():
    ax.axvspan(row["wy"] - 0.45, row["wy"] + 0.45,
               alpha=0.15, color="red", zorder=0)
ax.set_ylabel("Annual rainfall (mm)")
ax.set_title("Water-year rainfall\n(red = drought years)")
ax.legend(fontsize=8)
ax.grid(True, lw=0.2)

# (b) Annual gradient
ax = axes[1]
colors = ["firebrick" if d else "0.4" for d in wy_rain["drought"]]
ax.bar(wy_rain["wy"], wy_rain["P_gradient_total"], color=colors, width=0.7)
ax.axhline(grad_drought.mean(), color="firebrick", ls="--", lw=1,
           label=f"Drought mean ({grad_drought.mean():.0f})")
ax.axhline(grad_normal.mean(), color="0.4", ls="--", lw=1,
           label=f"Normal mean ({grad_normal.mean():.0f})")
ax.set_ylabel("Angola − Delta rainfall (mm)")
ax.set_title("Annual rainfall gradient")
ax.legend(fontsize=8)
ax.grid(True, lw=0.2)

# (c) Box plot comparing drought vs normal gradient
ax = axes[2]
bxp = ax.boxplot([grad_drought.values, grad_normal.values],
                 labels=["Drought", "Normal"],
                 patch_artist=True,
                 widths=0.5)
bxp["boxes"][0].set_facecolor("firebrick")
bxp["boxes"][0].set_alpha(0.5)
bxp["boxes"][1].set_facecolor("0.7")
ax.set_ylabel("Angola − Delta rainfall (mm)")
ax.set_title(f"Gradient comparison\n(Welch t-test p = {p_val:.3f})")
ax.grid(True, lw=0.2, axis="y")

plt.tight_layout()
plt.show()
''')

AHDI_MD = md(r'''
### Angolan Highland Drought Index vs TWS gradients

Construct a standardised precipitation anomaly index from the
Angolan-highlands CHIRPS rainfall:

$$\text{AHDI}_m = \frac{P_m - \bar{P}_{\text{clim}}(m)}{\sigma_{\text{clim}}(m)}$$

Then correlate the monthly index with the TWS gradient series (NW−NE and
SW−NW) at lags 0–6 months.

Do Angolan rainfall anomalies predict shifts in the regional water-storage gradient?
''')

AHDI_CELL = code(r'''
# ── Angolan Highland Drought Index (AHDI) — multi-scale ──
# Compute SPI-like indices at 3, 6, and 12-month accumulation windows.
# For each window k: rolling sum of P, then standardise by calendar-month
# climatology of that rolling sum.  df_rain is a continuous monthly series.
SPI_WINDOWS = [3, 6, 12]

for k in SPI_WINDOWS:
    roll_col = f"P_angola_{k}mo"
    idx_col  = f"AHDI_{k}"
    df_rain[roll_col] = df_rain["P_angola_mm"].rolling(k, min_periods=k).sum()
    grp = df_rain.groupby(df_rain["date"].dt.month)[roll_col]
    df_rain[idx_col] = (df_rain[roll_col] - grp.transform("mean")) / grp.transform("std")

# Default AHDI for backward compat = 6-month window
df_rain["AHDI"] = df_rain["AHDI_6"]

# Merge all AHDI variants with TWS gradient series on month
ahdi_cols = ["date"] + [f"AHDI_{k}" for k in SPI_WINDOWS] + ["AHDI"]
rain_m = df_rain[ahdi_cols].copy()
rain_m["month"] = rain_m["date"].dt.to_period("M")

nwne_m = df_nw_ne[["date", "NW_NE_diff_cm"]].copy()
nwne_m["month"] = nwne_m["date"].dt.to_period("M")

df_idx = (rain_m[["month"] + [f"AHDI_{k}" for k in SPI_WINDOWS] + ["AHDI"]]
          .merge(nwne_m[["month", "NW_NE_diff_cm"]], on="month", how="inner")
          .sort_values("month").reset_index(drop=True))
df_idx["date"] = df_idx["month"].dt.to_timestamp()

# Drop rows where longer-window indices are NaN (need k months of history)
df_idx_clean = df_idx.dropna(subset=[f"AHDI_{max(SPI_WINDOWS)}"]).copy()

print(f"Merged records: {len(df_idx_clean)}  "
      f"({df_idx_clean['date'].min().date()} → {df_idx_clean['date'].max().date()})")

# ── Contemporaneous Pearson correlations (all windows) ──
print("\n--- Contemporaneous r(AHDI_k, NW−NE) ---")
print(f"{'Window':>8s}  |  {'r (p)':>18s}")
print("-" * 35)
for k in SPI_WINDOWS:
    acol = f"AHDI_{k}"
    sub = df_idx_clean.dropna(subset=[acol])
    r1, p1 = stats.pearsonr(sub[acol], sub["NW_NE_diff_cm"])
    s1 = "*" if p1 < 0.05 else " "
    print(f"  {k:2d}-mo    | {r1:+.3f} ({p1:.4f}){s1}")

# ── Lagged cross-correlation (AHDI_6 leads NW−NE by 0–6 months) ──
# Shift on a FULL monthly grid, so "lag k" is always k calendar months (positional
# slicing of the GRACE-month-only table let lags jump across GRACE gaps).
max_lag = 6
full_idx = pd.date_range(df_idx_clean["date"].min(), df_idx_clean["date"].max(), freq="MS")
ahdi_full = df_rain.set_index("date")["AHDI"].reindex(full_idx)
nwne_full = df_nw_ne.set_index("date")["NW_NE_diff_cm"].reindex(full_idx)   # NaN in GRACE gaps
lag_rs, lag_ps, lag_ns = [], [], []
for lag in range(max_lag + 1):
    pair = pd.concat([ahdi_full.shift(lag), nwne_full], axis=1).dropna()
    r, p = stats.pearsonr(pair.iloc[:, 0].values, pair.iloc[:, 1].values)
    lag_rs.append(r)
    lag_ps.append(p)
    lag_ns.append(len(pair))

# ── Figure: dual-axis time series ──
r6, p6 = stats.pearsonr(df_idx_clean["AHDI_6"], df_idx_clean["NW_NE_diff_cm"])

fig, ax1 = plt.subplots(figsize=(14, 5))

# Left axis: AHDI-6
ax1.fill_between(df_idx_clean["date"], df_idx_clean["AHDI_6"], 0,
                 where=df_idx_clean["AHDI_6"] >= 0, alpha=0.25, color="steelblue")
ax1.fill_between(df_idx_clean["date"], df_idx_clean["AHDI_6"], 0,
                 where=df_idx_clean["AHDI_6"] < 0, alpha=0.25, color="firebrick")
ln1 = ax1.plot(df_idx_clean["date"], df_idx_clean["AHDI_6"],
               lw=1.2, color="steelblue", label="AHDI-6 (left)")
ax1.axhline(0, lw=0.5, color="0.5", ls="--")
ax1.set_ylabel("AHDI-6 (σ)", color="steelblue")
ax1.tick_params(axis="y", labelcolor="steelblue")
ax1.set_xlim(xlim)
ax1.grid(True, lw=0.2, alpha=0.4)

# Right axis: NW − NE TWS difference
ax2 = ax1.twinx()
ln2 = ax2.plot(df_idx_clean["date"], df_idx_clean["NW_NE_diff_cm"],
               lw=1.2, color="k", label="NW−NE TWS (right)")
ax2.set_ylabel("NW − NE TWS difference (cm)")

# Combined legend
lns = ln1 + ln2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc="lower left", fontsize=9)

ax1.set_title(f"AHDI-6 vs NW−NE GRACE TWS difference  "
              f"(r = {r6:+.2f}, p = {p6:.3f})")
fig.tight_layout()
plt.show()

# ── Scatter + lagged correlations ──
fig, (ax_s, ax_l) = plt.subplots(1, 2, figsize=(12, 5))
ax = ax_s
ax.scatter(df_idx_clean["AHDI"], df_idx_clean["NW_NE_diff_cm"], s=12, alpha=0.5,
           color="tab:blue")
m, b = np.polyfit(df_idx_clean["AHDI"], df_idx_clean["NW_NE_diff_cm"], 1)
xs = np.array([df_idx_clean["AHDI"].min(), df_idx_clean["AHDI"].max()])
ax.plot(xs, m * xs + b, color="tab:blue", lw=1.5, ls="--")
r0, p0 = stats.pearsonr(df_idx_clean["AHDI"], df_idx_clean["NW_NE_diff_cm"])
ax.set_xlabel("AHDI-6 (σ)")
ax.set_ylabel("NW − NE TWS (cm)")
ax.set_title(f"AHDI-6 vs NW−NE (r={r0:+.2f}, p={p0:.3f})")
ax.grid(True, lw=0.2)

# Lagged correlation (AHDI_6 → NW−NE)
ax = ax_l
lags = list(range(max_lag + 1))
ax.plot(lags, lag_rs, lw=1.5, color="tab:blue", marker="o", markersize=5)
for i, (r, p) in enumerate(zip(lag_rs, lag_ps)):
    if p < 0.05:
        ax.plot(i, r, color="tab:blue", marker="o", markersize=9,
                markeredgecolor="k", markeredgewidth=1.2)
ax.axhline(0, lw=0.5, color="0.5", ls="--")
ax.set_xlabel("Lag (months, AHDI-6 leads)")
ax.set_ylabel("Pearson r")
ax.set_title("Lagged correlation: AHDI-6 → NW−NE\n(large markers = p < 0.05)")
ax.set_xticks(lags)
ax.grid(True, lw=0.2)

plt.tight_layout()
plt.show()

# ── Print lag table ──
print("\nLag (mo)  |  NW−NE r (p)         n")
print("-" * 38)
for i in range(max_lag + 1):
    sig = "*" if lag_ps[i] < 0.05 else " "
    print(f"  {i:2d}      | {lag_rs[i]:+.3f} ({lag_ps[i]:.4f}){sig}  {lag_ns[i]:4d}")
''')

PREDICT_MD1 = md(r'''
### Predicting NW − NE TWS from rainfall & discharge

Multiple linear regression using **Angolan-highlands rainfall**,
**Delta rainfall**, and **Mohembo discharge** (with lags) to predict
the monthly NW − NE GRACE TWS difference.
''')

PREDICT_MD2 = md(r'''
#### Predictor variables

**Flux variables** (monthly water inputs):

| Variable | Description |
|---|---|
| `P_angola_mm` | CHIRPS rainfall over the Angolan highlands headwater catchment (mm) |
| `P_delta_mm` | CHIRPS rainfall directly over the Okavango Delta (mm) |
| `Qin_m3s` | Mohembo gauging-station discharge — inflow to the delta (m³ s⁻¹; gap-filled ≤ 2 months) |
| `P_gradient_mm` | `P_angola_mm − P_delta_mm` — headwater-to-delta rainfall gradient |
| `*_6mo` / `*_12mo` | 6- and 12-month rolling means of the above, capturing catchment memory |
| `*_lag1…lag3` | 1–3 month lags of rainfall and discharge, encoding travel-time delays |
| `NW_NE_lag1` | Previous month's NW − NE TWS difference (autoregressive term) |

**Season variables** (annual harmonic):

| Variable | Description |
|---|---|
| `sin_mo` | sin(2π · month / 12) — captures the phase of the annual flood cycle |
| `cos_mo` | cos(2π · month / 12) — together with sin_mo, fits amplitude and phase of any 12-month periodicity |

All rainfall / discharge features are built on the **continuous monthly series**
*before* merging with the GRACE target, so lags and rolling windows never span a
GRACE gap; the AR(1) term is `shift(1)` of the target on the same continuous grid
(NaN across gaps, so those months are dropped rather than bridged).

The target is `NW_NE_diff_cm`, the GRACE-derived TWS difference between the NW and NE mascon blocks (cm), which acts as a proxy for lateral water redistribution across the delta.
''')

PREDICT_NWNE_CELL = code(r'''
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error

# ── Continuous monthly predictor table (rainfall + Mohembo discharge) ──
# Features are built here, BEFORE merging with the GRACE target, so lags and
# rolling means are always in calendar months and never span GRACE gaps.
_rain_m = df_rain.set_index("date")[["P_angola_mm", "P_delta_mm"]]
_q_m = q_mohembo.set_index("date")["Qin_m3s"]
full_idx = pd.date_range(max(_rain_m.index.min(), _q_m.index.min()),
                         min(_rain_m.index.max(), _q_m.index.max()), freq="MS")
df_pred = pd.concat([_rain_m, _q_m], axis=1).reindex(full_idx)
df_pred.index.name = "date"

# Rainfall gradient
df_pred["P_gradient_mm"] = df_pred["P_angola_mm"] - df_pred["P_delta_mm"]

# Rolling means (6- and 12-month memory)
for col in ["P_angola_mm", "P_delta_mm", "P_gradient_mm", "Qin_m3s"]:
    df_pred[f"{col}_6mo"]  = df_pred[col].rolling(6,  min_periods=6).mean()
    df_pred[f"{col}_12mo"] = df_pred[col].rolling(12, min_periods=12).mean()

# Lagged predictors (1–3 months)
for lag in [1, 2, 3]:
    for col in ["P_angola_mm", "P_delta_mm", "Qin_m3s"]:
        df_pred[f"{col}_lag{lag}"] = df_pred[col].shift(lag)

# Seasonal harmonics
mo = df_pred.index.month
df_pred["sin_mo"] = np.sin(2 * np.pi * mo / 12)
df_pred["cos_mo"] = np.cos(2 * np.pi * mo / 12)

# ── Target on the same continuous grid; AR(1) = shift(1) → NaN across GRACE gaps ──
_tgt = df_nw_ne.set_index("date")["NW_NE_diff_cm"].reindex(full_idx)
df_model = df_pred.copy()
df_model["NW_NE_diff_cm"] = _tgt
df_model["NW_NE_lag1"] = _tgt.shift(1)
df_model = df_model.dropna().reset_index()
print(f"Model dataset: {len(df_model)} months  "
      f"({df_model['date'].min().date()} → {df_model['date'].max().date()})")

# ── Feature sets to compare ──
base_flux = ["P_angola_mm", "P_delta_mm", "Qin_m3s"]
feature_sets = {
    "Flux only": base_flux,
    "Flux + gradient + season": base_flux + ["P_gradient_mm", "sin_mo", "cos_mo"],
    "6-mo means + season": [f"{c}_6mo" for c in
        ["P_angola_mm", "P_delta_mm", "P_gradient_mm", "Qin_m3s"]] + ["sin_mo", "cos_mo"],
    "AR(1) only": ["NW_NE_lag1"],
    "AR(1) + flux + season": ["NW_NE_lag1", "sin_mo", "cos_mo"] + base_flux,
}

y = df_model["NW_NE_diff_cm"].values

# ── LOO cross-validated regression ──
results = {}
for name, feats in feature_sets.items():
    X = df_model[feats].values
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    loo = LeaveOneOut()
    y_pred = np.full_like(y, np.nan)
    for train_idx, test_idx in loo.split(X_sc):
        reg = LinearRegression()
        reg.fit(X_sc[train_idx], y[train_idx])
        y_pred[test_idx] = reg.predict(X_sc[test_idx])

    r2_loo = r2_score(y, y_pred)
    rmse_loo = np.sqrt(mean_squared_error(y, y_pred))

    reg_full = LinearRegression().fit(X_sc, y)
    results[name] = {
        "feats": feats, "r2": r2_loo, "rmse": rmse_loo,
        "coefs": reg_full.coef_, "intercept": reg_full.intercept_,
        "y_pred": y_pred,
    }
    # Store raw (unstandardised) AR(1) parameters
    if len(feats) == 1 and feats[0].endswith("_lag1"):
        mu_x, sd_x = scaler.mean_[0], scaler.scale_[0]
        slope_raw = reg_full.coef_[0] / sd_x
        intercept_raw = reg_full.intercept_ - reg_full.coef_[0] * mu_x / sd_x
        results[name]["ar1_slope"] = slope_raw
        results[name]["ar1_intercept"] = intercept_raw

    print(f"{name:35s}  n={len(feats):2d}  LOO R²={r2_loo:+.3f}  RMSE={rmse_loo:.2f} cm")

# ── Best model ──
best_name = max(results, key=lambda k: results[k]["r2"])
best = results[best_name]
print(f"\n{'='*60}")
print(f"Best: {best_name}  (R²={best['r2']:.3f}, RMSE={best['rmse']:.2f} cm)")
print(f"\nTop coefficients (standardised):")
for feat, coef in sorted(zip(best["feats"], best["coefs"]),
                          key=lambda x: abs(x[1]), reverse=True)[:8]:
    print(f"  {coef:+.3f}  {feat}")

# ── AR(1) raw parameters ──
for _ar_name, _ar_res in results.items():
    if "ar1_slope" in _ar_res:
        print(f"\n── {_ar_name}: raw AR(1) parameters ──")
        print(f"  Δ(t) = {_ar_res['ar1_intercept']:+.3f} + {_ar_res['ar1_slope']:.3f} · Δ(t−1)")
        print(f"  Persistence (slope):  {_ar_res['ar1_slope']:.3f}")
        print(f"  Long-run mean:        {_ar_res['ar1_intercept'] / (1 - _ar_res['ar1_slope']):+.2f} cm")
        print(f"  Half-life:            {-np.log(2) / np.log(abs(_ar_res['ar1_slope'])):.1f} months")

# ── Figure ──
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

ax = axes[0]
ax.plot(df_model["date"], y, lw=1, color="k", label="Observed")
ax.plot(df_model["date"], best["y_pred"], lw=1, color="tab:red",
        alpha=0.8, label=f"Predicted")
ax.fill_between(df_model["date"], y, best["y_pred"], alpha=0.15, color="tab:red")
ax.set_ylabel("NW − NE TWS (cm)")
ax.set_title(f"Observed vs predicted — {best_name}\n(LOO R²={best['r2']:.2f})")
ax.legend(fontsize=8)
ax.grid(True, lw=0.2)
ax.set_xlim(xlim)

ax = axes[1]
ax.scatter(y, best["y_pred"], s=15, alpha=0.5, color="tab:blue")
lims = [min(y.min(), best["y_pred"].min()) - 1,
        max(y.max(), best["y_pred"].max()) + 1]
ax.plot(lims, lims, "k--", lw=0.8)
ax.set_xlabel("Observed NW−NE (cm)")
ax.set_ylabel("Predicted NW−NE (cm)")
ax.set_title(f"1:1 plot (LOO R²={best['r2']:.2f})")
ax.set_aspect("equal")
ax.grid(True, lw=0.2)

ax = axes[2]
names = list(results.keys())
r2s = [results[n]["r2"] for n in names]
colors = ["tab:red" if n == best_name else "0.6" for n in names]
ax.barh(names, r2s, color=colors)
ax.set_xlabel("LOO R²")
ax.set_title("Model comparison")
ax.grid(True, lw=0.2, axis="x")
for i, v in enumerate(r2s):
    ax.text(max(v, 0) + 0.005, i, f"{v:.3f}", va="center", fontsize=9)

plt.tight_layout()
plt.show()
''')

PREDICT_SWNW_CELL = code(r'''
# ── Predicting SW − NW TWS from rainfall & discharge ──
# Same continuous predictor table (df_pred); target and AR(1) term on the full monthly grid.
_tgt_sw = df_sw_nw.set_index("date")["SW_NW_diff_cm"].reindex(df_pred.index)
df_model_sw = df_pred.copy()
df_model_sw["SW_NW_diff_cm"] = _tgt_sw
df_model_sw["SW_NW_lag1"] = _tgt_sw.shift(1)          # NaN across GRACE gaps
df_model_sw = df_model_sw.dropna().reset_index()
y_sw = df_model_sw["SW_NW_diff_cm"].values
print(f"SW−NW model dataset: {len(df_model_sw)} months  "
      f"({df_model_sw['date'].min().date()} → {df_model_sw['date'].max().date()})")

# ── Feature sets ──
base_flux_sw = ["P_angola_mm", "P_delta_mm", "Qin_m3s"]
feature_sets_sw = {
    "Flux only": base_flux_sw,
    "Flux + gradient + season": base_flux_sw + ["P_gradient_mm", "sin_mo", "cos_mo"],
    "6-mo means + season": [f"{c}_6mo" for c in
        ["P_angola_mm", "P_delta_mm", "P_gradient_mm", "Qin_m3s"]] + ["sin_mo", "cos_mo"],
    "AR(1) only": ["SW_NW_lag1"],
    "AR(1) + flux + season": ["SW_NW_lag1", "sin_mo", "cos_mo"] + base_flux_sw,
}

# ── LOO cross-validated regression ──
results_sw = {}
for name, feats in feature_sets_sw.items():
    X = df_model_sw[feats].values
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    loo = LeaveOneOut()
    y_pred = np.full_like(y_sw, np.nan)
    for train_idx, test_idx in loo.split(X_sc):
        reg = LinearRegression()
        reg.fit(X_sc[train_idx], y_sw[train_idx])
        y_pred[test_idx] = reg.predict(X_sc[test_idx])

    r2_loo = r2_score(y_sw, y_pred)
    rmse_loo = np.sqrt(mean_squared_error(y_sw, y_pred))

    reg_full = LinearRegression().fit(X_sc, y_sw)
    results_sw[name] = {
        "feats": feats, "r2": r2_loo, "rmse": rmse_loo,
        "coefs": reg_full.coef_, "intercept": reg_full.intercept_,
        "y_pred": y_pred,
    }
    # Store raw (unstandardised) AR(1) parameters
    if len(feats) == 1 and feats[0].endswith("_lag1"):
        mu_x, sd_x = scaler.mean_[0], scaler.scale_[0]
        slope_raw = reg_full.coef_[0] / sd_x
        intercept_raw = reg_full.intercept_ - reg_full.coef_[0] * mu_x / sd_x
        results_sw[name]["ar1_slope"] = slope_raw
        results_sw[name]["ar1_intercept"] = intercept_raw

    print(f"{name:35s}  n={len(feats):2d}  LOO R²={r2_loo:+.3f}  RMSE={rmse_loo:.2f} cm")

# ── Best model ──
best_name_sw = max(results_sw, key=lambda k: results_sw[k]["r2"])
best_sw = results_sw[best_name_sw]
print(f"\n{'='*60}")
print(f"Best: {best_name_sw}  (R²={best_sw['r2']:.3f}, RMSE={best_sw['rmse']:.2f} cm)")
print(f"\nTop coefficients (standardised):")
for feat, coef in sorted(zip(best_sw["feats"], best_sw["coefs"]),
                          key=lambda x: abs(x[1]), reverse=True)[:8]:
    print(f"  {coef:+.3f}  {feat}")

# ── AR(1) raw parameters ──
for _ar_name, _ar_res in results_sw.items():
    if "ar1_slope" in _ar_res:
        print(f"\n── {_ar_name}: raw AR(1) parameters ──")
        print(f"  Δ(t) = {_ar_res['ar1_intercept']:+.3f} + {_ar_res['ar1_slope']:.3f} · Δ(t−1)")
        print(f"  Persistence (slope):  {_ar_res['ar1_slope']:.3f}")
        print(f"  Long-run mean:        {_ar_res['ar1_intercept'] / (1 - _ar_res['ar1_slope']):+.2f} cm")
        print(f"  Half-life:            {-np.log(2) / np.log(abs(_ar_res['ar1_slope'])):.1f} months")

# ── Figure ──
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

ax = axes[0]
ax.plot(df_model_sw["date"], y_sw, lw=1, color="k", label="Observed")
ax.plot(df_model_sw["date"], best_sw["y_pred"], lw=1, color="tab:red",
        alpha=0.8, label="Predicted")
ax.fill_between(df_model_sw["date"], y_sw, best_sw["y_pred"], alpha=0.15, color="tab:red")
ax.set_ylabel("SW − NW TWS (cm)")
ax.set_title(f"Observed vs predicted — {best_name_sw}\n(LOO R²={best_sw['r2']:.2f})")
ax.legend(fontsize=8)
ax.grid(True, lw=0.2)
ax.set_xlim(xlim)

ax = axes[1]
ax.scatter(y_sw, best_sw["y_pred"], s=15, alpha=0.5, color="tab:blue")
lims = [min(y_sw.min(), best_sw["y_pred"].min()) - 1,
        max(y_sw.max(), best_sw["y_pred"].max()) + 1]
ax.plot(lims, lims, "k--", lw=0.8)
ax.set_xlabel("Observed SW−NW (cm)")
ax.set_ylabel("Predicted SW−NW (cm)")
ax.set_title(f"1:1 plot (LOO R²={best_sw['r2']:.2f})")
ax.set_aspect("equal")
ax.grid(True, lw=0.2)

ax = axes[2]
names_sw = list(results_sw.keys())
r2s_sw = [results_sw[n]["r2"] for n in names_sw]
colors_sw = ["tab:red" if n == best_name_sw else "0.6" for n in names_sw]
ax.barh(names_sw, r2s_sw, color=colors_sw)
ax.set_xlabel("LOO R²")
ax.set_title("Model comparison (SW − NW)")
ax.grid(True, lw=0.2, axis="x")
for i, v in enumerate(r2s_sw):
    ax.text(max(v, 0) + 0.005, i, f"{v:.3f}", va="center", fontsize=9)

plt.tight_layout()
plt.show()
''')

MEMORY_CELL = code(r'''
# ── Extended-memory rainfall model ──
# Long-memory predictors are built on the CONTINUOUS monthly predictor table (df_pred),
# so EWMAs, long rolling means, water-year sums and SPI never span GRACE gaps.
df_mem = df_pred.copy()

# ── 1. EWMA: exponentially weighted means with various half-lives ──
for hl in [3, 6, 12]:
    for col in ["P_angola_mm", "P_delta_mm", "Qin_m3s"]:
        df_mem[f"{col}_ewm{hl}"] = df_mem[col].ewm(halflife=hl).mean()

# ── 2. Longer rolling windows (18, 24 months) ──
for win in [18, 24]:
    for col in ["P_angola_mm", "P_delta_mm", "P_gradient_mm", "Qin_m3s"]:
        df_mem[f"{col}_{win}mo"] = df_mem[col].rolling(win, min_periods=win).mean()

# ── 3. Water-year cumulative rainfall (Oct–Sep) ──
# Water years are labelled by their *ending* year (Oct 2023–Sep 2024 → 2024);
# src/time_utils.water_year uses the *starting* year.
wy_arr = np.where(df_mem.index.month >= 10, df_mem.index.year + 1, df_mem.index.year)
for col in ["P_angola_mm", "P_delta_mm"]:
    df_mem[f"{col}_wy_cum"] = df_mem.groupby(wy_arr)[col].cumsum()

# ── 4. SPI-style standardized anomalies (rolling sum / climatological σ) ──
for win in [6, 12]:
    for col in ["P_angola_mm", "P_delta_mm"]:
        roll = df_mem[col].rolling(win, min_periods=win).sum()
        grp = roll.groupby(df_mem.index.month)        # standardise per calendar month
        df_mem[f"{col}_spi{win}"] = (roll - grp.transform("mean")) / grp.transform("std")

# ── 5. Longer lags (4–6 months) ──
for lag in [4, 5, 6]:
    for col in ["P_angola_mm", "P_delta_mm", "Qin_m3s"]:
        df_mem[f"{col}_lag{lag}"] = df_mem[col].shift(lag)

# ── Target + AR(1) on the same continuous grid ──
_tgt_mem = df_nw_ne.set_index("date")["NW_NE_diff_cm"].reindex(df_mem.index)
df_mem["NW_NE_diff_cm"] = _tgt_mem
df_mem["NW_NE_lag1"] = _tgt_mem.shift(1)
df_mem = df_mem.dropna().reset_index()
y_mem = df_mem["NW_NE_diff_cm"].values
print(f"Extended-memory dataset: {len(df_mem)} months  "
      f"({df_mem['date'].min().date()} → {df_mem['date'].max().date()})")

# ── Feature sets ──
base_flux = ["P_angola_mm", "P_delta_mm", "Qin_m3s"]
feature_sets_mem = {
    # Baselines (from previous cell)
    "AR(1) only (baseline)":
        ["NW_NE_lag1"],
    "Flux + season (baseline)":
        base_flux + ["P_gradient_mm", "sin_mo", "cos_mo"],
    "AR(1) + flux + season (baseline)":
        ["NW_NE_lag1", "sin_mo", "cos_mo"] + base_flux,

    # EWMA memory
    "EWMA-12 + season":
        [f"{c}_ewm12" for c in ["P_angola_mm", "P_delta_mm", "Qin_m3s"]]
        + ["sin_mo", "cos_mo"],
    "EWMA-6 + EWMA-12 + season":
        [f"{c}_ewm6" for c in ["P_angola_mm", "P_delta_mm", "Qin_m3s"]]
        + [f"{c}_ewm12" for c in ["P_angola_mm", "P_delta_mm", "Qin_m3s"]]
        + ["sin_mo", "cos_mo"],

    # Long rolling means
    "24-mo means + season":
        [f"{c}_24mo" for c in ["P_angola_mm", "P_delta_mm", "P_gradient_mm", "Qin_m3s"]]
        + ["sin_mo", "cos_mo"],

    # SPI-style
    "SPI-6 + SPI-12 + season":
        [f"{c}_spi6" for c in ["P_angola_mm", "P_delta_mm"]]
        + [f"{c}_spi12" for c in ["P_angola_mm", "P_delta_mm"]]
        + ["sin_mo", "cos_mo"],

    # Water-year cumulative
    "WY-cumulative + season":
        ["P_angola_mm_wy_cum", "P_delta_mm_wy_cum", "Qin_m3s", "sin_mo", "cos_mo"],

    # Kitchen sink: AR(1) + EWMA + SPI + season
    "AR(1) + EWMA-12 + SPI-12 + season":
        ["NW_NE_lag1"]
        + [f"{c}_ewm12" for c in ["P_angola_mm", "P_delta_mm", "Qin_m3s"]]
        + [f"{c}_spi12" for c in ["P_angola_mm", "P_delta_mm"]]
        + ["sin_mo", "cos_mo"],

    # Multi-scale: current + EWMA-6 + EWMA-12 + season
    "Multi-scale flux + season":
        base_flux
        + [f"{c}_ewm6" for c in ["P_angola_mm", "P_delta_mm", "Qin_m3s"]]
        + [f"{c}_ewm12" for c in ["P_angola_mm", "P_delta_mm", "Qin_m3s"]]
        + ["P_gradient_mm", "sin_mo", "cos_mo"],

    # Multi-scale + AR(1)
    "AR(1) + multi-scale flux + season":
        ["NW_NE_lag1"] + base_flux
        + [f"{c}_ewm6" for c in ["P_angola_mm", "P_delta_mm", "Qin_m3s"]]
        + [f"{c}_ewm12" for c in ["P_angola_mm", "P_delta_mm", "Qin_m3s"]]
        + ["P_gradient_mm", "sin_mo", "cos_mo"],
}

# ── LOO cross-validated Ridge regression (Ridge to handle collinearity) ──
from sklearn.model_selection import GridSearchCV

results_mem = {}
for name, feats in feature_sets_mem.items():
    X = df_mem[feats].values
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    # Quick α search via 5-fold CV, then LOO with best α
    ridge_cv = GridSearchCV(Ridge(), {"alpha": [0.01, 0.1, 1, 10, 100]},
                            cv=5, scoring="r2")
    ridge_cv.fit(X_sc, y_mem)
    best_alpha = ridge_cv.best_params_["alpha"]

    loo = LeaveOneOut()
    y_pred = np.full_like(y_mem, np.nan)
    for train_idx, test_idx in loo.split(X_sc):
        reg = Ridge(alpha=best_alpha)
        reg.fit(X_sc[train_idx], y_mem[train_idx])
        y_pred[test_idx] = reg.predict(X_sc[test_idx])

    r2_loo = r2_score(y_mem, y_pred)
    rmse_loo = np.sqrt(mean_squared_error(y_mem, y_pred))

    reg_full = Ridge(alpha=best_alpha).fit(X_sc, y_mem)
    results_mem[name] = {
        "feats": feats, "r2": r2_loo, "rmse": rmse_loo,
        "coefs": reg_full.coef_, "intercept": reg_full.intercept_,
        "y_pred": y_pred, "alpha": best_alpha,
    }
    # Store raw (unstandardised) AR(1) parameters for AR(1)-only models
    if feats == ["NW_NE_lag1"]:
        mu_x, sd_x = scaler.mean_[0], scaler.scale_[0]
        slope_raw = reg_full.coef_[0] / sd_x
        intercept_raw = reg_full.intercept_ - reg_full.coef_[0] * mu_x / sd_x
        results_mem[name]["ar1_slope"] = slope_raw
        results_mem[name]["ar1_intercept"] = intercept_raw

    print(f"{name:45s}  n={len(feats):2d}  α={best_alpha:5.2f}  "
          f"LOO R²={r2_loo:+.3f}  RMSE={rmse_loo:.2f} cm")

# ── Best model ──
best_name_mem = max(results_mem, key=lambda k: results_mem[k]["r2"])
best_mem = results_mem[best_name_mem]
print(f"\n{'='*65}")
print(f"Best: {best_name_mem}")
print(f"  LOO R²={best_mem['r2']:.3f},  RMSE={best_mem['rmse']:.2f} cm,  Ridge α={best_mem['alpha']}")
print(f"\nTop coefficients (standardised):")
for feat, coef in sorted(zip(best_mem["feats"], best_mem["coefs"]),
                          key=lambda x: abs(x[1]), reverse=True)[:10]:
    print(f"  {coef:+.3f}  {feat}")

# ── AR(1) raw parameters ──
for _ar_name, _ar_res in results_mem.items():
    if "ar1_slope" in _ar_res:
        print(f"\n── {_ar_name}: raw AR(1) parameters ──")
        print(f"  Δ(t) = {_ar_res['ar1_intercept']:+.3f} + {_ar_res['ar1_slope']:.3f} · Δ(t−1)")
        print(f"  Persistence (slope):  {_ar_res['ar1_slope']:.3f}")
        print(f"  Long-run mean:        {_ar_res['ar1_intercept'] / (1 - _ar_res['ar1_slope']):+.2f} cm")
        print(f"  Half-life:            {-np.log(2) / np.log(abs(_ar_res['ar1_slope'])):.1f} months")

# ── Figure ──
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Extended-memory rainfall models for NW − NE TWS", fontsize=13, y=1.02)

ax = axes[0]
ax.plot(df_mem["date"], y_mem, lw=1, color="k", label="Observed")
ax.plot(df_mem["date"], best_mem["y_pred"], lw=1, color="tab:red",
        alpha=0.8, label="Predicted")
ax.fill_between(df_mem["date"], y_mem, best_mem["y_pred"], alpha=0.15, color="tab:red")
ax.set_ylabel("NW − NE TWS (cm)")
ax.set_title(f"Observed vs predicted — {best_name_mem}\n(LOO R²={best_mem['r2']:.2f})")
ax.legend(fontsize=8)
ax.grid(True, lw=0.2)
ax.set_xlim(xlim)

ax = axes[1]
ax.scatter(y_mem, best_mem["y_pred"], s=15, alpha=0.5, color="tab:blue")
lims = [min(y_mem.min(), best_mem["y_pred"].min()) - 1,
        max(y_mem.max(), best_mem["y_pred"].max()) + 1]
ax.plot(lims, lims, "k--", lw=0.8)
ax.set_xlabel("Observed NW−NE (cm)")
ax.set_ylabel("Predicted NW−NE (cm)")
ax.set_title(f"1:1 plot (LOO R²={best_mem['r2']:.2f})")
ax.set_aspect("equal")
ax.grid(True, lw=0.2)

ax = axes[2]
sorted_names = sorted(results_mem, key=lambda k: results_mem[k]["r2"])
r2s = [results_mem[n]["r2"] for n in sorted_names]
colors = ["tab:red" if n == best_name_mem else "0.6" for n in sorted_names]
ax.barh(sorted_names, r2s, color=colors)
ax.set_xlabel("LOO R²")
ax.set_title("Model comparison (extended memory)")
ax.grid(True, lw=0.2, axis="x")
for i, v in enumerate(r2s):
    ax.text(max(v, 0) + 0.005, i, f"{v:.3f}", va="center", fontsize=8)

plt.tight_layout()
plt.show()
''')

# ---------------------------------------------------------------------------
# §13 — regime-adaptive ensemble ET (old cells 50–55)
# ---------------------------------------------------------------------------
ENSEMBLE_MD = md(r'''
## 13 — Regime-adaptive weighted ensemble ET

Each ET model $i$ receives a softmax weight that varies continuously with
the monthly water input $z_t = Q_{\text{in},t} + P_t$:

$$w_i(z) = \frac{\exp(a_i + b_i\, z)}{\sum_j \exp(a_j + b_j\, z)}$$

The ensemble ET is $\hat{E}_t = \sum_i w_i(z_t)\, E_{i,t}$, and the
parameters $\{a_i, b_i\}$ are calibrated by minimising the cumulative
budget residual:

$$\min_{\{a,b\}} \sum_t \bigl[Q_{\text{in},t} + P_t - \hat{E}_t(a,b) - \Delta S_t\bigr]^2$$

This lets the ensemble smoothly shift towards whichever model best
closes the budget in dry vs wet conditions.
''')

ENSEMBLE_QP_CELL = code(r'''
from scipy.optimize import minimize

# ── Build aligned arrays: ET models, Qin+P, ΔS ──
_mask_ens = df_balance[["Qin_km3", "P_km3", "dS_km3", "NW_NE_diff_cm"]].notna().all(axis=1)
et_ens = bal.et_km3_wide.loc[_mask_ens]          # per-product ET already on the balance grid

# Require at least 3 models per month
_nvalid = et_ens.notna().sum(axis=1)
et_ens = et_ens.loc[_nvalid >= 3]
_bal_ens = df_balance.loc[et_ens.index]

z_raw = (_bal_ens["Qin_km3"] + _bal_ens["P_km3"]).values
z_mu, z_sd = z_raw.mean(), z_raw.std()
z = (z_raw - z_mu) / z_sd

# Seasonal features
month_frac = _bal_ens.index.month.values * (2 * np.pi / 12)
sin_m = np.sin(month_frac)
cos_m = np.cos(month_frac)

# Lateral flux gradient (NW minus NE TWS, cm) — used post-hoc only
delta_nwne = _bal_ens["NW_NE_diff_cm"].values

ET_mat = et_ens.fillna(0.0).values
avail  = et_ens.notna().values.astype(float)
dS     = _bal_ens["dS_km3"].values
QplusP = z_raw
model_names = list(et_ens.columns)
M = len(model_names)
T = len(z)

# Features: [1, z, z², sin, cos, z·sin, z·cos]
N_PAR = 7
Z_feat = np.column_stack([np.ones(T), z, z**2, sin_m, cos_m,
                           z * sin_m, z * cos_m])

print(f"Ensemble calibration: {T} months, {M} ET models")
print(f"  Weight params: {N_PAR*M}, offset β: 1  →  {N_PAR*M+1} total")
print(f"z = (Qin+P), mean={z_mu:.3f} km³, std={z_sd:.3f} km³")
print(f"Date range: {et_ens.index.min():%Y-%m} – {et_ens.index.max():%Y-%m}")
for i, name in enumerate(model_names):
    n = int(avail[:, i].sum())
    print(f"  {name:25s}  {n}/{T} months available")

def softmax_weights_masked(coefs, feat, mask):
    """coefs: (N_PAR, M), feat: (T, N_PAR) → (T, M) weights."""
    logits = feat @ coefs
    logits = np.where(mask > 0, logits, -1e30)
    logits = logits / TEMPERATURE
    logits -= logits.max(axis=1, keepdims=True)
    w = np.exp(logits)
    w = np.where(mask > 0, w, 0.0)
    return w / w.sum(axis=1, keepdims=True)

LAMBDA_L2 = 0.5
TEMPERATURE = 4

def objective(params):
    coefs = params[:N_PAR * M].reshape(N_PAR, M)
    beta  = params[N_PAR * M]

    W = softmax_weights_masked(coefs, Z_feat, avail)
    et_hat = (W * ET_mat).sum(axis=1)
    resid = QplusP - et_hat - beta - dS
    l2 = LAMBDA_L2 * np.sum(params[:N_PAR * M]**2)
    return np.sum(resid**2) + l2

# ── Multi-start calibration (20 starts) ──
n_params = N_PAR * M + 1
rng = np.random.default_rng(42)
best_res = None
for trial in range(20):
    x0 = np.zeros(n_params)
    if trial > 0:
        x0[:N_PAR * M] = rng.normal(0, 0.3, N_PAR * M)
        x0[N_PAR * M] = rng.uniform(-2, 2)
    r = minimize(objective, x0, method="L-BFGS-B",
                 options={"maxiter": 12000, "ftol": 1e-15})
    if best_res is None or r.fun < best_res.fun:
        best_res = r

coefs_cal = best_res.x[:N_PAR * M].reshape(N_PAR, M)
beta_ens  = best_res.x[N_PAR * M]

print(f"\nOptimisation converged: {best_res.success}  (obj={best_res.fun:.2f})")
print(f"β (constant offset)   = {beta_ens:.3f} km³/month")
_mean_qp = np.mean(QplusP)
print(f"  → β / mean(Qin+P)  = {beta_ens/_mean_qp*100:.2f}%  "
      f"(mean Qin+P = {_mean_qp:.1f} km³/month)")
feat_names = ["a", "b_z", "b_z²", "b_sin", "b_cos", "b_z·sin", "b_z·cos"]
header = f"{'Model':25s} " + " ".join(f"{fn:>8s}" for fn in feat_names)
print(f"\n{header}")
for i, name in enumerate(model_names):
    vals = coefs_cal[:, i]
    print(f"{name:25s} " + " ".join(f"{v:+8.3f}" for v in vals))

# ── Compute ensembles ──
W_cal = softmax_weights_masked(coefs_cal, Z_feat, avail)
ET_ensemble = (W_cal * ET_mat).sum(axis=1)

W_uni = avail / avail.sum(axis=1, keepdims=True)
ET_uniform = (W_uni * ET_mat).sum(axis=1)

# Budget residuals (before α correction)
resid_ens = QplusP - ET_ensemble - beta_ens - dS
resid_uni = QplusP - ET_uniform - dS

# ── Post-hoc α·Δ(NW-NE) correction via OLS on residuals ──
# Sign convention (used consistently below):  Qin + P − ET − β + α·Δ ≈ ΔS,
# i.e. corrected residual = resid_ens + α·Δ.  OLS for α minimising ||resid_ens + α·Δ||²:
alpha_ens = -np.dot(resid_ens, delta_nwne) / np.dot(delta_nwne, delta_nwne)
resid_corrected = resid_ens + alpha_ens * delta_nwne

rmse_uni = np.sqrt(np.mean(resid_uni**2))
rmse_ens = np.sqrt(np.mean(resid_ens**2))
rmse_corr = np.sqrt(np.mean(resid_corrected**2))

print(f"\n── Post-hoc α·Δ(NW-NE) correction ──")
print(f"α = {alpha_ens:.5f} km³/cm  (OLS on ensemble residuals)")
print(f"\nRMSE (monthly residual):")
print(f"  Uniform:                     {rmse_uni:.4f} km³/month")
print(f"  Adaptive + β:                {rmse_ens:.4f} km³/month  ({(1-rmse_ens/rmse_uni)*100:.1f}%)")
print(f"  Adaptive + β + α·Δ:          {rmse_corr:.4f} km³/month  ({(1-rmse_corr/rmse_uni)*100:.1f}%)")
print(f"  Mean bias (uniform):         {np.mean(resid_uni):+.3f} km³/month")
print(f"  Mean bias (adaptive+β):      {np.mean(resid_ens):+.3f} km³/month")
print(f"  Mean bias (adaptive+β+α·Δ):  {np.mean(resid_corrected):+.3f} km³/month")

print(f"\nMean weight per model:")
for i, name in enumerate(model_names):
    wmean = W_cal[:, i][avail[:, i] > 0].mean()
    print(f"  {name:25s}  {wmean:.3f}")

# Cumulative budget
cum_ds   = np.cumsum(dS)
cum_uni  = np.cumsum(QplusP - ET_uniform)
cum_ens  = np.cumsum(QplusP - ET_ensemble - beta_ens)
cum_corr = np.cumsum(QplusP - ET_ensemble - beta_ens + alpha_ens * delta_nwne)

# ── Plots ──
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: weights vs Qin+P (all models, annual mean season → sin = cos = 0)
z_plot = np.linspace(z.min() - 0.5, z.max() + 0.5, 200)
feat_plot = np.column_stack([np.ones(200), z_plot, z_plot**2,
                             np.zeros(200), np.zeros(200),
                             np.zeros(200), np.zeros(200)])
mask_plot = np.ones((200, M))
W_plot = softmax_weights_masked(coefs_cal, feat_plot, mask_plot)
z_plot_real = z_plot * z_sd + z_mu

ax = axes[0, 0]
for i, name in enumerate(model_names):
    ax.plot(z_plot_real, W_plot[:, i], lw=1.5, label=name)
ax.set_xlabel("Monthly Qin + P (km³)")
ax.set_ylabel("Weight")
ax.set_title("Weights vs water input (annual mean season)")
ax.legend(fontsize=7, loc="best")
ax.grid(True, lw=0.2)

# Panel 2: realized seasonal weight cycle — median with IQR shading
_months_arr_qp = _bal_ens.index.month.values
ax = axes[0, 1]
for i, name in enumerate(model_names):
    med_w, lo_w, hi_w = [], [], []
    for mo in range(1, 13):
        w_mo = W_cal[_months_arr_qp == mo, i]
        med_w.append(np.median(w_mo))
        lo_w.append(np.percentile(w_mo, 25))
        hi_w.append(np.percentile(w_mo, 75))
    months_plot = np.arange(1, 13)
    ax.plot(months_plot, med_w, lw=1.5, marker="o", ms=4, label=name)
    ax.fill_between(months_plot, lo_w, hi_w, alpha=0.15)
ax.set_xlabel("Month")
ax.set_ylabel("Weight")
ax.set_xticks(months_plot)
ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
ax.set_title("Realized seasonal weights (median ± IQR)")
ax.legend(fontsize=7, loc="best")
ax.grid(True, lw=0.2)

# Panel 3: cumulative budget closure
ax = axes[1, 0]
dates_ens = _bal_ens.index
ax.plot(dates_ens, cum_ds, lw=2.5, ls=":", color="#c51b7d", label="∑ΔS (GRACE)")
ax.plot(dates_ens, cum_uni, lw=1, alpha=0.5, color="0.6", label=f"Uniform ({rmse_uni:.2f})")
ax.plot(dates_ens, cum_ens, lw=1.5, alpha=0.6, ls="--", color="#ff7f00",
        label=f"Adaptive + β ({rmse_ens:.2f})")
ax.plot(dates_ens, cum_corr, lw=2, color="#2ca02c",
        label=f"Adaptive + β + α·Δ ({rmse_corr:.2f})")
ax.set_ylabel("km³ (cumulative)")
ax.set_title("Cumulative budget closure")
ax.legend(fontsize=7)
ax.grid(True, lw=0.2)
ax.set_xlim(xlim)

# Panel 4: monthly residuals
ax = axes[1, 1]
ax.bar(dates_ens, resid_uni, width=25, color="0.7", alpha=0.4, label="Uniform")
ax.bar(dates_ens, resid_corrected, width=25, color="#2ca02c", alpha=0.7,
       label="Adaptive+β+α·Δ")
ax.axhline(0, lw=0.8, c="0.4", ls=":")
ax.set_ylabel("Residual (km³ month⁻¹)")
ax.set_title(f"Monthly budget residual\nRMSE: {rmse_uni:.3f} → {rmse_corr:.3f} km³")
ax.legend(fontsize=8)
ax.grid(True, lw=0.2)
ax.set_xlim(xlim)

plt.tight_layout()
plt.show()

# ── Weight table at dry / normal / wet × season ──
for pctl_label, pctl in [("Dry (10th pctl)", 10), ("Normal (50th)", 50), ("Wet (90th pctl)", 90)]:
    z_p = np.percentile(z_raw, pctl)
    z_s = (z_p - z_mu) / z_sd
    for season, sm, cm in [("Jan (wet)", np.sin(1*2*np.pi/12), np.cos(1*2*np.pi/12)),
                           ("Jul (dry)", np.sin(7*2*np.pi/12), np.cos(7*2*np.pi/12))]:
        feat_q = np.array([[1, z_s, z_s**2, sm, cm, z_s*sm, z_s*cm]])
        w = softmax_weights_masked(coefs_cal, feat_q, np.ones((1, M)))[0]

        top3 = np.argsort(w)[::-1][:3]
        top_str = ", ".join(f"{model_names[j]} {w[j]:.2f}" for j in top3)
        print(f"  {pctl_label} / {season}: {top_str}")
''')

DSWE_FILL_CELL = code(r'''
# ── Gap-fill Landsat DSWE with P + Q statistical model ──
# Build a monthly DSWE series, predict missing months using Qin, P, lags, and season
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict

# ── Observed monthly DSWE (Landsat) ──
_dswe_obs = df_landsat.set_index("date")["km2"].copy()
_dswe_obs.index = pd.to_datetime(_dswe_obs.index).to_period("M").to_timestamp()
_dswe_obs = _dswe_obs.groupby(_dswe_obs.index).mean()
_dswe_obs.name = "dswe_km2"

# ── Target index: the continuous balance grid (lags/rolling means below then never
#    span a Mohembo gap; months without Qin or P simply have NaN features) ──
_bal_qp = df_balance[["Qin_km3", "P_km3"]]
target_idx = _bal_qp.index

# ── Merge into a single dataframe ──
df_dswe = pd.DataFrame(index=target_idx)
df_dswe["Qin"] = _bal_qp["Qin_km3"]
df_dswe["P"]   = _bal_qp["P_km3"]
df_dswe["dswe_obs"] = _dswe_obs.reindex(target_idx)
df_dswe["month"] = df_dswe.index.month

# Seasonal harmonics
mo = df_dswe.index.month
df_dswe["sin1"] = np.sin(2 * np.pi * mo / 12)
df_dswe["cos1"] = np.cos(2 * np.pi * mo / 12)
df_dswe["sin2"] = np.sin(4 * np.pi * mo / 12)
df_dswe["cos2"] = np.cos(4 * np.pi * mo / 12)

# Lagged predictors (1, 2, 3 months)
for lag in [1, 2, 3]:
    df_dswe[f"Qin_lag{lag}"] = df_dswe["Qin"].shift(lag)
    df_dswe[f"P_lag{lag}"]   = df_dswe["P"].shift(lag)

# Rolling means (3- and 6-month)
for win in [3, 6]:
    df_dswe[f"Qin_{win}mo"] = df_dswe["Qin"].rolling(win, min_periods=win).mean()
    df_dswe[f"P_{win}mo"]   = df_dswe["P"].rolling(win, min_periods=win).mean()

# Lagged DSWE (carry forward last observed for prediction)
df_dswe["dswe_lag1"] = df_dswe["dswe_obs"].shift(1)
# Forward-fill the lag so that even after a gap, the last observation propagates
df_dswe["dswe_lag1_ff"] = df_dswe["dswe_lag1"].ffill()

# ── Feature list ──
feat_cols = ["Qin", "P", "sin1", "cos1", "sin2", "cos2",
             "Qin_lag1", "P_lag1", "Qin_lag2", "P_lag2", "Qin_lag3", "P_lag3",
             "Qin_3mo", "P_3mo", "Qin_6mo", "P_6mo",
             "dswe_lag1_ff"]

# ── Train on observed DSWE months (where all features available) ──
train_mask = df_dswe["dswe_obs"].notna() & df_dswe[feat_cols].notna().all(axis=1)
df_train = df_dswe.loc[train_mask].copy()

X_train = df_train[feat_cols].values
y_train = df_train["dswe_obs"].values

scaler_dswe = StandardScaler()
X_train_sc = scaler_dswe.fit_transform(X_train)

# Ridge regression (α=10 for some regularisation)
ridge_dswe = Ridge(alpha=10)
ridge_dswe.fit(X_train_sc, y_train)

# LOO cross-validation for accuracy assessment
y_loo = cross_val_predict(Ridge(alpha=10), X_train_sc, y_train, cv=LeaveOneOut())
rmse_loo_dswe = np.sqrt(np.mean((y_train - y_loo)**2))
ss_res = np.sum((y_train - y_loo)**2)
ss_tot = np.sum((y_train - y_train.mean())**2)
r2_loo_dswe = 1 - ss_res / ss_tot

print(f"Training: {len(df_train)} months with observed DSWE + all features")
print(f"LOO CV:  RMSE = {rmse_loo_dswe:.1f} km²,  R² = {r2_loo_dswe:.3f}")
print(f"Mean DSWE = {y_train.mean():.0f} km²,  std = {y_train.std():.0f} km²")

# Feature importances (standardised coefficients)
print(f"\nFeature coefficients (standardised):")
for fname, coef in sorted(zip(feat_cols, ridge_dswe.coef_), key=lambda x: -abs(x[1])):
    print(f"  {fname:18s}  {coef:+.1f}")

# ── Predict missing months ──
# For missing months where features are available, predict DSWE
pred_mask = df_dswe["dswe_obs"].isna() & df_dswe[feat_cols].notna().all(axis=1)
n_predicted = pred_mask.sum()

X_pred = df_dswe.loc[pred_mask, feat_cols].values
X_pred_sc = scaler_dswe.transform(X_pred)
dswe_predicted = ridge_dswe.predict(X_pred_sc)
# Clip to physical range
dswe_predicted = np.clip(dswe_predicted, 0, 15000)

# ── Build complete series ──
df_dswe["dswe_filled"] = df_dswe["dswe_obs"].copy()
df_dswe.loc[pred_mask, "dswe_filled"] = dswe_predicted
df_dswe["dswe_source"] = "observed"
df_dswe.loc[pred_mask, "dswe_source"] = "predicted"

n_filled = df_dswe["dswe_filled"].notna().sum()
n_obs_total = df_dswe["dswe_obs"].notna().sum()
print(f"\nGap-filling summary:")
print(f"  Observed months:      {n_obs_total}")
print(f"  Predicted (filled):   {n_predicted}")
print(f"  Total available:      {n_filled}")
print(f"  Still missing:        {df_dswe['dswe_filled'].isna().sum()}")

# ── Plot ──
fig, axes = plt.subplots(2, 1, figsize=(14, 7))

# Panel 1: time series with observed vs predicted
ax = axes[0]
obs_idx = df_dswe["dswe_source"] == "observed"
pred_idx = df_dswe["dswe_source"] == "predicted"
# Continuous filled line behind the markers
_filled = df_dswe["dswe_filled"].dropna()
ax.plot(_filled.index, _filled.values, lw=0.6, color="0.6", zorder=1)
ax.plot(df_dswe.index[obs_idx], df_dswe.loc[obs_idx, "dswe_filled"],
        "o", ms=3, color="steelblue", alpha=0.6, label="Observed (Landsat)", zorder=2)
ax.plot(df_dswe.index[pred_idx], df_dswe.loc[pred_idx, "dswe_filled"],
        "s", ms=4, color="orangered", alpha=0.8, label="Predicted (Ridge)", zorder=2)
ax.set_ylabel("Flood extent (km²)")
ax.set_title(f"Landsat DSWE gap-filling  (LOO R²={r2_loo_dswe:.2f}, RMSE={rmse_loo_dswe:.0f} km²)")
ax.legend(fontsize=9)
ax.grid(True, lw=0.2)

# Panel 2: LOO predicted vs observed (scatter)
ax = axes[1]
ax.scatter(y_train, y_loo, s=15, alpha=0.5, c="steelblue", edgecolors="none")
lims = [min(y_train.min(), y_loo.min()) - 200, max(y_train.max(), y_loo.max()) + 200]
ax.plot(lims, lims, "k--", lw=0.8)
ax.set_xlabel("Observed DSWE (km²)")
ax.set_ylabel("LOO predicted DSWE (km²)")
ax.set_title(f"Leave-one-out cross-validation  (n={len(y_train)})")
ax.set_aspect("equal")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.grid(True, lw=0.2)

plt.tight_layout()
plt.show()

# ── Export filled series for use in the ensemble cell below ──
dswe_filled_series = df_dswe["dswe_filled"].dropna()
print(f"\ndswe_filled_series: {len(dswe_filled_series)} months ready for ensemble")
''')

ENSEMBLE_DSWE_CELL = code(r'''
# ── Ensemble weighted by DSWE flood extent (Landsat + gap-filled) instead of Qin+P ──
from scipy.optimize import minimize

# ── Use gap-filled DSWE series from cell above ──
_dswe = dswe_filled_series.copy()

# ── Build aligned arrays ──
_mask_dswe = (df_balance[["Qin_km3", "P_km3", "dS_km3"]].notna().all(axis=1)
              & df_balance.index.isin(_dswe.index))
et_ens_d = bal.et_km3_wide.loc[_mask_dswe]

_nvalid_d = et_ens_d.notna().sum(axis=1)
et_ens_d = et_ens_d.loc[_nvalid_d >= 3]
_bal_d = df_balance.loc[et_ens_d.index]

# Conditioning variable: DSWE flood extent (km²)
dswe_raw = _dswe.reindex(et_ens_d.index).values
z_d_mu, z_d_sd = np.nanmean(dswe_raw), np.nanstd(dswe_raw)
z_d = (dswe_raw - z_d_mu) / z_d_sd

# Seasonal features
month_frac_d = _bal_d.index.month.values * (2 * np.pi / 12)
sin_m_d = np.sin(month_frac_d)
cos_m_d = np.cos(month_frac_d)

ET_mat_d = et_ens_d.fillna(0.0).values
avail_d  = et_ens_d.notna().values.astype(float)
dS_d     = _bal_d["dS_km3"].values
QpP_d    = (_bal_d["Qin_km3"] + _bal_d["P_km3"]).values
model_names_d = list(et_ens_d.columns)
M_d = len(model_names_d)
T_d = len(z_d)

# Features: [1, z, z², sin, cos, z·sin, z·cos]
N_PAR_d = 7
Z_feat_d = np.column_stack([np.ones(T_d), z_d, z_d**2, sin_m_d, cos_m_d,
                             z_d * sin_m_d, z_d * cos_m_d])

# Count observed vs gap-filled months in this ensemble
_obs_months = df_dswe.loc[et_ens_d.index, "dswe_source"]
n_obs_in_ens = (_obs_months == "observed").sum()
n_pred_in_ens = (_obs_months == "predicted").sum()

print(f"DSWE-conditioned ensemble (gap-filled): {T_d} months, {M_d} ET models")
print(f"  DSWE source: {n_obs_in_ens} observed + {n_pred_in_ens} gap-filled")
print(f"  Weight params: {N_PAR_d*M_d}, offset β: 1  →  {N_PAR_d*M_d+1} total")
print(f"z = DSWE flood extent, mean={z_d_mu:.1f} km², std={z_d_sd:.1f} km²")
print(f"Date range: {et_ens_d.index.min():%Y-%m} – {et_ens_d.index.max():%Y-%m}")
for i, name in enumerate(model_names_d):
    n = int(avail_d[:, i].sum())
    print(f"  {name:25s}  {n}/{T_d} months available")

LAMBDA_L2_d = 0.5
TEMPERATURE_d = 4

def softmax_masked_d(coefs, feat, mask):
    logits = feat @ coefs
    logits = np.where(mask > 0, logits, -1e30)
    logits = logits / TEMPERATURE_d
    logits -= logits.max(axis=1, keepdims=True)
    w = np.exp(logits)
    w = np.where(mask > 0, w, 0.0)
    return w / w.sum(axis=1, keepdims=True)

def objective_d(params):
    coefs = params[:N_PAR_d * M_d].reshape(N_PAR_d, M_d)
    beta  = params[N_PAR_d * M_d]
    W = softmax_masked_d(coefs, Z_feat_d, avail_d)
    et_hat = (W * ET_mat_d).sum(axis=1)
    resid = QpP_d - et_hat - beta - dS_d
    l2 = LAMBDA_L2_d * np.sum(params[:N_PAR_d * M_d]**2)
    return np.sum(resid**2) + l2

# ── Multi-start calibration (20 starts) ──
n_params_d = N_PAR_d * M_d + 1
rng_d = np.random.default_rng(99)
best_d = None
for trial in range(20):
    x0 = np.zeros(n_params_d)
    if trial > 0:
        x0[:N_PAR_d * M_d] = rng_d.normal(0, 0.3, N_PAR_d * M_d)
        x0[N_PAR_d * M_d] = rng_d.uniform(-2, 2)
    r = minimize(objective_d, x0, method="L-BFGS-B",
                 options={"maxiter": 12000, "ftol": 1e-15})
    if best_d is None or r.fun < best_d.fun:
        best_d = r

coefs_d = best_d.x[:N_PAR_d * M_d].reshape(N_PAR_d, M_d)
beta_d  = best_d.x[N_PAR_d * M_d]

print(f"\nOptimisation converged: {best_d.success}  (obj={best_d.fun:.2f})")
print(f"β (constant offset)   = {beta_d:.3f} km³/month")
_mean_qp_d = np.mean(QpP_d)
print(f"  → β / mean(Qin+P)  = {beta_d/_mean_qp_d*100:.2f}%  "
      f"(mean Qin+P = {_mean_qp_d:.1f} km³/month)")

# ── Compute ensembles ──
W_d = softmax_masked_d(coefs_d, Z_feat_d, avail_d)
ET_ens_d = (W_d * ET_mat_d).sum(axis=1)

W_uni_d = avail_d / avail_d.sum(axis=1, keepdims=True)
ET_uni_d = (W_uni_d * ET_mat_d).sum(axis=1)

resid_ens_d = QpP_d - ET_ens_d - beta_d - dS_d
resid_uni_d = QpP_d - ET_uni_d - dS_d

rmse_uni_d  = np.sqrt(np.mean(resid_uni_d**2))
rmse_ens_d  = np.sqrt(np.mean(resid_ens_d**2))
improv_d = (1 - rmse_ens_d / rmse_uni_d) * 100

print(f"\nRMSE (monthly residual):")
print(f"  Uniform:            {rmse_uni_d:.4f} km³/month")
print(f"  Adaptive + β:       {rmse_ens_d:.4f} km³/month  ({improv_d:.1f}%)")
print(f"  Mean bias (uniform):    {np.mean(resid_uni_d):+.3f} km³/month")
print(f"  Mean bias (adaptive+β): {np.mean(resid_ens_d):+.3f} km³/month")

print(f"\nMean weight per model:")
for i, name in enumerate(model_names_d):
    wmean = W_d[:, i][avail_d[:, i] > 0].mean()
    print(f"  {name:25s}  {wmean:.3f}")

# Comparison with Qin+P version
print(f"\n── Comparison: DSWE vs Qin+P conditioning ──")
print(f"  Qin+P  ({T} months): RMSE {rmse_ens:.4f} km³/month")
print(f"  DSWE   ({T_d} months): RMSE {rmse_ens_d:.4f} km³/month")

# ── Cumulative budget ──
cum_ds_d   = np.cumsum(dS_d)
cum_uni_d  = np.cumsum(QpP_d - ET_uni_d)
cum_ens_d  = np.cumsum(QpP_d - ET_ens_d - beta_d)

# ── Plots ──
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Ensemble ET — conditioned on DSWE flood extent (Landsat + gap-filled)",
             fontsize=13, y=0.98)

# Panel 1: weights vs DSWE flood extent (annual mean season → sin = cos = 0)
z_plot_d = np.linspace(z_d.min() - 0.5, z_d.max() + 0.5, 200)
feat_plot_d = np.column_stack([np.ones(200), z_plot_d, z_plot_d**2,
                               np.zeros(200), np.zeros(200),
                               np.zeros(200), np.zeros(200)])
W_plot_d = softmax_masked_d(coefs_d, feat_plot_d, np.ones((200, M_d)))
z_plot_real_d = z_plot_d * z_d_sd + z_d_mu

ax = axes[0, 0]
for i, name in enumerate(model_names_d):
    ax.plot(z_plot_real_d, W_plot_d[:, i], lw=1.5, label=name)
ax.set_xlabel("DSWE flood extent (km²)")
ax.set_ylabel("Weight")
ax.set_title("Weights vs flood extent (annual mean season)")
ax.legend(fontsize=7, loc="best")
ax.grid(True, lw=0.2)

# Panel 2: realized seasonal weight cycle — median with IQR shading
_months_arr = _bal_d.index.month.values          # month for each timestep
ax = axes[0, 1]
for i, name in enumerate(model_names_d):
    med_w, lo_w, hi_w = [], [], []
    for mo in range(1, 13):
        w_mo = W_d[_months_arr == mo, i]
        med_w.append(np.median(w_mo))
        lo_w.append(np.percentile(w_mo, 25))
        hi_w.append(np.percentile(w_mo, 75))
    months_d = np.arange(1, 13)
    ax.plot(months_d, med_w, lw=1.5, marker="o", ms=4, label=name)
    ax.fill_between(months_d, lo_w, hi_w, alpha=0.15)
ax.set_xlabel("Month")
ax.set_ylabel("Weight")
ax.set_xticks(months_d)
ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
ax.set_title("Realized seasonal weights (median ± IQR)")
ax.legend(fontsize=7, loc="best")
ax.grid(True, lw=0.2)

# Panel 3: cumulative budget closure
ax = axes[1, 0]
dates_d = _bal_d.index
ax.plot(dates_d, cum_ds_d, lw=2.5, ls=":", color="#c51b7d", label="∑ΔS (GRACE)")
ax.plot(dates_d, cum_uni_d, lw=1, alpha=0.5, color="0.6", label=f"Uniform ({rmse_uni_d:.2f})")
ax.plot(dates_d, cum_ens_d, lw=2, color="#2ca02c",
        label=f"Adaptive + β ({rmse_ens_d:.2f})")
ax.set_ylabel("km³ (cumulative)")
ax.set_title("Cumulative budget closure")
ax.legend(fontsize=7)
ax.grid(True, lw=0.2)

# Panel 4: monthly residuals
ax = axes[1, 1]
ax.bar(dates_d, resid_uni_d, width=25, color="0.7", alpha=0.4, label="Uniform")
ax.bar(dates_d, resid_ens_d, width=25, color="#2ca02c", alpha=0.7,
       label="Adaptive+β")
ax.axhline(0, lw=0.8, c="0.4", ls=":")
ax.set_ylabel("Residual (km³ month⁻¹)")
ax.set_title(f"Monthly budget residual\nRMSE: {rmse_uni_d:.3f} → {rmse_ens_d:.3f} km³")
ax.legend(fontsize=8)
ax.grid(True, lw=0.2)

plt.tight_layout()
plt.show()

# ── Weight table ──
for pctl_label, pctl in [("Low flood (10th)", 10), ("Median flood (50th)", 50), ("High flood (90th)", 90)]:
    z_p = np.percentile(dswe_raw, pctl)
    z_s = (z_p - z_d_mu) / z_d_sd
    for season, sm, cm in [("Jan (wet)", np.sin(1*2*np.pi/12), np.cos(1*2*np.pi/12)),
                           ("Jul (dry)", np.sin(7*2*np.pi/12), np.cos(7*2*np.pi/12))]:
        feat_q = np.array([[1, z_s, z_s**2, sm, cm, z_s*sm, z_s*cm]])
        w = softmax_masked_d(coefs_d, feat_q, np.ones((1, M_d)))[0]

        top3 = np.argsort(w)[::-1][:3]
        top_str = ", ".join(f"{model_names_d[j]} {w[j]:.2f}" for j in top3)
        print(f"  {pctl_label} / {season}: {top_str}")
''')

CONST_WEIGHT_CELL = code(r'''
# ── Constant-weight ensemble (one fixed weight per model, no conditioning) ──
# Baseline between equal-weight uniform and the fully adaptive ensemble.
# Parameters: M logits → softmax weights + β offset = M+1 total.
from scipy.optimize import minimize

def objective_const(params):
    logits_c = params[:M_d]
    beta_c   = params[M_d]
    # Masked softmax: for each timestep, zero out unavailable models
    logits_2d = np.tile(logits_c, (T_d, 1))
    logits_2d = np.where(avail_d > 0, logits_2d, -1e30)
    logits_2d -= logits_2d.max(axis=1, keepdims=True)
    w = np.exp(logits_2d)
    w = np.where(avail_d > 0, w, 0.0)
    w = w / w.sum(axis=1, keepdims=True)
    et_hat = (w * ET_mat_d).sum(axis=1)
    resid = QpP_d - et_hat - beta_c - dS_d
    return np.sum(resid**2)

# Multi-start (10 starts)
rng_c = np.random.default_rng(77)
best_c = None
for trial in range(10):
    x0 = np.zeros(M_d + 1)
    if trial > 0:
        x0[:M_d] = rng_c.normal(0, 0.5, M_d)
        x0[M_d]  = rng_c.uniform(-2, 2)
    r = minimize(objective_const, x0, method="L-BFGS-B",
                 options={"maxiter": 5000, "ftol": 1e-15})
    if best_c is None or r.fun < best_c.fun:
        best_c = r

logits_const = best_c.x[:M_d]
beta_const   = best_c.x[M_d]

# Recover constant weights (at full availability)
_exp = np.exp(logits_const - logits_const.max())
w_const = _exp / _exp.sum()

# Per-timestep weights (masking unavailable models)
logits_2d = np.tile(logits_const, (T_d, 1))
logits_2d = np.where(avail_d > 0, logits_2d, -1e30)
logits_2d -= logits_2d.max(axis=1, keepdims=True)
W_const = np.exp(logits_2d)
W_const = np.where(avail_d > 0, W_const, 0.0)
W_const = W_const / W_const.sum(axis=1, keepdims=True)

ET_const = (W_const * ET_mat_d).sum(axis=1)
resid_const = QpP_d - ET_const - beta_const - dS_d
rmse_const  = np.sqrt(np.mean(resid_const**2))

improv_const_vs_uni = (1 - rmse_const / rmse_uni_d) * 100
improv_adapt_vs_const = (1 - rmse_ens_d / rmse_const) * 100

_mean_qp_c = np.mean(QpP_d)
_beta_frac_c = beta_const / _mean_qp_c * 100
_beta_frac_d = beta_d / _mean_qp_c * 100

print("── Constant-weight ensemble (no conditioning variable) ──")
print(f"Parameters: {M_d} logits + 1 β_c = {M_d+1} total")
print(f"\nCalibrated constant weights:")
for i, name in enumerate(model_names_d):
    print(f"  {name:25s}  {w_const[i]:.3f}   (uniform: {1/M_d:.3f})")

print(f"\nOffset comparison:")
print(f"  β_c (constant wts)  = {beta_const:.3f} km³/month  →  β_c / mean(Qin+P) = {_beta_frac_c:.2f}%")
print(f"  β_d (adaptive DSWE) = {beta_d:.3f} km³/month  →  β_d / mean(Qin+P) = {_beta_frac_d:.2f}%")
print(f"  mean(Qin+P)         = {_mean_qp_c:.1f} km³/month")
if abs(beta_d) > 1e-9:
    print(f"  |β_c| / |β_d| = {abs(beta_const/beta_d):.1f}")

print(f"\nRMSE comparison (all {T_d} months):")
print(f"  Equal-weight uniform:          {rmse_uni_d:.4f} km³/month")
print(f"  Constant calibrated + β_c:     {rmse_const:.4f} km³/month  ({improv_const_vs_uni:+.1f}% vs uniform)")
print(f"  Adaptive (DSWE) + β_d:         {rmse_ens_d:.4f} km³/month  ({improv_adapt_vs_const:+.1f}% vs constant)")
print(f"\nInterpretation:")
print(f"  Re-weighting alone accounts for {improv_const_vs_uni:.1f}% improvement over uniform.")
print(f"  Making weights adaptive adds another {improv_adapt_vs_const:.1f}% on top.")

# ── Cumulative budget ──
cum_const = np.cumsum(QpP_d - ET_const - beta_const)

# ── Plots (4 panels, matching DSWE adaptive cell) ──
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Constant-weight ensemble — calibrated but not conditioned on DSWE",
             fontsize=13, y=0.98)

# Panel 1: weight bar chart (constant vs uniform)
ax = axes[0, 0]
x_pos = np.arange(M_d)
ax.bar(x_pos - 0.15, [1/M_d]*M_d, 0.3, label="Uniform", color="0.7", edgecolor="0.5")
ax.bar(x_pos + 0.15, w_const, 0.3, label="Calibrated", color="#2ca02c", edgecolor="0.3")
ax.set_xticks(x_pos)
ax.set_xticklabels([n.replace("_", "\n") for n in model_names_d], fontsize=7, rotation=45, ha="right")
ax.set_ylabel("Weight")
ax.set_title("Constant weights: uniform vs calibrated")
ax.legend(fontsize=8)
ax.grid(True, lw=0.2, axis="y")

# Panel 2: realized seasonal weights — constant has NO variability by design,
#           so overlay realized adaptive (IQR) for comparison
_months_arr = dates_d.month
ax = axes[0, 1]
for i, name in enumerate(model_names_d):
    # Adaptive median + IQR
    med_w, lo_w, hi_w = [], [], []
    for mo in range(1, 13):
        w_mo = W_d[_months_arr == mo, i]
        med_w.append(np.median(w_mo))
        lo_w.append(np.percentile(w_mo, 25))
        hi_w.append(np.percentile(w_mo, 75))
    months_c = np.arange(1, 13)
    ax.fill_between(months_c, lo_w, hi_w, alpha=0.12)
    ax.plot(months_c, med_w, lw=1.2, label=name)
ax.set_xlabel("Month")
ax.set_ylabel("Weight")
ax.set_xticks(np.arange(1, 13))
ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
ax.set_title("Adaptive realized weights (median ± IQR)")
ax.legend(fontsize=6, loc="best", ncol=2)
ax.grid(True, lw=0.2)

# Panel 3: cumulative budget closure (3 curves)
ax = axes[1, 0]
ax.plot(dates_d, cum_ds_d, lw=2.5, ls=":", color="#c51b7d", label="∑ΔS (GRACE)")
ax.plot(dates_d, cum_uni_d, lw=1, alpha=0.4, color="0.6",
        label=f"Uniform ({rmse_uni_d:.2f})")
ax.plot(dates_d, cum_const, lw=1.5, ls="--", color="#ff7f00",
        label=f"Constant + $β_c$ ({rmse_const:.2f})")
ax.plot(dates_d, cum_ens_d, lw=2, color="#2ca02c",
        label=f"Adaptive + $β_d$ ({rmse_ens_d:.2f})")
ax.set_ylabel("km³ (cumulative)")
ax.set_title("Cumulative budget closure")
ax.legend(fontsize=7)
ax.grid(True, lw=0.2)

# Panel 4: monthly residuals (constant vs adaptive)
ax = axes[1, 1]
ax.bar(dates_d, resid_const, width=25, color="#ff7f00", alpha=0.4,
       label=f"Constant + $β_c$")
ax.bar(dates_d, resid_ens_d, width=25, color="#2ca02c", alpha=0.7,
       label=f"Adaptive + $β_d$")
ax.axhline(0, lw=0.8, c="0.4", ls=":")
ax.set_ylabel("Residual (km³ month⁻¹)")
ax.set_title(f"Monthly budget residual\n"
             f"RMSE: Const {rmse_const:.3f} → Adaptive {rmse_ens_d:.3f} km³")
ax.legend(fontsize=8)
ax.grid(True, lw=0.2)

plt.tight_layout()
plt.show()
''')

# ---------------------------------------------------------------------------
# §14 — Darcy upper bound (old cells 56–57)
# ---------------------------------------------------------------------------
DARCY_MD = md(r'''
## 14 — Darcy upper bound on lateral subsurface efflux

Independent estimate using published Kalahari aquifer hydraulic conductivity
and the GRACE-derived head gradient between NW and NE mascon blocks:

$$Q_{\text{sub}} = K \;\frac{\Delta h}{\Delta x}\; W \; L$$

| Symbol | Meaning | Value |
|--------|---------|-------|
| $K$ | Hydraulic conductivity (range) | 0.1 – 10 m day⁻¹ |
| $\Delta h$ | TWS difference NW − NE (monthly, from GRACE) | variable |
| $\Delta x$ | Centroid-to-centroid distance between NW and NE blocks | computed |
| $W$ | Boundary width (N–S extent of shared block edge) | computed |
| $L$ | Aquifer thickness / inflow depth | 200 m |
''')

DARCY_CELL = code(r'''
# ── Darcy upper-bound on lateral subsurface efflux ──
from geopy.distance import geodesic

# Block centroids (NW_BLOCK is the delta-adjacent block west of NE_BLOCK, so the two share an edge)
nw_cy, nw_cx = NW_BLOCK["clat"], NW_BLOCK["clon"]
ne_cy, ne_cx = NE_BLOCK["clat"], NE_BLOCK["clon"]

# Centroid-to-centroid distance (Δx)
dx_km = geodesic((nw_cy, nw_cx), (ne_cy, ne_cx)).km
dx_m = dx_km * 1e3                       # (the old cell divided by 1000 again and left km)

# Shared boundary width W: N–S extent of the overlap between NW and NE blocks
shared_lat0 = max(NW_BLOCK["lat0"], NE_BLOCK["lat0"])
shared_lat1 = min(NW_BLOCK["lat1"], NE_BLOCK["lat1"])
assert shared_lat1 > shared_lat0, "NW and NE blocks do not share a N–S edge"
W_km = geodesic((shared_lat0, NW_BLOCK["lon1"]), (shared_lat1, NW_BLOCK["lon1"])).km
W_m = W_km * 1e3

# Aquifer thickness and hydraulic-conductivity range: the values stated in the
# table above (the pre-refactor cell used L = 10 000 m and K_low = 1 m/day instead).
L_m = 200.0          # m
K_low  = 0.1         # m/day
K_high = 10.0        # m/day

print(f"Centroid spacing Δx:     {dx_km:.0f} km")
print(f"Shared boundary width W: {W_km:.0f} km")
print(f"Aquifer thickness L:     {L_m:.0f} m")
print(f"K range:                 {K_low}–{K_high} m/day\n")

# Monthly TWS difference (NW − NE) in cm → convert to m for head gradient
dh_cm = df_balance["NW_NE_diff_cm"].dropna()
dh_m = dh_cm / 100.0  # cm → m

# Darcy flux:  Q = K × (Δh/Δx) × W × L   [m³/day]
# Convert to km³/month (× 30.44 days, ÷ 1e9)
days_per_month = 30.44

Q_darcy_low  = K_low  * (dh_m.abs() / dx_m) * W_m * L_m * days_per_month / 1e9
Q_darcy_high = K_high * (dh_m.abs() / dx_m) * W_m * L_m * days_per_month / 1e9

# Direction: positive dh means NW > NE → flow from NW to NE (inflow to delta)
#            negative dh means NE > NW → flow out (efflux)
efflux_mask = dh_m < 0

print("── Summary statistics (km³/month; 1e-3 km³ = 10⁶ m³) ──")
print(f"{'':30s} {'K_low':>10s} {'K_high':>10s}")
print(f"{'Max |Q_Darcy|':30s} {Q_darcy_low.max():10.2e} {Q_darcy_high.max():10.2e}")
print(f"{'Mean |Q_Darcy|':30s} {Q_darcy_low.mean():10.2e} {Q_darcy_high.mean():10.2e}")
print(f"{'Months with efflux (NE>NW)':30s} {efflux_mask.sum():>5d} / {len(dh_m)}")
if efflux_mask.any():
    print(f"{'Max efflux':30s} {Q_darcy_low[efflux_mask].max():10.2e} "
          f"{Q_darcy_high[efflux_mask].max():10.2e}")
    print(f"{'Mean efflux':30s} {Q_darcy_low[efflux_mask].mean():10.2e} "
          f"{Q_darcy_high[efflux_mask].mean():10.2e}")

# Compare with Mohembo inflow
mean_Qin = df_balance["Qin_km3"].mean()
print(f"\nMean Mohembo inflow:  {mean_Qin:.3f} km³/month")
print(f"Mean |Q_Darcy| at K_high: {Q_darcy_high.mean() * 1e3:.4f} ×10⁶ m³/month  "
      f"= {Q_darcy_high.mean() / mean_Qin * 100:.2e} % of Qin  (K={K_high} m/d, L={L_m:.0f} m)")

# ── Plot ──
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

# (a) Head difference time series
ax1.plot(dh_cm.index, dh_cm, lw=1, color="k")
ax1.axhline(0, lw=0.8, c="0.5", ls=":")
ax1.fill_between(dh_cm.index, dh_cm, 0,
                 where=dh_cm < 0, color="firebrick", alpha=0.3, label="Efflux (NE > NW)")
ax1.fill_between(dh_cm.index, dh_cm, 0,
                 where=dh_cm >= 0, color="steelblue", alpha=0.3, label="Influx (NW > NE)")
ax1.set_ylabel("TWS$_{NW}$ − TWS$_{NE}$ (cm)")
ax1.set_title("GRACE head gradient between NW and NE mascon blocks")
ax1.legend(fontsize=9)
ax1.grid(True, lw=0.2)

# (b) Darcy flux envelope vs Mohembo Q
ax2.fill_between(dh_cm.index, Q_darcy_low, Q_darcy_high,
                 color="orange", alpha=0.35, label=f"Darcy range (K={K_low}–{K_high} m/d, L={L_m:.0f} m)")
ax2.plot(dh_cm.index, Q_darcy_high, lw=0.8, color="darkorange")
ax2.plot(dh_cm.index, Q_darcy_low, lw=0.8, color="darkorange")
ax2.plot(df_balance.index, df_balance["Qin_km3"], lw=1, color="#0570b0",
         alpha=0.7, label="$Q_{in}$ (Mohembo)")
ax2.set_ylabel("km³ month⁻¹")
ax2.set_yscale("log")
ax2.set_title("Darcy lateral-flux upper bound vs Mohembo inflow (log scale)")
ax2.legend(fontsize=9)
ax2.grid(True, lw=0.2)
ax2.set_xlim(xlim)

plt.tight_layout()
plt.show()
''')


def build_aligned():
    cells = [
        TITLE_MD,
        md("## 1 — Setup"), setup_cell("grace_3block"), DELTA_CELL,
        md("## 2 — Discover GRACE mascon blocks"), DISCOVER_CELL, SELECT_CELL,
        md("## 3 — Earth Engine geometry + area"), EE_GEOM_CELL,
        md("## 4 — Monthly ET over the 3-block domain"), ET_CELL, ET_DIAG_MD, ET_DIAG_CELL,
        md("## 5 — CHIRPS precipitation"), CHIRPS_CELL,
        GRACE_MD, GRACE_CELL,
        PROXY_MD, PROXY_CELL,
        md("## 7 — Mohembo inflow"), MOHEMBO_CELL,
        BALANCE_MD, BALANCE_CELL,
        md("## 9 — Mass-balance plots"), plots_cell(True),
        md("## 10 — Per-ET-product diagnostics"), PER_MODEL_CELL, REGIME_MD, REGIME_CELL,
        TUNING_MD, *tuning_cells("NW_NE_diff_cm", "SW_NW_diff_cm", "S_NW − S_NE", "S_SW − S_NW", beta_le_0=False),
        md("## 11 — Save outputs"), save_cell(True),
        DRIVERS_MD, RAIN_PLOTS_CELL, CLIM_CELL,
        DROUGHT_MD, DROUGHT_CELL,
        AHDI_MD, AHDI_CELL,
        PREDICT_MD1, PREDICT_MD2, PREDICT_NWNE_CELL, PREDICT_SWNW_CELL, MEMORY_CELL,
        ENSEMBLE_MD, ENSEMBLE_QP_CELL, DSWE_FILL_CELL, ENSEMBLE_DSWE_CELL, CONST_WEIGHT_CELL,
        DARCY_MD, DARCY_CELL,
    ]
    write("mass_balance_grace_aligned.ipynb", cells)


if __name__ == "__main__":
    build_aligned()
