"""One-off generator for the mass-balance driver notebooks (kept for reference; the .ipynb files are the source of truth)."""
import sys
from pathlib import Path
import nbformat as nbf

OUT = Path("/Users/octaviacrompton/Projects/Okavango-water-balance/notebooks")
KERNEL = {"kernelspec": {"display_name": "Python (ee-map)", "language": "python", "name": "ee-map"},
          "language_info": {"name": "python", "version": "3.11.8"}}


def md(s):
    return nbf.v4.new_markdown_cell(s.strip("\n"))


def code(s):
    return nbf.v4.new_code_cell(s.strip("\n"))


# ---------------------------------------------------------------------------
# Shared cell templates
# ---------------------------------------------------------------------------
def setup_cell(geom_tag, extra_paths="", extra_imports=""):
    return code(f'''
import sys
from pathlib import Path
sys.path.insert(0, str(Path("..").resolve()))

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from IPython.display import IFrame, display
{extra_imports}
from src.gee_utils import ee_init
from src.grace_blocks import (discover_blocks, describe, neighbour_block, block_area_m2,
                              blocks_union, gdf_union, block_tws, domain_tws, tws_difference,
                              ee_geometry, block_map)
from src.et_products import compute_et_products, et_wide, summarize
from src.precip import compute_chirps
from src.discharge import load_mohembo_monthly
from src.balance import assemble_balance, monthly_climatology, write_outputs
from src.lateral_flux import per_model_fits, summarize_fits
from src import balance_plots as bp

# ── Paths ──
DELTA_SHP = Path("../data/regions/Delta_UCB_WGS84/Delta_UCB_WGS84.shp")
GRACE_NC  = Path("../data/grace_subsaharan_out/"
                 "GRCTellus.JPL.200204_202507.GLO.RL06.3M.MSCNv04CRI.nc")
{extra_paths}
assert DELTA_SHP.exists() and GRACE_NC.exists()

# ── Time window ──
START = "2002-04-01"                      # GRACE begins April 2002
END = (pd.Timestamp.today().normalize() + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d")

# ── Output ──
GEOM_TAG = "{geom_tag}"
FIG_DIR = Path("../figures/ET comparison") / GEOM_TAG
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Set False once the Earth Engine CSVs are cached in FIG_DIR ──
RUN_EE = True

print(f"Window : {{START}} → {{END}}")
print(f"Figures: {{FIG_DIR.resolve()}}")
''')


DELTA_CELL = code('''
gdf = gpd.read_file(DELTA_SHP).to_crs(epsg=4326)
delta_union = gdf_union(gdf)          # geopandas-version-safe union_all
centroid = delta_union.centroid
print(f"Delta bounds: {delta_union.bounds}")
''')

DISCOVER_CELL = code('''
blocks = discover_blocks(GRACE_NC, ref_geom=delta_union, pad_deg=8.0)
delta_blocks = [b for b in blocks if b["intersects_ref"]]
print(f"Total mascon blocks discovered: {len(blocks)}")
print("Blocks intersecting the delta polygon:")
for b in delta_blocks:
    print(f"  {describe(b)}  frac_delta={b['frac_ref']:.1%}")
''')

EE_GEOM_CELL = code('''
ee_init()
block_geom = ee_geometry(USE_BLOCKS)          # one polygon per block
block_union = blocks_union(USE_BLOCKS)        # shapely, for GLEAM
block_area_m2_ = sum(block_area_m2(b) for b in USE_BLOCKS)
ee_area = float(block_geom.area(maxError=1).getInfo())
print(f"Domain area: {block_area_m2_/1e6:,.0f} km² (geodesic)   EE: {ee_area/1e6:,.0f} km²")
''')

ET_CELL = code('''
df_et = compute_et_products(block_geom, block_area_m2_, START, END,
                            csv_path=FIG_DIR / "et_monthly.csv", run=RUN_EE,
                            gleam_geom=block_union)
display(summarize(df_et))
et_mm_wide = et_wide(df_et, "et_mm_mean")
bp.plot_et_products(et_mm_wide, f"Monthly ET over {DOMAIN_LABEL} (area-mean over valid pixels)",
                    save=FIG_DIR / "et_products.png")
plt.show()
''')

ET_DIAG_MD = md('''
## ET diagnostic — delta polygon vs GRACE domain

The same products over the **delta polygon**, for comparison with the mascon
domain.  Large differences mean the products respond to wetland conditions.
''')

ET_DIAG_CELL = code('''
delta_geom_ee = ee_geometry(delta_union)
delta_area_m2 = block_area_m2(delta_union)
df_et_delta = compute_et_products(delta_geom_ee, delta_area_m2, START, END,
                                  csv_path=FIG_DIR / "et_monthly_delta_poly.csv", run=RUN_EE,
                                  gleam_geom=delta_union)
bp.plot_et_two_domains(et_mm_wide, et_wide(df_et_delta, "et_mm_mean"), DOMAIN_LABEL, "Delta polygon",
                       f"ET: {DOMAIN_LABEL} vs delta polygon (mm/month)",
                       save=FIG_DIR / "et_delta_vs_grace_domain.png")
plt.show()
''')

CHIRPS_CELL = code('''
df_chirps = compute_chirps(block_geom, block_area_m2_, START, END,
                           csv_path=FIG_DIR / "chirps_monthly.csv", run=RUN_EE)
''')

GRACE_MD = md('''
## GRACE TWS over the domain

Block TWS comes from the **local JPL netCDF** (all pixels in a mascon share one
series), labelled by the midpoint of each solution's `time_bounds`.  The earlier
Earth Engine reduction labelled solutions by the day before their period and
put ~90 % of months one month early.
''')

GRACE_CELL = code('''
df_tws = domain_tws(GRACE_NC, USE_BLOCKS)
df_tws.to_csv(FIG_DIR / "grace_monthly.csv")
print(f"GRACE solutions: {len(df_tws)}  ({df_tws.index.min().date()} → {df_tws.index.max().date()})")
bp.plot_tws_series({DOMAIN_LABEL: df_tws["TWS_cm"]}, f"GRACE TWS anomaly — {DOMAIN_LABEL}")
plt.show()
''')

MOHEMBO_CELL = code('''
q_mohembo = load_mohembo_monthly(block_area_m2_)
fig, ax = plt.subplots(figsize=(13, 3.5))
ax.plot(q_mohembo["date"], q_mohembo["Qin_m3s"], lw=0.8)
ax.set_title("Mohembo monthly mean discharge (gap-filled ≤ 2 months)")
ax.set_ylabel("Q (m³/s)")
ax.grid(True, lw=0.2)
plt.tight_layout(); plt.show()
''')

BALANCE_MD = md('''
## Mass-balance assembly

All terms are placed on a continuous month-start grid.  ΔS uses a **centred
difference** `(S[t+1] − S[t−1]) / 2`, because a GRACE solution is the mean state
of its month while P, ET and Q are month totals.  A single missing GRACE month is
linearly interpolated before differencing (`fill_tws_gap_months=1`); longer
gaps, including the 11-month GRACE/GRACE-FO gap, are never bridged.  In the
ΔS panel, hollow markers show months whose ΔS depends on an interpolated
month.  ET is the median of the available products each month.
''')


def balance_cell(qin: bool, extra: str, eq: str):
    return code(f'''
bal = assemble_balance(df_et, df_chirps, df_tws, block_area_m2_,
                       qin={"q_mohembo" if qin else "None"},
                       extra={{{extra}}},
                       start=START, ds_scheme="centered",
                       fill_tws_gap_months=1)       # interpolate single missing GRACE months
df_balance = bal.df
print(bal.summary())
print("\\nResidual = {eq}")
df_balance.tail(6)
''')


def plots_cell(has_qin: bool):
    return code(f'''
bp.plot_balance_panels(bal, DOMAIN_LABEL, save=FIG_DIR / "mass_balance_panels.png"); plt.show()
bp.plot_terms(bal, f"Monthly mass-balance terms [{{GEOM_TAG}}]", save=FIG_DIR / "mass_balance_terms.png"); plt.show()
bp.plot_cumulative(bal, f"Cumulative mass balance [{{GEOM_TAG}}]", save=FIG_DIR / "cumulative_sums.png"); plt.show()
''')


PER_MODEL_CELL = code('''
bp.plot_per_model_cumulative(bal, DOMAIN_LABEL, fig_dir=FIG_DIR); plt.show()
bp.plot_monthly_divergence(bal, f"{DOMAIN_LABEL} — monthly residual by calendar month, per ET product",
                           save=FIG_DIR / "monthly_divergence_by_model.png"); plt.show()
''')

TUNING_MD = md('''
## Tunable lateral-flux terms

$$\\text{flux}_t + \\sum_k \\alpha_k\\,x_{k,t} + c \\approx \\Delta S_t$$

where $x_k$ are TWS head differences between adjacent mascon blocks (cm).  The
fit is on **monthly** residuals with an intercept $c$ that absorbs a constant
ET-product bias, and the reported improvement is relative to that bias-only
null.  (The earlier version regressed cumulative residuals on cumulative head
differences; since the head differences have a non-zero mean, that regressor
is nearly a straight line and α mostly fitted each product's drift.)  A real
lateral flux should give α of the **same sign for every ET product**.
''')


def tuning_cells(x1, x2, lab1, lab2, beta_le_0=True):
    bounds = "bounds=([-np.inf, -np.inf], [np.inf, 0])" if beta_le_0 else "bounds=None"
    c1 = code(f'''
results_1p, plot_1p = per_model_fits(bal, ["{x1}"])
summarize_fits(results_1p, "1-parameter: α × ({lab1})")
bp.plot_tuned_cumulative(results_1p, plot_1p, f"Tuned balance: flux + α·({lab1}) vs ΔS",
                         save=FIG_DIR / "tuned_alpha_cumulative.png"); plt.show()
bp.plot_alpha_sensitivity(bal, results_1p, "{x1}", title=f"Monthly RMSE vs α ({lab1})",
                          save=FIG_DIR / "rmse_vs_alpha.png"); plt.show()
''')
    c2 = code(f'''
results_2p, plot_2p = per_model_fits(bal, ["{x1}", "{x2}"], {bounds})
summarize_fits(results_2p, "2-parameter: α × ({lab1}) + β × ({lab2}){', β ≤ 0' if beta_le_0 else ''}")
bp.plot_tuned_cumulative(results_2p, plot_2p, f"Tuned balance: flux + α·({lab1}) + β·({lab2}) vs ΔS",
                         save=FIG_DIR / "tuned_2param_cumulative.png"); plt.show()
bp.plot_rmse_heatmap(bal, results_2p, "{x1}", "{x2}", title=f"Monthly RMSE surface: α ({lab1}) vs β ({lab2})",
                     save=FIG_DIR / "rmse_2d_heatmap.png"); plt.show()
''')
    return [c1, c2]


def save_cell(qin: bool):
    return code(f'''
write_outputs(bal, FIG_DIR, GEOM_TAG, USE_BLOCKS, qin={"q_mohembo" if qin else "None"})
results_1p.to_csv(FIG_DIR / "lateral_flux_1p.csv", index=False)
results_2p.to_csv(FIG_DIR / "lateral_flux_2p.csv", index=False)
''')


def write(name, cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"].update(KERNEL)
    p = OUT / name
    nbf.write(nb, p)
    print("wrote", p)


# ---------------------------------------------------------------------------
# 1 — NE block
# ---------------------------------------------------------------------------
def build_1block_ne():
    cells = [
        md('''
# Mass Balance — GRACE-aligned (NE mascon block)

Monthly water budget over the single JPL mascon block that holds most of the
Okavango Delta (NE quadrant, ~61 % of the delta polygon).

$$Q_{in} + P - ET - \\Delta S = Q_{out} + G \\;(\\text{residual})$$

All spatial reductions (ET, CHIRPS, GRACE) use the same block footprint.  The
pipeline lives in `src/` (`grace_blocks`, `et_products`, `precip`, `discharge`,
`balance`, `lateral_flux`, `balance_plots`); this notebook only chooses the
geometry and calls it.
'''),
        md("## 1 — Setup"), setup_cell("grace_1block_ne"), DELTA_CELL,
        md("## 2 — Discover GRACE mascon blocks"), DISCOVER_CELL,
        code('''
USE_BLOCKS = [b for b in delta_blocks if b["quadrant"] == "NE"]
NE_BLOCK = USE_BLOCKS[0]
DOMAIN_LABEL = "NE block"
NW_BLOCK = neighbour_block(NE_BLOCK, blocks, "W")     # delta-adjacent NW block
SE_BLOCK = neighbour_block(NE_BLOCK, blocks, "S")
print("Domain :", describe(NE_BLOCK))
print("West   :", describe(NW_BLOCK))
print("South  :", describe(SE_BLOCK))
m = block_map(blocks, {"#1b9e77": USE_BLOCKS, "#d95f02": [NW_BLOCK, SE_BLOCK]}, ref_gdf=gdf,
              center=(centroid.y, centroid.x), zoom_start=7, ref_name="Delta polygon")
map_path = FIG_DIR / "study_area_map.html"; m.save(str(map_path))
IFrame(src=str(map_path), width=800, height=500)
'''),
        md("## 3 — Earth Engine geometry + area"), EE_GEOM_CELL,
        md("## 4 — Monthly ET over the NE block"), ET_CELL, ET_DIAG_MD, ET_DIAG_CELL,
        md("## 5 — CHIRPS precipitation"), CHIRPS_CELL,
        GRACE_MD, GRACE_CELL,
        md("### Adjacent-block TWS differences (lateral-flux proxies)"),
        code('''
df_nw_ne = tws_difference(GRACE_NC, NW_BLOCK, NE_BLOCK, "NW_NE_diff_cm")
df_ne_se = tws_difference(GRACE_NC, NE_BLOCK, SE_BLOCK, "NE_SE_diff_cm")
bp.plot_tws_pair(df_nw_ne, df_nw_ne.columns[0], df_nw_ne.columns[1], "NW_NE_diff_cm", "NW", "NE",
                 "GRACE TWS: NW and NE blocks"); plt.show()
bp.plot_tws_pair(df_ne_se, df_ne_se.columns[0], df_ne_se.columns[1], "NE_SE_diff_cm", "NE", "SE",
                 "GRACE TWS: NE and SE blocks"); plt.show()
'''),
        md("## 7 — Mohembo inflow"), MOHEMBO_CELL,
        BALANCE_MD,
        balance_cell(True, '"NW_NE_diff_cm": df_nw_ne["NW_NE_diff_cm"], "NE_SE_diff_cm": df_ne_se["NE_SE_diff_cm"]',
                     "Qin + P − ET − ΔS  (= Qout + G)"),
        md("## 9 — Mass-balance plots"), plots_cell(True),
        md("## 10 — Per-ET-product diagnostics"), PER_MODEL_CELL,
        TUNING_MD, *tuning_cells("NW_NE_diff_cm", "NE_SE_diff_cm", "S_NW − S_NE", "S_NE − S_SE"),
        md("## 11 — Save outputs"), save_cell(True),
    ]
    write("mass_balance_grace_1block_ne.ipynb", cells)


# ---------------------------------------------------------------------------
# 2 — two east blocks
# ---------------------------------------------------------------------------
def build_2east():
    cells = [
        md('''
# Mass Balance — GRACE-aligned (2 east mascon blocks)

Monthly water budget over the **NE + SE** mascon blocks (~94 % of the delta
polygon).  The NW and SW blocks (1.6 % and 4.7 % of the delta) are excluded
from the domain but used as lateral-flux proxies.

$$Q_{in} + P - ET - \\Delta S = Q_{out} + G \\;(\\text{residual})$$

Pipeline: `src/` modules; see the NE-block notebook for the same structure.
'''),
        md("## 1 — Setup"), setup_cell("grace_2block_east"), DELTA_CELL,
        md("## 2 — Discover GRACE mascon blocks"), DISCOVER_CELL,
        code('''
USE_BLOCKS = [b for b in delta_blocks if b["quadrant"].endswith("E")]
NE_BLOCK = [b for b in USE_BLOCKS if b["quadrant"] == "NE"][0]
SE_BLOCK = [b for b in USE_BLOCKS if b["quadrant"] == "SE"][0]
DOMAIN_LABEL = "NE+SE blocks"
NW_BLOCK = neighbour_block(NE_BLOCK, blocks, "W")
SW_BLOCK = neighbour_block(SE_BLOCK, blocks, "W")
for lab, b in [("NE", NE_BLOCK), ("SE", SE_BLOCK), ("NW", NW_BLOCK), ("SW", SW_BLOCK)]:
    print(f"{lab}: {describe(b)}")
aligned = abs(NE_BLOCK["lon0"] - SE_BLOCK["lon0"]) < 1e-6 and abs(NE_BLOCK["lon1"] - SE_BLOCK["lon1"]) < 1e-6
print("NE and SE share the same longitude extent:", aligned)
m = block_map(blocks, {"#1b9e77": USE_BLOCKS, "#d95f02": [NW_BLOCK, SW_BLOCK]}, ref_gdf=gdf,
              center=(centroid.y, centroid.x), zoom_start=7, ref_name="Delta polygon")
map_path = FIG_DIR / "study_area_map.html"; m.save(str(map_path))
IFrame(src=str(map_path), width=800, height=500)
'''),
        md("## 3 — Earth Engine geometry + area"), EE_GEOM_CELL,
        md("## 4 — Monthly ET over the 2-block domain"), ET_CELL, ET_DIAG_MD, ET_DIAG_CELL,
        md("## 5 — CHIRPS precipitation"), CHIRPS_CELL,
        GRACE_MD, GRACE_CELL,
        md("### Adjacent-block TWS differences (lateral-flux proxies)"),
        code('''
df_nw_ne = tws_difference(GRACE_NC, NW_BLOCK, NE_BLOCK, "NW_NE_diff_cm")
df_sw_se = tws_difference(GRACE_NC, SW_BLOCK, SE_BLOCK, "SW_SE_diff_cm")
bp.plot_tws_pair(df_nw_ne, df_nw_ne.columns[0], df_nw_ne.columns[1], "NW_NE_diff_cm", "NW", "NE",
                 "GRACE TWS: NW and NE blocks"); plt.show()
bp.plot_tws_pair(df_sw_se, df_sw_se.columns[0], df_sw_se.columns[1], "SW_SE_diff_cm", "SW", "SE",
                 "GRACE TWS: SW and SE blocks"); plt.show()
'''),
        md("## 7 — Mohembo inflow"), MOHEMBO_CELL,
        BALANCE_MD,
        balance_cell(True, '"NW_NE_diff_cm": df_nw_ne["NW_NE_diff_cm"], "SW_SE_diff_cm": df_sw_se["SW_SE_diff_cm"]',
                     "Qin + P − ET − ΔS  (= Qout + G)"),
        md("## 9 — Mass-balance plots"), plots_cell(True),
        md("## 10 — Per-ET-product diagnostics"), PER_MODEL_CELL,
        TUNING_MD, *tuning_cells("NW_NE_diff_cm", "SW_SE_diff_cm", "S_NW − S_NE", "S_SW − S_SE", beta_le_0=False),
        md("## 11 — Save outputs"), save_cell(True),
    ]
    write("mass_balance_grace_2east.ipynb", cells)


# ---------------------------------------------------------------------------
# 3 — dry control (merges the former dry_pixel notebook: same block)
# ---------------------------------------------------------------------------
def build_dry_control():
    cells = [
        md('''
# Mass Balance — Dry Control Cell (south of the SE delta block)

Water budget for the single mascon block immediately **south of the SE delta
block**, deep in the Kalahari.  It is the null / control case: with no river
inflow and (ideally) no lateral exchange,

$$P - ET \\approx \\Delta S ,$$

so the residual $P - ET - \\Delta S$ should be ~0 with no seasonal signature.
(This notebook supersedes `mass_balance_grace_dry_pixel.ipynb`, which analysed
the identical block.)  Pipeline: `src/` modules.
'''),
        md("## 1 — Setup"), setup_cell("grace_dry_control"), DELTA_CELL,
        md("## 2 — Discover GRACE mascon blocks & select the dry control cell"), DISCOVER_CELL,
        code('''
SE_BLOCK = [b for b in delta_blocks if b["quadrant"] == "SE"][0]
NE_BLOCK = [b for b in delta_blocks if b["quadrant"] == "NE"][0]
DRY_BLOCK = neighbour_block(SE_BLOCK, blocks, "S")
SOUTH_BLOCK = neighbour_block(DRY_BLOCK, blocks, "S")
USE_BLOCKS = [DRY_BLOCK]
DOMAIN_LABEL = "Dry control cell"
print("SE (delta)  :", describe(SE_BLOCK))
print("Dry control :", describe(DRY_BLOCK), f" intersects delta: {DRY_BLOCK['intersects_ref']}")
print("South       :", describe(SOUTH_BLOCK) if SOUTH_BLOCK else None)
m = block_map(blocks, {"#d62728": USE_BLOCKS, "#1b9e77": [SE_BLOCK], "#d95f02": [SOUTH_BLOCK]}, ref_gdf=gdf,
              center=(centroid.y - 2, centroid.x), zoom_start=6, ref_name="Delta polygon")
map_path = FIG_DIR / "study_area_map_dry_control.html"; m.save(str(map_path))
IFrame(src=str(map_path), width=800, height=500)
'''),
        md("## 3 — Earth Engine geometry + area"), EE_GEOM_CELL,
        md("## 4 — Monthly ET over the dry control block"), ET_CELL, ET_DIAG_MD, ET_DIAG_CELL,
        md("## 5 — CHIRPS precipitation"), CHIRPS_CELL,
        GRACE_MD, GRACE_CELL,
        md("### Adjacent-block TWS differences (subsurface-flow proxies)"),
        code('''
df_n_dc = tws_difference(GRACE_NC, SE_BLOCK, DRY_BLOCK, "N_DC_diff_cm")
df_dc_s = tws_difference(GRACE_NC, DRY_BLOCK, SOUTH_BLOCK, "DC_S_diff_cm")
bp.plot_tws_pair(df_n_dc, df_n_dc.columns[0], df_n_dc.columns[1], "N_DC_diff_cm", "North (SE delta)", "Dry control",
                 "GRACE TWS: SE delta block and dry control cell"); plt.show()
bp.plot_tws_pair(df_dc_s, df_dc_s.columns[0], df_dc_s.columns[1], "DC_S_diff_cm", "Dry control", "South",
                 "GRACE TWS: dry control cell and south block"); plt.show()
'''),
        BALANCE_MD,
        balance_cell(False, '"N_DC_diff_cm": df_n_dc["N_DC_diff_cm"], "DC_S_diff_cm": df_dc_s["DC_S_diff_cm"]',
                     "P − ET − ΔS  (should be ≈ 0 for a closed cell)"),
        md("## 7 — Mass-balance plots"), plots_cell(False),
        md("## 7a — Per-ET-product diagnostics"), PER_MODEL_CELL,
        TUNING_MD, *tuning_cells("N_DC_diff_cm", "DC_S_diff_cm", "S_N − S_DC", "S_DC − S_S"),
        md("## 7f — Save outputs"), save_cell(False),
        md("## 8 — Compare with the delta blocks (TWS)"),
        code('''
tws_cmp = {"NE (delta core)": block_tws(GRACE_NC, NE_BLOCK),
           "SE (downstream delta)": block_tws(GRACE_NC, SE_BLOCK),
           "Dry control": block_tws(GRACE_NC, DRY_BLOCK)}
bp.plot_tws_series(tws_cmp, "GRACE TWS: delta blocks vs dry control cell", save=FIG_DIR / "tws_vs_delta_blocks.png")
plt.show()
for k, s in tws_cmp.items():
    print(f"{k:24s} std = {s.std():.2f} cm")
print(f"Dry control has {tws_cmp['Dry control'].std() / tws_cmp['NE (delta core)'].std():.0%} of the NE block's TWS variability")
'''),
        md("## 9 — Seasonality"),
        code('''
clim = monthly_climatology(bal.closure(), {"P_mm": "P", "ET_mm": "ET", "dS_mm": "ΔS", "resid_mm": "resid"})
bp.plot_climatology(clim, ["P", "ET", "ΔS", "resid"], ["steelblue", "#e6550d", "0.3", "grey"],
                    "Dry control cell — monthly climatology (closure months)", save=FIG_DIR / "climatology.png")
plt.show()
print(clim[[c for c in clim.columns if c.endswith("_mean")]].round(1).to_string())
'''),
        md("## 10 — Summary"),
        code('''
d = bal.closure()
print("═" * 60); print("DRY CONTROL CELL SUMMARY"); print("═" * 60)
print(f"Block: {describe(DRY_BLOCK)}   area {block_area_m2_/1e6:,.0f} km²")
print(f"Closure months: {len(d)} of {len(bal.df)}")
print("Mean annual fluxes (closure months):")
for c, lab in [("P_mm", "P"), ("ET_mm", "ET"), ("dS_mm", "ΔS"), ("resid_mm", "P − ET − ΔS")]:
    print(f"  {lab:12s} {d[c].mean() * 12:+7.0f} mm/yr")
print(f"P/ET = {d['P_mm'].mean() / d['ET_mm'].mean():.2f}  (≈ 1 expected for a closed dry cell)")
'''),
    ]
    write("mass_balance_grace_dry_control.ipynb", cells)


# ---------------------------------------------------------------------------
# 4 — Aral Sea
# ---------------------------------------------------------------------------
def build_aral():
    cells = [
        md('''
# Mass Balance — Aral Sea (GRACE mascon)

Sanity check of the method on a region with a large, well-known storage
trend.  The residual $P - ET - \\Delta S$ is the **net outflow** (river inflow
minus lake/irrigation loss is what ΔS already integrates, so a negative
residual means P − ET is smaller than the observed storage change and vice
versa).  Only blocks that overlap the historical lake bounding box by at least
`MIN_OVERLAP` are used, so the domain is not dominated by desert far from
the lake.  Pipeline: `src/` modules (WaPOR is Africa-only and is skipped).
'''),
        md("## 1 — Setup"),
        code('''
import sys
from pathlib import Path
sys.path.insert(0, str(Path("..").resolve()))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import IFrame, display
from shapely.geometry import box as shapely_box

from src.gee_utils import ee_init
from src.grace_blocks import (discover_blocks, describe, block_area_m2, blocks_union,
                              block_tws, domain_tws, ee_geometry, block_map)
from src.et_products import compute_et_products, et_wide, summarize
from src.precip import compute_chirps
from src.balance import assemble_balance, monthly_climatology, write_outputs
from src import balance_plots as bp

GRACE_NC = Path("../data/grace_subsaharan_out/GRCTellus.JPL.200204_202507.GLO.RL06.3M.MSCNv04CRI.nc")
assert GRACE_NC.exists()

# Historical (pre-1960) Aral Sea extent
ARAL_LAT_MIN, ARAL_LAT_MAX = 43.0, 46.8
ARAL_LON_MIN, ARAL_LON_MAX = 58.2, 61.5
aral_poly = shapely_box(ARAL_LON_MIN, ARAL_LAT_MIN, ARAL_LON_MAX, ARAL_LAT_MAX)
MIN_OVERLAP = 0.10          # keep blocks covering ≥ 10 % of the bbox

START = "2002-04-01"
END = (pd.Timestamp.today().normalize() + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d")
GEOM_TAG = "grace_aral_sea"
FIG_DIR = Path("../figures/ET comparison") / GEOM_TAG
FIG_DIR.mkdir(parents=True, exist_ok=True)
RUN_EE = True
DOMAIN_LABEL = "Aral Sea blocks"
print(f"Window : {START} → {END}")
'''),
        md("## 2 — Discover GRACE mascon blocks covering the Aral Sea"),
        code('''
blocks = discover_blocks(GRACE_NC, ref_geom=aral_poly, pad_deg=6.0)
print(f"Discovered {len(blocks)} blocks; those intersecting the Aral bbox:")
for b in [b for b in blocks if b["intersects_ref"]]:
    print(f"  {describe(b)}  frac_bbox={b['frac_ref']:.1%}")
USE_BLOCKS = [b for b in blocks if b["frac_ref"] >= MIN_OVERLAP]
print(f"\\nSelected (≥ {MIN_OVERLAP:.0%} overlap): {[b['block_id'] for b in USE_BLOCKS]}  "
      f"covering {sum(b['frac_ref'] for b in USE_BLOCKS):.0%} of the bbox")
m = block_map(blocks, {"#1b9e77": USE_BLOCKS}, center=((ARAL_LAT_MIN + ARAL_LAT_MAX) / 2, (ARAL_LON_MIN + ARAL_LON_MAX) / 2),
              zoom_start=6)
import folium
folium.Rectangle(bounds=[[ARAL_LAT_MIN, ARAL_LON_MIN], [ARAL_LAT_MAX, ARAL_LON_MAX]], color="#8c2d04", weight=3,
                 fill=False, tooltip="Aral Sea bbox (historical)").add_to(m)
map_path = FIG_DIR / "aral_sea_grace_map.html"; m.save(str(map_path))
IFrame(src=str(map_path), width=800, height=500)
'''),
        md("## 3 — Earth Engine geometry + area"), EE_GEOM_CELL,
        md("## 4 — Monthly ET"),
        code('''
df_et = compute_et_products(block_geom, block_area_m2_, START, END,
                            csv_path=FIG_DIR / "et_monthly.csv", run=RUN_EE,
                            gleam_geom=block_union, africa=False)
display(summarize(df_et))
et_mm_wide = et_wide(df_et, "et_mm_mean")
bp.plot_et_products(et_mm_wide, f"Monthly ET over {DOMAIN_LABEL} (area-mean over valid pixels)",
                    save=FIG_DIR / "et_products.png"); plt.show()
'''),
        md("## 5 — CHIRPS precipitation"), CHIRPS_CELL,
        GRACE_MD, GRACE_CELL,
        code('''
x_yr = (df_tws.index - df_tws.index[0]).days / 365.25
slope = np.polyfit(x_yr, df_tws["TWS_cm"], 1)[0]
print(f"Linear TWS trend: {slope:+.2f} cm/yr  ({slope / 100 * block_area_m2_ / 1e9:+.2f} km³/yr)")
'''),
        BALANCE_MD,
        balance_cell(False, "", "P − ET − ΔS  (net outflow; negative ⇒ storage falls faster than P − ET explains)"),
        md("## 8 — Plots"), plots_cell(False),
        md("## 9 — Seasonality and summary"),
        code('''
clim = monthly_climatology(bal.closure(), {"P_mm": "P", "ET_mm": "ET", "dS_mm": "ΔS", "resid_mm": "resid"})
bp.plot_climatology(clim, ["P", "ET", "ΔS", "resid"], ["steelblue", "#e6550d", "0.3", "grey"],
                    "Aral Sea blocks — monthly climatology (closure months)", save=FIG_DIR / "climatology.png"); plt.show()
d = bal.closure()
print(f"Closure months: {len(d)}")
for c, lab in [("P_mm", "P"), ("ET_mm", "ET"), ("dS_mm", "ΔS"), ("resid_mm", "P − ET − ΔS")]:
    print(f"  {lab:12s} {d[c].mean() * 12:+7.0f} mm/yr")
print(f"Cumulative ΔS over closure months: {d['dS_km3'].sum():+.1f} km³")
write_outputs(bal, FIG_DIR, GEOM_TAG, USE_BLOCKS)
'''),
    ]
    write("mass_balance_grace_aral_sea.ipynb", cells)


if __name__ == "__main__":
    which = sys.argv[1:] or ["ne", "2east", "dry", "aral"]
    {"ne": build_1block_ne, "2east": build_2east, "dry": build_dry_control, "aral": build_aral}
    for w in which:
        {"ne": build_1block_ne, "2east": build_2east, "dry": build_dry_control, "aral": build_aral}[w]()
