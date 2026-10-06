# GRACE mass balance of the Okavango Delta

Monthly water budget over JPL GRACE mascon blocks:

```
Qin + P − ET − ΔS = residual          (residual = Qout + G + errors)
```

| Term | Source | Module |
|---|---|---|
| Qin | GRDC monthly mean discharge at Mohembo (1357100), gap-filled ≤ 2 months | `src/discharge.py` |
| P | CHIRPS daily, summed to months (Earth Engine) | `src/precip.py` |
| ET | median of up to 8 products: MOD16A2GF, PML v2, TerraClimate, FLDAS, ERA5-Land, SSEBop, WaPOR v3 (Earth Engine) and GLEAM v4.2a (local netCDFs) | `src/et_products.py`, `src/ee_monthly.py` |
| ΔS | JPL RL06.3M mascon TWS from the local netCDF, centred difference | `src/grace_blocks.py`, `src/balance.py` |

## Notebooks

All five share the same structure and differ only in the domain. Each writes its
CSVs and figures to `figures/ET comparison/<GEOM_TAG>/` (git-ignored).

| Notebook | Domain | GEOM_TAG | Purpose |
|---|---|---|---|
| `mass_balance_grace_aligned.ipynb` | NE + SE + SW blocks, 98 % of the delta polygon | `grace_3block` | **Primary delta budget**; also the head-gradient drivers (§12), regime-adaptive ensemble ET (§13) and a Darcy bound on subsurface flux (§14) |
| `mass_balance_grace_2east.ipynb` | NE + SE blocks, 94 % | `grace_2block_east` | Delta budget without the sparsely covered SW block |
| `mass_balance_grace_1block_ne.ipynb` | NE block only, 61 % | `grace_1block_ne` | Delta core |
| `mass_balance_grace_dry_control.ipynb` | block south of the SE block (Kalahari) | `grace_dry_control` | Null test: no inflow, `P − ET ≈ ΔS` |
| `mass_balance_grace_aral_sea.ipynb` | blocks covering the Aral Sea | `grace_aral_sea` | Method sanity check on a large known storage trend |

Mascon block ids (`B018` etc.) are assigned in discovery order and are not stable
across code versions; identify blocks by their lon/lat extent, printed in each
notebook and saved to `geometry_info.txt`.

## Running

* Kernel: the notebooks are stored with the `ee-map` kernel
  (`~/anaconda3/envs/ee-map`); they also run on the base anaconda environment.
  Earth Engine must be authenticated (`src.gee_utils.ee_init`).
* `RUN_EE = False` (default) reads the Earth Engine products from the CSVs
  cached in the figure folder and computes them only if a CSV is missing.
  Set `RUN_EE = True` to force a recompute (~20 min per notebook).
* GRACE, GLEAM, Mohembo and the balance itself never need Earth Engine.
* Headless: `cd notebooks && jupyter nbconvert --to notebook --execute --inplace
  --ExecutePreprocessor.kernel_name=ee-map mass_balance_grace_<x>.ipynb`.
* Tests (no Earth Engine): `python -m pytest tests/test_mass_balance_modules.py`.
* `scripts/build_mass_balance_notebooks.py` (and `..._aligned_notebook.py`)
  regenerate the notebooks from templates. The `.ipynb` files are the source of
  truth; the builders are kept so that template-level changes can be reapplied.

## Conventions (and why)

* **GRACE comes only from the local netCDF**, labelled by the midpoint of each
  solution's `time_bounds`. The Earth Engine mascon asset's `system:index`
  dates are the day before each solution period; snapping them to months put
  ~90 % of solutions one month early and merged 11 pairs.
* **Every input is snapped to month start** (GLEAM stamps months at month end).
* **ΔS is a centred difference**, `(S[t+1] − S[t−1]) / 2`, on a continuous
  monthly grid, because a GRACE solution is the mean state of its month while
  P, ET and Q are month totals. Single missing GRACE months are linearly
  interpolated before differencing and flagged (`TWS_filled`,
  `dS_uses_filled_tws`; hollow markers in the ΔS panel). Gaps of two or more
  months, including the 11-month GRACE/GRACE-FO gap, are never bridged.
* **ET area means use valid pixels only**; the km³ total scales that mean to
  the full domain (`coverage` records the valid fraction). Months with no
  imagery are NaN, not 0. Sub-monthly composites (8-day MODIS/PML, WaPOR
  dekads) are pro-rated into calendar months by the mean daily rate of the
  overlapping images; months with under 75 % of their days covered are masked.
* **The ET median needs ≥ 4 products** (`min_et_products`). Coverage is 7–8
  products through 2024, 4 in 2025 and 3 in 2026.
* **Lateral-flux terms** (`α × TWS head difference` between adjacent blocks)
  are fitted by bounded least squares on *monthly* residuals with an
  intercept; the reported improvement is relative to that bias-only null, and
  α should keep one sign across ET products to mean anything. (Regressing
  cumulative residuals on cumulative head differences, as earlier versions
  did, mostly fitted each ET product's drift.)
* The month grid ends at the last complete calendar month.

## Known limitations

* The Mohembo record ends in February 2024, so delta-domain closure ends there.
* MOD16 is about half the other products over the delta even after masking
  (82 % coverage); it is a known low bias of the product over wetlands.
* The aligned notebook's ensemble section (§13) fits 57 parameters in-sample
  and is sensitive to the month set; treat it as exploratory. The head-gradient
  regressions (§12) are scored by leave-one-out on autocorrelated months, which
  is optimistic.
* GRDC stations 1357530 (Boro Junction) and 1357535 (Pantoon Site) carry an
  identical daily record from 1969 to 2016.

## Output files (per domain folder)

`mass_balance.csv` (monthly table: P, ET, TWS, ΔS, Qin, residual, flags),
`et_monthly.csv` / `et_monthly_delta_poly.csv` (per-product ET), `chirps_monthly.csv`,
`grace_monthly.csv`, `mohembo_monthly.csv`, `et_products_km3.csv`,
`lateral_flux_1p.csv` / `lateral_flux_2p.csv`, `geometry_info.txt`, and the PNG
figures.
