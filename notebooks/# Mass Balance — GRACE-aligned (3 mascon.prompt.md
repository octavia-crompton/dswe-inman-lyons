# Mass Balance — GRACE-aligned (3 mascon blocks)

Okavango Delta water budget with **P**, **ET**, **Q** and **GRACE ΔS** all
computed over the same three JPL mascon blocks (NE + SE + SW) that overlap the
delta polygon. Dropping the NW block (1.6% of the delta) eliminates a large
dry-Kalahari footprint while retaining 98.4% of the delta polygon coverage.

$$Q_{\text{in}} + P - ET \approx \Delta S + Q_{\text{out}} + G$$

## Selected mascon blocks

| Block | Position | Size | % of delta covered | % of block filled by delta |
|-------|----------|------|--------------------|---------------------------|
| B011 | **NE** | 3.5°×3.0° | **60.7%** | 11.6% |
| B007 | **SE** | 3.0°×3.0° | **33.0%** | 7.4% |
| B006 | SW | 3.0°×3.0° | 4.7% | 1.1% |

Dropped: B010 (NW) — 1.6% of delta, 0.3% of block.

## Notebook structure

| Section | Cells | Purpose |
|---------|-------|---------|
| **1 — Setup** | 3–4 | Imports, config, load delta shapefile |
| **2 — Mascon discovery** | 6–8 | Hash GRACE netCDF → discover blocks, select NE+SE+SW, folium map |
| **3 — EE geometry** | 10 | Build EE `MultiPolygon` from the 3 block boundaries |
| **4 — Monthly ET** | 12–14 | 7 ET products over 3-block union (with `RUN_EE` cache toggle) |
| **5 — CHIRPS P** | 16 | Monthly precip over 3-block union |
| **6 — GRACE TWS** | 18 | Monthly LWE anomaly over 3-block union |
| **7 — Mohembo Q** | 20 | Load discharge CSV, convert m³/s → km³/month (normalized to block area) |
| **8 — Assembly** | 22 | Combine all terms, compute ΔS, residual Qout+G |
| **9 — Plots** | 24 | Mass-balance terms (3-mo smoothed) + cumulative sums |
| **10 — Per-model** | 26 | Cumulative plot for each individual ET product |
| **11 — Save** | 28 | CSVs + geometry metadata to `figures/ET comparison/grace_3block/` |

## Key design decisions

- **Geometry**: 3 mascon blocks (NE+SE+SW), dropping NW — covers 98.4% of the delta
- **All terms consistent**: P, ET, GRACE all computed over the same 3-block footprint
- **`RUN_EE = True/False` toggle**: Set to `False` after the first run to skip EE calls and load from cached CSVs
- **Mohembo Q**: Point measurement at delta inflow — stays as-is; volume normalized to block area for mm comparison
- **SSEBop cleanup**: Zero-padded trailing months blanked to NaN
- **GRACE ΔS gaps**: Months with >45-day gaps set to NaN

## ET products

1. MOD16A2GF v6.1 (2000–)
2. PML v2 (2000–)
3. TerraClimate aet (1958–)
4. FLDAS Evap (1982–)
5. ERA5-Land totalET (1950–)
6. USGS SSEBop MODIS monthly (2003–)
7. WaPOR v3 AETI dekadal (2018–)

## Output files (`figures/ET comparison/grace_3block/`)

- `et_monthly.csv` — all ET products, monthly
- `chirps_monthly.csv` — CHIRPS precipitation
- `grace_monthly.csv` — GRACE TWS anomaly
- `mohembo_monthly.csv` — Mohembo discharge
- `mass_balance.csv` — assembled balance table
- `geometry_info.txt` — block coordinates and area
- `mass_balance_terms.png` — 3-mo smoothed terms plot
- `cumulative_sums.png` — cumulative ∑ΔS vs ∑(Qin+P−ET)
- `single-model/*/cumulative_sums.png` — per-ET-product cumulative plots
