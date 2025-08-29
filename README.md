# Okavango Flood Comparison — Inman–Lyons (IL) vs DSWE

Quick start guide for reproducing the **Okavango Delta** flood-extent comparison using **Inman–Lyons annual inundation maps** and **USGS DSWE** (monthly 30 m water masks).

---

## What this does
- Builds a **DSWE ImageCollection** from your monthly assets (adds `dswe` + `dswe_bin` bands).
- Loads **Inman–Lyons** yearly inundation products and harmonizes time windows.
- Compares **seasonal means**, **minima (p10)**, **variability (std, CV)**, and **period change** (late − early).
- Exports styled PNG/GeoTIFFs and previews layers in the GEE map.

---

## Data prerequisites (Earth Engine assets)

Make sure these exist (and are shared to anyone who will run the script):

- DSWE monthly masks (30 m):
  - `projects/ee-okavango/assets/water_masks/monthly_DSWE_Landsat_30m/*`
- Inman–Lyons inundation:
  - `projects/ee-okavango/assets/Inman_Lyons/FloodArea/*`
  - `projects/ee-okavango/assets/Inman_Lyons/Annual_inundation_maps/*`
- Optional boundaries:
  - `projects/ee-okavango/assets/shapes/1274m_contour` (highlands)
  - `projects/ee-okavango/assets/shapes/rainfall_regions` (zones)

> Tip: Assets under `projects/...` are **private by default**. Flip ACLs to public or share to your team.

---

## Quick start (GEE Code Editor)

1. Open the script in the **Code Editor**.
2. Check the **AOI** (Okavango polygon). Adjust if needed.
3. Set top-level parameters:
   ```js
   var startDate = '2003-01-01';
   var endDate   = '2025-01-01';
   var year0     = '2007';      // reference year for IL vs DSWE single-year compare
