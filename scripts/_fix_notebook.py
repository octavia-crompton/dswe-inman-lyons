#!/usr/bin/env python3
"""One-shot script to apply structural fixes 1-4 to the ET comparison notebook."""
import json

NB_PATH = "notebooks/ET comparison over the delta_2026-03-06.ipynb"

with open(NB_PATH) as f:
    nb = json.load(f)

cells = nb["cells"]
print(f"Before: {len(cells)} cells")

# ── 1) DELETE cells by id ──────────────────────────────
delete_ids = {
    "fb547112",   # cell 15: commented-out plot code
    "089bdc2b",   # cell 21: duplicate df_master build
    "aacdd851",   # cell 22: duplicate GRACE plot via df_master
    "bf0149fb",   # cell 23: empty
    "c1955f85",   # cell 30: empty
    "a3939f22",   # cell 35: empty
}

nb["cells"] = [c for c in cells if c.get("id") not in delete_ids]
cells = nb["cells"]
print(f"After deleting {len(delete_ids)} cells: {len(cells)} cells")

# ── 2) Make SSEBop cell idempotent (id=68f9685e) ──────
ssebop_idx = next(i for i, c in enumerate(cells) if c.get("id") == "68f9685e")
print(f"SSEBop cell at index {ssebop_idx}")

NEW_SSEBOP = [
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    'ssebop = "USGS_SSEBop_MODIS_monthly"\n',
    "tiny = 1e-6  # km\u00b3 or mm threshold for \"effectively zero\"\n",
    "\n",
    "# \u2500\u2500 Idempotent: only touch SSEBop rows if they still contain zero-padded values\n",
    'm = df_all["dataset"].eq(ssebop)\n',
    "\n",
    "if m.any():\n",
    "    # Count non-NaN zeros *before* we touch anything\n",
    '    zeros_km3 = (df_all.loc[m, "et_km3_total"].le(tiny) & df_all.loc[m, "et_km3_total"].notna()).sum()\n',
    "\n",
    "    if zeros_km3 > 0:\n",
    "        # Find last month with a non-trivial value\n",
    '        last_valid = df_all.loc[m & (df_all["et_km3_total"] > tiny), "date"].max()\n',
    '        print(f"SSEBop last nonzero month: {last_valid}  ({zeros_km3} zero-padded rows to blank)")\n',
    "\n",
    "        # Convert zeros/tiny values to NaN\n",
    '        df_all.loc[m & df_all["et_km3_total"].le(tiny), "et_km3_total"] = np.nan\n',
    '        df_all.loc[m & df_all["et_mm_mean"].le(tiny),   "et_mm_mean"]   = np.nan\n',
    "\n",
    "        # Blank everything after the last real value\n",
    "        if pd.notna(last_valid):\n",
    '            df_all.loc[m & (df_all["date"] > last_valid), ["et_km3_total", "et_mm_mean"]] = np.nan\n',
    "    else:\n",
    '        print("SSEBop: already cleaned (no zero-padded rows to fix)")\n',
    "else:\n",
    '    print("SSEBop not found in df_all \\u2014 skipping cleanup")\n',
    "\n",
    "# \u2500\u2500 Build df_combined (ET + Precip wide) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
    "df_et = df_all.copy()\n",
    'df_et["date"] = pd.to_datetime(df_et["date"])\n',
    "\n",
    "df_p = df_chirps_monthly.copy()\n",
    'df_p["date"] = pd.to_datetime(df_p["date"])\n',
    "\n",
    'et_mm_wide  = df_et.pivot_table(index="date", columns="dataset", values="et_mm_mean",  aggfunc="median")\n',
    'et_km3_wide = df_et.pivot_table(index="date", columns="dataset", values="et_km3_total", aggfunc="median")\n',
    "\n",
    'p_mm_wide  = df_p.pivot_table(index="date", columns="dataset", values="ppt_mm_mean",  aggfunc="median")\n',
    'p_km3_wide = df_p.pivot_table(index="date", columns="dataset", values="ppt_km3_total", aggfunc="median")\n',
    "\n",
    'et_mm_wide.columns  = [f"ETmm_{c}" for c in et_mm_wide.columns]\n',
    'et_km3_wide.columns = [f"ETkm3_{c}" for c in et_km3_wide.columns]\n',
    'p_mm_wide.columns   = [f"Pmm_{c}" for c in p_mm_wide.columns]\n',
    'p_km3_wide.columns  = [f"Pkm3_{c}" for c in p_km3_wide.columns]\n',
    "\n",
    "df_combined = pd.concat([et_mm_wide, et_km3_wide, p_mm_wide, p_km3_wide], axis=1).sort_index()\n",
    "df_combined.reset_index(inplace=True)\n",
    'df_combined.rename(columns={"index": "date"}, inplace=True)\n',
]

cells[ssebop_idx]["source"] = NEW_SSEBOP
cells[ssebop_idx]["outputs"] = []
cells[ssebop_idx]["execution_count"] = None

# ── 3) Insert to_month_start() utility cell before Mohembo ────
mohembo_idx = next(i for i, c in enumerate(cells) if c.get("id") == "3d78ec3b")
print(f"Mohembo cell at index {mohembo_idx}")

helper_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "a0b0c0d0",
    "metadata": {},
    "outputs": [],
    "source": [
        "# \u2500\u2500 Utility: normalize dates to month-start for all monthly merges \u2500\u2500\n",
        "import pandas as pd\n",
        "\n",
        "def to_month_start(x):\n",
        '    """Normalize datetimes to month-start (avoids GRACE vs ET/CHIRPS merge mismatch)."""\n',
        "    x = pd.to_datetime(x)\n",
        '    return x.dt.to_period("M").dt.to_timestamp(how="start")\n',
    ]
}
cells.insert(mohembo_idx, helper_cell)
print(f"Inserted to_month_start helper at index {mohembo_idx}")

# Re-find balance cell after insertion
balance_idx = next(i for i, c in enumerate(cells) if c.get("id") == "859e13ab")
print(f"df_balance cell at index {balance_idx}")

# ── 4) Rewrite df_balance cell: remove to_month_start def, add GRACE plot ────
NEW_BALANCE = [
    "\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "# --- normalize all dates to month-start (CRITICAL) ---\n",
    "df_all = df_all.copy()\n",
    "df_chirps_monthly = df_chirps_monthly.copy()\n",
    "df_grace = df_grace.copy()\n",
    "q_mohembo = q_mohembo.copy()\n",
    "\n",
    'df_all["date"] = to_month_start(df_all["date"])\n',
    'df_chirps_monthly["date"] = to_month_start(df_chirps_monthly["date"])\n',
    'df_grace["date"] = to_month_start(df_grace["date"])\n',
    'q_mohembo["date"] = to_month_start(q_mohembo["date"])\n',
    "\n",
    "# If any duplicates within a month, collapse\n",
    'df_all = df_all.groupby(["date", "dataset"], as_index=False).mean(numeric_only=True)\n',
    'df_chirps_monthly = df_chirps_monthly.groupby(["date", "dataset"], as_index=False).mean(numeric_only=True)\n',
    'df_grace = df_grace.groupby("date", as_index=False).mean(numeric_only=True)\n',
    'q_mohembo = q_mohembo.groupby("date", as_index=False).mean(numeric_only=True)\n',
    "\n",
    "# -----------------------\n",
    "# 1) ET ensemble median\n",
    "# -----------------------\n",
    'et_mm_wide  = df_all.pivot_table(index="date", columns="dataset", values="et_mm_mean",  aggfunc="median").sort_index()\n',
    'et_km3_wide = df_all.pivot_table(index="date", columns="dataset", values="et_km3_total", aggfunc="median").sort_index()\n',
    "\n",
    'ETmm_med  = et_mm_wide.median(axis=1, skipna=True).rename("ETmm")\n',
    'ETkm3_med = et_km3_wide.median(axis=1, skipna=True).rename("ETkm3")\n',
    "\n",
    "# -----------------------\n",
    "# 2) CHIRPS precip\n",
    "# -----------------------\n",
    'p_mm_wide  = df_chirps_monthly.pivot_table(index="date", columns="dataset", values="ppt_mm_mean",  aggfunc="median").sort_index()\n',
    'p_km3_wide = df_chirps_monthly.pivot_table(index="date", columns="dataset", values="ppt_km3_total", aggfunc="median").sort_index()\n',
    "\n",
    "if p_mm_wide.shape[1] == 0 or p_km3_wide.shape[1] == 0:\n",
    '    raise ValueError("No precip columns found in df_chirps_monthly after pivot. Check \'dataset\' values/bands.")\n',
    "\n",
    "pmm_col  = next((c for c in p_mm_wide.columns  if \"CHIRPS\" in str(c).upper()),  p_mm_wide.columns[0])\n",
    "pkm3_col = next((c for c in p_km3_wide.columns if \"CHIRPS\" in str(c).upper()), p_km3_wide.columns[0])\n",
    "\n",
    'Pmm  = p_mm_wide[pmm_col].rename("Pmm")\n',
    'Pkm3 = p_km3_wide[pkm3_col].rename("Pkm3")\n',
    "\n",
    "# -----------------------\n",
    "# 3) GRACE storage anomaly + monthly change (with gap checks)\n",
    "# -----------------------\n",
    'gr = (df_grace.set_index("date")[["grace_cm_mean", "grace_km3_total"]]\n',
    "      .sort_index()\n",
    '      .rename(columns={"grace_cm_mean": "GRACEcm", "grace_km3_total": "GRACEkm3"}))\n',
    "\n",
    "# sanity: GRACE should now be month-start\n",
    "bad_grace_dates = gr.index[gr.index.day != 1]\n",
    "if len(bad_grace_dates) > 0:\n",
    '    raise ValueError(f"GRACE dates not normalized to month-start (day!=1). Example: {bad_grace_dates[:5].tolist()}")\n',
    "\n",
    'dS_km3 = gr["GRACEkm3"].diff().rename("dS_km3")\n',
    'dS_cm  = gr["GRACEcm"].diff().rename("dS_cm")\n',
    "\n",
    "gap_days = gr.index.to_series().diff().dt.days\n",
    "if (gap_days > 45).any():\n",
    "    n_gaps = int((gap_days > 45).sum())\n",
    '    print(f"Warning: {n_gaps} GRACE gaps >45 days; setting dS to NaN across those steps.")\n',
    "\n",
    "dS_km3[gap_days > 45] = np.nan\n",
    "dS_cm[gap_days > 45]  = np.nan\n",
    "\n",
    "# -----------------------\n",
    "# 4) Qin (Mohembo) + mass-balance terms\n",
    "# -----------------------\n",
    'qin = (q_mohembo.set_index("date")[["Qin_m3s", "Qin_km3", "Qin_mm"]]\n',
    "       .sort_index())\n",
    "\n",
    "bad_qin_dates = qin.index[qin.index.day != 1]\n",
    "if len(bad_qin_dates) > 0:\n",
    '    raise ValueError(f"Qin dates not normalized to month-start (day!=1). Example: {bad_qin_dates[:5].tolist()}")\n',
    "\n",
    "df_balance = pd.concat([Pmm, Pkm3, ETmm_med, ETkm3_med, gr, dS_km3, dS_cm, qin], axis=1).sort_index()\n",
    "\n",
    "# -----------------------\n",
    "# 5) Error / closure checks\n",
    "# -----------------------\n",
    'df_balance["Qout_plus_G_km3"] = df_balance["Qin_km3"] + df_balance["Pkm3"] - df_balance["ETkm3"] - df_balance["dS_km3"]\n',
    'df_balance["Qout_plus_G_mm"]  = df_balance["Qin_mm"]  + df_balance["Pmm"]  - df_balance["ETmm"]  - (df_balance["dS_cm"] * 10.0)\n',
    "\n",
    'df_balance["has_closure"] = df_balance[["Qin_km3", "Pkm3", "ETkm3", "dS_km3"]].notna().all(axis=1)\n',
    "\n",
    'bad = df_balance.loc[df_balance["has_closure"] & (df_balance["Qout_plus_G_km3"] < 0)]\n',
    "if len(bad) > 0:\n",
    '    print(f"Note: {len(bad)} months have negative implied (Qout+G). "\n',
    '          "Could be storage/ET/P bias, timing mismatch, or true net inflow > outflow+G not holding for your control volume.")\n',
    "\n",
    "df_balance = df_balance.loc[df_balance.index.year > 2000].copy()\n",
    "\n",
    "# Quick-look: GRACE TWS anomaly over the delta\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.dates as mdates\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(12, 4))\n",
    'df_balance["GRACEkm3"].dropna().plot(ax=ax, title="GRACE TWS anomaly over Delta (km\\u00b3)")\n',
    'ax.set_ylabel("km\\u00b3 (anomaly)")\n',
    'ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))\n',
    'ax.spines["top"].set_visible(False)\n',
    'ax.spines["right"].set_visible(False)\n',
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "df_balance\n",
]

cells[balance_idx]["source"] = NEW_BALANCE
cells[balance_idx]["outputs"] = []
cells[balance_idx]["execution_count"] = None

# ── 5) Write back ─────────────────────────────────────
print(f"\nFinal: {len(cells)} cells")
with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1)
print("Notebook saved.")
