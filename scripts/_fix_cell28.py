#!/usr/bin/env python3
"""Patch cell 28 (index 27) of the ET notebook with updated color scheme."""
import json

nb_path = "notebooks/ET comparison over the delta.ipynb"
with open(nb_path, "r") as f:
    nb = json.load(f)

new_code = r'''import matplotlib.pyplot as plt

# --- use only months where closure is possible ---
dfc = df_balance[df_balance["has_closure"]].copy()

# Save median-ET plots in their own subfolder
median_dir = fig_dir / "median_ET"
median_dir.mkdir(parents=True, exist_ok=True)

# Explicit color scheme: blue=Qin, purple=P, green=ET, gray=ΔS
BAL_COLORS = {
    "Qin (Mohembo)": "#1f77b4",
    "P (CHIRPS)":    "#9467bd",
    "ET (median)":   "#2ca02c",
    "ΔS (GRACE)":    "#7f7f7f",
}

# -----------------------
# Plot 1: Terms in km³/month
# -----------------------
terms = dfc[["Qin_km3", "Pkm3", "ETkm3", "dS_km3", "Qout_plus_G_km3"]].copy()
plot_terms = pd.DataFrame(index=terms.index)
plot_terms["Qin (Mohembo)"] = terms["Qin_km3"]
plot_terms["P (CHIRPS)"]    = terms["Pkm3"]
plot_terms["ET (median)"]   = -terms["ETkm3"]
plot_terms["ΔS (GRACE)"]    = terms["dS_km3"]

fig, ax = plt.subplots(figsize=(12, 5))
smoothed = plot_terms.rolling(3, center=True).mean()
for col in smoothed.columns:
    ax.plot(smoothed.index, smoothed[col], linewidth=1, color=BAL_COLORS[col], label=col)
ax.axhline(0, lw=1, c="0.4")
ax.set_title(f"Okavango Delta monthly mass balance terms (3-mo smoothed) [{GEOM}]")
ax.set_ylabel("km³ / month")
ax.set_xlabel("")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
plt.tight_layout()
fig.savefig(median_dir / "mass_balance_terms.png", dpi=150, bbox_inches="tight")
plt.show()

# Cumulative colors
CUM_COLORS = {
    "∑ΔS (GRACE)": "#7f7f7f",
    "∑(Qin+P-ET)": "#9467bd",
}

# -----------------------
# Plot 2: Cumulative anomalies
# -----------------------
cum = pd.DataFrame(index=dfc.index)
cum["∑ΔS (GRACE)"] = dfc["dS_km3"].cumsum()
cum["∑(Qin+P-ET)"] = (dfc["Qin_km3"] + dfc["Pkm3"] - dfc["ETkm3"]).cumsum()

fig, ax = plt.subplots(figsize=(12, 4))
for col in cum.columns:
    ax.plot(cum.index, cum[col], linewidth=1.5, color=CUM_COLORS[col], label=col)
ax.axhline(0, lw=1, c="0.4")
ax.set_title(f"Cumulative sums [{GEOM}]")
ax.set_ylabel("km³ (cumulative)")
ax.set_xlabel("")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
plt.tight_layout()
fig.savefig(median_dir / "cumulative_sums.png", dpi=150, bbox_inches="tight")
plt.show()'''

# Convert to notebook source format (list of lines with \n)
lines = new_code.split("\n")
source_lines = [line + "\n" for line in lines[:-1]] + [lines[-1]]

nb["cells"][27]["source"] = source_lines

with open(nb_path, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Done – cell 28 patched with BAL_COLORS / CUM_COLORS color scheme.")
