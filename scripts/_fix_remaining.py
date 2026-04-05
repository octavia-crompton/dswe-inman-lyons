"""Check and fix ALL remaining call-sites that should use df_gapfilled."""
import json

NB = "notebooks/Mohembo time series.ipynb"

with open(NB) as f:
    nb = json.load(f)

changes = 0

for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"])
    new_src = src

    # plot_phase_by_water_year call-site (not the function def)
    if "plot_phase_by_water_year(df_combined," in new_src:
        new_src = new_src.replace(
            "plot_phase_by_water_year(df_combined,",
            "plot_phase_by_water_year(df_gapfilled,",
        )

    # phaseplots_wy_grid call-site
    if "phaseplots_wy_grid(\n    df_combined," in new_src:
        new_src = new_src.replace(
            "phaseplots_wy_grid(\n    df_combined,",
            "phaseplots_wy_grid(\n    df_gapfilled,",
        )

    # compile_wy_metrics call-site
    if "wy_metrics = compile_wy_metrics(\n    df_combined," in new_src:
        new_src = new_src.replace(
            "wy_metrics = compile_wy_metrics(\n    df_combined,",
            "wy_metrics = compile_wy_metrics(\n    df_gapfilled,",
        )

    # _filter_to_wys call-site
    if "_filter_to_wys(df_combined," in new_src:
        new_src = new_src.replace(
            "_filter_to_wys(df_combined,",
            "_filter_to_wys(df_gapfilled,",
        )

    # compile_limb_metrics call-site
    if "limb = compile_limb_metrics(\n    df_combined," in new_src:
        new_src = new_src.replace(
            "limb = compile_limb_metrics(\n    df_combined,",
            "limb = compile_limb_metrics(\n    df_gapfilled,",
        )

    if new_src != src:
        lines = new_src.split("\n")
        c["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]
        changes += 1
        print(f"  Fixed cell {i}")

print(f"\nTotal cells changed: {changes}")

with open(NB, "w") as f:
    json.dump(nb, f, indent=1)

# Final audit
with open(NB) as f:
    nb2 = json.load(f)
print("\nRemaining df_combined in analysis cells (>15):")
for i, c in enumerate(nb2["cells"]):
    if i <= 15:
        continue
    src = "".join(c["source"])
    pos = 0
    while True:
        p = src.find("df_combined", pos)
        if p == -1:
            break
        ctx = src[max(0, p - 20):p + 35].replace("\n", "\\n")
        print(f"  Cell {i}: ...{ctx}...")
        pos = p + 11
