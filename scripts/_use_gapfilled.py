"""Move gap-fill cell before the phase-plot cells and update all call-sites
to use df_gapfilled instead of df_combined."""
import json

NB = "notebooks/Mohembo time series.ipynb"

with open(NB) as f:
    nb = json.load(f)

# 1) Find gap-fill cell (contains "2nd-order polynomial gap-fill")
gf_idx = None
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"])
    if "2nd-order polynomial gap-fill" in src and "df_gapfilled" in src:
        gf_idx = i
        break
assert gf_idx is not None, "Could not find gap-fill cell"
print(f"Gap-fill cell at index {gf_idx}")

# 2) Find plot_phase_by_water_year cell (first phase-plot function)
phase_idx = None
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"])
    if "def plot_phase_by_water_year" in src:
        phase_idx = i
        break
assert phase_idx is not None, "Could not find plot_phase_by_water_year cell"
print(f"plot_phase_by_water_year cell at index {phase_idx}")

# 3) Move gap-fill cell to right before phase_idx
gf_cell = nb["cells"].pop(gf_idx)
# After pop, phase_idx may have shifted if gf was after it
if gf_idx < phase_idx:
    phase_idx -= 1
nb["cells"].insert(phase_idx, gf_cell)
print(f"Moved gap-fill cell to index {phase_idx}")

# 4) Update call-sites: df_combined -> df_gapfilled in specific patterns
# We only change the CALL sites, not function parameter names
replacements = [
    # plot_phase_by_water_year call
    ("plot_phase_by_water_year(df_combined,", "plot_phase_by_water_year(df_gapfilled,"),
    # phaseplots_wy_grid main call
    ("phaseplots_wy_grid(\n    df_combined,", "phaseplots_wy_grid(\n    df_gapfilled,"),
    # compile_wy_metrics call
    ("wy_metrics = compile_wy_metrics(\n    df_combined,", "wy_metrics = compile_wy_metrics(\n    df_gapfilled,"),
    # category plots - _filter_to_wys
    ("_filter_to_wys(df_combined,", "_filter_to_wys(df_gapfilled,"),
    # compile_limb_metrics call
    ("limb = compile_limb_metrics(\n    df_combined,", "limb = compile_limb_metrics(\n    df_gapfilled,"),
]

count = 0
for i, c in enumerate(nb["cells"]):
    new_source = []
    joined = "".join(c["source"])
    changed = False
    for old, new in replacements:
        if old in joined:
            joined = joined.replace(old, new)
            changed = True
            count += 1
    if changed:
        # Re-split into lines preserving the original line structure
        lines = joined.split("\n")
        new_lines = [line + "\n" for line in lines[:-1]]
        if lines[-1]:  # last line without trailing newline if original didn't have one
            new_lines.append(lines[-1])
        else:
            pass  # original ended with newline, already captured
        c["source"] = new_lines
        print(f"  Updated cell {i}")

print(f"Total replacements: {count}")

with open(NB, "w") as f:
    json.dump(nb, f, indent=1)

# Verify
with open(NB) as f:
    nb2 = json.load(f)
# Check gap-fill is before phase cells
for i, c in enumerate(nb2["cells"]):
    src = "".join(c["source"])
    if "2nd-order polynomial gap-fill" in src:
        print(f"\nGap-fill cell now at index {i}")
    if "def plot_phase_by_water_year" in src:
        print(f"plot_phase_by_water_year at index {i}")
        break

# Check no remaining df_combined in call-sites
remaining = []
for i, c in enumerate(nb2["cells"]):
    src = "".join(c["source"])
    for pattern in ["plot_phase_by_water_year(df_combined",
                     "phaseplots_wy_grid(\n    df_combined",
                     "compile_wy_metrics(\n    df_combined",
                     "_filter_to_wys(df_combined",
                     "compile_limb_metrics(\n    df_combined"]:
        if pattern in src:
            remaining.append((i, pattern))
if remaining:
    print(f"\nWARNING: unfixed references: {remaining}")
else:
    print("\nAll call-sites updated successfully.")
