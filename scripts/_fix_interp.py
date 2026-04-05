"""Fix polynomial interpolation to use positional index (avoid duplicate-x error)."""
import json

NB = "notebooks/Mohembo time series.ipynb"
with open(NB) as f:
    nb = json.load(f)

old_lines = [
    "    # poly interpolation (order=2) on the full series\n",
    '    filled = s.interpolate(method="polynomial", order=2, limit_direction="forward")\n',
]

new_lines = [
    "    # poly interpolation (order=2) \u2014 use positional index to avoid\n",
    '    # "duplicate x" errors when the DatetimeIndex has repeated months\n',
    "    filled = (\n",
    "        s.reset_index(drop=True)\n",
    '         .interpolate(method="polynomial", order=2, limit_direction="forward")\n',
    "    )\n",
    "    filled.index = s.index\n",
]

fixed = False
for i, c in enumerate(nb["cells"]):
    src = c["source"]
    # Find the two consecutive lines
    for j in range(len(src) - 1):
        if src[j] == old_lines[0] and src[j + 1] == old_lines[1]:
            c["source"] = src[:j] + new_lines + src[j + 2:]
            print(f"Fixed cell {i} (lines {j}-{j+1})")
            fixed = True
            break
    if fixed:
        break

if not fixed:
    print("ERROR: pattern not found")
    raise SystemExit(1)

with open(NB, "w") as f:
    json.dump(nb, f, indent=1)

print("Saved.")
