"""Patch the copied 1block_ne notebook: replace 2-block-east references with NE single block."""
import json

NB = "notebooks/mass_balance_grace_1block_ne.ipynb"

with open(NB) as f:
    nb = json.load(f)

REPLACEMENTS = [
    # Markdown header
    ("### Mass Balance — GRACE-aligned (2 east mascon blocks)",
     "### Mass Balance — GRACE-aligned (NE mascon block)"),
    # Markdown description
    ("computed over the **union of the two eastern JPL mascon blocks (NE + SE)** — the",
     "computed over the **northeast JPL mascon block (NE)** — the"),
    ("GRACE TWS, GLEAM) integrates over this same 2-block east footprint, not over the",
     "GRACE TWS, GLEAM) integrates over this same NE block footprint, not over the"),
    ("Using only the two eastern blocks focuses on the delta core while dropping the",
     "Using only the NE block focuses on the delta core while dropping the"),
    ("western blocks (NW + SW) that extend into dry Kalahari.",
     "other blocks that extend into dry Kalahari or away from the delta."),
    # GEOM_TAG
    ('GEOM_TAG = "grace_2block_east"', 'GEOM_TAG = "grace_1block_ne"'),
    # Block selection
    ('USE_BLOCKS = [b for b in delta_blocks if b["quadrant"].endswith("E")]',
     'USE_BLOCKS = [b for b in delta_blocks if b["quadrant"] == "NE"]'),
    ("# Drop the NW block (< 2 % of delta)", "# Select only the NE block"),
    ('print("Selected east blocks:",', 'print("Selected NE block:",'),
    # Area/union labels
    ("2-block east union area", "NE block area"),
    ("2-block east union", "NE block outline"),
    # East-alignment diagnostic
    ("East-alignment check (NE vs SE blocks):", "NE block check:"),
    # EE geometry comment
    ("# Build an EE MultiPolygon from the Shapely block geometries",
     "# Build an EE Polygon from the Shapely NE block geometry"),
    # Section headers
    ("## 4 \u2014 Monthly ET over 2-block east union", "## 4 \u2014 Monthly ET over NE block"),
    ("## 5 \u2014 CHIRPS precipitation over 2-block east union",
     "## 5 \u2014 CHIRPS precipitation over NE block"),
    ("## 6 \u2014 GRACE TWS over 2-block east union", "## 6 \u2014 GRACE TWS over NE block"),
    # Plot titles
    ("over 2-block east GRACE footprint", "over NE GRACE mascon block"),
    ("2-block east GRACE domain", "NE GRACE mascon block"),
    ("2-Block East GRACE Domain", "NE GRACE Mascon Block"),
    ("ET Diagnostic: Delta Polygon vs 2-Block East GRACE Domain",
     "ET Diagnostic: Delta Polygon vs NE GRACE Mascon Block"),
    # mass balance plot title tag
    ("Monthly ET over 2-block east GRACE footprint (area-mean)",
     "Monthly ET over NE GRACE mascon block (area-mean)"),
]

n_total = 0
for cell in nb["cells"]:
    new_lines = []
    for line in cell["source"]:
        for old, new in REPLACEMENTS:
            if old in line:
                line = line.replace(old, new)
                n_total += 1
        new_lines.append(line)
    cell["source"] = new_lines

    # Clear outputs so notebook runs fresh
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

with open(NB, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

print(f"Done – {n_total} replacements applied, outputs cleared.")
