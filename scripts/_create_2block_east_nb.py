"""Create a copy of mass_balance_grace_aligned.ipynb using only the 2 east GRACE cells."""
import json, re, sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "notebooks" / "mass_balance_grace_aligned.ipynb"
DST = Path(__file__).resolve().parent.parent / "notebooks" / "mass_balance_grace_2east.ipynb"

nb = json.loads(SRC.read_text(encoding="utf-8"))

def src(cell):
    return "".join(cell["source"])

def set_src(cell, text):
    cell["source"] = text.splitlines(True)
    # Clear outputs and execution count for a fresh notebook
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

replacements = [
    # ── Title / description ──
    ("# Mass Balance — GRACE-aligned (3 mascon blocks)",
     "# Mass Balance — GRACE-aligned (2 east mascon blocks)"),
    ("three JPL mascon blocks (NE + SE + SW)",
     "two eastern JPL mascon blocks (NE + SE)"),
    ("Dropping the NW block (1.6 % of the delta) eliminates a large\n"
     "dry-Kalahari footprint while retaining 98.4 % of the delta polygon coverage.",
     "Using only the two eastern blocks focuses on the delta core while\n"
     "dropping the western blocks (NW + SW) that extend into dry Kalahari."),

    # ── GEOM_TAG ──
    ('GEOM_TAG = "grace_3block"', 'GEOM_TAG = "grace_2block_east"'),

    # ── Block selection: keep only E quadrants ──
    ('USE_BLOCKS = [b for b in delta_blocks if b["quadrant"] != "NW"]',
     'USE_BLOCKS = [b for b in delta_blocks if b["quadrant"].endswith("E")]'),

    # ── Print labels ──
    ("Selected blocks:", "Selected east blocks:"),
    ('print(f"3-block union area   : ~{block_union_area_deg2:.1f} deg²")',
     'print(f"2-block east union area: ~{block_union_area_deg2:.1f} deg²")'),
    ('print(f"Delta polygon covered: {frac_covered:.1%}")',
     'print(f"Delta polygon covered : {frac_covered:.1%}")'),

    # ── Folium tooltip ──
    ('tooltip="3-block union"', 'tooltip="2-block east union"'),

    # ── EE area print ──
    ('print(f"3-block union area (EE): {block_area_m2 / 1e6:.0f} km²")',
     'print(f"2-block east union area (EE): {block_area_m2 / 1e6:.0f} km²")'),

    # ── Section headers ──
    ("## 4 — Monthly ET over 3-block union",
     "## 4 — Monthly ET over 2-block east union"),
    ("## 5 — CHIRPS precipitation over 3-block union",
     "## 5 — CHIRPS precipitation over 2-block east union"),
    ("## 6 — GRACE TWS over 3-block union",
     "## 6 — GRACE TWS over 2-block east union"),

    # ── GLEAM plot title ──
    ("GLEAM v4.2a — Monthly actual ET over 3-block GRACE footprint",
     "GLEAM v4.2a — Monthly actual ET over 2-block east GRACE footprint"),

    # ── All-ET plot title ──
    ("Monthly ET over 3-block GRACE footprint (area-mean)",
     "Monthly ET over 2-block east GRACE footprint (area-mean)"),

    # ── Diagnostic section header ──
    ("## ET Diagnostic: Delta Polygon vs 3-Block GRACE Domain",
     "## ET Diagnostic: Delta Polygon vs 2-Block East GRACE Domain"),

    # ── Diagnostic plot labels ──
    ('label="3-block GRACE domain"', 'label="2-block east GRACE domain"'),
    ('f"Domain: {m3:.0f}  |  Delta: {md:.0f} mm/mo"',
     'f"East domain: {m3:.0f}  |  Delta: {md:.0f} mm/mo"'),
    ('fig.suptitle("ET: Delta Polygon vs 3-Block GRACE Domain (mm/month)"',
     'fig.suptitle("ET: Delta Polygon vs 2-Block East GRACE Domain (mm/month)"'),

    # ── GLEAM comment ──
    ("# Mask: only pixels whose centre falls inside the 3-block union",
     "# Mask: only pixels whose centre falls inside the 2-block east union"),
    ("# Bounding box of the 3-block union for spatial clip",
     "# Bounding box of the 2-block east union for spatial clip"),
]

for cell in nb["cells"]:
    text = src(cell)
    changed = False
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new)
            changed = True
    if changed:
        set_src(cell, text)

# Also clear outputs for all code cells to get a clean notebook
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

DST.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Created: {DST}")
print(f"Key change: USE_BLOCKS filters to quadrant.endswith('E') → NE + SE only")
print(f"GEOM_TAG = 'grace_2block_east'")
