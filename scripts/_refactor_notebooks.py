#!/usr/bin/env python3
"""
Refactor notebooks: standardise EE init, replace inline GRDC parsing,
replace inline water_year, and replace inline lat/lon detection.

Run from repo root:
    python3 scripts/_refactor_notebooks.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"

changes_log = []


def log(nb: str, cell_idx: int, msg: str):
    changes_log.append(f"  [{nb}] cell {cell_idx}: {msg}")


def save_notebook(path: Path, nb: dict):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")


def src_lines(source: list[str]) -> str:
    """Join cell source for pattern matching."""
    return "".join(source)


# ── EE INIT ──────────────────────────────────────────────────────────────────
# Patterns we want to replace (top-level only, not indented / inside functions):
#   Pattern A:  "import ee\n", "ee.Initialize()\n"          → add ee_init import + call
#   Pattern B:  "ee.Authenticate()\n", "ee.Initialize()\n"  → replace with ee_init()
#   Pattern C:  try/except block                            → replace with ee_init()
#   Pattern D:  standalone "ee.Initialize()\n" (no indent)  → replace with ee_init()
#
# We skip:
#   - Lines inside functions (indented ≥4 spaces)
#   - Commented-out lines ("# ee.Initialize()")
#   - Lines with project= arg (ee.Initialize(project=...))


def _is_top_level_ee_init(line: str) -> bool:
    """True if line is a top-level ee.Initialize() call (not indented, not commented)."""
    stripped = line.rstrip("\n")
    if stripped.startswith("#"):
        return False
    if stripped.startswith(" ") or stripped.startswith("\t"):
        return False
    return bool(re.match(r"^ee\.Initialize\(\)\s*$", stripped))


def _is_top_level_ee_auth(line: str) -> bool:
    stripped = line.rstrip("\n")
    if stripped.startswith("#"):
        return False
    if stripped.startswith(" ") or stripped.startswith("\t"):
        return False
    return bool(re.match(r"^ee\.Authenticate\(\)\s*$", stripped))


def _has_ee_init_import(source: list[str]) -> bool:
    return any("from src.gee_utils import ee_init" in l for l in source)


def _has_import_ee(source: list[str]) -> bool:
    return any(re.match(r"^import ee\b", l.strip()) for l in source if not l.strip().startswith("#"))


def refactor_ee_init_cell(source: list[str]) -> tuple[list[str], str | None]:
    """Refactor a single cell's EE init pattern.  Returns (new_source, description)."""
    text = src_lines(source)

    # Skip cells that already use ee_init
    if "ee_init()" in text and "from src.gee_utils import ee_init" in text:
        return source, None

    # Skip cells with ee.Initialize(project=...)
    if "ee.Initialize(project=" in text:
        return source, None

    new_lines = []
    changed = False
    ee_init_import_added = False

    i = 0
    while i < len(source):
        line = source[i]
        stripped = line.rstrip("\n")

        # Pattern C: try/except block
        if stripped.strip() == "try:" and i + 1 < len(source):
            # Check if this is a try: ee.Initialize() / except: ee.Authenticate(); ee.Initialize()
            block = src_lines(source[i:min(i+6, len(source))])
            if "ee.Initialize()" in block and ("ee.Authenticate()" in block or "ee.Authenticate ()" in block):
                # Replace entire try/except block with ee_init()
                # Find end of except block
                j = i + 1
                in_except = False
                while j < len(source):
                    s = source[j].rstrip("\n")
                    if s.strip().startswith("except"):
                        in_except = True
                    elif in_except and (not s.startswith(" ") and not s.startswith("\t") and s.strip()):
                        break
                    elif in_except and "ee.Initialize()" in s:
                        j += 1
                        break
                    j += 1

                if not ee_init_import_added and not _has_ee_init_import(source):
                    new_lines.append("from src.gee_utils import ee_init\n")
                    ee_init_import_added = True
                new_lines.append("ee_init()\n")
                changed = True
                i = j
                continue

        # Pattern A/B: top-level ee.Authenticate() line
        if _is_top_level_ee_auth(line):
            # Skip it (ee_init handles auth)
            changed = True
            i += 1
            continue

        # Pattern D: top-level ee.Initialize() line
        if _is_top_level_ee_init(line):
            if not ee_init_import_added and not _has_ee_init_import(source):
                new_lines.append("from src.gee_utils import ee_init\n")
                ee_init_import_added = True
            new_lines.append("ee_init()\n")
            changed = True
            i += 1
            continue

        # Comment hints about ee.Authenticate
        if re.match(r"^#\s*ee\.Authenticate\(\)", stripped):
            # Remove stale auth hints
            changed = True
            i += 1
            continue

        if re.match(r"^# First time.*ee\.Authenticate", stripped):
            changed = True
            i += 1
            continue

        if re.match(r"^# ee\.Authenticate\(\)\s*(#.*)?$", stripped):
            changed = True
            i += 1
            continue

        new_lines.append(line)
        i += 1

    if changed:
        return new_lines, "replaced EE init with ee_init()"
    return source, None


# ── FLOOD_PREDICTION: inline GRDC lat/lon + water_year ───────────────────────

def refactor_flood_prediction(nb_path: Path):
    """Refactor flood_prediction_okavango.ipynb:
    - Replace inline lat/lon GRDC header parsing with read_grdc_cmd_file
    - Replace inline water_year() with import from src.time_utils
    - Replace inline read_grdc_daily with src.grdc_io imports
    """
    with open(nb_path) as f:
        nb = json.load(f)

    name = nb_path.name
    modified = False

    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        text = src_lines(cell["source"])

        # Cell 1: imports + paths + inline water_year + inline GRDC lat/lon parsing
        if "def water_year(dt):" in text and "GRDC" in text and "STATIONS" in text:
            new_source = []
            skip_water_year = False
            skip_grdc_meta = False
            added_imports = False

            for line in cell["source"]:
                s = line.rstrip("\n")

                # Add our imports after the existing imports block
                if s.startswith("from sklearn") and not added_imports:
                    new_source.append(line)
                    # We'll add our imports after the warnings block
                    continue

                # Replace water_year definition
                if "def water_year(dt):" in s:
                    skip_water_year = True
                    continue
                if skip_water_year:
                    if s.strip().startswith("return dt.year"):
                        skip_water_year = False
                        continue
                    continue

                # Skip inline GRDC lat/lon parsing block
                if "# Station GeoDataFrame for spatial plots" in s:
                    skip_grdc_meta = True
                    # Insert replacement code
                    new_source.append("# Station GeoDataFrame for spatial plots (using src.grdc_io)\n")
                    new_source.append("meta_rows = []\n")
                    new_source.append("for sid, info in STATIONS.items():\n")
                    new_source.append("    fp = GRDC / f\"{sid}_Q_Day.Cmd.txt\"\n")
                    new_source.append("    lat = lon = None\n")
                    new_source.append("    if fp.exists():\n")
                    new_source.append("        meta, _ = read_grdc_cmd_file(fp)\n")
                    new_source.append("        lat, lon = meta.get(\"lat\"), meta.get(\"lon\")\n")
                    new_source.append("    meta_rows.append({\"station_id\": sid, \"name\": info[\"name\"],\n")
                    new_source.append("                       \"river\": info[\"river\"], \"role\": info[\"role\"],\n")
                    new_source.append("                       \"lat\": lat, \"lon\": lon})\n")
                    continue

                if skip_grdc_meta:
                    # End of the inline block: look for the meta_df/station_gdf lines
                    if s.startswith("meta_df = pd.DataFrame"):
                        skip_grdc_meta = False
                        new_source.append(line)
                        continue
                    continue

                new_source.append(line)

            # Insert new imports
            final_source = []
            for line in new_source:
                final_source.append(line)
                if "from sklearn.exceptions import" in line:
                    final_source.append("\n")
                    final_source.append("from src.grdc_io import read_grdc_cmd_file, pick_discharge_column\n")
                    final_source.append("from src.time_utils import water_year\n")
                    added_imports = True

            cell["source"] = final_source
            modified = True
            log(name, i, "replaced inline water_year + GRDC lat/lon parsing with src imports")

        # Cell with read_grdc_daily: replace with src import
        if "def read_grdc_daily(" in text:
            new_source = []
            skip_func = False
            for line in cell["source"]:
                s = line.rstrip("\n")
                if s.startswith("def read_grdc_daily("):
                    skip_func = True
                    # Add a comment and import-based replacement
                    new_source.append("# read_grdc_daily replaced by src.grdc_io.read_grdc_cmd_file\n")
                    new_source.append("def read_grdc_daily(station_id: int, missing_val: float = -999.0) -> pd.DataFrame:\n")
                    new_source.append("    fp = GRDC / f\"{station_id}_Q_Day.Cmd.txt\"\n")
                    new_source.append("    if not fp.exists():\n")
                    new_source.append("        return pd.DataFrame()\n")
                    new_source.append("    _, ts = read_grdc_cmd_file(fp)\n")
                    new_source.append("    q_col = pick_discharge_column(ts)\n")
                    new_source.append("    return ts[[q_col]].rename(columns={q_col: 'Q_m3s'}).reset_index()\n")
                    continue
                if skip_func:
                    # Skip until we reach a line at indent 0 that starts a new statement
                    if s and not s.startswith(" ") and not s.startswith("\t"):
                        skip_func = False
                        new_source.append("\n")
                        new_source.append(line)
                    continue
                new_source.append(line)

            cell["source"] = new_source
            modified = True
            log(name, i, "replaced inline read_grdc_daily with src.grdc_io wrapper")

    if modified:
        save_notebook(nb_path, nb)


# ── MAIN ─────────────────────────────────────────────────────────────────────

def refactor_ee_init_all():
    """Iterate all notebooks and standardise EE initialization."""
    notebooks = sorted(NB_DIR.glob("*.ipynb"))

    for nb_path in notebooks:
        name = nb_path.name
        # Skip archive copies
        if "archive" in str(nb_path):
            continue

        with open(nb_path) as f:
            nb = json.load(f)

        modified = False
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue

            text = src_lines(cell["source"])
            # Only process cells that have EE init/auth patterns
            if "ee.Initialize" not in text and "ee.Authenticate" not in text:
                continue

            new_source, desc = refactor_ee_init_cell(cell["source"])
            if desc:
                cell["source"] = new_source
                modified = True
                log(name, i, desc)

        if modified:
            save_notebook(nb_path, nb)


def main():
    print("=== Refactoring notebooks ===\n")

    # 1) EE init standardization across all notebooks
    print("1) Standardising EE init...")
    refactor_ee_init_all()

    # 2) flood_prediction_okavango.ipynb: GRDC + water_year
    fp_nb = NB_DIR / "flood_prediction_okavango.ipynb"
    if fp_nb.exists():
        print("2) Refactoring flood_prediction_okavango.ipynb...")
        refactor_flood_prediction(fp_nb)

    # Summary
    print(f"\n=== {len(changes_log)} changes made ===")
    for entry in changes_log:
        print(entry)

    return 0 if changes_log else 1


if __name__ == "__main__":
    sys.exit(main())
