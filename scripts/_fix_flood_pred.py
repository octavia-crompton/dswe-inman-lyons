"""Fix flood_prediction_okavango.ipynb: water_year + GRDC lat/lon."""
import json

nb_path = "notebooks/flood_prediction_okavango.ipynb"
with open(nb_path) as f:
    nb = json.load(f)

# ── Cell 1: Replace inline water_year with import ──
cell1 = nb["cells"][1]
new_source = []
skip_wy = False
added_imports = False

for line in cell1["source"]:
    s = line.rstrip("\n")

    # Add imports after the warnings lines
    if "UndefinedMetricWarning" in s and not added_imports:
        new_source.append(line)
        new_source.append("\n")
        new_source.append("from src.grdc_io import read_grdc_cmd_file, pick_discharge_column\n")
        new_source.append("from src.time_utils import water_year\n")
        added_imports = True
        continue

    # Remove inline water_year
    if "# Water year: Oct 1" in s:
        continue
    if "def water_year(dt):" in s:
        skip_wy = True
        continue
    if skip_wy:
        if s.strip().startswith("return"):
            skip_wy = False
            continue
        continue

    new_source.append(line)

cell1["source"] = new_source
print("Cell 1 updated: added imports, removed inline water_year")

# ── Cell 3: Replace inline GRDC lat/lon parsing ──
cell3 = nb["cells"][3]
new_source = []
skip_inline_grdc = False

for line in cell3["source"]:
    s = line.rstrip("\n")

    if "# Station GeoDataFrame for spatial plots" in s:
        skip_inline_grdc = True
        new_source.append("# Station GeoDataFrame for spatial plots (using src.grdc_io)\n")
        new_source.append("meta_rows = []\n")
        new_source.append("for sid, info in STATIONS.items():\n")
        new_source.append('    fp = GRDC / f"{sid}_Q_Day.Cmd.txt"\n')
        new_source.append("    lat = lon = None\n")
        new_source.append("    if fp.exists():\n")
        new_source.append("        meta, _ = read_grdc_cmd_file(fp)\n")
        new_source.append('        lat, lon = meta.get("lat"), meta.get("lon")\n')
        new_source.append('    meta_rows.append({"station_id": sid, "name": info["name"],\n')
        new_source.append('                       "river": info["river"], "role": info["role"],\n')
        new_source.append('                       "lat": lat, "lon": lon})\n')
        new_source.append("\n")
        continue

    if skip_inline_grdc:
        if "meta_df = pd.DataFrame" in s:
            skip_inline_grdc = False
            new_source.append(line)
            continue
        continue

    new_source.append(line)

cell3["source"] = new_source
print("Cell 3 updated: replaced inline GRDC lat/lon parsing")

# Save
with open(nb_path, "w", encoding="utf-8", newline="\n") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

print("Saved flood_prediction_okavango.ipynb")
