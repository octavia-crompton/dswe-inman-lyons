"""Scan notebooks for EE init patterns."""
import json, glob

notebooks = sorted(glob.glob("notebooks/*.ipynb"))

for nbpath in notebooks:
    with open(nbpath) as f:
        nb = json.load(f)

    found = False
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell["source"])
        if "ee.Initialize" in src or "ee.Authenticate" in src:
            if not found:
                print(f"=== {nbpath} ===")
                found = True
            print(f"  Cell {i} ({cell['cell_type']}):")
            for line in cell["source"]:
                print(f"    {line!r}")
            print()
    if found:
        print()
