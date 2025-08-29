# DSWE + GEE Utilities for Okavango

Tools to upload Cloud Optimized GeoTIFFs (COGs) from Google Cloud Storage to Google Earth Engine (GEE), manage GEE assets, and export DSWE-derived imagery. The repo is organized as Python utilities plus Jupyter notebooks.

---

## Project structure

- `src/`
  - `gee_utils.py` — shared helpers for:
    - Listing GCS folders/files (`gsutil` wrappers)
    - Uploading images to GEE (`earthengine upload image`)
    - Asset ACL updates, listing, metadata inspection, counting
    - Creating/deleting GEE folders
- `notebooks/`
  - `bucket_to_gee.ipynb` — discover COGs in a GCS bucket and upload to a GEE asset folder
  - `gee_asset_management.ipynb` — inspect, count, and manage GEE assets (ACLs, metadata)
  - `manage_gee_assets_okavango.ipynb` — Okavango-focused asset maintenance tasks
  - `dswe_image_export.ipynb` — export DSWE-derived rasters (e.g., GeoTIFF/PNG)
  - `archive/` — date-prefixed archived notebooks (e.g., `2025-08-04_bucket_to_gee.ipynb`)
- `.gitignore` — excludes large data folders and local caches

Optional local data folders (ignored by git):
- `DSWE_images/`, `dswe-inman-lyons-points/`, etc.

---

## Requirements

- Python 3.10+
- Google Earth Engine Python API: `earthengine-api`
- Google Cloud SDK with `gsutil` installed

Authenticate once for Earth Engine and GCloud on this machine:

```bash
# Earth Engine
earthengine authenticate

# Google Cloud (for gsutil)
gcloud auth login
gcloud auth application-default login
```

---

## Using the notebooks

All notebooks import shared helpers from `src/gee_utils.py`. If running from the `notebooks/` folder, the notebooks add the project root to `sys.path` so imports work:

```python
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from src.gee_utils import (
    list_bucket_files, list_bucket_folders, list_bucket_subfolders,
    upload_to_gee, make_assets_public,
    check_asset_types, inspect_asset_metadata, count_assets_in_folder,
    create_gee_folder, delete_all_subfolders,
)
```

Notebook overview:
- `bucket_to_gee.ipynb`
  - Set `bucket_name`, `gee_asset_folder`
  - List files via `list_bucket_files` and call `upload_to_gee`
- `gee_asset_management.ipynb`
  - Use `check_asset_types`, `inspect_asset_metadata`, `count_assets_in_folder`
  - Use `make_assets_public` to flip ACLs
- `manage_gee_assets_okavango.ipynb`
  - Okavango-specific bulk tasks (cleanup, folder management)
- `dswe_image_export.ipynb`
  - DSWE export workflows (adjust to your assets and AOI)

---

## Notes

- Asset paths like `projects/ee-okavango/assets/...` are private by default. Use `make_assets_public` or share with specific users/groups.
- For large uploads, consider batching by folder and monitoring `earthengine upload tasks` in the CLI.
- Keep archived notebooks in `notebooks/archive/` with date prefixes for provenance.

---

## Development

- Shared utilities live in `src/gee_utils.py`. Prefer updating there vs duplicating code in notebooks.
- If Python can’t find `src`, set `PYTHONPATH` to the repo root or use the small `sys.path` snippet above.

---

<!--
Legacy content below referenced a GEE Code Editor script workflow for IL vs DSWE comparison.
It has been commented out because this repository now centers on Python + notebooks.

## What this does (legacy)
- Builds a DSWE ImageCollection from monthly assets, harmonizes IL time windows,
- Compares seasonal means, minima (p10), variability (std, CV), and period change,
- Exports styled PNG/GeoTIFFs and previews layers in the GEE map.

## Quick start (GEE Code Editor) — legacy
1. Open the script in the Code Editor.
2. Check the AOI (Okavango polygon). Adjust if needed.
3. Set top-level parameters in JS.
-->
