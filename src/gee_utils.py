"""Utility helpers for Google Cloud Storage (GCS) and Google Earth Engine (GEE).

This module centralizes helper functions that were previously duplicated across
multiple notebooks. They provide a thin, explicit wrapper around common shell
(`gsutil`, `earthengine`) and Earth Engine Python API calls for:

* Listing bucket contents (files, folders, subfolders)
* Uploading Cloud Optimized GeoTIFFs (with dry‑run support)
* Basic asset inspection (type, metadata, counting)
* Folder creation / bulk deletion of subfolders
* Recursive traversal of an asset tree
* Making an entire asset tree public
* Sharing an entire asset tree with a user or group (dry‑run capable)

Design notes:
* Functions prefer returning simple Python data (lists/dicts) when practical.
* Destructive / long‑running operations expose a dry‑run style flag where
    relevant (`upload_to_gee` already mirrors this pattern).
* Error handling is shallow: errors are caught, logged to stdout,
    and processing continues for subsequent assets. This is suitable for
    interactive / notebook workflows; consider raising exceptions or integrating
    structured logging for production pipelines.
"""

import os
import subprocess
import shlex
import ee

# Google Cloud Storage bucket utilities
def list_bucket_files(folder: str):
    """List COG GeoTIFF file paths in a GCS folder (non‑recursive).

    Args:
        folder: A fully qualified GCS URI (e.g., ``gs://bucket/path/``).

    Returns:
        List[str]: Only entries ending with ``cog.tif`` (COG convention used in project).
    """
    result = subprocess.run(["gsutil", "ls", folder], stdout=subprocess.PIPE, text=True)
    files = result.stdout.strip().split("\n")
    return [f for f in files if f.endswith("cog.tif")]

def list_bucket_folders(bucket_name: str):
    """List immediate objects / pseudo-folders in a bucket root.

    Args:
        bucket_name: Bucket name without ``gs://`` prefix.

    Returns:
        List[str]: Raw output lines from ``gsutil ls gs://<bucket>``.
    """
    result = subprocess.run(["gsutil", "ls", f"gs://{bucket_name}"], stdout=subprocess.PIPE, text=True)
    files = result.stdout.strip().split("\n")
    return [f for f in files]

def list_bucket_subfolders(bucket_name: str, folder: str):
    """List immediate children under a specific folder path within a bucket.

    Args:
        bucket_name: Bucket name without scheme.
        folder: Folder path suffix (must include trailing slash if targeting that convention).

    Returns:
        List[str]: Raw entries returned by ``gsutil ls``.
    """
    result = subprocess.run(["gsutil", "ls", f"gs://{bucket_name}{folder}"], stdout=subprocess.PIPE, text=True)
    files = result.stdout.strip().split("\n")
    return [f for f in files]

# Earth Engine asset management
def upload_to_gee(file_path: str, asset_folder: str, execute: bool = True):
    """Upload a local or ``gs://`` COG/GeoTIFF to a GEE image asset.

    Args:
        file_path: Local filesystem path or ``gs://`` URL to the source raster.
        asset_folder: Destination GEE folder (e.g., ``projects/ee-yourproj/assets/dswe``).
        execute: When False perform a dry run (return command without executing).

    Returns:
        Dict[str, Any]: keys ``asset_id``, ``file_path``, ``executed`` (bool), ``command`` (List[str]).
    """
    file_name = os.path.basename(file_path).split(".")[0]
    asset_id = f"{asset_folder}/{file_name}"
    cmd = [
        "earthengine",
        "upload",
        "image",
        "--asset_id=" + asset_id,
        file_path,
    ]
    print(f"Source: {file_path}")
    print(f"Asset:  {asset_id}")
    if not execute:
        # print("[DRY RUN] " + " ".join(shlex.quote(p) for p in cmd))
        return {"asset_id": asset_id, "file_path": file_path, "executed": False, "command": cmd}
    subprocess.run(cmd)
    return {"asset_id": asset_id, "file_path": file_path, "executed": True, "command": cmd}

def make_assets_public(folder_path: str):
    """Make all direct child assets inside a folder publicly readable.

    This does not recurse—use :func:`make_tree_public` for a full subtree.
    Errors per asset are logged and do not halt iteration.
    """
    try:
        assets = ee.data.listAssets({'parent': folder_path})['assets']
        for asset in assets:
            asset_id = asset['name']
            print(f"Making asset public: {asset_id}")
            try:
                ee.data.setAssetAcl(asset_id, {'allUsersCanRead': True})
                print(f"Successfully made public: {asset_id}")
            except Exception as e:
                print(f"Failed to update ACL for {asset_id}. Error: {e}")
    except Exception as e:
        print(f"Failed to list assets in folder: {folder_path}. Error: {e}")

def check_asset_types(asset_folder: str, max_assets: int = 10):
    """Print the asset type for up to ``max_assets`` children of a folder.

    Args:
        asset_folder: Parent asset path.
        max_assets: Maximum assets to query (pagination not continued beyond this call).
    """
    try:
        assets = ee.data.listAssets({'parent': asset_folder, 'pageSize': max_assets})['assets']
        print(f"Checking asset types in '{asset_folder}':\n")
        for asset in assets:
            asset_id = asset['name']
            asset_type = asset.get('type', 'Unknown')
            print(f"{asset_id}: Type = {asset_type}")
    except Exception as e:
        print(f"Error: {e}")

def inspect_asset_metadata(asset_folder: str, max_assets: int = 10):
    """Print full metadata dictionary for up to ``max_assets`` children.

    Args:
        asset_folder: Parent asset path.
        max_assets: Maximum assets to inspect.
    """
    try:
        assets = ee.data.listAssets({'parent': asset_folder, 'pageSize': max_assets})['assets']
        print(f"Inspecting metadata for the first {len(assets)} assets in '{asset_folder}':\n")
        for asset in assets:
            asset_id = asset['name']
            print(f"Metadata for {asset_id}:")
            metadata = ee.data.getAsset(asset_id)
            for key, value in metadata.items():
                print(f"  {key}: {value}")
            print("\n")
    except Exception as e:
        print(f"Error inspecting metadata: {e}")

def count_assets_in_folder(asset_folder: str):
    """Count assets (any type) directly under a folder using pagination.

    Args:
        asset_folder: Parent asset path.
    """
    try:
        total_assets = 0
        next_page_token = None
        while True:
            request = {'parent': asset_folder, 'pageSize': 100, 'pageToken': next_page_token}
            assets_response = ee.data.listAssets(request)
            assets = assets_response.get('assets', [])
            total_assets += len(assets)
            next_page_token = assets_response.get('nextPageToken', None)
            if not next_page_token:
                break
        print(f"Total assets in folder '{asset_folder}': {total_assets}")
    except Exception as e:
        print(f"Error counting assets: {e}")

def create_gee_folder(parent_folder: str, folder_name: str):
    """Create a new folder under a parent asset path.

    Args:
        parent_folder: Existing parent asset path.
        folder_name: Leaf folder name to create.
    """
    asset_path = f"{parent_folder}/{folder_name}"
    try:
        ee.data.createAsset({'type': 'folder'}, asset_path)
        print(f"Successfully created folder: {asset_path}")
    except Exception as e:
        print(f"Failed to create folder: {asset_path}. Error: {e}")

def delete_all_subfolders(gee_asset_folder: str):
    """Delete (non‑recursively) all immediate subfolders in a folder.

    Args:
        gee_asset_folder: Parent folder whose direct child folders will be removed.
    """
    try:
        assets = ee.data.listAssets({'parent': gee_asset_folder})['assets']
        folders = [asset['name'] for asset in assets if asset['type'] == 'FOLDER']
        for folder in folders:
            try:
                print(f"Deleting folder: {folder}")
                ee.data.deleteAsset(folder)
            except Exception as e:
                print(f"Failed to delete folder: {folder}. Error: {e}")
        print("All subfolders deleted successfully.")
    except Exception as e:
        print(f"Failed to list assets in folder: {gee_asset_folder}. Error: {e}")


def walk_assets(parent: str):
    """Recursively collect all asset paths beneath a parent.

    Includes the contents (not the parent itself) of folders and image collections.

    Args:
        parent: Asset path to start from.

    Returns:
        List[str]: All discovered child asset IDs.
    """
    out = []
    try:
        page = ee.data.listAssets(parent)
    except Exception as e:
        print(f"Failed to list assets under {parent}: {e}")
        return out
    for a in page.get('assets', []):
        aid, typ = a.get('name'), a.get('type')
        if not aid:
            continue
        out.append(aid)
        if typ in ('FOLDER', 'IMAGE_COLLECTION'):
            out.extend(walk_assets(aid))
    return out


def make_tree_public(root: str):
    """Recursively set an entire asset subtree (including root) to public read.

    Preserves existing readers / writers / owners, only toggling the public flag.

    Args:
        root: Asset root path (folder / image collection).
    """
    ids = [root] + walk_assets(root)
    for aid in ids:
        try:
            acl = ee.data.getAssetAcl(aid)
            # Preserve owners/writers/readers, just flip public flag
            acl['all_users_can_read'] = True
            ee.data.setAssetAcl(aid, acl)
            print(f"PUBLIC: {aid}")
        except Exception as e:
            print(f"FAILED: {aid} -> {e}")


def share_tree_with_user(root: str, email: str, role: str = 'READER', dry_run: bool = False, sleep_sec: float = 0.0):
    """Recursively share an entire asset subtree with a principal.

    Args:
        root: Asset root path (folder / image collection) e.g. ``projects/ee-yourproj/assets/foo``.
        email: Principal identifier. If no prefix (``user:``, ``group:``, etc.) is present,
            ``user:`` is prepended automatically.
        role: Either ``'READER'`` or ``'WRITER'``.
        dry_run: When True, only print intended ACL changes.
        sleep_sec: Optional delay between API calls (helpful for large trees / rate limiting).

    Notes:
        - Existing ACL entries are preserved; duplicates are avoided.
        - Root asset is included.
        - For very large trees, you may wish to batch or add exponential backoff.
    """
    import time

    # Normalize the principal format
    principal = email if (":" in email) else f"user:{email}"
    if role not in ("READER", "WRITER"):
        raise ValueError("role must be 'READER' or 'WRITER'")

    ids = [root] + walk_assets(root)
    updated = 0
    for aid in ids:
        try:
            acl = ee.data.getAssetAcl(aid)
            # Ensure keys exist
            readers = set(acl.get('readers', []))
            writers = set(acl.get('writers', []))

            if role == 'READER':
                if principal not in readers:
                    readers.add(principal)
                    acl['readers'] = sorted(readers)
                action = 'ADD-READER'
            else:
                if principal not in writers:
                    writers.add(principal)
                    acl['writers'] = sorted(writers)
                action = 'ADD-WRITER'

            print(f"{action}: {principal} -> {aid}")
            if not dry_run:
                ee.data.setAssetAcl(aid, acl)
                updated += 1
                if sleep_sec:
                    time.sleep(sleep_sec)
        except Exception as e:
            print(f"FAILED: {aid} -> {e}")
    if not dry_run:
        print(f"Updated ACLs on {updated} assets under {root}")
