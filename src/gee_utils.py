import os
import subprocess
import shlex
import ee

# Google Cloud Storage bucket utilities
def list_bucket_files(folder):
    result = subprocess.run([
        "gsutil", "ls", folder
    ], stdout=subprocess.PIPE, text=True)
    files = result.stdout.strip().split("\n")
    return [f for f in files if f.endswith("cog.tif")]

def list_bucket_folders(bucket_name):
    result = subprocess.run([
        "gsutil", "ls", f"gs://{bucket_name}"
    ], stdout=subprocess.PIPE, text=True)
    files = result.stdout.strip().split("\n")
    return [f for f in files]

def list_bucket_subfolders(bucket_name, folder):
    result = subprocess.run([
        "gsutil", "ls", f"gs://{bucket_name}{folder}"
    ], stdout=subprocess.PIPE, text=True)
    files = result.stdout.strip().split("\n")
    return [f for f in files]

# Earth Engine asset management
def upload_to_gee(file_path, asset_folder, execute: bool = True):
    """
    Upload a local/gs:// path to a GEE image asset.

    Parameters:
        file_path (str): Local path or gs:// URL to the COG/GeoTIFF.
        asset_folder (str): GEE folder where the image will be created.
        execute (bool): If False, only print the command (dry-run) and return.

    Returns:
        dict with keys: asset_id, file_path, executed (bool), command (list[str])
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

def make_assets_public(folder_path):
    """
    Make all assets in a specified GEE folder publicly readable.
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

def check_asset_types(asset_folder, max_assets=10):
    try:
        assets = ee.data.listAssets({'parent': asset_folder, 'pageSize': max_assets})['assets']
        print(f"Checking asset types in '{asset_folder}':\n")
        for asset in assets:
            asset_id = asset['name']
            asset_type = asset.get('type', 'Unknown')
            print(f"{asset_id}: Type = {asset_type}")
    except Exception as e:
        print(f"Error: {e}")

def inspect_asset_metadata(asset_folder, max_assets=10):
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

def count_assets_in_folder(asset_folder):
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

def create_gee_folder(parent_folder, folder_name):
    asset_path = f"{parent_folder}/{folder_name}"
    try:
        ee.data.createAsset({'type': 'folder'}, asset_path)
        print(f"Successfully created folder: {asset_path}")
    except Exception as e:
        print(f"Failed to create folder: {asset_path}. Error: {e}")

def delete_all_subfolders(gee_asset_folder):
    """
    Deletes all subfolders in a specified GEE asset folder.
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


def walk_assets(parent):
    """Recursively list all asset paths under a parent (folders, image collections, images, tables)."""
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


def make_tree_public(root):
    """Recursively set all assets under root to public read. Preserves other ACL fields."""
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
    """Recursively share all assets under root with a specific user/group.

    Args:
        root: Asset root path (folder/collection), e.g., 'projects/ee-yourproj/assets/foo'.
        email: Principal to grant access to. If it doesn't contain a prefix, 'user:' is assumed.
        role: 'READER' or 'WRITER'.
        dry_run: If True, print planned changes without applying.
        sleep_sec: Optional sleep between requests to avoid rate limits.

    Notes:
        - Preserves existing ACLs and de-duplicates principals.
        - Includes the root asset itself.
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
