import os
import subprocess
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
def upload_to_gee(file_path, asset_folder):
    file_name = os.path.basename(file_path).split(".")[0]
    asset_id = f"{asset_folder}/{file_name}"
    print(file_path)
    print(asset_id)
    subprocess.run([
        "earthengine",
        "upload",
        "image",
        "--asset_id=" + asset_id,
        file_path,
    ])

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
