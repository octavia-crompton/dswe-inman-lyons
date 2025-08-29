import src.gee_utils as g

def test_upload_to_gee_dry_run_builds_command():
    res = g.upload_to_gee("/tmp/example.tif", "projects/x/assets/demo", execute=False)
    assert res["executed"] is False
    assert res["asset_id"].endswith("/demo/example")
    assert isinstance(res["command"], list)
