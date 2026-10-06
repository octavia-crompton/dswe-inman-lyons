"""Tests for the front-velocity library (no data files needed except the OSM test, which skips if absent).

Run:  ~/anaconda3/envs/ee-map/bin/python -m pytest tests/test_okavango_fronts.py -q
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent


def _moving_front(speed_px_per_day: float, days: int = 40, ny: int = 40, nx: int = 60, px_deg: float = 0.01):
    """Binary daily mask of a straight N–S front advancing eastward at a known speed."""
    lat = -19.0 - np.arange(ny) * px_deg
    lon = 22.0 + np.arange(nx) * px_deg
    t = pd.date_range("2019-05-01", periods=days, freq="D")
    x = np.arange(nx)[None, None, :]
    pos = 10 + speed_px_per_day * np.arange(days)[:, None, None]
    wet = (x <= pos).astype("float32") * np.ones((1, ny, 1), dtype="float32")
    return xr.DataArray(wet, dims=("time", "lat", "lon"), coords={"time": t, "lat": lat, "lon": lon}, name="watermask")


def test_raw_binary_mask_gives_quantised_speeds_but_time_window_recovers_the_true_speed():
    from src.okavango_fronts import front_normal_velocity
    da = _moving_front(speed_px_per_day=0.5, days=120, nx=200)   # 0.5 px/day ≈ 525 m/day eastward
    px_m = 0.01 * 111_132
    # raw snapshots: |v| sits on a few geometric values regardless of the real speed
    raw = front_normal_velocity(da, "2019-05-05", "2019-05-25", front_value=0.5, bandwidth=0.05, smooth_px=1)
    v_raw = np.abs(raw["v_normal"].values[np.isfinite(raw["v_normal"].values)])
    assert len(np.unique(np.round(v_raw / 5) * 5)) <= 3          # a few geometric values only
    assert np.isclose(np.median(v_raw), 4 * px_m * np.cos(np.deg2rad(19)) / 20, rtol=0.05)  # 4 px/Δt here
    assert not np.isclose(np.median(v_raw), 0.5 * px_m * np.cos(np.deg2rad(19)), rtol=0.3)   # ≠ true speed
    # time-window averaging gives a fractional field and the true speed, with the advance sign (v < 0),
    # provided Δt is small relative to the window (here 10 d vs 30 d)
    win = front_normal_velocity(da, "2019-06-15", "2019-06-25", front_value=0.5, bandwidth=0.2,
                                time_window_days=30, min_grad_per_px=0.02)
    v = win["v_normal"].values[np.isfinite(win["v_normal"].values)]
    assert v.size > 0 and (v < 0).all()                      # water advancing → negative
    assert np.isclose(np.median(np.abs(v)), 0.5 * px_m * np.cos(np.deg2rad(19)), rtol=0.15)
    assert win.attrs["time_window_days"] == 30
    with pytest.warns(UserWarning, match="biased high"):          # Δt > window / 2 is flagged
        front_normal_velocity(da, "2019-06-05", "2019-06-30", front_value=0.5, bandwidth=0.2, time_window_days=30)


def test_gradient_floor_removes_flat_pixels():
    from src.okavango_fronts import front_normal_velocity
    da = _moving_front(speed_px_per_day=0.5)
    F = da.rolling(time=10, center=True).mean()
    # add a flat patch sitting exactly at 0.5 that must not count as front
    F[:, 30:35, 50:55] = 0.5
    strict = front_normal_velocity(F, "2019-05-10", "2019-05-25", front_value=0.5, bandwidth=0.1, min_grad_per_px=0.05)
    loose = front_normal_velocity(F, "2019-05-10", "2019-05-25", front_value=0.5, bandwidth=0.1, min_grad_per_px=0.0)
    # the patch's border is a real step and may count as front; its flat interior must not
    assert not strict["mask_front"].values[31:34, 51:54].any()
    assert loose["mask_front"].values[31:34, 51:54].all()
    assert np.nanmax(np.abs(strict["v_normal"].values)) < 1e4


def test_monthly_climatology_sign_and_counts():
    from src.okavango_fronts import front_normal_velocity_monthly_climatology
    da = _moving_front(speed_px_per_day=0.3, days=120)
    Fm = da.resample(time="MS").mean()
    clim = front_normal_velocity_monthly_climatology(Fm, front_value=0.5, bandwidth=0.1, max_gap_days=35)
    assert set(clim.data_vars) >= {"v_normal_mean", "pairs_count"}
    v = clim["v_normal_mean"].values
    assert np.nanmax(v) < 0                                   # steadily advancing front → all negative
    assert clim["pairs_count"].max() >= 1


OSM = ROOT / "data" / "raw" / "geofabrik_botswana" / "gis_osm_waterways_free_1.shp"


@pytest.mark.skipif(not OSM.exists(), reason="OSM extract not present")
def test_load_osm_channels_returns_named_delta_channels():
    from shapely.geometry import box
    from src.okavango_fronts import load_osm_channels
    ch = load_osm_channels(OSM, clip_geom=box(21.7, -20.3, 24.1, -18.2), fclass=("river",))
    assert (ch.geom_type == "LineString").all()
    assert {"Okavango", "Thaoge", "Khwai"} <= set(ch["name"].dropna())
    assert ch.crs.to_epsg() == 4326 and ch["length_km"].sum() > 1000
