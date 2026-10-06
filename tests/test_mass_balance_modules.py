"""Regression tests for the mass-balance modules (no Earth Engine needed).

Run:  ~/anaconda3/envs/ee-map/bin/python -m pytest tests/test_mass_balance_modules.py -q
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
GRACE_NC = ROOT / "data" / "grace_subsaharan_out" / "GRCTellus.JPL.200204_202507.GLO.RL06.3M.MSCNv04CRI.nc"
DELTA_SHP = ROOT / "data" / "regions" / "Delta_UCB_WGS84" / "Delta_UCB_WGS84.shp"

needs_data = pytest.mark.skipif(not (GRACE_NC.exists() and DELTA_SHP.exists()), reason="local data missing")


@pytest.fixture(scope="module")
def blocks():
    import geopandas as gpd
    from src.grace_blocks import discover_blocks, gdf_union
    delta = gdf_union(gpd.read_file(DELTA_SHP).to_crs(epsg=4326))
    return discover_blocks(GRACE_NC, delta, pad_deg=8.0)


@needs_data
def test_grace_months_are_unique_and_match_solution_periods(blocks):
    """Month labels come from time_bounds midpoints: no duplicates, no early labels."""
    from src.grace_blocks import block_tws, open_grace
    ne = [b for b in blocks if b["intersects_ref"] and b["quadrant"] == "NE"][0]
    s = block_tws(GRACE_NC, ne)
    assert not s.index.duplicated().any()
    assert (s.index.day == 1).all()
    ds = open_grace(GRACE_NC)
    tb = pd.to_datetime(ds["time_bounds"].values[:, 0])
    ds.close()
    # the August-2002 solution (period starts 2002-08-01) must be labelled 2002-08
    assert pd.Timestamp("2002-08-01") in s.index
    assert pd.Timestamp("2002-06-01") not in s.index and pd.Timestamp("2002-07-01") not in s.index
    assert tb[2].month == 8


@needs_data
def test_neighbour_search_uses_delta_adjacent_blocks(blocks):
    from src.grace_blocks import neighbour_block
    delta_blocks = [b for b in blocks if b["intersects_ref"]]
    ne = [b for b in delta_blocks if b["quadrant"] == "NE"][0]
    nw = [b for b in delta_blocks if b["quadrant"] == "NW"][0]
    se = [b for b in delta_blocks if b["quadrant"] == "SE"][0]
    assert neighbour_block(ne, blocks, "W")["block_id"] == nw["block_id"]
    assert neighbour_block(ne, blocks, "S")["block_id"] == se["block_id"]
    dry = neighbour_block(se, blocks, "S")
    assert dry is not None and not dry["intersects_ref"]
    assert dry["lat1"] <= se["lat0"] + 1e-6


@needs_data
def test_gleam_dates_are_month_start(blocks):
    from src.et_products import gleam_files, gleam_et_over_geometry
    files = gleam_files()
    if not files:
        pytest.skip("no GLEAM files")
    ne = [b for b in blocks if b["intersects_ref"] and b["quadrant"] == "NE"][0]
    df = gleam_et_over_geometry(files[:2], ne["geometry"], "2002-01-01", "2004-01-01")
    assert (df["date"].dt.day == 1).all()
    assert df["coverage"].between(0, 1).all()


def _synthetic_inputs():
    idx = pd.date_range("2002-04-01", "2004-03-01", freq="MS")
    et = pd.DataFrame({"dataset": "A", "date": idx, "et_mm_mean": 10.0, "et_km3_total": 1.0, "coverage": 1.0})
    et2 = et.assign(dataset="B", et_mm_mean=20.0, et_km3_total=2.0)
    # GLEAM-style month-END dates must be tolerated
    et3 = pd.DataFrame({"dataset": "C", "date": idx + pd.offsets.MonthEnd(0), "et_mm_mean": 30.0,
                        "et_km3_total": 3.0, "coverage": 1.0})
    chirps = pd.DataFrame({"dataset": "CHIRPS", "date": idx, "ppt_mm_mean": 50.0, "ppt_km3_total": 5.0, "coverage": 1.0})
    tws = pd.DataFrame({"TWS_cm": np.arange(len(idx), dtype=float), "TWS_km3": np.arange(len(idx), dtype=float) * 0.1},
                       index=idx)
    tws.index.name = "date"
    tws = tws.drop(pd.Timestamp("2003-06-01"))       # a GRACE gap
    return pd.concat([et, et2, et3]), chirps, tws


def test_assemble_balance_median_centered_diff_and_gap():
    from src.balance import assemble_balance
    et, chirps, tws = _synthetic_inputs()
    bal = assemble_balance(et, chirps, tws, area_m2=1e11, start="2002-04-01", ds_scheme="centered",
                           fill_tws_gap_months=0, min_et_products=3)
    df = bal.df
    assert df.index.freqstr == "MS"
    # median of the three products, month-end product included after snapping
    assert np.isclose(df.loc["2002-06-01", "ET_km3"], 2.0)
    assert df.loc["2002-06-01", "n_et_products"] == 3
    # centred difference of a unit ramp is 1 cm/month, NaN next to the gap
    assert np.isclose(df.loc["2002-08-01", "dS_cm"], 1.0)
    assert np.isnan(df.loc["2003-05-01", "dS_cm"]) and np.isnan(df.loc["2003-07-01", "dS_cm"])
    assert not df.loc["2003-05-01", "has_closure"]
    assert np.isclose(df.loc["2002-08-01", "resid_km3"], 5.0 - 2.0 - 0.1)
    assert "Qin_km3" not in df.columns and bal.flux_cols == ["P_km3"]


def test_assemble_balance_backward_diff():
    from src.balance import assemble_balance
    et, chirps, tws = _synthetic_inputs()
    bal = assemble_balance(et, chirps, tws, area_m2=1e11, start="2002-04-01", ds_scheme="backward",
                           fill_tws_gap_months=0, min_et_products=3)
    assert np.isclose(bal.df.loc["2002-08-01", "dS_cm"], 1.0)
    assert np.isnan(bal.df.loc["2003-07-01", "dS_cm"])       # gap not bridged
    assert np.isnan(bal.df.loc["2002-04-01", "dS_cm"])


def test_one_month_tws_gap_is_interpolated_longer_gap_is_not():
    from src.balance import assemble_balance
    et, chirps, tws = _synthetic_inputs()                  # single-month gap at 2003-06
    bal = assemble_balance(et, chirps, tws, area_m2=1e11, start="2002-04-01", min_et_products=3)   # default: fill 1
    df = bal.df
    assert bal.tws_fill_months == 1
    assert df.loc["2003-06-01", "TWS_filled"] and df["TWS_filled"].sum() == 1
    assert np.isclose(df.loc["2003-06-01", "TWS_cm"], 14.0, atol=0.05)      # linear in time
    for m in ("2003-05-01", "2003-07-01"):                  # neighbours now have a centred ΔS
        assert np.isclose(df.loc[m, "dS_cm"], 1.0, atol=0.02)
        assert df.loc[m, "dS_uses_filled_tws"]
    assert not df.loc["2002-08-01", "dS_uses_filled_tws"]
    # a two-month gap is left alone
    tws2 = tws.drop(pd.Timestamp("2003-07-01"))
    df2 = assemble_balance(et, chirps, tws2, area_m2=1e11, start="2002-04-01", min_et_products=3).df
    assert not df2["TWS_filled"].any()
    assert df2.loc["2003-06-01":"2003-07-01", "TWS_cm"].isna().all()
    assert np.isnan(df2.loc["2003-05-01", "dS_cm"]) and np.isnan(df2.loc["2003-08-01", "dS_cm"])


def test_et_median_requires_minimum_number_of_products():
    from src.balance import assemble_balance
    et, chirps, tws = _synthetic_inputs()                  # 3 products everywhere ...
    et = et[~((et["dataset"] == "C") & (et["date"] >= "2003-10-01"))]   # ... but only 2 from Oct 2003
    bal = assemble_balance(et, chirps, tws, area_m2=1e11, start="2002-04-01", min_et_products=3)
    df = bal.df
    assert bal.min_et_products == 3
    assert df.loc["2003-09-01", "n_et_products"] == 3 and np.isclose(df.loc["2003-09-01", "ET_km3"], 2.0)
    assert df.loc["2003-10-01", "n_et_products"] == 2
    assert np.isnan(df.loc["2003-10-01", "ET_km3"]) and df.loc["2003-10-01", "et_too_few_products"]
    assert not df.loc["2003-10-01", "has_closure"] and np.isnan(df.loc["2003-10-01", "resid_km3"])
    assert df["et_too_few_products"].sum() == 6                        # Oct 2003 – Mar 2004
    # default guard (4) with 3 products drops every month
    assert assemble_balance(et, chirps, tws, area_m2=1e11, start="2002-04-01").df["ET_km3"].isna().all()


def test_fit_lateral_recovers_alpha_and_intercept():
    from src.lateral_flux import fit_lateral, alpha_sensitivity
    rng = np.random.default_rng(0)
    x = rng.normal(0, 2, 300)
    resid = -(0.4 * x + 1.5) + rng.normal(0, 0.01, 300)   # resid + 0.4 x + 1.5 ≈ 0
    fit = fit_lateral(resid, x[:, None])
    assert np.isclose(fit["alpha"][0], 0.4, atol=0.01)
    assert np.isclose(fit["intercept"], 1.5, atol=0.01)
    assert fit["improvement_pct"] > 90
    curve = alpha_sensitivity(resid, x, np.array([0.0, 0.4]))
    assert curve[1] < curve[0]


def test_fit_lateral_bounds():
    from src.lateral_flux import fit_lateral
    x = np.linspace(-1, 1, 50)
    resid = -0.5 * x
    fit = fit_lateral(resid, x[:, None], bounds=([-np.inf], [0.0]))
    assert fit["alpha"][0] <= 0.0 + 1e-9


def test_period_rules_for_sub_monthly_composites():
    import datetime as dt
    from src.ee_monthly import period_end_8day, period_end_dekad, period_end_month, overlap_days
    jan29 = dt.datetime(2019, 1, 29)
    assert period_end_8day(jan29) == dt.datetime(2019, 2, 6)
    assert overlap_days(jan29, period_end_8day(jan29), dt.datetime(2019, 1, 1), dt.datetime(2019, 2, 1)) == 3
    assert overlap_days(jan29, period_end_8day(jan29), dt.datetime(2019, 2, 1), dt.datetime(2019, 3, 1)) == 5
    # last composite of the year (DOY 361) stops at 1 January
    assert period_end_8day(dt.datetime(2019, 12, 27)) == dt.datetime(2020, 1, 1)
    assert period_end_8day(dt.datetime(2020, 12, 26)) == dt.datetime(2021, 1, 1)      # leap year: 6 days
    # WaPOR dekads: 1-10, 11-20, 21-end of month
    assert period_end_dekad(dt.datetime(2019, 1, 1)) == dt.datetime(2019, 1, 11)
    assert period_end_dekad(dt.datetime(2019, 1, 11)) == dt.datetime(2019, 1, 21)
    assert period_end_dekad(dt.datetime(2019, 1, 21)) == dt.datetime(2019, 2, 1)       # 11 days
    assert period_end_dekad(dt.datetime(2019, 2, 21)) == dt.datetime(2019, 3, 1)       # 8 days
    assert period_end_dekad(dt.datetime(2019, 12, 21)) == dt.datetime(2020, 1, 1)
    assert period_end_month(dt.datetime(2019, 12, 1)) == dt.datetime(2020, 1, 1)
    # month count is exact (Earth Engine's fractional month difference dropped the last month)
    from src.ee_monthly import n_months
    assert n_months("2019-01-01", "2020-01-01") == 12
    assert n_months("2002-04-01", "2026-10-01") == 294
    assert n_months("2003-01-01", "2003-01-01") == 0


def test_prorate_month_uses_mean_daily_rate_and_masks_partial_months():
    from src.ee_monthly import prorate_month
    rates = [1.0, 2.0, 3.0, 4.0, 5.0]
    days = [5, 8, 8, 8, 2]                      # five composites overlapping a 31-day month
    expected = sum(r * d for r, d in zip(rates, days)) / 31 * 31
    assert np.isclose(prorate_month(rates, days, 31), expected)
    # one masked composite: the remaining ones are scaled up to the full month
    rates_m = [1.0, 2.0, 3.0, 4.0, np.nan]
    exp_m = (1 * 5 + 2 * 8 + 3 * 8 + 4 * 8) / 29 * 31
    assert np.isclose(prorate_month(rates_m, days, 31), exp_m)
    # ... unless that drops coverage below 75 % of the month (23 of 31 days)
    assert np.isnan(prorate_month([1.0, np.nan, 3.0, 4.0, 5.0], days, 31))
    # too little coverage → NaN (e.g. the partial month at the end of a record)
    assert np.isnan(prorate_month([2.0, 2.0], [8, 8], 31))               # 16 of 31 days < 75 %
    assert np.isclose(prorate_month([2.0, 2.0, 2.0], [8, 8, 8], 31), 62) # 24 of 31 days ≥ 75 %
    # a monthly product: one image, full overlap, unchanged
    assert np.isclose(prorate_month([3.0], [30], 30), 90.0)


def test_totals_to_df_handles_masked_months_and_coverage():
    from src.ee_monthly import totals_to_df
    totals = {"200201": {"m3": 1e9, "area": 1e9}, "200202": {"m3": None, "area": None},
              "200203": {"m3": 1e8, "area": 1e8}}
    df = totals_to_df(totals, "X", area_m2=1e9, prefix="et", min_coverage=0.5)
    assert np.isclose(df.loc[df.date == "2002-01-01", "et_mm_mean"].item(), 1000.0)
    assert np.isclose(df.loc[df.date == "2002-01-01", "et_km3_total"].item(), 1.0)   # 1 m over 1e9 m²
    assert np.isclose(df.loc[df.date == "2002-01-01", "coverage"].item(), 1.0)
    assert np.isnan(df.loc[df.date == "2002-02-01", "et_mm_mean"].item())
    assert np.isnan(df.loc[df.date == "2002-03-01", "et_mm_mean"].item())     # coverage 0.1 < 0.5
