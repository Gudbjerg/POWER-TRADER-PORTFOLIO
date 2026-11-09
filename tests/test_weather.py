"""Offline tests for the Weather Desk chain (data/weather.py, models/weather_impact.py)."""
import numpy as np
import pandas as pd
import pytest

from data import weather as W
from models import weather_impact as M


def test_power_curve_shape():
    cf = W.power_curve([0, 2.9, 3, 7.5, 12, 20, 25, np.nan])
    assert cf[0] == 0 and cf[1] == 0 and cf[2] == 0
    assert 0 < cf[3] < 1
    assert cf[4] == 1 and cf[5] == 1
    assert cf[6] == 0                       # cut-out
    assert np.isnan(cf[7])


def test_weighted_mean_renormalises_missing_sites():
    frame = pd.DataFrame({"a": [1.0, np.nan], "b": [3.0, 3.0]})
    out = W._weighted(frame, [1.0, 1.0])
    assert out.tolist() == [2.0, 3.0]


def _synthetic(n=150, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-04-01", periods=n, freq="D")
    wind_idx = rng.uniform(0.05, 0.6, n)
    solar_idx = rng.uniform(0.02, 0.3, n)
    hdd = rng.uniform(0, 15, n)
    hind = pd.DataFrame({"wind_idx": wind_idx, "solar_idx": solar_idx, "hdd": hdd,
                         "temp_c": 15.5 - hdd}, index=idx)
    dow = idx.dayofweek
    wind = 2 + 50 * wind_idx + rng.normal(0, 0.5, n)
    solar = 1 + 60 * solar_idx + rng.normal(0, 0.3, n)
    load = 52 + 0.8 * hdd - 8 * (dow == 5) - 10 * (dow == 6) + rng.normal(0, 0.5, n)
    ttf = pd.DataFrame({"date": idx, "price": 40 + rng.normal(0, 1, n).cumsum() * 0.1})
    ttf_prev = ttf.set_index("date")["price"].shift(1).to_numpy()
    price = -30 + 4.5 * (load - wind - solar) + 1.0 * np.nan_to_num(ttf_prev, nan=40) + rng.normal(0, 3, n)
    actuals = pd.DataFrame({"wind_gw": wind, "solar_gw": solar, "load_gw": load, "price": price}, index=idx)
    return hind, actuals, ttf


def test_fit_recovers_coefficients_and_validates():
    hind, actuals, ttf = _synthetic()
    model = M.fit_weather_model(hind, actuals, ttf, holdout_days=30)
    assert model.gw_per_wind_idx == pytest.approx(50, rel=0.05)
    assert model.solar.coef["solar_idx"] == pytest.approx(60, rel=0.05)
    assert model.load.coef["hdd"] == pytest.approx(0.8, rel=0.1)
    assert model.eur_per_gw == pytest.approx(4.5, rel=0.05)
    assert model.uses_gas
    v = model.validation
    assert v["n"] >= 25 and v["mae_model"] < v["mae_persistence"]


def test_price_equation_without_gas_history():
    hind, actuals, _ = _synthetic()
    model = M.fit_weather_model(hind, actuals, pd.DataFrame(), holdout_days=30)
    assert not model.uses_gas
    gen = model.generation(hind.iloc[-3:])
    assert model.price_from(gen["residual_gw"], None).notna().all()


def test_revision_more_wind_lowers_price():
    hind, actuals, ttf = _synthetic()
    model = M.fit_weather_model(hind, actuals, ttf)
    rev = pd.DataFrame({"latest": [0.4, 0.2], "previous": [0.2, 0.2]},
                       index=pd.date_range("2026-11-10", periods=2))
    out = M.revision_impact(model, rev)
    assert out.loc["2026-11-10", "wind_change_gw"] > 0
    assert out.loc["2026-11-10", "price_effect"] < 0
    assert out.loc["2026-11-11", "price_effect"] == pytest.approx(0)


def test_ensemble_band_is_ordered():
    hind, actuals, ttf = _synthetic()
    model = M.fit_weather_model(hind, actuals, ttf)
    days = pd.date_range("2026-11-10", periods=4)
    base = model.generation(hind.iloc[:4].set_axis(days))
    members = pd.DataFrame(np.random.default_rng(0).uniform(0, 0.8, (4, 51)), index=days)
    band = M.ensemble_band(model, members, base, 40.0)
    assert (band["wind_p10"] <= band["wind_p90"]).all()
    assert (band["price_p10"] <= band["price_p90"]).all()


def test_hydro_anomaly_and_leap_day():
    normals = pd.DataFrame({"precip_mm": 5.0, "temp_c": 2.0},
                           index=pd.Index([f"{m:02d}-{d:02d}" for m in range(1, 13) for d in range(1, 32)]))
    outlook = pd.DataFrame({"precip_mm": [10.0, 0.0, 5.0], "temp_c": [-1.0, 3.0, 4.0]},
                           index=pd.to_datetime(["2028-02-28", "2028-02-29", "2028-03-01"]))
    out = M.hydro_anomaly(outlook, normals)
    assert out["precip_normal_mm"] == pytest.approx(15.0)
    assert out["precip_ratio"] == pytest.approx(1.0)
    assert out["snow_share"] == pytest.approx(10 / 15)


def test_normals_fold_leap_day():
    idx = pd.to_datetime(["2024-02-28", "2024-02-29", "2025-02-28"])
    normals = W.normals_from_daily(pd.DataFrame({"precip_mm": [1.0, 3.0, 2.0], "temp_c": 0.0}, index=idx))
    assert list(normals.index) == ["02-28"]
    assert normals.loc["02-28", "precip_mm"] == pytest.approx(2.0)


def test_bundled_seed_files_are_valid():
    hind = pd.read_csv("data/weather_hindcast_v1.csv", parse_dates=["date"])
    assert len(hind) >= 120 and {"wind_idx", "solar_idx", "hdd"} <= set(hind.columns)
    assert hind["wind_idx"].between(0, 1).all()
    normals = pd.read_csv("data/weather_hydro_normals_v1.csv", dtype={"month_day": str})
    assert len(normals) == 365 and normals["precip_mm"].ge(0).all()


def test_storage_monte_carlo_draws_down_in_winter():
    from datetime import date
    from models.storage_monte_carlo import run_monte_carlo
    paths, dates = run_monte_carlo(72.7, {}, n_paths=200, today=date(2026, 11, 9))
    assert dates[0] == date(2026, 11, 10) and dates[-1] == date(2027, 11, 1)
    apr1 = dates.index(date(2027, 4, 1))
    assert np.median(paths[:, apr1]) < 72.7 - 20          # winter withdrawals happen
    assert np.median(paths[:, -1]) > np.median(paths[:, apr1])   # then the refill
