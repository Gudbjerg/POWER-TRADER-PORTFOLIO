"""
Offline tests for the data audit fixes: power-to-energy conversion, cross-border flows,
the feature matrix cache, day-ahead price sourcing, the merit order and model weights.
No network access: every external source is replaced by a fake.
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from data.energy_units import daily_energy, daily_mean, hourly_power


def _power(start: str, periods: int, freq: str = "15min", value: float = 1000.0,
           tz: str = "UTC") -> pd.Series:
    idx = pd.date_range(start, periods=periods, freq=freq, tz=tz)
    return pd.Series(value, index=idx, dtype=float)


# ── energy_units ─────────────────────────────────────────────────────────────

def test_quarter_hourly_and_hourly_power_give_the_same_energy():
    q = daily_energy(_power("2026-03-02", 96))
    h = daily_energy(_power("2026-03-02", 24, freq="h"))
    assert q.iloc[0] == pytest.approx(24_000.0)   # 1,000 MW for 24 h = 24,000 MWh
    assert h.iloc[0] == pytest.approx(q.iloc[0])


def test_partial_days_are_left_out():
    energy = daily_energy(_power("2026-03-02", 96 + 40))   # day two has 10 hours
    assert list(energy.index) == [pd.Timestamp("2026-03-02")]


def test_daily_mean_weighs_hours_equally_across_resolutions():
    # One day: first 12 hours hourly at 10, last 12 hours as 15-minute values at 30
    first = _power("2026-03-02 00:00", 12, freq="h", value=10.0, tz="Europe/Oslo")
    second = _power("2026-03-02 12:00", 48, freq="15min", value=30.0, tz="Europe/Oslo")
    mean = daily_mean(pd.concat([first, second]), tz="Europe/Oslo")
    assert mean.iloc[0] == pytest.approx(20.0)   # a plain mean of readings would give 26


def test_daily_mean_uses_the_local_delivery_day():
    prices = _power("2026-03-01 23:00", 25, freq="h", value=50.0, tz="UTC")   # 00:00 CET onwards
    prices.iloc[-1] = 500.0                                                    # next CET day
    mean = daily_mean(prices, tz="Europe/Oslo")
    assert list(mean.index) == [pd.Timestamp("2026-03-02")]
    assert mean.iloc[0] == pytest.approx(50.0)


def test_hourly_power_requires_a_timezone():
    with pytest.raises(ValueError):
        hourly_power(pd.Series([1.0], index=pd.DatetimeIndex(["2026-01-01"])))


# ── power_flows ──────────────────────────────────────────────────────────────

def test_corridor_daily_aligns_directions_reported_in_different_timezones():
    from data.power_flows import corridor_daily

    exp = _power("2026-03-01 23:00", 96, value=600.0, tz="UTC")              # 00:00 CET
    imp = _power("2026-03-02 00:00", 96, value=100.0, tz="Europe/Oslo")
    daily = corridor_daily(exp, imp, "NO2→GB")
    assert len(daily) == 1
    row = daily.iloc[0]
    assert row["export_mwh"] == pytest.approx(14_400.0)
    assert row["import_mwh"] == pytest.approx(2_400.0)
    assert row["net_flow_mwh"] == pytest.approx(-12_000.0)
    assert row["abs_net_mwh"] == pytest.approx(12_000.0)


def test_corridor_daily_treats_a_missing_direction_as_idle():
    from data.power_flows import corridor_daily

    daily = corridor_daily(_power("2026-03-02", 96, value=500.0, tz="Europe/Oslo"),
                           pd.Series(dtype=float), "NO2→DE")
    assert daily.iloc[0]["import_mwh"] == 0.0
    assert daily.iloc[0]["net_flow_mwh"] == pytest.approx(-12_000.0)


def test_nordlink_uses_the_de_lu_bidding_zone():
    from data.power_flows import CORRIDORS, ZONES

    nordlink = [c for c in CORRIDORS if c[3] == "NordLink"][0]
    assert ZONES[nordlink[1]] == "10Y1001A1001A82H"
    assert all(nb in ZONES for _, nb, _, _ in CORRIDORS)


# ── feature assembly ─────────────────────────────────────────────────────────

@pytest.fixture()
def fa(tmp_path, monkeypatch):
    import models.feature_assembly as fa
    monkeypatch.setattr(fa, "_RAW_FILE", str(tmp_path / "features_raw_v2.csv"))
    monkeypatch.setattr(fa, "_SEED_FILE", str(tmp_path / "seed_features_raw_v2.csv"))
    monkeypatch.setattr(fa, "_META_FILE", str(tmp_path / "features_meta.json"))
    return fa


def _raw_inputs(start: str, days: int, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days, freq="D", name="date")
    raw = pd.DataFrame({
        "no2": 60 + rng.normal(0, 5, days).cumsum() * 0.1,
        "nl": 80 + rng.normal(0, 5, days).cumsum() * 0.1,
        "ttf": 35 + rng.normal(0, 1, days).cumsum() * 0.1,
        "storage_fill": np.linspace(40, 90, days),
        "hydro_twh": np.linspace(30, 70, days),
        "de_wind_gwh": 300 + rng.normal(0, 50, days),
    }, index=idx)
    raw.loc[raw.index.dayofweek >= 5, "ttf"] = np.nan   # no TTF settlement at weekends
    return raw


def test_features_from_raw_keeps_trading_days_and_reports_coverage(fa):
    feats = fa.features_from_raw(_raw_inputs("2025-01-01", 300))
    assert feats["date"].dt.dayofweek.max() <= 4
    assert feats.attrs["coverage_report"] == {"core": True, "storage": True, "hydro": True, "wind": True}
    assert {"no_nl_spread", "storage_dev", "hydro_pct", "wind_de_z"} <= set(feats.columns)


def test_assembly_refreshes_only_the_tail_and_keeps_the_cache(fa, monkeypatch):
    today = fa._today_local()
    cached = _raw_inputs(str((today - timedelta(days=400)).date()), 390)
    fa._save_raw_cache(cached)
    calls = []

    def fake_fetch(start, end, history_start):
        calls.append((fa._day(start), fa._day(end)))
        fresh = _raw_inputs(str(fa._day(start).date()), (fa._day(end) - fa._day(start)).days, seed=9)
        fresh["no2"] = 999.0                       # recognisable fresh values
        return fresh

    monkeypatch.setattr(fa, "_fetch_inputs", fake_fetch)
    feats = fa.assemble_features(years=3)

    start, end = calls[0]
    last_complete = cached.dropna(subset=["no2", "nl", "ttf"]).index.max()
    assert start == last_complete - pd.Timedelta(days=fa.REFRESH_OVERLAP_DAYS)
    assert end == today + pd.Timedelta(days=1)
    old = feats[feats["date"] < start]
    new = feats[feats["date"] >= start]
    assert not old.empty and (old["no2"] != 999.0).all()
    assert not new.empty and (new["no2"] == 999.0).all()
    assert fa.get_feature_meta()["schema"] == fa.FEATURE_SCHEMA


def test_assembly_falls_back_to_the_cache_when_sources_fail(fa, monkeypatch):
    today = fa._today_local()
    fa._save_raw_cache(_raw_inputs(str((today - timedelta(days=200)).date()), 190))
    monkeypatch.setattr(fa, "_fetch_inputs", lambda *a, **k: pd.DataFrame())
    feats = fa.assemble_features(years=3)
    assert len(feats) > 100


def test_version_one_cache_is_not_read(fa, tmp_path, monkeypatch):
    # A schema-1 file under the old name must be ignored
    _raw_inputs("2025-01-01", 200).reset_index().to_csv(tmp_path / "features_cache.csv", index=False)
    monkeypatch.setattr(fa, "_fetch_inputs", lambda *a, **k: pd.DataFrame())
    assert fa.assemble_features(years=3).empty


def test_spot_gap_after_entsoe_is_filled_from_nord_pool(fa, monkeypatch):
    end_day = fa._today_local() + pd.Timedelta(days=1)
    start_day = end_day - pd.Timedelta(days=10)
    ent_idx = pd.date_range(start_day, end_day - pd.Timedelta(days=3), freq="D", name="date")
    ent = pd.DataFrame({"no2": 50.0, "nl": 70.0}, index=ent_idx)
    asked = {}

    def fake_np(days):
        asked["days"] = days
        idx = pd.date_range(end_day - pd.Timedelta(days=days), periods=days, freq="D", name="date")
        return pd.DataFrame({"no2": 51.0, "nl": 71.0}, index=idx)

    monkeypatch.setattr(fa, "_load_spot_entsoe", lambda s, e: ent)
    monkeypatch.setattr(fa, "_load_spot_nordpool", fake_np)
    spot = fa._load_spot(fa._local_midnight(start_day), fa._local_midnight(end_day))
    assert asked["days"] == 2
    assert spot.index.max() == end_day - pd.Timedelta(days=1)
    assert (spot.loc[ent_idx, "no2"] == 50.0).all()


# ── day-ahead prices ─────────────────────────────────────────────────────────

def test_dayahead_uses_entsoe_first_and_nord_pool_for_the_rest(monkeypatch):
    import data.spot_prices as sp

    today = pd.Timestamp.now(tz=sp.DELIVERY_TZ).date()
    days = [today - timedelta(days=i) for i in range(1, 4)]
    ent = pd.DataFrame([{"date": d, "zone": z, "price_eur_mwh": 50.0}
                        for d in days for z in ("NO1", "DE-LU")])
    monkeypatch.setattr(sp, "fetch_entsoe_daily", lambda zones, n: ent)
    asked = []

    def fake_nordpool(dates, areas):
        asked.append((tuple(dates), areas))
        return pd.DataFrame([{"date": pd.Timestamp(d).date(), "zone": z, "price_eur_mwh": 60.0}
                             for d in dates for z in areas])

    monkeypatch.setattr(sp, "_nordpool", fake_nordpool)
    df = sp.fetch_dayahead.__wrapped__(("NO1", "DE-LU", "SYS"), 3)
    assert "DE-LU" not in asked[0][1]                      # not on the Nord Pool portal
    got = df.set_index(["date", "zone"])["price_eur_mwh"]
    assert got[(days[0], "NO1")] == 50.0                   # ENTSO-E kept
    assert got[(days[0], "SYS")] == 60.0                   # only Nord Pool has SYS
    assert set(df["source"]) == {"ENTSO-E", "Nord Pool"}


# ── merit order ──────────────────────────────────────────────────────────────

def test_stack_uses_available_rather_than_installed_capacity():
    from models.supply_stack import build_stack, identify_marginal_fuel

    stack = build_stack(35.0, 70.0)
    assert stack["capacity_gw"].sum() < 120
    assert identify_marginal_fuel(stack, 60.0)["fuel"] not in ("wind", "solar")
    windy = build_stack(35.0, 70.0, {"wind": 55.0, "solar": 10.0})
    assert windy.set_index("fuel").loc["wind", "capacity_gw"] == 55.0
    assert identify_marginal_fuel(windy, 50.0)["fuel"] in ("wind", "solar", "hydro")


# ── model weights ────────────────────────────────────────────────────────────

def test_lstm_weights_round_trip_through_json():
    torch = pytest.importorskip("torch")
    from models import lstm_model

    model = lstm_model._build_model(5)
    restored = lstm_model._build_model(5)
    restored.load_state_dict(lstm_model._state_from_json(lstm_model._state_to_json(model.state_dict())))
    model.eval(); restored.eval()
    x = torch.randn(2, lstm_model.SEQ_LEN, 5)
    with torch.no_grad():
        assert torch.equal(model(x), restored(x))
