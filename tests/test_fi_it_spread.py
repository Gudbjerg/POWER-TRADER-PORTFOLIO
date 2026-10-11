"""Unit tests for models/fi_it_spread.py (synthetic, deterministic data)."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from models.fi_it_spread import add_rolling_mean, build_spread_frame, spread_statistics


def _frame(dates, values, col):
    return pd.DataFrame({"date": dates, col: values})


def test_build_spread_inner_join_and_sign():
    fi = _frame([date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3)], [10.0, 20.0, 30.0], "fi")
    it = _frame([pd.Timestamp("2026-01-02 00:00"), pd.Timestamp("2026-01-03 13:45"),
                 pd.Timestamp("2026-01-04")], [100.0, 90.0, 80.0], "it_north")
    out = build_spread_frame(fi, it)
    assert list(out.columns) == ["date", "fi", "it_north", "spread"]
    assert list(out["date"]) == [pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-03")]
    assert list(out["spread"]) == [-80.0, -60.0]   # FI minus IT-North


def test_build_spread_averages_duplicates_and_drops_missing():
    fi = _frame(["2026-01-01", "2026-01-01", "2026-01-02"], [10.0, 30.0, np.nan], "fi")
    it = _frame(["2026-01-01", "2026-01-02"], [50.0, 60.0], "it_north")
    out = build_spread_frame(fi, it)
    assert len(out) == 1
    assert out.loc[0, "fi"] == pytest.approx(20.0)
    assert out.loc[0, "spread"] == pytest.approx(-30.0)


def test_build_spread_empty_and_bad_columns():
    empty = build_spread_frame(pd.DataFrame(), _frame(["2026-01-01"], [1.0], "it_north"))
    assert empty.empty and list(empty.columns) == ["date", "fi", "it_north", "spread"]
    with pytest.raises(ValueError):
        build_spread_frame(_frame(["2026-01-01"], [1.0], "x"), _frame(["2026-01-01"], [1.0], "it_north"))


def test_rolling_mean_uses_calendar_days():
    dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-10", "2026-01-11"])
    df = pd.DataFrame({"date": dates, "spread": [1.0, 3.0, 5.0, 7.0]})
    out = add_rolling_mean(df, window_days=3, min_periods=1)
    # 11 Jan window (9 to 11 Jan) holds 10 and 11 Jan only, despite four rows of history
    assert out["spread_ma"].tolist() == [1.0, 2.0, 5.0, 6.0]


def _synthetic(n_days: int, end="2026-10-10"):
    dates = pd.date_range(end=end, periods=n_days, freq="D")
    rng = np.random.default_rng(7)
    spread = -100.0 + rng.normal(0.0, 10.0, n_days)
    return pd.DataFrame({"date": dates, "fi": 20.0 + spread * 0, "it_north": 20.0 - spread, "spread": spread})


def test_spread_statistics_matches_manual_computation():
    df = _synthetic(400)
    stats = spread_statistics(df, avg_window_days=30, z_window_days=365, min_z_obs=60)
    last30 = df["spread"].iloc[-30:]
    last365 = df["spread"].iloc[-365:]
    assert stats["latest_date"] == pd.Timestamp("2026-10-10")
    assert stats["latest_spread"] == pytest.approx(df["spread"].iloc[-1])
    assert stats["avg"] == pytest.approx(last30.mean())
    assert stats["n_avg"] == 30
    assert stats["n_fi_below"] == int((last30 < 0).sum())
    assert stats["n_z"] == 365
    expected_z = (df["spread"].iloc[-1] - last365.mean()) / last365.std(ddof=1)
    assert stats["z"] == pytest.approx(expected_z)
    assert stats["z_full_window"] is True
    assert stats["window_min"] == pytest.approx(last365.min())


def test_spread_statistics_partial_history_and_min_obs():
    df = _synthetic(63)
    stats = spread_statistics(df, z_window_days=365, min_z_obs=60)
    assert stats["n_z"] == 63
    assert stats["z_full_window"] is False
    assert np.isfinite(stats["z"])
    short = spread_statistics(_synthetic(40), min_z_obs=60)
    assert np.isnan(short["z"])


def test_spread_statistics_zero_variance_and_empty():
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=80, freq="D"),
        "fi": 10.0, "it_north": 50.0, "spread": -40.0,
    })
    stats = spread_statistics(df, min_z_obs=60)
    assert np.isnan(stats["z"])
    assert stats["share_fi_below"] == pytest.approx(1.0)
    assert spread_statistics(pd.DataFrame()) == {}
