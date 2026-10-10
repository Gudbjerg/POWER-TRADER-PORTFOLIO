"""Unit tests for models/lng_arbitrage.py (synthetic, deterministic data)."""
import numpy as np
import pandas as pd
import pytest

from models.lng_arbitrage import (
    MWH_PER_MMBTU,
    align_ttf_jkm,
    ordinal,
    rolling_percentile,
    spread_regime,
    spread_summary,
    usd_mmbtu_to_eur_mwh,
    usd_mmbtu_to_eur_mwh as convert,
)


# ── Unit conversion ──────────────────────────────────────────────────────────

def test_conversion_scalar_known_value():
    # 10 USD/MMBtu at 1.25 USD per EUR = 8 EUR/MMBtu = 8 / 0.293071 EUR/MWh
    assert usd_mmbtu_to_eur_mwh(10.0, 1.25) == pytest.approx(8.0 / 0.293071)
    assert MWH_PER_MMBTU == pytest.approx(0.293071)


def test_conversion_round_trip_one_mwh():
    # 0.293071 USD/MMBtu at parity is exactly 1 EUR/MWh
    assert convert(0.293071, 1.0) == pytest.approx(1.0)


def test_conversion_vector_and_invalid_fx():
    out = usd_mmbtu_to_eur_mwh(np.array([10.0, 10.0, 10.0]), np.array([1.0, 0.0, -1.0]))
    assert out[0] == pytest.approx(10.0 / 0.293071)
    assert np.isnan(out[1]) and np.isnan(out[2])


def test_conversion_series_keeps_index():
    idx = pd.date_range("2026-01-01", periods=2, freq="D")
    out = usd_mmbtu_to_eur_mwh(pd.Series([12.0, 15.0], index=idx), pd.Series([1.2, np.nan], index=idx))
    assert list(out.index) == list(idx)
    assert out.iloc[0] == pytest.approx(10.0 / 0.293071)
    assert np.isnan(out.iloc[1])


# ── Alignment ────────────────────────────────────────────────────────────────

def _s(dates, values, tz=None):
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    if tz:
        idx = idx.tz_localize(tz)
    return pd.Series(values, index=idx, dtype=float)


def test_align_inner_join_asof_fx_and_spread():
    ttf = _s(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"], [40, 41, 42, 43], tz="America/New_York")
    jkm = _s(["2026-01-05", "2026-01-07", "2026-01-08"], [12.0, 12.5, 13.0], tz="America/New_York")
    fx = _s(["2026-01-02", "2026-01-07"], [1.10, 1.20], tz="Europe/London")
    out = align_ttf_jkm(ttf, jkm, fx, fx_max_age_days=3)
    # 6 Jan has no JKM settlement (not interpolated); 5 Jan uses the 2 Jan FX close (3 days old)
    assert list(out["date"]) == [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-07"), pd.Timestamp("2026-01-08")]
    assert list(out["eurusd"]) == [1.10, 1.20, 1.20]
    expected_jkm = 12.0 / 1.10 / 0.293071
    assert out.loc[0, "jkm_eur_mwh"] == pytest.approx(expected_jkm)
    assert out.loc[0, "spread"] == pytest.approx(40.0 - expected_jkm)


def test_align_drops_dates_with_stale_fx():
    ttf = _s(["2026-01-10"], [40.0])
    jkm = _s(["2026-01-10"], [12.0])
    fx = _s(["2026-01-01"], [1.1])   # nine days old
    assert align_ttf_jkm(ttf, jkm, fx, fx_max_age_days=3).empty


def test_align_accepts_mixed_datetime_resolutions():
    dates = pd.to_datetime(["2026-02-02", "2026-02-03"])
    ttf = pd.Series([40.0, 41.0], index=dates.as_unit("s"))
    jkm = pd.Series([12.0, 12.5], index=dates.as_unit("us"))
    fx = pd.Series([1.10, 1.10], index=dates.as_unit("ns"))
    out = align_ttf_jkm(ttf, jkm, fx)
    assert len(out) == 2
    assert out.loc[1, "spread"] == pytest.approx(41.0 - 12.5 / 1.10 / 0.293071)


def test_align_empty_inputs():
    empty = align_ttf_jkm(pd.Series(dtype=float), _s(["2026-01-01"], [1.0]), _s(["2026-01-01"], [1.1]))
    assert empty.empty
    assert list(empty.columns) == ["date", "ttf", "jkm_usd_mmbtu", "eurusd", "jkm_eur_mwh", "spread"]


# ── Rolling percentile ───────────────────────────────────────────────────────

def test_rolling_percentile_increasing_series():
    idx = pd.date_range("2025-01-01", periods=10, freq="D")
    out = rolling_percentile(pd.Series(np.arange(10, dtype=float), index=idx), window_days=365, min_obs=3)
    assert out.iloc[:2].isna().all()
    # Latest value is the window maximum: (below + 0.5 * equal) / n = (9 + 0.5) / 10
    assert out.iloc[-1] == pytest.approx(95.0)
    assert out.iloc[2] == pytest.approx((2 + 0.5) / 3 * 100)


def test_rolling_percentile_window_excludes_old_data():
    idx = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-10", "2025-01-11"])
    s = pd.Series([100.0, 100.0, 1.0, 2.0], index=idx)
    out = rolling_percentile(s, window_days=5, min_obs=1)
    # On 11 Jan the 5-day window (7 to 11 Jan) holds only 1.0 and 2.0
    assert out.iloc[-1] == pytest.approx(75.0)


def test_rolling_percentile_ties_and_nan():
    idx = pd.date_range("2025-01-01", periods=4, freq="D")
    out = rolling_percentile(pd.Series([5.0, 5.0, np.nan, 5.0], index=idx), window_days=30, min_obs=2)
    assert np.isnan(out.iloc[0])          # one observation < min_obs
    assert out.iloc[1] == pytest.approx(50.0)
    assert np.isnan(out.iloc[2])          # NaN observation stays NaN
    assert out.iloc[3] == pytest.approx(50.0)


# ── Labels and summary ───────────────────────────────────────────────────────

@pytest.mark.parametrize("n,expected", [
    (1, "1st"), (2, "2nd"), (3, "3rd"), (4, "4th"), (11, "11th"), (12, "12th"),
    (13, "13th"), (21, "21st"), (22, "22nd"), (100, "100th"), (0, "0th"),
])
def test_ordinal(n, expected):
    assert ordinal(n) == expected


def test_spread_regime():
    assert spread_regime(1.5) == "Europe premium"
    assert spread_regime(-0.1) == "Asia premium"
    assert spread_regime(0.0) == "Parity"
    assert spread_regime(float("nan")) == "n/a"


def test_spread_summary_values():
    dates = pd.date_range("2025-06-01", periods=200, freq="D")
    spread = np.linspace(-10.0, 9.9, 200)
    df = pd.DataFrame({
        "date": dates, "ttf": 40.0 + spread, "jkm_usd_mmbtu": 12.0, "eurusd": 1.1,
        "jkm_eur_mwh": 40.0, "spread": spread,
    })
    out = spread_summary(df, avg_window_days=30, pct_window_days=365, min_pct_obs=120)
    assert out["spread"] == pytest.approx(9.9)
    assert out["regime"] == "Europe premium"
    assert out["avg"] == pytest.approx(spread[-30:].mean())
    assert out["n_window"] == 200
    assert out["percentile"] == pytest.approx(199.5 / 200 * 100)
    assert out["share_positive"] == pytest.approx((spread > 0).mean())
    assert spread_summary(pd.DataFrame()) == {}
