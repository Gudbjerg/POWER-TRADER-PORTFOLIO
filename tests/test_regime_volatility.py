"""Unit tests for models/regime_volatility.py (synthetic, deterministic data)."""
import numpy as np
import pandas as pd
import pytest

from models.regime_volatility import (
    realised_volatility,
    regime_transitions,
    transition_spike_proximity,
    volatility_by_regime,
)


def _prices(values, start="2026-01-05"):
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


# ── realised_volatility ──────────────────────────────────────────────────────

def test_abs_volatility_matches_manual_std():
    out = realised_volatility(_prices([10, 12, 11, 15, 14]), window=3)
    assert np.isnan(out.loc[0, "change"])
    assert list(out["change"].iloc[1:]) == [2.0, -1.0, 4.0, -1.0]
    assert np.isnan(out.loc[2, "rv"])                      # only two changes so far
    assert out.loc[3, "rv"] == pytest.approx(np.std([2.0, -1.0, 4.0], ddof=1))
    assert out.loc[4, "rv"] == pytest.approx(np.std([-1.0, 4.0, -1.0], ddof=1))


def test_log_volatility_guards_non_positive_prices():
    out = realised_volatility(_prices([10, 0, 5, 10, 20]), window=2, method="log")
    assert np.isnan(out.loc[1, "change"]) and np.isnan(out.loc[2, "change"])
    assert out.loc[3, "change"] == pytest.approx(np.log(2.0))
    assert np.isnan(out.loc[3, "rv"])                      # one valid change in the window
    assert out.loc[4, "rv"] == pytest.approx(0.0)          # ln2 and ln2


def test_constant_prices_have_zero_volatility_and_sorting():
    s = _prices([50.0] * 6)
    out = realised_volatility(s.iloc[::-1], window=3)     # unsorted input
    assert out["date"].is_monotonic_increasing
    assert out["rv"].dropna().eq(0.0).all()


def test_volatility_input_validation():
    with pytest.raises(ValueError):
        realised_volatility(_prices([1, 2, 3]), method="pct")
    with pytest.raises(ValueError):
        realised_volatility(_prices([1, 2, 3]), window=1)
    assert realised_volatility(pd.Series(dtype=float)).empty


# ── regime_transitions / volatility_by_regime ────────────────────────────────

def _regimes(labels, start="2026-01-05"):
    return pd.DataFrame({"date": pd.bdate_range(start, periods=len(labels)), "regime": labels})


def test_regime_transitions_detects_changes():
    tr = regime_transitions(_regimes(["A", "A", "B", "B", "A"]))
    assert list(tr["from_regime"]) == ["A", "B"]
    assert list(tr["to_regime"]) == ["B", "A"]
    assert list(tr["date"]) == list(pd.bdate_range("2026-01-05", periods=5)[[2, 4]])
    assert regime_transitions(pd.DataFrame()).empty


def test_volatility_by_regime_statistics():
    regimes = _regimes(["Calm", "Calm", "Stress", "Stress", "Stress"])
    vol = pd.DataFrame({
        "date": regimes["date"], "price": [10.0, 20.0, 30.0, 40.0, 50.0],
        "rv": [1.0, np.nan, 5.0, 7.0, 9.0],
    })
    out = volatility_by_regime(regimes, vol)
    assert list(out["regime"]) == ["Stress", "Calm"]       # sorted by mean rv
    stress = out.iloc[0]
    assert stress["days"] == 3 and stress["mean_rv"] == pytest.approx(7.0)
    assert stress["median_rv"] == pytest.approx(7.0) and stress["mean_price"] == pytest.approx(40.0)
    assert out.iloc[1]["days"] == 1                        # the NaN-rv day is excluded
    assert out["share"].sum() == pytest.approx(1.0)


# ── transition_spike_proximity ───────────────────────────────────────────────

def _proximity_fixture():
    n = 20
    labels = ["A"] * 9 + ["B"] * 6 + ["C"] * 5            # transitions at positions 9 and 15
    regimes = _regimes(labels)
    rv = np.ones(n)
    rv[10] = 10.0                                         # single spike day at position 10
    vol = pd.DataFrame({"date": regimes["date"], "price": 50.0, "rv": rv})
    return regimes, vol


def test_proximity_hand_computed_example():
    regimes, vol = _proximity_fixture()
    out = transition_spike_proximity(regimes, vol, spike_pctl=90.0, window=2)
    assert out["n_days"] == 20
    assert out["n_spike_days"] == 1
    assert out["n_transitions"] == 2
    assert out["n_near"] == 1                             # position 9 is within 2 days of 10
    assert out["observed_share"] == pytest.approx(0.5)
    assert out["baseline_share"] == pytest.approx(5 / 20)  # positions 8 to 12
    assert out["lift"] == pytest.approx(2.0)
    # Null shifts k = 5..15 (shifts below 2 x 2 + 1 are excluded): only k = 13, 14, 15
    # move a transition into positions 8 to 12, so p = (1 + 3) / (1 + 11)
    assert out["p_value"] == pytest.approx(4 / 12)
    assert out["near_flags"] == [True, False]


def test_proximity_excludes_warm_up_days():
    regimes, vol = _proximity_fixture()
    vol.loc[:9, "rv"] = np.nan                            # rv undefined for the first 10 days
    vol.loc[12, "rv"] = 10.0
    out = transition_spike_proximity(regimes, vol, spike_pctl=90.0, window=2)
    assert out["n_days"] == 10
    assert out["n_transitions"] == 1                      # the day-9 transition is not scored
    assert out["transition_dates"] == [regimes["date"].iloc[15]]


def test_proximity_without_transitions_and_empty_inputs():
    regimes = _regimes(["A"] * 12)
    vol = pd.DataFrame({"date": regimes["date"], "price": 1.0, "rv": np.arange(12, dtype=float)})
    out = transition_spike_proximity(regimes, vol, window=1)
    assert out["n_transitions"] == 0
    assert np.isnan(out["observed_share"]) and np.isnan(out["p_value"])
    assert transition_spike_proximity(pd.DataFrame(), vol) == {}
    assert transition_spike_proximity(regimes, pd.DataFrame()) == {}


def test_proximity_strong_clustering_is_significant():
    # Ten transitions, each one day before a spike, at irregular spacing in a
    # 400-day sample (periodic spacing would let cyclic shifts realign them)
    n = 400
    labels, current = [], 0
    spike_pos = [20, 55, 71, 120, 160, 201, 233, 290, 330, 371]
    switch = {p - 1 for p in spike_pos}
    for i in range(n):
        if i in switch:
            current += 1
        labels.append(f"R{current % 2}")
    regimes = _regimes(labels)
    rv = np.ones(n)
    rv[spike_pos] = 5.0
    vol = pd.DataFrame({"date": regimes["date"], "price": 1.0, "rv": rv})
    out = transition_spike_proximity(regimes, vol, spike_pctl=90.0, window=5)
    assert out["observed_share"] == pytest.approx(1.0)
    assert out["baseline_share"] == pytest.approx(110 / 400)
    assert out["p_value"] < 0.01
