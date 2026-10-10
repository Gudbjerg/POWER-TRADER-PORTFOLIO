"""Unit tests for models/coint_stability.py (synthetic, deterministic data)."""
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import coint

from models.coint_stability import (
    rolling_eg_pvalues,
    stability_label,
    stability_summary,
    window_end_dates,
)


# ── window_end_dates ─────────────────────────────────────────────────────────

def test_window_ends_anchored_on_latest_date():
    dates = pd.bdate_range("2024-01-01", periods=400)
    ends = window_end_dates(dates, window=180, step=21)
    assert ends[-1] == dates[-1]
    assert ends[0] >= dates[179]
    assert len(ends) == (399 - 179) // 21 + 1
    positions = [dates.get_loc(e) for e in ends]
    assert all(b - a == 21 for a, b in zip(positions, positions[1:]))


def test_window_ends_handle_unsorted_duplicates_and_short_history():
    dates = list(pd.bdate_range("2024-01-01", periods=30))
    shuffled = dates[::-1] + dates[:5]
    assert window_end_dates(shuffled, window=10, step=10) == [dates[9], dates[19], dates[29]]
    assert window_end_dates(dates, window=31, step=5) == []
    with pytest.raises(ValueError):
        window_end_dates(dates, window=1, step=5)
    with pytest.raises(ValueError):
        window_end_dates(dates, window=10, step=0)


# ── rolling_eg_pvalues ───────────────────────────────────────────────────────

def _pair(n=420, seed=5, cointegrated=True):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n)
    x = 50 + np.cumsum(rng.normal(0, 1, n))
    if cointegrated:
        y = 5 + 1.5 * x + rng.normal(0, 1, n)            # stationary residual
    else:
        y = 50 + np.cumsum(rng.normal(0, 1, n))          # independent random walk
    return dates, y, x


def test_rolling_pvalues_detect_cointegrated_pair():
    dates, y, x = _pair(cointegrated=True)
    ends = window_end_dates(dates, window=180, step=60)
    out = rolling_eg_pvalues(dates, y, x, ends, window=180)
    assert list(out["window_end"]) == ends
    assert (out["p_value"] < 0.05).all()


def test_rolling_pvalues_independent_walks_mostly_not_cointegrated():
    dates, y, x = _pair(seed=21, cointegrated=False)
    ends = window_end_dates(dates, window=180, step=30)
    out = rolling_eg_pvalues(dates, y, x, ends, window=180)
    assert (out["p_value"] >= 0.05).mean() >= 0.6


def test_rolling_pvalue_is_min_of_both_directions_on_exact_window():
    dates, y, x = _pair(seed=8)
    end = dates[299]
    out = rolling_eg_pvalues(dates, y, x, [end], window=180)
    p_ab = coint(y[120:300], x[120:300])[1]
    p_ba = coint(x[120:300], y[120:300])[1]
    assert out.loc[0, "p_ab"] == pytest.approx(p_ab)
    assert out.loc[0, "p_ba"] == pytest.approx(p_ba)
    assert out.loc[0, "p_value"] == pytest.approx(min(p_ab, p_ba))
    assert out.loc[0, "window_start"] == dates[120]


def test_rolling_pvalues_use_pair_sample_and_flag_short_windows():
    dates, y, x = _pair(n=260)
    y = y.copy()
    y[:40] = np.nan                                      # pair sample starts at row 40
    out = rolling_eg_pvalues(dates, y, x, [dates[200], dates[259]], window=180)
    assert np.isnan(out.loc[0, "p_value"])              # only 161 aligned obs by row 200
    assert out.loc[1, "window_start"] == dates[80]       # last 180 aligned obs up to row 259
    assert np.isfinite(out.loc[1, "p_value"])


def test_constant_window_returns_nan():
    dates = pd.bdate_range("2024-01-01", periods=200)
    out = rolling_eg_pvalues(dates, np.ones(200), np.arange(200.0), [dates[-1]], window=180)
    assert np.isnan(out.loc[0, "p_ab"]) and np.isnan(out.loc[0, "p_ba"])
    assert np.isnan(out.loc[0, "p_value"])


# ── stability_summary / stability_label ──────────────────────────────────────

def _pvals(values):
    ends = pd.bdate_range("2025-01-01", periods=len(values), freq="21B")
    return pd.DataFrame({"window_end": ends, "p_value": values}), ends


def test_summary_share_streak_and_last_break():
    frame, ends = _pvals([0.01, 0.02, 0.20, 0.03, 0.04, 0.50, 0.60])
    out = stability_summary(frame, alpha=0.05)
    assert out["n_windows"] == 7 and out["n_coint"] == 4
    assert out["share"] == pytest.approx(4 / 7)
    assert out["current_coint"] is False
    assert out["current_streak"] == 2
    assert out["last_break"] == ends[5]
    assert out["label"] == "Episodic"


def test_summary_without_breaks_and_with_nan_windows():
    frame, _ = _pvals([0.01, np.nan, 0.02, 0.03])
    out = stability_summary(frame)
    assert out["n_windows"] == 3
    assert out["current_coint"] is True and out["current_streak"] == 3
    assert out["last_break"] is None
    assert out["label"] == "Persistent"
    never, _ = _pvals([0.3, 0.4])
    assert stability_summary(never)["label"] == "Absent"
    assert stability_summary(pd.DataFrame())["label"] == "n/a"


@pytest.mark.parametrize("share,label", [
    (1.0, "Persistent"), (0.75, "Persistent"), (0.74, "Episodic"), (0.25, "Episodic"),
    (0.1, "Rare"), (0.0, "Absent"), (float("nan"), "n/a"),
])
def test_stability_label_thresholds(share, label):
    assert stability_label(share, stable_share=0.75, episodic_share=0.25) == label
