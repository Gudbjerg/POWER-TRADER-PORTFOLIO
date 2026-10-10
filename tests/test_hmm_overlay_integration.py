"""
Integration tests for the Layer 4 volatility overlay on top of a real (tiny)
Gaussian HMM, trained into a temporary directory so the committed weights are
never touched. Covers the trained path and the degradation paths: feature
matrix without NO2, and an unloadable weights file (Git LFS pointer).
"""
import numpy as np
import pandas as pd
import pytest

hmm_model = pytest.importorskip("models.hmm_model")
pytest.importorskip("hmmlearn")

from models.regime_volatility import (  # noqa: E402
    realised_volatility,
    transition_spike_proximity,
    volatility_by_regime,
)

FEATURES = ["no2_z", "nl_z", "spread_z", "ttf_z", "no2_vol5_z"]


def _synthetic_features(n: int = 260, seed: int = 11) -> pd.DataFrame:
    """Two alternating calm/volatile blocks so the HMM has structure to find."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=n)
    block = (np.arange(n) // 40) % 2                       # 0 calm, 1 volatile
    scale = np.where(block == 1, 6.0, 1.0)
    no2 = 60.0 + np.cumsum(rng.normal(0.0, scale))
    frame = pd.DataFrame({"date": dates, "no2": no2})
    for i, col in enumerate(FEATURES):
        frame[col] = block * (1.5 + 0.2 * i) + rng.normal(0.0, 0.4, n)
    return frame


@pytest.fixture()
def trained_model(tmp_path, monkeypatch):
    monkeypatch.setattr(hmm_model, "_MODEL_PATH", str(tmp_path / "hmm_model.joblib"))
    monkeypatch.setattr(hmm_model, "_META_PATH", str(tmp_path / "hmm_meta.json"))
    feats = _synthetic_features()
    meta = hmm_model.train_hmm(feats, FEATURES)
    assert "error" not in meta, meta
    return feats, tmp_path


def test_overlay_runs_on_decoded_history(trained_model):
    feats, _ = trained_model
    hist = hmm_model.get_regime_history(feats)
    assert len(hist) == len(feats)
    vol = realised_volatility(feats.set_index("date")["no2"], window=30)
    table = volatility_by_regime(hist, vol)
    stats = transition_spike_proximity(hist, vol, spike_pctl=90.0, window=5)
    assert table["days"].sum() == len(feats) - 30          # first 30 days have no full window
    assert stats["n_days"] == len(feats) - 30
    assert 0.0 <= stats["baseline_share"] <= 1.0
    if stats["n_transitions"]:
        assert 0.0 < stats["p_value"] <= 1.0


def test_inference_degrades_when_no2_features_missing(trained_model):
    feats, _ = trained_model
    no_no2 = feats.drop(columns=["no2", "no2_z", "no2_vol5_z"])
    assert hmm_model.predict_regime(no_no2) is None
    assert hmm_model.get_regime_history(no_no2).empty


def test_inference_degrades_with_lfs_pointer_weights(trained_model):
    feats, tmp_path = trained_model
    (tmp_path / "hmm_model.joblib").write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 3619\n"
    )
    assert hmm_model.predict_regime(feats) is None
    assert hmm_model.get_regime_history(feats).empty
