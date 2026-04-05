"""
Hidden Markov Model Market Regime Classifier (Model 2).

Classifies the European power market into one of four structural regimes daily:
  - Hydro-driven:         High Norwegian reservoirs, low Nordic prices, wide spread
  - Gas-driven:           TTF dominant, NL closely tracks gas costs, moderate hydro
  - Wind/Renewables:      High wind penetration, suppressed midday prices, low volatility
  - Geopolitical stress:  Low storage, high TTF, elevated volatility across all zones

Architecture: Gaussian HMM with 4 hidden states (hmmlearn).
States are unsupervised — regime labels are assigned post-hoc by inspecting
the mean feature vector (centroid) of each state.

Weights are saved to models/weights/hmm_model.joblib.
"""
from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
os.makedirs(_WEIGHTS_DIR, exist_ok=True)

_MODEL_PATH = os.path.join(_WEIGHTS_DIR, "hmm_model.joblib")
_META_PATH  = os.path.join(_WEIGHTS_DIR, "hmm_meta.json")

N_STATES    = 4
N_ITER      = 200
RANDOM_SEED = 42

REGIME_COLORS = {
    "Hydro-driven":        "#4caf8f",
    "Gas-driven":          "#4c8cbf",
    "Renewables-driven":   "#d4ac3a",
    "Geopolitical stress": "#f85149",
    "Unknown":             "#8b949e",
}


# ── Save / load ──────────────────────────────────────────────────────────────

def is_trained() -> bool:
    return os.path.exists(_MODEL_PATH) and os.path.exists(_META_PATH)


def load_meta() -> dict:
    try:
        with open(_META_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _save(model, meta: dict) -> None:
    try:
        import joblib
        joblib.dump(model, _MODEL_PATH)
    except Exception:
        pass
    with open(_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)


def _load():
    try:
        import joblib
        if not os.path.exists(_MODEL_PATH):
            return None
        return joblib.load(_MODEL_PATH)
    except Exception:
        return None


# ── Regime labelling ─────────────────────────────────────────────────────────

def _label_regimes(model, feature_cols: list[str]) -> dict[int, str]:
    """
    Assign a human-readable label to each HMM state by inspecting the state
    mean vector (centroid).

    Logic (using z-scored features):
      - Hydro-driven:       high hydro_pct, negative nl_z (low NL prices), negative spread_z (NO cheap vs NL)
      - Gas-driven:         high ttf_z, high nl_z, moderate storage
      - Renewables-driven:  low no2_vol5_z, low nl_z, (high wind_de_z if available)
      - Geopolitical stress:negative storage_dev (low storage), high ttf_z, high no2_vol5_z
    """
    means = model.means_   # shape (n_states, n_features)
    col_idx = {c: i for i, c in enumerate(feature_cols)}

    def _get(state_means, col, default=0.0):
        if col in col_idx:
            return float(state_means[col_idx[col]])
        return default

    labels = {}
    # Score each state on regime signatures
    for s in range(N_STATES):
        m = means[s]

        ttf_z     = _get(m, "ttf_z")
        nl_z      = _get(m, "nl_z")
        no2_z     = _get(m, "no2_z")
        spread_z  = _get(m, "spread_z")
        hydro_pct = _get(m, "hydro_pct", 50)
        stor_dev  = _get(m, "storage_dev", 0)
        vol_z     = _get(m, "no2_vol5_z")
        wind_z    = _get(m, "wind_de_z", 0)

        scores = {
            "Hydro-driven":        hydro_pct * 0.4 - spread_z * 0.3 - nl_z * 0.3,
            "Gas-driven":          ttf_z * 0.4 + nl_z * 0.3 - hydro_pct * 0.01,
            "Renewables-driven":   wind_z * 0.4 - vol_z * 0.3 - nl_z * 0.3,
            "Geopolitical stress": -stor_dev * 0.3 + ttf_z * 0.3 + vol_z * 0.4,
        }
        labels[s] = max(scores, key=scores.get)

    # Resolve ties — each regime label should be unique across states
    used = set()
    final = {}
    for s, label in sorted(labels.items(), key=lambda x: -abs(means[x[0]]).sum()):
        if label not in used:
            final[s] = label
            used.add(label)
        else:
            remaining = [r for r in REGIME_COLORS if r != "Unknown" and r not in used]
            final[s] = remaining[0] if remaining else "Unknown"
            if remaining:
                used.add(remaining[0])

    return final


# ── Training ─────────────────────────────────────────────────────────────────

def train_hmm(features_df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """
    Fit the Gaussian HMM on the provided feature matrix.

    Parameters
    ----------
    features_df : Output of assemble_features(), date-sorted.
    feature_cols : From get_hmm_feature_cols(features_df).

    Returns
    -------
    dict with training results.
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        return {"error": "hmmlearn not installed. Run: pip install hmmlearn"}

    df = features_df.copy().dropna(subset=feature_cols).sort_values("date").reset_index(drop=True)

    if len(df) < 100:
        return {"error": f"Insufficient data: {len(df)} rows (need ≥100)"}

    X = df[feature_cols].values.astype(np.float64)

    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type="full",
        n_iter=N_ITER,
        random_state=RANDOM_SEED,
        verbose=False,
    )
    model.fit(X)

    # Decode the full sequence
    log_prob, state_seq = model.decode(X, algorithm="viterbi")

    # Label regimes
    regime_labels = _label_regimes(model, feature_cols)
    label_seq     = [regime_labels[s] for s in state_seq]

    # State statistics
    state_counts = {regime_labels[s]: int((state_seq == s).sum()) for s in range(N_STATES)}
    state_pcts   = {k: round(v / len(state_seq) * 100, 1) for k, v in state_counts.items()}

    # Expected regime durations (inverse of off-diagonal transition probability)
    durations = {}
    for s in range(N_STATES):
        p_stay = model.transmat_[s, s]
        dur    = round(1.0 / max(1.0 - p_stay, 1e-6), 1)
        durations[regime_labels[s]] = dur

    meta = {
        "feature_cols":    feature_cols,
        "n_states":        N_STATES,
        "n_obs":           len(df),
        "date_min":        str(df["date"].min().date()),
        "date_max":        str(df["date"].max().date()),
        "regime_labels":   {str(k): v for k, v in regime_labels.items()},
        "state_pcts":      state_pcts,
        "state_durations": durations,
        "log_prob":        round(float(log_prob), 2),
        "trained_at":      datetime.utcnow().isoformat(),
    }

    _save(model, meta)
    return meta


# ── Inference ────────────────────────────────────────────────────────────────

def predict_regime(features_df: pd.DataFrame) -> dict | None:
    """
    Classify the current market regime using the trained HMM.

    Returns dict with: regime, regime_color, confidence, recent_history (last 30 days),
    or None if model is not trained.
    """
    meta = load_meta()
    if not meta:
        return None

    model = _load()
    if model is None:
        return None

    feature_cols  = meta["feature_cols"]
    regime_labels = {int(k): v for k, v in meta["regime_labels"].items()}

    df = features_df.copy().dropna(subset=feature_cols).sort_values("date")
    if len(df) < 5:
        return None

    X = df[feature_cols].values.astype(np.float64)

    try:
        _, state_seq  = model.decode(X, algorithm="viterbi")
        # Posterior probabilities for the latest observation
        log_posteriors = model.predict_proba(X[-1:])
        latest_state   = int(state_seq[-1])
        confidence     = float(log_posteriors[0, latest_state])
    except Exception:
        return None

    regime = regime_labels.get(latest_state, "Unknown")

    # Build recent history (last 30 days)
    n_hist = min(30, len(df))
    history = pd.DataFrame({
        "date":   df["date"].iloc[-n_hist:].values,
        "state":  state_seq[-n_hist:],
        "regime": [regime_labels.get(int(s), "Unknown") for s in state_seq[-n_hist:]],
    })

    return {
        "regime":        regime,
        "regime_color":  REGIME_COLORS.get(regime, "#8b949e"),
        "confidence":    round(confidence * 100, 1),
        "latest_date":   str(df["date"].iloc[-1].date()),
        "history":       history,
        "meta":          meta,
    }


def get_regime_history(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the full classified regime history as a DataFrame for charting.
    Columns: date, regime, color.
    """
    meta = load_meta()
    if not meta:
        return pd.DataFrame()

    model = _load()
    if model is None:
        return pd.DataFrame()

    feature_cols  = meta["feature_cols"]
    regime_labels = {int(k): v for k, v in meta["regime_labels"].items()}

    df = features_df.copy().dropna(subset=feature_cols).sort_values("date")
    if df.empty:
        return pd.DataFrame()

    X = df[feature_cols].values.astype(np.float64)
    try:
        _, state_seq = model.decode(X, algorithm="viterbi")
    except Exception:
        return pd.DataFrame()

    out = pd.DataFrame({
        "date":   df["date"].values,
        "state":  state_seq,
        "regime": [regime_labels.get(int(s), "Unknown") for s in state_seq],
    })
    out["color"] = out["regime"].map(REGIME_COLORS).fillna("#8b949e")
    return out
