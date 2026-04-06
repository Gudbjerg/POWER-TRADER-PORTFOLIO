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

    All features are normalised to [-1, 1] relative to their range across
    states before scoring, so scale differences (hydro_pct in 0-100 vs
    z-scored features in ±2) do not distort the regime assignment.

    Regime signatures (higher = more characteristic):
      - Hydro-driven:       high hydro, low NL price, wide negative spread
      - Gas-driven:         high TTF, high NL, low hydro
      - Renewables-driven:  high wind, low volatility, low NL price
      - Geopolitical stress:high volatility, high TTF, low storage
    """
    means = model.means_   # shape (n_states, n_features)
    col_idx = {c: i for i, c in enumerate(feature_cols)}

    def _norm(col) -> np.ndarray:
        """Normalise a feature across state means to [-1, 1]."""
        if col not in col_idx:
            return np.zeros(N_STATES)
        vals = means[:, col_idx[col]].astype(float)
        vmin, vmax = vals.min(), vals.max()
        rng = vmax - vmin
        if rng < 1e-9:
            return np.zeros(N_STATES)
        return (vals - vmin) / rng * 2 - 1  # [-1, 1]

    hydro_n  = _norm("hydro_pct")
    ttf_n    = _norm("ttf_z")
    nl_n     = _norm("nl_z")
    spread_n = _norm("spread_z")
    stor_n   = _norm("storage_dev")
    vol_n    = _norm("no2_vol5_z")
    wind_n   = _norm("wind_de_z")

    REGIME_ORDER = ["Hydro-driven", "Gas-driven", "Renewables-driven", "Geopolitical stress"]

    def _scores(s: int) -> dict[str, float]:
        return {
            "Hydro-driven":        hydro_n[s] * 0.5 - nl_n[s] * 0.3 - spread_n[s] * 0.2,
            "Gas-driven":          ttf_n[s] * 0.4 + nl_n[s] * 0.3 - hydro_n[s] * 0.3,
            "Renewables-driven":   wind_n[s] * 0.4 - vol_n[s] * 0.3 - nl_n[s] * 0.3,
            "Geopolitical stress": vol_n[s] * 0.4 + ttf_n[s] * 0.3 - stor_n[s] * 0.3,
        }

    # Assign each state its preferred label
    labels = {s: max(_scores(s), key=_scores(s).get) for s in range(N_STATES)}

    # Resolve ties: state with the strongest top-score wins its preferred regime
    used: set[str] = set()
    final: dict[int, str] = {}
    order = sorted(range(N_STATES), key=lambda s: -max(_scores(s).values()))
    for s in order:
        preferred = labels[s]
        if preferred not in used:
            final[s] = preferred
            used.add(preferred)
        else:
            remaining = [r for r in REGIME_ORDER if r not in used]
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
        _, state_seq     = model.decode(X, algorithm="viterbi")
        # Forward-backward posteriors over the full sequence, then take the
        # last step.  Running predict_proba on a single observation omits
        # transition context and can give 0% for the Viterbi-decoded state.
        posteriors_full  = model.predict_proba(X)    # shape (n_obs, n_states)
        latest_state     = int(state_seq[-1])
        confidence       = float(posteriors_full[-1, latest_state])
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
