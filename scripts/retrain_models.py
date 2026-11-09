"""
Retrain the Layer 4 models on the latest feature matrix and write their JSON files.

Runs in the `retrain` GitHub workflow (quarterly, or on demand) with ENTSOE_API_KEY and
AGSI_API_KEY set. Steps: refresh the feature matrix (starting from the copy the Space
ships, so only recent days are fetched), check that every input reaches the last weeks,
train the HMM and the LSTM exactly as the ML page's Retrain buttons do, and print a
summary. Exits non-zero, writing nothing new, when the data or a model fails a check,
so a bad run is never published.

Usage: python -m scripts.retrain_models
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

MAX_STALE_DAYS = 21   # an input whose last value is older than this blocks the retrain


def main() -> int:
    if not (os.getenv("ENTSOE_API_KEY") and os.getenv("AGSI_API_KEY")):
        print("ENTSOE_API_KEY and AGSI_API_KEY are required", file=sys.stderr)
        return 2

    from models.feature_assembly import assemble_features, get_hmm_feature_cols, get_lstm_feature_cols
    from models.hmm_model import train_hmm
    from models.lstm_model import train_lstm_full

    df = assemble_features(years=3)
    if df.empty:
        print("feature matrix is empty", file=sys.stderr)
        return 1
    last = df["date"].max()
    stale = {}
    for col in ("no2", "nl", "ttf", "storage_fill", "hydro_twh", "de_wind_gwh"):
        if col not in df.columns or df[col].notna().sum() == 0:
            stale[col] = "missing"
            continue
        age = (last - df.loc[df[col].notna(), "date"].max()).days
        if age > MAX_STALE_DAYS:
            stale[col] = f"{age} days old"
    print(f"feature matrix: {len(df)} trading days, {df['date'].min():%Y-%m-%d} to {last:%Y-%m-%d}")
    if stale:
        print(f"inputs too old or missing, not retraining: {stale}", file=sys.stderr)
        return 1

    hmm = train_hmm(df, get_hmm_feature_cols(df))
    if "error" in hmm:
        print(f"HMM failed: {hmm['error']}", file=sys.stderr)
        return 1
    lstm = train_lstm_full(df, get_lstm_feature_cols(df))
    if "error" in lstm:
        print(f"LSTM failed: {lstm['error']}", file=sys.stderr)
        return 1

    summary = {
        "hmm": {k: hmm.get(k) for k in ("n_obs", "date_min", "date_max", "state_pcts")},
        "lstm": {k: lstm.get(k) for k in ("test_mae_no2", "naive_mae_no2", "test_mae_nl",
                                          "naive_mae_nl", "n_train", "n_test", "trained_at")},
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
