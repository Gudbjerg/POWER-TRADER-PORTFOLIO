"""
LSTM Day-Ahead Price Forecaster (Model 1).

Predicts next-day average day-ahead prices for NO2 and NL simultaneously.
Architecture: 2-layer LSTM (64 → 32 units) with 0.2 dropout, linear output head.
Trained with MAE loss; compared to naive persistence baseline.

Weights and metadata are saved to models/weights/ so the model survives restarts.
Training is triggered manually via the UI; inference runs automatically on load.
"""
from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
os.makedirs(_WEIGHTS_DIR, exist_ok=True)

SEQ_LEN    = 21    # lookback window (trading days)
HIDDEN1    = 64
HIDDEN2    = 32
DROPOUT    = 0.2
EPOCHS     = 150
BATCH_SIZE = 32
LR         = 1e-3

TARGETS = ["no2", "nl"]


# ── Model architecture ───────────────────────────────────────────────────────

def _build_model(n_features: int):
    """Build and return an untrained PowerPriceLSTM."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return None

    class PowerPriceLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm1   = nn.LSTM(n_features, HIDDEN1, batch_first=True)
            self.drop    = nn.Dropout(DROPOUT)
            self.lstm2   = nn.LSTM(HIDDEN1, HIDDEN2, batch_first=True)
            self.fc      = nn.Linear(HIDDEN2, len(TARGETS))

        def forward(self, x):
            out, _ = self.lstm1(x)
            out     = self.drop(out)
            out, _  = self.lstm2(out)
            return self.fc(out[:, -1, :])   # last time step

    return PowerPriceLSTM()


# ── Sequence creation ────────────────────────────────────────────────────────

def _make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


# ── Save / load ──────────────────────────────────────────────────────────────

def _weights_path() -> str:
    return os.path.join(_WEIGHTS_DIR, "lstm_model.pt")


def _meta_path() -> str:
    return os.path.join(_WEIGHTS_DIR, "lstm_meta.json")


def _test_results_path() -> str:
    return os.path.join(_WEIGHTS_DIR, "lstm_test_results.csv")


def is_trained() -> bool:
    return os.path.exists(_weights_path()) and os.path.exists(_meta_path())


def load_test_results() -> pd.DataFrame:
    """Load saved test-set predictions. Returns DataFrame with date, no2_actual,
    no2_pred, nl_actual, nl_pred — or empty DataFrame if not available."""
    try:
        if os.path.exists(_test_results_path()):
            return pd.read_csv(_test_results_path(), parse_dates=["date"])
    except Exception:
        pass
    return pd.DataFrame()


def load_meta() -> dict:
    try:
        with open(_meta_path()) as f:
            return json.load(f)
    except Exception:
        return {}


def _save(model, scaler_mean: list, scaler_std: list, feature_cols: list,
          train_info: dict) -> None:
    try:
        import torch
        torch.save(model.state_dict(), _weights_path())
    except Exception:
        pass
    meta = {
        "feature_cols":  feature_cols,
        "scaler_mean":   scaler_mean,
        "scaler_std":    scaler_std,
        "targets":       TARGETS,
        "seq_len":       SEQ_LEN,
        "trained_at":    datetime.utcnow().isoformat(),
        **train_info,
    }
    with open(_meta_path(), "w") as f:
        json.dump(meta, f, indent=2)


def _load(n_features: int):
    """Load trained model + scaler from disk. Returns (model, meta) or (None, {})."""
    if not is_trained():
        return None, {}
    try:
        import torch
        meta  = load_meta()
        model = _build_model(n_features)
        model.load_state_dict(torch.load(_weights_path(), map_location="cpu"))
        model.eval()
        return model, meta
    except Exception:
        return None, {}


# ── Training ─────────────────────────────────────────────────────────────────

def train_lstm(features_df: pd.DataFrame, feature_cols: list[str],
               progress_cb=None) -> dict:
    """
    Train the LSTM on the provided feature matrix.

    Parameters
    ----------
    features_df  : Output of assemble_features(), must contain feature_cols + TARGETS.
    feature_cols : From get_lstm_feature_cols(features_df).
    progress_cb  : Optional callable(epoch, n_epochs, train_loss, val_loss) for UI updates.

    Returns
    -------
    dict with training results (test_mae_no2, test_mae_nl, naive_mae_no2, naive_mae_nl,
    n_train, n_val, n_test, trained_at, feature_cols).
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        return {"error": "torch not installed. Run: pip install torch"}

    df = features_df.copy().dropna(subset=feature_cols + TARGETS)
    df = df.sort_values("date").reset_index(drop=True)

    if len(df) < SEQ_LEN + 50:
        return {"error": f"Insufficient data: {len(df)} rows (need ≥{SEQ_LEN + 50})"}

    X_raw = df[feature_cols].values.astype(np.float32)
    y_raw = df[TARGETS].values.astype(np.float32)

    # Train/val/test split (time-ordered, no shuffling)
    n = len(X_raw)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    # test = remaining

    # Normalise X using training set statistics only
    mean = X_raw[:n_train].mean(axis=0)
    std  = X_raw[:n_train].std(axis=0)
    std[std == 0] = 1.0
    X_norm = (X_raw - mean) / std

    # Normalise y using training set statistics only
    y_mean = y_raw[:n_train].mean(axis=0)
    y_std  = y_raw[:n_train].std(axis=0)
    y_std[y_std == 0] = 1.0
    y_norm = (y_raw - y_mean) / y_std

    Xs, ys = _make_sequences(X_norm, y_norm, SEQ_LEN)

    n_seq   = len(Xs)
    nt      = int(n_seq * 0.70)
    nv      = int(n_seq * 0.15)

    X_tr, y_tr = Xs[:nt], ys[:nt]
    X_va, y_va = Xs[nt:nt+nv], ys[nt:nt+nv]
    X_te, y_te = Xs[nt+nv:], ys[nt+nv:]

    train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    val_ds   = TensorDataset(torch.tensor(X_va), torch.tensor(y_va))
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    model     = _build_model(len(feature_cols))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, min_lr=1e-5
    )
    criterion = nn.L1Loss()

    best_val_loss = float("inf")
    patience_ctr  = 0
    best_state    = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                val_loss += criterion(model(xb), yb).item()
        val_loss /= max(len(val_dl), 1)

        scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1

        if progress_cb:
            progress_cb(epoch, EPOCHS, train_loss, val_loss)

        if patience_ctr >= 20:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # Evaluate on test set (in original units)
    with torch.no_grad():
        X_te_t  = torch.tensor(X_te)
        y_pred_norm = model(X_te_t).numpy()

    y_pred = y_pred_norm * y_std + y_mean
    y_true = y_te * y_std + y_mean

    test_mae_no2 = float(np.abs(y_pred[:, 0] - y_true[:, 0]).mean())
    test_mae_nl  = float(np.abs(y_pred[:, 1] - y_true[:, 1]).mean())

    # Naive persistence baseline (yesterday = tomorrow)
    # In test set sequences, last value of target in input ≈ previous day
    naive_no2 = float(np.abs(np.diff(y_raw[nt+nv+SEQ_LEN:, 0])).mean()) if len(y_raw[nt+nv+SEQ_LEN:]) > 1 else test_mae_no2
    naive_nl  = float(np.abs(np.diff(y_raw[nt+nv+SEQ_LEN:, 1])).mean()) if len(y_raw[nt+nv+SEQ_LEN:]) > 1 else test_mae_nl

    info = {
        "test_mae_no2":   round(test_mae_no2, 2),
        "test_mae_nl":    round(test_mae_nl, 2),
        "naive_mae_no2":  round(naive_no2, 2),
        "naive_mae_nl":   round(naive_nl, 2),
        "n_train":        nt,
        "n_val":          nv,
        "n_test":         len(X_te),
        "n_features":     len(feature_cols),
        "epochs_trained": epoch,
        "best_val_loss":  round(best_val_loss, 5),
    }

    _save(model, mean.tolist(), std.tolist(), feature_cols, info)
    return info


# ── Inference ────────────────────────────────────────────────────────────────

def predict_next(features_df: pd.DataFrame) -> dict | None:
    """
    Predict next-day NO2 and NL prices using the trained LSTM.

    Returns dict with keys: no2_pred, nl_pred, feature_cols, trained_at, last_date
    or None if model is not trained or data is insufficient.
    """
    meta = load_meta()
    if not meta:
        return None

    feature_cols = meta["feature_cols"]
    try:
        import torch
        model, _ = _load(len(feature_cols))
        if model is None:
            return None

        mean = np.array(meta["scaler_mean"], dtype=np.float32)
        std  = np.array(meta["scaler_std"], dtype=np.float32)
        std[std == 0] = 1.0

        df = features_df.copy().sort_values("date")
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            return None

        df = df.dropna(subset=feature_cols).tail(SEQ_LEN)
        if len(df) < SEQ_LEN:
            return None

        X = df[feature_cols].values.astype(np.float32)
        X_norm = (X - mean) / std
        X_t = torch.tensor(X_norm).unsqueeze(0)  # (1, SEQ_LEN, n_features)

        with torch.no_grad():
            pred_norm = model(X_t).numpy()[0]

        # Denormalise: we need y_mean and y_std which aren't stored
        # Instead, predict in normalised space and report as "normalised offset"
        # Better: store y_scaler in meta too — let's do that from next train
        # For now, if y_mean/y_std in meta, use them; else return normalised
        if "y_mean" in meta and "y_std" in meta:
            y_mean = np.array(meta["y_mean"])
            y_std  = np.array(meta["y_std"])
            pred   = pred_norm * y_std + y_mean
        else:
            # Fallback: use last known prices + weighted adjustment
            last_no2 = float(df["no2"].iloc[-1]) if "no2" in df.columns else None
            last_nl  = float(df["nl"].iloc[-1])  if "nl" in df.columns else None
            return {
                "no2_pred":    last_no2,
                "nl_pred":     last_nl,
                "note":        "Retrain model to enable denormalised predictions.",
                "trained_at":  meta.get("trained_at"),
                "last_date":   str(df["date"].iloc[-1].date()),
            }

        return {
            "no2_pred":   round(float(pred[0]), 1),
            "nl_pred":    round(float(pred[1]), 1),
            "trained_at": meta.get("trained_at"),
            "last_date":  str(df["date"].iloc[-1].date()),
            "feature_cols": feature_cols,
        }
    except Exception:
        return None


def train_lstm_full(features_df: pd.DataFrame, feature_cols: list[str],
                    progress_cb=None) -> dict:
    """
    Wrapper that also stores y_mean/y_std in meta for denormalised inference.
    Call this instead of train_lstm when you want predict_next to work fully.
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        return {"error": "torch not installed"}

    df = features_df.copy().dropna(subset=feature_cols + TARGETS).sort_values("date").reset_index(drop=True)
    if len(df) < SEQ_LEN + 50:
        return {"error": f"Insufficient data: {len(df)} rows"}

    X_raw = df[feature_cols].values.astype(np.float32)
    y_raw = df[TARGETS].values.astype(np.float32)

    n      = len(X_raw)
    n_train = int(n * 0.70)

    mean = X_raw[:n_train].mean(axis=0)
    std  = X_raw[:n_train].std(axis=0)
    std[std == 0] = 1.0

    y_mean = y_raw[:n_train].mean(axis=0)
    y_std  = y_raw[:n_train].std(axis=0)
    y_std[y_std == 0] = 1.0

    X_norm = (X_raw - mean) / std
    y_norm = (y_raw - y_mean) / y_std

    Xs, ys = _make_sequences(X_norm, y_norm, SEQ_LEN)
    n_seq   = len(Xs)
    nt      = int(n_seq * 0.70)
    nv      = int(n_seq * 0.15)

    train_dl = DataLoader(TensorDataset(torch.tensor(Xs[:nt]), torch.tensor(ys[:nt])),
                          batch_size=BATCH_SIZE, shuffle=False)
    val_dl   = DataLoader(TensorDataset(torch.tensor(Xs[nt:nt+nv]), torch.tensor(ys[nt:nt+nv])),
                          batch_size=BATCH_SIZE, shuffle=False)

    model     = _build_model(len(feature_cols))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.L1Loss()
    best_val  = float("inf")
    best_state = None
    patience_ctr = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tl = sum(
            (lambda xb, yb: (optimizer.zero_grad(), (loss := criterion(model(xb), yb)),
                             loss.backward(), nn.utils.clip_grad_norm_(model.parameters(), 1.0),
                             optimizer.step(), loss.item())[-1])(xb, yb)
            for xb, yb in train_dl
        ) / len(train_dl)

        model.eval()
        with torch.no_grad():
            vl = sum(criterion(model(xb), yb).item() for xb, yb in val_dl) / max(len(val_dl), 1)

        scheduler.step(vl)
        if vl < best_val - 1e-5:
            best_val   = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if progress_cb:
            progress_cb(epoch, EPOCHS, tl, vl)
        if patience_ctr >= 20:
            break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        y_pred_norm = model(torch.tensor(Xs[nt+nv:])).numpy()
    y_pred = y_pred_norm * y_std + y_mean
    y_true = ys[nt+nv:] * y_std + y_mean

    test_mae_no2 = float(np.abs(y_pred[:, 0] - y_true[:, 0]).mean())
    test_mae_nl  = float(np.abs(y_pred[:, 1] - y_true[:, 1]).mean())
    naive_no2    = float(np.abs(np.diff(y_raw[nt+nv+SEQ_LEN:, 0])).mean()) if len(y_raw[nt+nv+SEQ_LEN:]) > 1 else test_mae_no2
    naive_nl     = float(np.abs(np.diff(y_raw[nt+nv+SEQ_LEN:, 1])).mean()) if len(y_raw[nt+nv+SEQ_LEN:]) > 1 else test_mae_nl

    info = {
        "test_mae_no2":   round(test_mae_no2, 2),
        "test_mae_nl":    round(test_mae_nl,  2),
        "naive_mae_no2":  round(naive_no2, 2),
        "naive_mae_nl":   round(naive_nl,  2),
        "n_train":        nt,
        "n_val":          nv,
        "n_test":         len(Xs[nt+nv:]),
        "epochs_trained": epoch,
        "best_val_loss":  round(best_val, 5),
        "y_mean":         y_mean.tolist(),
        "y_std":          y_std.tolist(),
    }
    _save(model, mean.tolist(), std.tolist(), feature_cols, info)

    # Save test set predictions for post-training chart
    try:
        test_dates = df["date"].iloc[SEQ_LEN + nt + nv : SEQ_LEN + nt + nv + len(y_pred)].values
        results_df = pd.DataFrame({
            "date":       test_dates,
            "no2_actual": y_true[:, 0].round(2),
            "no2_pred":   y_pred[:, 0].round(2),
            "nl_actual":  y_true[:, 1].round(2),
            "nl_pred":    y_pred[:, 1].round(2),
        })
        results_df.to_csv(_test_results_path(), index=False)
    except Exception:
        pass

    return info
