"""
Price Spike Detector.

Computes rolling 30-day z-scores on daily day-ahead prices for each bidding zone.
Identifies and ranks zones currently showing anomalous price levels.
"""
import numpy as np
import pandas as pd

ALERT_Z    = 2.5   # hard alert threshold
WARNING_Z  = 1.8   # soft warning threshold
WINDOW     = 30    # rolling window in trading days


def compute_zscores(spot_df: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """
    Compute rolling z-scores of day-ahead prices by zone.

    Parameters
    ----------
    spot_df : pd.DataFrame
        Long-format DataFrame with columns: date, zone, price_eur_mwh.
    window : int
        Rolling window in days (default: 30).

    Returns
    -------
    pd.DataFrame with columns: date, zone, price_eur_mwh, rolling_mean,
                                rolling_std, z_score, signal
        signal values: 'alert', 'warn', 'normal'
    """
    if spot_df.empty:
        return pd.DataFrame()

    records = []
    for zone in sorted(spot_df["zone"].unique()):
        sub = spot_df[spot_df["zone"] == zone].copy()
        sub = sub.sort_values("date").reset_index(drop=True)
        sub["rolling_mean"] = sub["price_eur_mwh"].rolling(window, min_periods=window // 2).mean()
        sub["rolling_std"]  = sub["price_eur_mwh"].rolling(window, min_periods=window // 2).std()
        sub["z_score"] = (sub["price_eur_mwh"] - sub["rolling_mean"]) / sub["rolling_std"].replace(0, np.nan)
        sub["zone"] = zone
        records.append(sub)

    df = pd.concat(records, ignore_index=True)

    def _signal(z):
        if pd.isna(z):
            return "normal"
        if abs(z) >= ALERT_Z:
            return "alert"
        if abs(z) >= WARNING_Z:
            return "warn"
        return "normal"

    df["signal"] = df["z_score"].apply(_signal)
    return df


def latest_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the most recent z-score reading per zone, sorted by |z_score| descending.

    Parameters
    ----------
    df : pd.DataFrame
        Output of compute_zscores().

    Returns
    -------
    pd.DataFrame with one row per zone, sorted by |z_score| descending.
    """
    if df.empty:
        return pd.DataFrame()

    latest_date = df["date"].max()
    latest = df[df["date"] == latest_date].copy()
    latest["abs_z"] = latest["z_score"].abs()
    latest = latest.sort_values("abs_z", ascending=False).drop(columns=["abs_z"])
    return latest.reset_index(drop=True)
