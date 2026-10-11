"""
FI minus IT-North day-ahead spread statistics (Layer 1, Prices tab; roadmap 3A).

Pure functions only (no I/O, no Streamlit), unit-tested in
tests/test_fi_it_spread.py. Inputs are daily delivery-day averages in EUR/MWh
as produced by data/italy_prices.py.

Spread convention: spread_t = FI_t minus IT-North_t. A negative value means
Finland cleared below northern Italy on that delivery day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import FI_IT_AVG_WINDOW_DAYS, FI_IT_Z_MIN_OBS, FI_IT_Z_WINDOW_DAYS


def _normalise_dates(values: pd.Series) -> pd.Series:
    out = pd.to_datetime(values)
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_localize(None)
    return out.dt.normalize()


def build_spread_frame(fi: pd.DataFrame, it_north: pd.DataFrame) -> pd.DataFrame:
    """
    Inner-join daily FI (columns: date, fi) and IT-North (columns: date,
    it_north) prices on delivery date and add spread = fi minus it_north.

    Dates are normalised to tz-naive midnights; duplicate dates within an input
    are averaged; rows with a missing price on either side are dropped.
    Returns DataFrame[date, fi, it_north, spread] sorted by date.
    """
    columns = ["date", "fi", "it_north", "spread"]
    if fi is None or it_north is None or fi.empty or it_north.empty:
        return pd.DataFrame(columns=columns)
    if not {"date", "fi"}.issubset(fi.columns) or not {"date", "it_north"}.issubset(it_north.columns):
        raise ValueError("expected columns fi[date, fi] and it_north[date, it_north]")

    left = fi[["date", "fi"]].copy()
    right = it_north[["date", "it_north"]].copy()
    left["date"] = _normalise_dates(left["date"])
    right["date"] = _normalise_dates(right["date"])
    left["fi"] = pd.to_numeric(left["fi"], errors="coerce")
    right["it_north"] = pd.to_numeric(right["it_north"], errors="coerce")
    left = left.groupby("date", as_index=False)["fi"].mean()
    right = right.groupby("date", as_index=False)["it_north"].mean()

    merged = left.merge(right, on="date", how="inner").dropna(subset=["fi", "it_north"])
    if merged.empty:
        return pd.DataFrame(columns=columns)
    merged["spread"] = merged["fi"] - merged["it_north"]
    return merged.sort_values("date").reset_index(drop=True)[columns]


def add_rolling_mean(
    df: pd.DataFrame,
    window_days: int = FI_IT_AVG_WINDOW_DAYS,
    min_periods: int = 10,
    column: str = "spread",
) -> pd.DataFrame:
    """
    Add `<column>_ma` holding the trailing calendar-day rolling mean (window
    `window_days` days, inclusive of the current day). Calendar-day windows keep
    the statistic honest when individual delivery days are missing.
    """
    out = df.copy()
    if out.empty:
        out[f"{column}_ma"] = pd.Series(dtype=float)
        return out
    series = out.set_index("date")[column].astype(float)
    rolled = series.rolling(f"{int(window_days)}D", min_periods=min_periods).mean()
    out[f"{column}_ma"] = rolled.to_numpy()
    return out


def spread_statistics(
    df: pd.DataFrame,
    avg_window_days: int = FI_IT_AVG_WINDOW_DAYS,
    z_window_days: int = FI_IT_Z_WINDOW_DAYS,
    min_z_obs: int = FI_IT_Z_MIN_OBS,
) -> dict:
    """
    Summary statistics of the latest FI minus IT-North spread.

    - avg: mean spread over the `avg_window_days` calendar days ending on the
      latest delivery day (inclusive); share_fi_below: share of those days with
      a negative spread.
    - z: (latest spread minus window mean) divided by the sample standard
      deviation (ddof=1) over the `z_window_days` calendar days ending on the
      latest delivery day (inclusive). Reported only when at least `min_z_obs`
      observations exist and the deviation is positive; otherwise NaN.
    - z_full_window: True when the available history spans the whole z window.

    Returns an empty dict for an empty frame.
    """
    if df is None or df.empty:
        return {}
    data = df.sort_values("date").reset_index(drop=True)
    last = data.iloc[-1]
    end = pd.Timestamp(last["date"])

    avg_mask = data["date"] > end - pd.Timedelta(days=int(avg_window_days))
    avg_slice = data.loc[avg_mask, "spread"].astype(float)

    z_mask = data["date"] > end - pd.Timedelta(days=int(z_window_days))
    z_slice = data.loc[z_mask, "spread"].astype(float)
    n_z = int(z_slice.notna().sum())
    z_mean = float(z_slice.mean()) if n_z else float("nan")
    z_std = float(z_slice.std(ddof=1)) if n_z > 1 else float("nan")
    latest = float(last["spread"])
    if n_z >= min_z_obs and np.isfinite(z_std) and z_std > 0:
        z = (latest - z_mean) / z_std
    else:
        z = float("nan")

    first_needed = end - pd.Timedelta(days=int(z_window_days) - 1)
    return {
        "latest_date": end,
        "latest_spread": latest,
        "latest_fi": float(last["fi"]),
        "latest_it_north": float(last["it_north"]),
        "avg": float(avg_slice.mean()) if len(avg_slice) else float("nan"),
        "n_avg": int(len(avg_slice)),
        "share_fi_below": float((avg_slice < 0).mean()) if len(avg_slice) else float("nan"),
        "n_fi_below": int((avg_slice < 0).sum()),
        "z": float(z),
        "z_mean": z_mean,
        "z_std": z_std,
        "n_z": n_z,
        "z_window_start": pd.Timestamp(data.loc[z_mask, "date"].iloc[0]) if n_z else None,
        "z_full_window": bool(pd.Timestamp(data["date"].iloc[0]) <= first_needed),
        "window_min": float(z_slice.min()) if n_z else float("nan"),
        "window_max": float(z_slice.max()) if n_z else float("nan"),
    }
