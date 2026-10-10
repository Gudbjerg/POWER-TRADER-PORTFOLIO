"""
TTF versus JKM inter-basin LNG price spread (Layer 3 "TTF vs JKM" tab and the
Layer 1 LNG Terminals KPI; roadmap 3B).

Pure functions only (no I/O, no Streamlit), unit-tested in
tests/test_lng_arbitrage.py.

Units and conventions
    TTF      ICE Endex front month (Yahoo Finance TTF=F), EUR/MWh.
    JKM      Platts JKM LNG futures front month (Yahoo Finance JKM=F), USD/MMBtu.
    EURUSD   EURUSD=X close, quoted as USD per EUR.
    Conversion: JKM[EUR/MWh] = JKM[USD/MMBtu] / EURUSD / 0.293071,
                since 1 MMBtu = 0.293071 MWh.
    Spread:  TTF minus JKM (EUR/MWh). Positive: Europe premium.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import (
    LNG_AVG_WINDOW_DAYS,
    LNG_FX_MAX_AGE_DAYS,
    LNG_PCTL_MIN_OBS,
    LNG_PCTL_WINDOW_DAYS,
)

MWH_PER_MMBTU = 0.293071   # 1 MMBtu = 293.071 kWh


def usd_mmbtu_to_eur_mwh(price_usd_mmbtu, eurusd):
    """
    Convert a gas price from USD/MMBtu to EUR/MWh.

    eurusd is the EUR/USD rate quoted as USD per EUR. Non-positive or missing
    FX rates yield NaN. Accepts scalars, numpy arrays or aligned pandas Series
    (a Series input returns a Series on the same index).
    """
    if isinstance(price_usd_mmbtu, pd.Series) or isinstance(eurusd, pd.Series):
        price = (pd.to_numeric(price_usd_mmbtu, errors="coerce")
                 if isinstance(price_usd_mmbtu, pd.Series) else float(price_usd_mmbtu))
        if isinstance(eurusd, pd.Series):
            fx = pd.to_numeric(eurusd, errors="coerce")
            fx = fx.where(fx > 0)
        else:
            fx = float(eurusd) if float(eurusd) > 0 else np.nan
        return price / fx / MWH_PER_MMBTU
    price = np.asarray(price_usd_mmbtu, dtype=float)
    fx = np.asarray(eurusd, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(fx > 0, price / fx / MWH_PER_MMBTU, np.nan)
    return float(out) if out.ndim == 0 else out


def _to_daily_frame(series: pd.Series, name: str) -> pd.DataFrame:
    """Date-indexed Series to DataFrame[date, name] with tz-naive midnight dates (last value per date)."""
    if series is None or len(series) == 0:
        return pd.DataFrame(columns=["date", name])
    idx = pd.DatetimeIndex(pd.to_datetime(series.index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    frame = pd.DataFrame({"date": idx.normalize(), name: pd.to_numeric(series, errors="coerce").to_numpy()})
    frame = frame.dropna(subset=[name]).sort_values("date")
    return frame.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)


def align_ttf_jkm(
    ttf: pd.Series,
    jkm_usd_mmbtu: pd.Series,
    eurusd: pd.Series,
    fx_max_age_days: int = LNG_FX_MAX_AGE_DAYS,
) -> pd.DataFrame:
    """
    Align date-indexed TTF (EUR/MWh), JKM (USD/MMBtu) and EURUSD (USD per EUR)
    series and compute the spread.

    - TTF and JKM are inner-joined on date: a spread exists only where both
      settle; JKM gaps are never interpolated.
    - EURUSD is matched as of the date: the latest FX close on or before it,
      at most `fx_max_age_days` calendar days old (FX and gas exchange holiday
      calendars differ). Dates without a usable FX rate are dropped.

    Returns DataFrame[date, ttf, jkm_usd_mmbtu, eurusd, jkm_eur_mwh, spread],
    spread = ttf minus jkm_eur_mwh, sorted by date.
    """
    columns = ["date", "ttf", "jkm_usd_mmbtu", "eurusd", "jkm_eur_mwh", "spread"]
    t = _to_daily_frame(ttf, "ttf")
    j = _to_daily_frame(jkm_usd_mmbtu, "jkm_usd_mmbtu")
    f = _to_daily_frame(eurusd, "eurusd")
    if t.empty or j.empty or f.empty:
        return pd.DataFrame(columns=columns)
    merged = t.merge(j, on="date", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=columns)
    merged = pd.merge_asof(
        merged.sort_values("date"),
        f.sort_values("date"),
        on="date",
        direction="backward",
        tolerance=pd.Timedelta(days=int(fx_max_age_days)),
    )
    merged = merged.dropna(subset=["eurusd"])
    merged = merged[merged["eurusd"] > 0]
    if merged.empty:
        return pd.DataFrame(columns=columns)
    merged["jkm_eur_mwh"] = usd_mmbtu_to_eur_mwh(merged["jkm_usd_mmbtu"], merged["eurusd"])
    merged["spread"] = merged["ttf"] - merged["jkm_eur_mwh"]
    return merged[columns].reset_index(drop=True)


def rolling_percentile(
    series: pd.Series,
    window_days: int = LNG_PCTL_WINDOW_DAYS,
    min_obs: int = LNG_PCTL_MIN_OBS,
) -> pd.Series:
    """
    Percentile rank (0 to 100) of each observation within the trailing window
    of `window_days` calendar days ending on, and including, its own date.

    Ties use the mid-rank convention: (count below + 0.5 x count equal) / n.
    Only past and current observations enter each window (no look-ahead).
    NaN until the window holds at least `min_obs` observations. The input
    must be indexed by dates.
    """
    if series is None or len(series) == 0:
        return pd.Series(dtype=float)
    s = series.sort_index()
    dates = pd.DatetimeIndex(pd.to_datetime(s.index)).to_numpy()
    values = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    window = np.timedelta64(int(window_days), "D")
    for i in range(len(values)):
        x = values[i]
        if np.isnan(x):
            continue
        lo = int(np.searchsorted(dates, dates[i] - window, side="right"))
        w = values[lo:i + 1]
        w = w[~np.isnan(w)]
        if len(w) < min_obs:
            continue
        below = np.count_nonzero(w < x)
        equal = np.count_nonzero(w == x)
        out[i] = (below + 0.5 * equal) / len(w) * 100.0
    return pd.Series(out, index=s.index, name="percentile")


def ordinal(n: int) -> str:
    """English ordinal for a non-negative integer: 1st, 2nd, 3rd, 4th, 11th, 12th, 13th, 21st, 100th."""
    n = int(n)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def spread_regime(spread: float) -> str:
    """Sign-based label: 'Europe premium' (TTF above JKM), 'Asia premium', 'Parity' or 'n/a'."""
    if spread is None or not np.isfinite(spread):
        return "n/a"
    if spread > 0:
        return "Europe premium"
    if spread < 0:
        return "Asia premium"
    return "Parity"


def spread_summary(
    df: pd.DataFrame,
    avg_window_days: int = LNG_AVG_WINDOW_DAYS,
    pct_window_days: int = LNG_PCTL_WINDOW_DAYS,
    min_pct_obs: int = LNG_PCTL_MIN_OBS,
) -> dict:
    """
    Latest-observation summary of an align_ttf_jkm() frame.

    avg: mean spread over the trailing `avg_window_days` calendar days;
    percentile: rolling_percentile of the latest spread within the trailing
    `pct_window_days` calendar days; share_positive: share of settlements in
    that window with a positive spread. Returns {} for an empty frame.
    """
    if df is None or df.empty:
        return {}
    data = df.sort_values("date").reset_index(drop=True)
    last = data.iloc[-1]
    end = pd.Timestamp(last["date"])
    avg_slice = data.loc[data["date"] > end - pd.Timedelta(days=int(avg_window_days)), "spread"]
    win_slice = data.loc[data["date"] > end - pd.Timedelta(days=int(pct_window_days)), "spread"]
    pct = rolling_percentile(
        data.set_index("date")["spread"], window_days=pct_window_days, min_obs=min_pct_obs,
    )
    return {
        "latest_date": end,
        "ttf": float(last["ttf"]),
        "jkm_usd_mmbtu": float(last["jkm_usd_mmbtu"]),
        "eurusd": float(last["eurusd"]),
        "jkm_eur_mwh": float(last["jkm_eur_mwh"]),
        "spread": float(last["spread"]),
        "regime": spread_regime(float(last["spread"])),
        "avg": float(avg_slice.mean()) if len(avg_slice) else float("nan"),
        "n_avg": int(len(avg_slice)),
        "percentile": float(pct.iloc[-1]) if len(pct) else float("nan"),
        "n_window": int(len(win_slice)),
        "share_positive": float((win_slice > 0).mean()) if len(win_slice) else float("nan"),
        "window_min": float(win_slice.min()) if len(win_slice) else float("nan"),
        "window_max": float(win_slice.max()) if len(win_slice) else float("nan"),
    }
