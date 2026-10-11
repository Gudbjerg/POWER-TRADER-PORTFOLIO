"""
JKM LNG and EUR/USD daily closes for the TTF vs JKM spread (roadmap 3B).

Sources (Yahoo Finance via yfinance, no key required):
    JKM=F     Platts JKM LNG futures, continuous front month, USD/MMBtu. A
              front-month settlement series that can contain missing days.
    EURUSD=X  EUR/USD close, quoted as USD per EUR.
    TTF=F     ICE Endex TTF front month, EUR/MWh, through the existing loader
              data.prices.fetch_ttf_prices.

load_ttf_jkm() returns the aligned frame (models.lng_arbitrage.align_ttf_jkm)
with the rolling one-year percentile attached, cached for both the Layer 1
LNG Terminals KPI and the Layer 3 "TTF vs JKM" tab.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

from config.settings import (
    LNG_FX_MAX_AGE_DAYS,
    LNG_HISTORY_DAYS,
    LNG_PCTL_MIN_OBS,
    LNG_PCTL_WINDOW_DAYS,
)
from models.lng_arbitrage import align_ttf_jkm, rolling_percentile

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

JKM_TICKER = "JKM=F"
EURUSD_TICKER = "EURUSD=X"

# Plausibility bounds used to reject unit or feed errors (not market views).
_JKM_MEDIAN_BOUNDS = (1.0, 200.0)      # USD/MMBtu
_EURUSD_MEDIAN_BOUNDS = (0.5, 2.0)     # USD per EUR


def _yf_close(ticker: str, start: str, end: str, attempts: int = 3) -> pd.Series:
    """Daily closes indexed by tz-naive dates; empty Series when unavailable."""
    if not YFINANCE_AVAILABLE:
        return pd.Series(dtype=float)
    for attempt in range(attempts):
        try:
            hist = yf.Ticker(ticker).history(start=start, end=end)
            if hist is None or hist.empty or "Close" not in hist.columns:
                return pd.Series(dtype=float)
            close = pd.to_numeric(hist["Close"], errors="coerce")
            idx = pd.DatetimeIndex(pd.to_datetime(close.index))
            if idx.tz is not None:
                idx = idx.tz_localize(None)
            close.index = idx.normalize()
            close = close[close > 0].dropna()
            return close[~close.index.duplicated(keep="last")].sort_index()
        except Exception:
            if attempt < attempts - 1:
                time.sleep(2.0 * (attempt + 1))
    return pd.Series(dtype=float)


@st.cache_data(ttl=7200, show_spinner=False)
def fetch_jkm_and_fx(days: int = LNG_HISTORY_DAYS) -> dict:
    """
    JKM front month (USD/MMBtu) and EURUSD (USD per EUR) daily closes.
    Returns dict: jkm (Series), eurusd (Series), error (str or None), fetched_at.
    """
    end = datetime.now(timezone.utc).date()
    start = (end - timedelta(days=int(days))).isoformat()
    end_s = end.isoformat()   # exclusive: completed sessions only, as in data.prices

    jkm = _yf_close(JKM_TICKER, start, end_s)
    fx = _yf_close(EURUSD_TICKER, start, end_s)
    errors = []
    if jkm.empty:
        errors.append(f"{JKM_TICKER} unavailable")
    elif not (_JKM_MEDIAN_BOUNDS[0] <= float(jkm.median()) <= _JKM_MEDIAN_BOUNDS[1]):
        errors.append(f"{JKM_TICKER} rejected (implausible units)")
        jkm = pd.Series(dtype=float)
    if fx.empty:
        errors.append(f"{EURUSD_TICKER} unavailable")
    elif not (_EURUSD_MEDIAN_BOUNDS[0] <= float(fx.median()) <= _EURUSD_MEDIAN_BOUNDS[1]):
        errors.append(f"{EURUSD_TICKER} rejected (implausible quote)")
        fx = pd.Series(dtype=float)
    return {
        "jkm": jkm,
        "eurusd": fx,
        "error": "; ".join(errors) if errors else None,
        "fetched_at": datetime.now(timezone.utc).replace(tzinfo=None),
    }


@st.cache_data(ttl=3600, show_spinner=False)
def load_ttf_jkm(days: int = LNG_HISTORY_DAYS) -> dict:
    """
    Aligned TTF vs JKM frame on an EUR/MWh basis with the rolling percentile.

    Returns dict:
        frame      : DataFrame[date, ttf, jkm_usd_mmbtu, eurusd, jkm_eur_mwh,
                     spread, percentile] (empty when a series is missing)
        error      : diagnostic string or None
        fetched_at : datetime (UTC)
    """
    from data.prices import fetch_ttf_prices

    ttf_df = fetch_ttf_prices(days=int(days))
    raw = fetch_jkm_and_fx(days=int(days))
    errors = [raw["error"]] if raw.get("error") else []
    if ttf_df is None or ttf_df.empty:
        errors.append("TTF=F unavailable")
        ttf = pd.Series(dtype=float)
    else:
        ttf = ttf_df.set_index("date")["price"]

    frame = align_ttf_jkm(ttf, raw["jkm"], raw["eurusd"], fx_max_age_days=LNG_FX_MAX_AGE_DAYS)
    if not frame.empty:
        pct = rolling_percentile(
            frame.set_index("date")["spread"],
            window_days=LNG_PCTL_WINDOW_DAYS,
            min_obs=LNG_PCTL_MIN_OBS,
        )
        frame = frame.assign(percentile=pct.to_numpy())
    elif not errors:
        errors.append("no overlapping TTF and JKM settlement dates")
    return {
        "frame": frame,
        "error": "; ".join(errors) if errors else None,
        "fetched_at": datetime.now(timezone.utc).replace(tzinfo=None),
    }
