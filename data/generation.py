"""
German power generation by fuel type via ENTSO-E A75
(Actual Generation Per Production Type).

Returns quarterly hard coal and lignite generation in TWh from 2019,
plus a 90-day daily series for the recent trend chart.

Disk cache: data/generation_cache.csv — persists across app restarts.
Only fetches quarters not already in the cache on subsequent calls.
"""
from __future__ import annotations

import os
import json
import time
import threading
import pandas as pd
import streamlit as st
from datetime import datetime

try:
    from entsoe import EntsoePandasClient
    ENTSOE_PY_AVAILABLE = True
except ImportError:
    ENTSOE_PY_AVAILABLE = False

DE_EIC     = "10Y1001A1001A83F"
START_YEAR = 2019
_CACHE_CSV  = os.path.join(os.path.dirname(__file__), "generation_cache.csv")
_CHUNK_TIMEOUT = 30   # seconds per annual ENTSO-E chunk before giving up

HARD_COAL_NAMES = {"Fossil Hard coal", "Hard coal"}
LIGNITE_NAMES   = {"Fossil Brown coal/Lignite", "Brown coal/Lignite", "Lignite"}


def _client():
    key = os.getenv("ENTSOE_API_KEY", "")
    if not key or not ENTSOE_PY_AVAILABLE:
        return None
    return EntsoePandasClient(api_key=key)


def _extract_coal_columns(df: pd.DataFrame) -> tuple:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    hard_coal = lignite = None
    for col in df.columns:
        if str(col) in HARD_COAL_NAMES:
            hard_coal = df[col]
        elif str(col) in LIGNITE_NAMES:
            lignite = df[col]
    return hard_coal, lignite


def _fetch_chunk_with_timeout(client, chunk_start, chunk_end) -> pd.DataFrame | None:
    """Fetch one annual chunk from ENTSO-E A75 with a hard timeout."""
    result = [None]
    exc    = [None]

    def _fetch():
        try:
            raw = client.query_generation(
                country_code=DE_EIC,
                start=chunk_start,
                end=chunk_end,
                psr_type=None,
            )
            result[0] = raw
        except Exception as e:
            exc[0] = e

    t = threading.Thread(target=_fetch, daemon=True)
    t.start()
    t.join(timeout=_CHUNK_TIMEOUT)

    if t.is_alive():
        return None   # timed out — skip this chunk
    if exc[0] is not None:
        return None
    return result[0]


def _load_cache() -> pd.DataFrame:
    try:
        if os.path.exists(_CACHE_CSV):
            df = pd.read_csv(_CACHE_CSV, parse_dates=["date"])
            return df
    except Exception:
        pass
    return pd.DataFrame()


def _save_cache(df: pd.DataFrame) -> None:
    try:
        df.to_csv(_CACHE_CSV, index=False)
    except Exception:
        pass


def _build_hourly_from_entsoe(client, fetch_from: pd.Timestamp, fetch_to: pd.Timestamp) -> pd.DataFrame:
    """Fetch hourly coal generation from ENTSO-E in annual chunks."""
    rows: list[pd.DataFrame] = []
    chunk_start = fetch_from
    while chunk_start < fetch_to:
        chunk_end = min(chunk_start + pd.Timedelta(days=365), fetch_to)
        raw = _fetch_chunk_with_timeout(client, chunk_start, chunk_end)
        if raw is not None and not raw.empty:
            hard_coal, lignite = _extract_coal_columns(raw)
            chunk_df = pd.DataFrame(index=raw.index)
            chunk_df["hard_coal_mwh"] = hard_coal if hard_coal is not None else 0.0
            chunk_df["lignite_mwh"]   = lignite   if lignite   is not None else 0.0
            rows.append(chunk_df)
        chunk_start = chunk_end

    if not rows:
        return pd.DataFrame()

    full = pd.concat(rows)
    full = full[~full.index.duplicated(keep="last")].sort_index()
    full.index = full.index.tz_convert("UTC")

    # Resample to daily GWh
    daily = full.resample("D").sum()
    daily["hard_coal_gwh"] = (daily["hard_coal_mwh"] / 1_000).round(2)
    daily["lignite_gwh"]   = (daily["lignite_mwh"]   / 1_000).round(2)
    daily = daily[["hard_coal_gwh", "lignite_gwh"]].reset_index()
    daily.columns = ["date", "hard_coal_gwh", "lignite_gwh"]
    daily["date"] = daily["date"].dt.tz_localize(None)
    return daily


@st.cache_data(ttl=21600)
def fetch_coal_generation() -> dict:
    """
    Fetch German hard coal and lignite generation from ENTSO-E A75.
    Uses disk cache — only fetches dates not already cached.
    """
    client = _client()
    empty = {"quarterly": pd.DataFrame(), "daily_90d": pd.DataFrame(),
             "source": None, "fetched_at": datetime.utcnow()}

    if client is None:
        cached = _load_cache()
        if not cached.empty:
            return _build_result(cached)
        return empty

    cached    = _load_cache()
    fetch_end = pd.Timestamp.now(tz="UTC").normalize()

    if cached.empty:
        fetch_from = pd.Timestamp(f"{START_YEAR}-01-01", tz="UTC")
    else:
        # Only fetch dates after the last cached date
        last_cached = pd.Timestamp(cached["date"].max(), tz="UTC")
        fetch_from  = last_cached - pd.Timedelta(days=7)   # overlap for safety

    new_data = _build_hourly_from_entsoe(client, fetch_from, fetch_end)

    if not new_data.empty:
        if not cached.empty:
            combined = pd.concat([cached, new_data], ignore_index=True)
            combined = (
                combined
                .sort_values("date")
                .drop_duplicates(subset=["date"], keep="last")
                .reset_index(drop=True)
            )
        else:
            combined = new_data
        _save_cache(combined)
        daily_full = combined
    elif not cached.empty:
        daily_full = cached
    else:
        return empty

    return _build_result(daily_full)


def _build_result(daily: pd.DataFrame) -> dict:
    """Convert daily GWh series to quarterly TWh and 90-day daily result dict."""
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.set_index("date").sort_index()

    # Quarterly TWh
    quarterly = daily.resample("QS").sum()
    quarterly["hard_coal_twh"] = (quarterly["hard_coal_gwh"] / 1_000).round(2)
    quarterly["lignite_twh"]   = (quarterly["lignite_gwh"]   / 1_000).round(2)
    quarterly = quarterly[["hard_coal_twh", "lignite_twh"]].reset_index()
    quarterly.columns = ["quarter", "hard_coal_twh", "lignite_twh"]
    # Drop current incomplete quarter if it has very few days
    if len(quarterly) > 1:
        last_q_start = quarterly["quarter"].iloc[-1]
        days_in_last_q = (daily.index[-1] - last_q_start).days
        if days_in_last_q < 14:
            quarterly = quarterly.iloc[:-1]

    # 90-day daily
    cutoff = daily.index[-1] - pd.Timedelta(days=90)
    daily_90d = daily[daily.index >= cutoff][["hard_coal_gwh", "lignite_gwh"]].reset_index()
    daily_90d.columns = ["date", "hard_coal_gwh", "lignite_gwh"]

    return {
        "quarterly":  quarterly,
        "daily_90d":  daily_90d,
        "source":     "ENTSO-E",
        "fetched_at": datetime.utcnow(),
    }


def get_generation_data() -> dict:
    return fetch_coal_generation()
