"""
German power generation by fuel type via ENTSO-E A75
(Actual Generation Per Production Type).

Returns quarterly hard coal and lignite generation in TWh from 2019,
plus a 90-day daily series for the recent trend chart.

ENTSO-E reports this series as average power (MW) over each 15-minute interval. Energy is
the hourly mean power summed over the hours of each UTC day, so the GWh and TWh figures are
energy whatever the reporting resolution.

Disk cache: generation_cache_v2.csv, one row per UTC day, kept in DATA_PERSIST_DIR when
that is set and next to this module otherwise. Later calls fetch only the days after the
last cached day, with a seven-day overlap that picks up late revisions. When the cache is
missing, the copy bundled next to this module is used as the starting point; the
deployment branch ships one, so a fresh server does not have to fetch the history from
2019 before the Live Monitor can render. Version 1 of the cache (generation_cache.csv)
summed the 15-minute readings as if they were hourly energy, which overstated every figure
fourfold; it is no longer read.
"""
from __future__ import annotations

from config.settings import ENTSOE_TIMEOUT_S

import os
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
_BUNDLED_DIR = os.path.dirname(os.path.abspath(__file__))
_PERSIST_DIR = os.getenv("DATA_PERSIST_DIR", _BUNDLED_DIR)
_CACHE_NAME  = "generation_cache_v2.csv"
_CACHE_CSV   = os.path.join(_PERSIST_DIR, _CACHE_NAME)
_SEED_CSV    = os.path.join(_BUNDLED_DIR, _CACHE_NAME)   # copy shipped with a deployment
_CACHE_COLUMNS = ["date", "hard_coal_gwh", "lignite_gwh"]
_CHUNK_TIMEOUT = 90   # seconds per quarterly ENTSO-E chunk before giving up
_MIN_HOURS_PER_DAY = 23   # hourly readings needed before a day counts as published

HARD_COAL_NAMES = {"Fossil Hard coal", "Hard coal"}
LIGNITE_NAMES   = {"Fossil Brown coal/Lignite", "Brown coal/Lignite", "Lignite"}


def _client():
    key = os.getenv("ENTSOE_API_KEY", "")
    if not key or not ENTSOE_PY_AVAILABLE:
        return None
    return EntsoePandasClient(api_key=key, timeout=ENTSOE_TIMEOUT_S)


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
    """Read the disk cache, or the bundled copy when the disk cache does not exist yet."""
    for path in dict.fromkeys((_CACHE_CSV, _SEED_CSV)):
        try:
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path, parse_dates=["date"])
            if not df.empty and set(_CACHE_COLUMNS) <= set(df.columns):
                return df[_CACHE_COLUMNS]
        except Exception:
            continue
    return pd.DataFrame()


def _save_cache(df: pd.DataFrame) -> None:
    try:
        if os.getenv("DATA_PERSIST_DIR"):
            import sys
            print(f"[generation] writing cache to {_CACHE_CSV}", file=sys.stderr)
        df.to_csv(_CACHE_CSV, index=False)
    except Exception:
        pass


def _daily_energy_gwh(power_mw: pd.DataFrame) -> pd.DataFrame:
    """Turn power readings in MW into energy per UTC day in GWh.

    ``power_mw`` has a tz-aware index and the columns ``hard_coal_mw`` and ``lignite_mw``.
    Each reading is the average power over its interval, so averaging to hourly power and
    summing the hours gives energy at any resolution; summing 15-minute readings directly
    would count every hour four times. Days with fewer than ``_MIN_HOURS_PER_DAY`` hourly
    readings (typically the current day) are left out, so they are fetched again later
    instead of being stored as low output.
    """
    if power_mw.empty:
        return pd.DataFrame(columns=_CACHE_COLUMNS)
    power = power_mw.copy()
    power.index = power.index.tz_convert("UTC")
    hourly = power.resample("h").mean()
    hours_reported = hourly.notna().any(axis=1).resample("D").sum()
    daily = hourly.resample("D").sum(min_count=1) / 1_000
    daily = daily[hours_reported.reindex(daily.index, fill_value=0) >= _MIN_HOURS_PER_DAY]
    daily = daily.fillna(0.0).round(2)
    return pd.DataFrame({
        "date":          daily.index.tz_localize(None),
        "hard_coal_gwh": daily["hard_coal_mw"].to_numpy(),
        "lignite_gwh":   daily["lignite_mw"].to_numpy(),
    })


def _build_hourly_from_entsoe(client, fetch_from: pd.Timestamp, fetch_to: pd.Timestamp) -> pd.DataFrame:
    """Fetch coal generation from ENTSO-E in quarterly chunks; return daily energy in GWh."""
    rows: list[pd.DataFrame] = []
    chunk_start = fetch_from
    while chunk_start < fetch_to:
        chunk_end = min(chunk_start + pd.Timedelta(days=92), fetch_to)
        raw = _fetch_chunk_with_timeout(client, chunk_start, chunk_end)
        if raw is not None and not raw.empty:
            hard_coal, lignite = _extract_coal_columns(raw)
            chunk_df = pd.DataFrame(index=raw.index)
            chunk_df["hard_coal_mw"] = hard_coal if hard_coal is not None else 0.0
            chunk_df["lignite_mw"]   = lignite   if lignite   is not None else 0.0
            rows.append(chunk_df)
        chunk_start = chunk_end

    if not rows:
        return pd.DataFrame()

    full = pd.concat(rows)
    full = full[~full.index.duplicated(keep="last")].sort_index()
    daily = _daily_energy_gwh(full)
    return daily if not daily.empty else pd.DataFrame()


@st.cache_data(ttl=21600)
def fetch_coal_generation() -> dict:
    """
    Fetch German hard coal and lignite generation from ENTSO-E A75.
    Uses the disk cache and fetches only the days after the last cached day.
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
            # Deduplicate before sorting: keep="last" must refer to the concatenation
            # order (fresh rows after cached ones), which an unstable sort would not keep.
            combined = pd.concat([cached, new_data], ignore_index=True)
            combined = (
                combined
                .drop_duplicates(subset=["date"], keep="last")
                .sort_values("date")
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
