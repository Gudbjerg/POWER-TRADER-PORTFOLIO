"""
German power generation by fuel type via ENTSO-E A75
(Actual Generation Per Production Type).

Returns quarterly hard coal and lignite generation in TWh from 2019,
plus a 90-day daily series for the recent trend chart.
"""
from __future__ import annotations

import os
import pandas as pd
import streamlit as st
from datetime import datetime

try:
    from entsoe import EntsoePandasClient
    ENTSOE_PY_AVAILABLE = True
except ImportError:
    ENTSOE_PY_AVAILABLE = False

DE_EIC     = "10Y1001A1001A83F"   # Germany bidding zone
START_YEAR = 2019                  # Marius's chart starts 2019

# ENTSO-E production type display names (A75 document)
HARD_COAL_NAMES = {"Fossil Hard coal", "Hard coal"}
LIGNITE_NAMES   = {"Fossil Brown coal/Lignite", "Brown coal/Lignite", "Lignite"}


def _client():
    key = os.getenv("ENTSOE_API_KEY", "")
    if not key or not ENTSOE_PY_AVAILABLE:
        return None
    return EntsoePandasClient(api_key=key)


def _extract_coal_columns(df: pd.DataFrame) -> tuple[pd.Series | None, pd.Series | None]:
    """
    Extract hard coal and lignite columns from an ENTSO-E generation DataFrame.
    Handles both flat and MultiIndex column formats returned by entsoe-py.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten: take the second level (production type name)
        df.columns = [col[1] if isinstance(col, tuple) else col for col in df.columns]

    hard_coal = None
    lignite   = None

    for col in df.columns:
        col_str = str(col)
        if col_str in HARD_COAL_NAMES:
            hard_coal = df[col]
        elif col_str in LIGNITE_NAMES:
            lignite = df[col]

    return hard_coal, lignite


@st.cache_data(ttl=21600)   # 6-hour cache — generation data updates daily
def fetch_coal_generation() -> dict:
    """
    Fetch German hard coal and lignite generation from ENTSO-E A75.

    Returns
    -------
    dict with keys:
      "quarterly"  : pd.DataFrame — columns: quarter (Period), hard_coal_twh, lignite_twh
      "daily_90d"  : pd.DataFrame — columns: date, hard_coal_gwh, lignite_gwh
      "source"     : str
      "fetched_at" : datetime
    """
    client = _client()
    if client is None:
        return {
            "quarterly":  pd.DataFrame(),
            "daily_90d":  pd.DataFrame(),
            "source":     None,
            "fetched_at": datetime.utcnow(),
        }

    end   = pd.Timestamp.now(tz="UTC").normalize()
    start = pd.Timestamp(f"{START_YEAR}-01-01", tz="UTC")

    quarterly_chunks: list[pd.DataFrame] = []

    # Fetch in annual chunks — ENTSO-E limits query windows
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + pd.Timedelta(days=365), end)
        try:
            raw = client.query_generation(
                country_code=DE_EIC,
                start=chunk_start,
                end=chunk_end,
                psr_type=None,
            )
            if raw is not None and not raw.empty:
                hard_coal, lignite = _extract_coal_columns(raw)
                chunk_df = pd.DataFrame(index=raw.index)
                chunk_df["hard_coal_mwh"] = hard_coal if hard_coal is not None else 0.0
                chunk_df["lignite_mwh"]   = lignite   if lignite   is not None else 0.0
                quarterly_chunks.append(chunk_df)
        except Exception:
            pass
        chunk_start = chunk_end

    if not quarterly_chunks:
        return {
            "quarterly":  pd.DataFrame(),
            "daily_90d":  pd.DataFrame(),
            "source":     None,
            "fetched_at": datetime.utcnow(),
        }

    full = pd.concat(quarterly_chunks)
    full = full[~full.index.duplicated(keep="last")].sort_index()
    full.index = full.index.tz_convert("UTC")

    # ── Quarterly aggregation (sum MWh per quarter, convert to TWh) ─────────
    quarterly = full.resample("QS").sum()
    quarterly["hard_coal_twh"] = (quarterly["hard_coal_mwh"] / 1_000_000).round(2)
    quarterly["lignite_twh"]   = (quarterly["lignite_mwh"]   / 1_000_000).round(2)
    quarterly = quarterly[["hard_coal_twh", "lignite_twh"]].reset_index()
    quarterly.columns = ["quarter", "hard_coal_twh", "lignite_twh"]
    quarterly["quarter"] = quarterly["quarter"].dt.tz_localize(None)

    # ── Daily 90-day series (GWh/day for recent trend) ──────────────────────
    recent_start = end - pd.Timedelta(days=90)
    recent = full[full.index >= recent_start].resample("D").sum()
    recent["hard_coal_gwh"] = (recent["hard_coal_mwh"] / 1_000).round(1)
    recent["lignite_gwh"]   = (recent["lignite_mwh"]   / 1_000).round(1)
    daily_90d = recent[["hard_coal_gwh", "lignite_gwh"]].reset_index()
    daily_90d.columns = ["date", "hard_coal_gwh", "lignite_gwh"]
    daily_90d["date"] = daily_90d["date"].dt.tz_localize(None)

    return {
        "quarterly":  quarterly,
        "daily_90d":  daily_90d,
        "source":     "ENTSO-E",
        "fetched_at": datetime.utcnow(),
    }


def get_generation_data() -> dict:
    return fetch_coal_generation()
