"""
GIE ALSI LNG terminal sendout fetcher.
Covers NW European import terminals: BE (Zeebrugge), NL (Gate), GB (South Hook/Dragon), FR (Dunkirk/Montoir).
Uses the same AGSI_API_KEY (free registration at agsi.gie.eu).

Requests go through data/gie.py (rate limits and gateway errors are retried). A country
whose fetch fails is reported in ``unavailable`` rather than silently left out of the
NW European total, and the failure is remembered for GIE_FAILURE_TTL_S only.
"""
import os
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from config.settings import GIE_FAILURE_TTL_S, LNG_COUNTRIES, LNG_SENDOUT_DROP_PCT
from data.gie import GieUnavailable, fetch_all_pages

ALSI_BASE = "https://alsi.gie.eu/api"


def _to_df(records: list[dict], country: str) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["gasDayStart"] = pd.to_datetime(df["gasDayStart"], errors="coerce")
    df = df.dropna(subset=["gasDayStart"])
    df = df.sort_values("gasDayStart").reset_index(drop=True)
    df["country"] = country
    df["country_label"] = LNG_COUNTRIES.get(country, country)
    for col in ["sendOut", "dtmi", "lngInventory", "full", "trend"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def _lng_history(country: str, date_from: str, date_to: str) -> pd.DataFrame:
    """Complete send-out history for one country; raises GieUnavailable (never cached)."""
    records = fetch_all_pages(ALSI_BASE, {"country": country, "from": date_from, "to": date_to, "size": 300})
    return _to_df(records, country)


@st.cache_data(ttl=GIE_FAILURE_TTL_S)
def fetch_lng_country(country: str, date_from: str, date_to: str) -> pd.DataFrame | None:
    """Send-out history for one country: empty without a key, None when GIE failed."""
    if not os.getenv("AGSI_API_KEY", "").strip():
        return pd.DataFrame()
    try:
        return _lng_history(country, date_from, date_to)
    except GieUnavailable:
        return None


def get_lng_data() -> dict:
    """
    Returns dict with keys:
      'sendout'      : DataFrame — per-country daily sendout (long format)
      'totals'       : DataFrame: date, total_sendout (GWh/d, as published by ALSI), ma7
      'wow_change_pct': float or None — week-on-week % change in total sendout
      'alert'        : bool — True if sendout dropped > LNG_SENDOUT_DROP_PCT
      'unavailable'  : list of LNG_COUNTRIES codes whose fetch failed (totals exclude them)
      'fetched_at'   : datetime
    """
    today     = datetime.utcnow().date()
    date_from = (today - timedelta(days=90)).strftime("%Y-%m-%d")
    date_to   = today.strftime("%Y-%m-%d")

    frames = []
    unavailable = []
    for country in LNG_COUNTRIES:
        df = fetch_lng_country(country, date_from, date_to)
        if df is None:
            unavailable.append(country)
        elif not df.empty:
            frames.append(df)

    if not frames:
        return {
            "sendout": pd.DataFrame(),
            "totals": pd.DataFrame(),
            "wow_change_pct": None,
            "alert": False,
            "unavailable": unavailable,
            "fetched_at": datetime.utcnow(),
        }

    combined = pd.concat(frames, ignore_index=True)

    # Daily totals across all countries
    totals = (
        combined.groupby("gasDayStart")["sendOut"]
        .sum()
        .reset_index()
        .rename(columns={"gasDayStart": "date", "sendOut": "total_sendout"})
        .sort_values("date")
        .reset_index(drop=True)
    )
    totals["ma7"] = totals["total_sendout"].rolling(7, min_periods=1).mean()

    # Week-on-week change
    wow_change_pct = None
    alert = False
    if len(totals) >= 14:
        recent_7 = totals["total_sendout"].iloc[-7:].mean()
        prior_7  = totals["total_sendout"].iloc[-14:-7].mean()
        if prior_7 > 0:
            wow_change_pct = (recent_7 - prior_7) / prior_7 * 100
            alert = wow_change_pct < -LNG_SENDOUT_DROP_PCT

    return {
        "sendout": combined,
        "totals": totals,
        "wow_change_pct": wow_change_pct,
        "alert": alert,
        "unavailable": unavailable,
        "fetched_at": datetime.utcnow(),
    }
