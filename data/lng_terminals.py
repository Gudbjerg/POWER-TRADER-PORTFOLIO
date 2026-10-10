"""
GIE ALSI LNG terminal sendout fetcher.
Covers NW European import terminals: BE (Zeebrugge), NL (Gate), GB (South Hook/Dragon), FR (Dunkirk/Montoir).
Uses the same AGSI_API_KEY — register free at agsi.gie.eu.
"""
import os
import requests
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from config.settings import LNG_COUNTRIES, LNG_SENDOUT_DROP_PCT

ALSI_BASE = "https://alsi.gie.eu/api"


def _headers() -> dict:
    return {"x-key": os.getenv("AGSI_API_KEY", "")}


def _fetch_pages(params: dict) -> list[dict]:
    records = []
    page = 1
    while True:
        try:
            resp = requests.get(ALSI_BASE, params={**params, "page": page}, headers=_headers(), timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            break

        if data.get("error") == "access denied":
            break

        batch = data.get("data", [])
        if not batch:
            break
        records.extend(batch)

        last_page = data.get("last_page", 1)
        if page >= last_page:
            break
        page += 1

    return records


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


@st.cache_data(ttl=3600)
def fetch_lng_country(country: str, date_from: str, date_to: str) -> pd.DataFrame:
    if not os.getenv("AGSI_API_KEY", ""):
        return pd.DataFrame()
    records = _fetch_pages({"country": country, "from": date_from, "to": date_to, "size": 300})
    return _to_df(records, country)


def get_lng_data() -> dict:
    """
    Returns dict with keys:
      'sendout'      : DataFrame — per-country daily sendout (long format)
      'totals'       : DataFrame — date, total_sendout_twh, ma7
      'wow_change_pct': float or None — week-on-week % change in total sendout
      'alert'        : bool — True if sendout dropped > LNG_SENDOUT_DROP_PCT
      'fetched_at'   : datetime
    """
    today     = datetime.utcnow().date()
    date_from = (today - timedelta(days=90)).strftime("%Y-%m-%d")
    date_to   = today.strftime("%Y-%m-%d")

    frames = []
    for country in LNG_COUNTRIES:
        df = fetch_lng_country(country, date_from, date_to)
        if not df.empty:
            frames.append(df)

    if not frames:
        return {
            "sendout": pd.DataFrame(),
            "totals": pd.DataFrame(),
            "wow_change_pct": None,
            "alert": False,
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
        "fetched_at": datetime.utcnow(),
    }
