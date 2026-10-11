"""
GIE AGSI+ gas storage data fetcher.

EU aggregate:  GET https://agsi.gie.eu/api/data/eu   (paginated)
Country level: GET https://agsi.gie.eu/api?country=DE (paginated)

Requires AGSI_API_KEY (free registration at https://agsi.gie.eu). Requests go through
data/gie.py: rate limits and gateway errors are retried, and a history that could not be
fetched completely is not returned at all. A complete history is cached for an hour; a
failed fetch returns an empty frame that is remembered for GIE_FAILURE_TTL_S only, so the
panel recovers on the next visit after a transient failure instead of an hour later.
"""
import os
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st

from config.settings import GIE_FAILURE_TTL_S
from data.gie import GieUnavailable, fetch_all_pages

AGSI_BASE = "https://agsi.gie.eu/api"
AGSI_EU   = "https://agsi.gie.eu/api/data/eu"


@st.cache_data(ttl=3600, show_spinner=False)
def _storage_history(url: str, params: tuple) -> pd.DataFrame:
    """Complete history from one AGSI+ endpoint; raises GieUnavailable (never cached)."""
    return _to_df(fetch_all_pages(url, dict(params)))


def _storage_or_empty(url: str, params: dict) -> pd.DataFrame:
    if not os.getenv("AGSI_API_KEY", "").strip():
        return pd.DataFrame()
    try:
        return _storage_history(url, tuple(sorted(params.items())))
    except GieUnavailable:
        return pd.DataFrame()


@st.cache_data(ttl=GIE_FAILURE_TTL_S)
def fetch_storage_eu(date_from: str, date_to: str) -> pd.DataFrame:
    """Fetch EU aggregate gas storage history (empty without a key or when GIE fails)."""
    return _storage_or_empty(AGSI_EU, {"from": date_from, "to": date_to, "size": 300})


@st.cache_data(ttl=GIE_FAILURE_TTL_S)
def fetch_storage_country(country: str, date_from: str, date_to: str) -> pd.DataFrame:
    """Fetch country-level gas storage history, e.g. country='DE' (empty on failure)."""
    return _storage_or_empty(
        AGSI_BASE, {"country": country, "from": date_from, "to": date_to, "size": 300})


def _to_df(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["gasDayStart"] = pd.to_datetime(df["gasDayStart"], errors="coerce")
    df = df.dropna(subset=["gasDayStart"])
    df = df.sort_values("gasDayStart").reset_index(drop=True)
    for col in ["full", "trend", "gasInStorage", "workingGasVolume",
                "injection", "withdrawal", "netWithdrawal"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def get_storage_data() -> dict:
    """
    Returns dict with keys:
      'europe'   : DataFrame of EU aggregate storage
      'germany'  : DataFrame of Germany storage
      'fetched_at': datetime
    """
    today = datetime.utcnow().date()
    date_from = (today - timedelta(days=365 * 6)).strftime("%Y-%m-%d")
    date_to   = today.strftime("%Y-%m-%d")

    europe  = fetch_storage_eu(date_from, date_to)
    germany = fetch_storage_country("DE", date_from, date_to)

    return {
        "europe":     europe,
        "germany":    germany,
        "fetched_at": datetime.utcnow(),
    }


def build_seasonal_bands(df: pd.DataFrame, value_col: str = "full") -> pd.DataFrame:
    """
    Compute day-of-year seasonal min/mean/max from all years
    except the current year. Returns DataFrame with columns:
    day_of_year, min, mean, max.
    """
    if df.empty or value_col not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["year"] = df["gasDayStart"].dt.year
    df["doy"]  = df["gasDayStart"].dt.dayofyear
    current_year = datetime.utcnow().year
    historical = df[df["year"] < current_year]

    return (
        historical.groupby("doy")[value_col]
        .agg(["min", "mean", "max"])
        .reset_index()
        .rename(columns={"doy": "day_of_year"})
    )
