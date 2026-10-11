"""
GIE AGSI+ gas storage data fetcher.

EU aggregate:  GET https://agsi.gie.eu/api/data/eu   (paginated)
Country level: GET https://agsi.gie.eu/api?country=DE (paginated)

Requires AGSI_API_KEY — register free at https://agsi.gie.eu
"""
import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st

AGSI_BASE = "https://agsi.gie.eu/api"
AGSI_EU   = "https://agsi.gie.eu/api/data/eu"


def _headers() -> dict:
    return {"x-key": os.getenv("AGSI_API_KEY", "")}


def _fetch_pages(url: str, params: dict) -> list[dict]:
    """Fetch all pages from an AGSI+ paginated endpoint."""
    records = []
    page = 1
    while True:
        try:
            resp = requests.get(url, params={**params, "page": page}, headers=_headers(), timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            # Network error or timeout — return whatever was fetched so far.
            # Components will display contextual messages if data is insufficient.
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


@st.cache_data(ttl=3600)
def fetch_storage_eu(date_from: str, date_to: str) -> pd.DataFrame:
    """Fetch EU aggregate gas storage history."""
    if not os.getenv("AGSI_API_KEY", ""):
        return pd.DataFrame()
    records = _fetch_pages(AGSI_EU, {"from": date_from, "to": date_to, "size": 300})
    return _to_df(records)


@st.cache_data(ttl=3600)
def fetch_storage_country(country: str, date_from: str, date_to: str) -> pd.DataFrame:
    """Fetch country-level gas storage history (e.g. country='DE')."""
    if not os.getenv("AGSI_API_KEY", ""):
        return pd.DataFrame()
    records = _fetch_pages(AGSI_BASE, {"country": country, "from": date_from, "to": date_to, "size": 300})
    return _to_df(records)


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
