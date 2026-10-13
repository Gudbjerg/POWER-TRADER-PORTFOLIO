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


HISTORY_YEARS = 6   # years of storage history behind the seasonal bands

# EU filling target (Regulation (EU) 2022/1032 as amended in 2025, applying to the end of 2027):
# 90% of capacity at any point between 1 October and 1 December, with deviations of up to
# 10 percentage points allowed in difficult market conditions.
EU_TARGET_PCT       = 90.0
EU_TARGET_FLEX_PP   = 10.0
EU_TARGET_LABEL     = "EU target 90% (any day 1 Oct to 1 Dec)"
EU_TARGET_SENTENCE  = (
    "The EU target is 90% fill at any point between 1 October and 1 December, with up to "
    "10 percentage points of deviation allowed in difficult market conditions (rules in force "
    "to the end of 2027)."
)


def calendar_doy(dates) -> pd.Series | int:
    """Day of year on a leap-year calendar: 29 February is 60 and 1 March is 61 in every year.

    Comparing years by plain day of year shifts every date after February by one day in leap
    years and leaves day 366 without a counterpart; this keeps each calendar date aligned.
    """
    if isinstance(dates, pd.Series):
        d = pd.to_datetime(dates)
        return (pd.to_datetime({"year": 2024, "month": d.dt.month, "day": d.dt.day})
                .dt.dayofyear.where(d.notna()))
    ts = pd.Timestamp(dates)
    return pd.Timestamp(year=2024, month=ts.month, day=ts.day).dayofyear


def storage_season(when) -> str:
    """Phase of the gas storage year for wording: injection, target window or withdrawal."""
    month = pd.Timestamp(when).month
    if 4 <= month <= 9:
        return "injection season"
    if month in (10, 11):
        return "EU target window"
    return "withdrawal season"


def history_window(years: int = HISTORY_YEARS) -> tuple[str, str]:
    """(date_from, date_to) of the storage history request, as AGSI+ date strings.

    Every caller uses the same window so the cached history is shared between pages.
    """
    today = datetime.utcnow().date()
    return ((today - timedelta(days=365 * years)).strftime("%Y-%m-%d"),
            today.strftime("%Y-%m-%d"))


def get_storage_data() -> dict:
    """
    Returns dict with keys:
      'europe'   : DataFrame of EU aggregate storage
      'germany'  : DataFrame of Germany storage
      'fetched_at': datetime
    """
    date_from, date_to = history_window()

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
    df["doy"]  = calendar_doy(df["gasDayStart"])
    current_year = datetime.utcnow().year
    historical = df[df["year"] < current_year]

    return (
        historical.groupby("doy")[value_col]
        .agg(["min", "mean", "max"])
        .reset_index()
        .rename(columns={"doy": "day_of_year"})
    )
