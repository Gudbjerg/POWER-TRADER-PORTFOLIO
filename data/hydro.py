"""
Norwegian hydro reservoir filling level fetcher via ENTSO-E.

Uses the Hydro Reservoirs & Water Value data (document type B31, EIC 10YNO-0--------C).
Returns weekly filling levels for Norway aggregated, with historical percentile bands.
"""
import os
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

try:
    from entsoe import EntsoePandasClient
    ENTSOE_PY_AVAILABLE = True
except ImportError:
    ENTSOE_PY_AVAILABLE = False

NO_EIC = "10YNO-0--------C"   # Norway aggregate bidding zone


def _client():
    key = os.getenv("ENTSOE_API_KEY", "")
    if not key or not ENTSOE_PY_AVAILABLE:
        return None
    return EntsoePandasClient(api_key=key)


@st.cache_data(ttl=3600)
def fetch_hydro_reservoirs(years: int = 6) -> pd.DataFrame:
    """
    Fetch weekly Norwegian hydro reservoir filling levels.

    Parameters
    ----------
    years : int
        Number of years of history to fetch (default: 6 for percentile bands).

    Returns
    -------
    pd.DataFrame with columns: week_start, filling_pct, year, week_of_year.
    """
    client = _client()
    if client is None:
        return pd.DataFrame()

    today = pd.Timestamp.now(tz="Europe/Oslo").normalize()
    start = today - pd.Timedelta(days=365 * years)

    try:
        raw = client.query_aggregate_water_reservoirs_and_hydro_storage(
            country_code=NO_EIC,
            start=start,
            end=today,
        )
    except Exception:
        return pd.DataFrame()

    if raw is None or (hasattr(raw, "empty") and raw.empty):
        return pd.DataFrame()

    # entsoe-py may return a Series or DataFrame
    if isinstance(raw, pd.Series):
        df = raw.rename("filling_mwh").reset_index()
        df.columns = ["week_start", "filling_mwh"]
    else:
        col = raw.columns[0]
        df = raw[[col]].rename(columns={col: "filling_mwh"}).reset_index()
        df.columns = ["week_start", "filling_mwh"]

    df["week_start"]  = pd.to_datetime(df["week_start"]).dt.tz_localize(None)
    df["filling_mwh"] = pd.to_numeric(df["filling_mwh"], errors="coerce")
    df = df.dropna(subset=["filling_mwh"])

    # Convert MWh → TWh for display (ENTSO-E B31 returns absolute MWh values)
    df["filling_twh"]  = df["filling_mwh"] / 1_000_000
    df["year"]         = df["week_start"].dt.year
    df["week_of_year"] = df["week_start"].dt.isocalendar().week.astype(int)
    df = df.sort_values("week_start").reset_index(drop=True)
    return df


def build_hydro_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute weekly historical percentile bands (10th, 50th, 90th) excluding current year.

    Returns DataFrame with columns: week_of_year, p10, p50, p90.
    """
    if df.empty:
        return pd.DataFrame()

    current_year = datetime.utcnow().year
    hist = df[df["year"] < current_year]
    if hist.empty:
        return pd.DataFrame()

    return (
        hist.groupby("week_of_year")["filling_twh"]
        .agg(p10=lambda x: x.quantile(0.10),
             p50=lambda x: x.quantile(0.50),
             p90=lambda x: x.quantile(0.90))
        .reset_index()
    )


def get_hydro_data() -> dict:
    df = fetch_hydro_reservoirs()
    percentiles = build_hydro_percentiles(df)
    return {
        "weekly":      df,
        "percentiles": percentiles,
        "fetched_at":  datetime.utcnow(),
    }
