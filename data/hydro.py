"""
Norwegian hydro reservoir filling level fetcher via ENTSO-E.

Uses the Hydro Reservoirs & Water Value data (document type B31, EIC 10YNO-0--------C).
Returns weekly filling levels for Norway aggregated, with historical percentile bands.
"""
from data.deadline import call_with_deadline
from config.settings import ENTSOE_TIMEOUT_S
import os
import pandas as pd
import streamlit as st

from data.deadline import cache_unless_empty
from datetime import datetime

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
    return EntsoePandasClient(api_key=key, timeout=ENTSOE_TIMEOUT_S)


@cache_unless_empty(ttl=3600)
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

    # One request for six years regularly runs past the deadline on a loaded Transparency
    # Platform; yearly requests side by side each finish well inside it. A year that still
    # fails is left out (the percentile bands then rest on the other years).
    from concurrent.futures import ThreadPoolExecutor

    edges = [start + pd.Timedelta(days=365 * i) for i in range(years)] + [today]

    def _year(lo, hi):
        try:
            return call_with_deadline(client.query_aggregate_water_reservoirs_and_hydro_storage,
                                      country_code=NO_EIC, start=lo, end=hi)
        except Exception as _exc:
            import sys
            print(f"[hydro] ENTSO-E B31 {lo:%Y-%m} failed: {type(_exc).__name__}", file=sys.stderr)  # no message: it can carry the API token
            return None

    with ThreadPoolExecutor(max_workers=3) as pool:
        parts = list(pool.map(lambda e: _year(*e), zip(edges[:-1], edges[1:])))
    parts = [x for x in parts if x is not None and not (hasattr(x, "empty") and x.empty)]
    if not parts:
        return pd.DataFrame()
    raw = pd.concat(parts)
    raw = raw[~raw.index.duplicated(keep="last")].sort_index()

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
    source = "ENTSO-E B31 (Norway aggregate)" if not df.empty else "unavailable"
    return {
        "weekly":      df,
        "percentiles": percentiles,
        "source":      source,
        "fetched_at":  datetime.utcnow(),
    }
