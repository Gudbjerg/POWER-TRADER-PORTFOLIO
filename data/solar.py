"""
Solar cannibalisation data fetcher for Germany.

Primary:  ENTSO-E Transparency Platform (Actual Generation A75 / Solar B16).
          Requires ENTSOE_API_KEY.
Fallback: energy-charts.info (Fraunhofer ISE) — public REST API, no key needed.
          Used automatically when ENTSO-E is unreachable or returns no data.

Returns hourly averages (hour 0-23) of day-ahead price and solar generation
averaged over the last `days` days for Germany.
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

DE_EIC = "10Y1001A1001A83F"
ENERGY_CHARTS_BASE = "https://api.energy-charts.info"


def _entsoe_client():
    key = os.getenv("ENTSOE_API_KEY", "")
    if not key or not ENTSOE_PY_AVAILABLE:
        return None
    return EntsoePandasClient(api_key=key)


def _fetch_entsoe(days: int) -> pd.DataFrame:
    """Try ENTSO-E. Returns hourly df or empty DataFrame."""
    client = _entsoe_client()
    if client is None:
        return pd.DataFrame()

    today = pd.Timestamp.now(tz="Europe/Berlin").normalize()
    start = today - pd.Timedelta(days=days)

    price_df = pd.DataFrame()
    solar_df = pd.DataFrame()

    try:
        prices = client.query_day_ahead_prices(DE_EIC, start=start, end=today)
        prices = prices.tz_convert("Europe/Berlin")
        price_df = pd.DataFrame({"price": prices.values}, index=prices.index)
    except Exception:
        pass

    try:
        gen = client.query_generation(DE_EIC, start=start, end=today, psr_type="B16")
        gen = gen.tz_convert("Europe/Berlin")
        if isinstance(gen.columns, pd.MultiIndex):
            gen.columns = ["_".join(c).strip() for c in gen.columns]
        solar_col = gen.columns[0]
        solar_df = pd.DataFrame({"solar_mw": gen[solar_col].values}, index=gen.index)
    except Exception:
        pass

    if price_df.empty or solar_df.empty:
        return pd.DataFrame()

    merged = price_df.join(solar_df, how="inner")
    merged["hour"] = merged.index.hour
    return merged.groupby("hour")[["price", "solar_mw"]].mean().reset_index()


def _fetch_energy_charts(days: int) -> pd.DataFrame:
    """
    Fallback: Fraunhofer ISE energy-charts.info public API.
    Fetches hourly solar generation and DE-LU day-ahead prices.
    """
    try:
        import requests
    except ImportError:
        return pd.DataFrame()

    now   = datetime.now()
    start = now - timedelta(days=days)
    # energy-charts.info accepts ISO 8601 with UTC offset
    fmt = "%Y-%m-%dT%H:%M+01:00"
    s   = start.strftime(fmt)
    e   = now.strftime(fmt)

    solar_series = pd.Series(dtype=float)
    price_series = pd.Series(dtype=float)

    try:
        r = requests.get(
            f"{ENERGY_CHARTS_BASE}/public_power",
            params={"country": "de", "production_type": "solar",
                    "time_step": "hour", "start": s, "end": e},
            timeout=12,
        )
        r.raise_for_status()
        data = r.json()
        ts = pd.to_datetime(data["unix_seconds"], unit="s", utc=True).tz_convert("Europe/Berlin")
        solar_series = pd.Series(data["power"], index=ts, name="solar_mw")
    except Exception:
        pass

    try:
        r = requests.get(
            f"{ENERGY_CHARTS_BASE}/price",
            params={"bzn": "DE-LU", "start": s, "end": e},
            timeout=12,
        )
        r.raise_for_status()
        data = r.json()
        ts = pd.to_datetime(data["unix_seconds"], unit="s", utc=True).tz_convert("Europe/Berlin")
        price_series = pd.Series(data["price"], index=ts, name="price")
    except Exception:
        pass

    if solar_series.empty or price_series.empty:
        return pd.DataFrame()

    merged = pd.concat([price_series, solar_series], axis=1).dropna()
    if merged.empty:
        return pd.DataFrame()

    merged["hour"] = merged.index.hour
    return merged.groupby("hour")[["price", "solar_mw"]].mean().reset_index()


@st.cache_data(ttl=3600)
def fetch_solar_cannibalisation(days: int = 14) -> dict:
    """
    Fetch hourly solar cannibalisation data, trying ENTSO-E first.

    Returns
    -------
    dict with keys:
        hourly  : pd.DataFrame with columns hour, price, solar_mw
        source  : "ENTSO-E" | "energy-charts.info (Fraunhofer ISE)" | None
    """
    df = _fetch_entsoe(days)
    if not df.empty:
        return {"hourly": df, "source": "ENTSO-E"}

    df = _fetch_energy_charts(days)
    if not df.empty:
        return {"hourly": df, "source": "energy-charts.info (Fraunhofer ISE)"}

    return {"hourly": pd.DataFrame(), "source": None}


def get_solar_data() -> dict:
    result = fetch_solar_cannibalisation()
    result["fetched_at"] = datetime.utcnow()
    return result
