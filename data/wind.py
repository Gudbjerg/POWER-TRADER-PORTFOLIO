"""
Wind generation fetcher for Germany (DE) and Norway (NO).

Primary:  ENTSO-E B18 (Wind Offshore) + B19 (Wind Onshore). Requires ENTSOE_API_KEY.
Fallback: energy-charts.info (Fraunhofer ISE) for DE — no key required.
          No fallback for NO (Norway wind is a minor generation source anyway).

Returns daily GWh totals per country.
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

DE_EIC = "10Y1001A1001A83F"   # Germany
NO_EIC = "10YNO-0--------C"   # Norway aggregate
ENERGY_CHARTS_BASE = "https://api.energy-charts.info"


def _client():
    key = os.getenv("ENTSOE_API_KEY", "")
    if not key or not ENTSOE_PY_AVAILABLE:
        return None
    return EntsoePandasClient(api_key=key)


def _entsoe_wind(client, eic: str, start, end) -> pd.Series:
    """Fetch hourly wind (B19 onshore + B18 offshore) from ENTSO-E, return daily GWh."""
    parts = []
    for psr in ("B19", "B18"):
        try:
            gen = client.query_generation(eic, start=start, end=end, psr_type=psr)
            gen = gen.tz_convert("UTC")
            if isinstance(gen.columns, pd.MultiIndex):
                gen.columns = ["_".join(str(c) for c in col).strip() for col in gen.columns]
            series = pd.to_numeric(gen.iloc[:, 0], errors="coerce")
            daily  = series.resample("D").sum() / 1000   # MWh → GWh
            parts.append(daily)
        except Exception:
            continue
    if not parts:
        return pd.Series(dtype=float)
    total = pd.concat(parts, axis=1).sum(axis=1)
    total.index = total.index.tz_localize(None)
    return total


def _energy_charts_wind(days: int) -> pd.Series:
    """Fallback: Fraunhofer ISE energy-charts.info for DE wind (no key needed)."""
    try:
        import requests
    except ImportError:
        return pd.Series(dtype=float)

    now   = datetime.now()
    start = now - timedelta(days=days)
    fmt   = "%Y-%m-%dT%H:%M+01:00"
    s, e  = start.strftime(fmt), now.strftime(fmt)

    parts = []
    for ptype in ("wind_onshore", "wind_offshore"):
        try:
            r = requests.get(
                f"{ENERGY_CHARTS_BASE}/public_power",
                params={"country": "de", "production_type": ptype,
                        "time_step": "hour", "start": s, "end": e},
                timeout=15,
            )
            r.raise_for_status()
            data = r.json()
            ts     = pd.to_datetime(data["unix_seconds"], unit="s", utc=True).tz_localize(None)
            series = pd.Series(data["power"], index=ts, dtype=float)  # MW hourly
            daily  = series.resample("D").sum() / 1000  # MWh sum → GWh
            parts.append(daily)
        except Exception:
            continue

    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts, axis=1).sum(axis=1)


@st.cache_data(ttl=3600)
def fetch_wind_daily(days: int = 730) -> dict:
    """
    Fetch daily wind generation totals for DE and NO.

    Returns
    -------
    dict:
        daily  : pd.DataFrame with columns date (datetime.date), de_wind_gwh, no_wind_gwh
        source : "ENTSO-E" | "energy-charts.info (DE only, NO unavailable)" | "unavailable"
    """
    client = _client()
    today  = pd.Timestamp.now(tz="UTC").normalize()
    start  = today - pd.Timedelta(days=days)

    de_series = pd.Series(dtype=float)
    no_series = pd.Series(dtype=float)
    source    = "unavailable"

    if client is not None:
        de_series = _entsoe_wind(client, DE_EIC, start, today)
        no_series = _entsoe_wind(client, NO_EIC, start, today)
        if not de_series.empty:
            source = "ENTSO-E"

    if de_series.empty:
        de_series = _energy_charts_wind(days)
        if not de_series.empty:
            source = "energy-charts.info (DE only, NO unavailable)"

    if de_series.empty:
        return {"daily": pd.DataFrame(), "source": "unavailable"}

    df = pd.DataFrame(index=de_series.index)
    df["de_wind_gwh"] = de_series
    if not no_series.empty:
        df["no_wind_gwh"] = no_series
    else:
        df["no_wind_gwh"] = float("nan")

    df.index.name = "date"
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.dropna(subset=["de_wind_gwh"]).sort_values("date").reset_index(drop=True)

    return {"daily": df, "source": source}


def get_wind_data() -> dict:
    result = fetch_wind_daily()
    result["fetched_at"] = datetime.utcnow()
    return result
