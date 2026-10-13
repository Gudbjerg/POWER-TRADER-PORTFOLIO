"""
Wind generation fetcher for Germany (DE) and Norway (NO).

Primary:  ENTSO-E B18 (Wind Offshore) + B19 (Wind Onshore). Requires ENTSOE_API_KEY.
Fallback: energy-charts.info (Fraunhofer ISE) for DE — no key required.
          No fallback for NO (Norway wind is a minor generation source anyway).

Returns daily GWh totals per country. ENTSO-E reports wind as average power per
15 minutes for Germany; every source is averaged to hourly power before the daily sum
(data/energy_units.py), so the totals are energy whatever the resolution.
"""
from config.settings import ENTSOE_TIMEOUT_S
from data.energy_units import daily_energy
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
    return EntsoePandasClient(api_key=key, timeout=ENTSOE_TIMEOUT_S)


_WIND_CHUNK_DAYS = 90   # long 15-minute requests time out; fetch in quarters


def _entsoe_wind(client, eic: str, start, end) -> pd.Series:
    """Daily wind energy in GWh (B19 onshore + B18 offshore) from ENTSO-E.

    Each production type is fetched in 90-day chunks so one slow or failed request costs
    one chunk, not the whole history. A day counts only when every production type that
    returned data has that day, so a missing onshore chunk never shows up as a day of
    offshore-only (far too low) output.
    """
    parts = {}
    for psr in ("B19", "B18"):
        chunks = []
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(chunk_start + pd.Timedelta(days=_WIND_CHUNK_DAYS), end)
            try:
                gen = client.query_generation(eic, start=chunk_start, end=chunk_end, psr_type=psr)
                if isinstance(gen, pd.DataFrame):
                    if isinstance(gen.columns, pd.MultiIndex):
                        gen.columns = ["_".join(str(c) for c in col).strip() for col in gen.columns]
                    gen = gen.iloc[:, 0]
                chunks.append(pd.to_numeric(gen, errors="coerce").dropna())
            except Exception:
                pass
            chunk_start = chunk_end
        if chunks:
            series = pd.concat(chunks)
            series = series[~series.index.duplicated(keep="last")]
            daily = daily_energy(series, tz="UTC") / 1000   # hourly-mean MW summed → GWh
            if not daily.empty:
                parts[psr] = daily
    if not parts:
        return pd.Series(dtype=float)
    frame = pd.concat(parts, axis=1)
    return frame.dropna(how="any").sum(axis=1)


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
            ts     = pd.to_datetime(data["unix_seconds"], unit="s", utc=True)
            series = pd.Series(data["power"], index=ts, dtype=float).dropna()  # MW
            daily  = daily_energy(series, tz="UTC") / 1000  # GWh per UTC day
            parts.append(daily)
        except Exception:
            continue

    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts, axis=1).sum(axis=1, min_count=1).dropna()


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


# ── Wind forecast (ENTSO-E A69) ───────────────────────────────────────────────

# Bidding zones for forecast — EIC codes
_FORECAST_ZONES = {
    "DE":  "10Y1001A1001A83F",
    "DK1": "10YDK-1--------W",
    "NO":  "10YNO-0--------C",
    "GB":  "10YGB----------A",
}


def _entsoe_wind_forecast(client, eic: str, start, end) -> pd.Series:
    """
    Fetch ENTSO-E A69 day-ahead wind generation forecast for one zone.
    Queries onshore (B19) + offshore (B18) and returns daily GWh forecast total.
    """
    parts = []
    for psr in ("B19", "B18"):
        try:
            fc = client.query_wind_and_solar_forecast(
                eic, start=start, end=end, psr_type=psr,
            )
            if fc is None or (hasattr(fc, "empty") and fc.empty):
                continue
            if isinstance(fc, pd.DataFrame):
                series = fc.iloc[:, 0]
            else:
                series = fc
            series = pd.to_numeric(series, errors="coerce").dropna()
            daily  = daily_energy(series, tz="UTC") / 1000   # GWh per UTC day
            parts.append(daily)
        except Exception:
            continue
    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts, axis=1).sum(axis=1, min_count=1).dropna()


@st.cache_data(ttl=3600)
def fetch_wind_forecast(days: int = 90) -> dict:
    """
    Fetch ENTSO-E A69 day-ahead wind generation forecasts for DE, DK1, NO, GB.

    Returns
    -------
    dict:
        forecast : pd.DataFrame — columns: date, <zone>_forecast_gwh per zone
        zones    : list[str] — zones that returned data
        source   : str — data source label
    """
    client = _client()
    if client is None:
        return {"forecast": pd.DataFrame(), "zones": [], "source": "unavailable"}

    today  = pd.Timestamp.now(tz="UTC").normalize()
    start  = today - pd.Timedelta(days=days)

    results: dict[str, pd.Series] = {}
    for zone, eic in _FORECAST_ZONES.items():
        series = _entsoe_wind_forecast(client, eic, start, today)
        if not series.empty:
            results[zone] = series

    if not results:
        return {"forecast": pd.DataFrame(), "zones": [], "source": "unavailable"}

    # Build wide DataFrame
    df = pd.DataFrame({f"{z}_forecast_gwh": s for z, s in results.items()})
    df.index.name = "date"
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date").reset_index(drop=True)

    return {
        "forecast": df,
        "zones":    list(results.keys()),
        "source":   "ENTSO-E A69",
    }


@st.cache_data(ttl=3600)
def fetch_de_renewables_today() -> dict:
    """Today's mean German wind and solar output expected by ENTSO-E (A69 day-ahead), in GW.

    Returns {"wind": GW, "solar": GW, "date": "YYYY-MM-DD"} or {} without a key or data.
    """
    client = _client()
    if client is None:
        return {}
    start = pd.Timestamp.now(tz="Europe/Berlin").normalize()
    try:
        fc = client.query_wind_and_solar_forecast(DE_EIC, start=start, end=start + pd.Timedelta(days=1))
    except Exception:
        return {}
    if fc is None or fc.empty:
        return {}
    if isinstance(fc, pd.Series):
        fc = fc.to_frame()
    cols = {str(c).lower(): c for c in fc.columns}
    wind = [c for k, c in cols.items() if "wind" in k]
    solar = [c for k, c in cols.items() if "solar" in k]
    out = {"date": str(start.date())}
    if wind:
        out["wind"] = round(float(fc[wind].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1).mean()) / 1000, 1)
    if solar:
        out["solar"] = round(float(pd.to_numeric(fc[solar[0]], errors="coerce").mean()) / 1000, 1)
    return out if len(out) > 1 else {}
