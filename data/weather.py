"""
Weather inputs for the Weather Desk: ECMWF forecasts and German power actuals, all keyless.

Sources
-------
* Open-Meteo (CC BY 4.0), model ``ecmwf_ifs025`` (ECMWF IFS at 0.25 degrees):
  - forecast API: the latest run, ten days ahead;
  - ensemble API: the 51-member ECMWF ensemble, for the spread of the wind outlook;
  - single-runs API: one named ECMWF run, so the newest run can be set against the run
    24 hours earlier for the same valid days (how much the outlook moved);
  - historical-forecast API: archived short-lead forecasts, used to calibrate the weather
    indices against what the German fleet produced;
  - ERA5 archive: 2016 to 2025 reanalysis, the precipitation and temperature normal for
    the Norwegian hydro catchments.
* energy-charts.info (Fraunhofer ISE; CC BY 4.0, data from SMARD/Bundesnetzagentur and
  ENTSO-E): German wind, solar and load in MW and DE-LU day-ahead prices.

Each site group is reduced to one daily index on the Europe/Berlin clock:
* wind:  mean capacity factor of a generic turbine power curve applied to 100 m wind;
* solar: mean global horizontal irradiance (kW/m2);
* heat:  heating degree days below 15.5 C (load rises in the cold);
* hydro: precipitation (mm) and temperature over western Norwegian catchments.
"""
from __future__ import annotations

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st

TZ = "Europe/Berlin"
MODEL = "ecmwf_ifs025"
TIMEOUT_S = 45
HEADERS = {"User-Agent": "power-trader-portfolio/2.3 (weather desk)"}

FORECAST_URL   = "https://api.open-meteo.com/v1/forecast"
ENSEMBLE_URL   = "https://ensemble-api.open-meteo.com/v1/ensemble"
SINGLE_RUN_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
HISTORICAL_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
ARCHIVE_URL    = "https://archive-api.open-meteo.com/v1/archive"
ENERGY_CHARTS  = "https://api.energy-charts.info"

# (label, latitude, longitude, weight). Wind weights follow where German capacity sits:
# the onshore fleet is concentrated in the north and east, offshore in the German Bight.
WIND_SITES: list[tuple[str, float, float, float]] = [
    ("Schleswig-Holstein",      54.30,  9.60, 0.13),
    ("Lower Saxony coast",      53.30,  7.80, 0.13),
    ("Lower Saxony inland",     52.70,  9.40, 0.10),
    ("Mecklenburg",             53.80, 12.20, 0.08),
    ("Brandenburg",             52.60, 13.70, 0.13),
    ("Saxony-Anhalt",           52.00, 11.60, 0.10),
    ("North Rhine-Westphalia",  51.70,  8.20, 0.07),
    ("Hesse / Thuringia",       50.90, 10.20, 0.05),
    ("North Sea offshore",      54.20,  6.60, 0.17),
    ("Baltic offshore",         54.70, 13.80, 0.04),
]
SOLAR_SITES: list[tuple[str, float, float, float]] = [
    ("Bavaria",                 48.50, 11.50, 0.30),
    ("Baden-Wurttemberg",       48.60,  9.00, 0.18),
    ("North Rhine-Westphalia",  51.50,  7.50, 0.15),
    ("Lower Saxony",            52.60,  9.50, 0.15),
    ("Brandenburg",             52.40, 13.50, 0.12),
    ("Saxony",                  51.00, 13.20, 0.10),
]
HEAT_SITES: list[tuple[str, float, float, float]] = [
    ("Berlin",     52.52, 13.40, 0.20),
    ("Hamburg",    53.55, 10.00, 0.15),
    ("Rhine-Ruhr", 51.45,  7.00, 0.30),
    ("Frankfurt",  50.11,  8.68, 0.15),
    ("Munich",     48.14, 11.58, 0.20),
]
# Western Norwegian catchments behind NO2 and NO5 (the largest reservoir volumes).
HYDRO_SITES: list[tuple[str, float, float, float]] = [
    ("Suldal / Ulla-Forre (NO2)",  59.50,  6.70, 0.25),
    ("Sira-Kvina (NO2)",           58.90,  6.90, 0.20),
    ("Hardanger / Voss (NO5)",     60.50,  6.80, 0.20),
    ("Sogn (NO5)",                 61.30,  7.30, 0.20),
    ("Telemark (NO2)",             59.80,  8.30, 0.15),
]

HDD_BASE_C = 15.5
CUT_IN_MS, RATED_MS, CUT_OUT_MS = 3.0, 12.0, 25.0


# ── Physics ───────────────────────────────────────────────────────────────────

def power_curve(speed_ms) -> np.ndarray:
    """Capacity factor of a generic modern turbine for hub-height wind speed (m/s).

    Zero below cut-in (3 m/s) and from cut-out (25 m/s), cubic between cut-in and rated
    speed (12 m/s), one at and above rated. NaN stays NaN.
    """
    v = np.asarray(speed_ms, dtype=float)
    cf = (v ** 3 - CUT_IN_MS ** 3) / (RATED_MS ** 3 - CUT_IN_MS ** 3)
    cf = np.clip(cf, 0.0, 1.0)
    cf = np.where((v < CUT_IN_MS) | (v >= CUT_OUT_MS), 0.0, cf)
    return np.where(np.isnan(v), np.nan, cf)


def _weighted(frame: pd.DataFrame, weights: list[float]) -> pd.Series:
    """Weighted mean across site columns, renormalised over the sites that reported."""
    w = pd.Series(weights, index=frame.columns, dtype=float)
    present = frame.notna()
    num = frame.fillna(0.0).mul(w, axis=1).sum(axis=1)
    den = present.mul(w, axis=1).sum(axis=1)
    return (num / den.replace(0.0, np.nan))


# ── HTTP ──────────────────────────────────────────────────────────────────────

class WeatherUnavailable(RuntimeError):
    """A weather or power source did not answer usefully."""


def _get_json(url: str, params: dict) -> dict | list:
    """GET with one retry; HTTP 429 (Open-Meteo's per-minute budget) waits before it."""
    import time
    last = "no answer"
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT_S)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}"
            if r.status_code == 429 and attempt < 2:
                time.sleep(min(float(r.headers.get("Retry-After", 20) or 20), 30.0))
                continue
            if 400 <= r.status_code < 500:
                break
        except (requests.RequestException, ValueError) as exc:
            last = type(exc).__name__
    raise WeatherUnavailable(f"{url.split('/')[2]} unavailable ({last})")


def _site_params(sites) -> dict:
    return {"latitude": ",".join(f"{s[1]:.2f}" for s in sites),
            "longitude": ",".join(f"{s[2]:.2f}" for s in sites)}


def _hourly_by_site(payload, variable: str, sites) -> pd.DataFrame:
    """One column per site for ``variable`` (hourly, local time index)."""
    items = payload if isinstance(payload, list) else [payload]
    cols = {}
    for site, item in zip(sites, items):
        h = item.get("hourly", {})
        if variable in h:
            cols[site[0]] = pd.Series(h[variable], index=pd.to_datetime(h["time"]), dtype=float)
    return pd.DataFrame(cols)


def _ensemble_by_site(payload, variable: str, sites) -> dict[str, pd.DataFrame]:
    """Per site: one column per ensemble member (control run first)."""
    items = payload if isinstance(payload, list) else [payload]
    out = {}
    for site, item in zip(sites, items):
        h = item.get("hourly", {})
        keys = [k for k in h if k == variable or k.startswith(f"{variable}_member")]
        out[site[0]] = pd.DataFrame({k: pd.Series(h[k], dtype=float) for k in keys},
                                    ).set_index(pd.to_datetime(h["time"]))
    return out


# ── Daily indices ─────────────────────────────────────────────────────────────

def _daily_indices(wind_ms: pd.DataFrame, ghi: pd.DataFrame, temp: pd.DataFrame) -> pd.DataFrame:
    """Daily wind, solar and heat indices from hourly site frames (local time)."""
    wind_cf = pd.DataFrame(power_curve(wind_ms.to_numpy()), index=wind_ms.index, columns=wind_ms.columns)
    wind_idx = _weighted(wind_cf, [s[3] for s in WIND_SITES if s[0] in wind_ms.columns])
    solar_idx = _weighted(ghi / 1000.0, [s[3] for s in SOLAR_SITES if s[0] in ghi.columns])
    t_mean = _weighted(temp, [s[3] for s in HEAT_SITES if s[0] in temp.columns])
    hourly = pd.DataFrame({"wind_idx": wind_idx, "solar_idx": solar_idx, "temp_c": t_mean})
    counts = hourly.notna().resample("D").sum()
    daily = hourly.resample("D").mean()
    daily = daily[(counts["wind_idx"] >= 20) & (counts["temp_c"] >= 20)]
    daily["hdd"] = (HDD_BASE_C - daily["temp_c"]).clip(lower=0.0)
    daily.index.name = "date"
    return daily


def _weather_request(url: str, extra: dict) -> pd.DataFrame:
    """Wind, irradiance and temperature indices for the request window in ``extra``."""
    common = {"models": MODEL, "timezone": TZ, "wind_speed_unit": "ms", **extra}
    wind = _hourly_by_site(_get_json(url, {**_site_params(WIND_SITES), "hourly": "wind_speed_100m", **common}),
                           "wind_speed_100m", WIND_SITES)
    ghi = _hourly_by_site(_get_json(url, {**_site_params(SOLAR_SITES), "hourly": "shortwave_radiation", **common}),
                          "shortwave_radiation", SOLAR_SITES)
    temp = _hourly_by_site(_get_json(url, {**_site_params(HEAT_SITES), "hourly": "temperature_2m", **common}),
                           "temperature_2m", HEAT_SITES)
    return _daily_indices(wind, ghi, temp)


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def fetch_forecast_indices(days: int = 10) -> pd.DataFrame:
    """Daily indices from the latest ECMWF run, today and the next ``days - 1`` days."""
    return _weather_request(FORECAST_URL, {"forecast_days": days})


_BUNDLED_DIR = os.path.dirname(os.path.abspath(__file__))
_HINDCAST_NAME = "weather_hindcast_v1.csv"
_HINDCAST_CSV = os.path.join(os.getenv("DATA_PERSIST_DIR", _BUNDLED_DIR), _HINDCAST_NAME)
_HINDCAST_SEED = os.path.join(_BUNDLED_DIR, _HINDCAST_NAME)   # copy shipped with a deployment
HINDCAST_OVERLAP_DAYS = 3


def _load_hindcast_cache() -> pd.DataFrame:
    for path in dict.fromkeys((_HINDCAST_CSV, _HINDCAST_SEED)):
        try:
            if os.path.exists(path):
                df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
                if {"wind_idx", "solar_idx", "temp_c", "hdd"} <= set(df.columns):
                    return df[~df.index.duplicated(keep="last")]
        except Exception:
            continue
    return pd.DataFrame()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def fetch_hindcast_indices(start: str, end: str) -> pd.DataFrame:
    """Daily indices from archived short-lead ECMWF forecasts between two dates.

    Archived forecasts do not change, so the days already fetched are kept on disk
    (``weather_hindcast_v1.csv``; a deployment ships a copy) and only the days after the
    last stored day are requested. That keeps the request inside Open-Meteo's free budget,
    which a shared cloud address otherwise exhausts. When the archive does not answer, the
    stored days are used as they are.
    """
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    cached = _load_hindcast_cache()
    fetch_from = lo
    if not cached.empty and cached.index.min() <= lo + pd.Timedelta(days=HINDCAST_OVERLAP_DAYS):
        fetch_from = max(lo, cached.index.max() - pd.Timedelta(days=HINDCAST_OVERLAP_DAYS))
    combined = cached
    if fetch_from <= hi:
        try:
            fresh = _weather_request(HISTORICAL_URL, {"start_date": fetch_from.date().isoformat(),
                                                      "end_date": hi.date().isoformat()})
            combined = pd.concat([cached, fresh]) if not cached.empty else fresh
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
            try:
                combined.round(5).to_csv(_HINDCAST_CSV, index_label="date")
            except OSError:
                pass
        except WeatherUnavailable:
            if cached.empty:
                raise
    out = combined[(combined.index >= lo) & (combined.index <= hi)]
    if len(out) < 60:
        raise WeatherUnavailable(f"only {len(out)} days of archived forecasts available")
    return out


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def fetch_wind_ensemble(days: int = 10) -> pd.DataFrame:
    """Daily wind index per ECMWF ensemble member (rows: days, columns: members)."""
    payload = _get_json(ENSEMBLE_URL, {**_site_params(WIND_SITES), "hourly": "wind_speed_100m",
                                       "models": MODEL, "timezone": TZ, "wind_speed_unit": "ms",
                                       "forecast_days": days})
    per_site = _ensemble_by_site(payload, "wind_speed_100m", WIND_SITES)
    members = sorted(set.intersection(*(set(f.columns) for f in per_site.values())))
    weights = [s[3] for s in WIND_SITES if s[0] in per_site]
    out = {}
    for m in members:
        cf = pd.DataFrame({site: power_curve(frame[m].to_numpy()) for site, frame in per_site.items()},
                          index=next(iter(per_site.values())).index)
        out[m] = _weighted(cf, weights)
    hourly = pd.DataFrame(out)
    counts = hourly.notna().resample("D").sum()
    daily = hourly.resample("D").mean()[counts.iloc[:, 0] >= 20]
    daily.index.name = "date"
    return daily


def _run_wind_index(run: pd.Timestamp, days: int) -> pd.Series:
    """Daily wind index from one ECMWF run (single-runs API), valid days from the run date."""
    payload = _get_json(SINGLE_RUN_URL, {**_site_params(WIND_SITES), "hourly": "wind_speed_100m",
                                         "models": MODEL, "timezone": TZ, "wind_speed_unit": "ms",
                                         "run": run.strftime("%Y-%m-%dT%H:%M"), "forecast_days": days})
    frame = _hourly_by_site(payload, "wind_speed_100m", WIND_SITES)
    cf = pd.DataFrame(power_curve(frame.to_numpy()), index=frame.index, columns=frame.columns)
    hourly = _weighted(cf, [s[3] for s in WIND_SITES])
    counts = hourly.notna().resample("D").sum()
    return hourly.resample("D").mean()[counts >= 20]


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def fetch_wind_revision(days: int = 7) -> pd.DataFrame:
    """Daily wind index from the newest available ECMWF run and from the run 24 hours earlier.

    Runs start at 00 and 12 UTC and reach the archive a few hours later, so the newest run
    is found by stepping back in 12-hour steps. Both runs are read for the same valid days,
    from today onwards. ``attrs`` carries the two run times.
    """
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    candidate = now.floor("12h")
    latest = None
    for _ in range(5):
        try:
            latest = _run_wind_index(candidate, days + 2)
            break
        except WeatherUnavailable:
            candidate -= pd.Timedelta(hours=12)
    if latest is None:
        raise WeatherUnavailable("no recent ECMWF run available")
    previous_run = candidate - pd.Timedelta(hours=24)
    previous = _run_wind_index(previous_run, days + 3)
    today = pd.Timestamp.now(tz=TZ).tz_localize(None).normalize()
    out = pd.DataFrame({"latest": latest, "previous": previous}).dropna()
    out = out[out.index >= today].iloc[:days]
    out.index.name = "date"
    out.attrs["latest_run"] = candidate
    out.attrs["previous_run"] = previous_run
    return out


# ── Norwegian hydro catchments ────────────────────────────────────────────────

@st.cache_data(ttl=3 * 3600, show_spinner=False)
def fetch_hydro_outlook(days: int = 10) -> pd.DataFrame:
    """Daily catchment precipitation (mm) and temperature (C) for the next ``days`` days."""
    payload = _get_json(FORECAST_URL, {**_site_params(HYDRO_SITES), "daily": "precipitation_sum,temperature_2m_mean",
                                       "models": MODEL, "timezone": "Europe/Oslo", "forecast_days": days})
    items = payload if isinstance(payload, list) else [payload]
    precip, temp = {}, {}
    for site, item in zip(HYDRO_SITES, items):
        d = item.get("daily", {})
        idx = pd.to_datetime(d.get("time", []))
        precip[site[0]] = pd.Series(d.get("precipitation_sum", []), index=idx, dtype=float)
        temp[site[0]] = pd.Series(d.get("temperature_2m_mean", []), index=idx, dtype=float)
    weights = [s[3] for s in HYDRO_SITES]
    out = pd.DataFrame({"precip_mm": _weighted(pd.DataFrame(precip), weights),
                        "temp_c": _weighted(pd.DataFrame(temp), weights)})
    out.index.name = "date"
    return out


_NORMALS_NAME = "weather_hydro_normals_v1.csv"
_NORMALS_SEED = os.path.join(_BUNDLED_DIR, _NORMALS_NAME)
CLIMATE_YEARS = (2016, 2025)


def era5_hydro_daily(start_year: int = CLIMATE_YEARS[0], end_year: int = CLIMATE_YEARS[1]) -> pd.DataFrame:
    """ERA5 daily catchment precipitation and temperature, weighted across the sites."""
    payload = _get_json(ARCHIVE_URL, {**_site_params(HYDRO_SITES),
                                      "start_date": f"{start_year}-01-01", "end_date": f"{end_year}-12-31",
                                      "daily": "precipitation_sum,temperature_2m_mean", "timezone": "Europe/Oslo"})
    items = payload if isinstance(payload, list) else [payload]
    precip, temp = {}, {}
    for site, item in zip(HYDRO_SITES, items):
        d = item.get("daily", {})
        idx = pd.to_datetime(d.get("time", []))
        precip[site[0]] = pd.Series(d.get("precipitation_sum", []), index=idx, dtype=float)
        temp[site[0]] = pd.Series(d.get("temperature_2m_mean", []), index=idx, dtype=float)
    weights = [s[3] for s in HYDRO_SITES]
    return pd.DataFrame({"precip_mm": _weighted(pd.DataFrame(precip), weights),
                         "temp_c": _weighted(pd.DataFrame(temp), weights)})


def normals_from_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Mean per calendar day (MM-DD); 29 February folds onto 28 February."""
    md = daily.index.strftime("%m-%d").str.replace("02-29", "02-28")
    out = daily.groupby(md).mean()
    out.index.name = "month_day"
    return out


@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False)
def fetch_hydro_normals() -> pd.DataFrame:
    """Calendar-day normals (ERA5 2016 to 2025) of catchment precipitation and temperature.

    A normal does not change, so the deployment ships it (``weather_hydro_normals_v1.csv``)
    and ERA5 is queried only when that file is missing.
    """
    try:
        if os.path.exists(_NORMALS_SEED):
            df = pd.read_csv(_NORMALS_SEED, dtype={"month_day": str}).set_index("month_day")
            if len(df) >= 365 and {"precip_mm", "temp_c"} <= set(df.columns):
                return df
    except Exception:
        pass
    return normals_from_daily(era5_hydro_daily())


# ── German actuals (energy-charts.info) ───────────────────────────────────────

def _ec_daily(unix_seconds, values) -> pd.Series:
    s = pd.Series(values, index=pd.to_datetime(unix_seconds, unit="s", utc=True), dtype=float)
    s = s.tz_convert(TZ)
    hourly = s.resample("h").mean()
    counts = hourly.notna().resample("D").sum()
    daily = hourly.resample("D").mean()[counts >= 23]
    daily.index = daily.index.tz_localize(None)
    return daily


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_de_actuals(start: str, end: str) -> pd.DataFrame:
    """Daily mean German wind, solar and load (GW) and DE-LU day-ahead price (EUR/MWh)."""
    power = _get_json(f"{ENERGY_CHARTS}/public_power", {"country": "de", "start": start, "end": end})
    series = {}
    for item in power.get("production_types", []):
        name = item["name"]["en"] if isinstance(item["name"], dict) else item["name"]
        series[name] = item["data"]
    ts = power.get("unix_seconds", [])

    def gw(*names):
        parts = [_ec_daily(ts, series[n]) for n in names if n in series]
        return sum(parts) / 1000.0 if parts else pd.Series(dtype=float)

    out = pd.DataFrame({
        "wind_gw":  gw("Wind onshore", "Wind offshore"),
        "solar_gw": gw("Solar"),
        "load_gw":  gw("Load"),
    })
    price = _get_json(f"{ENERGY_CHARTS}/price", {"bzn": "DE-LU", "start": start, "end": end})
    out["price"] = _ec_daily(price.get("unix_seconds", []), price.get("price", []))
    out.index.name = "date"
    return out


def calibration_window(days: int = 120) -> tuple[str, str]:
    """(start, end) dates of the calibration sample, ending yesterday."""
    end = date.today() - timedelta(days=1)
    return (end - timedelta(days=days - 1)).isoformat(), end.isoformat()
