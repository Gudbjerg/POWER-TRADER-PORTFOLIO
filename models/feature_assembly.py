"""
Feature assembly pipeline for ML models 1 and 2 and the pages that read the same matrix.

Joins all available data sources into a single daily feature matrix:
  - Day-ahead prices NO2 and NL  (ENTSO-E A44; Nord Pool fills recent days ENTSO-E lacks
                                  and replaces it without a key)
  - TTF front-month              (Yahoo Finance)
  - EU gas storage fill %        (GIE AGSI+, needs AGSI_API_KEY)
  - Norwegian hydro reservoirs   (ENTSO-E B31, weekly; each figure holds for its week)
  - German wind generation       (ENTSO-E B18/B19; energy-charts.info fallback)

Wind is treated as optional: models train without it and use it when it is available.
Prices are time-weighted averages over each delivery day in Central European time, the
day the Nordic and Continental day-ahead auctions clear, whatever the market time unit.

Persistence. The raw daily inputs (one row per calendar day) are stored in
features_raw_v2.csv, in DATA_PERSIST_DIR when that is set and in models/data otherwise; a
copy bundled in models/data seeds a fresh server. With a cache in place only the days from
REFRESH_OVERLAP_DAYS before its last complete day are fetched again, which also picks up
late publications and revisions. The derived features are recomputed over the whole
history on every assembly, so rolling windows are the same for old and new rows.
Version 1 of the cache (features_cache.csv) held wind energy four times too high (15-minute
readings summed as hourly energy) and prices averaged over UTC days; it is no longer read.

Pages read the matrix through load_feature_matrix(), cached for an hour and shared across
pages and sessions, so one assembly serves every page that needs it.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import pandas as pd

from config.settings import ENTSOE_TIMEOUT_S
from data.energy_units import daily_mean

# Lazy Streamlit import: this module can be imported outside Streamlit too
try:
    import streamlit as st
    _HAS_ST = True
except ImportError:
    _HAS_ST = False

FEATURE_SCHEMA       = 2              # stored with the cache; bump when a raw column changes meaning
PRICE_TZ             = "Europe/Oslo"  # delivery days of the day-ahead auctions (CET/CEST)
REFRESH_OVERLAP_DAYS = 14             # days re-fetched before the last complete cached day
MIN_WIND_WINDOW_DAYS = 90             # shortest wind request (the Quant page uses the same window)
NORDPOOL_MAX_DAYS    = 90             # Nord Pool serves recent delivery days only

_DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_PERSIST_DIR = os.getenv("DATA_PERSIST_DIR", _DATA_DIR)
_RAW_NAME    = f"features_raw_v{FEATURE_SCHEMA}.csv"
_RAW_FILE    = os.path.join(_PERSIST_DIR, _RAW_NAME)
_SEED_FILE   = os.path.join(_DATA_DIR, _RAW_NAME)   # copy shipped with a deployment
_META_FILE   = os.path.join(_PERSIST_DIR, "features_meta.json")

os.makedirs(_DATA_DIR, exist_ok=True)

_SPOT_ZONES = {
    "no2": "10YNO-2--------T",
    "nl":  "10YNL----------L",
}

# ── Feature definitions ──────────────────────────────────────────────────────

# Raw daily inputs, as stored in the cache
RAW_COLUMNS = ["no2", "nl", "ttf", "storage_fill", "hydro_twh", "de_wind_gwh"]
REQUIRED_RAW = ["no2", "nl", "ttf"]

# Features always included when source data is available
CORE_FEATURES = [
    "no2",            # NO2 day-ahead price (€/MWh)
    "nl",             # NL day-ahead price (€/MWh)
    "ttf",            # TTF front-month (€/MWh)
    "no_nl_spread",   # NO2 − NL (€/MWh) — cross-border arbitrage
    "ttf_nl_ratio",   # TTF / NL — gas-to-power cost ratio
    "no2_vol5",       # NO2 5-day rolling std (volatility proxy)
    "nl_vol5",        # NL 5-day rolling std
    "dow",            # day of week (0=Mon)
    "month",          # calendar month (1–12)
    "week",           # ISO week (1–53)
]

# Included when AGSI key available
STORAGE_FEATURES = ["storage_fill", "storage_dev"]

# Included when hydro available (ENTSO-E B31)
HYDRO_FEATURES = ["hydro_twh"]

# Included when wind available (ENTSO-E B18/B19 or Fraunhofer fallback)
WIND_FEATURES = ["de_wind_gwh"]

# HMM uses z-scored fundamentals only (no calendar, no ratio)
HMM_BASE_FEATURES = [
    "no2_z", "nl_z", "spread_z", "ttf_z",
    "storage_dev",   # already a deviation from seasonal norm
    "no2_vol5_z",
]
HMM_HYDRO_FEATURES = ["hydro_pct"]   # hydro as % of max observed
HMM_WIND_FEATURES  = ["wind_de_z"]   # z-scored wind GWh

_OPTIONAL_FILLED = ["storage_fill", "storage_dev", "hydro_twh", "hydro_pct",
                    "de_wind_gwh", "wind_de_z"]


# ── Data loading helpers ─────────────────────────────────────────────────────
# Each loader returns a frame indexed by naive daily dates, or an empty frame.

def _by_date(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col]).dt.normalize()
    out = out.set_index(date_col).sort_index()
    out.index.name = "date"
    return out[~out.index.duplicated(keep="last")]


def _day(ts: pd.Timestamp) -> pd.Timestamp:
    """Naive calendar day of a timestamp, read on its own wall clock."""
    ts = pd.Timestamp(ts)
    return (ts.tz_localize(None) if ts.tzinfo is not None else ts).normalize()


def _local_midnight(day: pd.Timestamp) -> pd.Timestamp:
    """Start of a delivery day as a tz-aware timestamp (midnight always exists in CET)."""
    return _day(day).tz_localize(PRICE_TZ)


def _load_spot_entsoe(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Daily average day-ahead price (EUR/MWh) for NO2 and NL from ENTSO-E A44.

    ``start`` and ``end`` are tz-aware delivery-day boundaries. Requests are split into
    chunks of 90 days (a year of 15-minute prices can exceed the request timeout). Returns columns ``no2`` and ``nl`` indexed
    by delivery day; empty without a key or when ENTSO-E returns nothing.
    """
    try:
        from entsoe import EntsoePandasClient
    except ImportError:
        return pd.DataFrame()

    key = os.getenv("ENTSOE_API_KEY", "")
    if not key:
        return pd.DataFrame()

    client = EntsoePandasClient(api_key=key, timeout=ENTSOE_TIMEOUT_S)
    columns: dict[str, pd.Series] = {}
    for name, eic in _SPOT_ZONES.items():
        chunks: list[pd.Series] = []
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(chunk_start + pd.Timedelta(days=90), end)   # long 15-minute requests time out
            try:
                s = client.query_day_ahead_prices(eic, start=chunk_start, end=chunk_end)
                if s is not None and not s.empty:
                    chunks.append(s)
            except Exception:
                pass
            chunk_start = chunk_end
        if not chunks:
            continue
        prices = pd.concat(chunks)
        prices = prices[~prices.index.duplicated(keep="last")]
        daily = daily_mean(pd.to_numeric(prices, errors="coerce").dropna(), tz=PRICE_TZ)
        if not daily.empty:
            columns[name] = daily

    if not columns:
        return pd.DataFrame()
    wide = pd.concat(columns, axis=1)
    wide.index.name = "date"
    return wide.sort_index()


def _load_spot_nordpool(days: int) -> pd.DataFrame:
    """Nord Pool daily averages for NO2 and NL over the last ``days`` days (at most 90)."""
    from data.spot_prices import fetch_spot_prices
    raw = fetch_spot_prices(days=min(days, NORDPOOL_MAX_DAYS))
    if raw.empty:
        return pd.DataFrame()
    wide = raw.pivot_table(index="date", columns="zone", values="price_eur_mwh", aggfunc="mean")
    wide = wide.rename(columns={"NO2": "no2", "NL": "nl"})
    wide = wide[[c for c in ("no2", "nl") if c in wide.columns]]
    if wide.empty:
        return pd.DataFrame()
    wide.columns.name = None
    return _by_date(wide.reset_index())


def _load_spot(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Daily NO2 and NL prices for delivery days in [start, end).

    ENTSO-E is the primary source. Nord Pool replaces it without a key and fills the
    recent days ENTSO-E has not published (Nord Pool serves only recent days).
    """
    spot = _load_spot_entsoe(start, end)
    first_day, last_day = _day(start), _day(end) - pd.Timedelta(days=1)
    if last_day < first_day:
        return pd.DataFrame()

    # First delivery day from which ENTSO-E has no complete pair of prices
    missing_from = first_day
    if {"no2", "nl"} <= set(spot.columns):
        complete = spot.dropna(subset=["no2", "nl"])
        if not complete.empty:
            missing_from = complete.index.max() + pd.Timedelta(days=1)
    if missing_from <= last_day:
        nordpool = _load_spot_nordpool((last_day - missing_from).days + 1)
        if not nordpool.empty:
            spot = nordpool if spot.empty else spot.combine_first(nordpool)
    if spot.empty:
        return spot
    return spot[(spot.index >= first_day) & (spot.index <= last_day)]


def _load_ttf() -> pd.DataFrame:
    """TTF front-month daily close (EUR/MWh); the seven-year request other pages share."""
    from models.ttf_backtest import fetch_ttf_history
    df = fetch_ttf_history(years=7)
    if df.empty or "price" not in df.columns:
        return pd.DataFrame()
    return _by_date(df[["date", "price"]].rename(columns={"price": "ttf"}))


def _load_storage() -> pd.DataFrame:
    """EU aggregate storage fill (%); the history window the storage pages share."""
    from data.gas_storage import fetch_storage_eu, history_window
    df = fetch_storage_eu(*history_window())
    if df.empty or "gasDayStart" not in df.columns or "full" not in df.columns:
        return pd.DataFrame()
    out = pd.DataFrame({
        "date":         pd.to_datetime(df["gasDayStart"]),
        "storage_fill": pd.to_numeric(df["full"], errors="coerce"),
    }).dropna()
    return _by_date(out) if not out.empty else pd.DataFrame()


def _load_hydro() -> pd.DataFrame:
    """Norwegian reservoir filling (TWh), weekly, held for the seven days of each week."""
    from data.hydro import fetch_hydro_reservoirs
    df = fetch_hydro_reservoirs()
    if df.empty or "filling_twh" not in df.columns:
        return pd.DataFrame()
    weekly = _by_date(df[["week_start", "filling_twh"]]
                      .rename(columns={"week_start": "date", "filling_twh": "hydro_twh"}))
    days = pd.date_range(weekly.index.min(), weekly.index.max() + pd.Timedelta(days=6), freq="D")
    daily = weekly.reindex(days).ffill(limit=6)
    daily.index.name = "date"
    return daily.dropna()


def _load_wind(days: int) -> pd.DataFrame:
    """German wind generation (GWh per day) over the last ``days`` days."""
    from data.wind import fetch_wind_daily
    df = fetch_wind_daily(days=days).get("daily", pd.DataFrame())
    if df.empty or "de_wind_gwh" not in df.columns:
        return pd.DataFrame()
    return _by_date(df[["date", "de_wind_gwh"]].dropna(subset=["de_wind_gwh"]))


def _fetch_inputs(start: pd.Timestamp, end: pd.Timestamp, history_start: pd.Timestamp) -> pd.DataFrame:
    """Raw daily inputs: prices and wind for [start, end), the shared histories from history_start."""
    lo_hist = _day(history_start)
    last_day = _day(end) - pd.Timedelta(days=1)
    wind_days = max(MIN_WIND_WINDOW_DAYS, (_day(end) - _day(start)).days + 1)

    parts = [_load_spot(start, end), _load_ttf(), _load_storage(), _load_hydro(), _load_wind(wind_days)]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame()
    raw = pd.concat(parts, axis=1)
    raw = raw[(raw.index >= lo_hist) & (raw.index <= last_day)]
    return raw.reindex(columns=[c for c in RAW_COLUMNS if c in raw.columns])


# ── Feature engineering ──────────────────────────────────────────────────────

def _rolling_zscore(series: pd.Series, window: int = 90) -> pd.Series:
    mean = series.rolling(window, min_periods=30).mean()
    std  = series.rolling(window, min_periods=30).std()
    return ((series - mean) / std.replace(0, float("nan"))).round(4)


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all derived features from the joined base DataFrame."""
    df = df.copy()

    # Core derived
    df["no_nl_spread"] = df["no2"] - df["nl"]
    df["ttf_nl_ratio"] = (df["ttf"] / df["nl"].replace(0, float("nan"))).clip(0, 10)
    df["no2_vol5"]     = df["no2"].rolling(5, min_periods=3).std()
    df["nl_vol5"]      = df["nl"].rolling(5, min_periods=3).std()

    # Calendar
    df["dow"]   = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["week"]  = df["date"].dt.isocalendar().week.astype(int)

    # Storage deviation from seasonal mean (expanding mean of the same day of year)
    if "storage_fill" in df.columns:
        df["doy"] = df["date"].dt.dayofyear
        seasonal_mean = (
            df.groupby("doy")["storage_fill"]
            .transform(lambda x: x.expanding(min_periods=2).mean())
        )
        df["storage_dev"] = (df["storage_fill"] - seasonal_mean).round(2)
        df = df.drop(columns=["doy"])

    # Hydro as % of rolling expanding max (no lookahead — only past observations used)
    if "hydro_twh" in df.columns:
        expanding_max = df["hydro_twh"].expanding(min_periods=10).quantile(0.98)
        expanding_max = expanding_max.replace(0, float("nan"))
        df["hydro_pct"] = (df["hydro_twh"] / expanding_max * 100).clip(0, 100).round(2)

    # HMM z-scored features
    df["no2_z"]      = _rolling_zscore(df["no2"])
    df["nl_z"]       = _rolling_zscore(df["nl"])
    df["spread_z"]   = _rolling_zscore(df["no_nl_spread"])
    df["ttf_z"]      = _rolling_zscore(df["ttf"])
    df["no2_vol5_z"] = _rolling_zscore(df["no2_vol5"])

    if "de_wind_gwh" in df.columns:
        df["wind_de_z"] = _rolling_zscore(df["de_wind_gwh"])

    return df


def features_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix from the raw daily inputs (indexed by date, one row per calendar day).

    Rolling windows run over calendar days; rows without a price or a TTF settlement
    (weekends, holidays) are dropped afterwards, so the matrix holds trading days.
    """
    if raw.empty or not set(REQUIRED_RAW) <= set(raw.columns):
        empty = pd.DataFrame()
        empty.attrs["coverage_report"] = get_available_feature_sets(empty)
        return empty

    base = raw.sort_index().reset_index()
    base = base.rename(columns={base.columns[0]: "date"})
    base["date"] = pd.to_datetime(base["date"])
    base = _compute_features(base)

    # Forward-fill short gaps in optional features: five days, except hydro, which ENTSO-E
    # publishes weekly with a lag of up to two weeks and which moves slowly.
    for col in _OPTIONAL_FILLED:
        if col in base.columns:
            base[col] = base[col].ffill(limit=14 if col.startswith("hydro") else 5)

    base = base.dropna(subset=REQUIRED_RAW).reset_index(drop=True)
    base.attrs["coverage_report"] = get_available_feature_sets(base)
    return base


# ── Main assembly function ───────────────────────────────────────────────────

def _today_local() -> pd.Timestamp:
    """Today's delivery day (naive date) on the auction clock."""
    return _day(pd.Timestamp.now(tz=PRICE_TZ))


def assemble_features(years: int = 3, use_cache: bool = True) -> pd.DataFrame:
    """
    Build the daily feature matrix from all available sources.

    Parameters
    ----------
    years     : Years of history requested when there is no cache to extend.
    use_cache : Start from the persisted raw inputs and fetch only the recent days.

    Returns
    -------
    pd.DataFrame with a date column and all computed features, one row per trading day
    up to today's delivery day. Rows without NO2, NL or TTF are dropped. The frame's
    ``attrs["coverage_report"]`` says which feature groups are populated.
    """
    end_day = _today_local() + pd.Timedelta(days=1)        # exclusive: includes today's delivery day
    full_start_day = end_day - pd.Timedelta(days=int(years * 365))

    cached = _load_raw_cache() if use_cache else pd.DataFrame()
    start_day = full_start_day
    if not cached.empty:
        complete = cached.dropna(subset=REQUIRED_RAW)
        if not complete.empty:
            resume = complete.index.max() - pd.Timedelta(days=REFRESH_OVERLAP_DAYS)
            start_day = max(full_start_day, resume)

    fresh = _fetch_inputs(_local_midnight(start_day), _local_midnight(end_day),
                          history_start=_local_midnight(full_start_day))
    if fresh.empty:
        raw = cached
    elif cached.empty:
        raw = fresh
    else:
        raw = fresh.combine_first(cached)
    if raw.empty:
        return features_from_raw(pd.DataFrame())
    raw = raw.reindex(columns=[c for c in RAW_COLUMNS if c in raw.columns]).sort_index()
    raw.index.name = "date"

    if not fresh.empty and use_cache:
        _save_raw_cache(raw)
    return features_from_raw(raw)


def _cache_data(**kwargs):
    """st.cache_data when Streamlit is available, otherwise no caching."""
    if _HAS_ST:
        return st.cache_data(**kwargs)
    return lambda fn: fn


@_cache_data(ttl=3600, show_spinner=False)
def load_feature_matrix(years: int = 3) -> pd.DataFrame:
    """The daily feature matrix for the pages: one assembly per hour, shared by all pages."""
    return assemble_features(years=years)


# ── Cache persistence ────────────────────────────────────────────────────────

def _save_raw_cache(raw: pd.DataFrame) -> None:
    try:
        out = raw.reset_index()
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
        out.to_csv(_RAW_FILE, index=False, float_format="%.6g")
        complete = raw.dropna(subset=[c for c in REQUIRED_RAW if c in raw.columns])
        meta = {
            "schema":     FEATURE_SCHEMA,
            "updated_at": datetime.utcnow().isoformat(timespec="seconds"),
            "n_rows":     int(len(complete)),
            "date_min":   str(complete.index.min().date()) if not complete.empty else None,
            "date_max":   str(complete.index.max().date()) if not complete.empty else None,
            "columns":    list(raw.columns),
        }
        with open(_META_FILE, "w") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass


def _load_raw_cache() -> pd.DataFrame:
    """The persisted raw inputs, or the bundled copy when none has been written yet."""
    for path in dict.fromkeys((_RAW_FILE, _SEED_FILE)):
        try:
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path, parse_dates=["date"])
            if df.empty or not set(REQUIRED_RAW) <= set(df.columns):
                continue
            df = _by_date(df)
            return df.reindex(columns=[c for c in RAW_COLUMNS if c in df.columns])
        except Exception:
            continue
    return pd.DataFrame()


def get_feature_meta() -> dict:
    """Return metadata about the persisted raw inputs (for UI display)."""
    try:
        if os.path.exists(_META_FILE):
            with open(_META_FILE) as f:
                meta = json.load(f)
            if meta.get("schema") == FEATURE_SCHEMA:
                return meta
    except Exception:
        pass
    raw = _load_raw_cache()
    complete = raw.dropna(subset=REQUIRED_RAW) if not raw.empty else raw
    if complete.empty:
        return {}
    return {
        "schema":   FEATURE_SCHEMA,
        "n_rows":   int(len(complete)),
        "date_min": str(complete.index.min().date()),
        "date_max": str(complete.index.max().date()),
        "columns":  list(raw.columns),
    }


def get_available_feature_sets(df: pd.DataFrame) -> dict:
    """Inspect which feature groups are present and sufficiently populated."""
    if df.empty:
        return {k: False for k in ("core", "storage", "hydro", "wind")}
    cols = set(df.columns)
    return {
        "core":    all(c in cols for c in ["no2", "nl", "ttf"]) and len(df) >= 100,
        "storage": "storage_fill" in cols and df["storage_fill"].notna().sum() >= 50,
        "hydro":   "hydro_twh"    in cols and df["hydro_twh"].notna().sum()    >= 50,
        "wind":    "de_wind_gwh"  in cols and df["de_wind_gwh"].notna().sum()  >= 50,
    }


def get_lstm_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return ordered list of feature columns to use for LSTM training."""
    avail = get_available_feature_sets(df)
    cols = list(CORE_FEATURES)
    if avail["storage"]:
        cols += STORAGE_FEATURES
    if avail["hydro"]:
        cols += HYDRO_FEATURES
    if avail["wind"]:
        cols += WIND_FEATURES
    return [c for c in cols if c in df.columns]


def get_hmm_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return ordered list of feature columns to use for HMM training."""
    avail = get_available_feature_sets(df)
    cols = list(HMM_BASE_FEATURES)
    if "storage_dev" not in df.columns and "storage_fill" in df.columns:
        cols = [c for c in cols if c != "storage_dev"]
    if avail["hydro"]:
        cols += HMM_HYDRO_FEATURES
    if avail["wind"]:
        cols += HMM_WIND_FEATURES
    return [c for c in cols if c in df.columns]
