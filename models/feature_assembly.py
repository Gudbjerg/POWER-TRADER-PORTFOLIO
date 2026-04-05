"""
Feature assembly pipeline for ML models 1 and 2.

Joins all available data sources into a single daily feature matrix:
  - Nord Pool day-ahead prices  (NO2, NL — always available)
  - TTF gas price               (always available via yfinance)
  - EU gas storage fill %       (available with AGSI_API_KEY)
  - Norwegian hydro reservoirs  (available with ENTSOE_API_KEY, weekly → daily)
  - German wind generation      (available with ENTSOE_API_KEY; Fraunhofer ISE fallback)

Wind is treated as optional — models are designed to train without it and
upgrade automatically when it becomes available.

Persists assembled features to models/data/features_cache.csv so training
data survives ENTSO-E outages and app restarts.
"""
from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Lazy Streamlit import — this module can be imported outside Streamlit too
try:
    import streamlit as st
    _HAS_ST = True
except ImportError:
    _HAS_ST = False

_DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
_CACHE_FILE  = os.path.join(_DATA_DIR, "features_cache.csv")
_META_FILE   = os.path.join(_DATA_DIR, "features_meta.json")

os.makedirs(_DATA_DIR, exist_ok=True)

# ── Feature definitions ──────────────────────────────────────────────────────

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

# Included when wind available (ENTSO-E B16/B18/B19 or Fraunhofer fallback)
WIND_FEATURES = ["de_wind_gwh"]

# HMM uses z-scored fundamentals only (no calendar, no ratio)
HMM_BASE_FEATURES = [
    "no2_z", "nl_z", "spread_z", "ttf_z",
    "storage_dev",   # already a deviation from seasonal norm
    "no2_vol5_z",
]
HMM_HYDRO_FEATURES = ["hydro_pct"]   # hydro as % of max observed
HMM_WIND_FEATURES  = ["wind_de_z"]   # z-scored wind GWh


# ── Data loading helpers ─────────────────────────────────────────────────────

def _load_spot_entsoe(years: int) -> pd.DataFrame:
    """
    Fetch daily average day-ahead prices for NO2 and NL from ENTSO-E A44.
    Queries in annual chunks (ENTSO-E limit) and concatenates.
    Returns DataFrame with columns: date, NO2, NL.
    """
    import os
    try:
        from entsoe import EntsoePandasClient
    except ImportError:
        return pd.DataFrame()

    key = os.getenv("ENTSOE_API_KEY", "")
    if not key:
        return pd.DataFrame()

    client = EntsoePandasClient(api_key=key)
    end   = pd.Timestamp.now(tz="UTC").normalize()
    start = end - pd.Timedelta(days=int(years * 365))

    zones = {
        "NO2": "10YNO-2--------T",
        "NL":  "10YNL----------L",
    }

    results: dict[str, pd.Series] = {}
    for zone_name, eic in zones.items():
        chunks: list[pd.Series] = []
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(chunk_start + pd.Timedelta(days=365), end)
            try:
                s = client.query_day_ahead_prices(eic, start=chunk_start, end=chunk_end)
                if s is not None and not s.empty:
                    chunks.append(s)
            except Exception:
                pass
            chunk_start = chunk_end
        if chunks:
            combined = pd.concat(chunks)
            combined = combined[~combined.index.duplicated(keep="last")]
            results[zone_name] = combined

    if not results:
        return pd.DataFrame()

    frames = []
    for zone_name, series in results.items():
        daily = (
            series
            .tz_convert("UTC")
            .resample("D")
            .mean()
            .rename(zone_name)
        )
        frames.append(daily)

    wide = pd.concat(frames, axis=1)
    wide.index = wide.index.tz_localize(None)
    wide.index.name = "date"
    wide = wide.reset_index()
    wide["date"] = pd.to_datetime(wide["date"]).dt.normalize()
    return wide.dropna(how="all", subset=["NO2", "NL"])


def _load_spot(days: int) -> pd.DataFrame:
    """
    Returns wide-format daily prices: date | NO2 | NL.
    Tries ENTSO-E A44 first (years of history); falls back to Nord Pool
    for recent data or when no ENTSO-E key is available.
    """
    years = max(1, days // 365)

    # Primary: ENTSO-E A44 — returns years of data in a few chunked calls
    df = _load_spot_entsoe(years)
    if not df.empty and len(df) > 100:
        return df

    # Fallback: Nord Pool public API (recent data only, ~30–90 days reliable)
    from data.spot_prices import fetch_spot_prices
    raw = fetch_spot_prices(days=min(days, 90))
    if raw.empty:
        return df  # return whatever ENTSO-E gave us, even if sparse

    wide = raw.pivot_table(index="date", columns="zone", values="price_eur_mwh", aggfunc="mean")
    wide.index = pd.to_datetime(wide.index)
    wide.columns.name = None
    wide = wide.reset_index().rename(columns={"index": "date"})

    # Merge ENTSO-E history with fresh Nord Pool rows
    if not df.empty:
        combined = pd.concat([df, wide], ignore_index=True)
        combined = (
            combined
            .sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True)
        )
        return combined

    return wide


def _load_ttf(years: int) -> pd.DataFrame:
    from models.ttf_backtest import fetch_ttf_history
    df = fetch_ttf_history(years=years)
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "price"]].rename(columns={"price": "ttf"})


def _load_storage(years: int) -> pd.DataFrame:
    from data.gas_storage import fetch_storage_eu
    today    = datetime.utcnow().date()
    date_from = (today - timedelta(days=365 * years)).strftime("%Y-%m-%d")
    date_to   = today.strftime("%Y-%m-%d")
    df = fetch_storage_eu(date_from, date_to)
    if df.empty or "gasDayStart" not in df.columns:
        return pd.DataFrame()
    out = df[["gasDayStart", "full"]].copy()
    out["date"] = pd.to_datetime(out["gasDayStart"]).dt.normalize()
    out["storage_fill"] = pd.to_numeric(out["full"], errors="coerce")
    return out[["date", "storage_fill"]].dropna()


def _load_hydro(years: int) -> pd.DataFrame:
    from data.hydro import fetch_hydro_reservoirs
    df = fetch_hydro_reservoirs(years=years)
    if df.empty or "filling_twh" not in df.columns:
        return pd.DataFrame()
    out = df[["week_start", "filling_twh"]].copy()
    out["date"] = pd.to_datetime(out["week_start"])
    out = out[["date", "filling_twh"]].rename(columns={"filling_twh": "hydro_twh"})
    # Resample weekly → daily via forward fill
    out = out.set_index("date").resample("D").ffill().reset_index()
    return out


def _load_wind(years: int) -> pd.DataFrame:
    from data.wind import fetch_wind_daily
    result = fetch_wind_daily(days=int(years * 365))
    df = result.get("daily", pd.DataFrame())
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "de_wind_gwh"]].dropna(subset=["de_wind_gwh"])


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

    # Storage deviation from seasonal mean (rolling 5-year DOY average)
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


# ── Main assembly function ───────────────────────────────────────────────────

def assemble_features(years: int = 3, use_cache: bool = True) -> pd.DataFrame:
    """
    Build the full daily feature matrix from all available sources.

    Parameters
    ----------
    years     : How many years of history to request from each source.
    use_cache : Merge with persisted cache (extends history across restarts).

    Returns
    -------
    pd.DataFrame with date column + all computed features.
    Rows with NaN in core features (no2, nl, ttf) are dropped.
    """
    days = int(years * 365)

    spot    = _load_spot(days)
    ttf     = _load_ttf(years)
    storage = _load_storage(years)
    hydro   = _load_hydro(years)
    wind    = _load_wind(years)

    if spot.empty or ttf.empty:
        # Try loading from disk cache
        return _load_cache()

    # Pivot spot prices: wide format
    for col in ("NO2", "NL"):
        if col not in spot.columns:
            return _load_cache()

    base = spot[["date", "NO2", "NL"]].copy()
    base.columns = ["date", "no2", "nl"]
    base["date"] = pd.to_datetime(base["date"])

    # Merge TTF
    ttf["date"] = pd.to_datetime(ttf["date"])
    base = base.merge(ttf, on="date", how="left")

    # Merge storage
    if not storage.empty:
        storage["date"] = pd.to_datetime(storage["date"])
        base = base.merge(storage, on="date", how="left")

    # Merge hydro
    if not hydro.empty:
        hydro["date"] = pd.to_datetime(hydro["date"])
        base = base.merge(hydro, on="date", how="left")

    # Merge wind
    if not wind.empty:
        wind["date"] = pd.to_datetime(wind["date"])
        base = base.merge(wind, on="date", how="left")

    base = base.sort_values("date").reset_index(drop=True)

    # Compute derived features
    base = _compute_features(base)

    # Forward-fill short gaps in optional features (max 5 days)
    for col in ["storage_fill", "storage_dev", "hydro_twh", "hydro_pct",
                "de_wind_gwh", "wind_de_z"]:
        if col in base.columns:
            base[col] = base[col].ffill(limit=5)

    # Drop rows missing core features
    base = base.dropna(subset=["no2", "nl", "ttf"]).reset_index(drop=True)

    # Merge with disk cache (fills in historical rows we may have lost)
    if use_cache:
        cached = _load_cache()
        if not cached.empty:
            combined = pd.concat([cached, base], ignore_index=True)
            combined = (
                combined
                .sort_values("date")
                .drop_duplicates(subset=["date"], keep="last")
                .reset_index(drop=True)
            )
            base = combined

    # Persist to disk
    _save_cache(base)

    return base


# ── Cache persistence ────────────────────────────────────────────────────────

def _save_cache(df: pd.DataFrame) -> None:
    try:
        df.to_csv(_CACHE_FILE, index=False)
        meta = {
            "updated_at": datetime.utcnow().isoformat(),
            "n_rows": len(df),
            "date_min": str(df["date"].min().date()),
            "date_max": str(df["date"].max().date()),
            "columns": list(df.columns),
        }
        with open(_META_FILE, "w") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass


def _load_cache() -> pd.DataFrame:
    try:
        if not os.path.exists(_CACHE_FILE):
            return pd.DataFrame()
        df = pd.read_csv(_CACHE_FILE, parse_dates=["date"])
        return df
    except Exception:
        return pd.DataFrame()


def get_feature_meta() -> dict:
    """Return metadata about the persisted feature cache (for UI display)."""
    try:
        if os.path.exists(_META_FILE):
            with open(_META_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    df = _load_cache()
    if df.empty:
        return {}
    return {
        "n_rows": len(df),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "columns": list(df.columns),
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
