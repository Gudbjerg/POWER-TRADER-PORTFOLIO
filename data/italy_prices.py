"""
IT-North day-ahead prices and the Finnish comparison series for the FI minus
IT-North spread (Layer 1, Prices tab; roadmap item 3A).

IT-North
    Primary:  energy-charts.info price API (Fraunhofer ISE), bidding zone
              "IT-North". No key required. Data licence: CC BY 4.0,
              Bundesnetzagentur | SMARD.de (attribution required).
              Requests are split into chunks of at most
              ENERGY_CHARTS_MAX_CHUNK_DAYS days; HTTP 429 and 5xx responses
              are retried with back-off.
    Fallback: ENTSO-E Transparency Platform, A44 day-ahead prices for
              IT_NORD (EIC 10Y1001A1001A73I) via entsoe-py, used for any
              chunk energy-charts.info fails to deliver when ENTSOE_API_KEY
              is set.

FI
    Primary:  Nord Pool Data Portal daily area averages, read through the
              existing cached fetch in data/spot_prices.py. The public
              endpoint only serves roughly the latest two months (older
              delivery dates return HTTP 401).
    Extension: ENTSO-E A44 for FI (EIC 10YFI-1--------U) when
              ENTSOE_API_KEY is set, which supplies the full one-year history.
              energy-charts.info also publishes FI prices, but under a
              restrictive licence (private and internal use only), so that
              series is deliberately not used.

All sub-daily prices (hourly before delivery day 2025-10-01, 15-minute
afterwards) are aggregated to arithmetic daily means per CET/CEST delivery
day (Europe/Rome, the SDAC delivery-day convention also used by Nord Pool).
"""
from __future__ import annotations

import os
import time
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

from config.settings import (
    ENERGY_CHARTS_MAX_CHUNK_DAYS,
    FI_IT_HISTORY_DAYS,
    FI_IT_MIN_DAY_COVERAGE,
    IT_NORTH_EIC,
)

ENERGY_CHARTS_PRICE_URL = "https://api.energy-charts.info/price"
IT_NORTH_BZN = "IT-North"
FI_EIC = "10YFI-1--------U"
DELIVERY_TZ = "Europe/Rome"   # CET/CEST, identical to the SDAC delivery-day clock

IT_NORTH_LICENCE_DEFAULT = (
    "CC BY 4.0 (creativecommons.org/licenses/by/4.0) from Bundesnetzagentur | SMARD.de"
)
ENTSOE_ATTRIBUTION = "ENTSO-E Transparency Platform (A44 day-ahead prices)"

_REQUEST_SPACING_S = 0.5   # pause between consecutive energy-charts.info requests
_MAX_ATTEMPTS = 3          # attempts per chunk (first try plus two retries)
_MAX_BACKOFF_S = 20.0      # cap on any single back-off sleep
_REQUEST_TIMEOUT_S = 15    # per-request timeout
_FETCH_BUDGET_S = 60.0     # wall-clock budget for one chunked history fetch
_MAX_DEAD_CHUNKS = 2       # consecutive unanswered chunks before the source is treated as down


# ── Pure helpers (unit-tested in tests/test_italy_prices.py) ─────────────────

def date_chunks(start: date, end: date, max_days: int) -> list[tuple[date, date]]:
    """
    Split the inclusive date range [start, end] into consecutive, non-overlapping
    inclusive chunks of at most `max_days` days each.
    """
    if max_days < 1:
        raise ValueError("max_days must be at least 1")
    if end < start:
        return []
    chunks: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=max_days - 1), end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def parse_energy_charts_price(payload: dict) -> pd.Series:
    """
    Convert an energy-charts.info /price JSON payload into a float Series in
    EUR/MWh indexed by UTC timestamps (interval start). Missing prices are
    dropped, duplicate timestamps keep the last value, and payloads in any unit
    other than EUR/MWh are rejected (empty Series).
    """
    empty = pd.Series(dtype=float, name="price")
    if not isinstance(payload, dict):
        return empty
    stamps = payload.get("unix_seconds")
    values = payload.get("price")
    if not stamps or not values or len(stamps) != len(values):
        return empty
    unit = str(payload.get("unit", "EUR / MWh")).replace(" ", "").upper()
    if unit != "EUR/MWH":
        return empty
    index = pd.DatetimeIndex(
        pd.to_datetime(np.asarray(stamps, dtype="int64"), unit="s", utc=True)
    ).as_unit("ns")
    prices = pd.to_numeric(pd.Series(values, dtype="object"), errors="coerce").to_numpy(dtype=float)
    series = pd.Series(prices, index=index, name="price")
    series = series[~series.index.duplicated(keep="last")].sort_index()
    return series.dropna()


def daily_average_by_delivery_day(
    prices: pd.Series,
    tz: str = DELIVERY_TZ,
    min_coverage: float = FI_IT_MIN_DAY_COVERAGE,
) -> pd.DataFrame:
    """
    Aggregate a sub-daily price series to arithmetic daily means per local
    delivery day.

    The index must be timezone-aware (interval start times). For every local
    day the native resolution is inferred as the median spacing between its
    timestamps, the expected number of intervals is the day length (23, 24 or
    25 hours around DST changes) divided by that resolution, and coverage is
    the observed share of expected intervals. Days below `min_coverage`, and
    days with a single observation (resolution not identifiable), are dropped.

    Returns a DataFrame with columns: date (tz-naive midnight), price,
    n_points, coverage.
    """
    columns = ["date", "price", "n_points", "coverage"]
    if prices is None or len(prices) == 0:
        return pd.DataFrame(columns=columns)
    if not isinstance(prices.index, pd.DatetimeIndex) or prices.index.tz is None:
        raise ValueError("prices must have a timezone-aware DatetimeIndex")

    series = prices.dropna().sort_index()
    series = series[~series.index.duplicated(keep="last")]
    if series.empty:
        return pd.DataFrame(columns=columns)

    local = series.tz_convert(tz)
    frame = pd.DataFrame({
        "price": local.to_numpy(dtype=float),
        "ts": local.index.as_unit("ns").asi8,   # UTC nanoseconds: spacing is unaffected by DST
        "day": local.index.tz_localize(None).normalize(),
    })

    rows = []
    for day, grp in frame.groupby("day", sort=True):
        n_points = len(grp)
        if n_points < 2:
            continue
        spacing_min = float(np.median(np.diff(grp["ts"].to_numpy()))) / 60e9
        if not np.isfinite(spacing_min) or spacing_min <= 0:
            continue
        day_start = pd.Timestamp(day).tz_localize(tz)
        day_end = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(tz)
        day_minutes = (day_end - day_start).total_seconds() / 60.0
        expected = day_minutes / spacing_min
        coverage = min(n_points / expected, 1.0) if expected > 0 else 0.0
        if coverage < min_coverage:
            continue
        rows.append({
            "date": pd.Timestamp(day),
            "price": float(grp["price"].mean()),
            "n_points": int(n_points),
            "coverage": round(float(coverage), 4),
        })
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values("date").reset_index(drop=True)


# ── Network helpers ──────────────────────────────────────────────────────────

def _get_json_with_retry(
    url: str,
    params: dict,
    timeout: int = _REQUEST_TIMEOUT_S,
    deadline: float | None = None,
) -> tuple[dict | None, str | None]:
    """
    GET a JSON document, retrying HTTP 429/5xx and transport errors with
    back-off. `deadline` (time.monotonic() value) stops further attempts and
    sleeps once reached, so an unresponsive source cannot stall the page.
    """
    last_error = None
    for attempt in range(_MAX_ATTEMPTS):
        if deadline is not None and time.monotonic() >= deadline:
            return None, last_error or "time budget exhausted"
        try:
            resp = requests.get(url, params=params, timeout=timeout)
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}"
            wait = min(2.0 * (2 ** attempt), _MAX_BACKOFF_S)
            if attempt < _MAX_ATTEMPTS - 1 and (deadline is None or time.monotonic() + wait < deadline):
                time.sleep(wait)
            continue
        if resp.status_code == 429 or resp.status_code >= 500:
            last_error = f"HTTP {resp.status_code}"
            retry_after = resp.headers.get("Retry-After", "")
            try:
                wait = float(retry_after)
            except ValueError:
                wait = 2.0 * (2 ** attempt)
            wait = min(max(wait, 1.0), _MAX_BACKOFF_S)
            if attempt < _MAX_ATTEMPTS - 1 and (deadline is None or time.monotonic() + wait < deadline):
                time.sleep(wait)
            continue
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}"
        try:
            return resp.json(), None
        except ValueError:
            return None, "invalid JSON response"
    return None, last_error or "request failed"


def _fetch_energy_charts(bzn: str, start: date, end: date) -> tuple[pd.Series, str | None, list]:
    """
    Fetch a bidding-zone price series from energy-charts.info in chunks.

    Only payloads published under a CC BY licence are accepted: energy-charts
    licenses some zones (for example FI) for private and internal use only,
    which is incompatible with a public dashboard.

    The whole fetch runs within _FETCH_BUDGET_S seconds, and after
    _MAX_DEAD_CHUNKS consecutive unanswered chunks the remaining chunks are
    skipped, so an outage degrades the panel instead of stalling the page.

    Returns (series, licence_text, failed) where `failed` lists
    (chunk_start, chunk_end, reason) for every chunk that delivered no data.
    """
    parts: list[pd.Series] = []
    licence = None
    failed: list[tuple[date, date, str]] = []
    deadline = time.monotonic() + _FETCH_BUDGET_S
    dead_chunks = 0
    for i, (c_start, c_end) in enumerate(date_chunks(start, end, ENERGY_CHARTS_MAX_CHUNK_DAYS)):
        if dead_chunks >= _MAX_DEAD_CHUNKS or time.monotonic() >= deadline:
            failed.append((c_start, c_end, "skipped: source unresponsive"))
            continue
        if i > 0:
            time.sleep(_REQUEST_SPACING_S)
        payload, err = _get_json_with_retry(
            ENERGY_CHARTS_PRICE_URL,
            {"bzn": bzn, "start": c_start.isoformat(), "end": c_end.isoformat()},
            deadline=deadline,
        )
        if payload is None:
            failed.append((c_start, c_end, err or "no response"))
            dead_chunks += 1
            continue
        dead_chunks = 0
        payload_licence = str(payload.get("license_info", ""))
        if "CC BY" not in payload_licence.upper():
            failed.append((c_start, c_end, "licence is not CC BY"))
            continue
        series = parse_energy_charts_price(payload)
        if series.empty:
            failed.append((c_start, c_end, "no data"))
            continue
        licence = licence or payload_licence
        parts.append(series)
    if not parts:
        return pd.Series(dtype=float, name="price"), licence, failed
    combined = pd.concat(parts).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined, licence, failed


def _fetch_entsoe(eic: str, start: date, end: date) -> pd.Series:
    """
    ENTSO-E A44 day-ahead prices for one bidding zone over [start, end]
    (inclusive delivery days), returned with a UTC index. Empty when no
    ENTSOE_API_KEY is configured, entsoe-py is missing, or the query fails.
    """
    key = os.getenv("ENTSOE_API_KEY", "").strip()
    if not key:
        return pd.Series(dtype=float, name="price")
    try:
        from entsoe import EntsoePandasClient
    except ImportError:
        return pd.Series(dtype=float, name="price")

    # Bounded timeout and retries: entsoe-py waits indefinitely by default.
    client = EntsoePandasClient(api_key=key, timeout=30, retry_count=2, retry_delay=5)
    parts: list[pd.Series] = []
    for c_start, c_end in date_chunks(start, end, 90):   # a year of 15-minute prices can time out
        try:
            series = client.query_day_ahead_prices(
                eic,
                start=pd.Timestamp(c_start).tz_localize(DELIVERY_TZ),
                end=(pd.Timestamp(c_end) + pd.Timedelta(days=1)).tz_localize(DELIVERY_TZ),
            )
        except Exception:
            continue
        if series is None or len(series) == 0:
            continue
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        parts.append(pd.to_numeric(series, errors="coerce").tz_convert("UTC"))
    if not parts:
        return pd.Series(dtype=float, name="price")
    combined = pd.concat(parts).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")].dropna()
    combined.name = "price"
    return combined


def _delivery_today() -> date:
    return pd.Timestamp.now(tz=DELIVERY_TZ).date()


# ── Cached loaders ───────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_it_north_daily(days: int = FI_IT_HISTORY_DAYS) -> dict:
    """
    Daily IT-North day-ahead prices (EUR/MWh) for the last `days` delivery days
    plus the next delivery day when already auctioned.

    Returns dict with keys:
        daily      : DataFrame[date, it_north]
        source     : human-readable source label (None when unavailable)
        licence    : licence / attribution text
        error      : diagnostic string for any chunk that could not be loaded
        fetched_at : datetime (UTC)
    """
    end = _delivery_today() + timedelta(days=1)
    start = end - timedelta(days=int(days))

    series, licence, failed = _fetch_energy_charts(IT_NORTH_BZN, start, end)
    sources = []
    licence_parts = []
    if not series.empty:
        sources.append("energy-charts.info (Fraunhofer ISE), data: Bundesnetzagentur | SMARD.de")
        licence_parts.append(licence or IT_NORTH_LICENCE_DEFAULT)

    # ENTSO-E fallback, chunk by chunk, for whatever energy-charts.info did not deliver.
    unresolved: list[str] = []
    filler: list[pd.Series] = []
    for c_start, c_end, reason in failed:
        fill = _fetch_entsoe(IT_NORTH_EIC, c_start, c_end)
        if fill.empty:
            unresolved.append(f"{c_start} to {c_end}: {reason}")
        else:
            filler.append(fill)
    if filler:
        fill_series = pd.concat(filler)
        series = pd.concat([fill_series, series]) if not series.empty else fill_series
        series = series.sort_index()
        series = series[~series.index.duplicated(keep="last")]   # energy-charts wins on overlap
        sources.append("ENTSO-E A44 (IT-North)")
        licence_parts.append(ENTSOE_ATTRIBUTION)

    daily = daily_average_by_delivery_day(series)
    daily = daily.rename(columns={"price": "it_north"})[["date", "it_north"]]
    return {
        "daily": daily,
        "source": " + ".join(sources) if sources else None,
        "licence": " | ".join(licence_parts) if licence_parts else None,
        "error": "; ".join(unresolved) if unresolved else None,
        "fetched_at": datetime.now(timezone.utc).replace(tzinfo=None),
    }


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fi_daily(days: int = FI_IT_HISTORY_DAYS) -> dict:
    """
    Daily FI day-ahead prices (EUR/MWh) for the last `days` delivery days.

    Nord Pool daily area averages are always used where available (keyless).
    When ENTSOE_API_KEY is set, ENTSO-E A44 supplies the delivery days the
    public Nord Pool window does not cover. On overlapping days the Nord Pool
    value is kept (both publish the same SDAC clearing prices).

    Returns dict with keys: daily (DataFrame[date, fi]), source, nordpool_days,
    entsoe_days, fetched_at.
    """
    from data.spot_prices import fetch_spot_prices

    end = _delivery_today()
    start = end - timedelta(days=int(days))

    # Same arguments as the Layer 1 365-day history call, so the cache entry is shared.
    raw = fetch_spot_prices(days=365)
    nordpool = pd.DataFrame(columns=["date", "fi"])
    if raw is not None and not raw.empty and "zone" in raw.columns:
        fi_rows = raw[raw["zone"] == "FI"][["date", "price_eur_mwh"]].copy()
        if not fi_rows.empty:
            fi_rows["date"] = pd.to_datetime(fi_rows["date"]).dt.normalize()
            nordpool = (
                fi_rows.rename(columns={"price_eur_mwh": "fi"})
                .groupby("date", as_index=False)["fi"].mean()
            )

    entsoe = pd.DataFrame(columns=["date", "fi"])
    if os.getenv("ENTSOE_API_KEY", "").strip():
        series = _fetch_entsoe(FI_EIC, start, end)
        if not series.empty:
            entsoe = (
                daily_average_by_delivery_day(series)
                .rename(columns={"price": "fi"})[["date", "fi"]]
            )

    frames = [f for f in (entsoe, nordpool) if not f.empty]
    if frames:
        combined = (
            pd.concat(frames, ignore_index=True)
            .sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")   # Nord Pool wins on overlap
        )
        combined = combined[
            (combined["date"] >= pd.Timestamp(start)) & (combined["date"] <= pd.Timestamp(end))
        ].reset_index(drop=True)
    else:
        combined = pd.DataFrame(columns=["date", "fi"])

    sources = []
    if not nordpool.empty:
        sources.append("Nord Pool Data Portal")
    if not entsoe.empty:
        sources.append("ENTSO-E A44 (FI)")
    nordpool_days = int(combined["date"].isin(nordpool["date"]).sum()) if not combined.empty else 0
    return {
        "daily": combined,
        "source": " + ".join(sources) if sources else None,
        "nordpool_days": nordpool_days,
        "entsoe_days": int(len(combined) - nordpool_days),
        "fetched_at": datetime.now(timezone.utc).replace(tzinfo=None),
    }
