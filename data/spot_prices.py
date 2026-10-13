"""
Day-ahead daily average prices for the Nordic and Continental bidding zones.

Two sources publish the same SDAC clearing prices:

* ENTSO-E Transparency Platform (A44), with ENTSOE_API_KEY: one request per zone and year,
  every bidding zone including DE-LU. Daily averages are time-weighted over each delivery
  day in Central European time (data/energy_units.daily_mean), so 15-minute and hourly
  periods weigh alike.
* Nord Pool Data Portal, keyless: one request per delivery day for all areas, recent days
  only, plus the Nordic system price (SYS), which ENTSO-E does not publish. DE-LU is not
  on the portal (EPEX licensing).

ENTSO-E is used first when a key is set; Nord Pool fills the zones and days ENTSO-E does
not cover (SYS, the latest delivery day before ENTSO-E publishes it) and replaces it
without a key. Nord Pool rate-limits shared cloud addresses, so its requests are retried
with a back-off and run four at a time.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pandas as pd
import requests
import streamlit as st

from config.settings import ENTSOE_TIMEOUT_S
from data.energy_units import daily_mean

NORDPOOL_URL = "https://dataportal-api.nordpoolgroup.com/api/DayAheadPrices"
NORDPOOL_AREAS = ["NO1", "NO2", "SE3", "DE-LU", "NL", "FI"]
HEADERS = {"Referer": "https://www.nordpoolgroup.com/", "User-Agent": "Mozilla/5.0"}
DELIVERY_TZ = "Europe/Oslo"          # SDAC delivery days run on Central European time
_NORDPOOL_WORKERS = 4
_NORDPOOL_RETRY_WAITS = (1.5, 4.0)   # seconds before the second and third attempt
NORDPOOL_MAX_DAYS = 90               # the public portal serves roughly the last two months
_ENTSOE_CHUNK_DAYS = 90              # a year of 15-minute prices in one request can time out

ENTSOE_ZONES = {
    "NO1":   "10YNO-1--------2",
    "NO2":   "10YNO-2--------T",
    "NO3":   "10YNO-3--------J",
    "NO4":   "10YNO-4--------9",
    "NO5":   "10Y1001A1001A48H",
    "SE3":   "10Y1001A1001A46L",
    "DE-LU": "10Y1001A1001A82H",
    "NL":    "10YNL----------L",
    "FI":    "10YFI-1--------U",
}

# Display labels for the chart
AREA_LABELS = {
    "NO1":   "NO1 (Oslo)",
    "NO2":   "NO2 (Kristiansand)",
    "SE3":   "SE3 (Stockholm)",
    "DE-LU": "DE/LU (Germany)",
    "NL":    "NL (Netherlands)",
    "FI":    "FI (Finland)",
}

_COLUMNS = ["date", "zone", "price_eur_mwh"]


def _fetch_day(date_str: str, areas: tuple[str, ...] = tuple(NORDPOOL_AREAS)) -> list[dict]:
    """Nord Pool daily averages for one delivery date: [{date, zone, price_eur_mwh}]."""
    for attempt in range(len(_NORDPOOL_RETRY_WAITS) + 1):
        try:
            resp = requests.get(
                NORDPOOL_URL,
                params={"currency": "EUR", "market": "DayAhead",
                        "deliveryArea": ",".join(areas), "date": date_str},
                headers=HEADERS, timeout=10,
            )
            if resp.status_code == 204:
                return []
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < len(_NORDPOOL_RETRY_WAITS):
                time.sleep(_NORDPOOL_RETRY_WAITS[attempt])
                continue
            if resp.status_code >= 400:
                return []          # 401 for dates outside the public window, 4xx otherwise
            data = resp.json()
        except requests.Timeout:
            return []
        except Exception:
            if attempt < len(_NORDPOOL_RETRY_WAITS):
                time.sleep(_NORDPOOL_RETRY_WAITS[attempt])
                continue
            return []
        return [
            {"date": date_str, "zone": e["areaCode"], "price_eur_mwh": e["price"]}
            for e in data.get("areaAverages", [])
            if e.get("price") is not None and e.get("areaCode") in areas
        ]
    return []


def _nordpool(dates: list[str], areas: tuple[str, ...]) -> pd.DataFrame:
    if not dates or not areas:
        return pd.DataFrame(columns=_COLUMNS)
    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=_NORDPOOL_WORKERS) as ex:
        for rows in ex.map(lambda d: _fetch_day(d, areas), dates):
            records.extend(rows)
    if not records:
        return pd.DataFrame(columns=_COLUMNS)
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["price_eur_mwh"] = pd.to_numeric(df["price_eur_mwh"], errors="coerce")
    return df.dropna(subset=["price_eur_mwh"])


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_entsoe_daily(zones: tuple[str, ...], days: int) -> pd.DataFrame:
    """ENTSO-E A44 daily averages for ``zones`` over the last ``days`` delivery days."""
    key = os.getenv("ENTSOE_API_KEY", "")
    zones = tuple(z for z in zones if z in ENTSOE_ZONES)
    if not key or not zones:
        return pd.DataFrame(columns=_COLUMNS)
    try:
        from entsoe import EntsoePandasClient
    except ImportError:
        return pd.DataFrame(columns=_COLUMNS)
    client = EntsoePandasClient(api_key=key, timeout=ENTSOE_TIMEOUT_S)
    end = pd.Timestamp.now(tz=DELIVERY_TZ).normalize() + pd.DateOffset(days=2)
    start = (end - pd.Timedelta(days=days + 2)).normalize()
    frames = []
    for zone in zones:
        chunks = []
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(chunk_start + pd.Timedelta(days=_ENTSOE_CHUNK_DAYS), end)
            try:
                s = client.query_day_ahead_prices(ENTSOE_ZONES[zone], start=chunk_start, end=chunk_end)
                if s is not None and not s.empty:
                    chunks.append(s)
            except Exception:
                pass
            chunk_start = chunk_end
        if not chunks:
            continue
        prices = pd.concat(chunks)
        prices = pd.to_numeric(prices[~prices.index.duplicated(keep="last")], errors="coerce").dropna()
        daily = daily_mean(prices, tz=DELIVERY_TZ)
        if daily.empty:
            continue
        frames.append(pd.DataFrame({"date": daily.index.date, "zone": zone,
                                    "price_eur_mwh": daily.to_numpy()}))
    if not frames:
        return pd.DataFrame(columns=_COLUMNS)
    return pd.concat(frames, ignore_index=True)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_dayahead(zones: tuple[str, ...], days: int = 30) -> pd.DataFrame:
    """Daily average day-ahead prices (EUR/MWh) for ``zones``, last ``days`` days through tomorrow.

    Long format: date (datetime.date), zone, price_eur_mwh, source.
    """
    today = pd.Timestamp.now(tz=DELIVERY_TZ).date()
    first = today - timedelta(days=days)
    wanted_dates = [first + timedelta(days=i) for i in range(days + 2)]   # through tomorrow

    ent = fetch_entsoe_daily(tuple(zones), days)
    ent = ent[(ent["date"] >= first)] if not ent.empty else ent
    if not ent.empty:
        ent = ent.assign(source="ENTSO-E")

    # Nord Pool for every (zone, date) ENTSO-E did not supply
    have = set(zip(ent["date"], ent["zone"])) if not ent.empty else set()
    np_zones = tuple(z for z in zones if z != "DE-LU")
    missing_dates = sorted({d for d in wanted_dates for z in np_zones if (d, z) not in have})
    # Days after today are only published after the auction (about 12:45 CET), and the
    # public portal does not serve dates older than about two months
    np_first = today - timedelta(days=NORDPOOL_MAX_DAYS)
    missing_dates = [d for d in missing_dates if np_first <= d <= today + timedelta(days=1)]
    nordpool = _nordpool([d.strftime("%Y-%m-%d") for d in missing_dates], np_zones)
    if not nordpool.empty:
        nordpool = nordpool[[(d, z) not in have for d, z in zip(nordpool["date"], nordpool["zone"])]]
        nordpool = nordpool.assign(source="Nord Pool")

    parts = [p for p in (ent, nordpool) if not p.empty]
    if not parts:
        return pd.DataFrame(columns=_COLUMNS + ["source"])
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset=["date", "zone"], keep="first")
    return df.sort_values(["date", "zone"]).reset_index(drop=True)


def fetch_spot_prices(days: int = 30) -> pd.DataFrame:
    """Daily average day-ahead prices for the platform's six reference zones."""
    return fetch_dayahead(tuple(NORDPOOL_AREAS), days)


def get_spot_price_data() -> dict:
    df = fetch_spot_prices()
    return {"prices": df, "fetched_at": datetime.utcnow()}
