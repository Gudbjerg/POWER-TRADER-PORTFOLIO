"""
Landing-page market strip: delayed quotes for gas, LNG, oil, carbon, FX and
Nordic/Continental day-ahead power.

Futures, carbon and FX come from Yahoo Finance daily bars (the latest bar can be
intraday while a market is open). Power is the day-ahead daily average from the Nord
Pool Data Portal via the existing ``data.spot_prices`` day fetcher. Every instrument
fails independently: an unavailable quote is returned with ``value=None`` and is
rendered as a middle dot.

DE-LU day-ahead prices are not published on the Nord Pool Data Portal (EPEX
licensing), so the Continental slot falls back to NL, the continental reference used
throughout the platform, and is labelled accordingly.
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

from data.spot_prices import _fetch_day as _fetch_nordpool_day

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:  # pragma: no cover - yfinance is a hard dependency in requirements
    YFINANCE_AVAILABLE = False

# (symbol, descriptor, ticker, unit, decimals, sanity range)
YF_INSTRUMENTS: list[tuple[str, str, str, str, int, tuple[float, float]]] = [
    ("TTF",     "Front month",     "TTF=F",    "EUR/MWh",   2, (5.0, 500.0)),
    ("JKM",     "Front month",     "JKM=F",    "USD/MMBtu", 2, (1.0, 150.0)),
    ("Brent",   "Front month",     "BZ=F",     "USD/bbl",   2, (10.0, 300.0)),
    ("EUA",     "Carbon, CO2.L",   "CO2.L",    "EUR/t",     2, (5.0, 250.0)),
    ("EUR/USD", "Spot FX",         "EURUSD=X", "USD per EUR", 4, (0.5, 2.0)),
]

# Power slots: (preferred zone, fallback zone or None, descriptor when preferred,
# descriptor when the fallback is used)
NORDPOOL_SLOTS: list[tuple[str, str | None, str, str]] = [
    ("NO2",   None, "Day-ahead avg", "Day-ahead avg"),
    ("DE-LU", "NL", "Day-ahead avg", "Day-ahead, DE-LU n/a"),
]
_NORDPOOL_DAYS = 8          # history for the sparkline and the day-over-day change
_YF_PERIOD = "1mo"          # daily bars for the sparkline and the day-over-day change


def _finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _empty_item(symbol: str, sub: str, unit: str, decimals: int, title: str) -> dict:
    return {
        "symbol": symbol, "sub": sub, "unit": unit, "decimals": decimals,
        "value": None, "change": None, "change_mode": "pct", "spark": [],
        "as_of": None, "title": title,
    }


def _yf_closes() -> pd.DataFrame:
    """Daily closes for all Yahoo instruments (columns = tickers). Empty on failure."""
    if not YFINANCE_AVAILABLE:
        return pd.DataFrame()
    tickers = [t for _, _, t, _, _, _ in YF_INSTRUMENTS]
    try:
        raw = yf.download(
            tickers, period=_YF_PERIOD, interval="1d", progress=False,
            auto_adjust=False, threads=True, group_by="column",
        )
    except Exception:
        return pd.DataFrame()
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            return pd.DataFrame()
        closes = raw["Close"].copy()
    else:
        if "Close" not in raw.columns:
            return pd.DataFrame()
        closes = raw[["Close"]].rename(columns={"Close": tickers[0]})
    closes.index = pd.to_datetime(closes.index)
    if getattr(closes.index, "tz", None) is not None:
        closes.index = closes.index.tz_localize(None)
    return closes.sort_index()


def _yf_items(closes: pd.DataFrame) -> list[dict]:
    items = []
    for symbol, sub, ticker, unit, dec, (lo, hi) in YF_INSTRUMENTS:
        item = _empty_item(symbol, sub, unit, dec, f"{symbol} ({ticker}) via Yahoo Finance")
        if not closes.empty and ticker in closes.columns:
            s = pd.to_numeric(closes[ticker], errors="coerce").dropna()
            s = s[(s > lo) & (s < hi)]        # discard obviously corrupt prints
            if not s.empty:
                last = float(s.iloc[-1])
                item["value"] = last
                item["as_of"] = s.index[-1].date().isoformat()
                if len(s) >= 2 and float(s.iloc[-2]) > 0:
                    item["change"] = (last / float(s.iloc[-2]) - 1.0) * 100.0
                item["spark"] = [float(v) for v in s.iloc[-22:]]
                item["title"] = f"{symbol} ({ticker}) via Yahoo Finance, last daily bar {item['as_of']}"
        items.append(item)
    return items


def _nordpool_history(days: int = _NORDPOOL_DAYS) -> pd.DataFrame:
    """Daily average day-ahead prices for the last ``days`` delivery days (long format)."""
    today = datetime.now(timezone.utc).date()
    dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]
    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for rows in ex.map(_fetch_nordpool_day, dates):
            records.extend(rows)
    if not records:
        return pd.DataFrame(columns=["date", "zone", "price_eur_mwh"])
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df["price_eur_mwh"] = pd.to_numeric(df["price_eur_mwh"], errors="coerce")
    return df.dropna(subset=["price_eur_mwh"]).sort_values("date")


def _nordpool_items(df: pd.DataFrame) -> list[dict]:
    items = []
    for zone, fallback, sub, fallback_sub in NORDPOOL_SLOTS:
        chosen, label_sub = zone, sub
        series = df[df["zone"] == zone].set_index("date")["price_eur_mwh"] if not df.empty else pd.Series(dtype=float)
        if series.empty and fallback:
            chosen, label_sub = fallback, fallback_sub
            series = df[df["zone"] == fallback].set_index("date")["price_eur_mwh"] if not df.empty else pd.Series(dtype=float)
        title = f"{chosen} day-ahead daily average via Nord Pool"
        if chosen != zone:
            title += f" ({zone} is not published on the Nord Pool Data Portal; {chosen} shown as the continental reference)"
        item = _empty_item(chosen, label_sub, "EUR/MWh", 2, title)
        series = series.sort_index()
        if not series.empty:
            last = float(series.iloc[-1])
            item["value"] = last
            item["as_of"] = series.index[-1].date().isoformat()
            item["change_mode"] = "abs"   # power can trade near zero or negative: show EUR/MWh change
            if len(series) >= 2:
                item["change"] = last - float(series.iloc[-2])
            item["spark"] = [float(v) for v in series.iloc[-_NORDPOOL_DAYS:]]
            item["title"] = f"{title}, delivery day {item['as_of']}"
        items.append(item)
    return items


@st.cache_data(ttl=900, show_spinner=False)
def fetch_market_strip() -> dict:
    """Quotes for the landing-page strip, cached for 15 minutes.

    Returns ``{"items": [...], "fetched_at": datetime}``. Each item carries ``symbol``,
    ``sub``, ``unit``, ``decimals``, ``value`` (float or None), ``change`` (float or
    None), ``change_mode`` ('pct' or 'abs'), ``spark`` (list of floats), ``as_of`` and
    ``title``.
    """
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_yf = ex.submit(_yf_closes)
        fut_np = ex.submit(_nordpool_history)
        try:
            closes = fut_yf.result()
        except Exception:
            closes = pd.DataFrame()
        try:
            power = fut_np.result()
        except Exception:
            power = pd.DataFrame(columns=["date", "zone", "price_eur_mwh"])
    items = _yf_items(closes) + _nordpool_items(power)
    return {"items": items, "fetched_at": datetime.now(timezone.utc)}
