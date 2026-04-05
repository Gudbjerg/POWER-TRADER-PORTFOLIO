"""
Cross-commodity price fetcher: TTF gas, wheat, corn futures via yfinance.
Used for the gas-to-fertiliser-to-agriculture spillover chain.
"""
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

TICKERS = {
    "TTF":   "TTF=F",    # ICE TTF front-month gas futures (EUR/MWh)
    "Wheat": "ZW=F",     # CBOT Wheat futures (USc/bushel)
    "Corn":  "ZC=F",     # CBOT Corn futures (USc/bushel)
}

COLORS = {
    "TTF":   "#e07b39",
    "Wheat": "#d4ac3a",
    "Corn":  "#3fb950",
}


@st.cache_data(ttl=3600)
def fetch_commodity_prices(days: int = 730) -> pd.DataFrame:
    """
    Fetch daily closing prices for TTF, wheat, and corn futures.
    Returns long-format DataFrame with columns: date, commodity, price, price_norm.
    price_norm is indexed to 100 at the earliest common date.
    """
    if not YFINANCE_AVAILABLE:
        return pd.DataFrame()

    end   = datetime.utcnow().date()
    start = (end - timedelta(days=days)).strftime("%Y-%m-%d")
    end_s = end.strftime("%Y-%m-%d")

    frames = {}
    for name, ticker in TICKERS.items():
        try:
            df = yf.Ticker(ticker).history(start=start, end=end_s)
            if df.empty:
                continue
            s = df["Close"].copy()
            s.index = pd.to_datetime(s.index).tz_localize(None)
            frames[name] = s
        except Exception:
            pass

    if len(frames) < 2:
        return pd.DataFrame()

    wide = pd.DataFrame(frames).dropna()
    if wide.empty:
        return pd.DataFrame()

    # Normalise to 100 at first common date
    base = wide.iloc[0]
    normed = wide.div(base) * 100

    records = []
    for col in wide.columns:
        for dt, raw, norm in zip(wide.index, wide[col], normed[col]):
            records.append({
                "date":       dt.date(),
                "commodity":  col,
                "price":      float(raw),
                "price_norm": float(norm),
            })

    df_out = pd.DataFrame(records).sort_values(["commodity", "date"]).reset_index(drop=True)
    return df_out


def get_commodity_data() -> dict:
    df = fetch_commodity_prices()
    return {"prices": df, "fetched_at": datetime.utcnow()}
