"""
TTF gas price fetcher via yfinance (TTF futures = 'TTF=F' on Yahoo Finance).
"""
import time
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


TTF_TICKER = "TTF=F"


@st.cache_data(ttl=7200)   # 2-hour cache — reduces Yahoo Finance request frequency
def fetch_ttf_prices(days: int = 120) -> pd.DataFrame:
    """Fetch TTF front-month futures price history. Retries up to 3 times on rate limit."""
    if not YFINANCE_AVAILABLE:
        return pd.DataFrame()

    end   = datetime.utcnow().date()
    start = end - timedelta(days=days)

    last_exc = None
    for attempt in range(3):
        try:
            ticker = yf.Ticker(TTF_TICKER)
            df = ticker.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
            if df.empty:
                return pd.DataFrame()
            df = df[["Close"]].rename(columns={"Close": "price"})
            df.index.name = "date"
            df = df.reset_index()
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
            df = df.sort_values("date").reset_index(drop=True)
            df["ma30"] = df["price"].rolling(30, min_periods=1).mean()
            df["ma90"] = df["price"].rolling(90, min_periods=1).mean()
            df["pct_change"] = df["price"].pct_change() * 100
            return df
        except Exception as e:
            last_exc = e
            if attempt < 2:
                time.sleep(3 * (attempt + 1))   # 3s, then 6s

    st.warning(f"TTF price data temporarily unavailable (Yahoo Finance rate limit). Will retry on next refresh.")
    return pd.DataFrame()


def get_ttf_data() -> dict:
    df = fetch_ttf_prices()
    spike = False
    spike_pct = 0.0
    if not df.empty and "pct_change" in df.columns:
        latest_change = df["pct_change"].iloc[-1]
        if abs(latest_change) > 10:
            spike = True
            spike_pct = latest_change
    return {
        "prices": df,
        "spike": spike,
        "spike_pct": spike_pct,
        "fetched_at": datetime.utcnow(),
    }
