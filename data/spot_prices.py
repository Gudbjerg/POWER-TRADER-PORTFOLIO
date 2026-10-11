"""
Day-ahead spot price fetcher using Nord Pool Data Portal API.
No API key required. Covers NO1, NO2, SE3, DE-LU, NL.
"""
import requests
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

NORDPOOL_URL = "https://dataportal-api.nordpoolgroup.com/api/DayAheadPrices"
NORDPOOL_AREAS = ["NO1", "NO2", "SE3", "DE-LU", "NL", "FI"]
HEADERS = {"Referer": "https://www.nordpoolgroup.com/"}

# Display labels for the chart
AREA_LABELS = {
    "NO1":   "NO1 (Oslo)",
    "NO2":   "NO2 (Kristiansand)",
    "SE3":   "SE3 (Stockholm)",
    "DE-LU": "DE/LU (Germany)",
    "NL":    "NL (Netherlands)",
    "FI":    "FI (Finland)",
}


def _fetch_day(date_str: str) -> list[dict]:
    """Fetch daily average prices for one date. Returns list of {zone, price_eur_mwh, date}."""
    try:
        resp = requests.get(
            NORDPOOL_URL,
            params={
                "currency": "EUR",
                "market": "DayAhead",
                "deliveryArea": ",".join(NORDPOOL_AREAS),
                "date": date_str,
            },
            headers=HEADERS,
            timeout=10,
        )
        if resp.status_code == 204:
            return []
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    records = []
    for entry in data.get("areaAverages", []):
        price = entry.get("price")
        if price is not None:
            records.append({
                "date": date_str,
                "zone": entry["areaCode"],
                "price_eur_mwh": price,
            })
    return records


@st.cache_data(ttl=3600)
def fetch_spot_prices(days: int = 30) -> pd.DataFrame:
    """Fetch daily average day-ahead prices for the last `days` days."""
    today = datetime.utcnow().date()
    dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days + 1)]
    records = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_fetch_day, d): d for d in dates}
        for future in as_completed(futures):
            records.extend(future.result())

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date").reset_index(drop=True)
    return df


def get_spot_price_data() -> dict:
    df = fetch_spot_prices()
    return {"prices": df, "fetched_at": datetime.utcnow()}
