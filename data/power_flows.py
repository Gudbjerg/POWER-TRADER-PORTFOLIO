"""
ENTSO-E cross-border physical flow fetcher.
Uses entsoe-py library. Requires ENTSOE_API_KEY in .env.
"""
import os
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

try:
    from entsoe import EntsoePandasClient
    ENTSOE_PY_AVAILABLE = True
except ImportError:
    ENTSOE_PY_AVAILABLE = False

# Bidding zone EIC codes
ZONES = {
    "NO1": "10YNO-1--------2",
    "NO2": "10YNO-2--------T",
    "SE3": "10Y1001A1001A46L",
    "DE":  "10Y1001A1001A83F",
    "NL":  "10YNL----------L",
    "DK1": "10YDK-1--------W",
    "GB":  "10YGB----------A",
}

# Pairs: (from_zone, to_zone) — positive = export from first zone
FLOW_PAIRS = [
    ("NO1", "SE3"),
    ("NO2", "SE3"),
    ("NO2", "DK1"),
    ("NO2", "DE"),
    ("NO2", "NL"),
    ("NO1", "GB"),
]


def _client():
    key = os.getenv("ENTSOE_API_KEY", "")
    if not key or not ENTSOE_PY_AVAILABLE:
        return None
    return EntsoePandasClient(api_key=key)


@st.cache_data(ttl=3600)
def fetch_all_flows(days: int = 30) -> pd.DataFrame:
    """Fetch net daily cross-border flows for Nordic corridor pairs."""
    client = _client()
    if client is None:
        return pd.DataFrame()

    today = pd.Timestamp.now(tz="Europe/Oslo").normalize()
    start = today - pd.Timedelta(days=days)

    records = []
    for zone_a, zone_b in FLOW_PAIRS:
        label = f"{zone_a}→{zone_b}"
        eic_a = ZONES[zone_a]
        eic_b = ZONES[zone_b]
        try:
            # Export: A → B
            exp = client.query_crossborder_flows(
                country_code_from=eic_a,
                country_code_to=eic_b,
                start=start,
                end=today,
            )
            # Import: B → A
            imp = client.query_crossborder_flows(
                country_code_from=eic_b,
                country_code_to=eic_a,
                start=start,
                end=today,
            )
            # Resample to daily, net = import - export (positive = net import into A)
            if not exp.empty and not imp.empty:
                net = imp.resample("D").sum() - exp.resample("D").sum()
            elif not imp.empty:
                net = imp.resample("D").sum()
            elif not exp.empty:
                net = -exp.resample("D").sum()
            else:
                continue

            for date, val in net.items():
                records.append({"date": date.date(), "pair": label, "net_flow_mwh": float(val)})

        except Exception:
            continue

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records).sort_values(["pair", "date"]).reset_index(drop=True)


def get_flow_data() -> dict:
    df = fetch_all_flows()
    return {"flows": df, "fetched_at": datetime.utcnow()}
