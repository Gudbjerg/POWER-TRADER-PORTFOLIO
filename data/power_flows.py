"""
ENTSO-E physical cross-border flows (A11) on Norway's external corridors.
Uses entsoe-py; requires ENTSOE_API_KEY.

ENTSO-E reports each direction of a border separately as average power (MW) per market
time unit, 15 minutes on these borders. For every corridor the two directions are put on
the same clock (Europe/Oslo), averaged to hourly power and turned into energy per day, so
the figures are MWh whatever the resolution. Sign convention: positive is a net import
into Norway, negative a net export.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from config.settings import ENTSOE_TIMEOUT_S
from data.energy_units import MIN_HOURS_PER_DAY, hourly_power

try:
    from entsoe import EntsoePandasClient
    ENTSOE_PY_AVAILABLE = True
except ImportError:
    ENTSOE_PY_AVAILABLE = False

FLOW_TZ = "Europe/Oslo"

# Bidding zone EIC codes. Germany's side of NordLink is the DE-LU bidding zone; the
# country code (10Y1001A1001A83F) returns no flow data for this border.
ZONES = {
    "NO1":   "10YNO-1--------2",
    "NO2":   "10YNO-2--------T",
    "SE3":   "10Y1001A1001A46L",
    "DE_LU": "10Y1001A1001A82H",
    "NL":    "10YNL----------L",
    "DK1":   "10YDK-1--------W",
    "GB":    "10YGB----------A",
}

# Norway's external corridors: (Norwegian zone, neighbour zone, label, link)
CORRIDORS: list[tuple[str, str, str, str]] = [
    ("NO1", "SE3",   "NO1→SE3", "AC interconnection (Hasle)"),
    ("NO2", "DK1",   "NO2→DK1", "Skagerrak 1-4"),
    ("NO2", "NL",    "NO2→NL",  "NorNed"),
    ("NO2", "DE_LU", "NO2→DE",  "NordLink"),
    ("NO2", "GB",    "NO2→GB",  "North Sea Link"),
]
FLOW_PAIRS = [(no, nb) for no, nb, _, _ in CORRIDORS]

_COLUMNS = ["date", "pair", "net_flow_mwh", "import_mwh", "export_mwh", "abs_net_mwh"]


def _client():
    import os
    key = os.getenv("ENTSOE_API_KEY", "")
    if not key or not ENTSOE_PY_AVAILABLE:
        return None
    return EntsoePandasClient(api_key=key, timeout=ENTSOE_TIMEOUT_S)


def corridor_daily(export_mw: pd.Series, import_mw: pd.Series, label: str) -> pd.DataFrame:
    """Daily energy for one corridor from its two directional power series.

    ``export_mw`` flows out of Norway and ``import_mw`` into it, each with a tz-aware index.
    An hour where only one direction reported counts the other as zero (ENTSO-E reports
    the idle direction as 0, so a gap there is a publication gap). Days with fewer than
    ``MIN_HOURS_PER_DAY`` hours are left out. Returns the columns in ``_COLUMNS``.
    """
    parts = {}
    if not export_mw.empty:
        parts["exp"] = hourly_power(export_mw.astype(float), FLOW_TZ)
    if not import_mw.empty:
        parts["imp"] = hourly_power(import_mw.astype(float), FLOW_TZ)
    if not parts:
        return pd.DataFrame(columns=_COLUMNS)
    hours = pd.concat(parts, axis=1)
    for col in ("exp", "imp"):
        if col not in hours.columns:
            hours[col] = float("nan")
    reported = hours.notna().any(axis=1)
    hours = hours[reported].fillna(0.0)
    if hours.empty:
        return pd.DataFrame(columns=_COLUMNS)
    hours["net"] = hours["imp"] - hours["exp"]
    hours["abs_net"] = hours["net"].abs()
    counts = reported[reported].resample("D").sum()
    daily = hours.resample("D").sum()
    daily = daily[counts.reindex(daily.index, fill_value=0) >= MIN_HOURS_PER_DAY]
    if daily.empty:
        return pd.DataFrame(columns=_COLUMNS)
    return pd.DataFrame({
        "date":         daily.index.tz_localize(None).date,
        "pair":         label,
        "net_flow_mwh": daily["net"].round(1).to_numpy(),
        "import_mwh":   daily["imp"].round(1).to_numpy(),
        "export_mwh":   daily["exp"].round(1).to_numpy(),
        "abs_net_mwh":  daily["abs_net"].round(1).to_numpy(),
    })


def _directional(client, frm: str, to: str, start, end) -> pd.Series:
    try:
        s = client.query_crossborder_flows(ZONES[frm], ZONES[to], start=start, end=end)
    except Exception as exc:
        import sys
        print(f"[flows] ENTSO-E {frm}->{to} unavailable: {type(exc).__name__}", file=sys.stderr)
        return pd.Series(dtype=float)
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return pd.to_numeric(s, errors="coerce").dropna()


@st.cache_data(ttl=3600)
def fetch_all_flows(days: int = 30) -> pd.DataFrame:
    """Net daily flows (MWh) on Norway's external corridors over the last ``days`` days."""
    client = _client()
    if client is None:
        return pd.DataFrame(columns=_COLUMNS)

    end = pd.Timestamp.now(tz=FLOW_TZ).normalize()
    start = end - pd.Timedelta(days=days)

    frames = []
    for no_zone, nb_zone, label, _ in CORRIDORS:
        export_mw = _directional(client, no_zone, nb_zone, start, end)
        import_mw = _directional(client, nb_zone, no_zone, start, end)
        daily = corridor_daily(export_mw, import_mw, label)
        if not daily.empty:
            frames.append(daily)

    if not frames:
        return pd.DataFrame(columns=_COLUMNS)
    return pd.concat(frames, ignore_index=True).sort_values(["pair", "date"]).reset_index(drop=True)


def get_flow_data() -> dict:
    df = fetch_all_flows()
    n_ok = df["pair"].nunique() if not df.empty else 0
    source = f"ENTSO-E ({n_ok}/{len(CORRIDORS)} corridors)" if n_ok > 0 else "unavailable"
    return {"flows": df, "source": source, "fetched_at": datetime.utcnow()}
