"""
TTF natural gas seasonal forward curve.

Constructs an 18-month implied forward strip anchored to today's spot price,
calibrated from 3 years of historical TTF seasonal patterns. Not a substitute
for live broker quotes — intended to show the structural shape of the curve
(winter premium, summer trough, Cal-year averages).
"""
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

TTF_TICKER = "TTF=F"

_MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}
_WINTER_MONTHS = {10, 11, 12, 1, 2, 3}


@st.cache_data(ttl=3600)
def _fetch_ttf_long() -> pd.DataFrame:
    """Fetch 3 years of daily TTF prices for seasonal calibration."""
    if not YFINANCE_AVAILABLE:
        return pd.DataFrame()

    end   = datetime.utcnow().date()
    start = end - timedelta(days=365 * 3 + 30)

    try:
        ticker = yf.Ticker(TTF_TICKER)
        df = ticker.history(start=str(start), end=str(end))
        if df.empty:
            return pd.DataFrame()
        df = df[["Close"]].rename(columns={"Close": "price"})
        df.index.name = "date"
        df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        return df.sort_values("date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def _seasonal_curve(hist: pd.DataFrame, spot: float) -> pd.DataFrame:
    """
    Build implied forward strip from historical seasonal indices.

    Returns DataFrame with columns:
        label, expiry, price, p25, p75, season, month_num
    """
    if hist.empty or spot <= 0:
        return pd.DataFrame()

    hist = hist.copy()
    hist["month"] = pd.to_datetime(hist["date"]).dt.month
    hist["year"]  = pd.to_datetime(hist["date"]).dt.year

    # Per-year seasonal index = monthly_avg / annual_avg
    ratios = []
    for year, grp in hist.groupby("year"):
        if len(grp) < 200:          # skip partial years at the edges
            continue
        ann_avg = grp["price"].mean()
        if ann_avg <= 0:
            continue
        for month, mgrp in grp.groupby("month"):
            ratios.append({"month": int(month), "ratio": mgrp["price"].mean() / ann_avg})

    if ratios:
        ratio_df = pd.DataFrame(ratios)
        idx = ratio_df.groupby("month")["ratio"].median().to_dict()
    else:
        # Fallback: raw monthly average / overall mean
        mavg = hist.groupby("month")["price"].mean()
        idx  = (mavg / mavg.mean()).to_dict()

    # Historical P25/P75 per calendar month (all years combined)
    stats = hist.groupby("month")["price"].agg(
        p25=lambda x: x.quantile(0.25),
        p75=lambda x: x.quantile(0.75),
    )

    today = pd.Timestamp.now()
    rows  = []
    for m_ahead in range(1, 19):
        dt        = today + pd.DateOffset(months=m_ahead)
        month_num = int(dt.month)
        price     = round(float(spot) * idx.get(month_num, 1.0), 2)
        p25 = round(float(stats.loc[month_num, "p25"]), 2) if month_num in stats.index else round(price * 0.85, 2)
        p75 = round(float(stats.loc[month_num, "p75"]), 2) if month_num in stats.index else round(price * 1.15, 2)
        rows.append({
            "label":     f"{_MONTH_NAMES[month_num]} {str(dt.year)[2:]}",
            "expiry":    dt.replace(day=1),
            "price":     price,
            "p25":       p25,
            "p75":       p75,
            "season":    "Winter" if month_num in _WINTER_MONTHS else "Summer",
            "month_num": month_num,
        })

    return pd.DataFrame(rows)


@st.cache_data(ttl=3600)
def get_forward_curve_data(spot_price: float) -> dict:
    """
    Return forward curve data dict:
        curve       — DataFrame (label, expiry, price, p25, p75, season)
        summary     — dict with summer_avg, winter_avg, winter_premium, cal26_avg, cal27_avg
        data_years  — number of historical years used for calibration
        fetched_at  — datetime
    """
    hist  = _fetch_ttf_long()
    curve = _seasonal_curve(hist, spot_price)

    summary: dict = {}
    if not curve.empty:
        summer = curve[curve["season"] == "Summer"]["price"]
        winter = curve[curve["season"] == "Winter"]["price"]

        summer_avg = round(float(summer.mean()), 1) if not summer.empty else None
        winter_avg = round(float(winter.mean()), 1) if not winter.empty else None
        summary["summer_avg"]      = summer_avg
        summary["winter_avg"]      = winter_avg
        summary["winter_premium"]  = (
            round(winter_avg - summer_avg, 1)
            if winter_avg is not None and summer_avg is not None else None
        )

        for cal_yr in [2026, 2027]:
            yr_slice = curve[curve["expiry"].dt.year == cal_yr]["price"]
            summary[f"cal{cal_yr}_avg"] = round(float(yr_slice.mean()), 1) if not yr_slice.empty else None

    n_years = hist["date"].dt.year.nunique() if not hist.empty else 0

    return {
        "curve":      curve,
        "summary":    summary,
        "data_years": n_years,
        "spot":       spot_price,
        "fetched_at": datetime.utcnow(),
    }
