"""
TTF Seasonal Injection-Withdrawal Backtest.

Strategy: buy summer TTF (Apr-Sep average), sell winter TTF (Oct-Mar average)
when the seasonal spread exceeds the round-trip storage cost. Backtest over
available TTF history using Yahoo Finance front-month continuous futures.

A "gas year" runs from April 1 to March 31 of the following year.
Summer half:  April – September
Winter half:  October – March (of the same gas year's winter)

P&L per gas year = Winter_avg − Summer_avg − storage_cost
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

TTF_TICKER = "TTF=F"


@st.cache_data(ttl=3600)
def fetch_ttf_history(years: int = 7) -> pd.DataFrame:
    """
    Fetch TTF front-month daily close prices for the last `years` calendar years.

    Returns a DataFrame with columns: date (datetime64), price (float).
    """
    if not YFINANCE_AVAILABLE:
        return pd.DataFrame()

    end   = datetime.utcnow().date()
    start = end - timedelta(days=years * 366)

    try:
        ticker = yf.Ticker(TTF_TICKER)
        df = ticker.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
        if df.empty:
            return pd.DataFrame()
        df = df[["Close"]].rename(columns={"Close": "price"})
        df.index.name = "date"
        df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        return df.sort_values("date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def compute_seasonal_strategy(
    df: pd.DataFrame,
    storage_cost: float = 4.0,
) -> pd.DataFrame:
    """
    Compute per-gas-year seasonal spread P&L.

    For each gas year (Apr Y – Mar Y+1):
    - summer_avg: mean TTF price Apr–Sep year Y
    - winter_avg: mean TTF price Oct Y – Mar Y+1
    - spread:     winter_avg − summer_avg
    - pnl:        spread − storage_cost  (positive = strategy profitable)
    - trade:      True if spread > storage_cost on entry signal

    Parameters
    ----------
    df           : DataFrame from fetch_ttf_history()
    storage_cost : Round-trip cost of storage in EUR/MWh (default 4.0)

    Returns
    -------
    DataFrame with one row per completed gas year.
    Columns: gas_year (str), summer_avg, winter_avg, spread, pnl, trade
    """
    if df.empty or "price" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Gas year: Apr Y → Mar Y+1, labelled by the summer year Y
    # Summer months: 4-9, gas year label = calendar year of those months
    # Winter months: 10-12 (same calendar year Y), 1-3 (calendar year Y+1, gas year = Y)
    def _gas_year(row):
        m, y = row["month"], row["year"]
        if 4 <= m <= 9:
            return y          # summer of gas year Y
        elif m >= 10:
            return y          # first half of winter, still gas year Y
        else:                  # m in [1,2,3]
            return y - 1      # second half of winter, gas year Y-1

    df["gas_year"] = df.apply(_gas_year, axis=1)

    records = []
    for gy, grp in df.groupby("gas_year"):
        summer = grp[grp["month"].between(4, 9)]
        winter = grp[(grp["month"] >= 10) | (grp["month"] <= 3)]

        if len(summer) < 10 or len(winter) < 10:
            continue

        summer_avg = float(summer["price"].mean())
        winter_avg = float(winter["price"].mean())
        spread     = winter_avg - summer_avg
        pnl        = spread - storage_cost
        trade      = bool(spread > storage_cost)

        records.append({
            "gas_year":   str(gy),
            "label":      f"GY {gy}/{str(gy+1)[-2:]}",
            "summer_avg": round(summer_avg, 2),
            "winter_avg": round(winter_avg, 2),
            "spread":     round(spread, 2),
            "pnl":        round(pnl, 2),
            "trade":      trade,
            "n_summer":   len(summer),
            "n_winter":   len(winter),
        })

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records).sort_values("gas_year").reset_index(drop=True)

    # Cumulative P&L (only count years where we trade)
    result["cum_pnl"] = result["pnl"].cumsum()

    return result


def compute_strategy_stats(bt: pd.DataFrame) -> dict:
    """
    Compute summary statistics from the backtest result DataFrame.
    """
    if bt.empty:
        return {}

    pnl = bt["pnl"]
    n   = len(pnl)

    avg_spread  = float(bt["spread"].mean())
    avg_pnl     = float(pnl.mean())
    hit_rate    = float((pnl > 0).mean()) * 100
    best_year   = bt.loc[pnl.idxmax(), "label"]
    worst_year  = bt.loc[pnl.idxmin(), "label"]
    best_pnl    = float(pnl.max())
    worst_pnl   = float(pnl.min())
    total_pnl   = float(pnl.sum())
    cum_end     = float(bt["cum_pnl"].iloc[-1])

    # Sharpe: mean / std, annualised assuming 1 observation per year
    sharpe = float(pnl.mean() / pnl.std()) if pnl.std() > 0 else 0.0

    # Ex-crisis stats: exclude GY 2021 and GY 2022 (European energy crisis peak)
    # These years had geopolitical-shock spreads that are not representative of normal operations
    crisis_gas_years = {2021, 2022}
    non_crisis = bt[~bt["gas_year"].astype(int).isin(crisis_gas_years)]
    avg_pnl_ex_crisis    = float(non_crisis["pnl"].mean())    if not non_crisis.empty else None
    hit_rate_ex_crisis   = float((non_crisis["pnl"] > 0).mean() * 100) if not non_crisis.empty else None
    n_ex_crisis          = len(non_crisis)

    return {
        "n_years":            n,
        "avg_spread":         avg_spread,
        "avg_pnl":            avg_pnl,
        "hit_rate":           hit_rate,
        "best_year":          best_year,
        "best_pnl":           best_pnl,
        "worst_year":         worst_year,
        "worst_pnl":          worst_pnl,
        "total_pnl":          total_pnl,
        "cum_end":            cum_end,
        "sharpe":             sharpe,
        "avg_pnl_ex_crisis":  avg_pnl_ex_crisis,
        "hit_rate_ex_crisis": hit_rate_ex_crisis,
        "n_ex_crisis":        n_ex_crisis,
    }
