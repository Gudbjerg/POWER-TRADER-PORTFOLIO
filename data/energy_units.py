"""
Power-to-energy conversion and daily averaging for ENTSO-E series.

ENTSO-E publishes generation, wind and solar forecasts, physical cross-border flows and
day-ahead prices as an average over each market time unit. That unit is 15 minutes for most
European areas (Germany throughout, the Nordic areas since the 15-minute market time unit),
and an hour for some series. Summing the readings over a day gives MWh only for hourly data
and four times the energy for 15-minute data. These helpers average to hourly values first,
so a day's energy is the sum of its hourly mean power, and a day's average price is the mean
of its hourly mean prices, whatever the reporting resolution.
"""
from __future__ import annotations

import pandas as pd

# A day counts as published when this many clock hours carry at least one reading. 23
# tolerates one missing hour and accepts the 23-hour day when clocks go forward.
MIN_HOURS_PER_DAY = 23


def hourly_power(power: pd.Series | pd.DataFrame, tz: str = "UTC") -> pd.Series | pd.DataFrame:
    """Average value per clock hour in ``tz``; hours without a reading are NaN.

    ``power`` must have a tz-aware DatetimeIndex (ENTSO-E series always do). The values are
    averages over their interval, such as power in MW or a price in EUR/MWh.
    """
    if power.empty:
        return power.copy()
    if getattr(power.index, "tz", None) is None:
        raise ValueError("power readings need a tz-aware index")
    out = power.copy()
    out.index = out.index.tz_convert(tz)
    out = out.sort_index()
    return out.resample("h").mean()


def _complete_days(hourly: pd.Series | pd.DataFrame, daily: pd.Series | pd.DataFrame,
                   min_hours: int) -> pd.Series | pd.DataFrame:
    """Keep the days of ``daily`` with at least ``min_hours`` reported hours; naive index."""
    reported = hourly.notna() if isinstance(hourly, pd.Series) else hourly.notna().any(axis=1)
    hours = reported.resample("D").sum()
    daily = daily[hours.reindex(daily.index, fill_value=0) >= min_hours]
    daily.index = daily.index.tz_localize(None)
    return daily


def daily_energy(power: pd.Series | pd.DataFrame, tz: str = "UTC",
                 min_hours: int = MIN_HOURS_PER_DAY) -> pd.Series | pd.DataFrame:
    """Energy per calendar day in ``tz`` from hourly mean power (MWh for MW input).

    Days with fewer than ``min_hours`` hourly readings are left out: they are still being
    published (typically the current day) and would read as low output. The index is the
    naive midnight of each day in ``tz``.
    """
    hourly = hourly_power(power, tz)
    if hourly.empty:
        return hourly
    return _complete_days(hourly, hourly.resample("D").sum(min_count=1), min_hours)


def daily_mean(values: pd.Series | pd.DataFrame, tz: str = "UTC",
               min_hours: int = MIN_HOURS_PER_DAY) -> pd.Series | pd.DataFrame:
    """Time-weighted average per calendar day in ``tz`` (the base price for prices).

    Each hour weighs the same whether it was reported as one hourly value or four
    15-minute values. Days with fewer than ``min_hours`` reported hours are left out. The
    index is the naive midnight of each day in ``tz``.
    """
    hourly = hourly_power(values, tz)
    if hourly.empty:
        return hourly
    return _complete_days(hourly, hourly.resample("D").mean(), min_hours)
