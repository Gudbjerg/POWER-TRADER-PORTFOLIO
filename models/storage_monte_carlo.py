"""
Storage Monte Carlo: the EU fill path from today to the next 1 November.

Empirical bootstrap: for each simulated calendar day a net daily change (injection minus
withdrawal, percentage points of working gas volume) is drawn with replacement from the
historical AGSI changes observed in that calendar month. The path therefore runs through
the winter drawdown when it starts in the withdrawal season (November to March) and
through the summer refill afterwards, instead of holding stocks flat until April.
The rate multiplier scales injections (positive changes) only; withdrawals follow history.
"""
import numpy as np
import pandas as pd
from datetime import date, timedelta

# Fallback net daily changes (pp/day) when AGSI history is insufficient: approximate
# long-run EU averages, negative in the withdrawal months.
_FALLBACK_RATES: dict[int, np.ndarray] = {
    1:  np.array([-0.45] * 31), 2: np.array([-0.40] * 28), 3: np.array([-0.25] * 31),
    4:  np.array([0.15] * 30),  5: np.array([0.36] * 31),  6: np.array([0.33] * 30),
    7:  np.array([0.28] * 31),  8: np.array([0.22] * 31),  9: np.array([0.16] * 30),
    10: np.array([0.06] * 31), 11: np.array([-0.15] * 30), 12: np.array([-0.35] * 31),
}


def compute_monthly_stats(df: pd.DataFrame) -> dict:
    """
    Empirical net daily changes in fill (pp/day) by calendar month from AGSI data.

    Excludes the current gas year's data from the current calendar year to avoid
    look-ahead bias. Keeps every day, injection and withdrawal alike, so the draws are
    net changes. Months with fewer than ten observations are left out (fallback used).

    Returns dict[int, np.ndarray]: month number (1-12) to the observed daily changes.
    """
    if df.empty or "full" not in df.columns or "gasDayStart" not in df.columns:
        return {}

    df = df.copy()
    df["gasDayStart"] = pd.to_datetime(df["gasDayStart"])
    df = df.sort_values("gasDayStart").reset_index(drop=True)
    df["delta"] = df["full"].diff()
    # A gap of more than one day would turn several days' change into one draw
    gap = df["gasDayStart"].diff().dt.days
    df = df[(gap == 1) & (df["gasDayStart"].dt.year < date.today().year)]
    df["month"] = df["gasDayStart"].dt.month

    stats: dict = {}
    for month in range(1, 13):
        vals = df.loc[df["month"] == month, "delta"].dropna().values
        if len(vals) >= 10:
            stats[month] = vals
    return stats


def run_monte_carlo(
    current_pct: float,
    monthly_stats: dict,
    n_paths: int = 1000,
    rate_multiplier: float = 1.0,
    today: date | None = None,
) -> tuple[np.ndarray, list]:
    """
    Simulate EU storage fill from tomorrow to the next 1 November.

    Each day draws a net change from that month's empirical distribution; positive draws
    (injection) are scaled by ``rate_multiplier``. Fill is bounded to 0-100%. The random
    seed is fixed per multiplier for reproducible comparisons.

    Returns (paths of shape (n_paths, n_days), list of dates).
    """
    today = today or date.today()
    target = date(today.year, 11, 1)
    if today >= target:
        target = date(today.year + 1, 11, 1)

    dates: list = []
    d = today + timedelta(days=1)
    while d <= target:
        dates.append(d)
        d += timedelta(days=1)

    n_days = len(dates)
    rng = np.random.default_rng(int(rate_multiplier * 100))

    prev = np.full(n_paths, current_pct, dtype=float)
    paths = np.empty((n_paths, n_days), dtype=float)

    for j, sim_date in enumerate(dates):
        pool = monthly_stats.get(sim_date.month, _FALLBACK_RATES[sim_date.month])
        draws = rng.choice(pool, size=n_paths, replace=True)
        increments = np.where(draws > 0, draws * rate_multiplier, draws)
        prev = np.clip(prev + increments, 0.0, 100.0)
        paths[:, j] = prev

    return paths, dates
