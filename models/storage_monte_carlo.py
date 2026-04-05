"""
Storage Refill Monte Carlo Simulation.

Empirical bootstrap: for each simulated calendar day, a daily injection rate
is drawn with replacement from the historical AGSI distribution for that month.
Injection season: April through October. Storage is held flat outside this window.
"""
import numpy as np
import pandas as pd
from datetime import date, timedelta

# Fallback injection rates (pct points/day) used when AGSI historical data
# is insufficient. Values approximate long-run European seasonal averages.
_FALLBACK_RATES: dict[int, np.ndarray] = {
    4:  np.array([0.28] * 30),
    5:  np.array([0.38] * 31),
    6:  np.array([0.33] * 30),
    7:  np.array([0.28] * 31),
    8:  np.array([0.22] * 31),
    9:  np.array([0.18] * 30),
    10: np.array([0.09] * 31),
}


def compute_monthly_stats(df: pd.DataFrame) -> dict:
    """
    Extract empirical daily injection rates by calendar month from AGSI data.

    Filters to injection season months (April-October), excludes the current
    year to avoid look-ahead bias, and retains only positive daily changes
    (net injection days).

    Parameters
    ----------
    df : pd.DataFrame
        AGSI storage DataFrame with columns 'gasDayStart' and 'full'.
        'full' is the fill level expressed as a percentage (0-100).

    Returns
    -------
    dict[int, np.ndarray]
        Maps month number (4-10) to an array of observed daily injection
        amounts in percentage points per day.
    """
    if df.empty or "full" not in df.columns or "gasDayStart" not in df.columns:
        return {}

    df = df.copy()
    df = df.sort_values("gasDayStart").reset_index(drop=True)
    current_year = date.today().year
    df = df[df["gasDayStart"].dt.year < current_year]
    df["month"] = df["gasDayStart"].dt.month
    df["delta"] = df["full"].diff()

    injection_days = df[df["month"].between(4, 10) & (df["delta"] > 0)]

    stats: dict = {}
    for month in range(4, 11):
        vals = injection_days[injection_days["month"] == month]["delta"].dropna().values
        if len(vals) >= 10:
            stats[month] = vals
    return stats


def run_monte_carlo(
    current_pct: float,
    monthly_stats: dict,
    n_paths: int = 1000,
    rate_multiplier: float = 1.0,
) -> tuple[np.ndarray, list]:
    """
    Run an empirical bootstrap Monte Carlo simulation for gas storage refill.

    For each simulated day, a daily injection increment is drawn with
    replacement from the historical distribution for that calendar month
    and scaled by rate_multiplier. The random seed is fixed per multiplier
    value for reproducibility. Storage is bounded between 0% and 100%.

    Parameters
    ----------
    current_pct : float
        Current EU gas storage fill level as a percentage (0-100).
    monthly_stats : dict
        Output of compute_monthly_stats(). Empty dict triggers fallback rates.
    n_paths : int
        Number of simulation paths (default: 1,000).
    rate_multiplier : float
        Scalar applied to all sampled injection increments. Values below 1.0
        simulate a slower-than-average refill season; above 1.0 an accelerated
        season.

    Returns
    -------
    paths : np.ndarray, shape (n_paths, n_days)
        Storage fill level for each path on each simulated day.
    dates : list[date]
        Corresponding calendar dates (length n_days).
    """
    today = date.today()
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
        month = sim_date.month
        if 4 <= month <= 10:
            pool = monthly_stats.get(month, _FALLBACK_RATES.get(month, np.array([0.2])))
            increments = rng.choice(pool, size=n_paths, replace=True) * rate_multiplier
        else:
            increments = np.zeros(n_paths)

        prev = np.clip(prev + increments, 0.0, 100.0)
        paths[:, j] = prev

    return paths, dates
