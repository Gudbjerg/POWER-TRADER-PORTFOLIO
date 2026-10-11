"""
Realised-volatility overlay for the HMM regime classifier (Layer 4; roadmap 3D).

Pure post-processing of two inputs (no I/O, no Streamlit), unit-tested in
tests/test_regime_volatility.py:
    - the NO2 day-ahead price column of the feature matrix (one row per TTF
      trading day, so a Monday change spans the weekend), and
    - the decoded regime history from models.hmm_model.get_regime_history.

Volatility measure (default, used in the UI): rolling sample standard
deviation (ddof=1) of daily absolute price changes, in EUR/MWh, over the last
REGIME_VOL_WINDOW trading days. Absolute changes are preferred to log changes
because Nordic day-ahead prices can approach zero and occasionally turn
negative, which makes log changes undefined or explosive, and because the HMM
itself measures volatility in EUR/MWh (5-day standard deviation of NO2). The
log-change variant is available (method="log"): changes involving a
non-positive price are set to NaN and excluded from the window.

Transition-spike proximity: a volatility spike is a trading day with realised
volatility above its in-sample REGIME_VOL_SPIKE_PCTL percentile. A regime
transition is a trading day whose decoded regime differs from the previous
day's. The observed share is the fraction of transitions with at least one
spike day within +/- REGIME_VOL_PROXIMITY_DAYS trading days. The chance
baseline is the fraction of all eligible trading days with that property,
i.e. the probability that a transition placed on a random eligible day would
qualify. A one-sided circular-shift test (every cyclic shift of the
transition sequence against the fixed spike sequence, excluding shifts of
fewer than 2 x window + 1 days) gives an exact, deterministic p-value that
preserves the clustering of both sequences:
p = (1 + shifts with at least the observed number of near-spike transitions)
    / (1 + number of shifts).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import (
    REGIME_VOL_PROXIMITY_DAYS,
    REGIME_VOL_SPIKE_PCTL,
    REGIME_VOL_WINDOW,
)


def realised_volatility(
    prices: pd.Series,
    window: int = REGIME_VOL_WINDOW,
    method: str = "abs",
    min_periods: int | None = None,
) -> pd.DataFrame:
    """
    Rolling realised volatility of a daily price series.

    prices: Series indexed by date (one observation per trading day).
    method: "abs" for P_t minus P_{t-1} in EUR/MWh, or "log" for
            ln(P_t / P_{t-1}) with non-positive prices guarded (NaN change).
    min_periods: valid changes required in the window (default: `window`).

    Returns DataFrame[date, price, change, rv] sorted by date.
    """
    if method not in ("abs", "log"):
        raise ValueError("method must be 'abs' or 'log'")
    if window < 2:
        raise ValueError("window must be at least 2")
    columns = ["date", "price", "change", "rv"]
    if prices is None or len(prices) == 0:
        return pd.DataFrame(columns=columns)

    s = pd.to_numeric(prices, errors="coerce")
    s.index = pd.DatetimeIndex(pd.to_datetime(s.index))
    s = s[~s.index.duplicated(keep="last")].sort_index().dropna()
    if s.empty:
        return pd.DataFrame(columns=columns)

    if method == "abs":
        change = s.diff()
    else:
        prev = s.shift(1)
        valid = (s > 0) & (prev > 0)
        change = pd.Series(np.nan, index=s.index)
        change[valid] = np.log(s[valid] / prev[valid])

    mp = window if min_periods is None else int(min_periods)
    rv = change.rolling(window, min_periods=mp).std(ddof=1)
    return pd.DataFrame({
        "date": s.index,
        "price": s.to_numpy(),
        "change": change.to_numpy(),
        "rv": rv.to_numpy(),
    }).reset_index(drop=True)


def _align(regimes: pd.DataFrame, vol: pd.DataFrame) -> pd.DataFrame:
    """Regime history (date, regime) left-joined with realised volatility on date, sorted."""
    if regimes is None or regimes.empty or vol is None or vol.empty:
        return pd.DataFrame(columns=["date", "regime", "price", "rv"])
    r = regimes[["date", "regime"]].copy()
    r["date"] = pd.to_datetime(r["date"]).dt.normalize()
    r = r.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    v = vol[["date", "price", "rv"]].copy()
    v["date"] = pd.to_datetime(v["date"]).dt.normalize()
    v = v.drop_duplicates(subset=["date"], keep="last")
    return r.merge(v, on="date", how="left").reset_index(drop=True)


def regime_transitions(regimes: pd.DataFrame) -> pd.DataFrame:
    """
    Trading days on which the decoded regime differs from the previous day's.
    Returns DataFrame[date, from_regime, to_regime].
    """
    if regimes is None or regimes.empty:
        return pd.DataFrame(columns=["date", "from_regime", "to_regime"])
    r = regimes[["date", "regime"]].copy()
    r["date"] = pd.to_datetime(r["date"]).dt.normalize()
    r = r.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    prev = r["regime"].shift(1)
    mask = prev.notna() & (r["regime"] != prev)
    return pd.DataFrame({
        "date": r.loc[mask, "date"].to_numpy(),
        "from_regime": prev[mask].to_numpy(),
        "to_regime": r.loc[mask, "regime"].to_numpy(),
    })


def volatility_by_regime(regimes: pd.DataFrame, vol: pd.DataFrame) -> pd.DataFrame:
    """
    Realised-volatility statistics per regime on days where both exist.
    Returns DataFrame[regime, days, share, mean_rv, median_rv, mean_price]
    sorted by mean_rv descending.
    """
    columns = ["regime", "days", "share", "mean_rv", "median_rv", "mean_price"]
    merged = _align(regimes, vol).dropna(subset=["rv"])
    if merged.empty:
        return pd.DataFrame(columns=columns)
    out = (
        merged.groupby("regime")
        .agg(days=("rv", "size"), mean_rv=("rv", "mean"), median_rv=("rv", "median"),
             mean_price=("price", "mean"))
        .reset_index()
    )
    out["share"] = out["days"] / out["days"].sum()
    return out.sort_values("mean_rv", ascending=False).reset_index(drop=True)[columns]


def _near_mask(spike: np.ndarray, window: int) -> np.ndarray:
    """True where a spike lies within +/- `window` positions (inclusive)."""
    n = len(spike)
    csum = np.concatenate([[0], np.cumsum(spike.astype(int))])
    idx = np.arange(n)
    lo = np.clip(idx - window, 0, n)
    hi = np.clip(idx + window + 1, 0, n)
    return (csum[hi] - csum[lo]) > 0


def transition_spike_proximity(
    regimes: pd.DataFrame,
    vol: pd.DataFrame,
    spike_pctl: float = REGIME_VOL_SPIKE_PCTL,
    window: int = REGIME_VOL_PROXIMITY_DAYS,
) -> dict:
    """
    Share of regime transitions within +/- `window` trading days of a
    volatility spike, against the chance baseline and a circular-shift p-value.

    Positions are trading days of the regime history. Eligible days are those
    with a defined realised volatility; spikes are eligible days with
    rv > the `spike_pctl` percentile of eligible rv. Only transitions on
    eligible days are scored, and the baseline is measured on the same days,
    so warm-up and sample-edge effects affect both sides equally.

    Returns a dict (empty when fewer than three eligible days exist):
        n_days, n_spike_days, threshold, n_transitions, n_near,
        observed_share, baseline_share, lift, p_value, spike_pctl, window,
        transition_dates, near_flags.
    """
    merged = _align(regimes, vol)
    if merged.empty:
        return {}
    regime = merged["regime"].to_numpy()
    rv = merged["rv"].to_numpy(dtype=float)
    eligible = ~np.isnan(rv)
    if eligible.sum() < 3:
        return {}

    threshold = float(np.nanpercentile(rv, spike_pctl))
    spike = eligible & (rv > threshold)
    near = _near_mask(spike, int(window))

    is_transition = np.zeros(len(regime), dtype=bool)
    is_transition[1:] = regime[1:] != regime[:-1]
    scored = is_transition & eligible

    n_tr = int(scored.sum())
    n_near = int((scored & near).sum())
    baseline = float(near[eligible].mean())
    observed = n_near / n_tr if n_tr else float("nan")
    lift = observed / baseline if (n_tr and baseline > 0) else float("nan")

    # Exact one-sided circular-shift test on the eligible block. Shifts shorter
    # than 2 x window + 1 days leave a transition inside its own neighbourhood
    # and carry no information, so the null set is every cyclic shift k with
    # min_shift <= k <= m - min_shift; the observed arrangement counts once.
    p_value = float("nan")
    t_el = scored[eligible]
    n_el = near[eligible]
    m = len(t_el)
    min_shift = 2 * int(window) + 1
    shifts = range(min_shift, m - min_shift + 1)
    if n_tr and len(shifts) > 0:
        observed_hits = int(np.count_nonzero(t_el & n_el))
        null_hits = np.array([np.count_nonzero(np.roll(t_el, k) & n_el) for k in shifts])
        p_value = float((1 + np.count_nonzero(null_hits >= observed_hits)) / (1 + len(null_hits)))

    return {
        "n_days": int(eligible.sum()),
        "n_spike_days": int(spike.sum()),
        "threshold": threshold,
        "n_transitions": n_tr,
        "n_near": n_near,
        "observed_share": float(observed),
        "baseline_share": baseline,
        "lift": float(lift),
        "p_value": p_value,
        "spike_pctl": float(spike_pctl),
        "window": int(window),
        "transition_dates": list(merged.loc[scored, "date"]),
        "near_flags": list(near[scored]),
    }
