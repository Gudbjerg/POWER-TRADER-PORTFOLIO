"""
Rolling Engle-Granger cointegration stability (Layer 2, Cointegration Scanner
tab; deferred roadmap item D2_alt).

Re-runs the scanner's Engle-Granger test on rolling windows to separate pairs
that are persistently cointegrated from pairs that are only episodically so.

Test settings match the scanner exactly: statsmodels.tsa.stattools.coint with
its defaults (constant term, augmented Engle-Granger, ADF lag length chosen by
AIC), run in both regression directions, keeping the lower p-value.

Windows: every window holds COINT_ROLL_WINDOW observations of the pair's own
aligned sample (rows where both prices exist). Window end dates come from the
scanner's common date grid, anchored on the latest date and spaced
COINT_ROLL_STEP grid rows apart, so all pairs share the same x axis.

Pure functions plus statsmodels (no I/O, no Streamlit), unit-tested in
tests/test_coint_stability.py.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from config.settings import (
    COINT_EPISODIC_SHARE,
    COINT_ROLL_ALPHA,
    COINT_ROLL_STEP,
    COINT_ROLL_WINDOW,
    COINT_STABLE_SHARE,
)


def window_end_dates(dates, window: int = COINT_ROLL_WINDOW, step: int = COINT_ROLL_STEP) -> list:
    """
    Window end dates on a common grid: the latest date, then every `step`
    rows back, keeping only ends preceded by at least `window` grid rows
    (inclusive of the end). Returned in ascending order.
    """
    if window < 2 or step < 1:
        raise ValueError("window must be >= 2 and step >= 1")
    grid = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(dates)))).dropna().unique().sort_values()
    n = len(grid)
    if n < window:
        return []
    positions = range(n - 1, window - 2, -step)
    return sorted(grid[p] for p in positions)


def _eg_pvalue(y: np.ndarray, x: np.ndarray) -> float:
    """Engle-Granger p-value of y on x with the scanner's (default) settings; NaN on failure."""
    from statsmodels.tsa.stattools import coint

    if np.nanstd(y) == 0 or np.nanstd(x) == 0:
        return float("nan")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, pvalue, _ = coint(y, x)
        return float(pvalue)
    except Exception:
        return float("nan")


def rolling_eg_pvalues(
    dates,
    y_a,
    y_b,
    end_dates: list,
    window: int = COINT_ROLL_WINDOW,
) -> pd.DataFrame:
    """
    Engle-Granger p-values for one pair on rolling windows.

    For each end date E the window is the last `window` observations of the
    pair's aligned sample dated on or before E. Both directions (a on b,
    b on a) are tested and p_value is the lower of the two, as in the
    full-sample scanner. Windows without enough observations, or where both
    tests fail, carry NaN.

    Returns DataFrame[window_end, window_start, p_value, p_ab, p_ba].
    """
    columns = ["window_end", "window_start", "p_value", "p_ab", "p_ba"]
    frame = pd.DataFrame({
        "date": pd.to_datetime(pd.Series(list(dates))),
        "a": pd.to_numeric(pd.Series(list(y_a)), errors="coerce"),
        "b": pd.to_numeric(pd.Series(list(y_b)), errors="coerce"),
    }).dropna().sort_values("date")
    d = frame["date"].to_numpy()
    a_all = frame["a"].to_numpy(dtype=float)
    b_all = frame["b"].to_numpy(dtype=float)

    rows = []
    for end in end_dates:
        end_ts = pd.Timestamp(end)
        hi = int(np.searchsorted(d, end_ts.to_datetime64(), side="right"))
        lo = hi - int(window)
        if lo < 0:
            rows.append({"window_end": end_ts, "window_start": pd.NaT,
                         "p_value": np.nan, "p_ab": np.nan, "p_ba": np.nan})
            continue
        ya, yb = a_all[lo:hi], b_all[lo:hi]
        p_ab = _eg_pvalue(ya, yb)
        p_ba = _eg_pvalue(yb, ya)
        p = np.nanmin([p_ab, p_ba]) if not (np.isnan(p_ab) and np.isnan(p_ba)) else np.nan
        rows.append({"window_end": end_ts, "window_start": pd.Timestamp(d[lo]),
                     "p_value": float(p), "p_ab": p_ab, "p_ba": p_ba})
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def stability_label(
    share: float,
    stable_share: float = COINT_STABLE_SHARE,
    episodic_share: float = COINT_EPISODIC_SHARE,
) -> str:
    """Persistent (share >= stable_share), Episodic (>= episodic_share), Rare (> 0), Absent (0) or n/a."""
    if share is None or not np.isfinite(share):
        return "n/a"
    if share >= stable_share:
        return "Persistent"
    if share >= episodic_share:
        return "Episodic"
    if share > 0:
        return "Rare"
    return "Absent"


def stability_summary(pvals: pd.DataFrame, alpha: float = COINT_ROLL_ALPHA) -> dict:
    """
    Summarise a rolling_eg_pvalues() frame (windows with NaN p-values are ignored).

    - n_windows, n_coint, share: windows tested, windows with p < alpha, share.
    - current_coint: status of the latest window.
    - current_streak: consecutive latest windows sharing the latest status.
    - last_break: end date of the latest window that turned from p < alpha to
      p >= alpha (None when cointegration never broke within the sample).
    - label: stability_label(share).
    """
    if pvals is None or pvals.empty:
        return {"n_windows": 0, "n_coint": 0, "share": float("nan"), "current_coint": None,
                "current_streak": 0, "last_break": None, "label": "n/a"}
    valid = pvals.dropna(subset=["p_value"]).sort_values("window_end")
    if valid.empty:
        return {"n_windows": 0, "n_coint": 0, "share": float("nan"), "current_coint": None,
                "current_streak": 0, "last_break": None, "label": "n/a"}
    status = (valid["p_value"].to_numpy() < alpha)
    ends = list(valid["window_end"])
    n = len(status)
    current = bool(status[-1])
    streak = 1
    for s in status[-2::-1]:
        if bool(s) != current:
            break
        streak += 1
    last_break = None
    for i in range(n - 1, 0, -1):
        if status[i - 1] and not status[i]:
            last_break = pd.Timestamp(ends[i])
            break
    share = float(status.mean())
    return {
        "n_windows": int(n),
        "n_coint": int(status.sum()),
        "share": share,
        "current_coint": current,
        "current_streak": int(streak),
        "last_break": last_break,
        "label": stability_label(share),
    }
