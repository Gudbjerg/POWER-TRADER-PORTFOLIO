"""
Gas-to-Power OLS Regression.

Regresses German day-ahead electricity price on TTF front-month gas price using
rolling 60-day OLS. Tracks the time-varying gas-power relationship, rolling R-squared,
and standardised residuals. A residual z-score exceeding ±2 indicates the power price
is deviating materially from what the prevailing gas price implies.
"""
import numpy as np
import pandas as pd


def prepare_data(
    ttf_df: pd.DataFrame,
    spot_df: pd.DataFrame,
    power_zone: str = "NL",
) -> pd.DataFrame:
    """
    Merge TTF and a day-ahead power price series on calendar date.

    Uses NL (Netherlands) by default. NL is the most natural pairing with TTF
    (both Netherlands-based markets) and is available from Nord Pool without
    additional credentials. DE-LU is listed by Nord Pool but returns null prices
    as it falls under EPEX Spot licensing.

    Parameters
    ----------
    ttf_df : pd.DataFrame
        Output of get_ttf_data()['prices']. Requires 'date' and 'price' columns.
    spot_df : pd.DataFrame
        Long-format spot price DataFrame. Requires 'date', 'zone', 'price_eur_mwh'.
    power_zone : str
        Bidding zone to use as the power price proxy (default: 'NL').

    Returns
    -------
    pd.DataFrame with columns: date, ttf_price, power_price. Inner join on date.
    """
    if ttf_df.empty or spot_df.empty:
        return pd.DataFrame()

    de = spot_df[spot_df["zone"] == power_zone][["date", "price_eur_mwh"]].copy()
    de = de.rename(columns={"price_eur_mwh": "power_price"})
    de["date"] = pd.to_datetime(de["date"]).dt.date

    ttf = ttf_df[["date", "price"]].copy()
    ttf = ttf.rename(columns={"price": "ttf_price"})
    ttf["date"] = pd.to_datetime(ttf["date"]).dt.date

    merged = de.merge(ttf, on="date", how="inner")
    merged = merged.dropna(subset=["power_price", "ttf_price"])
    merged = merged.sort_values("date").reset_index(drop=True)
    return merged


def _rolling_ols(x: np.ndarray, y: np.ndarray, window: int):
    """
    Compute rolling OLS of y ~ x over a sliding window.

    Returns (slopes, intercepts, r2s) — each an array of length len(x).
    Leading entries within the first `window` steps are NaN.
    """
    n = len(x)
    slopes     = np.full(n, np.nan)
    intercepts = np.full(n, np.nan)
    r2s        = np.full(n, np.nan)

    for i in range(window - 1, n):
        xi = x[i - window + 1:i + 1]
        yi = y[i - window + 1:i + 1]
        mask = ~(np.isnan(xi) | np.isnan(yi))
        if mask.sum() < max(10, window // 3):
            continue
        xm, ym = xi[mask], yi[mask]
        coeffs = np.polyfit(xm, ym, 1)
        fitted = np.polyval(coeffs, xm)
        ss_res = np.sum((ym - fitted) ** 2)
        ss_tot = np.sum((ym - ym.mean()) ** 2)
        slopes[i]     = coeffs[0]
        intercepts[i] = coeffs[1]
        r2s[i]        = max(0.0, 1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return slopes, intercepts, r2s


def run_full_ols(df: pd.DataFrame) -> dict:
    """
    Run a single OLS regression over the full sample period.

    Appropriate when the sample is too short for a meaningful rolling window
    (fewer than ~80 overlapping observations). Returns slope, intercept, R-squared,
    and standardised residuals for the full period.

    Parameters
    ----------
    df : pd.DataFrame
        Output of prepare_data(). Columns: date, ttf_price, power_price.

    Returns
    -------
    dict with keys: slope, intercept, r2, fitted (array), residual (array),
                    residual_zscore (array), n_obs (int).
    Returns empty dict if fewer than 20 observations.
    """
    if df.empty or len(df) < 20:
        return {}

    x = df["ttf_price"].values.astype(float)
    y = df["power_price"].values.astype(float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]

    coeffs   = np.polyfit(x, y, 1)
    fitted   = np.polyval(coeffs, x)
    residual = y - fitted

    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2     = max(0.0, 1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    res_std = residual.std()
    residual_z = residual / res_std if res_std > 0 else residual

    return {
        "slope":          float(coeffs[0]),
        "intercept":      float(coeffs[1]),
        "r2":             float(r2),
        "fitted":         fitted,
        "residual":       residual,
        "residual_zscore": residual_z,
        "n_obs":          int(mask.sum()),
    }


def run_regression(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    Run rolling OLS regression of power price on TTF gas price.

    Parameters
    ----------
    df : pd.DataFrame
        Output of prepare_data(). Columns: date, ttf_price, power_price.
    window : int
        Rolling window in calendar days with price data (default: 60).

    Returns
    -------
    pd.DataFrame with additional columns:
        fitted           : model-implied power price from rolling OLS
        residual         : actual minus fitted (EUR/MWh)
        rolling_r2       : rolling R-squared (0-1)
        rolling_slope    : rolling beta (power price sensitivity to TTF)
        residual_zscore  : residual normalised by rolling standard deviation
    """
    if df.empty or len(df) < window:
        return pd.DataFrame()

    x = df["ttf_price"].values.astype(float)
    y = df["power_price"].values.astype(float)

    slopes, intercepts, r2s = _rolling_ols(x, y, window)

    fitted   = slopes * x + intercepts
    residual = y - fitted

    res_series = pd.Series(residual)
    roll_std   = res_series.rolling(window, min_periods=window // 2).std()
    roll_mean  = res_series.rolling(window, min_periods=window // 2).mean()
    residual_z = ((res_series - roll_mean) / roll_std).values

    out = df.copy()
    out["fitted"]          = fitted
    out["residual"]        = residual
    out["rolling_r2"]      = r2s
    out["rolling_slope"]   = slopes
    out["residual_zscore"] = residual_z
    return out
