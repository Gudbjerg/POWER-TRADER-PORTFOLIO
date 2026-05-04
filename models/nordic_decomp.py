"""
Nordic price decomposition via rolling multivariate OLS.

Regresses NO2 day-ahead price on continental price (NL), gas price (TTF),
and optionally hydro reservoir level (hydro_pct) and German wind output
(de_wind_gwh). Regressors are globally standardised before fitting so each
beta represents EUR/MWh per one-standard-deviation change in that driver,
making magnitudes directly comparable across variables with different units.

Rolling 90-day window via statsmodels RollingOLS. Degrades gracefully:
  - 4-factor (NL + TTF + hydro + wind) when all data available
  - 3-factor (NL + TTF + hydro) when wind missing
  - 2-factor (NL + TTF) when neither hydro nor wind available

Connects forward to Phase D1 cointegration scanner: the same rolling
regression machinery, extended with cointegration testing, underpins
pair-trading signal generation for the Nordic-Continental spread.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    from statsmodels.regression.rolling import RollingOLS
    import statsmodels.api as sm
    _STATSMODELS = True
except ImportError:
    _STATSMODELS = False


# ── Display labels and colours ────────────────────────────────────────────────

FACTOR_LABELS: dict[str, str] = {
    "nl":          "Continental price (NL)",
    "ttf":         "Gas price (TTF)",
    "hydro_pct":   "Hydro reservoir level",
    "de_wind_gwh": "German wind output",
}

FACTOR_COLORS: dict[str, str] = {
    "nl":          "#58a6ff",
    "ttf":         "#e07b39",
    "hydro_pct":   "#3fb950",
    "de_wind_gwh": "#a371f7",
}


# ── Feature selection ─────────────────────────────────────────────────────────

def select_features(df: pd.DataFrame) -> tuple[list[str], str]:
    """
    Choose regression features based on data availability.

    Returns
    -------
    (feature_cols, model_label)
    """
    has_hydro = (
        "hydro_pct" in df.columns
        and df["hydro_pct"].notna().sum() >= 50
    )
    has_wind = (
        "de_wind_gwh" in df.columns
        and df["de_wind_gwh"].notna().sum() >= 50
    )

    features = ["nl", "ttf"]
    if has_hydro:
        features.append("hydro_pct")
    if has_wind:
        features.append("de_wind_gwh")

    labels = {
        2: "2-factor model (NL + TTF only — hydro and wind unavailable)",
        3: "3-factor model (NL + TTF + hydro — wind unavailable)",
        4: "4-factor model (NL + TTF + hydro + wind)",
    }
    return features, labels[len(features)]


# ── Rolling OLS ───────────────────────────────────────────────────────────────

def run_rolling_decomposition(
    df: pd.DataFrame,
    window: int = 90,
) -> tuple[pd.DataFrame, list[str], str]:
    """
    Rolling multivariate OLS: NO2 ~ NL + TTF [+ hydro_pct] [+ de_wind_gwh]

    Regressors are globally standardised (mean 0, std 1) so beta magnitudes
    represent EUR/MWh per one-standard-deviation change in the driver. This
    makes coefficients comparable across factors with different units.

    Parameters
    ----------
    df     : Feature matrix from assemble_features(). Must contain 'date', 'no2',
             'nl', 'ttf'. Optional: 'hydro_pct', 'de_wind_gwh'.
    window : Rolling window in trading days (default 90, minimum ~45 for stability).

    Returns
    -------
    (results_df, feature_cols, model_label)

    results_df columns: date, beta_<col> for each feature, r2.
    NaN rows correspond to observations within the first window period.
    """
    if not _STATSMODELS:
        return pd.DataFrame(), [], "statsmodels unavailable — pip install statsmodels"

    feature_cols, model_label = select_features(df)
    target = "no2"

    required = ["date", target] + feature_cols
    clean = df[required].dropna().copy()
    if len(clean) < window + 10:
        return pd.DataFrame(), feature_cols, model_label

    clean = clean.sort_values("date").reset_index(drop=True)

    # Standardise globally — betas are EUR/MWh per 1 std-dev of the full sample
    X_raw  = clean[feature_cols]
    X_std  = X_raw.std().replace(0, 1.0)
    X_mean = X_raw.mean()
    X_scaled = ((X_raw - X_mean) / X_std).values

    y = clean[target].values
    X = sm.add_constant(X_scaled, prepend=True, has_constant="add")

    try:
        fit    = RollingOLS(y, X, window=window, min_nobs=max(window // 2, 20)).fit()
        params = fit.params   # ndarray shape (n_obs, n_params)
        r2     = fit.rsquared
    except Exception:
        return pd.DataFrame(), feature_cols, model_label

    results = pd.DataFrame({"date": clean["date"].values})
    for i, col in enumerate(feature_cols):
        results[f"beta_{col}"] = params[:, i + 1]   # column 0 is intercept
    results["r2"] = r2

    return results, feature_cols, model_label


# ── Derived analytics ─────────────────────────────────────────────────────────

def current_contributions(
    df: pd.DataFrame,
    results: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, float]:
    """
    Compute each factor's contribution to the current NO2 level.

    Contribution = beta_i × z_i, where z_i is the current standardised value
    of factor i. Units: EUR/MWh. Positive = upward pressure on NO2.

    Returns empty dict if data or results are insufficient.
    """
    if results.empty or df.empty or not feature_cols:
        return {}

    latest_result = results.dropna(subset=[f"beta_{c}" for c in feature_cols]).iloc[-1]
    latest_data   = df.sort_values("date").dropna(subset=feature_cols).iloc[-1]

    X_std  = df[feature_cols].std().replace(0, 1.0)
    X_mean = df[feature_cols].mean()

    out: dict[str, float] = {}
    for col in feature_cols:
        beta = latest_result.get(f"beta_{col}", np.nan)
        raw  = latest_data.get(col, np.nan)
        if pd.isna(beta) or pd.isna(raw):
            out[col] = 0.0
        else:
            z_i = (raw - X_mean[col]) / X_std[col]
            out[col] = float(beta * z_i)
    return out


def dominant_driver(results: pd.DataFrame, feature_cols: list[str]) -> tuple[str, float]:
    """
    Return (factor_key, |beta|) for the factor with the largest absolute beta
    in the most recent observation.
    """
    if results.empty or not feature_cols:
        return "unknown", 0.0

    latest = results.dropna(subset=[f"beta_{c}" for c in feature_cols]).iloc[-1]
    best_col, best_abs = feature_cols[0], 0.0
    for col in feature_cols:
        b = abs(float(latest.get(f"beta_{col}", 0.0) or 0.0))
        if b > best_abs:
            best_abs, best_col = b, col
    return best_col, best_abs


# ── Charts ────────────────────────────────────────────────────────────────────

def make_beta_chart(
    results: pd.DataFrame,
    feature_cols: list[str],
    window: int = 90,
) -> go.Figure:
    """
    Rolling beta time series — one line per factor.
    Y-axis: EUR/MWh per one-std-dev change in the factor.
    """
    fig = go.Figure()

    valid = results.dropna(subset=[f"beta_{c}" for c in feature_cols])

    for col in feature_cols:
        label = FACTOR_LABELS.get(col, col)
        color = FACTOR_COLORS.get(col, "#8b949e")
        fig.add_trace(go.Scatter(
            x=valid["date"],
            y=valid[f"beta_{col}"],
            mode="lines",
            name=label,
            line=dict(color=color, width=1.8),
            hovertemplate=f"<b>{label}</b><br>β = %{{y:.2f}} EUR/MWh per σ<extra></extra>",
        ))

    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.2)", width=1))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        height=320,
        margin=dict(l=50, r=20, t=20, b=40),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(size=10, color="#8b949e")),
        yaxis=dict(
            title=f"β (EUR/MWh per 1 std dev) — {window}-day rolling",
            gridcolor="rgba(255,255,255,0.06)",
            tickfont=dict(size=10, color="#8b949e"),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(size=10, color="#8b949e"),
        ),
        hovermode="x unified",
    )
    return fig


def make_contribution_bar(
    contributions: dict[str, float],
    no2_actual: float | None = None,
) -> go.Figure:
    """
    Today's factor contribution bar chart.
    Each bar shows how much that driver is adding/subtracting from NO2 (EUR/MWh).
    """
    if not contributions:
        return go.Figure()

    labels = [FACTOR_LABELS.get(k, k) for k in contributions]
    values = list(contributions.values())
    colors = [
        (FACTOR_COLORS.get(k, "#8b949e") if v >= 0 else "#f85149")
        for k, v in contributions.items()
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        hovertemplate="%{x}: %{y:+.1f} EUR/MWh<extra></extra>",
        name="Factor contribution",
    ))

    if no2_actual is not None:
        fig.add_hline(
            y=no2_actual,
            line=dict(color="#e6edf3", width=1.5, dash="dot"),
            annotation_text=f"NO2 actual: €{no2_actual:.1f}/MWh",
            annotation_font=dict(color="#e6edf3", size=11),
            annotation_position="top right",
        )

    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.2)", width=1))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        height=280,
        margin=dict(l=50, r=20, t=20, b=60),
        xaxis=dict(tickfont=dict(size=11, color="#8b949e")),
        yaxis=dict(
            title="Contribution (EUR/MWh)",
            gridcolor="rgba(255,255,255,0.06)",
            tickfont=dict(size=10, color="#8b949e"),
        ),
        showlegend=False,
    )
    return fig
