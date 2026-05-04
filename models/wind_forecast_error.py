"""
Wind forecast error tracker.

Compares ENTSO-E A69 day-ahead wind generation forecast against actual
generation (ENTSO-E B18/B19 or Fraunhofer ISE fallback for DE).

Metrics:
  - Daily forecast error (GWh and %) per country
  - Rolling 7-day RMSE per country
  - Correlation: |forecast error| vs |day-over-day NO2 price change|
    (tests whether large wind forecast misses co-move with intraday/
     next-day price volatility)

Countries covered: DE (most reliable — Fraunhofer fallback for actuals),
DK1, NO, GB (ENTSO-E A69 data availability subject to server status).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ── Zone display config ────────────────────────────────────────────────────────

ZONE_LABELS: dict[str, str] = {
    "DE":  "Germany (DE)",
    "DK1": "Denmark West (DK1)",
    "NO":  "Norway (NO)",
    "GB":  "Great Britain (GB)",
}

ZONE_COLORS: dict[str, str] = {
    "DE":  "#e07b39",
    "DK1": "#58a6ff",
    "NO":  "#3fb950",
    "GB":  "#a371f7",
}


# ── Error computation ─────────────────────────────────────────────────────────

def compute_errors(
    forecast_df: pd.DataFrame,
    actual_daily_df: pd.DataFrame,
    zones: list[str],
) -> pd.DataFrame:
    """
    Join forecast and actual wind generation; compute daily error per zone.

    Parameters
    ----------
    forecast_df    : Output of fetch_wind_forecast()['forecast'].
                     Columns: date, <zone>_forecast_gwh
    actual_daily_df: Output of fetch_wind_daily()['daily'].
                     Columns: date, de_wind_gwh [, no_wind_gwh]
    zones          : List of zones to process (those with forecast data).

    Returns
    -------
    pd.DataFrame with columns: date, <zone>_error_gwh, <zone>_error_pct
    for each zone that has both forecast and actual data.
    """
    if forecast_df.empty or actual_daily_df.empty or not zones:
        return pd.DataFrame()

    fc  = forecast_df.copy()
    act = actual_daily_df.copy()
    fc["date"]  = pd.to_datetime(fc["date"]).dt.normalize()
    act["date"] = pd.to_datetime(act["date"]).dt.normalize()

    # Actual data column name mapping
    _actual_col = {
        "DE": "de_wind_gwh",
        "NO": "no_wind_gwh",
    }

    merged = fc.merge(act, on="date", how="inner")
    out = pd.DataFrame({"date": merged["date"]})

    for zone in zones:
        fc_col = f"{zone}_forecast_gwh"
        if fc_col not in merged.columns:
            continue
        act_col = _actual_col.get(zone)
        if act_col is None or act_col not in merged.columns:
            continue

        fc_vals  = pd.to_numeric(merged[fc_col], errors="coerce")
        act_vals = pd.to_numeric(merged[act_col], errors="coerce")
        error    = fc_vals - act_vals    # positive = over-forecast

        out[f"{zone}_error_gwh"] = error.round(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = np.where(act_vals > 0, error / act_vals * 100.0, np.nan)
        out[f"{zone}_error_pct"] = np.round(pct, 1)

    if out.shape[1] <= 1:  # only date column
        return pd.DataFrame()

    return out.sort_values("date").reset_index(drop=True)


def compute_rolling_rmse(
    error_df: pd.DataFrame,
    zones: list[str],
    window: int = 7,
) -> pd.DataFrame:
    """
    Compute rolling RMSE for each zone's GWh forecast error.

    Returns
    -------
    pd.DataFrame with columns: date, <zone>_rmse_gwh
    """
    if error_df.empty or not zones:
        return pd.DataFrame()

    out = pd.DataFrame({"date": error_df["date"]})
    for zone in zones:
        col = f"{zone}_error_gwh"
        if col not in error_df.columns:
            continue
        sq_err = error_df[col] ** 2
        rmse   = sq_err.rolling(window, min_periods=max(window // 2, 2)).mean() ** 0.5
        out[f"{zone}_rmse_gwh"] = rmse.round(1)

    return out if out.shape[1] > 1 else pd.DataFrame()


def compute_price_correlation(
    error_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    zone: str = "DE",
    window: int = 30,
) -> pd.DataFrame:
    """
    Compute rolling correlation between |wind forecast error| and |price change|.

    Uses the absolute day-over-day change in NO2 day-ahead price as the price
    volatility proxy (derived from the feature matrix).

    Returns
    -------
    pd.DataFrame with columns: date, abs_error_gwh, abs_price_change, rolling_corr
    """
    error_col = f"{zone}_error_gwh"
    if error_df.empty or error_col not in error_df.columns:
        return pd.DataFrame()
    if feature_df.empty or "no2" not in feature_df.columns:
        return pd.DataFrame()

    err = error_df[["date", error_col]].copy()
    err["date"] = pd.to_datetime(err["date"]).dt.normalize()
    err["abs_error_gwh"] = err[error_col].abs()

    price = feature_df[["date", "no2"]].copy()
    price["date"] = pd.to_datetime(price["date"]).dt.normalize()
    price = price.sort_values("date")
    price["abs_price_change"] = price["no2"].diff().abs()

    merged = err.merge(price[["date", "abs_price_change"]], on="date", how="inner")
    merged = merged.dropna(subset=["abs_error_gwh", "abs_price_change"])

    if len(merged) < window:
        return pd.DataFrame()

    merged["rolling_corr"] = (
        merged["abs_error_gwh"]
        .rolling(window, min_periods=window // 2)
        .corr(merged["abs_price_change"])
        .round(3)
    )

    return merged[["date", "abs_error_gwh", "abs_price_change", "rolling_corr"]].reset_index(drop=True)


# ── Charts ────────────────────────────────────────────────────────────────────

def make_error_bar_chart(
    error_df: pd.DataFrame,
    zones: list[str],
    last_n_days: int = 60,
) -> go.Figure:
    """
    Daily forecast error bar chart per zone (last N days).
    Positive = over-forecast; negative = under-forecast.
    """
    fig = go.Figure()

    cutoff = pd.Timestamp.now() - pd.Timedelta(days=last_n_days)
    df = error_df[pd.to_datetime(error_df["date"]) >= cutoff].copy()

    for zone in zones:
        col = f"{zone}_error_gwh"
        if col not in df.columns:
            continue
        vals   = df[col]
        colors = ["#f85149" if v > 0 else "#3fb950" for v in vals]
        fig.add_trace(go.Bar(
            x=df["date"],
            y=vals,
            name=ZONE_LABELS.get(zone, zone),
            marker_color=colors,
            opacity=0.75,
            hovertemplate=f"<b>{ZONE_LABELS.get(zone, zone)}</b><br>Error: %{{y:.1f}} GWh<extra></extra>",
        ))

    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.25)", width=1))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        barmode="group",
        height=320,
        margin=dict(l=55, r=20, t=20, b=40),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(size=10, color="#8b949e")),
        yaxis=dict(
            title="Forecast error (GWh)  [positive = over-forecast]",
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


def make_rmse_chart(
    rmse_df: pd.DataFrame,
    zones: list[str],
) -> go.Figure:
    """Rolling 7-day RMSE line chart per zone."""
    fig = go.Figure()

    for zone in zones:
        col = f"{zone}_rmse_gwh"
        if col not in rmse_df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=rmse_df["date"],
            y=rmse_df[col],
            mode="lines",
            name=ZONE_LABELS.get(zone, zone),
            line=dict(color=ZONE_COLORS.get(zone, "#8b949e"), width=1.8),
            hovertemplate=(
                f"<b>{ZONE_LABELS.get(zone, zone)}</b><br>"
                "7-day RMSE: %{y:.1f} GWh<extra></extra>"
            ),
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        height=280,
        margin=dict(l=55, r=20, t=20, b=40),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(size=10, color="#8b949e")),
        yaxis=dict(
            title="7-day rolling RMSE (GWh)",
            gridcolor="rgba(255,255,255,0.06)",
            tickfont=dict(size=10, color="#8b949e"),
            rangemode="tozero",
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(size=10, color="#8b949e"),
        ),
        hovermode="x unified",
    )
    return fig


def make_correlation_scatter(
    corr_df: pd.DataFrame,
    zone: str = "DE",
) -> go.Figure:
    """
    Scatter: |wind forecast error| (GWh) vs |price change| (EUR/MWh).
    Annotated with rolling 30-day correlation.
    """
    if corr_df.empty:
        return go.Figure()

    latest_corr = corr_df["rolling_corr"].dropna().iloc[-1] if not corr_df["rolling_corr"].dropna().empty else float("nan")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=corr_df["abs_error_gwh"],
        y=corr_df["abs_price_change"],
        mode="markers",
        marker=dict(
            color=ZONE_COLORS.get(zone, "#58a6ff"),
            size=5,
            opacity=0.55,
        ),
        hovertemplate=(
            "Error: %{x:.1f} GWh<br>"
            "|ΔPrice|: %{y:.1f} EUR/MWh<extra></extra>"
        ),
        name="Observations",
    ))

    # Rolling correlation annotation
    corr_label = f"Rolling 30d ρ = {latest_corr:.2f}" if not np.isnan(latest_corr) else "Rolling corr: n/a"
    fig.add_annotation(
        xref="paper", yref="paper", x=0.98, y=0.98,
        text=corr_label, showarrow=False,
        font=dict(size=12, color="#e6edf3"),
        bgcolor="rgba(22,27,34,0.85)",
        bordercolor="rgba(255,255,255,0.15)",
        borderwidth=1,
        xanchor="right", yanchor="top",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        height=300,
        margin=dict(l=55, r=20, t=20, b=50),
        xaxis=dict(
            title=f"|Forecast error| (GWh) — {ZONE_LABELS.get(zone, zone)}",
            gridcolor="rgba(255,255,255,0.06)",
            tickfont=dict(size=10, color="#8b949e"),
        ),
        yaxis=dict(
            title="|NO2 day-over-day price change| (EUR/MWh)",
            gridcolor="rgba(255,255,255,0.06)",
            tickfont=dict(size=10, color="#8b949e"),
        ),
        showlegend=False,
    )
    return fig
