"""
Day-ahead spot price chart: NO1, NO2, SE3 versus DE-LU, NL, FI.
Includes 7-day rolling Nordic and Continental average lines.
Data source: Nord Pool Data Portal (no API key required).
"""
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

ZONE_COLORS = {
    "NO1":   "#4caf8f",
    "NO2":   "#2980b9",
    "SE3":   "#7ec8e3",
    "DE-LU": "#d4ac3a",
    "NL":    "#c0392b",
    "FI":    "#8e6bbf",
}

ZONE_LABELS = {
    "NO1":   "NO1 (Oslo)",
    "NO2":   "NO2 (Kristiansand)",
    "SE3":   "SE3 (Stockholm)",
    "DE-LU": "DE/LU (Germany)",
    "NL":    "NL (Netherlands)",
    "FI":    "FI (Finland)",
}

COLORS = {
    "nordic_avg":      "#3fb950",
    "continental_avg": "#e07b39",
    "grid":      "rgba(255,255,255,0.06)",
    "text":      "#c9d1d9",
    "bg":        "#0d1117",
    "panel_bg":  "#161b22",
}


def render_spot_prices_chart(spot_data: dict):
    df         = spot_data.get("prices", pd.DataFrame())
    fetched_at = spot_data.get("fetched_at")

    st.subheader("Day-Ahead Spot Prices: Nordic and Continental (EUR/MWh)")

    if df.empty:
        st.warning("Spot price data unavailable.")
        if fetched_at:
            st.caption(f"Last updated: {fetched_at.strftime('%Y-%m-%d %H:%M UTC')}")
        st.caption("Source: Nord Pool Data Portal | nordpoolgroup.com")
        return

    # Build wide-format daily averages for rolling calculations
    wide = df.pivot_table(index="date", columns="zone", values="price_eur_mwh").sort_index()
    nordic_cols = [z for z in ["NO1", "NO2"] if z in wide.columns]
    cont_cols   = [z for z in ["DE-LU", "NL"] if z in wide.columns]

    fig = go.Figure()

    # Individual zone lines (thin)
    for zone in ["NO1", "NO2", "SE3", "DE-LU", "NL", "FI"]:
        subset = df[df["zone"] == zone].sort_values("date")
        if subset.empty:
            continue
        label = ZONE_LABELS.get(zone, zone)
        fig.add_trace(go.Scatter(
            x=subset["date"],
            y=subset["price_eur_mwh"],
            line=dict(color=ZONE_COLORS.get(zone, "#888"), width=1.2),
            name=label,
            hovertemplate=f"{label}: €%{{y:.1f}}/MWh<extra></extra>",
        ))

    # 7-day rolling Nordic average (thicker overlay)
    if nordic_cols:
        nordic_avg = wide[nordic_cols].mean(axis=1).rolling(7, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=nordic_avg.index,
            y=nordic_avg.values,
            line=dict(color=COLORS["nordic_avg"], width=2.5, dash="dot"),
            name="Nordic avg (7d MA)",
            hovertemplate="Nordic 7d MA: €%{y:.1f}/MWh<extra></extra>",
        ))

    # 7-day rolling Continental average (thicker overlay)
    # DE-LU is not available from Nord Pool (EPEX licensing), so in practice this is NL only.
    if cont_cols:
        cont_avg = wide[cont_cols].mean(axis=1).rolling(7, min_periods=1).mean()
        cont_label = (
            "NL (7d MA)"
            if cont_cols == ["NL"]
            else "Continental avg (7d MA)"
        )
        fig.add_trace(go.Scatter(
            x=cont_avg.index,
            y=cont_avg.values,
            line=dict(color=COLORS["continental_avg"], width=2.5, dash="dot"),
            name=cont_label,
            hovertemplate=f"{cont_label}: €%{{y:.1f}}/MWh<extra></extra>",
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["panel_bg"],
        font=dict(color=COLORS["text"], size=12),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor=COLORS["grid"], title=None),
        yaxis=dict(title="EUR/MWh", showgrid=True, gridcolor=COLORS["grid"]),
        hovermode="x unified",
    )

    st.plotly_chart(fig, width="stretch")

    # Insight on latest date
    if not df.empty:
        latest_date = df["date"].max()
        latest = df[df["date"] == latest_date].set_index("zone")["price_eur_mwh"]
        n_zones = [z for z in ["NO1", "NO2"] if z in latest.index]
        c_zones = [z for z in ["DE-LU", "NL"] if z in latest.index]
        n_avg = latest[n_zones].mean() if n_zones else None
        c_avg = latest[c_zones].mean() if c_zones else None

        if n_avg is not None and c_avg is not None:
            spread = c_avg - n_avg
            cont_label_ins = "NL" if c_zones == ["NL"] else "Continental"
            if abs(spread) > 5:
                direction = "above" if spread > 0 else "below"
                st.caption(
                    f"Latest ({latest_date}): {cont_label_ins} €{abs(spread):.1f}/MWh {direction} Nordic average. "
                    f"{'Strong export incentive for Norwegian hydro.' if spread > 0 else 'Interconnectors flowing northward.'}"
                )
            else:
                st.caption(
                    f"Latest ({latest_date}): Nordic and {cont_label_ins} prices broadly aligned "
                    f"(spread: €{spread:+.1f}/MWh)."
                )

    if fetched_at:
        st.caption(f"Last updated: {fetched_at.strftime('%Y-%m-%d %H:%M UTC')}")
    st.caption("Source: Nord Pool Data Portal, Day-Ahead Market Prices | nordpoolgroup.com")
