"""
Nordic cross-border power flow chart: net import/export bar chart by corridor.
"""
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from datetime import date, timedelta

PAIR_COLORS = {
    "NO1->SE3": "#4cc9f0",
    "NO2->SE3": "#4361ee",
    "NO2->DK1": "#3a0ca3",
    "NO2->DE":  "#7209b7",
    "NO2->NL":  "#c1666b",
    "NO1->GB":  "#2a9d8f",
}

COLORS = {
    "grid":     "rgba(255,255,255,0.06)",
    "text":     "#c9d1d9",
    "bg":       "#0d1117",
    "panel_bg": "#161b22",
}


def render_flows_chart(flow_data: dict):
    df         = flow_data.get("flows", pd.DataFrame())
    fetched_at = flow_data.get("fetched_at")

    st.subheader("Nordic Cross-Border Power Flows (Net MWh/day)")

    if df.empty:
        from utils.helpers import has_entsoe_key
        if has_entsoe_key():
            st.warning("Power flow data unavailable. ENTSO-E server may be temporarily unavailable. Data will load automatically when the server recovers.")
        else:
            st.warning("Power flow data unavailable. Add ENTSOE_API_KEY to .env to enable this panel.")
        if fetched_at:
            st.caption(f"Last updated: {fetched_at.strftime('%Y-%m-%d %H:%M UTC')}")
        st.caption("Source: ENTSO-E Transparency Platform, Physical Cross-Border Flows (A11) | transparency.entsoe.eu")
        return

    # Normalise arrow characters in pair labels to plain text
    df = df.copy()
    df["pair"] = df["pair"].str.replace("→", "->", regex=False)

    fig = go.Figure()

    for pair in df["pair"].unique():
        subset = df[df["pair"] == pair].sort_values("date")
        color = PAIR_COLORS.get(pair, "#888888")
        fig.add_trace(go.Bar(
            x=subset["date"],
            y=subset["net_flow_mwh"],
            name=pair,
            marker_color=color,
            hovertemplate=f"{pair}: %{{y:,.0f}} MWh<extra></extra>",
        ))

    fig.update_layout(
        barmode="group",
        template="plotly_dark",
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["panel_bg"],
        font=dict(color=COLORS["text"], size=12),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor=COLORS["grid"], title=None),
        yaxis=dict(
            title="Net MWh/day (positive = import into Norway, negative = export)",
            showgrid=True,
            gridcolor=COLORS["grid"],
        ),
        hovermode="x unified",
    )

    st.plotly_chart(fig, width="stretch")

    # 7-day dominant corridor insight
    if not df.empty:
        recent = df[df["date"] >= date.today() - timedelta(days=7)]
        if not recent.empty:
            avg_net  = recent.groupby("pair")["net_flow_mwh"].mean()
            dominant = avg_net.abs().idxmax()
            val      = avg_net[dominant]
            direction = "net importing" if val > 0 else "net exporting"
            st.caption(
                f"Last 7 days: the {dominant} corridor is {direction} "
                f"({abs(val):,.0f} MWh/day average)."
            )

    if fetched_at:
        st.caption(f"Last updated: {fetched_at.strftime('%Y-%m-%d %H:%M UTC')}")
    st.caption("Source: ENTSO-E Transparency Platform, Physical Cross-Border Flows (doc type A11) | transparency.entsoe.eu")
