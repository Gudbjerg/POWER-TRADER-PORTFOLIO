"""
TTF gas price chart with 30/90-day moving averages, spike alert, and reference threshold lines.
"""
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

COLORS = {
    "ttf":      "#e07b39",
    "ma30":     "#2a9d8f",
    "ma90":     "#b5a642",
    "grid":     "rgba(255,255,255,0.06)",
    "text":     "#c9d1d9",
    "bg":       "#0d1117",
    "panel_bg": "#161b22",
}

# Reference price thresholds with label and line colour
TTF_THRESHOLDS = [
    (20,  "Pre-crisis avg (~20)",  "rgba(80,160,80,0.35)"),
    (35,  "Elevated (35)",         "rgba(200,160,50,0.35)"),
    (50,  "High risk (50)",        "rgba(200,80,80,0.35)"),
]


def render_ttf_chart(ttf_data: dict, events: list | None = None):
    df         = ttf_data.get("prices", pd.DataFrame())
    spike      = ttf_data.get("spike", False)
    spike_pct  = ttf_data.get("spike_pct", 0.0)
    fetched_at = ttf_data.get("fetched_at")

    st.subheader("TTF Natural Gas Price (EUR/MWh)")

    if df.empty:
        st.warning("TTF price data unavailable. Check yfinance installation or network access.")
        st.caption("Source: ICE TTF front-month futures via Yahoo Finance (ticker: TTF=F)")
        return

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["price"],
        line=dict(color=COLORS["ttf"], width=2),
        name="TTF spot",
        hovertemplate="TTF: €%{y:.2f}/MWh<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["ma30"],
        line=dict(color=COLORS["ma30"], width=1.5, dash="dash"),
        name="30-day MA",
        hovertemplate="30d MA: €%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["ma90"],
        line=dict(color=COLORS["ma90"], width=1.5, dash="dot"),
        name="90-day MA",
        hovertemplate="90d MA: €%{y:.2f}<extra></extra>",
    ))

    # Geopolitical event overlay
    if events:
        from data.events import CATEGORY_COLORS, IMPACT_DASH
        x_min = df["date"].min()
        x_max = df["date"].max()
        for ev in events:
            ev_date = pd.Timestamp(ev["date"])
            if not (x_min <= ev_date <= x_max):
                continue
            color = CATEGORY_COLORS.get(ev.get("category", "policy"), "#888")
            dash  = IMPACT_DASH.get(ev.get("impact", "warn"), "dot")
            fig.add_vline(
                x=ev_date.timestamp() * 1000,
                line_dash=dash,
                line_color=color,
                line_width=1.2,
                annotation_text=ev["label"],
                annotation_position="top left",
                annotation_font=dict(size=9, color=color),
                annotation_textangle=-90,
            )

    # Reference threshold lines
    for level, label, color in TTF_THRESHOLDS:
        fig.add_hline(
            y=level,
            line_dash="dot",
            line_color=color,
            line_width=1,
            annotation_text=label,
            annotation_position="top right",
            annotation_font=dict(size=9, color=color),
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["panel_bg"],
        font=dict(color=COLORS["text"], size=12),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor=COLORS["grid"]),
        yaxis=dict(title="EUR/MWh", showgrid=True, gridcolor=COLORS["grid"]),
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    latest_price = df["price"].iloc[-1] if not df.empty else None
    ma30_latest  = df["ma30"].iloc[-1]  if not df.empty else None

    if spike:
        direction = "up" if spike_pct > 0 else "down"
        st.error(
            f"Price alert: TTF moved {direction} {abs(spike_pct):.1f}% today. "
            f"Current price: €{latest_price:.2f}/MWh."
        )
    elif latest_price is not None and ma30_latest is not None:
        diff_pct = (latest_price - ma30_latest) / ma30_latest * 100
        rel = "above" if diff_pct >= 0 else "below"
        st.caption(
            f"TTF at €{latest_price:.2f}/MWh, {abs(diff_pct):.1f}% {rel} the 30-day average."
        )

    if fetched_at:
        st.caption(f"Last updated: {fetched_at.strftime('%Y-%m-%d %H:%M UTC')}")
    st.caption("Source: ICE TTF Natural Gas front-month futures via Yahoo Finance (ticker: TTF=F)")
