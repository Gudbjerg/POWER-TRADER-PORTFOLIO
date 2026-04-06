"""
Solar cannibalisation (duck curve) chart for Germany.
Average intraday day-ahead price versus average solar generation by hour of day.
"""
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

COLORS = {
    "price":    "#c0392b",
    "solar":    "#d4ac3a",
    "grid":     "rgba(255,255,255,0.06)",
    "text":     "#c9d1d9",
    "bg":       "#0d1117",
    "panel_bg": "#161b22",
}


def render_solar_chart(solar_data: dict):
    df         = solar_data.get("hourly", pd.DataFrame())
    fetched_at = solar_data.get("fetched_at")
    source     = solar_data.get("source")

    st.subheader("Solar Cannibalisation: Germany Intraday Price vs Solar Output (14-day average)")

    if df.empty:
        from utils.helpers import has_entsoe_key
        if has_entsoe_key():
            st.warning(
                "Solar data unavailable. ENTSO-E server is temporarily down and the "
                "energy-charts.info fallback also returned no data. Will retry on next refresh."
            )
        else:
            st.warning("Solar data unavailable. Add ENTSOE_API_KEY to .env to enable this panel.")
        if fetched_at:
            st.caption(f"Last updated: {fetched_at.strftime('%Y-%m-%d %H:%M UTC')}")
        st.caption("Source: ENTSO-E Transparency Platform (primary) | energy-charts.info / Fraunhofer ISE (fallback)")
        return

    if source and source != "ENTSO-E":
        st.info(
            f"Using fallback data source: **{source}**. "
            "ENTSO-E Transparency Platform is currently unavailable. "
            "Data quality is equivalent; source will switch back to ENTSO-E automatically."
        )

    fig = go.Figure()

    # Solar generation on secondary axis (bars)
    fig.add_trace(go.Bar(
        x=df["hour"],
        y=df["solar_mw"],
        name="Solar generation (MW)",
        marker_color=COLORS["solar"],
        opacity=0.45,
        yaxis="y2",
        hovertemplate="Hour %{x}: %{y:,.0f} MW<extra></extra>",
    ))

    # Day-ahead price on primary axis
    fig.add_trace(go.Scatter(
        x=df["hour"],
        y=df["price"],
        line=dict(color=COLORS["price"], width=2.5),
        name="Day-ahead price (EUR/MWh)",
        hovertemplate="Hour %{x}: €%{y:.2f}/MWh<extra></extra>",
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["panel_bg"],
        font=dict(color=COLORS["text"], size=12),
        margin=dict(l=10, r=10, t=10, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            title="Hour of day (CET)",
            tickmode="linear",
            tick0=0, dtick=2,
            showgrid=True,
            gridcolor=COLORS["grid"],
        ),
        yaxis=dict(
            title="EUR/MWh",
            showgrid=True,
            gridcolor=COLORS["grid"],
        ),
        yaxis2=dict(
            title="Solar output (MW)",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Duck curve insight
    if not df.empty and "price" in df.columns and "solar_mw" in df.columns:
        midday_price  = df[df["hour"].between(11, 14)]["price"].mean()
        evening_price = df[df["hour"].between(17, 20)]["price"].mean()
        morning_price = df[df["hour"].between(7, 9)]["price"].mean()
        cannibalisation = morning_price - midday_price if morning_price > 0 else 0

        if cannibalisation > 5:
            st.caption(
                f"Duck curve visible: midday average €{midday_price:.1f}/MWh versus "
                f"morning €{morning_price:.1f}/MWh. Solar generation is suppressing midday prices "
                f"by approximately €{cannibalisation:.1f}/MWh. "
                f"Evening ramp average: €{evening_price:.1f}/MWh."
            )
        else:
            st.caption(
                f"Midday average: €{midday_price:.1f}/MWh. Evening ramp average: €{evening_price:.1f}/MWh."
            )

    if fetched_at:
        st.caption(f"Last updated: {fetched_at.strftime('%Y-%m-%d %H:%M UTC')}")
    if source == "ENTSO-E":
        st.caption("Source: ENTSO-E Transparency Platform, Actual Generation (A75, Solar B16) and Day-Ahead Prices (A44), Germany | transparency.entsoe.eu")
    else:
        st.caption(f"Source: {source or 'unavailable'}: solar generation and DE-LU day-ahead prices | energy-charts.info")
