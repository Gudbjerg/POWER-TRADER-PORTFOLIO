"""
Norwegian hydro reservoir filling level chart.
Current year vs historical 10th/50th/90th percentile bands.
ENTSO-E B31 data is returned in MWh; displayed here in TWh.
"""
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from datetime import datetime

COLORS = {
    "current":  "#4caf8f",
    "p50":      "#8b949e",
    "band":     "rgba(139,148,158,0.15)",
    "band_mid": "rgba(139,148,158,0.25)",
    "grid":     "rgba(255,255,255,0.06)",
    "text":     "#c9d1d9",
    "bg":       "#0d1117",
    "panel_bg": "#161b22",
}


def render_hydro_chart(hydro_data: dict):
    df          = hydro_data.get("weekly", pd.DataFrame())
    percentiles = hydro_data.get("percentiles", pd.DataFrame())
    fetched_at  = hydro_data.get("fetched_at")

    st.subheader("Norwegian Hydro Reservoir Level (TWh)")

    if df.empty:
        from utils.helpers import has_entsoe_key
        if has_entsoe_key():
            st.warning(
                "Hydro reservoir data unavailable. ENTSO-E server may be temporarily unavailable. "
                "Data will load automatically when the server recovers."
            )
        else:
            st.warning("Hydro reservoir data unavailable. Add ENTSOE_API_KEY to .env.")
        if fetched_at:
            st.caption(f"Last updated: {fetched_at.strftime('%Y-%m-%d %H:%M UTC')}")
        st.caption("Source: ENTSO-E Transparency Platform, Hydro Reservoirs and Water Storage (B31) | transparency.entsoe.eu")
        return

    current_year = datetime.utcnow().year
    current = df[df["year"] == current_year].copy()
    if current.empty:
        current = df[df["year"] == df["year"].max()].copy()

    fig = go.Figure()

    # Percentile bands
    if not percentiles.empty and not current.empty:
        week_map = current.set_index("week_of_year")["week_start"].to_dict()
        band_df  = percentiles[percentiles["week_of_year"].isin(week_map)].copy()
        band_df["date"] = band_df["week_of_year"].map(week_map)
        band_df = band_df.sort_values("date")

        fig.add_trace(go.Scatter(
            x=list(band_df["date"]) + list(band_df["date"])[::-1],
            y=list(band_df["p90"]) + list(band_df["p10"])[::-1],
            fill="toself",
            fillcolor=COLORS["band"],
            line=dict(color="rgba(0,0,0,0)", width=0),
            name="Historical P10-P90",
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=band_df["date"],
            y=band_df["p50"],
            line=dict(color=COLORS["p50"], width=1.5, dash="dot"),
            name="Historical median (P50)",
            hovertemplate="P50: %{y:.1f} TWh<extra></extra>",
        ))

    # Current year line
    fig.add_trace(go.Scatter(
        x=current["week_start"],
        y=current["filling_twh"],
        line=dict(color=COLORS["current"], width=2.2),
        name=f"{current_year} (current)",
        hovertemplate="%{x|%b %d}: %{y:.1f} TWh<extra></extra>",
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["panel_bg"],
        font=dict(color=COLORS["text"], size=12),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor=COLORS["grid"], title=None),
        yaxis=dict(title="Reservoir content (TWh)", showgrid=True, gridcolor=COLORS["grid"],
                   rangemode="tozero"),
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Insight text — all comparisons in TWh vs percentile bands
    if not current.empty:
        latest_twh  = float(current["filling_twh"].iloc[-1])
        latest_date = current["week_start"].iloc[-1]

        if not percentiles.empty:
            latest_week = int(current["week_of_year"].iloc[-1])
            row = percentiles[percentiles["week_of_year"] == latest_week]
            if not row.empty:
                p10 = float(row["p10"].values[0])
                p50 = float(row["p50"].values[0])
                p90 = float(row["p90"].values[0])
                diff = latest_twh - p50
                rel  = "above" if diff >= 0 else "below"

                if latest_twh < p10:
                    st.error(
                        f"Hydro alert: Norwegian reservoirs at {latest_twh:.1f} TWh "
                        f"({latest_date.strftime('%b %d')}), below the historical 10th percentile "
                        f"({p10:.1f} TWh). Strongly bullish for NO1/NO2 prices; "
                        "Norway likely importing power from the Continent."
                    )
                elif latest_twh < p50:
                    st.warning(
                        f"Reservoirs at {latest_twh:.1f} TWh ({latest_date.strftime('%b %d')}), "
                        f"{abs(diff):.1f} TWh below the historical median ({p50:.1f} TWh). "
                        "Below-average hydro supports higher Nordic prices."
                    )
                else:
                    st.caption(
                        f"Reservoirs at {latest_twh:.1f} TWh ({latest_date.strftime('%b %d')}), "
                        f"{abs(diff):.1f} TWh {rel} the historical median ({p50:.1f} TWh). "
                        f"{'Supportive for Norwegian exports.' if diff >= 0 else 'Monitor for price implications.'}"
                    )
        else:
            st.caption(
                f"Reservoir content: {latest_twh:.1f} TWh as of {latest_date.strftime('%b %d, %Y')}."
            )

    if fetched_at:
        st.caption(f"Last updated: {fetched_at.strftime('%Y-%m-%d %H:%M UTC')}")
    st.caption(
        "Source: ENTSO-E Transparency Platform, Hydro Reservoirs and Water Storage (B31), "
        "Norway aggregate (EIC: 10YNO-0--------C) | transparency.entsoe.eu"
    )
