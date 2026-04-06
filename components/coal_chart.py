"""
German coal generation chart: hard coal and lignite stacked quarterly bars.
Replicates and automates the fuel switching indicator posted by Marius Slette.
"""
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

COLORS = {
    "hard_coal": "#2c2c3e",   # near-black (sort kull)
    "lignite":   "#c0754a",   # amber-brown (brunkull)
    "grid":      "rgba(255,255,255,0.06)",
    "text":      "#c9d1d9",
    "bg":        "#0d1117",
    "panel_bg":  "#161b22",
    "ttf_line":  "#e8c547",
}


def render_coal_chart(gen_data: dict, ttf_price: float | None = None):
    quarterly  = gen_data.get("quarterly",  pd.DataFrame())
    daily_90d  = gen_data.get("daily_90d",  pd.DataFrame())
    fetched_at = gen_data.get("fetched_at")
    source     = gen_data.get("source")

    st.subheader("German Coal Generation: Fuel Switching Indicator")

    if quarterly.empty:
        from utils.helpers import has_entsoe_key
        if has_entsoe_key():
            st.warning(
                "German generation data unavailable. ENTSO-E server may be temporarily "
                "unavailable. Data will load automatically when the server recovers."
            )
        else:
            st.warning("German generation data unavailable. Add ENTSOE_API_KEY to .env to enable this panel.")
        if fetched_at:
            st.caption(f"Last updated: {fetched_at.strftime('%Y-%m-%d %H:%M UTC')}")
        st.caption("Source: ENTSO-E Transparency Platform, Actual Generation Per Production Type (A75) | transparency.entsoe.eu")
        return

    # ── Quarterly stacked bar chart ──────────────────────────────────────────
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=quarterly["quarter"],
        y=quarterly["lignite_twh"],
        name="Lignite (brunkull)",
        marker_color=COLORS["lignite"],
        hovertemplate="%{x|%Y-%m}<br>Lignite: %{y:.1f} TWh<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        x=quarterly["quarter"],
        y=quarterly["hard_coal_twh"],
        name="Hard coal (sort kull)",
        marker_color=COLORS["hard_coal"],
        hovertemplate="%{x|%Y-%m}<br>Hard coal: %{y:.1f} TWh<extra></extra>",
    ))

    fig.update_layout(
        barmode="stack",
        template="plotly_dark",
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["panel_bg"],
        font=dict(color=COLORS["text"], size=12),
        margin=dict(l=10, r=10, t=10, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title=None, showgrid=True, gridcolor=COLORS["grid"]),
        yaxis=dict(title="TWh per quarter", showgrid=True, gridcolor=COLORS["grid"]),
        hovermode="x unified",
    )

    st.plotly_chart(fig, width="stretch")

    # ── Fuel switching insight ───────────────────────────────────────────────
    if not quarterly.empty and len(quarterly) >= 4:
        # Compare most recent quarter vs same quarter prior year
        latest_q = quarterly.iloc[-1]
        prior_q  = quarterly.iloc[-5] if len(quarterly) >= 5 else None
        latest_total = float(latest_q["hard_coal_twh"]) + float(latest_q["lignite_twh"])

        if prior_q is not None:
            prior_total = float(prior_q["hard_coal_twh"]) + float(prior_q["lignite_twh"])
            change_pct  = (latest_total - prior_total) / prior_total * 100 if prior_total > 0 else 0
            direction   = "up" if change_pct > 5 else ("down" if change_pct < -5 else "broadly stable")
            st.caption(
                f"Most recent quarter: {latest_total:.1f} TWh total coal generation "
                f"({latest_q['hard_coal_twh']:.1f} TWh hard coal, "
                f"{latest_q['lignite_twh']:.1f} TWh lignite). "
                f"Year-on-year: {direction} ({change_pct:+.0f}%). "
                f"Coal re-enters the German merit order when TTF exceeds approximately "
                f"EUR 50/MWh, displacing gas-fired generation."
            )
        else:
            st.caption(
                f"Most recent quarter: {latest_total:.1f} TWh total coal generation. "
                f"Coal acts as a capacity reserve (superreserve) when gas prices are elevated."
            )

    if ttf_price is not None:
        threshold = 50.0
        if ttf_price > threshold:
            st.caption(
                f"TTF is currently at EUR {ttf_price:.1f}/MWh, above the approximate "
                f"EUR {threshold:.0f}/MWh fuel-switching threshold. "
                "Hard coal and lignite are economically competitive with gas-fired generation."
            )
        else:
            st.caption(
                f"TTF is currently at EUR {ttf_price:.1f}/MWh, below the approximate "
                f"EUR {threshold:.0f}/MWh fuel-switching threshold. "
                "Gas-fired generation is competitive; coal dispatch is reduced."
            )

    # ── 90-day daily chart ───────────────────────────────────────────────────
    if not daily_90d.empty:
        st.markdown("**Recent 90-day daily coal dispatch**")
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=daily_90d["date"],
            y=daily_90d["lignite_gwh"],
            name="Lignite (GWh/day)",
            marker_color=COLORS["lignite"],
            hovertemplate="Lignite: %{y:.0f} GWh<extra></extra>",
        ))
        fig2.add_trace(go.Bar(
            x=daily_90d["date"],
            y=daily_90d["hard_coal_gwh"],
            name="Hard coal (GWh/day)",
            marker_color=COLORS["hard_coal"],
            hovertemplate="Hard coal: %{y:.0f} GWh<extra></extra>",
        ))
        fig2.update_layout(
            barmode="stack",
            template="plotly_dark",
            paper_bgcolor=COLORS["bg"],
            plot_bgcolor=COLORS["panel_bg"],
            font=dict(color=COLORS["text"], size=12),
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            xaxis=dict(showgrid=True, gridcolor=COLORS["grid"]),
            yaxis=dict(title="GWh/day", showgrid=True, gridcolor=COLORS["grid"]),
            hovermode="x unified",
        )
        st.plotly_chart(fig2, width="stretch")

    if fetched_at:
        st.caption(f"Last updated: {fetched_at.strftime('%Y-%m-%d %H:%M UTC')}")
    st.caption(
        "Source: ENTSO-E Transparency Platform, Actual Generation Per Production Type "
        "(A75, Germany), hard coal (B05) and lignite (B04) | transparency.entsoe.eu"
    )
