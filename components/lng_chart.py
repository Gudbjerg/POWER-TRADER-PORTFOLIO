"""
LNG terminal sendout chart: NW European terminals stacked by country.
Data source: GIE ALSI (same API key as AGSI).
"""
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

COUNTRY_COLORS = {
    "BE": "#e07b39",
    "NL": "#2a9d8f",
    "GB": "#b5a642",
    "FR": "#4361ee",
}

COLORS = {
    "ma_line":  "#c0392b",
    "grid":     "rgba(255,255,255,0.06)",
    "text":     "#c9d1d9",
    "bg":       "#0d1117",
    "panel_bg": "#161b22",
}


def render_lng_chart(lng_data: dict):
    df         = lng_data.get("sendout", pd.DataFrame())
    totals     = lng_data.get("totals",  pd.DataFrame())
    wow_change = lng_data.get("wow_change_pct")
    alert      = lng_data.get("alert", False)
    fetched_at = lng_data.get("fetched_at")

    st.subheader("NW European LNG Terminal Sendout (TWh/day)")

    if df.empty:
        st.warning(
            "LNG terminal data unavailable. "
            "Add AGSI_API_KEY to .env (same key as gas storage, free registration at agsi.gie.eu)."
        )
        if fetched_at:
            st.caption(f"Last updated: {fetched_at.strftime('%Y-%m-%d %H:%M UTC')}")
        st.caption("Source: Gas Infrastructure Europe (GIE), ALSI | alsi.gie.eu")
        return

    cutoff  = pd.Timestamp.now().normalize() - pd.Timedelta(days=60)
    df_plot = df[df["gasDayStart"] >= cutoff].copy()

    fig = go.Figure()

    missing_countries = []
    for country, label in [
        ("BE", "Belgium (Zeebrugge)"),
        ("NL", "Netherlands (Gate LNG)"),
        ("GB", "Great Britain"),
        ("FR", "France"),
    ]:
        subset = (
            df_plot[df_plot["country"] == country]
            .sort_values("gasDayStart")
            .dropna(subset=["sendOut"])
        )
        if subset.empty:
            missing_countries.append(label)
            continue
        fig.add_trace(go.Bar(
            x=subset["gasDayStart"],
            y=subset["sendOut"],
            name=label,
            marker_color=COUNTRY_COLORS.get(country, "#888"),
            hovertemplate=f"{label}: %{{y:.2f}} TWh/d<extra></extra>",
        ))

    # 7-day MA total overlay
    if not totals.empty and "date" in totals.columns:
        totals_plot = totals[totals["date"] >= cutoff]
        fig.add_trace(go.Scatter(
            x=totals_plot["date"],
            y=totals_plot["ma7"],
            name="7-day MA (total)",
            line=dict(color=COLORS["ma_line"], width=2, dash="dot"),
            hovertemplate="7d MA: %{y:.2f} TWh/d<extra></extra>",
        ))

    fig.update_layout(
        barmode="stack",
        template="plotly_dark",
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["panel_bg"],
        font=dict(color=COLORS["text"], size=12),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor=COLORS["grid"], title=None),
        yaxis=dict(title="TWh/day", showgrid=True, gridcolor=COLORS["grid"]),
        hovermode="x unified",
    )

    st.plotly_chart(fig, width="stretch")

    if missing_countries:
        st.caption(
            f"No sendout data available for: {', '.join(missing_countries)}. "
            "GIE ALSI may not have reported data for this terminal in the selected window."
        )

    if alert and wow_change is not None:
        st.error(
            f"Sendout alert: NW European LNG sendout fell {abs(wow_change):.1f}% week-on-week. "
            "Potential supply disruption. Monitor terminal status at alsi.gie.eu."
        )
    elif wow_change is not None:
        direction = "up" if wow_change >= 0 else "down"
        note = (
            "Supportive for European gas supply."
            if wow_change >= 0
            else "Watch for further declines; a sustained drop would tighten the gas balance."
        )
        st.caption(f"Sendout {direction} {abs(wow_change):.1f}% versus prior 7 days. {note}")

    if fetched_at:
        st.caption(f"Last updated: {fetched_at.strftime('%Y-%m-%d %H:%M UTC')}")
    st.caption("Source: Gas Infrastructure Europe (GIE), ALSI | alsi.gie.eu")
