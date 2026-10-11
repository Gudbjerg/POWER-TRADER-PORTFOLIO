"""
LNG terminal sendout chart: NW European terminals stacked by country.
Data source: GIE ALSI (same API key as AGSI).
"""
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from config.settings import GIE_FAILURE_TTL_S, LNG_COUNTRIES
from utils.helpers import has_agsi_key
from utils.theme import CHART, PLOTLY_TEMPLATE

COUNTRY_COLORS = {
    "BE": "#e07b39",
    "NL": "#2a9d8f",
    "GB": "#b5a642",
    "FR": "#4361ee",
}

COLORS = {
    "ma_line":  "#c0392b",
    "grid":     CHART["grid"],
    "text":     CHART["text"],
    "bg":       CHART["bg"],
    "panel_bg": CHART["panel_bg"],
}


def render_lng_chart(lng_data: dict):
    df         = lng_data.get("sendout", pd.DataFrame())
    totals     = lng_data.get("totals",  pd.DataFrame())
    wow_change = lng_data.get("wow_change_pct")
    alert      = lng_data.get("alert", False)
    fetched_at = lng_data.get("fetched_at")
    unavailable = lng_data.get("unavailable") or []
    retry_note = f"The panel retries within {max(GIE_FAILURE_TTL_S // 60, 1)} minutes."

    st.subheader("NW European LNG Terminal Sendout (GWh/day)")

    if df.empty:
        if has_agsi_key():
            st.warning(f"LNG terminal data unavailable: GIE ALSI did not answer. {retry_note}")
        else:
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
            if country not in unavailable:   # failed fetches are reported below
                missing_countries.append(label)
            continue
        fig.add_trace(go.Bar(
            x=subset["gasDayStart"],
            y=subset["sendOut"],
            name=label,
            marker_color=COUNTRY_COLORS.get(country, "#888"),
            hovertemplate=f"{label}: %{{y:,.1f}} GWh/d<extra></extra>",
        ))

    # 7-day MA total overlay
    if not totals.empty and "date" in totals.columns:
        totals_plot = totals[totals["date"] >= cutoff]
        fig.add_trace(go.Scatter(
            x=totals_plot["date"],
            y=totals_plot["ma7"],
            name="7-day MA (total)",
            line=dict(color=COLORS["ma_line"], width=2, dash="dot"),
            hovertemplate="7d MA: %{y:,.1f} GWh/d<extra></extra>",
        ))

    fig.update_layout(
        barmode="stack",
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["panel_bg"],
        font=dict(color=COLORS["text"], size=12),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor=COLORS["grid"], title=None),
        yaxis=dict(title="GWh/day", showgrid=True, gridcolor=COLORS["grid"]),
        hovermode="x unified",
    )

    st.plotly_chart(fig, width="stretch")

    if unavailable:
        names = ", ".join(LNG_COUNTRIES.get(c, c) for c in unavailable)
        st.warning(
            f"GIE ALSI did not answer for {names}. The chart and the NW European total leave "
            f"them out until the next successful fetch. {retry_note}"
        )

    if missing_countries:
        _gb_missing = any("Great Britain" in c or c == "GB" for c in missing_countries)
        if _gb_missing:
            st.info(
                "Great Britain (South Hook / Dragon LNG) data unavailable from GIE ALSI. "
                "South Hook (21 bcm/yr) and Dragon LNG (5 bcm/yr) represent the largest "
                "NW European LNG import cluster. The total shown above understates true sendout "
                "until GB data resumes. Check terminal status at alsi.gie.eu.",
                icon="ℹ️",
            )
        else:
            st.caption(
                f"No sendout data for: {', '.join(missing_countries)}. "
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
