"""
TTF natural gas seasonal forward curve chart.
18-month implied strip: winter bars in blue, summer in teal.
P25-P75 historical range shown as error bars.
"""
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

COLORS = {
    "winter":   "#4c8cbf",
    "summer":   "#4caf8f",
    "range":    "rgba(139,148,158,0.35)",
    "spot":     "#e07b39",
    "grid":     "rgba(255,255,255,0.06)",
    "text":     "#c9d1d9",
    "bg":       "#0d1117",
    "panel_bg": "#161b22",
}

TTF_THRESHOLDS = [
    (20,  "Pre-crisis avg (~20)",  "rgba(80,160,80,0.75)"),
    (35,  "Elevated (35)",         "rgba(200,160,50,0.75)"),
    (50,  "High risk (50)",        "rgba(200,80,80,0.75)"),
]


def render_forward_curve_chart(fwd_data: dict):
    curve      = fwd_data.get("curve", pd.DataFrame())
    summary    = fwd_data.get("summary", {})
    spot       = fwd_data.get("spot", 0.0)
    data_years = fwd_data.get("data_years", 0)
    fetched_at = fwd_data.get("fetched_at")

    st.subheader("TTF Natural Gas: Seasonal Forward Curve")

    if curve.empty:
        st.warning("Forward curve unavailable. TTF price data required.")
        return

    labels = curve["label"].tolist()
    fig    = go.Figure()

    # Reference thresholds
    for level, label, color in TTF_THRESHOLDS:
        fig.add_hline(
            y=level, line_dash="dash", line_color=color, line_width=1.5,
            annotation_text=label,
            annotation_position="top right",
            annotation_font=dict(size=10, color=color),
        )

    # Winter bars
    w = curve[curve["season"] == "Winter"]
    fig.add_trace(go.Bar(
        x=w["label"], y=w["price"],
        name="Winter (Oct-Mar)",
        marker_color=COLORS["winter"],
        marker_opacity=0.82,
        error_y=dict(
            type="data", symmetric=False,
            array=(w["p75"] - w["price"]).clip(lower=0).tolist(),
            arrayminus=(w["price"] - w["p25"]).clip(lower=0).tolist(),
            color=COLORS["range"], thickness=1.5, width=4,
        ),
        hovertemplate="%{x}: €%{y:.1f}/MWh<br>3yr range: €%{customdata[0]:.0f}-€%{customdata[1]:.0f}<extra></extra>",
        customdata=list(zip(w["p25"], w["p75"])),
    ))

    # Summer bars
    s = curve[curve["season"] == "Summer"]
    fig.add_trace(go.Bar(
        x=s["label"], y=s["price"],
        name="Summer (Apr-Sep)",
        marker_color=COLORS["summer"],
        marker_opacity=0.82,
        error_y=dict(
            type="data", symmetric=False,
            array=(s["p75"] - s["price"]).clip(lower=0).tolist(),
            arrayminus=(s["price"] - s["p25"]).clip(lower=0).tolist(),
            color=COLORS["range"], thickness=1.5, width=4,
        ),
        hovertemplate="%{x}: €%{y:.1f}/MWh<br>3yr range: €%{customdata[0]:.0f}-€%{customdata[1]:.0f}<extra></extra>",
        customdata=list(zip(s["p25"], s["p75"])),
    ))

    # Current spot line
    if spot > 0:
        fig.add_hline(
            y=spot,
            line_dash="dashdot",
            line_color=COLORS["spot"],
            line_width=1.8,
            annotation_text=f"  Spot: €{spot:.1f}/MWh",
            annotation_position="top left",
            annotation_font=dict(color=COLORS["spot"], size=10),
        )

    # Winter premium annotation box
    wp = summary.get("winter_premium")
    if wp is not None:
        sign  = "+" if wp >= 0 else ""
        acolor = COLORS["winter"] if wp >= 0 else COLORS["summer"]
        fig.add_annotation(
            xref="paper", yref="paper", x=0.99, y=0.97,
            text=f"Winter premium: {sign}€{wp:.1f}/MWh",
            showarrow=False,
            font=dict(size=11, color=acolor),
            bgcolor=COLORS["panel_bg"],
            bordercolor=acolor,
            borderwidth=1,
            borderpad=6,
            align="right",
            xanchor="right",
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["panel_bg"],
        font=dict(color=COLORS["text"], size=12),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            showgrid=False, title=None,
            categoryorder="array", categoryarray=labels,
        ),
        yaxis=dict(
            title="EUR/MWh",
            showgrid=True, gridcolor=COLORS["grid"],
            rangemode="tozero",
        ),
        bargap=0.2,
        hovermode="x unified",
    )

    st.plotly_chart(fig, width="stretch")

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        if spot > 0:
            st.metric("TTF Spot", f"€{spot:.1f}/MWh")
    with k2:
        if summary.get("summer_avg") is not None:
            st.metric("Summer avg (fwd)", f"€{summary['summer_avg']:.1f}/MWh")
    with k3:
        if summary.get("winter_avg") is not None:
            st.metric("Winter avg (fwd)", f"€{summary['winter_avg']:.1f}/MWh")
    with k4:
        if summary.get("cal2026_avg") is not None:
            st.metric("Cal 26 avg (fwd)", f"€{summary['cal2026_avg']:.1f}/MWh")

    cal27 = summary.get("cal2027_avg")
    if cal27 is not None:
        st.caption(f"Cal 27 average (implied): €{cal27:.1f}/MWh")

    note = (
        f"Seasonally-implied forward strip anchored to today's TTF spot, "
        f"calibrated from {data_years} years of historical seasonal patterns. "
        "Error bars show the 3-year historical P25-P75 range per calendar month "
        "(includes 2022-23 crisis period; bars may appear asymmetric or one-sided "
        "when current prices are below the historical range). "
        "Not a substitute for live OTC forward quotes or exchange settlement prices."
    )
    st.caption(note)
    if fetched_at:
        st.caption(f"Last updated: {fetched_at.strftime('%Y-%m-%d %H:%M UTC')}")
    st.caption("Source: ICE TTF Natural Gas front-month futures history via Yahoo Finance (TTF=F)")
