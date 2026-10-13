"""
Nordic cross-border power flow chart: net import/export bar chart by corridor.
"""
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from datetime import date, timedelta
from utils.theme import CHART, PLOTLY_TEMPLATE

PAIR_COLORS = {
    "NO1->SE3": "#1c8fb0",
    "NO2->DK1": "#2f45c6",
    "NO2->DE":  "#a0168a",
    "NO2->NL":  "#b8494f",
    "NO2->GB":  "#c77b2e",   # North Sea Link
}

COLORS = {
    "grid":     CHART["grid"],
    "text":     CHART["text"],
    "bg":       CHART["bg"],
    "panel_bg": CHART["panel_bg"],
}


def render_flows_chart(flow_data: dict):
    df         = flow_data.get("flows", pd.DataFrame())
    fetched_at = flow_data.get("fetched_at")

    st.subheader("Norwegian Cross-Border Power Flows (net GWh per day)")

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
        color = PAIR_COLORS.get(pair, "#667b83")
        fig.add_trace(go.Bar(
            x=subset["date"],
            y=subset["net_flow_mwh"] / 1_000,
            name=pair.replace("->", " → "),
            marker_color=color,
            hovertemplate=f"{pair.replace('->', ' → ')}: %{{y:,.1f}} GWh<extra></extra>",
        ))

    fig.update_layout(
        barmode="group",
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["panel_bg"],
        font=dict(color=COLORS["text"], size=12),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor=COLORS["grid"], title=None),
        yaxis=dict(
            title="Net GWh/day (positive = import into Norway)",
            showgrid=True,
            gridcolor=COLORS["grid"],
        ),
        hovermode="x unified",
    )

    st.plotly_chart(fig, width="stretch")

    # North Sea Link balance over the fetched window
    nsl = df[df["pair"] == "NO2->GB"].dropna(subset=["net_flow_mwh"])
    if not nsl.empty:
        export_days = int((nsl["net_flow_mwh"] < 0).sum())   # negative = Norway exporting
        import_days = int((nsl["net_flow_mwh"] > 0).sum())
        n_days = len(nsl)
        gross_exp = float(nsl["export_mwh"].sum()) / 1_000 if "export_mwh" in nsl else float("nan")
        gross_imp = float(nsl["import_mwh"].sum()) / 1_000 if "import_mwh" in nsl else float("nan")
        st.info(
            f"North Sea Link (NO2 to GB), last {n_days} days: net export on {export_days} days, "
            f"net import on {import_days} days. Gross flow {gross_exp:,.0f} GWh to Britain and "
            f"{gross_imp:,.0f} GWh to Norway."
        )

    # 7-day dominant corridor insight
    if not df.empty:
        recent = df[df["date"] >= date.today() - timedelta(days=7)]
        if not recent.empty:
            avg_net  = recent.groupby("pair")["net_flow_mwh"].mean()
            dominant = avg_net.abs().idxmax()
            val      = avg_net[dominant]
            direction = "net importing" if val > 0 else "net exporting"
            st.caption(
                f"Last 7 days: the {dominant.replace('->', ' → ')} corridor is {direction} "
                f"({abs(val) / 1_000:,.1f} GWh/day on average)."
            )

    if fetched_at:
        st.caption(f"Last updated: {fetched_at.strftime('%Y-%m-%d %H:%M UTC')}")
    st.caption("Source: ENTSO-E Transparency Platform, Physical Cross-Border Flows (doc type A11) | transparency.entsoe.eu")
