"""
Layer 3: Geopolitical and Macro Signals
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Macro Signals",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from utils.helpers import apply_dark_theme, kpi_card, delta_span, commentary
from data.prices import get_ttf_data
from data.events import load_events, events_in_range, CATEGORY_COLORS
from data.commodities import get_commodity_data
from components.prices_chart import render_ttf_chart

apply_dark_theme()

# ── Load data ────────────────────────────────────────────────────────────────
with st.spinner(""):
    ttf         = get_ttf_data()
    commodities = get_commodity_data()

events_all = load_events()

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("## Layer 3: Geopolitical and Macro Signals")
st.caption(
    "Geopolitical event context, EU gas supply mix, and cross-commodity price transmission. "
    "Data sources: ICE/Yahoo Finance, manually curated event timeline."
)
st.divider()

tab_geo, tab_mix, tab_spill = st.tabs([
    "Geopolitical Overlay",
    "EU Gas Supply Mix",
    "Cross-Commodity Spillover",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1: GEOPOLITICAL EVENT OVERLAY
# ════════════════════════════════════════════════════════════════════════════
with tab_geo:
    st.markdown("### TTF Gas Price with Geopolitical Event Timeline")
    st.caption(
        "Vertical lines mark key supply disruptions, geopolitical escalations, and policy shifts. "
        "Red = geopolitical, amber = supply disruption, blue = policy, green = weather."
    )

    ttf_df = ttf["prices"]
    if not ttf_df.empty:
        chart_events = events_in_range(ttf_df["date"].min(), ttf_df["date"].max())
    else:
        chart_events = []

    render_ttf_chart(ttf, events=chart_events)

    st.markdown("#### Full Event Timeline")
    cat_filter = st.multiselect(
        "Filter by category",
        options=["geopolitical", "supply", "policy", "weather"],
        default=["geopolitical", "supply", "policy", "weather"],
    )

    filtered_events = [ev for ev in events_all if ev.get("category") in cat_filter]
    filtered_events = sorted(filtered_events, key=lambda x: x["date"], reverse=True)

    for ev in filtered_events:
        impact = ev.get("impact", "warn")
        cat    = ev.get("category", "policy")
        color  = CATEGORY_COLORS.get(cat, "#888")
        date_s = ev["date"].strftime("%Y-%m-%d")
        border_cls = {"critical": "commentary-critical", "warn": "commentary-warn"}.get(impact, "")
        st.markdown(
            f'<div class="commentary {border_cls}" style="border-left-color:{color};">'
            f'<strong>{date_s}: {ev["label"]}</strong><br>'
            f'<span style="color:#8b949e;font-size:0.78rem;text-transform:uppercase;'
            f'letter-spacing:.5px;">{cat}</span><br>'
            f'{ev["description"]}</div>',
            unsafe_allow_html=True,
        )

    with st.expander("About this timeline", expanded=False):
        st.markdown("""
        Events are manually curated from public sources including ENTSO-E announcements,
        GIE supply reports, Reuters Energy, and Montel News. They cover four categories:

        - **Geopolitical (red):** Military conflicts, diplomatic escalations, sanctions affecting gas supply routes
        - **Supply (amber):** Pipeline outages, LNG terminal disruptions, scheduled maintenance events
        - **Policy (blue):** EU regulatory changes, storage mandates, carbon market developments
        - **Weather (green):** Extreme temperature events affecting storage drawdown or injection pace

        Vertical lines on the TTF chart are filtered to the visible date range. All events are
        documented in `data/events.json` and can be extended as new events occur.
        """)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2: EU GAS SUPPLY MIX
# ════════════════════════════════════════════════════════════════════════════
with tab_mix:
    st.markdown("### EU Gas Supply Mix (2020-2026)")
    st.caption(
        "Approximate EU gas import supply shares by origin. "
        "Based on GIE, Eurostat, and IEA data. "
        "Q1-Q2 2026 estimate: Norway ~27%, USA ~29%, Algeria ~12%, Russia ~3%, Qatar ~8%, other ~21%."
    )

    _mix = {
        "year":    [2020,  2021,  2022,  2023,  2024,  2025,  2026],
        "Norway":  [28.0,  27.0,  33.0,  34.0,  31.0,  29.0,  27.0],
        "USA":     [5.0,   7.0,   15.0,  19.0,  22.0,  27.0,  29.0],
        "Russia":  [40.0,  40.0,  20.0,  8.0,   4.0,   3.0,   3.0],
        "Algeria": [8.0,   9.0,   10.0,  11.0,  12.0,  13.0,  12.0],
        "Qatar":   [5.0,   5.0,   6.0,   7.0,   8.0,   8.0,   8.0],
        "Other":   [14.0,  12.0,  16.0,  21.0,  23.0,  20.0,  21.0],
    }
    mix_df = pd.DataFrame(_mix)

    MIX_COLORS = {
        "Norway":  "#4caf8f",
        "USA":     "#4361ee",
        "Russia":  "#f85149",
        "Algeria": "#d4ac3a",
        "Qatar":   "#8e6bbf",
        "Other":   "#8b949e",
    }
    C = {
        "bg": "#0d1117", "panel_bg": "#161b22",
        "grid": "rgba(255,255,255,0.06)", "text": "#c9d1d9",
    }

    fig_mix = go.Figure()
    for supplier in ["Norway", "USA", "Algeria", "Qatar", "Other", "Russia"]:
        fig_mix.add_trace(go.Scatter(
            x=mix_df["year"],
            y=mix_df[supplier],
            name=supplier,
            stackgroup="one",
            mode="lines",
            line=dict(width=0.5, color=MIX_COLORS[supplier]),
            fillcolor=MIX_COLORS[supplier],
            opacity=0.85,
            hovertemplate=f"{supplier}: %{{y:.1f}}%<extra></extra>",
        ))

    fig_mix.add_vline(
        x=2022, line_dash="dash", line_color="rgba(10,10,10,0.85)", line_width=1.5,
        annotation_text="Russia invades Ukraine", annotation_position="top left",
        annotation_font=dict(size=9, color="#0d1117"), annotation_textangle=-90,
    )
    fig_mix.update_layout(
        template="plotly_dark",
        paper_bgcolor=C["bg"], plot_bgcolor=C["panel_bg"],
        font=dict(color=C["text"], size=12),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor=C["grid"], title=None,
                   tickmode="array", tickvals=list(range(2020, 2027))),
        yaxis=dict(title="Share of EU gas imports (%)", showgrid=True,
                   gridcolor=C["grid"], range=[0, 105]),
        hovermode="x unified",
    )
    st.plotly_chart(fig_mix, width="stretch")

    st.markdown(
        commentary(
            "Russia's share of EU gas imports fell from approximately 40% in 2021 to 3% in Q1 2026, "
            "a structural shift driven by the 2022 invasion embargo and pipeline abandonment. "
            "US LNG replaced the majority of this volume, rising from 7% to 29%. "
            "This concentration in US LNG creates a new geopolitical risk vector: US tariff policy "
            "and Hormuz Strait transit risk are now material factors for European gas supply security.",
            "warn",
        ),
        unsafe_allow_html=True,
    )

    with st.expander("Data notes", expanded=False):
        st.markdown("""
        Supply shares are approximate annual estimates based on GIE ALSI terminal flow data,
        Eurostat energy import statistics, and IEA Gas Market Report publications.
        The 2026 figure is a Q1 estimate. Quarterly figures vary around annual averages.
        Russia includes residual TurkStream flows to Southern Europe.
        "Other" includes Azerbaijan (Southern Gas Corridor), Libya, Nigeria LNG, and spot cargoes.
        """)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3: CROSS-COMMODITY SPILLOVER
# ════════════════════════════════════════════════════════════════════════════
with tab_spill:
    st.markdown("### Cross-Commodity Spillover: Gas to Agriculture")
    st.caption(
        "Gas price transmission chain: TTF gas drives European nitrogen fertiliser production costs, "
        "which flow through to global wheat and corn prices. "
        "All series normalised to 100 at the earliest common date."
    )

    comm_df = commodities["prices"]
    fetched = commodities["fetched_at"]

    if comm_df.empty:
        st.warning(
            "Commodity price data unavailable. Requires yfinance and internet access. "
            "Tickers: TTF=F (gas), ZW=F (wheat), ZC=F (corn)."
        )
    else:
        COMM_COLORS = {"TTF": "#e07b39", "Wheat": "#d4ac3a", "Corn": "#3fb950"}
        C = {
            "bg": "#0d1117", "panel_bg": "#161b22",
            "grid": "rgba(255,255,255,0.06)", "text": "#c9d1d9",
        }

        # Normalised price index chart
        fig_comm = go.Figure()
        for comm in ["TTF", "Wheat", "Corn"]:
            sub = comm_df[comm_df["commodity"] == comm].sort_values("date")
            if sub.empty:
                continue
            fig_comm.add_trace(go.Scatter(
                x=sub["date"], y=sub["price_norm"],
                name=comm,
                line=dict(color=COMM_COLORS.get(comm, "#888"), width=2),
                hovertemplate=f"{comm}: %{{y:.1f}}<extra></extra>",
            ))

        fig_comm.update_layout(
            title=dict(text="Normalised Price Index (base = 100 at start date)", font=dict(size=12)),
            template="plotly_dark",
            paper_bgcolor=C["bg"], plot_bgcolor=C["panel_bg"],
            font=dict(color=C["text"], size=12),
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(showgrid=True, gridcolor=C["grid"], title=None),
            yaxis=dict(title="Index (100 = start date)", showgrid=True, gridcolor=C["grid"]),
            hovermode="x unified",
        )
        st.plotly_chart(fig_comm, width="stretch")

        # Rolling correlations
        st.markdown("#### Rolling 90-Day Pearson Correlations")
        wide = comm_df.pivot_table(index="date", columns="commodity", values="price_norm")
        wide = wide.sort_index().dropna()

        if len(wide) >= 90:
            pairs = [("TTF", "Wheat"), ("TTF", "Corn"), ("Wheat", "Corn")]
            pair_colors = {"TTF-Wheat": "#d4ac3a", "TTF-Corn": "#3fb950", "Wheat-Corn": "#8e6bbf"}

            fig_corr = go.Figure()
            for a, b in pairs:
                if a not in wide.columns or b not in wide.columns:
                    continue
                roll_corr = wide[a].rolling(90).corr(wide[b])
                label = f"{a}-{b}"
                fig_corr.add_trace(go.Scatter(
                    x=wide.index, y=roll_corr,
                    name=label,
                    line=dict(color=pair_colors.get(label, "#888"), width=1.8),
                    hovertemplate=f"{label}: %{{y:.2f}}<extra></extra>",
                ))
            fig_corr.add_hline(y=0, line_dash="solid",
                               line_color="rgba(255,255,255,0.2)", line_width=0.8)
            fig_corr.add_hline(y=0.5, line_dash="dot",
                               line_color="rgba(255,255,255,0.15)", line_width=0.8,
                               annotation_text="0.5", annotation_font=dict(size=9))
            fig_corr.update_layout(
                template="plotly_dark",
                paper_bgcolor=C["bg"], plot_bgcolor=C["panel_bg"],
                font=dict(color=C["text"], size=12),
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(showgrid=True, gridcolor=C["grid"], title=None),
                yaxis=dict(title="Pearson correlation", showgrid=True,
                           gridcolor=C["grid"], range=[-1, 1]),
                hovermode="x unified",
                height=250,
            )
            st.plotly_chart(fig_corr, width="stretch")

            # Latest KPIs
            c1, c2, c3 = st.columns(3)
            for col_st, (a, b) in zip([c1, c2, c3], pairs):
                if a in wide.columns and b in wide.columns:
                    val = float(wide[a].rolling(90).corr(wide[b]).iloc[-1])
                    color = "green" if abs(val) >= 0.5 else ("amber" if abs(val) >= 0.3 else "blue")
                    with col_st:
                        st.markdown(
                            kpi_card(f"Corr: {a}-{b}", f"{val:.2f}",
                                     delta_span("90-day rolling Pearson", color)),
                            unsafe_allow_html=True,
                        )
        else:
            st.caption("Insufficient history for rolling correlations (requires at least 90 overlapping days).")

        with st.expander("Transmission mechanism", expanded=True):
            st.markdown("""
            **Gas to fertiliser:** Nitrogen fertilisers (ammonium nitrate, urea) are produced via the
            Haber-Bosch process, which consumes natural gas both as a hydrogen feedstock and as process
            energy. Gas accounts for approximately 70-80% of fertiliser production costs at European plants.
            When TTF rises sharply, European nitrogen fertiliser production becomes uneconomic and plants
            curtail output, driving fertiliser prices higher and reducing European supply.

            **Fertiliser to agriculture:** Nitrogen fertiliser is the primary input cost for cereal
            production globally. Higher fertiliser prices raise farmer break-even costs, support grain
            prices, and can reduce planted area in price-sensitive markets. The lag between a gas price
            shock and grain price response is typically 3-9 months, reflecting crop production cycles.

            **2022 case study:** TTF reached approximately 350 EUR/MWh in August 2022. Several major
            European fertiliser producers (BASF, Yara, CF Industries) curtailed output by 50-80%.
            Global wheat prices rose approximately 70% from January to May 2022, partly driven by the
            Russia-Ukraine harvest disruption and partly by fertiliser cost pass-through to global supply.

            **Current context:** At current TTF price levels, European fertiliser plants are broadly
            economic (approximately 30-40 EUR/MWh is the rough breakeven threshold for Haber-Bosch
            economics). The gas-to-agriculture transmission mechanism is partially dormant. A sustained
            TTF spike above 50 EUR/MWh would reactivate it.
            """)

    if fetched:
        st.caption(f"Last updated: {fetched.strftime('%Y-%m-%d %H:%M UTC')}")
    st.caption(
        "Sources: ICE/Yahoo Finance (TTF=F, ZW=F, ZC=F) | "
        "GIE ALSI | IEA Gas Market Report | Eurostat"
    )


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    """<div style="color:#484f58;font-size:0.72rem;line-height:1.8;">
    Built by Tobias Gudbjerg &nbsp;|&nbsp;
    Gas price: <a href="https://finance.yahoo.com" style="color:#484f58;">ICE/Yahoo Finance</a> &nbsp;|&nbsp;
    Supply mix: GIE ALSI, Eurostat, IEA &nbsp;|&nbsp;
    Agricultural futures: CBOT via Yahoo Finance<br>
    For informational purposes only. Not financial advice.
    </div>""",
    unsafe_allow_html=True,
)
