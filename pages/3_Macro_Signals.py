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
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from utils.helpers import apply_dark_theme, kpi_card, delta_span, commentary
from data.prices import get_ttf_data
from data.events import load_events, events_in_range, CATEGORY_COLORS
from data.commodities import get_commodity_data
from components.prices_chart import render_ttf_chart

apply_dark_theme()


def _mom_delta(v: float) -> str:
    if np.isnan(v):
        return delta_span("n/a", "blue")
    color = "red" if v > 3 else ("green" if v < -3 else "blue")
    return delta_span(f"{'▲' if v > 0 else '▼'} {abs(v):.1f}%", color)


# ── 7×7 Correlation Grid ─────────────────────────────────────────────────────
_TICKERS_7X7 = {
    "TTF":    "TTF=F",   # ICE TTF front-month gas (EUR/MWh)
    "Brent":  "BZ=F",    # ICE Brent crude (USD/bbl)
    "Coal":   "MTF=F",   # ICE Rotterdam coal / API2 proxy (USD/t)
    "EUA":    "CO2.L",   # EU carbon allowances (ICE, EUR/t)
    "Copper": "HG=F",    # COMEX copper (USD/lb, LME proxy)
    "Alum":   "ALI=F",   # COMEX aluminium (USD/t, LME proxy)
    "BDI":    "BDRY",    # Breakwave Dry Bulk ETF (Baltic Dry proxy)
}


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_7x7_prices(days: int = 180) -> pd.DataFrame:
    """Daily close prices for 7 macro/commodity assets via yfinance. Wide format, date-indexed."""
    try:
        import yfinance as _yf
    except ImportError:
        return pd.DataFrame()
    _end  = pd.Timestamp.now()
    _strt = (_end - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    _ends = _end.strftime("%Y-%m-%d")
    _frames: dict = {}
    for _nm, _tkr in _TICKERS_7X7.items():
        try:
            _raw = _yf.Ticker(_tkr).history(start=_strt, end=_ends)
            if not _raw.empty and "Close" in _raw.columns:
                _s = _raw["Close"].copy()
                _s.index = pd.to_datetime(_s.index).tz_localize(None)
                _frames[_nm] = _s
        except Exception:
            pass
    if len(_frames) < 2:
        return pd.DataFrame()
    _wide = pd.DataFrame(_frames).sort_index().ffill(limit=3)
    _wide.index = pd.to_datetime(_wide.index).normalize()
    _wide.index.name = "date"
    return _wide.dropna(thresh=max(2, len(_wide.columns) - 2))


@st.cache_data(show_spinner=False)
def _compute_lead_lag(wide: pd.DataFrame, max_lag: int = 10, window: int = 90) -> list[dict]:
    """
    For each asset pair, find lag τ ∈ [−max_lag, max_lag] maximising |Pearson ρ| of daily
    log-returns over the last `window` trading days. Positive τ: left asset leads right.
    Returns list of dicts sorted by |best_corr| descending.
    """
    _ret  = np.log(wide / wide.shift(1)).iloc[-window:].copy()
    _cols = list(_ret.columns)
    _res: list[dict] = []
    for _i, _a in enumerate(_cols):
        for _b in _cols[_i + 1:]:
            _sa = _ret[_a].dropna()
            _sb = _ret[_b].dropna()
            _bl, _bc = 0, 0.0
            for _lag in range(-max_lag, max_lag + 1):
                _al = pd.concat([_sa.shift(_lag), _sb], axis=1).dropna()
                if len(_al) < 30:
                    continue
                _c = float(_al.iloc[:, 0].corr(_al.iloc[:, 1]))
                if not np.isnan(_c) and abs(_c) > abs(_bc):
                    _bl, _bc = _lag, _c
            _ct = pd.concat([_sa, _sb], axis=1).dropna()
            _contemp = (float(_ct.iloc[:, 0].corr(_ct.iloc[:, 1]))
                        if len(_ct) >= 30 else float("nan"))
            _res.append({
                "pair":      f"{_a} / {_b}",
                "a": _a,     "b": _b,
                "best_lag":  _bl,
                "best_corr": round(float(_bc), 3),
                "contemp":   round(float(_contemp), 3) if not np.isnan(_contemp) else float("nan"),
                "lag_label": (
                    f"{_a} leads {_b} by {_bl}d" if _bl > 0
                    else (f"{_b} leads {_a} by {-_bl}d" if _bl < 0
                          else "Contemporaneous")
                ),
            })
    _res.sort(key=lambda r: -abs(r["best_corr"]))
    return _res


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

tab_geo, tab_mix, tab_spill, tab_grid = st.tabs([
    "Geopolitical Overlay",
    "EU Gas Supply Mix",
    "Cross-Commodity Spillover",
    "7×7 Correlation Grid",
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

    # ── Momentum ribbon ───────────────────────────────────────────────────────
    _mom_df = ttf["prices"].copy() if not ttf["prices"].empty else pd.DataFrame()
    if not _mom_df.empty and len(_mom_df) >= 60:
        _mom_df["date"] = pd.to_datetime(_mom_df["date"])
        _mom_df = _mom_df.sort_values("date").reset_index(drop=True)
        for _n in (5, 20, 60):
            _mom_df[f"mom{_n}"] = (_mom_df["price"] / _mom_df["price"].shift(_n) - 1) * 100

        _m5  = float(_mom_df["mom5"].dropna().iloc[-1])  if _mom_df["mom5"].notna().any()  else float("nan")
        _m20 = float(_mom_df["mom20"].dropna().iloc[-1]) if _mom_df["mom20"].notna().any() else float("nan")
        _m60 = float(_mom_df["mom60"].dropna().iloc[-1]) if _mom_df["mom60"].notna().any() else float("nan")

        st.markdown("#### TTF Price Momentum (5 / 20 / 60-day)")
        _mc1, _mc2, _mc3, _mc4 = st.columns(4)
        with _mc1:
            st.markdown(
                kpi_card("5-day momentum", f"{_m5:+.1f}%" if not np.isnan(_m5) else "n/a", _mom_delta(_m5)),
                unsafe_allow_html=True,
            )
        with _mc2:
            st.markdown(
                kpi_card("20-day momentum", f"{_m20:+.1f}%" if not np.isnan(_m20) else "n/a", _mom_delta(_m20)),
                unsafe_allow_html=True,
            )
        with _mc3:
            st.markdown(
                kpi_card("60-day momentum", f"{_m60:+.1f}%" if not np.isnan(_m60) else "n/a", _mom_delta(_m60)),
                unsafe_allow_html=True,
            )
        with _mc4:
            _n_pos = sum(v > 0 for v in (_m5, _m20, _m60) if not np.isnan(v))
            _trend = "Uptrend" if _n_pos == 3 else ("Downtrend" if _n_pos == 0 else "Mixed / consolidating")
            _trend_color = "red" if _n_pos == 3 else ("green" if _n_pos == 0 else "blue")
            st.markdown(
                kpi_card("Trend alignment", _trend,
                         delta_span(f"{_n_pos}/3 timeframes positive", _trend_color)),
                unsafe_allow_html=True,
            )

        # Momentum line chart
        _fig_mom = go.Figure()
        for _n, _col, _lbl in [(5, "#f85149", "5d"), (20, "#e07b39", "20d"), (60, "#58a6ff", "60d")]:
            _col_name = f"mom{_n}"
            _sub = _mom_df.dropna(subset=[_col_name])
            _fig_mom.add_trace(go.Scatter(
                x=_sub["date"], y=_sub[_col_name],
                name=f"{_lbl} momentum",
                line=dict(color=_col, width=1.5),
                hovertemplate=f"{_lbl}: %{{y:+.1f}}%<extra></extra>",
            ))
        _fig_mom.add_hline(y=0, line=dict(color="rgba(255,255,255,0.25)", width=1))
        for _band, _opacity in [(5, 0.08), (10, 0.05)]:
            _fig_mom.add_hrect(y0=_band, y1=50, fillcolor="rgba(248,81,73,{})".format(_opacity),
                               line_width=0)
            _fig_mom.add_hrect(y0=-50, y1=-_band, fillcolor="rgba(63,185,80,{})".format(_opacity),
                               line_width=0)
        _fig_mom.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            xaxis=dict(title=None, gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(title="Return (%)", gridcolor="rgba(255,255,255,0.06)",
                       zeroline=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                        font=dict(size=10)),
            margin=dict(l=60, r=20, t=30, b=50),
            height=300,
            hovermode="x unified",
        )
        st.plotly_chart(_fig_mom, use_container_width=True)
        st.caption(
            "Momentum = (current price / price N days ago − 1) × 100. "
            "Red shading = historically elevated (+5/+10%). Green = historically cheap (−5/−10%)."
        )

    st.divider()
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
    st.plotly_chart(fig_mix, use_container_width=True)

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
        st.plotly_chart(fig_comm, use_container_width=True)

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
            st.plotly_chart(fig_corr, use_container_width=True)

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


# ════════════════════════════════════════════════════════════════════════════
# TAB 4: 7×7 CORRELATION GRID
# ════════════════════════════════════════════════════════════════════════════
with tab_grid:
    st.markdown("### 7×7 Cross-Commodity Correlation Grid")
    st.caption(
        "90-day rolling Pearson correlations of daily log-returns across TTF gas, Brent crude, "
        "API2 coal, EU carbon (EUA), copper, aluminium, and the Baltic Dry Index. "
        "All data via Yahoo Finance. Signals update on page load."
    )

    with st.spinner("Loading cross-commodity prices…"):
        _w7 = _fetch_7x7_prices(days=180)

    _loaded  = list(_w7.columns) if not _w7.empty else []
    _missing = [k for k in _TICKERS_7X7 if k not in _loaded]
    if _missing:
        st.caption(
            f"Assets unavailable from Yahoo Finance: {', '.join(_missing)}. "
            f"Grid shows {len(_loaded)} of 7 assets."
        )

    if _w7.empty or len(_loaded) < 2:
        st.warning(
            "Insufficient cross-commodity data. "
            "Requires internet access and yfinance (pip install yfinance)."
        )
    else:
        _ll_pairs = _compute_lead_lag(_w7, max_lag=10, window=90)

        # ── KPI row ──────────────────────────────────────────────────────────
        _strong_pairs    = [r for r in _ll_pairs if abs(r["best_corr"]) >= 0.5]
        _best            = _ll_pairs[0] if _ll_pairs else None
        _best_lag_signal = next(
            (r for r in _ll_pairs if r["best_lag"] != 0 and abs(r["best_corr"]) >= 0.4), None
        )

        _kc1, _kc2, _kc3 = st.columns(3)
        with _kc1:
            if _best:
                _bc_col = "red" if _best["best_corr"] > 0 else "green"
                st.markdown(
                    kpi_card(
                        "Strongest pair (90d ρ)",
                        _best["pair"],
                        delta_span(f"ρ = {_best['best_corr']:+.2f}", _bc_col),
                    ),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(kpi_card("Strongest pair", "n/a", delta_span("–", "blue")),
                            unsafe_allow_html=True)
        with _kc2:
            st.markdown(
                kpi_card(
                    "Strong pairs (|ρ| ≥ 0.5)",
                    str(len(_strong_pairs)),
                    delta_span(f"of {len(_ll_pairs)} total pairs", "blue"),
                ),
                unsafe_allow_html=True,
            )
        with _kc3:
            if _best_lag_signal:
                _ll_col = "red" if _best_lag_signal["best_corr"] > 0 else "green"
                st.markdown(
                    kpi_card(
                        "Best lead-lag signal",
                        _best_lag_signal["lag_label"],
                        delta_span(f"ρ = {_best_lag_signal['best_corr']:+.2f}", _ll_col),
                    ),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    kpi_card("Best lead-lag signal", "No lagged signal ≥ 0.4", delta_span("–", "blue")),
                    unsafe_allow_html=True,
                )

        # ── Correlation heatmap ───────────────────────────────────────────────
        st.markdown("#### 90-Day Correlation Matrix")
        _ret90  = np.log(_w7 / _w7.shift(1)).iloc[-90:].dropna(how="all")
        _cm     = _ret90.corr()
        _assets = list(_cm.columns)
        _z_vals = _cm.values.tolist()
        _z_text = [
            [f"{v:.2f}" if not np.isnan(v) else "" for v in row]
            for row in _cm.values
        ]

        _fig_hm = go.Figure(go.Heatmap(
            z=_z_vals,
            x=_assets,
            y=_assets,
            text=_z_text,
            texttemplate="%{text}",
            colorscale=[
                [0.00, "#2166ac"],
                [0.25, "#74add1"],
                [0.50, "#161b22"],
                [0.75, "#d6604d"],
                [1.00, "#b2182b"],
            ],
            zmin=-1, zmax=1,
            showscale=True,
            colorbar=dict(
                title="ρ",
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["-1", "-0.5", "0", "+0.5", "+1"],
                thickness=12,
                len=0.8,
                tickfont=dict(size=10),
            ),
            hoverongaps=False,
            hovertemplate="%{y} / %{x}: %{z:.3f}<extra></extra>",
        ))
        _fig_hm.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#161b22",
            font=dict(color="#c9d1d9", size=11),
            margin=dict(l=10, r=10, t=10, b=10),
            height=420,
            xaxis=dict(side="bottom", tickfont=dict(size=11)),
            yaxis=dict(tickfont=dict(size=11), autorange="reversed"),
        )
        st.plotly_chart(_fig_hm, use_container_width=True)
        st.caption(
            "Computed over the 90 most recent trading days using daily log-returns. "
            "Diagonal = 1.0 (self-correlation, shown for reference)."
        )

        # ── Lead-lag table ────────────────────────────────────────────────────
        st.markdown("#### Lead-Lag Analysis — Top 15 Pairs by |ρ|")
        st.caption(
            "For each pair the lag τ ∈ [−10d, +10d] maximising |ρ| is reported. "
            "Positive τ = left asset leads right asset by τ days."
        )
        _top15 = _ll_pairs[:15]
        if _top15:
            _ll_css = """
            <style>
            .ll-tbl{width:100%;border-collapse:collapse;font-size:0.80rem;color:#c9d1d9;}
            .ll-tbl th{background:#21262d;color:#8b949e;font-weight:600;
                       padding:6px 10px;text-align:left;border-bottom:1px solid #30363d;}
            .ll-tbl td{padding:5px 10px;border-bottom:1px solid #21262d;}
            .cb-wrap{background:#21262d;border-radius:3px;height:6px;
                     width:100px;display:inline-block;vertical-align:middle;margin-left:6px;}
            .cb-fill{height:6px;border-radius:3px;}
            </style>
            """
            _rows = ""
            for _r in _top15:
                _bc  = _r["best_corr"]
                _ct  = _r["contemp"]
                _lag = _r["best_lag"]
                _bar_w = int(abs(_bc) * 100)
                _bar_c = "#d6604d" if _bc > 0 else "#74add1"
                _bc_c  = "#d6604d" if _bc > 0 else "#74add1"
                _lag_c = "#58a6ff" if _lag != 0 else "#8b949e"
                _ct_s  = f"{_ct:+.2f}" if not np.isnan(_ct) else "–"
                _rows += (
                    f"<tr>"
                    f"<td>{_r['pair']}</td>"
                    f"<td><span style='color:{_bc_c};font-weight:600;'>{_bc:+.3f}</span>"
                    f"<span class='cb-wrap'><span class='cb-fill' "
                    f"style='width:{_bar_w}px;background:{_bar_c};'></span></span></td>"
                    f"<td><span style='color:{_lag_c};'>{_lag:+d}d</span></td>"
                    f"<td>{_r['lag_label']}</td>"
                    f"<td>{_ct_s}</td>"
                    f"</tr>"
                )
            st.markdown(
                _ll_css
                + "<table class='ll-tbl'><thead><tr>"
                + "<th>Pair</th>"
                + "<th>Best ρ (optimal lag)</th>"
                + "<th>Lag</th>"
                + "<th>Lead direction</th>"
                + "<th>Contemp ρ</th>"
                + f"</tr></thead><tbody>{_rows}</tbody></table>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("No pairs with sufficient data to rank.")

        # ── Methodology expander ──────────────────────────────────────────────
        with st.expander("Methodology and asset definitions", expanded=False):
            st.markdown("""
            **Asset universe**

            | Label  | Ticker  | Description |
            |--------|---------|-------------|
            | TTF    | TTF=F   | ICE TTF Natural Gas front-month (EUR/MWh) |
            | Brent  | BZ=F    | ICE Brent Crude Oil front-month (USD/bbl) |
            | Coal   | MTF=F   | ICE Rotterdam API2 Coal front-month (USD/t) |
            | EUA    | CO2.L   | EU Emission Allowance — ICE EUA Dec (EUR/tCO₂) |
            | Copper | HG=F    | COMEX High-Grade Copper (USD/lb, LME proxy) |
            | Alum   | ALI=F   | COMEX Aluminium (USD/t, LME proxy) |
            | BDI    | BDRY    | Breakwave Dry Bulk Shipping ETF — Baltic Dry proxy |

            **Correlation methodology**

            Correlations are computed on **daily log-returns** (`log(P_t / P_{t-1})`) over a
            rolling 90-trading-day window. Log-returns remove price-level drift and produce
            more stationary series than raw prices or price ratios.

            **Lead-lag analysis**

            For each pair (A, B) we test lags τ ∈ [−10, +10] trading days. At lag τ we compute
            Pearson ρ between `A.shift(τ)` and `B`. The τ maximising |ρ| is the "best lag."
            Positive τ means A leads B by τ days. A signal is flagged as actionable when
            |ρ| ≥ 0.4 and τ ≠ 0.

            **Positioning relevance (Statkraft context)**

            Cross-commodity correlations are most useful for:
            - **Calendar spreading**: gas/carbon/coal correlations drive clean dark/spark spread compression
            - **Macro hedging**: copper and aluminium lead demand signals for European industrial load
            - **Baltic Dry as leading indicator**: shipping costs often lead energy demand by 4–8 weeks

            **Limitations:** Lagged correlations are regime-dependent and unstable. A high historical
            ρ does not imply causation. Always validate against a structural mechanism before trading.
            """)

    st.caption(
        "Sources: Yahoo Finance (TTF=F, BZ=F, MTF=F, CO2.L, HG=F, ALI=F, BDRY) | "
        "90-day rolling window | Updated on page load"
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
