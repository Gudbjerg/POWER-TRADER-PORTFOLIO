"""
Home: landing page of the European Gas & Power Market Intelligence Platform.
"""
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="European Gas & Power: Market Intelligence Platform")

from utils.helpers import apply_dark_theme, has_entsoe_key, has_agsi_key
from utils.globe import GLOBE_VIEWS, globe_link, globe_view_url
from utils.theme import (
    ACCENTS, callout, chip, layer_card, page_footer, render, section_header, ticker_strip,
)
from data.market_strip import fetch_market_strip

apply_dark_theme()


# ── Hero ───────────────────────────────────────────────────────────────────────
with st.container(key="ptp_hero"):
    _hero_l, _hero_r = st.columns([1.75, 1], gap="large", vertical_alignment="center")
    with _hero_l:
        render(
            '<div class="ptp-hero">'
            f'<div class="ptp-kicker" style="--ph-accent:{ACCENTS["home"]}">Gas · Power · LNG · Carbon</div>'
            '<div class="ptp-hero__title" role="heading" aria-level="1">European Gas &amp; Power '
            '<span>Market Intelligence Platform</span></div>'
            '<div class="ptp-hero__lead">A five-layer market intelligence platform built to mirror the '
            'analytical workflow of a European gas and power trader.</div>'
            '<div class="ptp-hero__body">Live fundamental surveillance, quantitative signal generation, '
            'cross-commodity macro analysis, machine learning, portfolio risk simulation, and BESS project '
            'economics, integrated across seven pages and 35 analytical modules.</div>'
            "</div>"
        )
        with st.container(horizontal=True, gap="small", key="ptp_cta"):
            with st.container(width="content", key="ptp_cta_primary"):
                st.page_link("views/1_Live_Monitor.py", label="Open Live Monitor",
                             icon=":material/arrow_forward:", icon_position="right")
            st.page_link("views/5_Mispricing_Dashboard.py", label="Mispricing Dashboard",
                         icon=":material/track_changes:")
            globe_link("north_sea", "Open Energy Globe")
    with _hero_r:
        render(
            '<div class="ptp-hero-stats">'
            '<div class="ptp-hero-stat"><div class="ptp-hero-stat__value">5</div>'
            '<div class="ptp-hero-stat__label">Analytical layers</div></div>'
            '<div class="ptp-hero-stat"><div class="ptp-hero-stat__value">7</div>'
            '<div class="ptp-hero-stat__label">Pages</div></div>'
            '<div class="ptp-hero-stat"><div class="ptp-hero-stat__value">35</div>'
            '<div class="ptp-hero-stat__label">Analytical modules</div></div>'
            '<div class="ptp-hero-stat"><div class="ptp-hero-stat__value">14</div>'
            '<div class="ptp-hero-stat__label">Quant models</div></div>'
            "</div>"
        )

# ── Market snapshot (filled at the end of the run so the page renders first) ─────
render(
    '<div class="ptp-strip-head"><div class="ptp-strip-head__title">Market snapshot</div>'
    '<div class="ptp-strip-head__meta">Delayed · refreshed every 15 minutes</div></div>'
)
_strip_slot = st.empty()
_strip_slot.markdown(
    ticker_strip(
        [{"symbol": s, "sub": "Loading", "value": None} for s in
         ("TTF", "JKM", "Brent", "EUA", "EUR/USD", "NO2", "NL")]
    ),
    unsafe_allow_html=True,
)

# ── Data sources ───────────────────────────────────────────────────────────────
_agsi_chip = (
    chip("GIE AGSI+ / ALSI: Active", "up", dot=True) if has_agsi_key()
    else chip("GIE AGSI+ / ALSI: Add AGSI_API_KEY", "warn", dot=True)
)
_entsoe_chip = (
    chip("ENTSO-E Transparency: Active", "up", dot=True) if has_entsoe_key()
    else chip("ENTSO-E: Add ENTSOE_API_KEY", "warn", dot=True)
)
render(
    '<div class="ptp-status-row"><span class="ptp-status-row__label">Data sources</span>'
    + _agsi_chip + _entsoe_chip
    + chip("Nord Pool / Yahoo Finance / ICE: Active (no key required)", "up", dot=True)
    + "</div>"
)
render(callout(
    "First load fetches live data from ENTSO-E, GIE AGSI+, Nord Pool, and Yahoo Finance. "
    "Allow 1–2 minutes for all panels to populate. Subsequent loads are cached and near-instant.",
    "neutral",
))

# ── Layer cards ────────────────────────────────────────────────────────────────
section_header(
    "Platform layers",
    "From raw market observation to composite signals, portfolio risk and project economics.",
)

_LAYERS = [
    dict(key="l1", page="views/1_Live_Monitor.py", link="Open Live Monitor",
         kicker="Layer 1 · Markets", title="Live Market Monitor", stat="10 panels", icon="pulse",
         desc="Ten real-time panels: EU and German gas storage against 5-year bands, LNG sendout, "
              "TTF spot and forward strip, Nordic and FI minus IT-North day-ahead spreads, hydro, "
              "flows and solar."),
    dict(key="l2", page="views/2_Quant_Analysis.py", link="Open Quant Analysis",
         kicker="Layer 2 · Analytics", title="Quantitative Analysis", stat="14 models", icon="chart",
         desc="Storage refill Monte Carlo, gas-to-power regression, German merit order, multi-pair "
              "cointegration scanner with rolling stability, and TTF forward curve PCA with "
              "calendar-spread and butterfly signals."),
    dict(key="l3", page="views/3_Macro_Signals.py", link="Open Macro Signals",
         kicker="Layer 3 · Markets", title="Geopolitical & Macro Signals", stat="5 panels", icon="globe",
         desc="TTF timeline with 28 annotated market events, EU supply-mix evolution, gas-to-"
              "agriculture spillover, a 7×7 cross-commodity correlation grid and the TTF vs JKM "
              "inter-basin LNG spread."),
    dict(key="l4", page="views/4_ML_Models.py", link="Open ML Models",
         kicker="Layer 4 · Analytics", title="ML Models", stat="3 models", icon="cpu",
         desc="LSTM next-day power price forecaster benchmarked against a naive baseline, a 4-state "
              "Gaussian HMM regime classifier with a realised-volatility overlay, and a FinBERT news "
              "sentiment signal with Granger tests."),
    dict(key="l5", page="views/5_Mispricing_Dashboard.py", link="Open Mispricing Dashboard",
         kicker="Layer 5 · Signals aggregator", title="Mispricing Dashboard", stat="8 signals", icon="target",
         desc="Eight rich/cheap signals from Layers 1 to 4 ranked by historical percentile, with "
              "direction, confidence classification and composite commentary."),
    dict(key="risk", page="views/6_Risk_Dashboard.py", link="Open Risk Dashboard",
         kicker="D4 · Risk simulator", title="Risk Dashboard", stat="6 signals", icon="shield",
         desc="Portfolio simulator on the D3 signal suite: equity curves, Sharpe, 1-day 95%/99% VaR "
              "and Expected Shortfall, four stress scenarios and rolling signal correlations."),
    dict(key="bess", page="views/7_BESS_Optimiser.py", link="Open BESS Optimiser",
         kicker="Project toolkit", title="BESS Optimiser", stat="4 markets", icon="battery",
         desc="Battery storage economics for Nordic power: DAM arbitrage, FCR-N, FCR-D and intraday "
              "revenue stack with cash flow, NPV, IRR, payback and ±20% sensitivities."),
]


def _layer_grid(layers: list[dict]) -> None:
    cols = st.columns(len(layers), gap="medium")
    for col, layer in zip(cols, layers):
        with col:
            with st.container(key=f"ptp_layer_{layer['key']}"):
                render(layer_card(layer["kicker"], layer["title"], layer["desc"], layer["stat"],
                                  icon=layer["icon"], accent=ACCENTS[layer["key"]]))
                st.page_link(layer["page"], label=layer["link"],
                             icon=":material/arrow_forward:", icon_position="right")


_layer_grid(_LAYERS[:4])
_layer_grid(_LAYERS[4:])

# ── Companion 3D globe (shown only when GLOBE_URL is configured) ─────────────────
if globe_view_url("north_sea"):
    section_header(
        "Energy Globe",
        "The physical system behind the prices, in 3D: Norwegian shelf facilities and export "
        "pipelines, European LNG terminals with live send-out, and the maritime chokepoints that "
        "LNG and crude cargoes pass through.",
    )
    with st.container(horizontal=True, gap="small", key="ptp_globe_links"):
        globe_link("north_sea", "Norwegian shelf and pipelines")
        globe_link("lng_europe", "European LNG terminals")
        globe_link("lng_routes", "LNG shipping chokepoints")
        globe_link("hormuz", "Strait of Hormuz")

# ── About ──────────────────────────────────────────────────────────────────────
with st.expander("About this platform", expanded=False):
    render(
        "<div class='ptp-about'>"
        "Built by <strong>Tobias Gudbjerg</strong>, currently in equity sales at ABG Sundal Collier, "
        "to apply commodity markets and quantitative methods to European gas and power trading. "
        "The analytical framework draws on coursework from Bayes Business School: "
        "SMM591 Commodity Derivatives &amp; Trading, SMM284 Applied Machine Learning, "
        "SMM620 FX Trading &amp; Hedging, and SMM921 Market Microstructure. "
        "All data is sourced from public APIs (GIE AGSI+, ENTSO-E Transparency, Nord Pool, ICE/Yahoo Finance); "
        "all models are implemented from scratch in Python."
        "</div>"
    )
    _ac1, _ac2 = st.columns(2)
    with _ac1:
        render(
            "<div class='ptp-about-stack'>"
            "<div class='ptp-about-stack__title'>Analytics</div>"
            "scikit-learn &nbsp;·&nbsp; statsmodels &nbsp;·&nbsp; PyTorch<br>"
            "hmmlearn &nbsp;·&nbsp; HuggingFace Transformers (FinBERT)"
            "</div>"
        )
    with _ac2:
        render(
            "<div class='ptp-about-stack'>"
            "<div class='ptp-about-stack__title'>Data &amp; Infrastructure</div>"
            "Python 3.11+ &nbsp;·&nbsp; Streamlit &nbsp;·&nbsp; Plotly<br>"
            "Pandas &nbsp;·&nbsp; NumPy &nbsp;·&nbsp; yfinance &nbsp;·&nbsp; entsoe-py<br>"
            "Deployed on Render (EU Frankfurt) · HuggingFace Spaces (backup)"
            "</div>"
        )

page_footer([
    ("Gas storage and LNG", "GIE AGSI+/ALSI", "https://agsi.gie.eu"),
    ("Power data", "ENTSO-E Transparency Platform", "https://transparency.entsoe.eu"),
    ("Spot prices", "Nord Pool", "https://www.nordpoolgroup.com"),
    ("TTF", "Yahoo Finance / ICE", "https://finance.yahoo.com"),
])

# ── Fill the market snapshot (cached for 15 minutes) ──────────────────────────
_strip = fetch_market_strip()
_strip_slot.markdown(
    ticker_strip(
        _strip["items"],
        note=f"Day-over-day change. Futures, carbon and FX: last daily bar via Yahoo Finance. "
             f"Power: Nord Pool day-ahead daily average, change in EUR/MWh. "
             f"Last fetch {_strip['fetched_at']:%Y-%m-%d %H:%M} UTC.",
    ),
    unsafe_allow_html=True,
)
