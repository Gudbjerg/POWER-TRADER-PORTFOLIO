"""
European Gas & Power Market Intelligence Platform — Landing Page
"""
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="European Gas & Power — Market Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from utils.helpers import has_entsoe_key, has_agsi_key, apply_dark_theme

apply_dark_theme()

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("# European Gas & Power Market Intelligence Platform")
st.markdown(
    "<p style='color:#8b949e;font-size:1.05rem;line-height:1.7;max-width:860px;'>"
    "A five-layer market intelligence platform built to mirror the analytical workflow of a European gas and power trader. "
    "Live fundamental surveillance, quantitative signal generation, cross-commodity macro analysis, "
    "machine learning, and portfolio risk simulation — integrated across six pages and 20 analytical modules."
    "</p>",
    unsafe_allow_html=True,
)
st.info(
    "First load fetches live data from ENTSO-E, GIE AGSI+, Nord Pool, and Yahoo Finance. "
    "Allow 1–2 minutes for all panels to populate. Subsequent loads are cached and near-instant.",
    icon="⏱",
)

st.divider()

# ── Data source status ─────────────────────────────────────────────────────────
st.markdown("#### Data Source Status")
s1, s2, s3 = st.columns(3)
with s1:
    if has_agsi_key():
        st.success("GIE AGSI+ / ALSI: Active")
    else:
        st.warning("GIE AGSI+ / ALSI: Add AGSI_API_KEY")
with s2:
    if has_entsoe_key():
        st.success("ENTSO-E Transparency: Active")
    else:
        st.warning("ENTSO-E: Add ENTSOE_API_KEY")
with s3:
    st.success("Nord Pool / Yahoo Finance / ICE: Active (no key required)")

st.divider()

# ── Layer cards ────────────────────────────────────────────────────────────────
st.markdown("#### Platform Layers")

c1, c2 = st.columns(2)

with c1:
    st.markdown("""
<div style="background:#161b22;border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:20px 22px;margin-bottom:12px;">
  <div style="color:#58a6ff;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;font-weight:600;">Layer 1</div>
  <div style="color:#e6edf3;font-size:1.05rem;font-weight:600;margin:4px 0 10px;">Live Market Monitor</div>
  <div style="color:#8b949e;font-size:0.88rem;line-height:1.65;">
    Fundamental surveillance across ten real-time panels: EU and German gas storage against the 5-year
    historical band, NW European LNG terminal sendout (Zeebrugge, Gate, South Hook, Dunkirk), TTF spot
    with 30/90-day moving averages and spike alerts, day-ahead prices across NO1/NO2/SE3/NL/FI with
    Nordic–Continental spread history, TTF seasonal forward strip (M+1–M+18), Norwegian hydro reservoir
    levels (P10/P50/P90), Nordic cross-border flow utilisation percentages, and the German intraday
    solar duck curve with cannibalisation signal.
  </div>
</div>
""", unsafe_allow_html=True)
    st.page_link("pages/1_Live_Monitor.py", label="Open Live Monitor →")

with c2:
    st.markdown("""
<div style="background:#161b22;border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:20px 22px;margin-bottom:12px;">
  <div style="color:#3fb950;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;font-weight:600;">Layer 2 · 14 models</div>
  <div style="color:#e6edf3;font-size:1.05rem;font-weight:600;margin:4px 0 10px;">Quantitative Analysis</div>
  <div style="color:#8b949e;font-size:0.88rem;line-height:1.65;">
    Full quant stack for gas and power trading: storage refill Monte Carlo (1,000 bootstrap paths against
    the EU 90% mandate), gas-to-power OLS regression with rolling residual z-score, German merit order
    with live ENTSO-E demand, multi-pair cointegration scanner across a 6-asset universe
    (Engle-Granger, OU half-life, ranked entry signals), TTF forward curve PCA decomposing the M+1–M+18
    strip into level/slope/curvature factors with calendar-spread and butterfly trade signals,
    plus TTF seasonal norm tracker, hydro lead/lag, and further models.
  </div>
</div>
""", unsafe_allow_html=True)
    st.page_link("pages/2_Quant_Analysis.py", label="Open Quant Analysis →")

c3, c4 = st.columns(2)

with c3:
    st.markdown("""
<div style="background:#161b22;border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:20px 22px;margin-bottom:12px;">
  <div style="color:#d29922;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;font-weight:600;">Layer 3 · 4 panels</div>
  <div style="color:#e6edf3;font-size:1.05rem;font-weight:600;margin:4px 0 10px;">Geopolitical & Macro Signals</div>
  <div style="color:#8b949e;font-size:0.88rem;line-height:1.65;">
    Four macro panels: TTF price timeline with 19 annotated supply disruptions and a 5/20/60-day momentum
    ribbon, EU gas supply-mix evolution 2020–2026 (Russia phase-out, US LNG ramp, Algerian shift),
    gas-to-fertiliser-to-agriculture spillover chain with rolling 90-day correlations, and a 7×7
    cross-commodity correlation grid (TTF, Brent, API2 coal, EUA, copper, aluminium, Baltic Dry)
    with 90-day Pearson heatmap and ±10-day lead-lag analysis.
  </div>
</div>
""", unsafe_allow_html=True)
    st.page_link("pages/3_Macro_Signals.py", label="Open Macro Signals →")

with c4:
    st.markdown("""
<div style="background:#161b22;border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:20px 22px;margin-bottom:12px;">
  <div style="color:#8b949e;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;font-weight:600;">Layer 4</div>
  <div style="color:#e6edf3;font-size:1.05rem;font-weight:600;margin:4px 0 10px;">ML Models</div>
  <div style="color:#8b949e;font-size:0.88rem;line-height:1.65;">
    Machine learning applied to energy price formation: a 2-layer PyTorch LSTM next-day power price
    forecaster with honest baseline comparison (model flagged if MAE exceeds the naïve benchmark),
    a 4-state Gaussian HMM regime classifier (hydro-driven, gas-driven, renewables-driven, geopolitical
    stress) with current regime badge and 30-day history, and a FinBERT news sentiment signal with
    Granger causality test against TTF returns.
  </div>
</div>
""", unsafe_allow_html=True)
    st.page_link("pages/4_ML_Models.py", label="Open ML Models →")

st.markdown("""
<div style="background:#161b22;border:1px solid rgba(255,255,255,0.08);border-left:3px solid #f85149;border-radius:0 8px 8px 0;padding:20px 22px;margin-top:4px;margin-bottom:12px;">
  <div style="color:#f85149;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;font-weight:600;">Layer 5 · Signals Aggregator</div>
  <div style="color:#e6edf3;font-size:1.05rem;font-weight:600;margin:4px 0 10px;">Mispricing Dashboard</div>
  <div style="color:#8b949e;font-size:0.88rem;line-height:1.65;">
    Eight rich/cheap signals from Layers 1–4 ranked into a single composite view: historical percentile
    context (how extreme is the current reading vs 2–3 years of history), direction badge
    (Bullish/Bearish/Neutral), confidence classification (High/Medium/Low by distance from the 50th
    percentile), and composite commentary. Signals sorted by extremity — most off-centre first.
    Covers TTF seasonal position, EU storage fill, NO2/NL spread z-score, NO2 vs TTF regression
    residual, Clean Spark Spread, Norwegian hydro level, TTF vs storage residual, and marginal fuel regime.
  </div>
</div>
""", unsafe_allow_html=True)
st.page_link("pages/5_Mispricing_Dashboard.py", label="Open Mispricing Dashboard →")

st.markdown("""
<div style="background:#161b22;border:1px solid rgba(255,255,255,0.08);border-left:3px solid #d4ac3a;border-radius:0 8px 8px 0;padding:20px 22px;margin-top:4px;margin-bottom:12px;">
  <div style="color:#d4ac3a;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;font-weight:600;">D4 · Risk Simulator</div>
  <div style="color:#e6edf3;font-size:1.05rem;font-weight:600;margin:4px 0 10px;">Risk Dashboard</div>
  <div style="color:#8b949e;font-size:0.88rem;line-height:1.65;">
    Portfolio VaR and stress-scenario simulator built on the D3 signal suite. Select any combination of
    the six backtestable signals (TTF seasonal, EU storage bias, NO2/NL spread z-score, NO2 vs TTF OLS
    residual, Norwegian hydro, TTF vs storage residual) and assign notional sizes to simulate historical
    daily P&L. Outputs: per-signal equity curves, annualised Sharpe, 1-day 95%/99% VaR and Expected
    Shortfall (non-parametric bootstrap), four named stress scenarios (cold snap, Norwegian outage,
    Hormuz extension, EUR/USD shock), and 30-day rolling signal correlation matrix.
    For analytical purposes only — not financial advice.
  </div>
</div>
""", unsafe_allow_html=True)
st.page_link("pages/6_Risk_Dashboard.py", label="Open Risk Dashboard →")

st.divider()

# ── About ──────────────────────────────────────────────────────────────────────
with st.expander("About this platform", expanded=False):
    st.markdown(
        "<div style='color:#c9d1d9;font-size:0.90rem;line-height:1.85;max-width:820px;'>"
        "Built by <strong>Tobias Gudbjerg</strong>, currently in equity sales at ABG Sundal Collier, "
        "to apply commodity markets and quantitative methods to European gas and power trading. "
        "The analytical framework draws on coursework from Bayes Business School: "
        "SMM591 Commodity Derivatives &amp; Trading, SMM284 Applied Machine Learning, "
        "SMM620 FX Trading &amp; Hedging, and SMM921 Market Microstructure. "
        "All data is sourced from public APIs (GIE AGSI+, ENTSO-E Transparency, Nord Pool, ICE/Yahoo Finance); "
        "all models are implemented from scratch in Python."
        "</div>",
        unsafe_allow_html=True,
    )
    _ac1, _ac2 = st.columns(2)
    with _ac1:
        st.markdown(
            "<div style='color:#8b949e;font-size:0.82rem;line-height:1.9;margin-top:10px;'>"
            "<strong style='color:#c9d1d9;'>Analytics</strong><br>"
            "scikit-learn &nbsp;·&nbsp; statsmodels &nbsp;·&nbsp; PyTorch<br>"
            "hmmlearn &nbsp;·&nbsp; HuggingFace Transformers (FinBERT)"
            "</div>",
            unsafe_allow_html=True,
        )
    with _ac2:
        st.markdown(
            "<div style='color:#8b949e;font-size:0.82rem;line-height:1.9;margin-top:10px;'>"
            "<strong style='color:#c9d1d9;'>Data &amp; Infrastructure</strong><br>"
            "Python 3.11+ &nbsp;·&nbsp; Streamlit &nbsp;·&nbsp; Plotly<br>"
            "Pandas &nbsp;·&nbsp; NumPy &nbsp;·&nbsp; yfinance &nbsp;·&nbsp; entsoe-py<br>"
            "Deployed on HuggingFace Spaces"
            "</div>",
            unsafe_allow_html=True,
        )

st.divider()
st.markdown(
    "<div style='color:#484f58;font-size:0.72rem;line-height:1.8;'>"
    "Built by Tobias Gudbjerg &nbsp;|&nbsp;"
    "Gas storage and LNG: <a href='https://agsi.gie.eu' style='color:#484f58;'>GIE AGSI+/ALSI</a> &nbsp;|&nbsp;"
    "Power data: <a href='https://transparency.entsoe.eu' style='color:#484f58;'>ENTSO-E Transparency Platform</a> &nbsp;|&nbsp;"
    "Spot prices: <a href='https://www.nordpoolgroup.com' style='color:#484f58;'>Nord Pool</a> &nbsp;|&nbsp;"
    "TTF: <a href='https://finance.yahoo.com' style='color:#484f58;'>Yahoo Finance / ICE</a><br>"
    "For informational purposes only. Not financial advice."
    "</div>",
    unsafe_allow_html=True,
)
