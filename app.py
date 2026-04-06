"""
European Power and Gas Analysis Platform - Landing Page
"""
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="European Power and Gas Analysis Platform",
    page_icon="P",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.helpers import has_entsoe_key, has_agsi_key, apply_dark_theme

apply_dark_theme()

# ── Hero ───────────────────────────────────────────────────────────────────
st.markdown("# European Power and Gas Analysis Platform")
st.markdown(
    "<p style='color:#8b949e;font-size:1.05rem;line-height:1.7;max-width:820px;'>"
    "A four-layer market intelligence platform for European power and gas markets. "
    "Integrates live fundamental data, quantitative models, geopolitical context, "
    "and market analytics: the tools a physical energy trader uses daily, built from scratch."
    "</p>",
    unsafe_allow_html=True,
)
st.info(
    "First page load fetches live data from ENTSO-E, GIE AGSI+, Nord Pool, and Yahoo Finance. "
    "Allow 1–2 minutes for all panels to populate, particularly on Layer 1 and Layer 4. "
    "Subsequent loads are cached and near-instant.",
    icon="⏱",
)

st.divider()

# ── Data source status ─────────────────────────────────────────────────────
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

# ── Layer cards ────────────────────────────────────────────────────────────
st.markdown("#### Platform Layers")

c1, c2 = st.columns(2)

with c1:
    st.markdown("""
<div style="background:#161b22;border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:20px 22px;margin-bottom:12px;">
  <div style="color:#58a6ff;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;font-weight:600;">Layer 1</div>
  <div style="color:#e6edf3;font-size:1.05rem;font-weight:600;margin:4px 0 10px;">Live Market Monitor</div>
  <div style="color:#8b949e;font-size:0.88rem;line-height:1.65;">
    Real-time European energy data refreshed hourly: EU and German gas storage versus the 5-year historical band,
    NW European LNG terminal sendout (Zeebrugge, Gate, South Hook, Dunkirk), TTF gas price with moving averages
    and spike alerts, day-ahead spot prices across NO1, NO2, SE3, NL and FI, the TTF seasonal forward curve,
    plus Norwegian hydro reservoir levels, Nordic cross-border flows, and the solar duck curve (ENTSO-E).
  </div>
</div>
""", unsafe_allow_html=True)
    st.page_link("pages/1_Live_Monitor.py", label="Open Live Monitor")

with c2:
    st.markdown("""
<div style="background:#161b22;border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:20px 22px;margin-bottom:12px;">
  <div style="color:#3fb950;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;font-weight:600;">Layer 2</div>
  <div style="color:#e6edf3;font-size:1.05rem;font-weight:600;margin:4px 0 10px;">Quantitative Analysis</div>
  <div style="color:#8b949e;font-size:0.88rem;line-height:1.65;">
    Quantitative models used by physical power traders: gas-to-power OLS regression with rolling residual
    z-score tracker, storage refill Monte Carlo simulation across 1,000 paths with an empirical injection
    rate bootstrap, and a rolling 30-day price spike detector covering all major Nordic and Continental
    bidding zones.
  </div>
</div>
""", unsafe_allow_html=True)
    st.page_link("pages/2_Quant_Analysis.py", label="Open Quant Analysis")

c3, c4 = st.columns(2)

with c3:
    st.markdown("""
<div style="background:#161b22;border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:20px 22px;margin-bottom:12px;">
  <div style="color:#d29922;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;font-weight:600;">Layer 3</div>
  <div style="color:#e6edf3;font-size:1.05rem;font-weight:600;margin:4px 0 10px;">Geopolitical and Macro Signals</div>
  <div style="color:#8b949e;font-size:0.88rem;line-height:1.65;">
    Macro and geopolitical context for European energy markets: TTF price history with annotated supply
    event overlays (Iran nuclear risk, US LNG tariff threats, Norwegian outages), EU gas supply mix by
    origin from 2020 to present (Russia collapse, US LNG rise), and a cross-commodity spillover chain
    linking gas prices to fertiliser and agricultural markets.
  </div>
</div>
""", unsafe_allow_html=True)
    st.page_link("pages/3_Macro_Signals.py", label="Open Macro Signals")

with c4:
    st.markdown("""
<div style="background:#161b22;border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:20px 22px;margin-bottom:12px;">
  <div style="color:#8b949e;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;font-weight:600;">Layer 4</div>
  <div style="color:#e6edf3;font-size:1.05rem;font-weight:600;margin:4px 0 10px;">ML and Advanced Models</div>
  <div style="color:#8b949e;font-size:0.88rem;line-height:1.65;">
    Machine learning applied to energy market forecasting: LSTM next-day power price forecaster,
    Hidden Markov Model market regime classifier (hydro, gas, wind, stress states),
    and NLP news sentiment signal using FinBERT on energy headlines.
    Train buttons included — models run on available hardware.
  </div>
</div>
""", unsafe_allow_html=True)
    st.page_link("pages/4_ML_Models.py", label="Open ML Models")

st.divider()

# ── Research Pipeline ──────────────────────────────────────────────────────
st.markdown("#### Research Pipeline")
st.markdown(
    "<p style='color:#8b949e;font-size:0.88rem;margin-bottom:16px;'>"
    "Independent analyses and platform extensions in progress or under consideration. "
    "Each is framed as a specific market question, not just a feature."
    "</p>",
    unsafe_allow_html=True,
)

rp1, rp2, rp3 = st.columns(3)

with rp1:
    st.markdown("""
<div style="background:#161b22;border:1px solid rgba(255,255,255,0.08);border-left:3px solid #4caf8f;border-radius:0 8px 8px 0;padding:16px 18px;margin-bottom:10px;">
  <div style="color:#4caf8f;font-size:0.68rem;text-transform:uppercase;letter-spacing:1px;font-weight:600;margin-bottom:6px;">Independent Analysis</div>
  <div style="color:#e6edf3;font-size:0.92rem;font-weight:600;margin-bottom:8px;">Norwegian Hydro and NO2 Winter Prices</div>
  <div style="color:#8b949e;font-size:0.82rem;line-height:1.65;">
    What is the historical relationship between autumn reservoir levels and NO2 winter day-ahead prices?
    Using ENTSO-E B31 data, build a simple regression and form a quantitative view on winter 2026/27
    given current reservoir drawdown. Does low hydro in September predict a winter premium, and by how much?
  </div>
</div>
<div style="background:#161b22;border:1px solid rgba(255,255,255,0.08);border-left:3px solid #4caf8f;border-radius:0 8px 8px 0;padding:16px 18px;">
  <div style="color:#4caf8f;font-size:0.68rem;text-transform:uppercase;letter-spacing:1px;font-weight:600;margin-bottom:6px;">Platform: Layer 2</div>
  <div style="color:#e6edf3;font-size:0.92rem;font-weight:600;margin-bottom:8px;">Nordic Price Decomposition</div>
  <div style="color:#8b949e;font-size:0.82rem;line-height:1.65;">
    Decompose NO2 day-ahead prices into hydro, wind, gas, and Continental components using
    multivariate regression on ENTSO-E generation and hydro data. Identify which factor
    is driving the marginal price on any given day. This is the analytical foundation for the HMM regime model.
  </div>
</div>
""", unsafe_allow_html=True)

with rp2:
    st.markdown("""
<div style="background:#161b22;border:1px solid rgba(255,255,255,0.08);border-left:3px solid #58a6ff;border-radius:0 8px 8px 0;padding:16px 18px;margin-bottom:10px;">
  <div style="color:#58a6ff;font-size:0.68rem;text-transform:uppercase;letter-spacing:1px;font-weight:600;margin-bottom:6px;">Independent Analysis</div>
  <div style="color:#e6edf3;font-size:0.92rem;font-weight:600;margin-bottom:8px;">Qatar LNG Diversion: Event Study</div>
  <div style="color:#8b949e;font-size:0.82rem;line-height:1.65;">
    Reconstruct the March 2026 Qatar LNG diversion using GIE ALSI sendout data, TTF prices,
    and the JKM-TTF spread. How quickly did the LNG sendout drop translate into a TTF price response?
    What was the lag? How long did it persist? A precise, data-driven account of one specific
    supply shock, using data already in this platform.
  </div>
</div>
<div style="background:#161b22;border:1px solid rgba(255,255,255,0.08);border-left:3px solid #58a6ff;border-radius:0 8px 8px 0;padding:16px 18px;">
  <div style="color:#58a6ff;font-size:0.68rem;text-transform:uppercase;letter-spacing:1px;font-weight:600;margin-bottom:6px;">Platform: Layer 2</div>
  <div style="color:#e6edf3;font-size:0.92rem;font-weight:600;margin-bottom:8px;">Merit Order and Supply Stack</div>
  <div style="color:#8b949e;font-size:0.82rem;line-height:1.65;">
    Build a simplified German merit order using ENTSO-E generation capacity data,
    fuel prices (coal, gas, carbon), and capacity factors. Visualise the supply stack
    and estimate where the marginal plant sits at current gas and coal prices.
    Shows how much of the stack is displaced when gas falls below a given threshold.
  </div>
</div>
""", unsafe_allow_html=True)

with rp3:
    st.markdown("""
<div style="background:#161b22;border:1px solid rgba(255,255,255,0.08);border-left:3px solid #d29922;border-radius:0 8px 8px 0;padding:16px 18px;margin-bottom:10px;">
  <div style="color:#d29922;font-size:0.68rem;text-transform:uppercase;letter-spacing:1px;font-weight:600;margin-bottom:6px;">Independent Analysis</div>
  <div style="color:#e6edf3;font-size:0.92rem;font-weight:600;margin-bottom:8px;">Backtested TTF Seasonal Injection Strategy</div>
  <div style="color:#8b949e;font-size:0.82rem;line-height:1.65;">
    Can a simple rules-based strategy (buy summer TTF, sell winter TTF when the seasonal spread
    exceeds storage cost) generate consistent returns historically? Backtest over 2019-2025
    using ICE settlement prices. Account for storage costs (~€4/MWh round-trip), transaction costs,
    and capital. Compare Sharpe ratio against a naive buy-and-hold benchmark.
  </div>
</div>
<div style="background:#161b22;border:1px solid rgba(255,255,255,0.08);border-left:3px solid #d29922;border-radius:0 8px 8px 0;padding:16px 18px;">
  <div style="color:#d29922;font-size:0.68rem;text-transform:uppercase;letter-spacing:1px;font-weight:600;margin-bottom:6px;">Platform: Layer 4</div>
  <div style="color:#e6edf3;font-size:0.92rem;font-weight:600;margin-bottom:8px;">FinBERT Energy Sentiment Signal</div>
  <div style="color:#8b949e;font-size:0.82rem;line-height:1.65;">
    Daily sentiment scoring of energy headlines (Reuters, Montel, Recharge) using FinBERT.
    Does news sentiment Granger-cause next-day TTF price changes? The March 2026 Hormuz
    escalation (a 40% TTF spike in three days) is a natural test case: was the sentiment
    shift detectable before prices fully adjusted?
  </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── About ──────────────────────────────────────────────────────────────────
st.markdown("#### About This Platform")
col_about, col_stack = st.columns([2, 1])
with col_about:
    st.markdown(
        "<div style='color:#8b949e;font-size:0.88rem;line-height:1.75;'>"
        "Built to demonstrate the intersection of energy market knowledge and data engineering. "
        "Each layer mirrors the analytical workflow of a physical power trader: "
        "monitoring live fundamentals, running quantitative supply-demand models, "
        "contextualising moves against geopolitical events, and applying machine learning "
        "to identify structural market regimes.<br><br>"
        "Data is sourced exclusively from public APIs (GIE, ENTSO-E, Nord Pool, ICE/Yahoo Finance). "
        "All models are built from scratch in Python."
        "</div>",
        unsafe_allow_html=True,
    )
with col_stack:
    st.markdown(
        "<div style='color:#8b949e;font-size:0.82rem;line-height:1.8;'>"
        "<strong style='color:#c9d1d9;'>Stack</strong><br>"
        "Python 3.11+ &nbsp;·&nbsp; Streamlit<br>"
        "Plotly &nbsp;·&nbsp; Pandas &nbsp;·&nbsp; NumPy<br>"
        "scikit-learn &nbsp;·&nbsp; statsmodels<br>"
        "yfinance &nbsp;·&nbsp; entsoe-py<br>"
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
