"""
Home: landing page of the European Gas & Power Market Intelligence Platform.
"""
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="European Gas & Power: Market Intelligence Platform")

from utils.helpers import apply_dark_theme, has_entsoe_key, has_agsi_key
from utils.globe import globe_link
from utils.theme import (
    ACCENTS, callout, chip, interconnector, page_footer, render, section_header, tap, ticker_strip,
)
from data.market_strip import fetch_market_strip
from data.spot_prices import fetch_dayahead

apply_dark_theme()


def _link_prices() -> tuple[dict, dict, str, str | None, str | None]:
    """Latest delivery day with prices at both ends of NordLink (NorNed when DE-LU is missing)."""
    df = fetch_dayahead(("NO2", "DE-LU", "NL"), 7)
    if df.empty:
        return {}, {}, "DE-LU", None, None
    days = sorted(df["date"].unique(), reverse=True)
    rows = {d: df[df["date"] == d].set_index("zone")["price_eur_mwh"] for d in days[:5]}
    # NordLink when DE-LU is published for one of the two newest days (it reaches ENTSO-E
    # after the Nordic prices); NorNed otherwise
    for zone, window in (("DE-LU", days[:2]), ("NL", days[:5])):
        for day in window:
            row = rows[day]
            if "NO2" in row.index and zone in row.index:
                return ({"NO2": float(row["NO2"])}, {zone: float(row[zone])}, zone,
                        day.strftime("%d %b %Y") if hasattr(day, "strftime") else str(day), None)
    return {}, {}, "DE-LU", None, None


_LINKS = {"DE-LU": ("Germany", "NordLink", "1,400 MW"), "NL": ("Netherlands", "NorNed", "700 MW")}


def _render_link(slot, left: dict, right: dict, zone: str, as_of: str | None) -> None:
    place, link, cap = _LINKS[zone]
    slot.markdown(
        interconnector(
            {"zone": "NO2", "place": "Southern Norway", "price": left.get("NO2")},
            {"zone": zone, "place": place, "price": right.get(zone)},
            link=link, capacity=cap, as_of=as_of,
        ),
        unsafe_allow_html=True,
    )


# ── Hero: the interconnector between the hydro system and the gas-fired system ──
with st.container(key="ptp_hero"):
    render(
        '<div class="ptp-hero">'
        '<div class="ptp-hero__title" role="heading" aria-level="1">'
        "European gas and power, read from the cable</div>"
        '<div class="ptp-hero__lead">Norway prices power off water in reservoirs; Germany prices it off '
        "gas, coal and wind. The interconnector between them is where the two systems meet, and this "
        "platform is built to explain the price on each end.</div>"
        '<div class="ptp-hero__body">Eight pages follow a trader\'s workflow: live fundamentals, '
        "quantitative models, macro and geopolitics, machine learning, ranked rich and cheap signals, "
        "portfolio risk, battery project economics and a 3D globe of the infrastructure.</div>"
        "</div>"
    )
    _ic_slot = st.empty()
    _render_link(_ic_slot, {}, {}, "DE-LU", None)
    with st.container(horizontal=True, gap="small", key="ptp_cta"):
        with st.container(width="content", key="ptp_cta_primary"):
            st.page_link("views/1_Live_Monitor.py", label="Open the Live Monitor",
                         icon=":material/monitoring:")
        st.page_link("views/5_Mispricing_Dashboard.py", label="See today's rich and cheap signals",
                     icon=":material/track_changes:")
        globe_link("europe", "Open the Energy Globe")

# ── Market snapshot (filled at the end of the run so the page renders first) ─────
render(
    '<div class="ptp-strip-head"><div class="ptp-strip-head__title">Market snapshot</div>'
    '<div class="ptp-strip-head__meta">Delayed quotes, refreshed every 15 minutes</div></div>'
)
_strip_slot = st.empty()
_strip_slot.markdown(
    ticker_strip(
        [{"symbol": s, "sub": "Loading", "value": None} for s in
         ("TTF", "JKM", "Brent", "EUA", "EUR/USD", "NO2", "DE-LU")]
    ),
    unsafe_allow_html=True,
)

# ── Data sources ───────────────────────────────────────────────────────────────
render(
    '<div class="ptp-status-row"><span class="ptp-status-row__label">Data sources</span>'
    + chip("GIE AGSI+ and ALSI", "up" if has_agsi_key() else "warn", dot=True,
           title="Connected" if has_agsi_key() else "AGSI_API_KEY not set")
    + chip("ENTSO-E Transparency Platform", "up" if has_entsoe_key() else "warn", dot=True,
           title="Connected" if has_entsoe_key() else "ENTSOE_API_KEY not set")
    + chip("Nord Pool", "up", dot=True, title="Keyless")
    + chip("Yahoo Finance", "up", dot=True, title="Keyless")
    + "</div>"
)
render(callout(
    "The first visit after a restart fetches live data from every source and can take a minute or two. "
    "After that, pages load from a shared cache that refreshes every hour.",
    "neutral",
))

# ── Single-line diagram of the platform ─────────────────────────────────────────
section_header(
    "How the platform is wired",
    "Each page taps into the same data bus: observation first, then models, then signals and "
    "risk, then a project that uses them.",
)

_GROUPS = [
    ("Observe the market", [
        dict(key="l1", page="views/1_Live_Monitor.py", link="Open Live Monitor",
             title="Live Monitor", stat="10 panels",
             desc="EU and German gas storage against five-year bands, LNG send-out, TTF and its forward "
                  "strip, Nordic, Continental and Italian day-ahead prices, Norway's cross-border flows, "
                  "German coal burn, solar cannibalisation and hydro reservoirs."),
        dict(key="l3", page="views/3_Macro_Signals.py", link="Open Macro Signals",
             title="Macro Signals", stat="5 panels",
             desc="TTF with annotated market events, the EU supply mix, gas-to-agriculture spillover, a "
                  "seven-asset correlation grid and the TTF versus JKM inter-basin LNG spread."),
        dict(key="globe", page="views/8_Energy_Globe.py", link="Open Energy Globe",
             title="Energy Globe", stat="4 layers",
             desc="A 3D globe of the physical system: Norwegian shelf facilities and export pipelines, "
                  "European LNG terminals sized by live send-out, and the shipping chokepoints."),
    ]),
    ("Model it", [
        dict(key="l2", page="views/2_Quant_Analysis.py", link="Open Quant Analysis",
             title="Quant Analysis", stat="14 models",
             desc="Storage refill Monte Carlo, gas-to-power regression, the German merit order on live "
                  "load and renewables, cointegration with rolling stability and TTF forward curve PCA."),
        dict(key="l4", page="views/4_ML_Models.py", link="Open ML Models",
             title="ML Models", stat="3 models",
             desc="An LSTM next-day price forecaster judged against a naive baseline, a four-state regime "
                  "model with a volatility overlay, and FinBERT news sentiment with Granger tests."),
    ]),
    ("Act on it", [
        dict(key="l5", page="views/5_Mispricing_Dashboard.py", link="Open Mispricing Dashboard",
             title="Mispricing Dashboard", stat="8 signals",
             desc="Rich and cheap signals from the other pages, ranked by how far today sits in its own "
                  "history, with direction and confidence."),
        dict(key="risk", page="views/6_Risk_Dashboard.py", link="Open Risk Dashboard",
             title="Risk Dashboard", stat="6 signals",
             desc="A demo portfolio built from the signals: equity curves, Sharpe, 1-day VaR and "
                  "Expected Shortfall, stress scenarios and rolling correlations."),
        dict(key="bess", page="views/7_BESS_Optimiser.py", link="Open BESS Optimiser",
             title="BESS Optimiser", stat="4 revenue streams",
             desc="Battery economics in the Nordic market: day-ahead arbitrage, FCR-N, FCR-D and intraday "
                  "revenue, cash flow, NPV, IRR, payback and sensitivities."),
    ]),
]

for _i, (_group, _taps) in enumerate(_GROUPS):
    render(f'<div class="ptp-sld-group">{_group}</div>')
    with st.container(key=f"ptp_bus_{_i}"):
        for _t in _taps:
            with st.container(key=f"ptp_tap_{_t['key']}"):
                render(tap(_t["title"], _t["desc"], _t["stat"], accent=ACCENTS[_t["key"]]))
                st.page_link(_t["page"], label=_t["link"])

# ── Energy Globe views ─────────────────────────────────────────────────────────
section_header(
    "Jump into the globe",
    "Preset views of the physical system behind the prices.",
)
with st.container(horizontal=True, gap="small", key="ptp_globe_links"):
    globe_link("europe", "European gas infrastructure")
    globe_link("north_sea", "Norwegian shelf and pipelines")
    globe_link("lng_europe", "European LNG terminals")
    globe_link("lng_routes", "LNG shipping chokepoints")
    globe_link("hormuz", "Strait of Hormuz")

# ── About ──────────────────────────────────────────────────────────────────────
with st.expander("About this platform", expanded=False):
    render(
        "<div class='ptp-about'>"
        "Built by <strong>Tobias Gudbjerg</strong>, Equity Sales Trader at Fearnley Securities in Oslo, "
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
            "scikit-learn, statsmodels, PyTorch<br>"
            "hmmlearn, Hugging Face Transformers (FinBERT)"
            "</div>"
        )
    with _ac2:
        render(
            "<div class='ptp-about-stack'>"
            "<div class='ptp-about-stack__title'>Data and infrastructure</div>"
            "Python, Streamlit, Plotly, pandas, NumPy, yfinance, entsoe-py<br>"
            "Hosted on Hugging Face Spaces; tested with pytest and GitHub Actions"
            "</div>"
        )

page_footer([
    ("Gas storage and LNG", "GIE AGSI+/ALSI", "https://agsi.gie.eu"),
    ("Power data", "ENTSO-E Transparency Platform", "https://transparency.entsoe.eu"),
    ("Day-ahead prices", "ENTSO-E and Nord Pool", "https://www.nordpoolgroup.com"),
    ("TTF", "Yahoo Finance / ICE", "https://finance.yahoo.com"),
])

# ── Fill the interconnector and the market snapshot (cached) ──────────────────
_left, _right, _zone, _as_of, _ = _link_prices()
if _left and _right:
    _render_link(_ic_slot, _left, _right, _zone, _as_of)

_strip = fetch_market_strip()
_strip_slot.markdown(
    ticker_strip(
        _strip["items"],
        note=f"Day-over-day change. Futures, carbon and FX: last daily bar via Yahoo Finance. "
             f"Power: day-ahead daily average (ENTSO-E, Nord Pool), change in EUR/MWh. "
             f"Last fetch {_strip['fetched_at']:%Y-%m-%d %H:%M} UTC.",
    ),
    unsafe_allow_html=True,
)
