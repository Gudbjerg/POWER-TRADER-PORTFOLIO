"""
Layer 1: European Power and Gas Live Market Monitor
"""
import time as _time
import streamlit as st
from dotenv import load_dotenv

_PAGE_T0 = _time.perf_counter()

load_dotenv()

st.set_page_config(
    page_title="Live Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Imports ────────────────────────────────────────────────────────────────
from data.gas_storage     import get_storage_data, build_seasonal_bands, calendar_doy, EU_TARGET_SENTENCE
from data.prices          import get_ttf_data
from data.power_flows     import get_flow_data
from data.spot_prices     import get_spot_price_data, fetch_spot_prices
from data.solar           import get_solar_data
from data.lng_terminals   import get_lng_data
from data.hydro           import get_hydro_data
from data.forward_curve   import get_forward_curve_data
from data.generation      import get_generation_data
from components.storage_chart        import render_storage_chart
from components.prices_chart         import render_ttf_chart
from components.flows_chart          import render_flows_chart
from components.spot_prices_chart    import render_spot_prices_chart
from components.solar_chart          import render_solar_chart
from components.lng_chart            import render_lng_chart
from components.hydro_chart          import render_hydro_chart
from components.forward_curve_chart  import render_forward_curve_chart
from components.coal_chart           import render_coal_chart
from utils.globe     import globe_link
from utils.helpers   import (
    has_entsoe_key, has_agsi_key, apply_dark_theme,
    pill, commentary, kpi_card, delta_span,
)
from utils.theme     import ACCENTS, CHART, PLOTLY_TEMPLATE, page_header, page_footer
from utils.scenarios import (
    storage_status, ttf_status, spread_status, lng_status, market_summary,
    storage_refill_pace,
)
import pandas as pd
from datetime import date as _date, timedelta as _timedelta
from config.settings import (
    INTERCONNECTOR_CAPACITY_MW,
    INTERCONNECTOR_UTIL_HIGH_PCT, INTERCONNECTOR_UTIL_MED_PCT,
    SPREAD_CHART_REF_EUR,
)

apply_dark_theme()


@st.cache_data(ttl=3600, show_spinner=False)
def _get_spot_history_l1():
    return fetch_spot_prices(days=365)


_NO_ZONES_ZONAL = ["NO1", "NO2", "NO3", "NO4", "NO5", "SYS"]
_NO_ZONE_LABELS = {
    "NO1": "NO1 (Oslo)",
    "NO2": "NO2 (Kristiansand)",
    "NO3": "NO3 (Molde)",
    "NO4": "NO4 (Tromsø)",
    "NO5": "NO5 (Bergen)",
    "SYS": "SYS (System)",
}
_NO_ZONE_COLORS = {
    "NO1": "#2f45c6", "NO2": "#2f7a47", "NO3": "#a86a12",
    "NO4": "#c2611a", "NO5": "#6e4a9e", "SYS": "#c2391f",
}


def _fetch_norwegian_zonal(days: int = 90) -> pd.DataFrame:
    """Day-ahead daily averages for NO1 to NO5 (ENTSO-E, Nord Pool fallback) and SYS (Nord Pool)."""
    from data.spot_prices import fetch_dayahead
    return fetch_dayahead(tuple(_NO_ZONES_ZONAL), days)


# ── Load all data (cached) ─────────────────────────────────────────────────
with st.spinner(""):
    storage    = get_storage_data()
    ttf        = get_ttf_data()
    flows      = get_flow_data()
    spots      = get_spot_price_data()
    solar      = get_solar_data()
    lng        = get_lng_data()
    hydro      = get_hydro_data()
    generation = get_generation_data()

_ttf_spot = float(ttf["prices"]["price"].iloc[-1]) if not ttf["prices"].empty else 0.0
fwd = get_forward_curve_data(_ttf_spot)

eu_df      = storage["europe"]
de_df      = storage["germany"]
ttf_df     = ttf["prices"]
spot_df    = spots["prices"]
lng_totals = lng.get("totals", pd.DataFrame())


# ── Derive KPI values ──────────────────────────────────────────────────────
def _latest_storage(df):
    if df.empty:
        return None, None, None, None
    row = df.iloc[-1]
    doy = calendar_doy(row["gasDayStart"])
    bands = build_seasonal_bands(df)
    if bands.empty:
        return float(row["full"]), None, None, doy
    b = bands[bands["day_of_year"] == doy]
    mn  = float(b["min"].values[0])  if not b.empty else None
    avg = float(b["mean"].values[0]) if not b.empty else None
    return float(row["full"]), mn, avg, doy


eu_pct, eu_min, eu_mean, _ = _latest_storage(eu_df)
de_pct, de_min, de_mean, _ = _latest_storage(de_df)
ttf_price     = float(ttf_df["price"].iloc[-1]) if not ttf_df.empty else None
ttf_ma30      = float(ttf_df["ma30"].iloc[-1])  if not ttf_df.empty else None
ttf_ma90      = float(ttf_df["ma90"].iloc[-1])  if not ttf_df.empty else None
ttf_spike     = ttf["spike"]
ttf_spike_pct = ttf["spike_pct"]

# Nordic-Continental spread
nordic_avg = continental_avg = None
continental_label = "Continental"
if not spot_df.empty:
    # Latest delivery day with both a Nordic and a Continental price (tomorrow's auction
    # result can arrive from one source before the other)
    for latest_date in sorted(spot_df["date"].unique(), reverse=True):
        latest = spot_df[spot_df["date"] == latest_date].set_index("zone")["price_eur_mwh"]
        n_zones = [z for z in ["NO1", "NO2"] if z in latest.index]
        c_zones = [z for z in ["DE-LU", "NL"] if z in latest.index]
        if n_zones and c_zones:
            nordic_avg      = float(latest[n_zones].mean())
            continental_avg = float(latest[c_zones].mean())
            continental_label = "/".join(c_zones)
            break

# LNG KPI
lng_total_today = None
lng_wow  = lng.get("wow_change_pct")
lng_alert = lng.get("alert", False)
if not lng_totals.empty and "total_sendout" in lng_totals.columns:
    lng_total_today = float(lng_totals["total_sendout"].iloc[-1])


# ── Scenarios ──────────────────────────────────────────────────────────────
eu_status, eu_headline, eu_detail = (
    storage_status(eu_pct, eu_min, eu_mean, "EU") if eu_pct is not None else ("ok", "", "")
)
de_status, de_headline, de_detail = (
    storage_status(de_pct, de_min, de_mean, "Germany") if de_pct is not None else ("ok", "", "")
)
ttf_stat, ttf_headline, ttf_detail = (
    ttf_status(ttf_price, ttf_ma30, ttf_ma90, ttf_spike, ttf_spike_pct)
    if ttf_price is not None else ("ok", "", "")
)
sp_stat, sp_headline, sp_detail = (
    spread_status(nordic_avg, continental_avg, continental_label)
    if nordic_avg is not None and continental_avg is not None else ("ok", "No spot data", "")
)
lng_stat, lng_headline, lng_detail = lng_status(lng_wow)


# ── Header ─────────────────────────────────────────────────────────────────
_hdr_col, _btn_col = st.columns([5, 1], vertical_alignment="center")
with _hdr_col:
    page_header(
        "Layer 1", "Live Market Monitor",
        "Real-time European power and gas data, refreshed hourly.",
        sources=["GIE AGSI+/ALSI", "Nord Pool", "ENTSO-E", "ICE/Yahoo Finance"],
        accent=ACCENTS["l1"],
    )
with _btn_col:
    with st.container(horizontal=True, horizontal_alignment="right"):
        if st.button("Refresh data", icon=":material/refresh:"):
            st.cache_data.clear()
            st.rerun()

missing = []
if not has_agsi_key():   missing.append("AGSI_API_KEY (agsi.gie.eu)")
if not has_entsoe_key(): missing.append("ENTSOE_API_KEY (transparency.entsoe.eu)")
if missing:
    st.info("Add to .env to unlock all panels: " + " and ".join(missing))


# ── KPI bar ────────────────────────────────────────────────────────────────
# GIE-backed cards say why a figure is missing: no key, or a fetch that failed and is
# retried within GIE_FAILURE_TTL_S (data/gie.py).
_gie_note = "AGSI key required" if not has_agsi_key() else "GIE unavailable, retrying shortly"
_lng_unavailable = lng.get("unavailable") or []

k1, k2, k3, k4, k5, k6 = st.columns(6)

with k1:
    if eu_pct is not None:
        color = "red" if eu_status == "critical" else ("amber" if eu_status == "warn" else "green")
        if eu_mean is not None and eu_min is not None:
            d = delta_span(f"{eu_pct - eu_mean:+.1f}pp vs 5yr avg, min {eu_min:.1f}%", color)
        else:
            d = delta_span(f"{eu_pct:.1f}%, bands unavailable", color)
        st.markdown(kpi_card("EU Gas Storage", f"{eu_pct:.1f}%", d), unsafe_allow_html=True)
    else:
        st.markdown(kpi_card("EU Gas Storage", "n/a", delta_span(_gie_note, "amber")), unsafe_allow_html=True)

with k2:
    if de_pct is not None:
        color = "red" if de_status == "critical" else ("amber" if de_status == "warn" else "green")
        if de_mean is not None and de_min is not None:
            d = delta_span(f"{de_pct - de_mean:+.1f}pp vs 5yr avg, min {de_min:.1f}%", color)
        else:
            d = delta_span(f"{de_pct:.1f}%, bands unavailable", color)
        st.markdown(kpi_card("DE Gas Storage", f"{de_pct:.1f}%", d), unsafe_allow_html=True)
    else:
        st.markdown(kpi_card("DE Gas Storage", "n/a", delta_span(_gie_note, "amber")), unsafe_allow_html=True)

with k3:
    if ttf_price is not None:
        color = "red" if ttf_stat in ("critical", "warn") else "green"
        d = delta_span(f"{(ttf_price / ttf_ma30 - 1) * 100:+.1f}% vs 30d MA, 90d avg €{ttf_ma90:.1f}", color)
        st.markdown(kpi_card("TTF (EUR/MWh)", f"€{ttf_price:.2f}", d), unsafe_allow_html=True)
    else:
        st.markdown(kpi_card("TTF (EUR/MWh)", "n/a", ""), unsafe_allow_html=True)

with k4:
    if nordic_avg is not None:
        d = delta_span(f"NO1/NO2 €{nordic_avg:.0f}, {continental_label} €{continental_avg:.0f}", "blue")
        st.markdown(kpi_card("Nordic Spot (EUR/MWh)", f"€{nordic_avg:.0f}", d), unsafe_allow_html=True)
    else:
        st.markdown(kpi_card("Nordic Spot (EUR/MWh)", "n/a", ""), unsafe_allow_html=True)

with k5:
    if nordic_avg is not None and continental_avg is not None:
        spread = continental_avg - nordic_avg
        color = "amber" if abs(spread) > 15 else "green"
        if spread > 0:
            sub = f"{continental_label} premium, export incentive"
        elif spread < 0:
            sub = "Nordic premium, import signal"
        else:
            sub = "markets aligned"
        d = delta_span(sub, color)
        st.markdown(kpi_card("N-C Spread (EUR/MWh)", f"€{spread:+.0f}", d), unsafe_allow_html=True)
    else:
        st.markdown(kpi_card("N-C Spread (EUR/MWh)", "n/a", ""), unsafe_allow_html=True)

with k6:
    if _lng_unavailable:
        # A total without one of the four countries is not the NW European total.
        _missing = ", ".join(_lng_unavailable)
        st.markdown(kpi_card("NW EU LNG Sendout", "n/a",
                             delta_span(f"GIE ALSI unavailable for {_missing}, retrying shortly", "amber")),
                    unsafe_allow_html=True)
    elif lng_total_today is not None:
        color = "red" if lng_alert else ("amber" if lng_wow is not None and lng_wow < -7 else "green")
        d = delta_span(f"{lng_wow:+.1f}% vs prior 7d" if lng_wow is not None else "calculating", color)
        st.markdown(kpi_card("NW EU LNG Sendout", f"{lng_total_today:,.0f} GWh/d", d), unsafe_allow_html=True)
    else:
        st.markdown(kpi_card("NW EU LNG Sendout", "n/a", delta_span(_gie_note, "amber")), unsafe_allow_html=True)

st.divider()


# ── Tabs ───────────────────────────────────────────────────────────────────
tab_overview, tab_gas, tab_lng, tab_prices, tab_flows, tab_generation, tab_solar, tab_hydro = st.tabs([
    "Overview",
    "Gas Storage",
    "LNG Terminals",
    "Prices",
    "Power Flows",
    "Generation",
    "Solar",
    "Hydro Reservoirs",
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.markdown("### Market Summary")

    if eu_pct is not None and ttf_price is not None:
        summary = market_summary(
            eu_pct, eu_min, eu_mean,
            de_pct, de_min, de_mean,
            ttf_price, ttf_ma30,
            nordic_avg, continental_avg, continental_label,
        )
        st.markdown(commentary(summary, eu_status), unsafe_allow_html=True)

    st.markdown("### Signal Overview")
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown("**EU Gas Storage**")
        st.markdown(f"{pill(eu_status)} &nbsp; {eu_headline}", unsafe_allow_html=True)
        if eu_detail:
            with st.expander("Analysis", expanded=True):
                st.markdown(eu_detail)

        st.markdown("**Germany Gas Storage**")
        st.markdown(f"{pill(de_status)} &nbsp; {de_headline}", unsafe_allow_html=True)
        if de_detail:
            with st.expander("Analysis", expanded=True):
                st.markdown(de_detail)

        st.markdown("**NW EU LNG Sendout**")
        st.markdown(f"{pill(lng_stat)} &nbsp; {lng_headline}", unsafe_allow_html=True)
        if lng_detail:
            with st.expander("Analysis", expanded=True):
                st.markdown(lng_detail)

    with r1c2:
        st.markdown("**TTF Gas Price**")
        st.markdown(f"{pill(ttf_stat)} &nbsp; {ttf_headline}", unsafe_allow_html=True)
        if ttf_detail:
            with st.expander("Analysis", expanded=True):
                st.markdown(ttf_detail)

        st.markdown("**Nordic-Continental Spread**")
        st.markdown(f"{pill(sp_stat)} &nbsp; {sp_headline}", unsafe_allow_html=True)
        if sp_detail:
            with st.expander("Analysis", expanded=True):
                st.markdown(sp_detail)

    st.divider()
    st.markdown("### Context")

    _today = _date.today()
    _nov1  = _date(_today.year if _today.month < 11 else _today.year + 1, 11, 1)
    _days_to_nov = (_nov1 - _today).days

    w1, w2, w3 = st.columns(3)
    with w1:
        st.markdown("**Storage refill pace**")
        if eu_pct is not None:
            _pace_info = storage_refill_pace(eu_pct, build_seasonal_bands(eu_df), _today)
            if ttf_price is not None and ttf_price > 35:
                _lng_note = f"TTF at €{ttf_price:.0f}/MWh supports LNG cargo diversion to Europe over Asia."
            elif ttf_price is not None and ttf_price > 20:
                _lng_note = f"TTF at €{ttf_price:.0f}/MWh provides moderate LNG import incentive."
            elif ttf_price is not None:
                _lng_note = f"TTF at €{ttf_price:.0f}/MWh; LNG import economics are weak at current prices."
            else:
                _lng_note = ""
            if _pace_info["verdict"] == "met":
                _pace_text = (
                    f"With EU storage at {eu_pct:.1f}%, the target is already met "
                    f"({_pace_info['days']} days before 1 November). "
                )
            elif _pace_info["norm_pp"] is not None:
                _pace_text = (
                    f"With EU storage at {eu_pct:.1f}% and {_pace_info['days']} days remaining, "
                    f"the required net injection is approximately {_pace_info['required_pp']:.0f}pp "
                    f"({_pace_info['required_per_day']:.2f}pp/day), {_pace_info['verdict']} the "
                    f"5-year average for the same window ({max(_pace_info['norm_pp'], 0.0):.0f}pp, "
                    f"{max(_pace_info['norm_per_day'], 0.0):.2f}pp/day). "
                )
            else:
                _pace_text = (
                    f"With EU storage at {eu_pct:.1f}% and {_pace_info['days']} days remaining, "
                    f"the required net injection is approximately {_pace_info['required_pp']:.0f}pp "
                    f"({_pace_info['required_per_day']:.2f}pp/day); the 5-year comparison is unavailable. "
                )
            st.caption(
                EU_TARGET_SENTENCE + " Pace is measured to 1 November, the usual end of injections. "
                + _pace_text
                + _lng_note
            )
        else:
            _storage_hint = (
                "Add AGSI_API_KEY to see current storage level." if not has_agsi_key()
                else "Current storage level unavailable: GIE AGSI+ did not answer; retrying shortly."
            )
            st.caption(
                f"{EU_TARGET_SENTENCE} {_days_to_nov} days to 1 November. {_storage_hint}"
            )
    with w2:
        st.markdown("**LNG supply and Iran risk**")
        st.caption(
            "European LNG imports are the primary swing supply source. "
            "US tariff uncertainty and any disruption to "
            "Hormuz Strait LNG transit would tighten European supply balances. "
            "Monitor Sabine Pass, Freeport, and Qatar loadings."
        )
    with w3:
        st.markdown("**Norwegian hydro export incentive**")
        if nordic_avg is not None and continental_avg is not None:
            spread = continental_avg - nordic_avg
            if spread > 5:
                st.caption(
                    f"{continental_label} at €{continental_avg:.0f}/MWh versus Nordic €{nordic_avg:.0f}/MWh "
                    f"(+€{spread:.0f}/MWh Continental premium). "
                    "Strong incentive to export Norwegian hydro via NordLink, NorNed, and NSN. "
                    "Norwegian reservoir levels (ENTSO-E B31 data) determine the upper bound on export capacity."
                )
            elif spread < -5:
                st.caption(
                    f"Nordic at €{nordic_avg:.0f}/MWh versus {continental_label} €{continental_avg:.0f}/MWh "
                    f"(+€{abs(spread):.0f}/MWh Nordic premium). "
                    "Reversed spread: Norway may be importing from the Continent. "
                    "Check reservoir levels and interconnector availability."
                )
            else:
                st.caption(
                    f"Nordic €{nordic_avg:.0f}/MWh and {continental_label} €{continental_avg:.0f}/MWh broadly aligned "
                    f"(spread: €{spread:+.0f}/MWh). No strong directional flow incentive."
                )
        else:
            st.caption("Spot price data required to calculate export incentive.")

    st.divider()
    entsoe_note = "Awaiting API key" if not has_entsoe_key() else "Active"
    st.caption(
        f"ENTSO-E panels (Flows, Solar): {entsoe_note}. "
        f"Storage updated: {storage['fetched_at'].strftime('%Y-%m-%d %H:%M UTC')}. "
        f"Prices updated: {ttf['fetched_at'].strftime('%Y-%m-%d %H:%M UTC')}."
    )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: GAS STORAGE
# ═══════════════════════════════════════════════════════════════════════════
with tab_gas:
    c1, c2 = st.columns(2)
    with c1:
        render_storage_chart(storage, region="europe")
        if eu_detail:
            st.markdown(commentary(eu_detail, eu_status), unsafe_allow_html=True)
    with c2:
        render_storage_chart(storage, region="germany")
        if de_detail:
            st.markdown(commentary(de_detail, de_status), unsafe_allow_html=True)

    with st.expander("How to read this chart", expanded=True):
        st.markdown("""
        **Red line:** current year's storage fill level as a percentage of working gas volume.

        **Shaded band:** range between the 5-year historical minimum and maximum for each calendar date.

        **Dotted line:** 5-year historical average fill level.

        **Dashed horizontal line and tinted window:** EU filling target of 90%, to be reached on any day between 1 October and 1 December.

        A current level below the shaded band indicates storage has not been this low on this date in any of the preceding five years. The 2022 emergency regulation (EU 2022/1032) required 90% by 1 November; the 2025 amendment turned this into a window from 1 October to 1 December, allows deviations of up to 10 percentage points in difficult market conditions and extends the rules to the end of 2027. Germany's storage capacity (approximately 24 bcm) represents roughly 25% of its annual gas consumption.
        """)

    with st.expander("Key supply and demand risks", expanded=True):
        st.markdown("""
        **Supply side risks:**
        - Norwegian pipeline maintenance (Asgard, Troll): scheduled summer outages reduce export capacity
        - LNG: US export volumes, Qatari loadings, Hormuz Strait transit risk
        - Russian pipeline residual volumes via TurkStream

        **Demand side risks:**
        - Colder-than-seasonal spring temperatures accelerating withdrawal
        - Industrial gas demand recovery in Germany (chemicals, fertiliser production)

        **Injection season benchmarks:**
        - April 1 reference level: approximately 28%
        - EU target: 90% on any day between 1 October and 1 December (up to 10 pp lower in difficult market conditions)
        - Historical average injection: approximately 52 percentage points over the April to October period
        """)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: LNG TERMINALS
# ═══════════════════════════════════════════════════════════════════════════
with tab_lng:
    render_lng_chart(lng)
    globe_link("lng_europe", "LNG terminals on the 3D globe")

    # ── 3B: TTF vs JKM spread card (roadmap 3B) ──────────────────────────────
    # Keyless inter-basin price signal next to the send-out chart; the full
    # history and methodology live in Layer 3, "TTF vs JKM" tab.
    from data.lng_prices import load_ttf_jkm as _jkm_load
    from models.lng_arbitrage import spread_summary as _jkm_summary, ordinal as _jkm_ordinal
    import math as _jkm_math

    _jkm_l1 = _jkm_load()
    _jkm_l1_df = _jkm_l1.get("frame", pd.DataFrame())
    _jkm_c1, _jkm_c2 = st.columns([1, 3])
    if _jkm_l1_df.empty:
        with _jkm_c1:
            st.markdown(
                kpi_card("TTF minus JKM", "n/a", delta_span("JKM=F / EURUSD=X unavailable", "amber")),
                unsafe_allow_html=True,
            )
    else:
        _jkm_l1_s = _jkm_summary(_jkm_l1_df)
        _jkm_l1_pct = (
            f" · {_jkm_ordinal(round(_jkm_l1_s['percentile']))} percentile (1y)"
            if _jkm_math.isfinite(_jkm_l1_s["percentile"]) else ""
        )
        with _jkm_c1:
            st.markdown(
                kpi_card(
                    "TTF minus JKM",
                    f"€{_jkm_l1_s['spread']:+.2f}/MWh",
                    delta_span(f"{_jkm_l1_s['regime']}{_jkm_l1_pct}",
                               "green" if _jkm_l1_s["spread"] > 0 else "red"),
                ),
                unsafe_allow_html=True,
            )
        with _jkm_c2:
            _jkm_l1_dir = (
                "TTF above JKM favours deliveries into North-West European terminals, supporting LNG send-out."
                if _jkm_l1_s["spread"] > 0 else
                "JKM above TTF tends to pull flexible cargoes towards Asia, a headwind for NW European LNG send-out."
            )
            st.caption(
                f"Inter-basin price signal for flexible LNG cargoes (settlement {_jkm_l1_s['latest_date']:%d %b %Y}; "
                f"TTF €{_jkm_l1_s['ttf']:.2f}/MWh, JKM €{_jkm_l1_s['jkm_eur_mwh']:.2f}/MWh converted from "
                f"${_jkm_l1_s['jkm_usd_mmbtu']:.3f}/MMBtu). {_jkm_l1_dir} Diversion pays only when the gap "
                "exceeds the shipping cost differential. History and methodology: Layer 3, TTF vs JKM tab."
            )
    # ── end 3B: TTF vs JKM spread card ───────────────────────────────────────

    with st.expander("Understanding LNG sendout", expanded=True):
        st.markdown("""
        **Definition:** LNG (liquefied natural gas) arrives by tanker at import terminals, where it is regasified and injected into the pipeline grid. The sendout rate (GWh/day, as published by GIE ALSI) measures how much gas is delivered to the European grid from each terminal.

        **Market significance:** LNG is Europe's swing supply source. Cargoes can be diverted between Europe and Asia depending on relative benchmark prices (TTF versus JKM). When European TTF prices are elevated relative to Asian LNG, cargoes flow to Europe, supporting storage injection. A sustained decline in sendout tightens the European gas balance.

        **Key terminals monitored:**

        | Terminal | Country | Nameplate capacity |
        |----------|---------|-------------------|
        | Zeebrugge | Belgium | 9 bcm/yr |
        | Gate LNG | Netherlands (Rotterdam) | 12 bcm/yr |
        | Dunkirk | France | 13 bcm/yr |
        | Montoir-de-Bretagne | France | 10 bcm/yr |

        Great Britain's South Hook (21 bcm/yr) and Dragon LNG (5 bcm/yr) are the largest NW European import cluster, but they have not reported to GIE ALSI since the end of 2020 and are therefore not in the totals.

        **Alert threshold:** A week-on-week decline exceeding 15% in aggregate sendout is flagged as a potential supply disruption signal. US flex LNG cargoes (Sabine Pass, Freeport, Corpus Christi) respond to the TTF versus Henry Hub price differential.
        """)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4: PRICES
# ═══════════════════════════════════════════════════════════════════════════
with tab_prices:
    render_ttf_chart(ttf)
    if ttf_detail:
        st.markdown(commentary(ttf_detail, ttf_stat), unsafe_allow_html=True)

    st.divider()
    render_forward_curve_chart(fwd)

    with st.expander("Understanding the seasonal forward curve", expanded=False):
        st.markdown("""
        **What this shows:** An 18-month implied forward strip for TTF natural gas, constructed by anchoring
        today's spot price to a three-year historical seasonal index. Each bar represents the implied price
        for that delivery month, scaled by how that month has historically traded relative to the annual average.
        Error bars show the P25-P75 historical range for each calendar month.

        **Winter/Summer spread:** Natural gas is structurally more expensive in winter (October-March) than
        summer (April-September) due to heating demand. The winter premium over summer typically ranges from
        €5-20/MWh in a balanced market, widening sharply when storage is below target heading into autumn.

        **How traders use the forward curve:**
        - **Injection economics:** If summer prices are low relative to winter, there is an economic
          incentive to buy gas in summer (inject into storage) and sell it in winter (withdraw).
          The spread must exceed storage costs (approximately €3-6/MWh round-trip) to be profitable.
        - **Cal-year hedging:** Physical traders hedge annual supply obligations using calendar strips
          (Cal 26, Cal 27). The Cal price reflects the full-year average.
        - **Structure signals:** A steep contango (summer cheap, winter expensive) reflects expected
          tightness in winter supply. Backwardation (summer above winter) is unusual and signals
          near-term supply stress.

        **Important limitation:** This curve is derived from historical seasonal patterns, not live OTC
        broker quotes or exchange settlement prices. For actual forward hedging, use EEX or ICE settlement
        prices. This chart is intended to illustrate the structural shape of the curve and seasonal dynamics.
        """)

    st.divider()
    render_spot_prices_chart(spots)
    if sp_detail:
        st.markdown(commentary(sp_detail, sp_stat), unsafe_allow_html=True)

    # ── B6: Nordic-Continental spread history ─────────────────────────────────
    st.divider()
    st.markdown("#### Nordic–Continental Spread History (NO2 – NL)")
    st.caption(
        "NL day-ahead minus NO2 day-ahead (EUR/MWh). Positive = Continental premium: "
        "Norway has an export incentive and NordLink/NorNed flow toward the Continent."
    )

    _spread_df = _get_spot_history_l1()
    if not _spread_df.empty:
        _no2  = _spread_df[_spread_df["zone"] == "NO2"][["date", "price_eur_mwh"]].rename(
            columns={"price_eur_mwh": "no2"}
        )
        _nl   = _spread_df[_spread_df["zone"] == "NL"][["date", "price_eur_mwh"]].rename(
            columns={"price_eur_mwh": "nl"}
        )
        _sp   = _no2.merge(_nl, on="date", how="inner").sort_values("date")
        _sp["spread"] = _sp["nl"] - _sp["no2"]   # NL - NO2: positive = export incentive

        if not _sp.empty:
            _current_spread = float(_sp["spread"].iloc[-1])
            _sp_color = "red" if _current_spread > SPREAD_CHART_REF_EUR else ("green" if _current_spread < -SPREAD_CHART_REF_EUR else "blue")
            _sp_label = (
                "strong NL premium, NordLink near capacity" if _current_spread > SPREAD_CHART_REF_EUR
                else "Nordic premium (atypical, check reservoirs)" if _current_spread < -SPREAD_CHART_REF_EUR
                else "balanced"
            )
            st.markdown(
                kpi_card(
                    "Current NL–NO2 spread",
                    f"€{_current_spread:+.1f}/MWh",
                    delta_span(_sp_label, _sp_color),
                ),
                unsafe_allow_html=True,
            )

            import plotly.graph_objects as _go
            _fig_sp = _go.Figure()
            _fig_sp.add_trace(_go.Scatter(
                x=_sp["date"], y=_sp["spread"],
                mode="lines",
                name="NL − NO2 spread",
                line=dict(color="#2f45c6", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(47,69,198,0.08)",
                hovertemplate="NL−NO2: €%{y:.1f}/MWh<extra></extra>",
            ))
            _fig_sp.add_hline(y=0, line=dict(color=CHART["zero"], width=1))
            _fig_sp.add_hline(
                y=SPREAD_CHART_REF_EUR, line=dict(color="#a86a12", width=1, dash="dot"),
                annotation_text=f"  +€{SPREAD_CHART_REF_EUR:.0f} (NordLink historically >90% utilised)",
                annotation_font=dict(color="#a86a12", size=10),
                annotation_position="right",
            )
            _fig_sp.add_hline(
                y=-SPREAD_CHART_REF_EUR, line=dict(color="#2f7a47", width=1, dash="dot"),
                annotation_text=f"  −€{SPREAD_CHART_REF_EUR:.0f} (Nordic premium, atypical)",
                annotation_font=dict(color="#2f7a47", size=10),
                annotation_position="right",
            )
            _fig_sp.update_layout(
                template=PLOTLY_TEMPLATE,
                paper_bgcolor=CHART["bg"],
                plot_bgcolor=CHART["panel_bg"],
                height=220,
                margin=dict(l=50, r=140, t=10, b=35),
                xaxis=dict(gridcolor=CHART["grid"], tickfont=dict(size=10, color=CHART["muted"])),
                yaxis=dict(
                    title="EUR/MWh",
                    gridcolor=CHART["grid"],
                    tickfont=dict(size=10, color=CHART["muted"]),
                    zeroline=False,
                ),
                showlegend=False,
                hovermode="x unified",
            )
            st.plotly_chart(_fig_sp, width="stretch")

    with st.expander("Price interpretation reference", expanded=True):
        st.markdown("""
        **TTF (EUR/MWh) reference levels:**

        | Range | Market signal |
        |-------|--------------|
        | Below 20 | Pre-crisis normal. Strong injection economics. |
        | 20 to 35 | Post-crisis baseline. Moderate supply risk reflected. |
        | 35 to 50 | Elevated. Storage tightness or supply uncertainty priced in. |
        | Above 50 | High. Industrial demand destruction becomes a factor. Coal switching increases. |
        | Above 100 | Crisis level (2022 peak: approximately 350 EUR/MWh). |

        **Nordic-Continental spread (NL − NO2):**
        Computed as `NL day-ahead − NO2 day-ahead` (EUR/MWh). Sources: ENTSO-E Transparency Platform, with Nord Pool for days ENTSO-E has not yet published; 365-day history.
        Reference levels: ±€20/MWh are empirical thresholds; above +€20, NordLink has historically run above 90% utilisation.

        - Positive spread (Continental above Nordic): Norwegian hydro has a clear export incentive. Interconnectors are likely constrained.
        - Near zero: Markets are broadly coupled under normal conditions.
        - Negative spread (Nordic above Continental): Atypical. Check Norwegian reservoir levels and interconnector availability.

        **Finland (FI) in context:** Finland is a net power importer, connected to Sweden (SE3) and the Baltic states. The FI versus NO spread often reflects nuclear availability (Olkiluoto) and Swedish hydro levels. Finland and Italy represent contrasting generation mixes exposed differently to gas price shocks.
        """)

    # ── Norwegian zonal vs system price ──────────────────────────────────────
    st.divider()
    st.markdown("#### Norgespris debate: zonal vs system pricing")
    st.caption(
        "Norwegian day-ahead prices by bidding zone (NO1–NO5) vs the Nordic system price (SYS). "
        "When zones diverge from SYS, congestion on internal transmission bottlenecks is the driver. "
        "Sources: ENTSO-E Transparency Platform (zones) and Nord Pool Data Portal (SYS). 90-day history."
    )

    with st.spinner("Loading Norwegian zonal prices…"):
        _no_zonal_df = _fetch_norwegian_zonal(days=90)

    if _no_zonal_df.empty:
        st.caption("Norwegian zonal prices are unavailable right now: neither ENTSO-E nor Nord Pool answered.")
    else:
        import plotly.graph_objects as _go_no

        # KPI row: latest price per zone
        _latest_no_date = _no_zonal_df["date"].max()
        _latest_no = (
            _no_zonal_df[_no_zonal_df["date"] == _latest_no_date]
            .set_index("zone")["price_eur_mwh"]
        )
        _sys_price = float(_latest_no["SYS"]) if "SYS" in _latest_no.index else None
        _kpi_cols = st.columns(len(_NO_ZONES_ZONAL))
        for _ci, _zone in enumerate(_NO_ZONES_ZONAL):
            with _kpi_cols[_ci]:
                if _zone in _latest_no.index:
                    _zp = float(_latest_no[_zone])
                    if _zone == "SYS":
                        _delta = delta_span("system reference", "blue")
                    elif _sys_price is not None:
                        _diff = _zp - _sys_price
                        _color = "red" if abs(_diff) > 10 else ("amber" if abs(_diff) > 3 else "green")
                        _delta = delta_span(f"{'▲' if _diff >= 0 else '▼'} {abs(_diff):.1f} vs SYS", _color)
                    else:
                        _delta = delta_span("n/a", "blue")
                    st.markdown(
                        kpi_card(_NO_ZONE_LABELS[_zone], f"€{_zp:.1f}", _delta),
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        kpi_card(_NO_ZONE_LABELS[_zone], "n/a", delta_span("no data", "amber")),
                        unsafe_allow_html=True,
                    )

        # Line chart: all zones over 90 days
        _fig_no = _go_no.Figure()
        for _zone in _NO_ZONES_ZONAL:
            _sub = (
                _no_zonal_df[_no_zonal_df["zone"] == _zone]
                .sort_values("date")
            )
            if _sub.empty:
                continue
            _dash = "dot" if _zone == "SYS" else "solid"
            _width = 2.2 if _zone == "SYS" else 1.4
            _fig_no.add_trace(_go_no.Scatter(
                x=_sub["date"], y=_sub["price_eur_mwh"],
                name=_NO_ZONE_LABELS[_zone],
                line=dict(color=_NO_ZONE_COLORS[_zone], width=_width, dash=_dash),
                hovertemplate=f"{_NO_ZONE_LABELS[_zone]}: €%{{y:.1f}}/MWh<extra></extra>",
            ))

        _fig_no.update_layout(
            template=PLOTLY_TEMPLATE,
            paper_bgcolor=CHART["bg"],
            plot_bgcolor=CHART["panel_bg"],
            height=260,
            margin=dict(l=50, r=20, t=10, b=35),
            xaxis=dict(gridcolor=CHART["grid"], tickfont=dict(size=10, color=CHART["muted"])),
            yaxis=dict(
                title="EUR/MWh",
                gridcolor=CHART["grid"],
                tickfont=dict(size=10, color=CHART["muted"]),
                zeroline=False,
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        font=dict(size=10)),
            hovermode="x unified",
        )
        st.plotly_chart(_fig_no, width="stretch")

        with st.expander("Norwegian zonal pricing context", expanded=False):
            st.markdown("""
            **Why five zones?** Norway is divided into five bidding zones (NO1–NO5) reflecting
            geographic bottlenecks in the transmission grid. When the grid can carry surplus freely,
            all zones clear at the same price. Congestion on key lines (particularly the north–south
            corridor) causes zones to diverge.

            **System price (SYS):** The Nordic system price is a theoretical unconstrained clearing
            price assuming no transmission limits within the Nordic area. It serves as the settlement
            reference for financial power contracts (Nord Pool financial market). Physical spot prices
            (the zonal prices) are what generators and consumers actually receive.

            **Key bottlenecks driving divergence:**
            - NO1 (Oslo) vs NO2 (Kristiansand/SW Norway): Affected by north–south flow on Sørlandsbanen lines.
            - NO3 (Central/Mid-Norway) vs NO1/NO5: Snøhvit and inland hydro reservoir imbalances push NO3 low in wet periods.
            - NO4 (North Norway / Tromsø): Structurally cheap: large hydro surplus, limited southward export capacity.
            - NO5 (West Norway / Bergen): High hydro density; diverges from NO1 on congestion into central grid.

            **Norgespris debate:** Norway is considering moving to a single national price zone
            (a "Norwegian system price" for physical delivery) to reduce inter-zonal distortions for
            consumers and industrial users. Critics argue this removes price signals needed to incentivise
            transmission investment and demand flexibility. The debate is active in the Storting (Norwegian parliament).
            """)

    # ── 3A: FI minus IT-North spread (roadmap 3A) ────────────────────────────
    # Self-contained block: data (data/italy_prices.py), statistics
    # (models/fi_it_spread.py), KPI row, two-panel chart, commentary, methodology.
    import math as _fiit_math
    import plotly.graph_objects as _fiit_go
    from plotly.subplots import make_subplots as _fiit_subplots
    from data.italy_prices import fetch_it_north_daily as _fiit_fetch_it, fetch_fi_daily as _fiit_fetch_fi
    from models.fi_it_spread import (
        build_spread_frame as _fiit_build,
        add_rolling_mean as _fiit_add_ma,
        spread_statistics as _fiit_statistics,
    )
    from config.settings import (
        FI_IT_HISTORY_DAYS as _FIIT_HIST_D,
        FI_IT_AVG_WINDOW_DAYS as _FIIT_AVG_D,
        FI_IT_Z_WINDOW_DAYS as _FIIT_Z_D,
        FI_IT_Z_MIN_OBS as _FIIT_Z_MIN,
        FI_IT_MIN_DAY_COVERAGE as _FIIT_COV,
    )

    st.divider()
    st.markdown("#### FI minus IT-North Day-Ahead Spread")
    st.caption(
        "Finland (FI) minus northern Italy (IT-North) day-ahead price, daily averages per CET "
        "delivery day (EUR/MWh). Contrasts a Nordic system clearing on low-marginal-cost supply "
        "with a Continental zone where gas-fired plants usually set the price."
    )

    with st.spinner("Loading FI and IT-North day-ahead prices…"):
        _fiit_it = _fiit_fetch_it(days=_FIIT_HIST_D)
        _fiit_fi = _fiit_fetch_fi(days=_FIIT_HIST_D)

    _fiit_it_df = _fiit_it.get("daily", pd.DataFrame())
    _fiit_fi_df = _fiit_fi.get("daily", pd.DataFrame())
    _fiit_df = _fiit_build(_fiit_fi_df, _fiit_it_df)

    if _fiit_it_df.empty or _fiit_fi_df.empty:
        _fiit_missing = []
        if _fiit_fi_df.empty:
            _fiit_missing.append("FI (Nord Pool Data Portal unreachable)")
        if _fiit_it_df.empty:
            _fiit_missing.append(
                "IT-North (energy-charts.info unreachable; the ENTSO-E fallback requires ENTSOE_API_KEY)"
            )
        st.caption("FI minus IT-North spread unavailable. Missing: " + "; ".join(_fiit_missing) + ".")
    elif len(_fiit_df) < 5:
        st.caption(
            f"FI minus IT-North spread unavailable: only {len(_fiit_df)} overlapping delivery days "
            "between the FI and IT-North series."
        )
    else:
        _fiit_df = _fiit_add_ma(_fiit_df, window_days=_FIIT_AVG_D)
        _fs = _fiit_statistics(
            _fiit_df, avg_window_days=_FIIT_AVG_D, z_window_days=_FIIT_Z_D, min_z_obs=_FIIT_Z_MIN,
        )
        _fiit_z = _fs["z"]
        _fiit_z_ok = _fiit_math.isfinite(_fiit_z)
        _fiit_win_lbl = "1-year" if _fs["z_full_window"] else f"{_fs['n_z']}-day"
        _fiit_date_s = _fs["latest_date"].strftime("%d %b %Y")

        _fk1, _fk2, _fk3, _fk4 = st.columns(4)
        with _fk1:
            st.markdown(
                kpi_card(
                    "FI minus IT-North (latest)",
                    f"€{_fs['latest_spread']:+.1f}/MWh",
                    delta_span(
                        f"{_fiit_date_s}: FI €{_fs['latest_fi']:.1f}, IT-North €{_fs['latest_it_north']:.1f}",
                        "blue",
                    ),
                ),
                unsafe_allow_html=True,
            )
        with _fk2:
            st.markdown(
                kpi_card(
                    f"{_FIIT_AVG_D}-day average spread",
                    f"€{_fs['avg']:+.1f}/MWh",
                    delta_span(f"FI below IT-North on {_fs['n_fi_below']} of {_fs['n_avg']} days", "blue"),
                ),
                unsafe_allow_html=True,
            )
        with _fk3:
            if _fiit_z_ok:
                _fiit_z_col = "red" if abs(_fiit_z) >= 2 else ("amber" if abs(_fiit_z) >= 1 else "green")
                _fiit_z_note = f"vs {_fiit_win_lbl} mean €{_fs['z_mean']:+.1f}, σ €{_fs['z_std']:.1f}"
                if not _fs["z_full_window"]:
                    _fiit_z_note += " (partial year)"
                st.markdown(
                    kpi_card("Z-score vs 1-year", f"{_fiit_z:+.2f}σ", delta_span(_fiit_z_note, _fiit_z_col)),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    kpi_card(
                        "Z-score vs 1-year", "n/a",
                        delta_span(f"needs ≥{_FIIT_Z_MIN} days, have {_fs['n_z']}", "amber"),
                    ),
                    unsafe_allow_html=True,
                )
        with _fk4:
            st.markdown(
                kpi_card(
                    f"{_fiit_win_lbl.capitalize()} spread range",
                    f"€{_fs['window_min']:+.0f} to €{_fs['window_max']:+.0f}",
                    delta_span(f"{_fs['n_z']} delivery days", "blue"),
                ),
                unsafe_allow_html=True,
            )

        # Two-panel chart: price levels (top), spread with rolling mean (bottom)
        _fiit_fig = _fiit_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.55, 0.45],
        )
        _fiit_it_plot = _fiit_it_df.copy()
        _fiit_it_plot["date"] = pd.to_datetime(_fiit_it_plot["date"])
        _fiit_fi_plot = _fiit_fi_df.copy()
        _fiit_fi_plot["date"] = pd.to_datetime(_fiit_fi_plot["date"])
        _fiit_fig.add_trace(_fiit_go.Scatter(
            x=_fiit_it_plot["date"], y=_fiit_it_plot["it_north"], name="IT-North",
            line=dict(color="#c2611a", width=1.5),
            hovertemplate="IT-North: €%{y:.1f}/MWh<extra></extra>",
        ), row=1, col=1)
        _fiit_fig.add_trace(_fiit_go.Scatter(
            x=_fiit_fi_plot["date"], y=_fiit_fi_plot["fi"], name="FI (Finland)",
            line=dict(color="#6e4a9e", width=1.5),
            hovertemplate="FI: €%{y:.1f}/MWh<extra></extra>",
        ), row=1, col=1)
        _fiit_fig.add_trace(_fiit_go.Scatter(
            x=_fiit_df["date"], y=_fiit_df["spread"], name="FI minus IT-North",
            line=dict(color="#2f45c6", width=1.3),
            fill="tozeroy", fillcolor="rgba(47,69,198,0.08)",
            hovertemplate="FI minus IT-North: €%{y:.1f}/MWh<extra></extra>",
        ), row=2, col=1)
        _fiit_fig.add_trace(_fiit_go.Scatter(
            x=_fiit_df["date"], y=_fiit_df["spread_ma"], name=f"{_FIIT_AVG_D}-day mean",
            line=dict(color="#a86a12", width=1.6, dash="dot"),
            hovertemplate=f"{_FIIT_AVG_D}-day mean: €%{{y:.1f}}/MWh<extra></extra>",
        ), row=2, col=1)
        _fiit_fig.add_hline(y=0, line=dict(color="rgba(19,47,56,0.25)", width=1), row=2, col=1)
        _fiit_fig.update_layout(
            template=PLOTLY_TEMPLATE,
            paper_bgcolor="#f5f7f4",
            plot_bgcolor="#f5f7f4",
            height=440,
            margin=dict(l=50, r=20, t=10, b=35),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        font=dict(size=10)),
            hovermode="x unified",
        )
        _fiit_fig.update_xaxes(gridcolor="rgba(19,47,56,0.06)", tickfont=dict(size=10, color="#667b83"))
        _fiit_fig.update_yaxes(gridcolor="rgba(19,47,56,0.06)", tickfont=dict(size=10, color="#667b83"),
                               zeroline=False)
        _fiit_fig.update_yaxes(title_text="EUR/MWh", row=1, col=1)
        _fiit_fig.update_yaxes(title_text="Spread (EUR/MWh)", row=2, col=1)
        st.plotly_chart(_fiit_fig, width="stretch")

        _fiit_span_note = (
            f"Spread shown for {len(_fiit_df)} overlapping delivery days "
            f"({_fiit_df['date'].iloc[0]:%d %b %Y} to {_fiit_df['date'].iloc[-1]:%d %b %Y}). "
            f"FI history: {_fiit_fi.get('source') or 'n/a'}, {len(_fiit_fi_df)} days"
        )
        if _fiit_fi.get("entsoe_days", 0) == 0 and len(_fiit_fi_df) < _FIIT_Z_D * 0.9:
            _fiit_span_note += (
                " (the public Nord Pool endpoint serves about two months; "
                "set ENTSOE_API_KEY to extend FI to one year)"
            )
        st.caption(_fiit_span_note + ".")

        # Commentary: every figure below is computed from the series above
        _fiit_text = (
            f"On {_fiit_date_s}, Finland cleared at €{_fs['latest_fi']:.1f}/MWh against "
            f"€{_fs['latest_it_north']:.1f}/MWh in IT-North, a spread of €{_fs['latest_spread']:+.1f}/MWh. "
            f"Across the {_fs['n_avg']} delivery days in the trailing {_FIIT_AVG_D}-day window the spread "
            f"averaged €{_fs['avg']:+.1f}/MWh, with Finland pricing below IT-North on "
            f"{_fs['n_fi_below']} of {_fs['n_avg']} days. "
        )
        if _fiit_z_ok:
            _fiit_text += (
                f"The latest reading is {_fiit_z:+.2f}σ from its {_fiit_win_lbl} mean of "
                f"€{_fs['z_mean']:+.1f}/MWh. "
            )
        _fiit_text += (
            "A deeply negative FI minus IT-North spread is the expected configuration when Italian "
            "prices are set by gas-fired plants at the margin while Finland clears on low-marginal-cost "
            "supply (hydro, nuclear including Olkiluoto 3, and wind). The spread narrows when Finnish "
            "supply tightens (nuclear outages, low-wind spells, cold snaps) or when gas costs fall."
        )
        st.markdown(
            commentary(_fiit_text, "warn" if (_fiit_z_ok and abs(_fiit_z) >= 2) else "ok"),
            unsafe_allow_html=True,
        )

        with st.expander("FI minus IT-North spread: methodology and sources", expanded=False):
            st.markdown(f"""
            **Definition:** spread = FI day-ahead price minus IT-North day-ahead price (EUR/MWh),
            using daily averages per CET/CEST delivery day. No unit or currency conversion is required.

            **FI data:** Nord Pool Data Portal daily area averages (no key required). The public
            endpoint only serves roughly the latest two months of delivery days; older dates return
            HTTP 401. When `ENTSOE_API_KEY` is set, ENTSO-E A44 day-ahead prices for FI
            (EIC 10YFI-1--------U) extend the history to {_FIIT_HIST_D} days; Nord Pool values are kept
            on overlapping days because both publish the same coupled (SDAC) clearing prices.

            **IT-North data:** energy-charts.info price API (Fraunhofer ISE), bidding zone IT-North,
            requested in chunks of at most 90 days. Data from Bundesnetzagentur | SMARD.de, licensed
            under CC BY 4.0 (creativecommons.org/licenses/by/4.0). Fallback: ENTSO-E A44 for IT-North
            (EIC 10Y1001A1001A73I) when `ENTSOE_API_KEY` is set. energy-charts.info also publishes FI
            prices, but under a licence restricted to private and internal use, so that series is not used.

            **Daily aggregation:** quarter-hourly prices (from delivery day 1 October 2025) or hourly
            prices (before) are averaged arithmetically per delivery day in Europe/Rome time
            (23, 24 or 25 hours around daylight-saving changes). Days with less than
            {_FIIT_COV:.0%} of the expected intervals are excluded.

            **Statistics:** the {_FIIT_AVG_D}-day average covers the trailing {_FIIT_AVG_D} calendar days
            ending on the latest common delivery day. The z-score is (latest spread minus mean) divided
            by the sample standard deviation over the trailing {_FIIT_Z_D} calendar days, reported only
            with at least {_FIIT_Z_MIN} daily observations and flagged as partial when the history is
            shorter than a full year.

            **Interpretation and limitations:** Finland and Italy are not directly interconnected,
            so the spread is not a tradable arbitrage. It is a structural comparison of the marginal
            technology in each zone: low-marginal-cost Nordic supply against gas-fired generation in
            Italy. IT-North is one of several Italian bidding zones and hosts Italy's interconnections
            with France, Switzerland, Austria and Slovenia.
            """)

    _fiit_src_bits = ["FI: " + (_fiit_fi.get("source") or "unavailable")]
    _fiit_src_bits.append("IT-North: " + (_fiit_it.get("source") or "unavailable"))
    if _fiit_it.get("licence"):
        _fiit_src_bits.append("Licence: " + _fiit_it["licence"])
    st.caption("Sources: " + " | ".join(_fiit_src_bits))
    if _fiit_it.get("error"):
        st.caption(f"IT-North chunks not loaded: {_fiit_it['error']}")
    # ── end 3A: FI minus IT-North spread ─────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5: POWER FLOWS
# ═══════════════════════════════════════════════════════════════════════════
with tab_flows:
    if not has_entsoe_key():
        st.info(
            "ENTSO-E API key required. Register free at transparency.entsoe.eu, "
            "then add ENTSOE_API_KEY to .env."
        )
    else:
        render_flows_chart(flows)
        _flows_src = flows.get("source", "ENTSO-E")
        st.caption(f"Cross-border flows · Source: {_flows_src}")

        # ── B5: Interconnector utilisation KPIs ───────────────────────────────
        # Latest complete day per corridor. Utilisation is the mean absolute hourly net
        # flow over the link's rating; NO1-SE3 has no fixed rating (flow-based capacity).
        _flows_df = flows.get("flows", pd.DataFrame())
        if not _flows_df.empty:
            _flows_df = _flows_df.copy()
            _flows_df["pair"] = _flows_df["pair"].str.replace("→", "->", regex=False)
            _corridors = ["NO2->DE", "NO2->GB", "NO2->DK1", "NO2->NL", "NO1->SE3"]
            st.markdown("##### Interconnector utilisation: latest complete day")
            _util_cols = st.columns(len(_corridors))
            for _ci, _pair in enumerate(_corridors):
                _rows = _flows_df[_flows_df["pair"] == _pair].sort_values("date")
                _label = _pair.replace("->", " → ")
                with _util_cols[_ci]:
                    if _rows.empty:
                        st.markdown(kpi_card(_label, "n/a", delta_span("no data from ENTSO-E", "amber")),
                                    unsafe_allow_html=True)
                        continue
                    _row = _rows.iloc[-1]
                    _net = float(_row["net_flow_mwh"])
                    _dir = "net import" if _net > 0 else "net export"
                    _day = pd.Timestamp(_row["date"]).strftime("%d %b")
                    _cap_mw = INTERCONNECTOR_CAPACITY_MW.get(_pair)
                    if _cap_mw:
                        _util_pct = min(float(_row["abs_net_mwh"]) / (_cap_mw * 24.0) * 100.0, 100.0)
                        _u_color = ("red" if _util_pct > INTERCONNECTOR_UTIL_HIGH_PCT
                                    else "amber" if _util_pct > INTERCONNECTOR_UTIL_MED_PCT else "green")
                        st.markdown(kpi_card(_label, f"{_util_pct:.0f}%",
                                             delta_span(f"{abs(_net) / 1_000:.1f} GWh {_dir}, {_day}", _u_color)),
                                    unsafe_allow_html=True)
                    else:
                        st.markdown(kpi_card(_label, f"{abs(_net) / 1_000:.1f} GWh",
                                             delta_span(f"{_dir}, {_day}; AC border, no fixed rating", "blue")),
                                    unsafe_allow_html=True)

        with st.expander("Understanding Norwegian cross-border flows", expanded=True):
            st.markdown("""
            **Corridors shown:**
            - NO2 to DE (NordLink, 1.4 GW HVDC): Norwegian hydro to Germany
            - NO2 to GB (North Sea Link, 1.4 GW HVDC): Norway to Great Britain, in service since 2021
            - NO2 to DK1 (Skagerrak 1-4, 1.64 GW HVDC): Norway to western Denmark
            - NO2 to NL (NorNed, 0.7 GW HVDC): Norway to the Netherlands
            - NO1 to SE3 (AC interconnection): Eastern Norway to central Sweden

            **Sign convention:** positive values are a net import into Norway, negative values a net export.

            **Units:** ENTSO-E reports each direction as average power per 15 minutes. Both directions are
            averaged to hourly power on Norwegian time and summed into energy per day (GWh).

            **Utilisation:** mean absolute hourly net flow on the latest complete day divided by the link's
            rating. Ratings are nameplate values for the HVDC links; available capacity can be lower during
            maintenance or when the TSOs reduce it. NO1-SE3 has no fixed rating because Nordic capacity on AC
            borders comes from flow-based capacity calculation. Colours: above 90% red (constrained), above
            70% amber, otherwise green.

            **What to watch:**
            - Sustained Norwegian exports draw reservoirs down; compare with the Hydro Reservoirs tab
            - Net imports (positive values) mean Norwegian supply is short of demand at Continental prices
            - Compare the flow direction with the Nordic-Continental price spread in the Prices tab
            """)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 6: GENERATION (GERMAN COAL / FUEL SWITCHING)
# ═══════════════════════════════════════════════════════════════════════════
with tab_generation:
    if not has_entsoe_key():
        st.info(
            "ENTSO-E API key required. Register free at transparency.entsoe.eu, "
            "then add ENTSOE_API_KEY to .env."
        )
    else:
        render_coal_chart(generation, ttf_price=ttf_price)

        with st.expander("Coal as a fuel switching indicator", expanded=True):
            st.markdown("""
            **The merit order and fuel switching:**
            Germany's power system dispatches generation in order of marginal cost: nuclear and renewables
            run first, then lignite, then hard coal, then gas, with oil peakers last.
            When gas prices rise above approximately EUR 50/MWh, coal-fired generation
            becomes cheaper at the margin, causing utilities to switch from gas to coal.
            This process is called fuel switching and is a primary mechanism linking gas prices to power prices.

            **Lignite versus hard coal:**
            - Lignite (brunkull): domestic German brown coal, very low fuel cost but high CO2 intensity.
              Plants are mostly located in the Rhineland and Lusatia mining regions. Nearly always dispatched
              when available due to near-zero variable fuel cost.
            - Hard coal (sort kull): imported steam coal, marginal cost more sensitive to global coal prices
              and carbon (EUA) costs. More responsive to the gas-coal switching dynamic.

            **What the 2021-2022 energy crisis showed:**
            When TTF exceeded EUR 100/MWh in August 2022, gas-to-coal switching pushed German coal
            generation to the highest annual total in the series above, about 166 TWh in 2022.
            Hard coal and lignite together ran at roughly 37 to 46 TWh per quarter from late 2021
            to early 2023. As gas prices normalised in 2023-2024, coal dispatch fell sharply.
            Renewed gas price pressure in 2025-2026 is now causing a partial reversal.

            **Relevance for Norwegian hydro:**
            High German coal generation signals that gas is expensive and therefore Continental power
            prices are elevated. This improves the economics of exporting Norwegian hydro via NordLink
            and NorNed, increasing the flow incentive to Germany and the Netherlands.
            """)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 7: SOLAR
# ═══════════════════════════════════════════════════════════════════════════
with tab_solar:
    if not has_entsoe_key():
        st.info(
            "ENTSO-E API key required. Register free at transparency.entsoe.eu, "
            "then add ENTSOE_API_KEY to .env."
        )
    else:
        render_solar_chart(solar)

        with st.expander("The solar cannibalisation effect", expanded=True):
            st.markdown("""
            **The duck curve:** As solar capacity has expanded, midday power prices in high-irradiance markets (Germany, Spain) have compressed, sometimes reaching zero or negative values. Morning and evening ramps have become steeper and more commercially significant.

            **Structural implications:**
            - Midday solar surplus depresses spot prices, displacing gas-fired generation
            - Evening ramp (approximately 16:00 to 20:00): solar output falls and gas or hydro must ramp rapidly, producing price spikes
            - Cannibalisation: each additional MW of solar capacity installed earns less per MWh, as it suppresses the very prices it sells into

            **Relevance for Norwegian hydro:** Norwegian hydro is the most flexible dispatchable resource in Northern Europe. The evening ramp period creates high-value export windows; optimal dispatch timing is determined by the intraday price curve shown above.
            """)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 7: HYDRO RESERVOIRS
# ═══════════════════════════════════════════════════════════════════════════
with tab_hydro:
    if not has_entsoe_key():
        st.info(
            "ENTSO-E API key required for hydro reservoir data. "
            "Register free at transparency.entsoe.eu, then add ENTSOE_API_KEY to .env."
        )
    else:
        render_hydro_chart(hydro)
        _hydro_src = hydro.get("source", "ENTSO-E B31")
        st.caption(f"Hydro reservoirs · Source: {_hydro_src}")

        with st.expander("Why Norwegian hydro levels matter for power prices", expanded=True):
            st.markdown("""
            **Hydro as the European price anchor:** Norwegian hydropower (~140 TWh/yr) is the largest flexible generation source in Northern Europe. Reservoir filling levels directly determine Norway's ability to export power to Germany (NordLink), the Netherlands (NorNed), and Great Britain (NSN).

            **Seasonal dynamics:**
            - Spring snowmelt (April-June) typically refills reservoirs from winter lows
            - Autumn drawdown begins as winter heating demand rises
            - Cold, dry winters accelerate depletion; wet springs rebuild buffers

            **Price signal interpretation:**
            - Reservoirs below historical 10th percentile: strongly bullish for NO2 spot prices; Norway likely importing from Continental Europe
            - Below seasonal median: mild upward price pressure; reduced export capacity
            - Above seasonal median: Norway incentivised to export, placing downward pressure on Continental prices via interconnectors

            **Key interconnectors from Norway:**

            | Link | Capacity | Markets connected |
            |------|----------|------------------|
            | NordLink | 1,400 MW | NO2 ↔ Germany (DE-LU) |
            | NorNed | 700 MW | NO2 ↔ Netherlands (NL) |
            | NSN | 1,400 MW | NO1 ↔ Great Britain |
            | Skagerrak | 1,700 MW | NO2 ↔ Denmark (DK1) |

            **Data source:** ENTSO-E Transparency Platform, Hydro Reservoirs and Water Storage (B31), Norway aggregate (EIC: 10YNO-0--------C).
            """)


# ── Footer ─────────────────────────────────────────────────────────────────
page_footer([
    ("Gas storage and LNG data", "Gas Infrastructure Europe (GIE) AGSI+/ALSI", "https://agsi.gie.eu"),
    ("Power data", "ENTSO-E Transparency Platform", "https://transparency.entsoe.eu"),
    ("Spot prices", "Nord Pool", "https://www.nordpoolgroup.com"),
    ("TTF", "Yahoo Finance / ICE", "https://finance.yahoo.com"),
])
