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
from data.gas_storage     import get_storage_data, build_seasonal_bands
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
from utils.helpers   import (
    has_entsoe_key, has_agsi_key, apply_dark_theme,
    pill, commentary, kpi_card, delta_span,
)
from utils.scenarios import (
    storage_status, ttf_status, spread_status, lng_status, market_summary,
)
import pandas as pd
from datetime import date as _date, timedelta as _timedelta
from config.settings import (
    INTERCONNECTOR_CAPACITY_MW,
    INTERCONNECTOR_UTIL_HIGH_PCT, INTERCONNECTOR_UTIL_MED_PCT,
    SPREAD_CHART_REF_EUR,
)

apply_dark_theme()


@st.cache_data(ttl=3600, persist="disk", show_spinner=False)
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
    "NO1": "#58a6ff", "NO2": "#3fb950", "NO3": "#d4ac3a",
    "NO4": "#e07b39", "NO5": "#8e6bbf", "SYS": "#f85149",
}


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_norwegian_zonal(days: int = 90) -> pd.DataFrame:
    """Day-ahead prices for NO1/NO2/NO3/NO4/NO5 + SYS via Nord Pool API."""
    import requests as _req
    from concurrent.futures import ThreadPoolExecutor as _TPE, as_completed as _acf
    from datetime import datetime as _dt, timedelta as _td
    _NORDPOOL_URL = "https://dataportal-api.nordpoolgroup.com/api/DayAheadPrices"
    _HEADERS = {"Referer": "https://www.nordpoolgroup.com/"}
    today = _dt.utcnow().date()
    dates = [(today - _td(days=i)).strftime("%Y-%m-%d") for i in range(days + 1)]

    def _fetch_day_no(date_str):
        try:
            r = _req.get(
                _NORDPOOL_URL,
                params={"currency": "EUR", "market": "DayAhead",
                        "deliveryArea": ",".join(_NO_ZONES_ZONAL), "date": date_str},
                headers=_HEADERS, timeout=10,
            )
            if r.status_code == 204:
                return []
            r.raise_for_status()
            rows = []
            for entry in r.json().get("areaAverages", []):
                if entry.get("price") is not None and entry["areaCode"] in _NO_ZONES_ZONAL:
                    rows.append({"date": date_str, "zone": entry["areaCode"],
                                 "price_eur_mwh": entry["price"]})
            return rows
        except Exception:
            return []

    records = []
    with _TPE(max_workers=8) as ex:
        for fut in _acf({ex.submit(_fetch_day_no, d): d for d in dates}):
            records.extend(fut.result())

    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.sort_values("date").reset_index(drop=True)


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
    doy = row["gasDayStart"].dayofyear
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
if not spot_df.empty:
    latest_date = spot_df["date"].max()
    latest = spot_df[spot_df["date"] == latest_date].set_index("zone")["price_eur_mwh"]
    n_zones = [z for z in ["NO1", "NO2"] if z in latest.index]
    c_zones = [z for z in ["DE-LU", "NL"] if z in latest.index]
    if n_zones: nordic_avg      = float(latest[n_zones].mean())
    if c_zones: continental_avg = float(latest[c_zones].mean())

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
    spread_status(nordic_avg, continental_avg)
    if nordic_avg is not None and continental_avg is not None else ("ok", "No spot data", "")
)
lng_stat, lng_headline, lng_detail = lng_status(lng_wow)


# ── Header ─────────────────────────────────────────────────────────────────
_hdr_col, _btn_col = st.columns([5, 1])
with _hdr_col:
    st.markdown("## Layer 1 · Live Market Monitor")
    st.caption("Real-time European power and gas data, refreshed hourly. Sources: GIE AGSI+/ALSI, Nord Pool, ENTSO-E, ICE/Yahoo Finance")
with _btn_col:
    st.markdown("<div style='padding-top:18px;'>", unsafe_allow_html=True)
    if st.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

missing = []
if not has_agsi_key():   missing.append("AGSI_API_KEY (agsi.gie.eu)")
if not has_entsoe_key(): missing.append("ENTSOE_API_KEY (transparency.entsoe.eu)")
if missing:
    st.info("Add to .env to unlock all panels: " + " and ".join(missing))


# ── KPI bar ────────────────────────────────────────────────────────────────
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
        st.markdown(kpi_card("EU Gas Storage", "n/a", delta_span("AGSI key required", "amber")), unsafe_allow_html=True)

with k2:
    if de_pct is not None:
        color = "red" if de_status == "critical" else ("amber" if de_status == "warn" else "green")
        if de_mean is not None and de_min is not None:
            d = delta_span(f"{de_pct - de_mean:+.1f}pp vs 5yr avg, min {de_min:.1f}%", color)
        else:
            d = delta_span(f"{de_pct:.1f}%, bands unavailable", color)
        st.markdown(kpi_card("DE Gas Storage", f"{de_pct:.1f}%", d), unsafe_allow_html=True)
    else:
        st.markdown(kpi_card("DE Gas Storage", "n/a", delta_span("AGSI key required", "amber")), unsafe_allow_html=True)

with k3:
    if ttf_price is not None:
        color = "red" if ttf_stat in ("critical", "warn") else "green"
        d = delta_span(f"{(ttf_price / ttf_ma30 - 1) * 100:+.1f}% vs 30d MA, 90d avg €{ttf_ma90:.1f}", color)
        st.markdown(kpi_card("TTF (EUR/MWh)", f"€{ttf_price:.2f}", d), unsafe_allow_html=True)
    else:
        st.markdown(kpi_card("TTF (EUR/MWh)", "n/a", ""), unsafe_allow_html=True)

with k4:
    if nordic_avg is not None:
        d = delta_span(f"NO avg €{nordic_avg:.0f}, NL €{continental_avg:.0f}", "blue")
        st.markdown(kpi_card("Nordic Spot (EUR/MWh)", f"€{nordic_avg:.0f}", d), unsafe_allow_html=True)
    else:
        st.markdown(kpi_card("Nordic Spot (EUR/MWh)", "n/a", ""), unsafe_allow_html=True)

with k5:
    if nordic_avg is not None and continental_avg is not None:
        spread = continental_avg - nordic_avg
        color = "amber" if abs(spread) > 15 else "green"
        if spread > 0:
            sub = "NL premium, export incentive"
        elif spread < 0:
            sub = "Nordic premium, import signal"
        else:
            sub = "markets aligned"
        d = delta_span(sub, color)
        st.markdown(kpi_card("N-C Spread (EUR/MWh)", f"€{spread:+.0f}", d), unsafe_allow_html=True)
    else:
        st.markdown(kpi_card("N-C Spread (EUR/MWh)", "n/a", ""), unsafe_allow_html=True)

with k6:
    if lng_total_today is not None:
        color = "red" if lng_alert else ("amber" if lng_wow is not None and lng_wow < -7 else "green")
        d = delta_span(f"{lng_wow:+.1f}% vs prior 7d" if lng_wow is not None else "calculating", color)
        st.markdown(kpi_card("NW EU LNG Sendout", f"{lng_total_today:.1f} TWh/d", d), unsafe_allow_html=True)
    else:
        st.markdown(kpi_card("NW EU LNG Sendout", "n/a", delta_span("AGSI key required", "amber")), unsafe_allow_html=True)

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
            nordic_avg, continental_avg,
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
            _required_pp = max(0.0, 90.0 - eu_pct)
            _pace = "above" if _required_pp > 52 else "in line with"
            if ttf_price is not None and ttf_price > 35:
                _lng_note = f"TTF at €{ttf_price:.0f}/MWh supports LNG cargo diversion to Europe over Asia."
            elif ttf_price is not None and ttf_price > 20:
                _lng_note = f"TTF at €{ttf_price:.0f}/MWh provides moderate LNG import incentive."
            elif ttf_price is not None:
                _lng_note = f"TTF at €{ttf_price:.0f}/MWh; LNG import economics are weak at current prices."
            else:
                _lng_note = ""
            st.caption(
                f"EU regulation requires 90% fill by November 1. "
                f"With EU storage at {eu_pct:.1f}% and {_days_to_nov} days remaining, "
                f"the required net injection is approximately {_required_pp:.0f}pp, "
                f"{_pace} the 5-year average injection pace. "
                + _lng_note
            )
        else:
            st.caption(
                f"EU regulation requires 90% fill by November 1. "
                f"{_days_to_nov} days remaining. Add AGSI_API_KEY to see current storage level."
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
                    f"NL at €{continental_avg:.0f}/MWh versus Nordic €{nordic_avg:.0f}/MWh "
                    f"(+€{spread:.0f}/MWh NL premium). "
                    "Strong incentive to export Norwegian hydro via NordLink, NorNed, and NSN. "
                    "Norwegian reservoir levels (ENTSO-E B31 data) determine the upper bound on export capacity."
                )
            elif spread < -5:
                st.caption(
                    f"Nordic at €{nordic_avg:.0f}/MWh versus NL €{continental_avg:.0f}/MWh "
                    f"(+€{abs(spread):.0f}/MWh Nordic premium). "
                    "Reversed spread: Norway may be importing from the Continent. "
                    "Check reservoir levels and interconnector availability."
                )
            else:
                st.caption(
                    f"Nordic €{nordic_avg:.0f}/MWh and NL €{continental_avg:.0f}/MWh broadly aligned "
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

        **Dashed horizontal line:** EU regulatory mandate of 90% fill by November 1.

        A current level below the shaded band indicates storage has not been this low on this date in any of the preceding five years. The current EU mandate requires 90% fill by November 1 (extended regulation from the original 80% emergency target under EU 2022/1032). Germany's storage capacity (approximately 24 bcm) represents roughly 25% of its annual gas consumption.
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
        - Target by November 1: 90% (current EU mandate; 83% was achieved by October 1, 2025)
        - Historical average injection: approximately 52 percentage points over the April to October period
        """)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: LNG TERMINALS
# ═══════════════════════════════════════════════════════════════════════════
with tab_lng:
    render_lng_chart(lng)

    with st.expander("Understanding LNG sendout", expanded=True):
        st.markdown("""
        **Definition:** LNG (liquefied natural gas) arrives by tanker at import terminals, where it is regasified and injected into the pipeline grid. The sendout rate (TWh/day) measures how much gas is delivered to the European grid from each terminal.

        **Market significance:** LNG is Europe's swing supply source. Cargoes can be diverted between Europe and Asia depending on relative benchmark prices (TTF versus JKM). When European TTF prices are elevated relative to Asian LNG, cargoes flow to Europe, supporting storage injection. A sustained decline in sendout tightens the European gas balance.

        **Key terminals monitored:**

        | Terminal | Country | Nameplate capacity |
        |----------|---------|-------------------|
        | Zeebrugge | Belgium | 9 bcm/yr |
        | Gate LNG | Netherlands (Rotterdam) | 12 bcm/yr |
        | South Hook | Great Britain (Wales) | 21 bcm/yr |
        | Dragon LNG | Great Britain (Wales) | 5 bcm/yr |
        | Dunkirk | France | 13 bcm/yr |
        | Montoir-de-Bretagne | France | 10 bcm/yr |

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
                line=dict(color="#58a6ff", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(88,166,255,0.08)",
                hovertemplate="NL−NO2: €%{y:.1f}/MWh<extra></extra>",
            ))
            _fig_sp.add_hline(y=0, line=dict(color="rgba(255,255,255,0.25)", width=1))
            _fig_sp.add_hline(
                y=SPREAD_CHART_REF_EUR, line=dict(color="#d4ac3a", width=1, dash="dot"),
                annotation_text=f"  +€{SPREAD_CHART_REF_EUR:.0f} (NordLink historically >90% utilised)",
                annotation_font=dict(color="#d4ac3a", size=10),
                annotation_position="right",
            )
            _fig_sp.add_hline(
                y=-SPREAD_CHART_REF_EUR, line=dict(color="#3fb950", width=1, dash="dot"),
                annotation_text=f"  −€{SPREAD_CHART_REF_EUR:.0f} (Nordic premium, atypical)",
                annotation_font=dict(color="#3fb950", size=10),
                annotation_position="right",
            )
            _fig_sp.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0d1117",
                plot_bgcolor="#161b22",
                height=220,
                margin=dict(l=50, r=140, t=10, b=35),
                xaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(size=10, color="#8b949e")),
                yaxis=dict(
                    title="EUR/MWh",
                    gridcolor="rgba(255,255,255,0.06)",
                    tickfont=dict(size=10, color="#8b949e"),
                    zeroline=False,
                ),
                showlegend=False,
                hovermode="x unified",
            )
            st.plotly_chart(_fig_sp, use_container_width=True)

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
        Computed as `NL day-ahead − NO2 day-ahead` (EUR/MWh). Source: Nord Pool public endpoint, 365-day history.
        Reference levels: ±€20/MWh are empirical thresholds — above +€20, NordLink has historically run above 90% utilisation.

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
        "Source: Nord Pool Data Portal. 90-day history."
    )

    with st.spinner("Loading Norwegian zonal prices…"):
        _no_zonal_df = _fetch_norwegian_zonal(days=90)

    if _no_zonal_df.empty:
        st.caption("Norwegian zonal data unavailable. Nord Pool API did not return results.")
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
                        _delta = delta_span("—", "blue")
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
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#161b22",
            height=260,
            margin=dict(l=50, r=20, t=10, b=35),
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(size=10, color="#8b949e")),
            yaxis=dict(
                title="EUR/MWh",
                gridcolor="rgba(255,255,255,0.06)",
                tickfont=dict(size=10, color="#8b949e"),
                zeroline=False,
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        font=dict(size=10)),
            hovermode="x unified",
        )
        st.plotly_chart(_fig_no, use_container_width=True)

        with st.expander("Norwegian zonal pricing context", expanded=False):
            st.markdown("""
            **Why five zones?** Norway is divided into five bidding zones (NO1–NO5) reflecting
            geographic bottlenecks in the transmission grid. When the grid can carry surplus freely,
            all zones clear at the same price. Congestion on key lines — particularly the north–south
            corridor — causes zones to diverge.

            **System price (SYS):** The Nordic system price is a theoretical unconstrained clearing
            price assuming no transmission limits within the Nordic area. It serves as the settlement
            reference for financial power contracts (Nord Pool financial market). Physical spot prices
            (the zonal prices) are what generators and consumers actually receive.

            **Key bottlenecks driving divergence:**
            - NO1 (Oslo) vs NO2 (Kristiansand/SW Norway): Affected by north–south flow on Sørlandsbanen lines.
            - NO3 (Central/Mid-Norway) vs NO1/NO5: Snøhvit and inland hydro reservoir imbalances push NO3 low in wet periods.
            - NO4 (North Norway / Tromsø): Structurally cheap — large hydro surplus, limited southward export capacity.
            - NO5 (West Norway / Bergen): High hydro density; diverges from NO1 on congestion into central grid.

            **Norgespris debate:** Norway is considering moving to a single national price zone
            (a "Norwegian system price" for physical delivery) to reduce inter-zonal distortions for
            consumers and industrial users. Critics argue this removes price signals needed to incentivise
            transmission investment and demand flexibility. The debate is active in the Storting (Norwegian parliament).
            """)


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
        _flows_df = flows.get("flows", pd.DataFrame())
        if not _flows_df.empty:
            _latest_day = _flows_df["date"].max()
            _today_flows = _flows_df[_flows_df["date"] == _latest_day].copy()
            _today_flows["pair"] = _today_flows["pair"].str.replace("→", "->", regex=False)

            if not _today_flows.empty:
                st.markdown("##### Interconnector utilisation: latest day")
                _util_cols = st.columns(len(INTERCONNECTOR_CAPACITY_MW))
                for _ci, (_pair, _cap_mw) in enumerate(INTERCONNECTOR_CAPACITY_MW.items()):
                    _row = _today_flows[_today_flows["pair"] == _pair]
                    with _util_cols[_ci]:
                        if not _row.empty:
                            _flow_mwh = abs(float(_row["net_flow_mwh"].iloc[0]))
                            _cap_mwh  = _cap_mw * 24.0  # MWh/day at full capacity
                            _util_pct = min(_flow_mwh / _cap_mwh * 100.0, 100.0)
                            _dir      = "→ NO" if float(_row["net_flow_mwh"].iloc[0]) > 0 else "→ EU"
                            _u_color  = "red" if _util_pct > INTERCONNECTOR_UTIL_HIGH_PCT else ("amber" if _util_pct > INTERCONNECTOR_UTIL_MED_PCT else "green")
                            st.markdown(
                                kpi_card(
                                    _pair.replace("->", " → "),
                                    f"{_util_pct:.0f}%",
                                    delta_span(f"{_flow_mwh/1000:.1f} GWh {_dir}", _u_color),
                                ),
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                kpi_card(_pair.replace("->", " → "), "n/a",
                                         delta_span("no data", "amber")),
                                unsafe_allow_html=True,
                            )

        with st.expander("Understanding Nordic cross-border flows", expanded=True):
            st.markdown("""
            **Key interconnector corridors:**
            - NO2 to DE (NordLink, 1.4 GW): primary corridor linking Norwegian hydro to Germany
            - NO2 to NL (NorNed, 0.7 GW): Norway-Netherlands
            - NO1 to GB (NSN, 1.4 GW): Norway-United Kingdom, commissioned 2021
            - NO2 to DK1 (Skagerrak cables, 1.7 GW): Norway-Denmark West

            **Sign convention:** Positive values indicate net import into Norway. Negative values indicate net export from Norway.

            **Utilisation formula:** `utilisation = |net_flow_MWh/day| ÷ (capacity_MW × 24h) × 100%`.
            Capacities are nameplate thermal limits from ENTSO-E; actual available capacity may be lower
            due to N-1 reliability margins or maintenance. Colour coding: >90% = red (constrained), >70% = amber, ≤70% = green.

            **Key signals to monitor:**
            - Sustained high Norwegian exports imply reservoir draw-down; monitor NVE reservoir data
            - Import spikes (positive values) indicate Norwegian generation is below domestic demand
            - Compare flow direction against the Nordic-Continental price spread in the Prices tab
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
            When TTF exceeded EUR 100/MWh in August 2022, German coal generation surged to levels
            not seen since the mid-2010s. Both hard coal and lignite output reached approximately
            40 TWh per quarter. As gas prices normalised in 2023-2024, coal dispatch fell sharply.
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

st.divider()
st.markdown(
    """<div style="color:#484f58;font-size:0.72rem;line-height:1.8;">
    Built by Tobias Gudbjerg &nbsp;|&nbsp;
    Gas storage and LNG data: <a href="https://agsi.gie.eu" style="color:#484f58;">Gas Infrastructure Europe (GIE) AGSI+/ALSI</a> &nbsp;|&nbsp;
    Power data: <a href="https://transparency.entsoe.eu" style="color:#484f58;">ENTSO-E Transparency Platform</a> &nbsp;|&nbsp;
    Spot prices: <a href="https://www.nordpoolgroup.com" style="color:#484f58;">Nord Pool</a> &nbsp;|&nbsp;
    TTF: <a href="https://finance.yahoo.com" style="color:#484f58;">Yahoo Finance / ICE</a><br>
    For informational purposes only. Not financial advice.
    </div>""",
    unsafe_allow_html=True,
)
