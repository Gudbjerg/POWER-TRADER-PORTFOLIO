"""
Layer 1: European Power and Gas Live Market Monitor
"""
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Live Monitor",
    page_icon="E",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Imports ────────────────────────────────────────────────────────────────
from data.gas_storage     import get_storage_data, build_seasonal_bands
from data.prices          import get_ttf_data
from data.power_flows     import get_flow_data
from data.spot_prices     import get_spot_price_data
from data.solar           import get_solar_data
from data.lng_terminals   import get_lng_data
from data.hydro           import get_hydro_data
from data.forward_curve   import get_forward_curve_data
from components.storage_chart        import render_storage_chart
from components.prices_chart         import render_ttf_chart
from components.flows_chart          import render_flows_chart
from components.spot_prices_chart    import render_spot_prices_chart
from components.solar_chart          import render_solar_chart
from components.lng_chart            import render_lng_chart
from components.hydro_chart          import render_hydro_chart
from components.forward_curve_chart  import render_forward_curve_chart
from utils.helpers   import (
    has_entsoe_key, has_agsi_key, apply_dark_theme,
    pill, commentary, kpi_card, delta_span,
)
from utils.scenarios import (
    storage_status, ttf_status, spread_status, lng_status, market_summary,
)
import pandas as pd
from datetime import date as _date, timedelta as _timedelta

apply_dark_theme()


# ── Load all data (cached) ─────────────────────────────────────────────────
with st.spinner(""):
    storage = get_storage_data()
    ttf     = get_ttf_data()
    flows   = get_flow_data()
    spots   = get_spot_price_data()
    solar   = get_solar_data()
    lng     = get_lng_data()
    hydro   = get_hydro_data()

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
    st.markdown("## Layer 1: Live Market Monitor")
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
tab_overview, tab_gas, tab_lng, tab_prices, tab_flows, tab_solar, tab_hydro = st.tabs([
    "Overview",
    "Gas Storage",
    "LNG Terminals",
    "Prices",
    "Power Flows",
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
            _required_pp = max(0.0, 80.0 - eu_pct)
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
                f"EU regulation requires 80% fill by November 1. "
                f"With EU storage at {eu_pct:.1f}% and {_days_to_nov} days remaining, "
                f"the required net injection is approximately {_required_pp:.0f}pp, "
                f"{_pace} the 5-year average injection pace. "
                + _lng_note
            )
        else:
            st.caption(
                f"EU regulation requires 80% fill by November 1. "
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

        **Nordic-Continental spread interpretation:**
        - Positive spread (Continental above Nordic): Norwegian hydro has a clear export incentive. Interconnectors are likely constrained.
        - Near zero: Markets are broadly coupled under normal conditions.
        - Negative spread (Nordic above Continental): Atypical. Check Norwegian reservoir levels and interconnector availability.

        **Finland (FI) in context:** Finland is a net power importer, connected to Sweden (SE3) and the Baltic states. The FI versus NO spread often reflects nuclear availability (Olkiluoto) and Swedish hydro levels. Finland and Italy represent contrasting generation mixes exposed differently to gas price shocks.
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

        with st.expander("Understanding Nordic cross-border flows", expanded=True):
            st.markdown("""
            **Key interconnector corridors:**
            - NO2 to DE (NordLink, 1.4 GW): primary corridor linking Norwegian hydro to Germany
            - NO2 to NL (NorNed, 0.7 GW): Norway-Netherlands
            - NO1 to GB (NSN, 1.4 GW): Norway-United Kingdom, commissioned 2021
            - NO2 to DK1 (Skagerrak cables, 1.7 GW): Norway-Denmark West

            **Sign convention:** Positive values indicate net import into Norway. Negative values indicate net export from Norway.

            **Key signals to monitor:**
            - Sustained high Norwegian exports imply reservoir draw-down; monitor NVE reservoir data
            - Import spikes (positive values) indicate Norwegian generation is below domestic demand
            - Compare flow direction against the Nordic-Continental price spread in the Prices tab
            """)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 6: SOLAR
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
