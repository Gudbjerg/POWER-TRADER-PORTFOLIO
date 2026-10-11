"""
Energy Globe: the physical system behind European gas prices, in 3D.

Norwegian shelf facilities and export pipelines (Sodir), European LNG import terminals
with live GIE ALSI send-out and stocks, and the maritime chokepoints that LNG and crude
cargoes pass through (EIA), on an interactive CesiumJS globe, with the same data in
tables below it. Open a curated view with ``?view=<key>`` (keys in ``utils/globe.py``).
"""
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Energy Globe",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

import pandas as pd  # noqa: E402

from components.energy_globe import CESIUM_VERSION, render_energy_globe  # noqa: E402
from data.energy_globe import (  # noqa: E402
    FACILITY_CATEGORIES,
    PIPELINE_MEDIA,
    fetch_alsi_terminals,
    globe_payload,
    lng_terminal_markers,
    lng_totals,
    load_layers,
)
from utils.globe import GLOBE_VIEWS, external_globe_url, resolve_view  # noqa: E402
from utils.helpers import apply_dark_theme, delta_span, kpi_card  # noqa: E402
from utils.theme import ACCENTS, callout, page_footer, page_header, render, section_header  # noqa: E402

apply_dark_theme()

view = resolve_view(st.query_params.get("view"))

page_header(
    "Infrastructure · 3D", "Energy Globe",
    "The physical system behind the prices: Norwegian shelf facilities and export pipelines, "
    "European LNG import terminals with live send-out and stocks, and the maritime chokepoints "
    "between the Gulf, Asia and Europe.",
    sources=["Sodir FactMaps", "GEM GGIT", "GIE ALSI", "U.S. EIA"],
    accent=ACCENTS["globe"],
)

# ── Data ───────────────────────────────────────────────────────────────────────
layers = load_layers()
with st.spinner("Fetching live LNG terminal figures from GIE ALSI…"):
    alsi = fetch_alsi_terminals()
markers = lng_terminal_markers(layers["facilities"], layers["facilities_source"], alsi)
totals = lng_totals(markers)
live = alsi.get("status") == "ok"

_pipelines = [f for f in layers["pipelines"] if (f.get("geometry") or {}).get("type") != "Point"]
_shelf = [f for f in layers["facilities"] if f["properties"].get("category") != "lng_import"]
_network_km = sum(float(f["properties"].get("length_km") or 0.0) for f in _pipelines)

# ── KPIs ───────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
with k1:
    if live and totals["send_out_gwh_d"] is not None:
        change = totals["send_out_change_gwh_d"]
        if change is None:
            delta = delta_span(f"Gas day {alsi.get('gas_day') or 'n/a'}", "blue")
        else:
            tone = "green" if change > 0 else ("red" if change < 0 else "blue")
            delta = delta_span(f"{change:+,.0f} GWh/d day on day · gas day {alsi.get('gas_day')}", tone)
        render(kpi_card("EU LNG send-out", f"{totals['send_out_gwh_d']:,.0f} GWh/d", delta))
    else:
        render(kpi_card("EU LNG send-out", "n/a", delta_span("Live GIE ALSI figures unavailable", "amber")))
with k2:
    if live and totals["utilisation_pct"] is not None:
        render(kpi_card("Terminal utilisation", f"{totals['utilisation_pct']:.0f}%",
                        delta_span("Send-out as a share of declared send-out capacity", "blue")))
    else:
        render(kpi_card("Terminal utilisation", "n/a", delta_span("Needs live GIE ALSI figures", "amber")))
with k3:
    if live and totals["fullness_pct"] is not None:
        render(kpi_card("LNG tank stocks", f"{totals['fullness_pct']:.0f}%",
                        delta_span(f"Of tank capacity across {totals['terminals_live']} reporting terminals",
                                   "blue")))
    else:
        render(kpi_card("LNG tank stocks", "n/a", delta_span("Needs live GIE ALSI figures", "amber")))
with k4:
    render(kpi_card("Shelf pipeline network", f"{_network_km:,.0f} km",
                    delta_span(f"{len(_pipelines)} export lines · {len(_shelf)} shelf facilities", "blue")))

if not live:
    render(callout(
        f"{alsi.get('message') or 'Live GIE ALSI figures are unavailable.'} "
        "Every LNG terminal still appears on the globe with its capacity; the other layers do not "
        "depend on the key.",
        "warn" if alsi.get("status") != "no_key" else "neutral",
    ))
elif alsi.get("errors"):
    st.caption(
        f"{len(alsi['errors'])} of {alsi.get('requested', 0)} GIE ALSI facilities returned no figure in "
        "this refresh; they are listed under Data and licences."
    )

# ── Globe ──────────────────────────────────────────────────────────────────────
render_energy_globe(globe_payload(layers, markers, alsi, view, GLOBE_VIEWS))
st.caption(
    "Gauge markers: size follows LNG send-out, the arc shows utilisation of declared send-out "
    "capacity, grey rings have no live figure. Live figures refresh every three hours (GIE ALSI "
    f"publishes one value per gas day). 3D engine: CesiumJS {CESIUM_VERSION}. Basemaps: Esri World "
    "Dark Gray, NASA VIIRS Black Marble night lights or Esri World Imagery."
)
_external = external_globe_url(view)
if _external:
    st.link_button("Open this view in the stand-alone globe", _external, icon=":material/open_in_new:")

# ── Tables ─────────────────────────────────────────────────────────────────────
section_header("Infrastructure data", "Everything on the globe, as tables.")
tab_lng, tab_choke, tab_pipes, tab_fac, tab_data = st.tabs(
    ["LNG terminals", "Chokepoints", "Export pipelines", "Shelf facilities", "Data and licences"]
)

with tab_lng:
    rows = []
    for m in markers:
        rows.append({
            "Terminal": m["name"],
            "Country": m.get("nation"),
            "Type": "FSRU" if m.get("fsru") else "Onshore",
            "Gas day": m.get("gas_day"),
            "Send-out (GWh/d)": m.get("send_out_gwh_d"),
            "Change d/d (GWh/d)": m.get("send_out_change_gwh_d"),
            "Utilisation (%)": m.get("utilisation_pct"),
            "Stock fullness (%)": m.get("fullness_pct"),
            "Inventory (GWh)": m.get("inventory_gwh"),
            "Capacity (bcm/y)": m.get("capacity_bcm_y"),
            "Live figure": (m.get("data_status") or "reported").capitalize() if m.get("live") else m.get("live_note"),
        })
    lng_df = pd.DataFrame(rows)
    # Without live figures the live columns are empty for every terminal: leave them out.
    _live_cols = ["Gas day", "Send-out (GWh/d)", "Change d/d (GWh/d)", "Utilisation (%)",
                  "Stock fullness (%)", "Inventory (GWh)"]
    lng_df = lng_df.drop(columns=[c for c in _live_cols if lng_df[c].isna().all()])
    st.dataframe(
        lng_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Send-out (GWh/d)": st.column_config.NumberColumn(format="%.1f"),
            "Change d/d (GWh/d)": st.column_config.NumberColumn(format="%+.1f"),
            "Utilisation (%)": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
            "Stock fullness (%)": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
            "Inventory (GWh)": st.column_config.NumberColumn(format="%.0f"),
            "Capacity (bcm/y)": st.column_config.NumberColumn(format="%.1f"),
        },
    )
    st.caption(
        "Utilisation: send-out divided by DTRS (declared total reference send-out). Stock fullness: "
        "inventory divided by DTMI (declared total maximum inventory). Terminals sharing one marker "
        "(Wilhelmshaven 1 and 2) are summed. UK and Turkish terminals do not report to GIE ALSI; their "
        "capacity and start-up year come from Global Energy Monitor."
    )

with tab_choke:
    rows = []
    for f in layers["chokepoints"]:
        p = f["properties"]
        lng_text = (
            f"{p['lng_flow_bcf_d']:.1f}" if p.get("lng_flow_bcf_d") is not None
            else (f"below {p['lng_flow_below_bcf_d']:.1f}" if p.get("lng_flow_below_bcf_d") is not None else None)
        )
        rows.append({
            "Chokepoint": p.get("name"),
            "Region": p.get("region"),
            "Connects": p.get("connects"),
            "Oil (mb/d)": p.get("oil_flow_mb_d"),
            "Oil period": p.get("oil_flow_period"),
            "LNG (bcf/d)": lng_text,
            "LNG period": p.get("lng_flow_period"),
        })
    chokepoints_df = pd.DataFrame(rows).sort_values("Oil (mb/d)", ascending=False, na_position="last")
    st.dataframe(
        chokepoints_df, width="stretch", hide_index=True,
        column_config={"Oil (mb/d)": st.column_config.NumberColumn(format="%.1f")},
    )
    st.caption(
        "Source: U.S. Energy Information Administration, World Oil Transit Chokepoints (Mar 2026). "
        "Gates on the globe are approximate counting areas drawn for this layer, not legal or "
        "navigational boundaries."
    )

with tab_pipes:
    rows = []
    for f in _pipelines:
        p = f["properties"]
        rows.append({
            "Pipeline": p.get("name"),
            "System": p.get("system"),
            "Medium": PIPELINE_MEDIA.get(p.get("medium"), p.get("medium")),
            "Diameter (in)": p.get("diameter_in"),
            "Length (km)": p.get("length_km"),
            "From": p.get("from"),
            "To": p.get("to"),
            "Operator": p.get("operator"),
            "In service since": p.get("status_since"),
        })
    pipes_df = pd.DataFrame(rows).sort_values("Length (km)", ascending=False, na_position="last")
    st.dataframe(
        pipes_df, width="stretch", hide_index=True,
        column_config={"Length (km)": st.column_config.NumberColumn(format="%.1f")},
    )
    by_medium = pipes_df.groupby("Medium", dropna=False)["Length (km)"].sum().sort_values(ascending=False)
    st.caption(
        "Network length by medium: "
        + ", ".join(f"{medium} {km:,.0f} km" for medium, km in by_medium.items())
        + ". Source: Sodir FactMaps (NLOD); routes are the surveyed centre lines Sodir publishes."
    )

with tab_fac:
    rows = []
    for f in _shelf:
        p = f["properties"]
        rows.append({
            "Facility": p.get("name"),
            "Category": FACILITY_CATEGORIES.get(p.get("category"), p.get("category")),
            "Kind": p.get("kind"),
            "Field": p.get("field"),
            "Operator": p.get("operator"),
            "Status": "Under installation" if p.get("status") == "installation" else "Operating",
            "Start-up": p.get("startup"),
            "Water depth (m)": p.get("water_depth_m"),
            "_rank": p.get("display_rank", 0),
        })
    fac_df = (pd.DataFrame(rows).sort_values(["Category", "_rank", "Facility"], ascending=[True, False, True])
              .drop(columns="_rank"))
    st.dataframe(fac_df, width="stretch", hide_index=True)
    st.caption(
        "Fixed surface facilities in service or under installation on the Norwegian shelf (Sodir), "
        "onshore plants and landfalls in mainland Norway, and offshore wind farms."
    )

with tab_data:
    sources_rows = []
    for key in ("facilities", "pipelines", "chokepoints"):
        source = layers[f"{key}_source"]
        for name, licence in (source.get("licences") or {}).items():
            sources_rows.append({"Dataset": source.get("name") or key, "Source": name, "Licence and use": licence})
    sources_rows.extend([
        {"Dataset": "LNG Terminals (live)", "Source": "GIE ALSI",
         "Licence and use": "Send-out, inventory, DTRS and DTMI per facility and gas day; source referral GIE ALSI."},
        {"Dataset": "Basemap", "Source": "Esri World Dark Gray Base",
         "Licence and use": "Esri, HERE, Garmin, © OpenStreetMap contributors, and the GIS user community."},
        {"Dataset": "Basemap", "Source": "NASA GIBS, VIIRS Black Marble 2016",
         "Licence and use": "NASA Earth Observing System Data and Information System (EOSDIS); no use restrictions."},
        {"Dataset": "Basemap", "Source": "Esri World Imagery",
         "Licence and use": "Source: Esri, Vantor, Earthstar Geographics, and the GIS User Community."},
        {"Dataset": "3D engine", "Source": f"CesiumJS {CESIUM_VERSION}",
         "Licence and use": "Apache License 2.0; loaded from jsDelivr with Subresource Integrity."},
    ])
    st.dataframe(pd.DataFrame(sources_rows), width="stretch", hide_index=True)
    approx = layers["facilities_source"].get("approximate_locations") or []
    if approx:
        st.caption(
            f"{len(approx)} LNG terminal positions are approximate in the source (mostly FSRUs placed at "
            "the port or town); their details card says so."
        )
    if alsi.get("status") == "ok":
        st.caption(
            f"Last GIE ALSI refresh {alsi.get('fetched_at') or 'n/a'} (UTC): {alsi.get('requested', 0)} "
            f"facilities requested, latest gas day {alsi.get('gas_day') or 'n/a'}."
        )
    if alsi.get("excluded"):
        with st.expander(f"Facilities in the ALSI listing that were not requested ({len(alsi['excluded'])})"):
            st.dataframe(pd.DataFrame(alsi["excluded"]).rename(
                columns={"eic": "EIC", "name": "Facility", "reason": "Reason"}), width="stretch", hide_index=True)
    if alsi.get("errors"):
        with st.expander(f"Facilities without a figure in this refresh ({len(alsi['errors'])})"):
            st.dataframe(pd.DataFrame(alsi["errors"]).rename(
                columns={"eic": "EIC", "name": "Facility", "reason": "Reason"}), width="stretch", hide_index=True)
    st.caption("Dataset provenance, selection rules and checksums: data/globe/README.md in the repository.")

page_footer(
    [
        ("Offshore facilities and pipelines", "Sodir FactMaps (NLOD)", "https://factmaps.sodir.no"),
        ("LNG terminals", "Global Energy Monitor GGIT (CC BY 4.0)",
         "https://globalenergymonitor.org/projects/global-gas-infrastructure-tracker/"),
        ("Live LNG figures", "GIE ALSI", "https://alsi.gie.eu"),
        ("Chokepoint flows", "U.S. EIA (Mar 2026)",
         "https://www.eia.gov/international/analysis/special-topics/World_Oil_Transit_Chokepoints"),
    ],
    note="Positions marked approximate in the source are placed at the port or town. Flows are "
         "EIA period averages, not live vessel counts.",
)
