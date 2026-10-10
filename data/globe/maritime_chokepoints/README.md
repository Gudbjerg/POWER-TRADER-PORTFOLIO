# Maritime chokepoints

Counting gates for the **Chokepoints** layer (`maritime-chokepoints`): ten
narrow sea passages and canals that carry a large share of the world's oil and
liquefied natural gas (LNG) trade, with the oil and LNG flows the U.S. Energy
Information Administration (EIA) reports for them.

- Runtime file: `chokepoints.geojsonl` (one GeoJSON feature per line:
  `Polygon`, or `MultiPolygon` for the Turkish and Danish Straits)
- Provenance, gate centre lines, land checks and the cited EIA rows:
  [`source.json`](source.json)
- Build script: [`scripts/build-maritime-chokepoints.py`](../../../../scripts/build-maritime-chokepoints.py)

The app serves this file through `/api/vessels/chokepoints.geojsonl`, which adds
live counts of the vessels the server's AISStream cache holds inside each gate
(see "Live counts" below). The counts are not part of this dataset.

| Source | Used for | Licence and attribution |
|---|---|---|
| Gates drawn for this layer (`GATES` in the build script) | The counting areas and label points | Drawn for this project; no third-party data. |
| [EIA, World Oil Transit Chokepoints](https://www.eia.gov/international/analysis/special-topics/World_Oil_Transit_Chokepoints), last updated March 3, 2026 | Oil and LNG flows (Table 1 and the LNG rows of Tables 2 to 8) | Public domain (U.S. government publication). EIA asks for an acknowledgment with the publication date: "Source: U.S. Energy Information Administration (Mar 2026)". |
| [Natural Earth](https://www.naturalearthdata.com) 10 m countries, bundled in `../natural_earth/countries.json` | Checking that open-sea centre lines lie at sea | Public domain. Nothing is copied. |

## Gates

Each gate is a corridor of a stated half-width either side of a hand-placed
centre line along the main traffic route through the narrows, with a label
point at the narrowest or most representative place. They are approximate
counting areas, not legal, navigational or EIA boundaries.

| Gate | Half-width | Covers |
|---|---|---|
| Strait of Hormuz | 14 km | Around the Musandam peninsula, between it and Larak and Qeshm |
| Bab el-Mandeb | 7 km | The main channel west of Perim island |
| Suez Canal | 3 km | Port Said to Suez, including the Bitter Lakes |
| Strait of Malacca | 12 km | Between One Fathom Bank and Malacca, clear of the Port Klang and Singapore anchorages |
| Turkish Straits | 1.5 km and 2 km | The Bosporus and the Dardanelles (two parts) |
| Danish Straits | 3 km and 4 km | The Øresund and the Great Belt (two parts); the Little Belt is not included |
| Strait of Gibraltar | 6 km | Cape Spartel to Gibraltar and Ceuta, through the narrows off Tarifa |
| Panama Canal | 2.5 km | Colón to Balboa, through Gatún Lake and the Culebra Cut |
| Cape of Good Hope | 30 km | The shipping route off the Cape Peninsula and Cape Agulhas |
| Dover Strait | 10 km | Between Dover and Cap Gris-Nez, along the traffic separation scheme |

The build samples every open-sea centre line each kilometre and fails if any
sample lies inside a Natural Earth country outline. The two canals, and the
Bosporus and Dardanelles (narrower than that outline's 0.01 degree
resolution), are exempt and were checked against imagery instead.

## EIA figures

`oil_flow_mb_d` is the latest column of EIA Table 1, in million barrels per
day of crude oil and petroleum liquids, and `lng_flow_bcf_d` the latest value
of the matching LNG row of Tables 2 to 8, in billion cubic feet per day.
`oil_flow_period` and `lng_flow_period` name the period: `1H25`, the first half
of 2025, or `FY2025` for the Panama Canal, which EIA reports by fiscal year
(October 2024 to September 2025). Notes:

- Suez Canal: the oil figure includes the SUMED pipeline (2.5 million b/d
  capacity), whose crude oil does not cross the canal by ship.
- Turkish Straits: EIA's Table 1 figure is for the Dardanelles; the Bosporus
  figure is `oil_flow_bosporus_mb_d`. LNG carriers may not pass the Bosporus,
  so the LNG figure is for the Dardanelles.
- Danish Straits: the figures exclude the Kiel Canal.
- Panama Canal: EIA gives no LNG value, only that LNG cargoes "declined to less
  than 0.3 Bcf/d" in FY 2025, stored as `lng_flow_below_bcf_d`.
- Strait of Gibraltar and Dover Strait: not covered by the EIA report, so they
  carry no figures. No figure from any other source is substituted.

The build reads these values from the page content the EIA site loads and
fails when the page's "Last Updated" date changes, so a newer edition is
reviewed before its figures are adopted.

## Properties

`id`, `name`, `kind` (`strait`, `straits`, `canal` or `route`), `region`,
`connects`, `label_point` (`[lon, lat]`), `gate` (what the gate covers),
`gate_half_width_km` (the widest part), `notes`, and where EIA reports:
`oil_flow_mb_d`, `oil_flow_period`, `oil_flow_bosporus_mb_d`,
`lng_flow_bcf_d` or `lng_flow_below_bcf_d`, `lng_flow_period`, `flows_source`.

## Live counts

With `AISSTREAM_API_KEY` set, the route adds `vessels_total`, `tankers` (AIS
ship types 80 to 89), `lng_carriers` (tankers whose name contains "LNG" as a
word), `cargo` (70 to 79), `other_vessels` (other valid types),
`unknown_type` (no type reported yet), `underway` (at least 1 knot),
`stationary`, `speed_unknown` and `counts_window_min` (30: vessels with a
position in the last 30 minutes). Limitations:

- AISStream relays terrestrial AIS receivers; coverage varies by region and
  is thin far from shore, so counts are a lower bound of the vessels present.
- Ship type arrives in separate static messages, so recently seen vessels are
  often `unknown_type`, and the LNG name heuristic misses carriers whose names
  do not contain "LNG".
- A gate counts every vessel inside it, including ones at anchor or in port
  areas the gate overlaps, not only vessels transiting.
- Counts come from the server's AIS cache, so they follow the AISStream
  subscription: the default covers the whole world, but with
  `AISSTREAM_BOUNDING_BOXES` narrowed to a region, gates outside it read zero
  rather than unknown. The cache keeps the 50,000 most recently updated
  vessels; when more report within 30 minutes, the oldest positions drop out
  first and the counts cover a shorter window.

Without the key, the route serves these gates unchanged with the
`X-GEV-Key-Required: aisstream` header, and the layer shows the gates and EIA
flows with a notice that live counts need the key.
