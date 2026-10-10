# Energy pipelines

Features for the **Energy Pipelines** layer (`local-energy-pipelines`): the
transportation pipelines the Norwegian Offshore Directorate (Sodir) publishes
for the Norwegian shelf, as routes, and the receiving terminals outside Norway
where they land, as points.

- Runtime file: `pipelines.geojsonl` (one GeoJSON feature per line:
  `LineString` or `MultiLineString` routes, `Point` receiving terminals)
- Provenance, counts, terminal cross-checks and exclusions: [`source.json`](source.json)
- Build script: [`scripts/build-energy-pipelines.py`](../../../../scripts/build-energy-pipelines.py)

This database is derived from one openly licensed source, separately from the
MIT code. Two further sources are used only to check positions; nothing from
them is redistributed.

| Source | Used for | Licence and attribution |
|---|---|---|
| [Sodir FactMaps](https://factmaps.sodir.no), layer 311 "Pipelines", WGS84 service | Pipeline routes and attributes | [Norwegian Licence for Open Government Data (NLOD)](https://data.norge.no/nlod/en). Contains data under the NLOD distributed by the Norwegian Offshore Directorate (Sodir). |
| Sodir FactMaps, layer 307 "All facilities", WGS84 service | Naming and checking receiving terminals | NLOD, as above. |
| [Global Energy Monitor](https://globalenergymonitor.org), Global Gas Infrastructure Tracker, Gas Pipelines, December 2023 release (Europe Gas Tracker 2024-05) | Independent check of each gas landfall (distance only) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Not redistributed. |
| Published terminal site coordinates (cited per terminal in `source.json`) | Independent check of three UK terminals (distance only) | Not redistributed. |

## Selection rules

- **Pipelines:** Sodir layer 311 records in phase `IN SERVICE` or
  `INSTALLATION`. Decommissioned and abandoned lines are excluded, as are six
  in-service or installation records Sodir publishes without geometry (Gina
  Krog, the Johan Sverdrup hot tap and Yggdrasil); all are listed in
  `source.json` with their reasons.
- **Geometry:** Sodir's own WGS84 coordinates (Sodir applies the ED50 to
  WGS84 transformation), rounded to 5 decimals, with consecutive duplicate
  vertices and parts of fewer than two distinct vertices dropped. Orientation
  is as published, so a route does not always run from `from` to `to`. Sodir
  states a detail level of about 50 m and omits track within 500 m of a
  facility.
- **Receiving terminals:** for each line whose Sodir destination is a delivery
  point outside Norway, the end vertex nearest that destination is the
  landfall. Landfalls are assigned to the nearest Sodir delivery point listed
  in the build script, and every terminal is checked against Sodir layer 307
  and at least one independent source (the matching GEM route, a published
  site coordinate, or both). A terminal is marked `approximate` when any check
  disagrees by more than 5 km. Europipe I and II land at Dornum although
  Sodir names Emden as their destination; Europipe I gas continues 48 km
  onshore to the Emden metering station.

## Properties

Routes: `id`, `name` (the Sodir map label; four lines without one are named in
`source.json`), `kind` (`pipeline`), `segment` (Sodir pipeline name),
`system` (Sodir system, e.g. `GASSLED`), `medium` (`gas`, `oil`, `condensate`
or `co2`), `diameter_in`, `max_water_depth_m`, `from`, `to`, `operator`,
`status` (`operating` or `installation`), `status_since` (date of the current
phase, Europe/Oslo), `length_km` (WGS84 ellipsoidal length of the published
track), `npdid`, `fact_page`, `display_rank` (0 to 3, label priority),
`location_accuracy` (`surveyed`, or `schematic` for the Baltic Pipe),
`source`, `licence`.

Receiving terminals: `id`, `name`, `kind` (`receiving terminal`), `systems`
(names of the lines that land there), `nation`, `display_rank`,
`location_accuracy` (`surveyed` or `approximate`), `source`, `licence`.

Absent values are omitted rather than written as null.

## Known limitations

- Only pipelines Sodir publishes for the Norwegian shelf are included. UK,
  Danish, Dutch and German pipelines, and onshore networks abroad, are not.
- Dunkerque and Teesside are `approximate`. For Dunkerque the Sodir layer 307
  point lies 14.6 km east of the pipeline end, while the GEM route and the
  Gassco terminal address (Mardyck, Loon-Plage) agree with the pipeline end.
  For Teesside the Sodir track and layer 307 point both lie about 7.4 km west
  of the Seal Sands oil terminal.
- The Baltic Pipe is a schematic Sodir route on a 0.01 degree grid, drawn
  straight across Denmark; its length is not a surveyed length.
- Receiving-terminal points sit at the pipeline landfall, which can be a few
  kilometres from the terminal plant itself (3 to 4 km at most terminals).
- Sixteen Sodir delivery points abroad remain without a terminal point because
  no published line ends there (for example Den Helder, Gdańsk and Sullom Voe);
  they are listed in `source.json`.
- `status_since` is Sodir's date for the current phase, which is not always the
  first start-up (the Baltic Pipe shows 2026-08-18).
- Sodir publishes "no operator" (`ingen oper`) for the Baltic Pipe and the
  Øygarden CO2 line; they carry no `operator` property (listed in
  `source.json`).

## Rebuild

Maintainer tool; requires Python 3.9+ and `openpyxl==3.1.5`.

```sh
python scripts/build-energy-pipelines.py --cache /tmp/energy-pipelines
python scripts/build-energy-pipelines.py --cache /tmp/energy-pipelines --offline --verify
```

The first command downloads all inputs into the cache and records their URLs,
retrieval time and SHA-256 in `fetch-manifest.json`; the GEM workbook is pinned
by checksum. An unknown phase or medium, a landfall far from every listed
terminal, or a terminal without an independent check fails the build until it
is reviewed.
