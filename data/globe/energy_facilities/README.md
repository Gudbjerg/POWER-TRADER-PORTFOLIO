# Energy facilities

Point features for the **Energy Facilities** layer (`local-energy-facilities`):
offshore oil and gas installations on the Norwegian continental shelf and
adjacent UK sector, onshore processing plants and pipeline landfalls in
mainland Norway, and LNG import terminals in operation across the European gas
market.

- Runtime file: `facilities.geojsonl` (one GeoJSON Point feature per line)
- Provenance, counts, exclusions and the ALSI crosswalk: [`source.json`](source.json)
- Build script: [`scripts/build-energy-facilities.py`](../../../../scripts/build-energy-facilities.py)

This database combines two openly licensed sources, separately from the MIT code.

| Source | Used for | Licence and attribution |
|---|---|---|
| [Sodir FactMaps](https://factmaps.sodir.no), layer 307 "All facilities", WGS84 service | Offshore installations, onshore plants and landfalls | [Norwegian Licence for Open Government Data (NLOD)](https://data.norge.no/nlod/en). Contains data under the NLOD distributed by the Norwegian Offshore Directorate (Sodir). |
| [Global Energy Monitor](https://globalenergymonitor.org), Global Gas Infrastructure Tracker, LNG Terminals, October 2023 release (Europe Gas Tracker 2024-05) | LNG terminal positions, capacity, start year, owners outside ALSI | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Global Energy Monitor, Global Gas Infrastructure Tracker, October 2023 release. Modified: filtered, aggregated per terminal, merged with ALSI status. |
| [GIE ALSI](https://alsi.gie.eu) facility listing | Which EU terminals operate today; names and operators of ALSI terminals | Operational dates and facility names only. Source: GIE ALSI. |

## Selection rules

- **Sodir:** fixed surface facilities in phase `IN SERVICE` or `INSTALLATION`.
  Coordinates are Sodir's own WGS84 values (Sodir applies the ED50 to WGS84
  transformation; the build never converts datums). Virtual trading points are
  dropped. Onshore records are kept only in mainland Norway; onshore delivery
  points elsewhere are deferred to the pipeline phase because several carry
  coordinate errors in the source (each is listed with a note in `source.json`).
- **LNG:** GEM import terminals in the European gas market (EU, UK, Norway,
  Türkiye, Western Balkans, Ukraine, Moldova, Switzerland). A terminal ALSI
  reports is included only while ALSI shows it in operation on the build date.
  A terminal outside ALSI is included when GEM lists it as operating with at
  least 1 bcm/y of operating capacity, which leaves out small bunkering and
  industrial depots.
- **Co-located records** that share one site point (25 m onshore and for LNG,
  10 m offshore) become one marker; the others are listed in its `colocated`
  property.

## Properties

`id`, `name`, `category` (`lng_import`, `onshore`, `offshore_platform`,
`offshore_storage`, `offshore_wind`), `kind`, `source_kind`, `functions`,
`status` (`operating` or `installation`), `operator`, `field` (Sodir field),
`system` (Sodir transport and utilisation facility, e.g. Gassled), `nation`,
`startup` (ISO date, or a year for GEM), `water_depth_m`, `capacity_bcm_y`,
`display_rank` (0 to 3, label priority), `location_accuracy` (`surveyed`,
`exact` or `approximate`), `source`, `source_ref`, `url`, `licence`, and for
LNG `status_source`, `unit` and `capacity_note` where they apply. Absent values
are omitted rather than written as null.

## Known limitations

- Twelve LNG positions are GEM `approximate` (mostly FSRUs placed at the port
  or town). Wilhelmshaven terminals 1 and 2 share one approximate GEM point and
  are shown as one marker. The list is in `source.json`
  (`approximate_locations`).
- LNG capacity and start years come from GEM's October 2023 release. Terminals
  that started later carry GEM's planned capacity with a `capacity_note`;
  Mukran has no capacity because GEM lists two ambiguous phases.
- Terminals outside ALSI (UK, Türkiye) use GEM's 2023 status.
- `nation` is Sodir's jurisdiction field, not a geocoded country.

## Rebuild

Maintainer tool; requires Python 3.9+ and `openpyxl==3.1.5`.

```sh
python scripts/build-energy-facilities.py --cache /tmp/energy-facilities
python scripts/build-energy-facilities.py --cache /tmp/energy-facilities --offline --verify
```

The first command downloads all inputs into the cache and records their URLs,
retrieval time and SHA-256 in `fetch-manifest.json`; the GEM workbook is pinned
by checksum. A new ALSI terminal fails the build until it is added to the
crosswalk or the exclusions in the script.
