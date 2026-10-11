# Energy Globe datasets

Static infrastructure layers for the **Energy Globe** page
([views/8_Energy_Globe.py](../../views/8_Energy_Globe.py)), loaded by
[data/energy_globe.py](../energy_globe.py).

| Folder | Runtime file | Features | Sources and licences |
|---|---|---|---|
| `energy_facilities/` | `facilities.geojsonl` | 162 points: 94 offshore installations, 5 offshore storage units, 14 onshore plants and landfalls, 11 offshore wind farms, 38 LNG import terminals | Sodir FactMaps (NLOD); Global Energy Monitor, Global Gas Infrastructure Tracker (CC BY 4.0); operating status from GIE ALSI |
| `energy_pipelines/` | `pipelines.geojsonl` | 66 Norwegian shelf export pipelines (gas, oil, condensate, CO2) and 8 receiving terminals | Sodir FactMaps (NLOD) |
| `maritime_chokepoints/` | `chokepoints.geojsonl` | 10 maritime chokepoints (counting gates drawn for the layer) with oil and LNG flows | U.S. Energy Information Administration, World Oil Transit Chokepoints, March 2026 (public domain) |

Each folder keeps the files exactly as built: the runtime GeoJSON Lines file, a
`README.md` with the selection rules, properties and known limitations, and a
`source.json` with input checksums, counts, exclusions and (for facilities) the
crosswalk from GIE ALSI facility EIC codes to the LNG terminal markers. The
`output_sha256` in each `source.json` matches the runtime file; the tests check it.

The datasets were built by the scripts `scripts/build-energy-facilities.py`,
`scripts/build-energy-pipelines.py` and `scripts/build-maritime-chokepoints.py` in the
companion globe repository (a fork of gods-eye-view, branch `feat/energy-phase2-3`,
commit `3579382`). Paths to scripts inside the per-folder READMEs refer to that
repository.

Live figures are not part of these files. The Energy Globe page joins GIE ALSI
send-out, stocks and capacities to the LNG terminal markers at run time (with
`AGSI_API_KEY`, the same GIE key the rest of the platform uses).

Attribution required when the data is shown:

- Contains data under the Norwegian Licence for Open Government Data (NLOD)
  distributed by the Norwegian Offshore Directorate (Sodir).
- Global Energy Monitor, Global Gas Infrastructure Tracker, October 2023 release,
  CC BY 4.0 (modified: filtered, aggregated per terminal, merged with ALSI status).
- Source: U.S. Energy Information Administration (Mar 2026).
- Source: GIE ALSI (live LNG terminal figures).
