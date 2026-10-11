"""Energy Globe data: bundled datasets, GIE ALSI parsing and pacing, and the terminal join."""
import hashlib
import json
from pathlib import Path

import pytest

from data import energy_globe as eg
from utils.globe import GLOBE_VIEWS

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "alsi-sample.json"
TODAY = "2026-10-10"


@pytest.fixture(scope="module")
def sample():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def layers():
    return eg.load_layers()


def _units(sample) -> list[dict]:
    facilities, _ = eg.select_alsi_facilities(sample["listing"], today=TODAY)
    units = []
    for facility in facilities:
        body = sample["facilities"].get(facility["eic"])
        if body is not None:
            units.append(eg.build_alsi_unit(facility, eg.read_alsi_facility_records(body)))
    return units


# ── Bundled datasets ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("key", list(eg.DATASETS))
def test_runtime_files_match_their_recorded_checksum(key):
    folder, filename = eg.DATASETS[key]
    source = json.loads((eg.GLOBE_DATA_DIR / folder / "source.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256((eg.GLOBE_DATA_DIR / folder / filename).read_bytes()).hexdigest()
    assert digest == source["output_sha256"]


def test_layer_counts(layers):
    assert len(layers["facilities"]) == 162
    assert len(layers["pipelines"]) == 74
    assert len(layers["chokepoints"]) == 10
    lng = [f for f in layers["facilities"] if f["properties"]["category"] == "lng_import"]
    assert len(lng) == 38


def test_every_feature_has_an_id_name_and_valid_geometry(layers):
    for key in eg.DATASETS:
        ids = set()
        for feature in layers[key]:
            props = feature["properties"]
            assert props.get("id") and props.get("name"), props
            assert props["id"] not in ids, props["id"]
            ids.add(props["id"])
            geometry = feature["geometry"]
            if geometry["type"] == "Point":
                lon, lat = geometry["coordinates"][:2]
                assert -180 <= lon <= 180 and -90 <= lat <= 90, props["id"]


def test_load_layers_returns_an_independent_copy(layers):
    first = eg.load_layers()
    first["facilities"].clear()
    assert len(eg.load_layers()["facilities"]) == 162


# ── ALSI parsing ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("raw", "expected"),
    [("12.5", 12.5), (" 7 ", 7.0), ("-", None), ("", None), (None, None), ("null", None),
     ("abc", None), ("nan", None), (float("inf"), None), (True, None), (3, 3.0)],
)
def test_alsi_number(raw, expected):
    assert eg.alsi_number(raw) == expected


def test_listing_selection_rules(sample):
    facilities, excluded = eg.select_alsi_facilities(sample["listing"], today=TODAY)
    reasons = " | ".join(e["reason"] for e in excluded)
    assert facilities, "the fixture has facilities in operation"
    assert all(not f["country"].startswith("GB") for f in facilities)
    assert "18W000000000GVMT" not in {f["eic"] for f in facilities}
    assert "Virtual balancing" in reasons
    assert len({f["eic"] for f in facilities}) == len(facilities)
    keys = [(f["country"], f["name"], f["eic"]) for f in facilities]
    assert keys == sorted(keys)


def test_listing_skips_future_ceased_and_placeholder_dates():
    listing = [{"name": "Op", "country": "XX", "eic": "C1", "facilities": [
        {"eic": "A", "name": "Future", "country": "XX", "operational_start_date": "2030-01-01"},
        {"eic": "B", "name": "Ceased", "country": "XX", "operational_start_date": "2010-01-01",
         "operational_end_date": "2020-01-01"},
        {"eic": "C", "name": "Placeholder", "country": "XX", "operational_start_date": "3000-01-01"},
        {"eic": "D", "name": "Open-ended", "country": "XX", "operational_start_date": "2010-01-01",
         "operational_end_date": "3000-01-01"},
        {"eic": "E", "name": "No start", "country": "XX"},
    ]}]
    facilities, excluded = eg.select_alsi_facilities(listing, today=TODAY)
    assert [f["eic"] for f in facilities] == ["D"]
    assert facilities[0]["end"] is None
    assert {e["eic"] for e in excluded} == {"A", "B", "C", "E"}


def test_duplicate_listing_prefers_the_entry_without_an_end_date():
    listing = [{"name": "Op", "country": "IT", "eic": "C1", "facilities": [
        {"eic": "P", "name": "Old operator", "country": "IT", "operational_start_date": "2015-01-01",
         "operational_end_date": "2027-01-01"},
        {"eic": "P", "name": "New operator", "country": "IT", "operational_start_date": "2010-01-01"},
    ]}]
    facilities, excluded = eg.select_alsi_facilities(listing, today=TODAY)
    assert [f["name"] for f in facilities] == ["New operator"]
    assert "Duplicate listing" in excluded[0]["reason"]


def test_listing_input_is_validated():
    with pytest.raises(TypeError):
        eg.select_alsi_facilities({"not": "a list"}, today=TODAY)
    with pytest.raises(ValueError):
        eg.select_alsi_facilities([], today="yesterday")


def test_access_denied_is_an_auth_error(sample):
    with pytest.raises(eg.AlsiAuthError):
        eg.read_alsi_facility_records(sample["accessDenied"])
    with pytest.raises(ValueError):
        eg.read_alsi_facility_records({"error": "something else"})
    with pytest.raises(ValueError):
        eg.read_alsi_facility_records(["not", "a", "dict"])


def test_records_are_newest_first_and_dated(sample):
    body = next(iter(sample["facilities"].values()))
    records = eg.read_alsi_facility_records(body)
    days = [r["gasDayStart"] for r in records]
    assert days == sorted(days, reverse=True)


def test_unit_figures(sample):
    zeebrugge = next(u for u in _units(sample) if u["eic"] == "21W0000000001245")
    # Fixture, gas day 2026-10-08: send-out 24 (previous day 24.7), DTRS 724.1,
    # inventory 727.69 GWh of a 3,803.42 GWh DTMI, 108.29 thousand m3 of LNG.
    assert zeebrugge["gas_day"] == "2026-10-08"
    assert zeebrugge["send_out_gwh_d"] == 24.0
    assert zeebrugge["send_out_change_gwh_d"] == pytest.approx(-0.7)
    assert zeebrugge["utilisation_pct"] == pytest.approx(round(24 / 724.1 * 100, 1))
    assert zeebrugge["fullness_pct"] == pytest.approx(round(727.69 / 3803.42 * 100, 1))
    assert zeebrugge["inventory_lng_m3"] == 108290
    assert zeebrugge["data_status"] == "confirmed"
    assert zeebrugge["terminal_type"] == "onshore"


def test_vessel_name_is_read_from_the_facility_name():
    assert eg.alsi_vessel("Wilhelmshaven LNG Terminal 1 (FSRU Hoegh Esperanza)") == "FSRU Hoegh Esperanza"
    assert eg.alsi_vessel("Zeebrugge (Historical data prior to 1 March 2025)") is None


def test_dash_values_stay_missing():
    facility = {"eic": "X", "name": "X", "country": "BE", "operator": None, "type": "LAND"}
    unit = eg.build_alsi_unit(facility, [{"gasDayStart": "2026-10-08", "sendOut": "-", "dtrs": "-",
                                          "inventory": {"gwh": "-"}, "dtmi": {"gwh": "0"}}])
    assert unit["send_out_gwh_d"] is None and unit["utilisation_pct"] is None
    assert unit["fullness_pct"] is None and unit["send_out_change_gwh_d"] is None
    assert eg.build_alsi_unit(facility, []) is None


# ── Fetch pass (injected transport) ───────────────────────────────────────────

def _transport(sample, fail=None):
    """Fake ALSI: the fixture's bodies, and Zeebrugge's body for facilities it lacks."""
    filler = sample["facilities"]["21W0000000001245"]

    def get_json(url, key):
        if url == eg.ALSI_LISTING_URL:
            assert key is None, "the listing must be requested without the key"
            return sample["listing"]
        assert key == "test-key"
        eic = url.split("facility=")[1].split("&")[0]
        if fail and eic in fail:
            raise fail[eic]
        return sample["facilities"].get(eic, filler)
    return get_json


def test_pass_collects_units_and_errors(sample):
    facilities, _ = eg.select_alsi_facilities(sample["listing"], today=TODAY)
    broken = facilities[-1]["eic"]
    result = eg.run_alsi_pass("test-key", TODAY, workers=2,
                              get_json=_transport(sample, {broken: TimeoutError("slow")}))
    assert result["requested"] == len(facilities)
    assert len(result["units"]) == len(facilities) - 1
    assert result["gas_day"] == "2026-10-08"
    assert [e["eic"] for e in result["errors"]] == [broken]
    assert result["errors"][0]["reason"] == "TimeoutError"


def test_pass_stops_on_a_rejected_key(sample):
    facilities, _ = eg.select_alsi_facilities(sample["listing"], today=TODAY)
    fail = {facilities[0]["eic"]: eg.AlsiAuthError()}
    with pytest.raises(eg.AlsiAuthError):
        eg.run_alsi_pass("test-key", TODAY, get_json=_transport(sample, fail), workers=1)


def test_pass_without_enough_data_is_not_usable(sample):
    with pytest.raises(RuntimeError):
        eg.run_alsi_pass("test-key", TODAY, get_json=lambda url, key: (
            sample["listing"] if url == eg.ALSI_LISTING_URL else {"data": []}), workers=1)


def test_fetch_without_a_key_never_touches_the_network(monkeypatch):
    monkeypatch.delenv("AGSI_API_KEY", raising=False)

    def boom(*args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("network used without a key")

    monkeypatch.setattr(eg, "_get_json", boom)
    result = eg.fetch_alsi_terminals()
    assert result["status"] == "no_key" and result["units"] == []
    assert "AGSI_API_KEY" in result["message"]


def test_pacer_spaces_request_starts(monkeypatch):
    clock = {"t": 1000.0}
    sleeps = []
    monkeypatch.setattr(eg.time, "monotonic", lambda: clock["t"])

    def fake_sleep(seconds):
        sleeps.append(seconds)
        clock["t"] += seconds

    monkeypatch.setattr(eg.time, "sleep", fake_sleep)
    pacer = eg._Pacer(spacing_s=0.25, per_minute=3)
    for _ in range(4):
        pacer.wait()
    # Two spacing waits, then the fourth start waits for the first to leave the minute.
    assert sleeps[0] == pytest.approx(0.25) and sleeps[1] == pytest.approx(0.25)
    assert sum(sleeps) == pytest.approx(60.0, abs=0.01)


# ── Join to markers ───────────────────────────────────────────────────────────

def test_crosswalk_index_maps_eics_to_lng_markers(layers):
    index = eg.alsi_marker_index(layers["facilities"], layers["facilities_source"])
    lng_ids = {f["properties"]["id"] for f in layers["facilities"]
               if f["properties"]["category"] == "lng_import"}
    assert index and set(index.values()) <= lng_ids
    assert index["21W0000000000079"] == "gem-T0492"      # Rotterdam Gate
    assert index["21W0000000001245"] == "gem-T0426"      # Zeebrugge
    # Wilhelmshaven 2 is merged into the Wilhelmshaven 1 marker (one GEM point for both).
    assert index["21W000000000129W"] == index["37W000000000111J"] == "gem-T0458"


def test_markers_join_live_units(sample, layers):
    units = _units(sample)
    alsi = {"status": "ok", "units": units}
    markers = eg.lng_terminal_markers(layers["facilities"], layers["facilities_source"], alsi)
    assert len(markers) == 38
    live = [m for m in markers if m["live"]]
    assert {m["id"] for m in live} == {"gem-T0458", "gem-T0426"}   # Wilhelmshaven 1, Zeebrugge
    assert markers[0]["id"] == "gem-T0458", "live terminals sort first, by send-out"
    assert markers[0]["send_out_gwh_d"] == pytest.approx(136.1)
    assert all(m["live_note"] is None for m in live)
    static = [m for m in markers if not m["live"]]
    assert any(m["live_note"] == "Not reported to GIE ALSI" for m in static)   # UK, Türkiye


def test_units_sharing_a_marker_are_summed():
    units = [
        {"eic": "A", "send_out_gwh_d": 100.0, "send_out_change_gwh_d": 5.0, "dtrs_gwh_d": 200.0,
         "inventory_gwh": 50.0, "dtmi_gwh": 100.0, "inventory_lng_m3": 1000, "gas_day": "2026-10-08",
         "data_status": "confirmed", "updated_at": "2026-10-09 10:00:00", "name": "A"},
        {"eic": "B", "send_out_gwh_d": 50.0, "send_out_change_gwh_d": -2.0, "dtrs_gwh_d": 100.0,
         "inventory_gwh": 30.0, "dtmi_gwh": 100.0, "inventory_lng_m3": 500, "gas_day": "2026-10-08",
         "data_status": "estimated", "updated_at": "2026-10-09 11:00:00", "name": "B"},
    ]
    agg = eg._aggregate_units(units)
    assert agg["send_out_gwh_d"] == 150.0 and agg["send_out_change_gwh_d"] == 3.0
    assert agg["utilisation_pct"] == 50.0 and agg["fullness_pct"] == 40.0
    assert agg["inventory_lng_m3"] == 1500
    assert agg["data_status"] == "mixed: confirmed, estimated"
    assert agg["updated_at"] == "2026-10-09 11:00:00"
    assert len(agg["units"]) == 2


def test_unmatched_unit_gets_a_marker_at_alsi_coordinates(layers):
    unit = {"eic": "NEW-EIC", "name": "New FSRU", "country": "IT", "terminal_type": "FSRU",
            "send_out_gwh_d": 10.0, "alsi_lon": 12.0, "alsi_lat": 44.0, "gas_day": "2026-10-08"}
    markers = eg.lng_terminal_markers(layers["facilities"], layers["facilities_source"],
                                      {"status": "ok", "units": [unit]})
    new = next(m for m in markers if m["id"] == "alsi-NEW-EIC")
    assert (new["lon"], new["lat"]) == (12.0, 44.0)
    assert new["nation"] == "Italy" and new["fsru"] and new["live"]


def test_totals_need_every_terminal_for_the_daily_change():
    markers = [
        {"live": True, "send_out_gwh_d": 100.0, "send_out_change_gwh_d": 4.0, "dtrs_gwh_d": 400.0,
         "inventory_gwh": 10.0, "dtmi_gwh": 20.0},
        {"live": True, "send_out_gwh_d": 50.0, "send_out_change_gwh_d": None, "dtrs_gwh_d": 100.0,
         "inventory_gwh": None, "dtmi_gwh": None},
        {"live": False},
    ]
    totals = eg.lng_totals(markers)
    assert totals["send_out_gwh_d"] == 150.0
    assert totals["send_out_change_gwh_d"] is None
    assert totals["utilisation_pct"] == 30.0
    assert totals["fullness_pct"] == 50.0
    assert totals["terminals_live"] == 2 and totals["terminals_total"] == 3


# ── Payload ───────────────────────────────────────────────────────────────────

def test_payload_shape(layers):
    alsi = {"status": "no_key", "message": "x", "units": []}
    markers = eg.lng_terminal_markers(layers["facilities"], layers["facilities_source"], alsi)
    payload = eg.globe_payload(layers, markers, alsi, "hormuz", GLOBE_VIEWS)
    assert payload["view"] == "hormuz"
    assert len(payload["facilities"]) == 124          # LNG terminals are their own layer
    assert len(payload["pipelines"]) == 66 and len(payload["receiving"]) == 8
    assert len(payload["lng"]) == 38 and len(payload["chokepoints"]) == 10
    assert set(payload["views"]) == set(GLOBE_VIEWS)
    assert all(len(p["parts"]) >= 1 for p in payload["pipelines"])
    assert all(len(c["label"]) == 2 and c["rings"] for c in payload["chokepoints"])
    json.dumps(payload, allow_nan=False)
    assert eg.globe_payload(layers, markers, alsi, "nope", GLOBE_VIEWS)["view"] in GLOBE_VIEWS
