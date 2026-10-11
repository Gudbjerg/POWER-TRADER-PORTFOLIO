"""
Energy Globe data: bundled infrastructure datasets and live LNG terminal figures.

Bundled datasets (``data/globe/``, provenance in ``data/globe/README.md``):

* ``energy_facilities/facilities.geojsonl``: 162 points. Norwegian continental shelf
  installations, storage units, onshore plants and landfalls (Sodir FactMaps, NLOD),
  offshore wind farms, and 38 LNG import terminals in operation across the European
  gas market (Global Energy Monitor GGIT, CC BY 4.0; operating status from GIE ALSI).
* ``energy_pipelines/pipelines.geojsonl``: 66 Norwegian shelf export pipelines and
  8 receiving terminals (Sodir FactMaps, NLOD).
* ``maritime_chokepoints/chokepoints.geojsonl``: 10 maritime chokepoints with the oil
  and LNG flows published by the U.S. Energy Information Administration (public domain).

Live figures: GIE ALSI publishes send-out, stocks and capacities per LNG facility and
gas day. ``fetch_alsi_terminals`` requests the public facility listing and then each
facility in operation; the GIE key travels only in the ``x-key`` header. ``AGSI_API_KEY``
holds it, because one GIE account covers both AGSI+ and ALSI. ``lng_terminal_markers``
joins those figures to the bundled terminal markers through the EIC crosswalk in
``energy_facilities/source.json``.

Units follow the ALSI data definitions: send-out and DTRS (declared total reference
send-out) in GWh per gas day; inventory and DTMI (declared total maximum inventory) in
GWh at gross calorific value; LNG volumes in cubic metres of LNG. A gas day runs from
05:00 to 05:00 UTC in winter (04:00 in summer).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode

import requests
import streamlit as st

from config.settings import (
    GLOBE_ALSI_DEADLINE_S,
    GLOBE_ALSI_MIN_SHARE,
    GLOBE_ALSI_PER_MINUTE,
    GLOBE_ALSI_SPACING_S,
    GLOBE_ALSI_TIMEOUT_S,
    GLOBE_ALSI_TTL_S,
    GLOBE_ALSI_WORKERS,
)

# ══════════════════════════════════════════════════════════════════════════════
# Bundled datasets
# ══════════════════════════════════════════════════════════════════════════════

GLOBE_DATA_DIR = Path(__file__).resolve().parent / "globe"

# Layer key -> (folder, runtime file) under GLOBE_DATA_DIR.
DATASETS: dict[str, tuple[str, str]] = {
    "facilities":  ("energy_facilities", "facilities.geojsonl"),
    "pipelines":   ("energy_pipelines", "pipelines.geojsonl"),
    "chokepoints": ("maritime_chokepoints", "chokepoints.geojsonl"),
}

FACILITY_CATEGORIES: dict[str, str] = {
    "offshore_platform": "Offshore installation",
    "offshore_storage":  "Offshore storage",
    "onshore":           "Onshore plant or landfall",
    "offshore_wind":     "Offshore wind farm",
    "lng_import":        "LNG import terminal",
}

PIPELINE_MEDIA: dict[str, str] = {
    "gas": "Gas",
    "oil": "Oil",
    "condensate": "Condensate",
    "co2": "CO2",
}

_COORD_DIGITS = 5   # about 1 m; keeps the embedded payload small


def read_geojsonl(path: Path) -> list[dict]:
    """Parse a GeoJSON Lines file: one Feature per non-empty line."""
    features: list[dict] = []
    with Path(path).open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict) or obj.get("type") != "Feature" \
                    or not isinstance(obj.get("geometry"), dict):
                raise ValueError(f"{Path(path).name}:{number}: not a GeoJSON Feature")
            features.append(obj)
    return features


@lru_cache(maxsize=1)
def _bundled_json() -> str:
    """The bundled layers and their provenance, parsed once and frozen as JSON text."""
    out: dict[str, Any] = {}
    for key, (folder, filename) in DATASETS.items():
        out[key] = read_geojsonl(GLOBE_DATA_DIR / folder / filename)
        out[f"{key}_source"] = json.loads(
            (GLOBE_DATA_DIR / folder / "source.json").read_text(encoding="utf-8")
        )
    return json.dumps(out)


def load_layers() -> dict[str, Any]:
    """Return a fresh copy of the bundled layers.

    Keys: ``facilities``, ``pipelines``, ``chokepoints`` (lists of GeoJSON Features) and
    ``facilities_source``, ``pipelines_source``, ``chokepoints_source`` (the parsed
    ``source.json`` of each dataset). Callers may modify the result freely.
    """
    return json.loads(_bundled_json())


def _point(feature: dict) -> tuple[float | None, float | None]:
    geometry = feature.get("geometry") or {}
    if geometry.get("type") != "Point":
        return None, None
    coords = geometry.get("coordinates") or []
    if len(coords) < 2:
        return None, None
    lon, lat = coords[0], coords[1]
    if not (_finite(lon) and _finite(lat)) or abs(lon) > 180 or abs(lat) > 90:
        return None, None
    return float(lon), float(lat)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _round_coords(coords: Iterable) -> list[list[float]]:
    return [[round(float(c[0]), _COORD_DIGITS), round(float(c[1]), _COORD_DIGITS)] for c in coords]


# ══════════════════════════════════════════════════════════════════════════════
# GIE ALSI: parsing (pure functions, shared by the fetcher and the tests)
# ══════════════════════════════════════════════════════════════════════════════

ALSI_ORIGIN = "https://alsi.gie.eu"
ALSI_LISTING_URL = f"{ALSI_ORIGIN}/api/about?show=listing"   # public, no key
ALSI_DATA_URL = f"{ALSI_ORIGIN}/api"                          # per facility, needs the key

# Facilities ALSI lists that are not physical terminals.
ALSI_EXCLUDED_FACILITIES: dict[str, str] = {
    "18W000000000GVMT": "Virtual balancing LNG tank, not a physical terminal",
}
_UK_PREFIX = "GB"          # UK terminals stopped reporting to ALSI after Brexit (GB and GB*)
_PLACEHOLDER_YEAR = 2999   # ALSI writes year 3000 for "no date published"
_DATA_STATUS = {"C": "confirmed", "E": "estimated", "N": "no data"}
_DAY_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")
_VESSEL_RE = re.compile(r"\((FSRU\b[^()]*)\)\s*$")

# ISO country codes ALSI uses, for terminals placed from ALSI's own coordinates.
ALSI_COUNTRIES: dict[str, str] = {
    "BE": "Belgium", "CY": "Cyprus", "DE": "Germany", "EE": "Estonia", "ES": "Spain",
    "FI": "Finland", "FR": "France", "GR": "Greece", "HR": "Croatia", "IE": "Ireland",
    "IT": "Italy", "LT": "Lithuania", "LV": "Latvia", "MT": "Malta", "NL": "Netherlands",
    "PL": "Poland", "PT": "Portugal", "SE": "Sweden",
}


class AlsiAuthError(RuntimeError):
    """ALSI rejected the API key (HTTP 401 or 403, or HTTP 200 with 'access denied')."""

    def __init__(self, message: str = "ALSI rejected the API key") -> None:
        super().__init__(message)


class AlsiRateLimited(RuntimeError):
    """ALSI answered HTTP 429."""


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text and text not in ("null", "undefined") else None


def _iso_day(value: Any) -> str | None:
    """'YYYY-MM-DD' (or a longer ISO string) as a validated day string, or None."""
    match = _DAY_RE.match(_text(value) or "")
    if not match:
        return None
    try:
        return date(int(match[1]), int(match[2]), int(match[3])).isoformat()
    except ValueError:
        return None


def alsi_number(value: Any) -> float | None:
    """An ALSI number: published as a string, '-' when there is no data."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None
    text = _text(value)
    if not text or text == "-":
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def alsi_vessel(name: Any) -> str | None:
    """The floating unit an ALSI facility name ends with, e.g. 'FSRU Hoegh Esperanza'."""
    match = _VESSEL_RE.search(_text(name) or "")
    return match[1].strip() if match else None


def select_alsi_facilities(listing: Any, today: str) -> tuple[list[dict], list[dict]]:
    """Choose the facilities to request from the public ALSI listing.

    Skipped, each with a reason: entries without an EIC, UK entries (GB and GB*), the
    Spanish virtual balancing tank, facilities whose start date is missing or a
    placeholder (year 3000), facilities that have not started yet, and facilities whose
    operational end date has passed (a placeholder end date means none is published).
    A facility listed twice (Panigaglia after an operator change) is requested once,
    preferring the entry without an end date, then the latest start.

    Returns ``(facilities, excluded)``; facilities are sorted by country, name and EIC.
    """
    if not isinstance(listing, list):
        raise TypeError("ALSI listing must be a list of companies")
    day = _iso_day(today)
    if not day:
        raise ValueError("today must be a YYYY-MM-DD date")
    excluded: list[dict] = []
    candidates: dict[str, dict] = {}
    for company in listing:
        company = company if isinstance(company, dict) else {}
        facilities = company.get("facilities")
        for facility in facilities if isinstance(facilities, list) else []:
            if not isinstance(facility, dict):
                continue
            eic = _text(facility.get("eic"))
            name = _text(facility.get("name"))
            country = _text(facility.get("country")) or _text(company.get("country")) or ""

            def skip(reason: str, eic=eic, name=name) -> None:
                excluded.append({"eic": eic, "name": name, "reason": reason})

            if not eic:
                skip("No facility EIC published")
                continue
            if country.upper().startswith(_UK_PREFIX):
                skip("UK terminal; UK operators stopped reporting to ALSI after Brexit")
                continue
            if eic in ALSI_EXCLUDED_FACILITIES:
                skip(ALSI_EXCLUDED_FACILITIES[eic])
                continue
            start = _iso_day(facility.get("operational_start_date"))
            if not start:
                skip("No operational start date published")
                continue
            if int(start[:4]) >= _PLACEHOLDER_YEAR:
                skip(f"Placeholder operational start date ({start})")
                continue
            if start > day:
                skip(f"Not in operation yet (starts {start})")
                continue
            end_raw = _iso_day(facility.get("operational_end_date"))
            end = end_raw if end_raw and int(end_raw[:4]) < _PLACEHOLDER_YEAR else None
            if end and end <= day:
                skip(f"Ceased operation ({end})")
                continue
            operator = _text(company.get("name"))
            record = {
                "eic": eic,
                "name": name,
                "type": _text(facility.get("type")),
                "country": country[:-1] if country.endswith("*") else country,
                "company": _text(facility.get("company")) or _text(company.get("eic")),
                "operator": re.sub(r"\s+", " ", operator) if operator else None,
                "start": start,
                "end": end,
            }
            previous = candidates.get(eic)
            if previous is None:
                candidates[eic] = record
                continue
            if (previous["end"] is None) != (record["end"] is None):
                keep = previous if previous["end"] is None else record
            else:
                keep = previous if previous["start"] >= record["start"] else record
            dropped = record if keep is previous else previous
            excluded.append({
                "eic": eic,
                "name": dropped["name"],
                "reason": f"Duplicate listing of {eic}; requested once as {keep['name']}",
            })
            candidates[eic] = keep
    facilities_out = sorted(
        candidates.values(),
        key=lambda f: (f["country"], str(f["name"] or ""), f["eic"]),
    )
    return facilities_out, excluded


def alsi_facility_url(facility: dict, origin: str = ALSI_ORIGIN) -> str:
    """Per-facility request for the latest two gas days (for the day-on-day change)."""
    params = urlencode({
        "country": facility["country"],
        "company": facility["company"],
        "facility": facility["eic"],
        "size": "2",
    })
    return f"{origin.rstrip('/')}/api?{params}"


def read_alsi_facility_records(body: Any) -> list[dict]:
    """Daily records of one facility response, newest gas day first.

    ALSI answers a missing or invalid key with HTTP 200 and ``{"error": "access denied"}``;
    that raises ``AlsiAuthError`` rather than reading as an empty facility.
    """
    if not isinstance(body, dict):
        raise ValueError("Malformed ALSI response")
    error = _text(body.get("error"))
    if error:
        if re.search(r"access denied|api key", f"{error} {_text(body.get('message')) or ''}", re.I):
            raise AlsiAuthError()
        raise ValueError(f"ALSI error: {error}")
    data = body.get("data")
    if not isinstance(data, list):
        raise ValueError("Malformed ALSI response")
    records = [r for r in data if isinstance(r, dict) and _iso_day(r.get("gasDayStart"))]
    records.sort(key=lambda r: _iso_day(r["gasDayStart"]), reverse=True)
    return records


def build_alsi_unit(facility: dict, records: list[dict]) -> dict | None:
    """One ALSI facility's latest figures, or None without records.

    Values ALSI publishes as '-' stay None rather than becoming zero. ``alsi_lon`` and
    ``alsi_lat`` carry ALSI's own coordinates when they are valid.
    """
    latest = records[0] if records else None
    if not latest:
        return None
    gas_day = _iso_day(latest.get("gasDayStart"))
    previous = None
    if gas_day:
        previous_day = (date.fromisoformat(gas_day) - timedelta(days=1)).isoformat()
        previous = next((r for r in records if _iso_day(r.get("gasDayStart")) == previous_day), None)
    send_out = alsi_number(latest.get("sendOut"))
    previous_send_out = alsi_number(previous.get("sendOut")) if previous else None
    dtrs = alsi_number(latest.get("dtrs"))
    inventory = latest.get("inventory") if isinstance(latest.get("inventory"), dict) else {}
    dtmi = latest.get("dtmi") if isinstance(latest.get("dtmi"), dict) else {}
    inventory_gwh = alsi_number(inventory.get("gwh"))
    inventory_lng = alsi_number(inventory.get("lng"))     # thousand m3 of LNG
    dtmi_gwh = alsi_number(dtmi.get("gwh"))
    kind = _text(latest.get("type")) or facility.get("type")
    name = _text(latest.get("name")) or facility.get("name")
    lon = alsi_number(latest.get("longitude"))
    lat = alsi_number(latest.get("latitude"))
    has_position = lon is not None and lat is not None and abs(lon) <= 180 and abs(lat) <= 90
    return {
        "eic": facility["eic"],
        "name": name,
        "vessel": alsi_vessel(name),
        "country": facility.get("country"),
        "operator": facility.get("operator"),
        "terminal_type": "FSRU" if kind == "FSRU" else "onshore",
        "gas_day": gas_day,
        "send_out_gwh_d": send_out,
        "send_out_change_gwh_d": (
            round(send_out - previous_send_out, 1)
            if send_out is not None and previous_send_out is not None else None
        ),
        "utilisation_pct": (
            round(send_out / dtrs * 100.0, 1) if send_out is not None and dtrs and dtrs > 0 else None
        ),
        "inventory_gwh": inventory_gwh,
        "inventory_lng_m3": round(inventory_lng * 1000) if inventory_lng is not None else None,
        "fullness_pct": (
            round(inventory_gwh / dtmi_gwh * 100.0, 1)
            if inventory_gwh is not None and dtmi_gwh and dtmi_gwh > 0 else None
        ),
        "dtrs_gwh_d": dtrs,
        "dtmi_gwh": dtmi_gwh,
        "data_status": _DATA_STATUS.get((_text(latest.get("status")) or "").upper()),
        "updated_at": _text(latest.get("updatedAt")),
        "alsi_lon": lon if has_position else None,
        "alsi_lat": lat if has_position else None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# GIE ALSI: fetching (paced, deadline-bound, cached)
# ══════════════════════════════════════════════════════════════════════════════

class _Pacer:
    """Spaces request starts and keeps a rolling per-minute budget, process-wide."""

    def __init__(self, spacing_s: float, per_minute: int) -> None:
        self._lock = threading.Lock()
        self._starts: deque[float] = deque()
        self._spacing = float(spacing_s)
        self._per_minute = int(per_minute)

    def wait(self) -> None:
        with self._lock:
            while True:
                now = time.monotonic()
                while self._starts and self._starts[0] <= now - 60.0:
                    self._starts.popleft()
                gap = self._starts[-1] + self._spacing - now if self._starts else 0.0
                budget = (
                    self._starts[0] + 60.0 - now
                    if len(self._starts) >= self._per_minute else 0.0
                )
                wait = max(gap, budget)
                if wait <= 0:
                    break
                time.sleep(min(wait, 5.0))
            self._starts.append(time.monotonic())


_PACER = _Pacer(GLOBE_ALSI_SPACING_S, GLOBE_ALSI_PER_MINUTE)
_SESSIONS = threading.local()


def _session() -> requests.Session:
    session = getattr(_SESSIONS, "session", None)
    if session is None:
        session = requests.Session()
        _SESSIONS.session = session
    return session


def _describe(exc: BaseException) -> str:
    """Short, key-free description of a request failure."""
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return f"HTTP {exc.response.status_code}"
    if isinstance(exc, requests.Timeout):
        return "request timed out"
    if isinstance(exc, requests.ConnectionError):
        return "connection failed"
    if isinstance(exc, (ValueError, RuntimeError)) and str(exc):
        return str(exc)
    return type(exc).__name__


def _get_json(url: str, key: str | None) -> Any:
    """One ALSI request. The key travels only in the ``x-key`` header and is never logged."""
    _PACER.wait()
    headers = {"Accept": "application/json"}
    if key:
        headers["x-key"] = key
    response = _session().get(url, headers=headers, timeout=(10, GLOBE_ALSI_TIMEOUT_S))
    if response.status_code in (401, 403):
        raise AlsiAuthError(f"ALSI rejected the API key (HTTP {response.status_code})")
    if response.status_code == 429:
        raise AlsiRateLimited("ALSI rate limit reached (HTTP 429)")
    response.raise_for_status()
    try:
        return response.json()
    except ValueError as exc:
        raise ValueError("Malformed ALSI response") from exc


def _fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def run_alsi_pass(key: str, today: str, get_json=_get_json,
                  deadline_s: float = GLOBE_ALSI_DEADLINE_S,
                  workers: int = GLOBE_ALSI_WORKERS) -> dict:
    """Fetch the listing and every facility in operation; raise when the pass is unusable.

    A rejected key or a rate limit stops the pass (``AlsiAuthError`` / ``AlsiRateLimited``).
    Other per-facility failures are recorded in ``errors``. The pass raises when fewer
    than ``GLOBE_ALSI_MIN_SHARE`` of the facilities returned data, so that a broken
    refresh is never cached. ``get_json`` is injectable for tests.
    """
    deadline = time.monotonic() + deadline_s
    listing = get_json(ALSI_LISTING_URL, None)
    facilities, excluded = select_alsi_facilities(listing, today=today)
    if not facilities:
        raise ValueError("The ALSI listing has no facility in operation")
    stop = threading.Event()
    fatal: list[BaseException] = []

    def one(facility: dict) -> tuple[dict, dict | None, str | None]:
        if stop.is_set():
            return facility, None, "Skipped after ALSI refused an earlier request"
        if time.monotonic() > deadline:
            return facility, None, "Time budget for this refresh exceeded"
        try:
            records = read_alsi_facility_records(get_json(alsi_facility_url(facility), key))
        except (AlsiAuthError, AlsiRateLimited) as exc:
            stop.set()
            fatal.append(exc)
            return facility, None, _describe(exc)
        except Exception as exc:  # one facility failing must not sink the pass
            return facility, None, _describe(exc)
        unit = build_alsi_unit(facility, records)
        return facility, unit, None if unit else "No data returned"

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        results = list(pool.map(one, facilities))
    if fatal:
        raise fatal[0]
    units: list[dict] = []
    errors: list[dict] = []
    for facility, unit, reason in results:
        if unit is not None:
            units.append(unit)
        else:
            errors.append({"eic": facility["eic"], "name": facility["name"],
                           "reason": reason or "No data returned"})
    if not units or len(units) < GLOBE_ALSI_MIN_SHARE * len(facilities):
        raise RuntimeError(f"only {len(units)} of {len(facilities)} facilities returned data")
    return {
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gas_day": max((u["gas_day"] for u in units if u.get("gas_day")), default=None),
        "requested": len(facilities),
        "units": units,
        "excluded": excluded,
        "errors": errors,
    }


@st.cache_data(ttl=GLOBE_ALSI_TTL_S, show_spinner=False)
def _alsi_pass_cached(key_fingerprint: str, today: str) -> dict:
    # The fingerprint only partitions the cache per key; the key itself is read here so
    # that it never becomes part of a cache key.
    del key_fingerprint
    return run_alsi_pass(os.getenv("AGSI_API_KEY", "").strip(), today)


def fetch_alsi_terminals() -> dict:
    """Live GIE ALSI figures for every EU LNG facility in operation, cached for three hours.

    Returns a dict with ``status`` ('ok', 'no_key', 'auth_failed', 'rate_limited' or
    'error'), ``message`` (plain text for the page, None when 'ok'), ``units`` (one record
    per ALSI facility, see ``build_alsi_unit``), ``excluded`` and ``errors`` (facility,
    reason), ``gas_day``, ``fetched_at`` (ISO UTC) and ``requested``. Never raises.
    """
    base = {"units": [], "excluded": [], "errors": [], "gas_day": None,
            "fetched_at": None, "requested": 0}
    key = os.getenv("AGSI_API_KEY", "").strip()
    if not key:
        return {**base, "status": "no_key",
                "message": "Add AGSI_API_KEY (the free GIE key) to load live terminal figures."}
    today = datetime.now(timezone.utc).date().isoformat()
    try:
        data = _alsi_pass_cached(_fingerprint(key), today)
    except AlsiAuthError:
        return {**base, "status": "auth_failed", "message": "GIE ALSI rejected the API key."}
    except AlsiRateLimited:
        return {**base, "status": "rate_limited",
                "message": "GIE ALSI rate limit reached; live figures return on a later refresh."}
    except Exception as exc:  # network, upstream or parsing failure: degrade to static data
        return {**base, "status": "error",
                "message": f"GIE ALSI is unavailable right now ({_describe(exc)})."}
    return {**data, "status": "ok", "message": None}


# ══════════════════════════════════════════════════════════════════════════════
# Joining live figures to the bundled LNG terminal markers
# ══════════════════════════════════════════════════════════════════════════════

def alsi_marker_index(facilities: list[dict], provenance: dict) -> dict[str, str]:
    """Map ALSI facility EIC to the id of the bundled LNG terminal marker it belongs to.

    EIC to GEM terminal id through the crosswalk in ``energy_facilities/source.json``,
    then to the LNG feature whose ``source_ref`` is that id. A GEM terminal merged into a
    co-located marker (Wilhelmshaven 2) maps to the marker whose ``colocated`` list names it.
    """
    by_gem: dict[str, str] = {}
    by_name: dict[str, str] = {}
    for feature in facilities or []:
        props = feature.get("properties") or {}
        if props.get("category") != "lng_import" or _point(feature)[0] is None:
            continue
        feature_id = _text(props.get("id"))
        if not feature_id:
            continue
        if _text(props.get("source_ref")):
            by_gem[props["source_ref"]] = feature_id
        for name in props.get("colocated") or []:
            by_name[name] = feature_id
    gem_names = {
        str(entry["id"])[4:]: entry.get("name")
        for entry in (provenance or {}).get("exclusions") or []
        if re.fullmatch(r"gem-T\d+", str(entry.get("id") or ""))
    }
    index: dict[str, str] = {}
    for entry in (provenance or {}).get("alsi_crosswalk") or []:
        eic, gem = _text(entry.get("eic")), _text(entry.get("gem"))
        if not eic or not gem or eic in index:
            continue
        target = by_gem.get(gem) or by_name.get(gem_names.get(gem))
        if target:
            index[eic] = target
    return index


def _aggregate_units(units: list[dict]) -> dict:
    """Combine the ALSI units joined to one marker (Wilhelmshaven 1 and 2 share a marker)."""
    if not units:
        return {
            "live": False, "gas_day": None, "mixed_days": False, "send_out_gwh_d": None,
            "send_out_change_gwh_d": None, "utilisation_pct": None, "fullness_pct": None,
            "inventory_gwh": None, "inventory_lng_m3": None, "dtrs_gwh_d": None,
            "dtmi_gwh": None, "data_status": None, "updated_at": None, "units": [],
        }

    def total(field: str) -> float | None:
        values = [u[field] for u in units if u.get(field) is not None]
        return round(sum(values), 2) if values else None

    with_send = [u for u in units if u.get("send_out_gwh_d") is not None]
    change = (
        round(sum(u["send_out_change_gwh_d"] for u in with_send), 1)
        if with_send and all(u.get("send_out_change_gwh_d") is not None for u in with_send)
        else None
    )
    util_pairs = [(u["send_out_gwh_d"], u["dtrs_gwh_d"]) for u in units
                  if u.get("send_out_gwh_d") is not None and (u.get("dtrs_gwh_d") or 0) > 0]
    full_pairs = [(u["inventory_gwh"], u["dtmi_gwh"]) for u in units
                  if u.get("inventory_gwh") is not None and (u.get("dtmi_gwh") or 0) > 0]
    statuses = sorted({u["data_status"] for u in units if u.get("data_status")})
    days = {u["gas_day"] for u in units if u.get("gas_day")}
    lng_m3 = total("inventory_lng_m3")
    return {
        "live": True,
        "gas_day": max(days) if days else None,
        "mixed_days": len(days) > 1,
        "send_out_gwh_d": total("send_out_gwh_d"),
        "send_out_change_gwh_d": change,
        "utilisation_pct": (
            round(sum(s for s, _ in util_pairs) / sum(d for _, d in util_pairs) * 100.0, 1)
            if util_pairs else None
        ),
        "fullness_pct": (
            round(sum(i for i, _ in full_pairs) / sum(d for _, d in full_pairs) * 100.0, 1)
            if full_pairs else None
        ),
        "inventory_gwh": total("inventory_gwh"),
        "inventory_lng_m3": int(lng_m3) if lng_m3 is not None else None,
        "dtrs_gwh_d": total("dtrs_gwh_d"),
        "dtmi_gwh": total("dtmi_gwh"),
        "data_status": (statuses[0] if len(statuses) == 1
                        else ("mixed: " + ", ".join(statuses) if statuses else None)),
        "updated_at": max((u["updated_at"] for u in units if u.get("updated_at")), default=None),
        "units": [
            {
                "name": u.get("vessel") or u.get("name"),
                "send_out_gwh_d": u.get("send_out_gwh_d"),
                "utilisation_pct": u.get("utilisation_pct"),
                "fullness_pct": u.get("fullness_pct"),
                "gas_day": u.get("gas_day"),
            }
            for u in units
        ],
    }


def lng_terminal_markers(facilities: list[dict], provenance: dict, alsi: dict | None) -> list[dict]:
    """LNG import terminals for the globe and the table, with live ALSI figures joined.

    Every bundled terminal becomes a marker. ALSI units join their marker through
    ``alsi_marker_index``; a unit without a bundled marker gets a marker at ALSI's own
    coordinates (or none, ``lon``/``lat`` None, when ALSI publishes no valid position).
    Markers without live figures carry ``live_note`` explaining why. Sorted with live
    terminals first, by send-out, then by name.
    """
    alsi = alsi or {}
    status = alsi.get("status")
    index = alsi_marker_index(facilities, provenance)
    crosswalk_targets = set(index.values())
    markers: dict[str, dict] = {}
    for feature in facilities or []:
        props = feature.get("properties") or {}
        if props.get("category") != "lng_import":
            continue
        lon, lat = _point(feature)
        if lon is None:
            continue
        markers[props["id"]] = {
            "id": props["id"],
            "name": props.get("name"),
            "lon": lon,
            "lat": lat,
            "nation": props.get("nation"),
            "operator": props.get("operator"),
            "kind": props.get("kind") or "LNG import terminal",
            "fsru": "FSRU" in str(props.get("kind") or "") or "FSRU" in str(props.get("source_kind") or ""),
            "status": props.get("status"),
            "startup": props.get("startup"),
            "capacity_bcm_y": props.get("capacity_bcm_y"),
            "capacity_note": props.get("capacity_note"),
            "location_accuracy": props.get("location_accuracy"),
            "url": props.get("url"),
            "source": props.get("source"),
            "position_source": "Energy Facilities dataset",
            "_units": [],
        }
    for unit in alsi.get("units") or []:
        target = index.get(unit.get("eic"))
        if target in markers:
            markers[target]["_units"].append(unit)
            continue
        marker_id = f"alsi-{unit.get('eic')}"
        markers[marker_id] = {
            "id": marker_id,
            "name": unit.get("name"),
            "lon": unit.get("alsi_lon"),
            "lat": unit.get("alsi_lat"),
            "nation": ALSI_COUNTRIES.get(unit.get("country") or "", unit.get("country")),
            "operator": unit.get("operator"),
            "kind": ("LNG import terminal (FSRU)" if unit.get("terminal_type") == "FSRU"
                     else "LNG import terminal"),
            "fsru": unit.get("terminal_type") == "FSRU",
            "status": "operating",
            "startup": None,
            "capacity_bcm_y": None,
            "capacity_note": None,
            "location_accuracy": "GIE ALSI coordinates" if unit.get("alsi_lon") is not None else None,
            "url": None,
            "source": "GIE ALSI",
            "position_source": "GIE ALSI" if unit.get("alsi_lon") is not None else None,
            "_units": [unit],
        }
    out: list[dict] = []
    for marker in markers.values():
        units = marker.pop("_units")
        marker.update(_aggregate_units(units))
        if marker["live"]:
            marker["live_note"] = None
        elif marker["id"] not in crosswalk_targets:
            marker["live_note"] = "Not reported to GIE ALSI"
        elif status == "ok":
            marker["live_note"] = "No ALSI figure in this refresh"
        else:
            marker["live_note"] = "Live figures unavailable"
        out.append(marker)
    out.sort(key=lambda m: (0 if m["live"] else 1, -(m.get("send_out_gwh_d") or 0.0),
                            str(m.get("name") or "")))
    return out


def lng_totals(markers: list[dict]) -> dict:
    """Send-out, utilisation and stock fullness across terminals with live figures."""
    live = [m for m in markers if m.get("live")]
    send = [m["send_out_gwh_d"] for m in live if m.get("send_out_gwh_d") is not None]
    change = [m["send_out_change_gwh_d"] for m in live
              if m.get("send_out_gwh_d") is not None and m.get("send_out_change_gwh_d") is not None]
    util_pairs = [(m["send_out_gwh_d"], m["dtrs_gwh_d"]) for m in live
                  if m.get("send_out_gwh_d") is not None and (m.get("dtrs_gwh_d") or 0) > 0]
    full_pairs = [(m["inventory_gwh"], m["dtmi_gwh"]) for m in live
                  if m.get("inventory_gwh") is not None and (m.get("dtmi_gwh") or 0) > 0]
    return {
        "terminals_live": len(live),
        "terminals_total": len(markers),
        "send_out_gwh_d": round(sum(send), 1) if send else None,
        # Day-on-day change only when every reporting terminal has both days.
        "send_out_change_gwh_d": round(sum(change), 1) if send and len(change) == len(send) else None,
        "utilisation_pct": (round(sum(s for s, _ in util_pairs) / sum(d for _, d in util_pairs) * 100.0, 1)
                            if util_pairs else None),
        "fullness_pct": (round(sum(i for i, _ in full_pairs) / sum(d for _, d in full_pairs) * 100.0, 1)
                         if full_pairs else None),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Payload for the globe component
# ══════════════════════════════════════════════════════════════════════════════

def _drop_empty(record: dict) -> dict:
    return {k: v for k, v in record.items() if v is not None and v != [] and v != ""}


def _facility_record(feature: dict) -> dict | None:
    props = feature.get("properties") or {}
    lon, lat = _point(feature)
    if lon is None:
        return None
    return _drop_empty({
        "id": props.get("id"),
        "name": props.get("name"),
        "cat": props.get("category"),
        "kind": props.get("kind"),
        "status": props.get("status"),
        "operator": props.get("operator"),
        "field": props.get("field"),
        "system": props.get("system"),
        "nation": props.get("nation"),
        "startup": props.get("startup"),
        "depth": props.get("water_depth_m"),
        "rank": props.get("display_rank", 0),
        "acc": props.get("location_accuracy"),
        "url": props.get("url"),
        "colocated": props.get("colocated"),
        "lon": round(lon, _COORD_DIGITS),
        "lat": round(lat, _COORD_DIGITS),
    })


def _pipeline_records(features: list[dict]) -> tuple[list[dict], list[dict]]:
    lines: list[dict] = []
    terminals: list[dict] = []
    for feature in features or []:
        props = feature.get("properties") or {}
        geometry = feature.get("geometry") or {}
        kind = geometry.get("type")
        if kind == "Point":
            lon, lat = _point(feature)
            if lon is None:
                continue
            systems = props.get("systems")
            terminals.append(_drop_empty({
                "id": props.get("id"),
                "name": props.get("name"),
                "kind": props.get("kind"),
                "nation": props.get("nation"),
                "systems": systems if isinstance(systems, list) else None,
                "operator": props.get("operator"),
                "lon": round(lon, _COORD_DIGITS),
                "lat": round(lat, _COORD_DIGITS),
            }))
            continue
        if kind == "LineString":
            parts = [geometry.get("coordinates") or []]
        elif kind == "MultiLineString":
            parts = geometry.get("coordinates") or []
        else:
            continue
        parts = [_round_coords(p) for p in parts if len(p) >= 2]
        if not parts:
            continue
        lines.append(_drop_empty({
            "id": props.get("id"),
            "name": props.get("name"),
            "segment": props.get("segment"),
            "system": props.get("system"),
            "medium": props.get("medium"),
            "dia": props.get("diameter_in"),
            "len": props.get("length_km"),
            "depth": props.get("max_water_depth_m"),
            "from": props.get("from"),
            "to": props.get("to"),
            "operator": props.get("operator"),
            "status": props.get("status"),
            "since": props.get("status_since"),
            "url": props.get("fact_page"),
            "parts": parts,
        }))
    return lines, terminals


def _chokepoint_record(feature: dict) -> dict | None:
    props = feature.get("properties") or {}
    geometry = feature.get("geometry") or {}
    if geometry.get("type") == "Polygon":
        polygons = [geometry.get("coordinates") or []]
    elif geometry.get("type") == "MultiPolygon":
        polygons = geometry.get("coordinates") or []
    else:
        return None
    rings = [_round_coords(polygon[0]) for polygon in polygons if polygon and len(polygon[0]) >= 4]
    label = props.get("label_point")
    if not (isinstance(label, list) and len(label) == 2 and _finite(label[0]) and _finite(label[1])):
        return None
    return _drop_empty({
        "id": props.get("id"),
        "name": props.get("name"),
        "kind": props.get("kind"),
        "region": props.get("region"),
        "connects": props.get("connects"),
        "gate": props.get("gate"),
        "oil": props.get("oil_flow_mb_d"),
        "oil_period": props.get("oil_flow_period"),
        "bosporus": props.get("oil_flow_bosporus_mb_d"),
        "lng": props.get("lng_flow_bcf_d"),
        "lng_below": props.get("lng_flow_below_bcf_d"),
        "lng_period": props.get("lng_flow_period"),
        "flows_source": props.get("flows_source"),
        "notes": props.get("notes"),
        "label": [round(float(label[0]), _COORD_DIGITS), round(float(label[1]), _COORD_DIGITS)],
        "rings": rings,
    })


def _lng_record(marker: dict) -> dict | None:
    if marker.get("lon") is None or marker.get("lat") is None:
        return None
    record = {k: marker.get(k) for k in (
        "id", "name", "nation", "operator", "kind", "fsru", "status", "startup",
        "capacity_bcm_y", "capacity_note", "location_accuracy", "url", "live", "live_note",
        "gas_day", "mixed_days", "send_out_gwh_d", "send_out_change_gwh_d", "utilisation_pct",
        "fullness_pct", "inventory_gwh", "inventory_lng_m3", "dtrs_gwh_d", "dtmi_gwh",
        "data_status", "units",
    )}
    record["lon"] = round(float(marker["lon"]), _COORD_DIGITS)
    record["lat"] = round(float(marker["lat"]), _COORD_DIGITS)
    record["units"] = record["units"] if record.get("units") and len(record["units"]) > 1 else None
    return _drop_empty(record) | {"live": bool(marker.get("live")), "fsru": bool(marker.get("fsru"))}


def globe_payload(layers: dict, lng_markers: list[dict], alsi: dict, view: str,
                  views: dict[str, dict]) -> dict:
    """Everything the globe component draws, as plain JSON-serialisable data.

    The payload is deterministic for the same inputs, so the embedded globe is not
    rebuilt on reruns that do not change the data.
    """
    facilities = [
        record for record in (
            _facility_record(f) for f in layers.get("facilities") or []
            if (f.get("properties") or {}).get("category") != "lng_import"
        ) if record
    ]
    pipelines, receiving = _pipeline_records(layers.get("pipelines") or [])
    chokepoints = [r for r in (_chokepoint_record(f) for f in layers.get("chokepoints") or []) if r]
    lng = [r for r in (_lng_record(m) for m in lng_markers) if r]
    return {
        "view": view if view in views else next(iter(views)),
        "views": {
            key: {"label": spec["label"], "title": spec["title"], "lat": spec["lat"],
                  "lon": spec["lon"], "alt": spec["alt"], "pitch": spec.get("pitch", -90)}
            for key, spec in views.items()
        },
        "facilities": facilities,
        "pipelines": pipelines,
        "receiving": receiving,
        "lng": lng,
        "chokepoints": chokepoints,
        "meta": {
            "live_status": alsi.get("status"),
            "live_message": alsi.get("message"),
            "gas_day": alsi.get("gas_day"),
            "fetched_at": alsi.get("fetched_at"),
        },
    }
