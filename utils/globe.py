"""Deep links into the companion 3D Energy Globe.

The globe (a CesiumJS application) restores a complete view from its share-link hash:
the camera position (``lat``, ``lon``, ``alt``, ``heading``, ``pitch``, ``roll``) and the
enabled layers, encoded as share-link tokens in ``l``. This module builds such links for
a few curated views so that a panel can open the matching infrastructure in 3D.

Configuration: set ``GLOBE_URL`` to the globe's base address, for example
``http://localhost:5173`` for a local development server or the address of a hosted
deployment. Without it (or with anything other than an http(s) address) every helper
returns ``None`` or renders nothing, so the platform runs unchanged.
"""
from __future__ import annotations

import os
from urllib.parse import urlencode, urlsplit

import streamlit as st

# Share-link tokens the globe registers for these layers. They are part of the globe's
# public link format (src/data/layerStateTokenReservations.json in the globe repository)
# and never change once published.
LAYER_TOKENS: dict[str, str] = {
    "energy_facilities": "3",   # Energy Facilities: Norwegian shelf and European LNG terminals
    "energy_pipelines": "4",    # Energy Pipelines: Norwegian shelf export lines and receiving terminals
    "lng_live": "5",            # LNG Terminals (live): GIE ALSI send-out and stocks
    "chokepoints": "6",         # Chokepoints: maritime gates with live AIS counts and EIA flows
}

# Curated views. The camera looks straight down (pitch -90 degrees) from ``alt`` metres.
GLOBE_VIEWS: dict[str, dict] = {
    "north_sea": {
        "lat": 58.6, "lon": 3.6, "alt": 1_700_000,
        "layers": ("energy_facilities", "energy_pipelines"),
        "title": "Norwegian shelf: offshore facilities, export pipelines and receiving terminals",
    },
    "lng_europe": {
        "lat": 49.5, "lon": 4.0, "alt": 3_200_000,
        "layers": ("lng_live", "energy_facilities"),
        "title": "European LNG import terminals with live send-out and stocks",
    },
    "lng_routes": {
        "lat": 24.0, "lon": 40.0, "alt": 14_000_000,
        "layers": ("chokepoints", "lng_live"),
        "title": "LNG shipping lanes: maritime chokepoints between the Gulf, Asia and Europe",
    },
    "hormuz": {
        "lat": 26.4, "lon": 56.4, "alt": 900_000,
        "layers": ("chokepoints",),
        "title": "Strait of Hormuz: live tanker and LNG carrier counts",
    },
}


def globe_base_url() -> str | None:
    """Return the configured globe address, or None when the globe is not configured."""
    raw = os.getenv("GLOBE_URL", "").strip()
    if not raw:
        return None
    parts = urlsplit(raw)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return None
    # Any fragment on the configured address would be replaced by the view hash.
    return raw.split("#", 1)[0]


def globe_view_url(view: str) -> str | None:
    """Return a deep link that opens ``view`` (a key of GLOBE_VIEWS) on the globe."""
    base = globe_base_url()
    spec = GLOBE_VIEWS.get(view)
    if base is None or spec is None:
        return None
    tokens = [LAYER_TOKENS[layer] for layer in spec["layers"]]
    params = {
        "v": "2",
        "lat": f"{spec['lat']:.4f}",
        "lon": f"{spec['lon']:.4f}",
        "alt": str(int(spec["alt"])),
        "heading": "0",
        "pitch": "-90",
        "roll": "0",
        "l": ".".join(tokens),
    }
    return f"{base}#{urlencode(params, safe='.')}"


def globe_link(view: str, label: str = "View on the 3D globe") -> None:
    """Render a link button to a globe view; renders nothing when the globe is not configured."""
    url = globe_view_url(view)
    if url is None:
        return
    st.page_link(url, label=label, icon=":material/public:", help=GLOBE_VIEWS[view]["title"])
