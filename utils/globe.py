"""Links to the Energy Globe.

The Energy Globe is a page of this platform (``views/8_Energy_Globe.py``): a CesiumJS
globe of Norwegian shelf facilities and export pipelines, European LNG terminals with
live GIE ALSI figures, and maritime chokepoints. ``globe_link`` renders a button that
opens the page on one of the curated views below; the view travels as the ``view``
query parameter, so a link such as ``/Energy_Globe?view=hormuz`` also works on its own.

Optional: the same views can be opened in the stand-alone gods-eye-view globe (the
CesiumJS application the datasets were first built for). Set ``GLOBE_URL`` to its base
address, for example ``http://127.0.0.1:5173``; ``external_globe_url`` then encodes the
camera position and the globe's share-link layer tokens in the URL hash. Without it (or
with anything other than an http(s) address) the external link is not offered.
"""
from __future__ import annotations

import os
from urllib.parse import urlencode, urlsplit

import streamlit as st
from streamlit.errors import StreamlitAPIException

GLOBE_PAGE = "views/8_Energy_Globe.py"

# Share-link tokens the stand-alone globe registers for these layers. They are part of
# its public link format (src/data/layerStateTokenReservations.json in the globe
# repository) and never change once published.
LAYER_TOKENS: dict[str, str] = {
    "energy_facilities": "3",   # Energy Facilities: Norwegian shelf and European LNG terminals
    "energy_pipelines": "4",    # Energy Pipelines: Norwegian shelf export lines and receiving terminals
    "lng_live": "5",            # LNG Terminals (live): GIE ALSI send-out and stocks
    "chokepoints": "6",         # Chokepoints: maritime gates with live AIS counts and EIA flows
}

# Curated views: target ``lat``/``lon``, camera distance ``alt`` (metres) and the camera
# ``pitch`` (degrees) the platform's globe uses; external links always look straight down.
# ``layers`` lists the stand-alone globe's layers for the external link; the platform's
# own globe shows every layer and lets the reader switch them.
GLOBE_VIEWS: dict[str, dict] = {
    "europe": {
        "label": "Europe",
        "lat": 51.5, "lon": 7.0, "alt": 5_600_000, "pitch": -72,
        "layers": ("energy_facilities", "energy_pipelines", "lng_live"),
        "title": "European gas infrastructure: Norwegian shelf, export pipelines and LNG import terminals",
    },
    "north_sea": {
        "label": "Norwegian shelf",
        "lat": 58.6, "lon": 3.6, "alt": 1_700_000, "pitch": -62,
        "layers": ("energy_facilities", "energy_pipelines"),
        "title": "Norwegian shelf: offshore facilities, export pipelines and receiving terminals",
    },
    "lng_europe": {
        "label": "European LNG",
        "lat": 49.5, "lon": 4.0, "alt": 3_200_000, "pitch": -70,
        "layers": ("lng_live", "energy_facilities"),
        "title": "European LNG import terminals with live send-out and stocks",
    },
    "lng_routes": {
        "label": "LNG routes",
        "lat": 30.0, "lon": 38.0, "alt": 13_000_000, "pitch": -90,
        "layers": ("chokepoints", "lng_live"),
        "title": "LNG shipping lanes: maritime chokepoints between the Gulf, Asia and Europe",
    },
    "hormuz": {
        "label": "Hormuz",
        "lat": 26.2, "lon": 55.2, "alt": 1_500_000, "pitch": -62,
        "layers": ("chokepoints",),
        "title": "Strait of Hormuz: about a fifth of world oil and LNG flows pass through it (EIA)",
    },
}

DEFAULT_VIEW = "europe"


def resolve_view(value: object) -> str:
    """Return ``value`` when it names a curated view, otherwise the default view."""
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    text = str(value).strip() if value is not None else ""
    return text if text in GLOBE_VIEWS else DEFAULT_VIEW


def globe_link(view: str = DEFAULT_VIEW, label: str = "View on the 3D globe") -> None:
    """Render a button that opens the Energy Globe page on ``view``.

    Through ``app.py`` the page is always registered and the link renders. A page module
    run on its own (``streamlit run views/...`` or the page smoke tests) has no router, so
    the Energy Globe page does not exist there and Streamlit cannot build the link: it
    raises ``KeyError`` (1.56) or ``StreamlitPageNotFoundError``. The link is then left
    out instead of failing the page.
    """
    view = resolve_view(view)
    try:
        st.page_link(
            GLOBE_PAGE,
            label=label,
            icon=":material/public:",
            help=GLOBE_VIEWS[view]["title"],
            query_params={"view": view},
        )
    except (KeyError, StreamlitAPIException):
        return


def globe_base_url() -> str | None:
    """Return the configured stand-alone globe address, or None when it is not configured."""
    raw = os.getenv("GLOBE_URL", "").strip()
    if not raw:
        return None
    parts = urlsplit(raw)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return None
    # Any fragment on the configured address would be replaced by the view hash.
    return raw.split("#", 1)[0]


def external_globe_url(view: str) -> str | None:
    """Deep link that opens ``view`` in the stand-alone globe, or None when not configured."""
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
