"""Embedded 3D Energy Globe for the Energy Globe page.

The globe is a self-contained HTML document (``energy_globe.html``) rendered with
``st.iframe``. It loads CesiumJS from jsDelivr, pinned to one version and checked with
Subresource Integrity, and draws the payload built by ``data.energy_globe.globe_payload``:
Norwegian shelf facilities, export pipelines and receiving terminals, LNG import
terminals (gauge markers: size for send-out, arc for utilisation) and maritime
chokepoints. Camera views, layer switches, basemap choice, hover labels and the details
card all run inside the frame, so using them never reruns the Streamlit script.

Basemaps: Esri World Dark Gray Base by default, NASA GIBS VIIRS Black Marble night
lights and Esri World Imagery on request, and Natural Earth II (bundled with CesiumJS)
underneath them as an offline fallback. None of them needs a key. If CesiumJS or WebGL
is unavailable the frame says so and the page's tables still carry the data.

The frame is an ``about:srcdoc`` document. The template therefore hands CesiumJS
absolute worker URLs and answers its transferable-buffer probe up front; without that,
CesiumJS resolves its bare worker module ids against ``about:srcdoc``, every worker
fails, and terrain, imagery and line geometry never load.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import streamlit as st

from config.settings import GLOBE_HEIGHT_PX

CESIUM_VERSION = "1.138.0"
CESIUM_BASE_URL = f"https://cdn.jsdelivr.net/npm/cesium@{CESIUM_VERSION}/Build/Cesium/"
# sha384 of Build/Cesium/Cesium.js and Build/Cesium/Widgets/widgets.css in the npm
# package cesium@1.138.0 (identical to the files jsDelivr serves).
CESIUM_JS_SRI = "sha384-gwvyEd+NJ2w32hmgkIBgNk001SFVYPVftkf+WkDlk8PgE6LE3zQ9dPk3LWlkmL5O"
CESIUM_CSS_SRI = "sha384-ghEeMdcWWzRv/BPeUcX835vcKDGrxvROXisl/Btpv3GeekBUXTSPVcFJpI1Tcrgp"

_TEMPLATE_PATH = Path(__file__).with_name("energy_globe.html")
_PAYLOAD_TOKEN = "__PAYLOAD_JSON__"


@lru_cache(maxsize=1)
def _template() -> str:
    text = _TEMPLATE_PATH.read_text(encoding="utf-8")
    if text.count(_PAYLOAD_TOKEN) != 1:
        raise RuntimeError("energy_globe.html must contain the payload placeholder exactly once")
    return (
        text.replace("__CESIUM_BASE__", CESIUM_BASE_URL)
        .replace("__CESIUM_JS_SRI__", CESIUM_JS_SRI)
        .replace("__CESIUM_CSS_SRI__", CESIUM_CSS_SRI)
    )


def _clean(value: Any) -> Any:
    """Replace non-finite floats with None so the payload is strict JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def payload_json(payload: dict) -> str:
    """Serialise the payload for an inline ``<script type="application/json">`` element.

    ``<``, ``>`` and ``&`` are written as Unicode escapes, so no text from the data (for
    example a facility name from an upstream API) can close the script element or open
    markup. U+2028 and U+2029 are escaped for older JavaScript parsers.
    """
    text = json.dumps(_clean(payload), separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return (
        text.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace(" ", "\\u2028")
        .replace(" ", "\\u2029")
    )


def build_globe_html(payload: dict) -> str:
    """The complete HTML document for the globe frame."""
    # The payload goes in last, so its content is never scanned for template tokens.
    return _template().replace(_PAYLOAD_TOKEN, payload_json(payload), 1)


def render_energy_globe(payload: dict, height: int = GLOBE_HEIGHT_PX) -> None:
    """Render the globe in an iframe of fixed ``height`` CSS pixels, full container width."""
    st.iframe(build_globe_html(payload), height=int(height))
