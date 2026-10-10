"""The embedded globe document (components/energy_globe.py and energy_globe.html)."""
import json
import os
import re
import shutil
import subprocess

import pytest

from components import energy_globe as ce
from data import energy_globe as eg
from utils.globe import GLOBE_VIEWS

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


@pytest.fixture(scope="module")
def payload():
    layers = eg.load_layers()
    alsi = {"status": "no_key", "message": "Add AGSI_API_KEY", "units": []}
    markers = eg.lng_terminal_markers(layers["facilities"], layers["facilities_source"], alsi)
    return eg.globe_payload(layers, markers, alsi, "north_sea", GLOBE_VIEWS)


def _embedded_payload(html: str) -> dict:
    match = re.search(r'<script id="payload" type="application/json">(.*?)</script>', html, re.S)
    assert match, "payload script element missing"
    return json.loads(match.group(1))


def test_cesium_is_pinned_and_integrity_checked(payload):
    html = ce.build_globe_html(payload)
    assert f'src="{ce.CESIUM_BASE_URL}Cesium.js" integrity="{ce.CESIUM_JS_SRI}" crossorigin="anonymous"' in html
    assert f'href="{ce.CESIUM_BASE_URL}Widgets/widgets.css" integrity="{ce.CESIUM_CSS_SRI}"' in html
    assert f"cesium@{ce.CESIUM_VERSION}/" in ce.CESIUM_BASE_URL
    assert re.fullmatch(r"sha384-[A-Za-z0-9+/]{64}", ce.CESIUM_JS_SRI)
    assert not re.search(r"__[A-Z_]+__", html), "unfilled template token"


def test_payload_round_trips(payload):
    html = ce.build_globe_html(payload)
    assert _embedded_payload(html) == json.loads(json.dumps(payload))


def test_upstream_text_cannot_break_out_of_the_script_element(payload):
    hostile = json.loads(json.dumps(payload))
    hostile["lng"][0]["name"] = '</script><script>alert(1)</script> & <b>'
    hostile["lng"][0]["operator"] = "line separator"
    html = ce.build_globe_html(hostile)
    payload_block = re.search(r'<script id="payload" type="application/json">(.*?)</script>', html, re.S).group(1)
    assert "<" not in payload_block and ">" not in payload_block and "&" not in payload_block
    assert _embedded_payload(html)["lng"][0]["name"] == hostile["lng"][0]["name"]


def test_non_finite_numbers_become_null(payload):
    odd = json.loads(json.dumps(payload))
    odd["lng"][0]["capacity_bcm_y"] = float("nan")
    odd["lng"][1]["capacity_bcm_y"] = float("inf")
    data = _embedded_payload(ce.build_globe_html(odd))
    assert data["lng"][0]["capacity_bcm_y"] is None and data["lng"][1]["capacity_bcm_y"] is None


def test_frame_handles_about_srcdoc_workers():
    template = (ce._TEMPLATE_PATH).read_text(encoding="utf-8")
    assert "useAbsoluteWorkerUrls" in template
    assert "_canTransferArrayBuffer" in template
    # Keyless basemaps only.
    assert "cartocdn" not in template
    assert "World_Dark_Gray_Base" in template and "VIIRS_Black_Marble" in template


def test_the_frame_degrades_without_cesium_or_webgl():
    template = (ce._TEMPLATE_PATH).read_text(encoding="utf-8")
    assert 'typeof Cesium === "undefined"' in template
    assert "could not start WebGL" in template


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_inline_script_is_valid_javascript(payload, tmp_path):
    html = ce.build_globe_html(payload)
    scripts = re.findall(r"<script>(.*?)</script>", html, re.S)
    path = tmp_path / "globe.js"
    path.write_text(scripts[-1], encoding="utf-8")
    result = subprocess.run(["node", "--check", str(path)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


def test_page_runs_without_a_key(monkeypatch):
    """The page needs no network without AGSI_API_KEY, so this runs offline."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.chdir(ROOT)
    monkeypatch.delenv("AGSI_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    at = AppTest.from_file(os.path.join(ROOT, "views/8_Energy_Globe.py"), default_timeout=120)
    at.query_params["view"] = "hormuz"
    at.run()
    assert not at.exception, [str(getattr(e, "value", e)) for e in at.exception]
    assert len(at.tabs) == 5
    assert any("AGSI_API_KEY" in m.value for m in at.markdown)
