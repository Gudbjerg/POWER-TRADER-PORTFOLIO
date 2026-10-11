"""Links to the Energy Globe page and to the optional stand-alone globe (utils/globe.py)."""
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from utils import globe

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("GLOBE_URL", "http://localhost:5173/")


def _hash_params(url: str) -> dict[str, str]:
    fragment = urlsplit(url).fragment
    return {k: v[0] for k, v in parse_qs(fragment, keep_blank_values=True).items()}


# ── Views and the internal page ───────────────────────────────────────────────

def test_views_carry_everything_the_globe_needs():
    assert globe.DEFAULT_VIEW in globe.GLOBE_VIEWS
    for name, spec in globe.GLOBE_VIEWS.items():
        assert {"label", "title", "lat", "lon", "alt", "pitch", "layers"} <= set(spec), name
        assert -90 <= spec["lat"] <= 90 and -180 <= spec["lon"] <= 180, name
        assert spec["alt"] > 0 and -90 <= spec["pitch"] < 0, name


@pytest.mark.parametrize(
    ("value", "expected"),
    [("hormuz", "hormuz"), (" north_sea ", "north_sea"), (["lng_routes"], "lng_routes"),
     (None, globe.DEFAULT_VIEW), ("", globe.DEFAULT_VIEW), ("unknown", globe.DEFAULT_VIEW),
     ([], globe.DEFAULT_VIEW), ("<script>", globe.DEFAULT_VIEW)],
)
def test_resolve_view_accepts_only_curated_views(value, expected):
    assert globe.resolve_view(value) == expected


def test_link_is_left_out_when_a_page_runs_without_the_router(tmp_path, monkeypatch):
    """Page smoke tests run a view module on its own; the globe link must not break it."""
    from streamlit.testing.v1 import AppTest

    script = tmp_path / "standalone_page.py"
    script.write_text(
        "import streamlit as st\n"
        "from utils.globe import globe_link\n"
        "globe_link('hormuz', 'Strait of Hormuz')\n"
        "st.write('rendered after the link')\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(ROOT)
    monkeypatch.syspath_prepend(str(ROOT))
    at = AppTest.from_file(str(script), default_timeout=60).run()
    assert not at.exception, [str(getattr(e, "value", e)) for e in at.exception]
    assert any("rendered after the link" in m.value for m in at.markdown)


def test_globe_page_is_registered_in_the_router():
    router = (ROOT / "app.py").read_text(encoding="utf-8")
    assert f'st.Page("{globe.GLOBE_PAGE}"' in router
    assert 'url_path="Energy_Globe"' in router


# ── Optional stand-alone globe ────────────────────────────────────────────────

def test_external_links_are_off_without_configuration(monkeypatch):
    monkeypatch.delenv("GLOBE_URL", raising=False)
    assert globe.globe_base_url() is None
    assert globe.external_globe_url("north_sea") is None


@pytest.mark.parametrize("value", ["", "   ", "localhost:5173", "ftp://globe.example", "javascript:alert(1)"])
def test_only_http_addresses_are_accepted(monkeypatch, value):
    monkeypatch.setenv("GLOBE_URL", value)
    assert globe.external_globe_url("north_sea") is None


def test_external_link_carries_camera_and_layer_tokens(configured):
    url = globe.external_globe_url("north_sea")
    assert url.startswith("http://localhost:5173/#")
    params = _hash_params(url)
    assert params["v"] == "2"
    assert params["pitch"] == "-90"
    assert float(params["lat"]) == pytest.approx(58.6)
    assert float(params["lon"]) == pytest.approx(3.6)
    assert params["alt"] == "1700000"
    # Facilities (3) and pipelines (4), dot-separated as the globe's codec expects.
    assert params["l"] == "3.4"


def test_existing_fragment_on_the_base_is_replaced(monkeypatch):
    monkeypatch.setenv("GLOBE_URL", "https://globe.example/app/#lat=1&lon=2")
    url = globe.external_globe_url("hormuz")
    assert url.startswith("https://globe.example/app/#v=2&")
    assert url.count("#") == 1
    assert _hash_params(url)["l"] == "6"


def test_unknown_view_returns_none(configured):
    assert globe.external_globe_url("does-not-exist") is None


def test_every_view_uses_registered_tokens_once(configured):
    known = set(globe.LAYER_TOKENS.values())
    for name in globe.GLOBE_VIEWS:
        tokens = _hash_params(globe.external_globe_url(name))["l"].split(".")
        assert tokens and set(tokens) <= known
        assert len(tokens) == len(set(tokens)), name
