"""Deep links into the companion 3D Energy Globe (utils/globe.py)."""
from urllib.parse import parse_qs, urlsplit

import pytest

from utils import globe


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("GLOBE_URL", "http://localhost:5173/")


def _hash_params(url: str) -> dict[str, str]:
    fragment = urlsplit(url).fragment
    return {k: v[0] for k, v in parse_qs(fragment, keep_blank_values=True).items()}


def test_links_are_hidden_without_configuration(monkeypatch):
    monkeypatch.delenv("GLOBE_URL", raising=False)
    assert globe.globe_base_url() is None
    assert globe.globe_view_url("north_sea") is None


@pytest.mark.parametrize("value", ["", "   ", "localhost:5173", "ftp://globe.example", "javascript:alert(1)"])
def test_only_http_addresses_are_accepted(monkeypatch, value):
    monkeypatch.setenv("GLOBE_URL", value)
    assert globe.globe_view_url("north_sea") is None


def test_view_link_carries_camera_and_layer_tokens(configured):
    url = globe.globe_view_url("north_sea")
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
    url = globe.globe_view_url("hormuz")
    assert url.startswith("https://globe.example/app/#v=2&")
    assert url.count("#") == 1
    assert _hash_params(url)["l"] == "6"


def test_unknown_view_returns_none(configured):
    assert globe.globe_view_url("does-not-exist") is None


def test_every_view_uses_registered_tokens_once(configured):
    known = set(globe.LAYER_TOKENS.values())
    for name in globe.GLOBE_VIEWS:
        tokens = _hash_params(globe.globe_view_url(name))["l"].split(".")
        assert tokens and set(tokens) <= known
        assert len(tokens) == len(set(tokens)), name
