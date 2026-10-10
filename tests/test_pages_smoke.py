"""
Smoke tests: the router and every page must execute end to end without raising.
Pages fetch live data (Nord Pool, Yahoo Finance, energy-charts.info), so the tests
carry the ``network`` marker; deselect them offline with ``pytest -m "not network"``.
Missing API keys (ENTSO-E, GIE) are an expected condition and must degrade
gracefully rather than raise.
"""
import os

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Seconds allowed for one full script run. With live ENTSO-E and GIE keys a cold
# cache makes the heaviest page (Quant Analysis: 14 tabs over a three-year
# feature matrix) take several minutes; override with PTP_SMOKE_TIMEOUT.
TIMEOUT_S = int(os.getenv("PTP_SMOKE_TIMEOUT", "900"))

PAGES = [
    "views/1_Live_Monitor.py",
    "views/2_Quant_Analysis.py",
    "views/3_Macro_Signals.py",
    "views/4_ML_Models.py",
    "views/5_Mispricing_Dashboard.py",
    "views/6_Risk_Dashboard.py",
    "views/7_BESS_Optimiser.py",
]


def _run(path: str):
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(os.path.join(ROOT, path), default_timeout=TIMEOUT_S).run()


@pytest.mark.network
@pytest.mark.parametrize("page", PAGES)
def test_page_runs_without_exception(page, monkeypatch):
    monkeypatch.chdir(ROOT)
    at = _run(page)
    messages = [str(getattr(exc, "value", exc)) for exc in at.exception]
    assert not at.exception, f"{page} raised: {messages}"


@pytest.mark.network
@pytest.mark.parametrize("globe_url", ["", "http://localhost:5173"])
def test_router_renders_the_overview(globe_url, monkeypatch):
    """app.py routes to the overview page, with and without the companion globe configured."""
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("GLOBE_URL", globe_url)
    at = _run("app.py")
    messages = [str(getattr(exc, "value", exc)) for exc in at.exception]
    assert not at.exception, f"app.py raised: {messages}"
