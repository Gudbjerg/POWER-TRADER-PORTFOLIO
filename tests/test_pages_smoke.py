"""
Smoke tests: every page changed by the Q4 roadmap work must execute end to end
without raising. Pages fetch live data (Nord Pool, Yahoo Finance,
energy-charts.info), so the tests carry the ``network`` marker; deselect them
offline with ``pytest -m "not network"``. Missing API keys (ENTSO-E, GIE) are an
expected condition and must degrade gracefully rather than raise.
"""
import os

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

CHANGED_PAGES = [
    "pages/1_Live_Monitor.py",
    "pages/3_Macro_Signals.py",
]


@pytest.mark.network
@pytest.mark.parametrize("page", CHANGED_PAGES)
def test_page_runs_without_exception(page, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.chdir(ROOT)
    at = AppTest.from_file(os.path.join(ROOT, page), default_timeout=300).run()
    messages = [str(getattr(exc, "value", exc)) for exc in at.exception]
    assert not at.exception, f"{page} raised: {messages}"
