"""Schema and loader tests for data/events.json (Layer 3 geopolitical overlay)."""
import json
import os

import pandas as pd
import pytest

from data.events import CATEGORY_COLORS, IMPACT_DASH, events_in_range, load_events

EVENTS_FILE = os.path.join(os.path.dirname(__file__), os.pardir, "data", "events.json")
REQUIRED = {"date", "label", "description", "category", "impact"}


@pytest.fixture(scope="module")
def raw_events():
    with open(EVENTS_FILE, encoding="utf-8") as f:
        return json.load(f)


def test_schema_and_vocabularies(raw_events):
    assert raw_events, "events.json must not be empty"
    for ev in raw_events:
        assert REQUIRED.issubset(ev), ev
        pd.Timestamp(ev["date"])                           # parseable ISO date
        assert ev["category"] in CATEGORY_COLORS
        assert ev["impact"] in {"critical", "warn"}
        assert ev["impact"] in IMPACT_DASH
        assert ev["label"].strip() and ev["description"].strip()


def test_events_sorted_by_date(raw_events):
    dates = [ev["date"] for ev in raw_events]
    assert dates == sorted(dates)


def test_sourced_events_are_well_formed(raw_events):
    sourced = [ev for ev in raw_events if "source" in ev]
    assert sourced, "expanded entries carry a source URL"
    for ev in sourced:
        assert ev["source"].startswith("https://")
        assert "2026-04-20" <= ev["date"] <= "2026-10-10"
    # Every entry added for the 2026-04-20 onwards window carries a source
    assert all("source" in ev for ev in raw_events if ev["date"] >= "2026-04-20")


def test_no_em_dashes_in_user_facing_text(raw_events):
    for ev in raw_events:
        assert "—" not in ev["label"], ev["label"]
    for ev in raw_events:
        if "source" in ev:
            assert "—" not in ev["description"], ev["description"]


def test_loader_ignores_unknown_keys_and_filters_range():
    events = load_events()
    assert all(isinstance(ev["date"], pd.Timestamp) for ev in events)
    in_window = events_in_range("2026-06-01", "2026-07-31")
    assert in_window and all(
        pd.Timestamp("2026-06-01") <= ev["date"] <= pd.Timestamp("2026-07-31") for ev in in_window
    )
    assert any("source" in ev for ev in in_window)          # extra key passes through untouched
