"""
Energy market event loader for geopolitical overlay charts.
"""
import json
import os
import pandas as pd

_DIR = os.path.dirname(__file__)
_EVENTS_FILE = os.path.join(_DIR, "events.json")

CATEGORY_COLORS = {
    "geopolitical": "#f85149",
    "supply":       "#d29922",
    "policy":       "#58a6ff",
    "weather":      "#3fb950",
}

IMPACT_DASH = {
    "critical": "dash",
    "warn":     "dot",
    "ok":       "dot",
}


def load_events() -> list[dict]:
    """Load events from events.json and parse dates."""
    with open(_EVENTS_FILE, "r") as f:
        events = json.load(f)
    for ev in events:
        ev["date"] = pd.Timestamp(ev["date"])
    return events


def events_in_range(date_min, date_max) -> list[dict]:
    """Filter events to those within [date_min, date_max]."""
    events = load_events()
    date_min = pd.Timestamp(date_min)
    date_max = pd.Timestamp(date_max)
    return [ev for ev in events if date_min <= ev["date"] <= date_max]
