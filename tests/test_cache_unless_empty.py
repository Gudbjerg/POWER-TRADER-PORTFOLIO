"""An empty loader result is not kept for the full cache lifetime."""
import pandas as pd

from data.deadline import _is_empty, cache_unless_empty


def test_empty_results_are_refetched_and_full_ones_cached():
    calls = {"n": 0}
    answers = [pd.DataFrame(), pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [2]})]

    @cache_unless_empty(ttl=3600, retry_after=0)
    def loader(key: str) -> pd.DataFrame:
        calls["n"] += 1
        return answers[min(calls["n"] - 1, 2)]

    assert loader("x").empty            # source down: empty frame passed through
    assert loader("x")["a"].iloc[0] == 1  # fetched again, now full
    assert loader("x")["a"].iloc[0] == 1  # full result served from the long cache
    assert calls["n"] == 2


def test_is_empty_on_dicts_of_frames():
    assert _is_empty({"daily": pd.DataFrame(), "source": "unavailable"})
    assert not _is_empty({"daily": pd.DataFrame({"a": [1]})})
    assert not _is_empty({"source": "x"})


def test_two_loaders_never_share_a_cache_entry():
    @cache_unless_empty(ttl=3600)
    def first() -> pd.DataFrame:
        return pd.DataFrame({"src": ["first"]})

    @cache_unless_empty(ttl=3600)
    def second() -> dict:
        return {"daily": pd.DataFrame({"src": ["second"]})}

    assert first()["src"].iloc[0] == "first"
    assert second()["daily"]["src"].iloc[0] == "second"
