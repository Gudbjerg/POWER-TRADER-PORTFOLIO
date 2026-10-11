"""GIE AGSI+/ALSI access (data/gie.py) and the storage and LNG fetchers built on it."""
import pandas as pd
import pytest
import requests

from config.settings import GIE_MAX_RETRY_AFTER_S, GIE_RETRY_WAITS_S
from data import gas_storage as gs
from data import gie
from data import lng_terminals as lt


class _Response:
    def __init__(self, status=200, payload=None, headers=None):
        self.status_code = status
        self._payload = payload if payload is not None else {}
        self.headers = headers or {}

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _Server:
    """Plays back a scripted sequence of answers and records each request."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls: list[dict] = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params or {}), "headers": headers})
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


@pytest.fixture
def sleeps(monkeypatch):
    waited: list[float] = []
    monkeypatch.setattr(gie, "_sleep", waited.append)
    return waited


def _serve(monkeypatch, *answers) -> _Server:
    server = _Server(*answers)
    monkeypatch.setattr(gie.requests, "get", server.get)
    return server


def _page(rows, last_page=1):
    return _Response(200, {"last_page": last_page, "data": rows})


# ── Single requests ───────────────────────────────────────────────────────────

def test_rate_limit_is_retried_after_the_requested_wait(monkeypatch, sleeps):
    server = _serve(monkeypatch, _Response(429, headers={"Retry-After": "3"}), _page([{"a": 1}]))
    assert gie.get_json("https://agsi.gie.eu/api", {"page": 1})["data"] == [{"a": 1}]
    assert len(server.calls) == 2
    assert sleeps == [max(GIE_RETRY_WAITS_S[0], 3.0)]


def test_long_retry_after_is_capped(monkeypatch, sleeps):
    _serve(monkeypatch, _Response(429, headers={"Retry-After": "120"}), _page([]))
    gie.get_json("https://agsi.gie.eu/api", {})
    assert sleeps == [GIE_MAX_RETRY_AFTER_S]


def test_gateway_errors_give_up_after_the_last_attempt(monkeypatch, sleeps):
    server = _serve(monkeypatch, *[_Response(503) for _ in range(len(GIE_RETRY_WAITS_S) + 1)])
    with pytest.raises(gie.GieUnavailable, match="HTTP 503"):
        gie.get_json("https://agsi.gie.eu/api", {})
    assert len(server.calls) == len(GIE_RETRY_WAITS_S) + 1
    assert sleeps == list(GIE_RETRY_WAITS_S)


def test_connection_errors_are_retried(monkeypatch, sleeps):
    server = _serve(monkeypatch, requests.exceptions.ConnectionError("reset"), _page([{"a": 1}]))
    assert gie.get_json("https://alsi.gie.eu/api", {})["data"] == [{"a": 1}]
    assert len(server.calls) == 2 and sleeps == [GIE_RETRY_WAITS_S[0]]


def test_a_read_timeout_is_not_retried(monkeypatch, sleeps):
    server = _serve(monkeypatch, requests.exceptions.ReadTimeout("slow"))
    with pytest.raises(gie.GieUnavailable, match="did not answer"):
        gie.get_json("https://agsi.gie.eu/api", {})
    assert len(server.calls) == 1 and sleeps == []


@pytest.mark.parametrize("answer", [
    _Response(200, {"error": "access denied"}),
    _Response(404),
    _Response(200, ValueError("not json")),
    _Response(200, ["not", "an", "object"]),
])
def test_permanent_problems_fail_at_once(monkeypatch, sleeps, answer):
    server = _serve(monkeypatch, answer)
    with pytest.raises(gie.GieUnavailable):
        gie.get_json("https://agsi.gie.eu/api", {})
    assert len(server.calls) == 1 and sleeps == []


def test_the_key_goes_in_the_x_key_header(monkeypatch, sleeps):
    monkeypatch.setenv("AGSI_API_KEY", "test-key")
    server = _serve(monkeypatch, _page([]))
    gie.get_json("https://agsi.gie.eu/api", {})
    assert server.calls[0]["headers"] == {"x-key": "test-key"}


# ── Pagination ────────────────────────────────────────────────────────────────

def test_every_page_is_fetched(monkeypatch, sleeps):
    server = _serve(monkeypatch, _page([{"d": 1}], 3), _page([{"d": 2}], 3), _page([{"d": 3}], 3))
    rows = gie.fetch_all_pages("https://agsi.gie.eu/api/data/eu", {"size": 300})
    assert rows == [{"d": 1}, {"d": 2}, {"d": 3}]
    assert [c["params"]["page"] for c in server.calls] == [1, 2, 3]
    assert all(c["params"]["size"] == 300 for c in server.calls)


def test_a_failed_page_fails_the_whole_history(monkeypatch, sleeps):
    failures = [_Response(503) for _ in range(len(GIE_RETRY_WAITS_S) + 1)]
    _serve(monkeypatch, _page([{"d": 1}], 3), *failures)
    with pytest.raises(gie.GieUnavailable):
        gie.fetch_all_pages("https://agsi.gie.eu/api/data/eu", {})


def test_an_empty_first_page_is_an_empty_history(monkeypatch, sleeps):
    _serve(monkeypatch, _page([], 0))
    assert gie.fetch_all_pages("https://alsi.gie.eu/api", {}) == []


# ── Cached fetchers ───────────────────────────────────────────────────────────

@pytest.fixture
def fresh_caches(monkeypatch):
    monkeypatch.setenv("AGSI_API_KEY", "test-key")
    caches = [gs.fetch_storage_eu, gs.fetch_storage_country, gs._storage_history,
              lt.fetch_lng_country, lt._lng_history]
    for cached in caches:
        cached.clear()
    yield
    for cached in caches:
        cached.clear()


def _storage_rows():
    return [{"gasDayStart": "2026-10-01", "full": "73.1"}, {"gasDayStart": "2026-10-02", "full": "73.5"}]


def test_a_failed_storage_fetch_is_not_kept_for_an_hour(monkeypatch, sleeps, fresh_caches):
    outcomes = iter([gie.GieUnavailable("HTTP 429"), _storage_rows()])

    def fake_fetch_all_pages(url, params):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(gs, "fetch_all_pages", fake_fetch_all_pages)
    assert gs.fetch_storage_eu("2020-10-01", "2026-10-02").empty

    # Once the short failure cache lapses (cleared here), the next call fetches again
    # because the hour-long cache never stored the failure.
    gs.fetch_storage_eu.clear()
    frame = gs.fetch_storage_eu("2020-10-01", "2026-10-02")
    assert frame["full"].tolist() == [73.1, 73.5]

    # A complete history stays in the hour-long cache.
    gs.fetch_storage_eu.clear()
    assert gs.fetch_storage_eu("2020-10-01", "2026-10-02")["full"].tolist() == [73.1, 73.5]


def test_without_a_key_nothing_is_requested(monkeypatch, sleeps, fresh_caches):
    monkeypatch.setenv("AGSI_API_KEY", " ")
    monkeypatch.setattr(gs, "fetch_all_pages", lambda *a, **k: pytest.fail("no request expected"))
    assert gs.fetch_storage_country("DE", "2020-10-01", "2026-10-02").empty


def test_lng_totals_name_the_countries_that_failed(monkeypatch, sleeps, fresh_caches):
    def fake_fetch_all_pages(url, params):
        if params["country"] == "GB":
            raise gie.GieUnavailable("HTTP 429")
        return [{"gasDayStart": d.strftime("%Y-%m-%d"), "sendOut": "100"}
                for d in pd.date_range(end=pd.Timestamp.now().normalize(), periods=20, freq="D")]

    monkeypatch.setattr(lt, "fetch_all_pages", fake_fetch_all_pages)
    data = lt.get_lng_data()
    assert data["unavailable"] == ["GB"]
    assert set(data["sendout"]["country"]) == {"BE", "NL", "FR"}
    assert data["totals"]["total_sendout"].iloc[-1] == 300.0


def test_lng_without_failures_reports_none(monkeypatch, sleeps, fresh_caches):
    monkeypatch.setattr(lt, "fetch_all_pages", lambda url, params: [])
    data = lt.get_lng_data()
    assert data["unavailable"] == []
    assert data["totals"].empty
