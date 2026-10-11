"""
Shared HTTP access to the GIE transparency APIs: AGSI+ (gas storage) and ALSI (LNG).

Both APIs take the same key (``AGSI_API_KEY``) in the ``x-key`` header and paginate
with ``page`` and ``last_page``. GIE allows 60 requests per minute per address; on
shared cloud egress other tenants count against that limit, so HTTP 429 and gateway
errors are treated as transient and retried (see the GIE request policy in
``config/settings.py``). Anything that still fails raises :class:`GieUnavailable`;
callers decide what to show and how long to remember the failure. A paginated fetch
either returns every page or raises, so a partial history is never cached as if it
were complete.
"""
from __future__ import annotations

import os
import time
from email.utils import parsedate_to_datetime

import requests

from config.settings import (
    GIE_MAX_RETRY_AFTER_S,
    GIE_RETRY_WAITS_S,
    GIE_TIMEOUT_S,
)

_RETRY_STATUS = {429, 500, 502, 503, 504}
_sleep = time.sleep   # replaced in tests


class GieUnavailable(RuntimeError):
    """GIE did not deliver a usable answer (after retries where retrying makes sense)."""


def has_key() -> bool:
    return bool(os.getenv("AGSI_API_KEY", "").strip())


def _headers() -> dict:
    return {"x-key": os.getenv("AGSI_API_KEY", "")}


def _retry_after_s(response: requests.Response) -> float | None:
    """Seconds requested by a Retry-After header (delta-seconds or HTTP date), if any."""
    value = (response.headers.get("Retry-After") or "").strip()
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = parsedate_to_datetime(value).timestamp() - time.time()
        except (TypeError, ValueError, OverflowError):
            return None
    return min(max(seconds, 0.0), GIE_MAX_RETRY_AFTER_S)


def get_json(url: str, params: dict) -> dict:
    """GET one GIE page and return its JSON object.

    Retries HTTP 429, 5xx gateway answers and connection failures up to
    ``len(GIE_RETRY_WAITS_S)`` times. A read timeout, any other HTTP error, a body that
    is not a JSON object, or GIE's ``{"error": "access denied"}`` raises
    :class:`GieUnavailable` straight away.
    """
    attempts = len(GIE_RETRY_WAITS_S) + 1
    problem = "no attempt made"
    for attempt in range(attempts):
        try:
            response = requests.get(url, params=params, headers=_headers(), timeout=GIE_TIMEOUT_S)
        except requests.exceptions.ReadTimeout as exc:
            raise GieUnavailable(f"GIE did not answer within {GIE_TIMEOUT_S} s") from exc
        except requests.exceptions.RequestException as exc:
            problem = f"connection failed ({type(exc).__name__})"
            wait = None
        else:
            if response.status_code in _RETRY_STATUS:
                problem = f"HTTP {response.status_code}"
                wait = _retry_after_s(response)
            else:
                if response.status_code >= 400:
                    raise GieUnavailable(f"HTTP {response.status_code}")
                try:
                    data = response.json()
                except ValueError as exc:
                    raise GieUnavailable("GIE answered with a body that is not JSON") from exc
                if not isinstance(data, dict):
                    raise GieUnavailable("GIE answered with an unexpected JSON shape")
                if str(data.get("error", "")).strip().lower() == "access denied":
                    raise GieUnavailable("GIE rejected the API key (access denied)")
                return data
        if attempt + 1 < attempts:
            default_wait = GIE_RETRY_WAITS_S[attempt]
            _sleep(max(default_wait, wait or 0.0))
    raise GieUnavailable(f"GIE request failed after {attempts} attempts: {problem}")


def fetch_all_pages(url: str, params: dict) -> list[dict]:
    """Return the ``data`` records of every page, or raise :class:`GieUnavailable`."""
    records: list[dict] = []
    page = 1
    while True:
        payload = get_json(url, {**params, "page": page})
        batch = payload.get("data") or []
        if not isinstance(batch, list):
            raise GieUnavailable("GIE answered with an unexpected data field")
        records.extend(batch)
        try:
            last_page = int(payload.get("last_page") or 1)
        except (TypeError, ValueError):
            last_page = 1
        if not batch or page >= last_page:
            return records
        page += 1
