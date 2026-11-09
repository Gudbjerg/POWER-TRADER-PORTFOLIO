"""
Wall-clock deadline for blocking API calls.

``requests`` timeouts bound each socket read, not the whole response: a server that
trickles bytes (the ENTSO-E Transparency Platform does under load) can hold a call open
for many minutes. ``call_with_deadline`` runs the call in a daemon thread and gives up
after ``seconds``; the abandoned thread finishes or dies on its own and its result is
discarded, so a page never waits longer than the deadline for one request.
"""
from __future__ import annotations

import threading
from typing import Any, Callable

ENTSOE_DEADLINE_S = 60   # longest wait for one ENTSO-E request, retries included


def call_with_deadline(fn: Callable[..., Any], *args, seconds: float = ENTSOE_DEADLINE_S, **kwargs) -> Any:
    """Return ``fn(*args, **kwargs)``; raise TimeoutError after ``seconds``, re-raise its errors."""
    box: dict[str, Any] = {}

    def run() -> None:
        try:
            box["value"] = fn(*args, **kwargs)
        except BaseException as exc:   # noqa: BLE001 - re-raised in the caller's thread
            box["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        raise TimeoutError(f"{getattr(fn, '__name__', 'call')} exceeded {seconds:.0f} s")
    if "error" in box:
        raise box["error"]
    return box.get("value")


class _NoData(Exception):
    """Raised inside a cached loader so an empty result is not cached for the full lifetime."""

    def __init__(self, value: Any):
        super().__init__("no data")
        self.value = value


def _is_empty(value: Any) -> bool:
    import pandas as pd
    if value is None:
        return True
    if isinstance(value, pd.DataFrame):
        return value.empty
    if isinstance(value, dict):
        frames = [v for v in value.values() if isinstance(v, pd.DataFrame)]
        return bool(frames) and all(f.empty for f in frames)
    return False


def cache_unless_empty(ttl: int = 3600, retry_after: int = 300):
    """Like ``st.cache_data(ttl=ttl)``, but an empty result is remembered only for ``retry_after``.

    A source that is briefly down (ENTSO-E often is) would otherwise blank its panel for the
    whole cache lifetime. A full result is cached for ``ttl``; an empty one (an empty frame,
    or a dict whose frames are all empty) is served for at most ``retry_after`` seconds and
    then fetched again. Streamlit does not cache a call that raises, which is what keeps the
    empty result out of the long-lived cache.
    """
    import functools
    import streamlit as st

    def deco(fn):
        @st.cache_data(ttl=ttl, show_spinner=False)
        def _full(*args, **kwargs):
            out = fn(*args, **kwargs)
            if _is_empty(out):
                raise _NoData(out)
            return out

        @st.cache_data(ttl=retry_after, show_spinner=False)
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return _full(*args, **kwargs)
            except _NoData as empty:
                return empty.value

        return wrapper
    return deco
