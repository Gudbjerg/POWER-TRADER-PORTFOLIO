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
