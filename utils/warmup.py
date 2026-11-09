"""
Background cache warm-up.

Streamlit computes a cached loader the first time a visitor needs it, so after a restart
(or once a cache entry expires) the first visitor to each page waits for every source that
page reads. The warm-up starts once per server process, on the first page view, and fills
the shared caches page by page in a background thread while the visitor reads the page
they opened; it repeats every ``REWARM_S`` seconds so the hourly caches rarely expire in
front of a visitor. Each loader is called with the same arguments its page uses, so the
page and the warm-up share one cache entry. Failures are logged and skipped: a source that
does not answer is fetched again by the page as before.
"""
from __future__ import annotations

import sys
import threading
import time

REWARM_S = 45 * 60          # below the one-hour lifetime of most market-data caches
START_DELAY_S = 5           # let the first page render before the warm-up competes with it

_started = False
_lock = threading.Lock()


def _steps():
    """(label, callable) pairs in the order visitors usually move through the pages."""
    from data.gas_storage import get_storage_data
    from data.prices import get_ttf_data
    from data.spot_prices import get_spot_price_data
    from data.lng_terminals import get_lng_data
    from data.power_flows import get_flow_data
    from data.generation import get_generation_data
    from data.hydro import get_hydro_data
    from data.solar import get_solar_data
    from models.feature_assembly import load_feature_matrix
    from models.ttf_backtest import fetch_ttf_history
    from data import weather as W
    from models.weather_impact import load_weather_model

    return [
        ("storage", get_storage_data),
        ("ttf", get_ttf_data),
        ("day-ahead prices", get_spot_price_data),
        ("lng", get_lng_data),
        ("flows", get_flow_data),
        ("generation", get_generation_data),
        ("hydro", get_hydro_data),
        ("solar", get_solar_data),
        ("ttf history", lambda: fetch_ttf_history(years=7)),
        ("feature matrix", lambda: load_feature_matrix(years=3)),
        ("weather model", load_weather_model),
        ("weather forecast", lambda: W.fetch_forecast_indices(10)),
        ("weather ensemble", lambda: W.fetch_wind_ensemble(10)),
        ("weather revision", lambda: W.fetch_wind_revision(7)),
        ("catchments", lambda: (W.fetch_hydro_outlook(10), W.fetch_hydro_normals())),
    ]


def _timed(label, fn) -> None:
    t0 = time.monotonic()
    try:
        fn()
        print(f"[warmup] {label}: {time.monotonic() - t0:.1f}s", file=sys.stderr)
    except Exception as exc:   # a failed source is retried by its page
        print(f"[warmup] {label} failed: {type(exc).__name__}", file=sys.stderr)


def _run_once() -> None:
    from concurrent.futures import ThreadPoolExecutor
    # Four at a time: fast enough to finish within a few minutes, gentle enough on the
    # rate-limited sources (GIE, Yahoo Finance, Open-Meteo) that share the Space's address.
    with ThreadPoolExecutor(max_workers=4) as pool:
        for label, fn in _steps():
            pool.submit(_timed, label, fn)


def _loop() -> None:
    time.sleep(START_DELAY_S)
    while True:
        _run_once()
        time.sleep(REWARM_S)


def start_warmup() -> None:
    """Start the warm-up thread once per process (safe to call on every rerun)."""
    global _started
    import os
    if _started or os.getenv("PTP_DISABLE_WARMUP") or "pytest" in sys.modules:
        return
    with _lock:
        if _started:
            return
        _started = True
    try:
        import logging
        # Cached loaders run fine outside a script run; Streamlit only logs that the
        # thread has no script context. No context is attached, so nothing the loaders
        # might write (a warning box, say) can land in a visitor's page.
        for name in ("streamlit.runtime.scriptrunner_utils.script_run_context",
                     "streamlit.runtime.caching.cache_data_api"):
            logging.getLogger(name).setLevel(logging.ERROR)
        threading.Thread(target=_loop, name="ptp-warmup", daemon=True).start()
    except Exception as exc:
        print(f"[warmup] not started: {type(exc).__name__}", file=sys.stderr)
