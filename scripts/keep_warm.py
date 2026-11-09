"""
Open the platform's landing page in a headless browser so the Space stays awake and its
background warm-up (utils/warmup.py) runs. A plain HTTP request is not enough: Streamlit
runs the app only when a browser session connects over its websocket.

Usage: python -m scripts.keep_warm [url]
"""
from __future__ import annotations

import sys
import time

from playwright.sync_api import sync_playwright

URL = "https://tobgud-power-trader-portfolio.hf.space/"


def main(url: str = URL) -> None:
    t0 = time.monotonic()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.goto(url, timeout=300_000)
        # Wait until the landing page has rendered its header (the app is awake and running)
        page.wait_for_selector(".ptp-hero, [data-testid='stMain']", timeout=300_000)
        page.wait_for_timeout(60_000)   # keep the session open while the warm-up starts
        browser.close()
    print(f"warm: {url} in {time.monotonic() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else URL)
