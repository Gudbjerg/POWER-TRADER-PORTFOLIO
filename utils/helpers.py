"""Shared utilities: API key checks plus the legacy HTML helpers used across pages.

The visual language lives in ``utils.theme``. The helpers below keep their original
signatures and behaviour (each returns an HTML string for ``st.markdown(...,
unsafe_allow_html=True)``) and delegate their styling to the design system.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Streamlit Cloud secrets → environment variables ────────────────────────
# Allows the same os.getenv() calls throughout the codebase to work both
# locally (via .env) and on Streamlit Community Cloud (via st.secrets).
def _sync_secrets_to_env() -> None:
    try:
        import streamlit as st
        for key in ("AGSI_API_KEY", "ENTSOE_API_KEY"):
            if not os.getenv(key):
                val = st.secrets.get(key, "")
                if val:
                    os.environ[key] = val
    except Exception:
        pass

_sync_secrets_to_env()


def has_entsoe_key() -> bool:
    return bool(os.getenv("ENTSOE_API_KEY", "").strip())


def has_agsi_key() -> bool:
    return bool(os.getenv("AGSI_API_KEY", "").strip())


def has_torch() -> bool:
    """Check if PyTorch and transformers are installed (required for Layer 4 ML models)."""
    try:
        import torch          # noqa: F401
        import transformers   # noqa: F401
        return True
    except ImportError:
        return False


def apply_dark_theme():
    """Apply the platform design system. Call once per page, after set_page_config.

    The router in ``app.py`` injects the stylesheet once per run for every page, so this
    is normally a no-op. It still injects the stylesheet when a page module runs on its
    own (for example under Streamlit's AppTest).
    """
    from utils.theme import inject_theme
    inject_theme()


# ── HTML component helpers ─────────────────────────────────────────────────

PILL = {"critical": "pill-critical", "warn": "pill-warn", "ok": "pill-ok", "good": "pill-good"}
BORDER = {"critical": "commentary-critical", "warn": "commentary-warn", "ok": "", "good": "commentary-good"}
STATUS_LABEL = {"critical": "Critical", "warn": "Elevated", "ok": "Normal", "good": "Healthy"}
_STATUS_TONE = {"critical": "critical", "warn": "warn", "ok": "info", "good": "good"}


def pill(status: str) -> str:
    return f'<span class="{PILL[status]}">{STATUS_LABEL[status]}</span>'


def commentary(text: str, status: str = "ok") -> str:
    """Status-coloured commentary block. ``text`` may contain inline HTML."""
    from utils.theme import callout
    return callout(text, _STATUS_TONE[status])


def kpi_card(label: str, value: str, delta_html: str = "", accent: str | None = None,
             spark=None) -> str:
    """KPI tile. Extra optional ``accent`` (colour) and ``spark`` (list of numbers)."""
    from utils.theme import kpi_card as _kpi_card
    return _kpi_card(label, value, delta_html, accent=accent, spark=spark)


def delta_span(text: str, color: str = "blue") -> str:
    return f'<span class="delta-{color}">{text}</span>'


def ordinal(value) -> str:
    """English ordinal for a whole number: 1st, 2nd, 3rd, 4th, 11th, 12th, 13th, 21st, 72nd.

    Non-numeric or non-finite input returns "n/a" rather than raising.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if number != number or number in (float("inf"), float("-inf")):
        return "n/a"
    n = int(round(number))
    tail = abs(n) % 100
    suffix = "th" if 10 <= tail <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(abs(n) % 10, "th")
    return f"{n}{suffix}"
