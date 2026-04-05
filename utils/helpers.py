"""Shared utilities: API key checks, CSS theme, and HTML component helpers."""
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


def apply_dark_theme():
    """Inject consistent dark theme CSS. Call once per page, after set_page_config."""
    import streamlit as st
    st.markdown("""
<style>
  .stApp { background-color: #0d1117; }
  .block-container { padding-top: 2.5rem; max-width: 1400px; }
  h1 { color: #e6edf3 !important; font-size: 1.5rem !important; letter-spacing: -0.3px; }
  h2 { color: #8b949e !important; font-size: 1.1rem !important; font-weight: 500; margin-bottom: 0.3rem; }
  h3 { color: #8b949e !important; font-size: 0.95rem !important; font-weight: 500; }
  hr { border-color: rgba(255,255,255,0.07); margin: 0.6rem 0; }
  .kpi-card {
    background: #161b22;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 4px;
  }
  .kpi-label { color: #8b949e; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 4px; }
  .kpi-value { color: #e6edf3; font-size: 1.5rem; font-weight: 600; line-height: 1.1; }
  .kpi-delta { font-size: 0.78rem; margin-top: 3px; }
  .delta-red   { color: #f85149; }
  .delta-amber { color: #d29922; }
  .delta-green { color: #3fb950; }
  .delta-blue  { color: #58a6ff; }
  .pill-critical { background:#3d1a1a; color:#f85149; border:1px solid #f85149; border-radius:4px; padding:2px 8px; font-size:0.72rem; font-weight:600; letter-spacing:.5px; }
  .pill-warn     { background:#2d2200; color:#d29922; border:1px solid #d29922; border-radius:4px; padding:2px 8px; font-size:0.72rem; font-weight:600; letter-spacing:.5px; }
  .pill-ok       { background:#0d2240; color:#58a6ff; border:1px solid #58a6ff; border-radius:4px; padding:2px 8px; font-size:0.72rem; font-weight:600; letter-spacing:.5px; }
  .pill-good     { background:#0d2d17; color:#3fb950; border:1px solid #3fb950; border-radius:4px; padding:2px 8px; font-size:0.72rem; font-weight:600; letter-spacing:.5px; }
  .commentary {
    background: #161b22;
    border-left: 3px solid #58a6ff;
    border-radius: 0 6px 6px 0;
    padding: 12px 16px;
    color: #c9d1d9;
    font-size: 0.88rem;
    line-height: 1.6;
    margin: 8px 0 16px 0;
  }
  .commentary-critical { border-left-color: #f85149; }
  .commentary-warn     { border-left-color: #d29922; }
  .commentary-good     { border-left-color: #3fb950; }
  .stTabs [data-baseweb="tab-list"] { gap: 4px; background: transparent; }
  .stTabs [data-baseweb="tab"] {
    background: #161b22;
    border-radius: 6px 6px 0 0;
    color: #8b949e;
    font-size: 0.82rem;
    padding: 6px 16px;
    border: 1px solid rgba(255,255,255,0.07);
    border-bottom: none;
  }
  .stTabs [aria-selected="true"] { background: #21262d; color: #e6edf3 !important; }
  .stTabs [data-baseweb="tab-panel"] {
    background: #21262d;
    border-radius: 0 6px 6px 6px;
    border: 1px solid rgba(255,255,255,0.07);
    padding: 16px;
  }
  .stAlert { border-radius: 6px; font-size: 0.85rem; }
  div[data-testid="stCaption"] { color: #484f58 !important; font-size: 0.72rem !important; }
</style>
""", unsafe_allow_html=True)


# ── HTML component helpers ─────────────────────────────────────────────────

PILL = {"critical": "pill-critical", "warn": "pill-warn", "ok": "pill-ok", "good": "pill-good"}
BORDER = {"critical": "commentary-critical", "warn": "commentary-warn", "ok": "", "good": "commentary-good"}
STATUS_LABEL = {"critical": "CRITICAL", "warn": "ELEVATED", "ok": "NORMAL", "good": "HEALTHY"}


def pill(status: str) -> str:
    return f'<span class="{PILL[status]}">{STATUS_LABEL[status]}</span>'


def commentary(text: str, status: str = "ok") -> str:
    return f'<div class="commentary {BORDER[status]}">{text}</div>'


def kpi_card(label: str, value: str, delta_html: str) -> str:
    return f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-delta">{delta_html}</div>
    </div>"""


def delta_span(text: str, color: str = "blue") -> str:
    return f'<span class="delta-{color}">{text}</span>'
