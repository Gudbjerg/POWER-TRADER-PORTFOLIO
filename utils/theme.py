"""
Design system for the European Gas & Power Market Intelligence Platform.

One module owns the visual language of the app:

* Design tokens (colours, typography, radii, spacing, per-layer accents).
* ``inject_theme()``: the global stylesheet (fonts, app chrome, top navigation,
  tabs, expanders, buttons, page links, tables, alerts, metrics, sidebar).
* HTML components returned as strings for ``st.markdown(..., unsafe_allow_html=True)``:
  ``chip``, ``kpi_card``, ``sparkline``, ``callout``, ``ticker_strip``, ``layer_card``.
* Page-structure renderers that write to the app directly: ``page_header``,
  ``section_header``, ``page_footer``.
* A registered Plotly template (``PLOTLY_TEMPLATE``) plus ``CHART`` tokens for the
  chart chrome (backgrounds, grid, axis text). Series colours stay with each chart.

Text arguments documented as plain text are HTML-escaped. Arguments whose name ends
in ``_html`` (and ``callout`` text, which mirrors the legacy ``commentary`` helper)
are inserted as HTML.
"""
from __future__ import annotations

import html as _html
import math
import re
from typing import Iterable, Sequence

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ══════════════════════════════════════════════════════════════════════════════
# Tokens
# ══════════════════════════════════════════════════════════════════════════════

COLORS: dict[str, str] = {
    # Surfaces (darkest to lightest)
    "bg":            "#0a0e14",
    "surface_1":     "#0f141c",
    "surface_2":     "#141b25",
    "surface_3":     "#1a2330",
    # Lines
    "border":        "rgba(148,163,184,0.14)",
    "border_soft":   "rgba(148,163,184,0.08)",
    "border_strong": "rgba(148,163,184,0.26)",
    # Text
    "text":          "#e6edf3",
    "text_2":        "#c4cdd8",
    "muted":         "#8b98a9",
    "faint":         "#5b6676",
    # Brand
    "accent":        "#f5b301",   # energy amber (gas flame)
    "accent_2":      "#38bdf8",   # power cyan
    # Semantic
    "up":            "#3fb950",
    "down":          "#f85149",
    "warn":          "#d29922",
    "info":          "#58a6ff",
}

# Per-layer accents, used for page headers, KPI rules and the landing page cards.
ACCENTS: dict[str, str] = {
    "home":  COLORS["accent"],
    "l1":    "#38bdf8",   # Live Monitor: cyan
    "l2":    "#3fb950",   # Quant Analysis: green
    "l3":    "#f5b301",   # Macro Signals: amber
    "l4":    "#a78bfa",   # ML Models: violet
    "l5":    "#fb7185",   # Mispricing Dashboard: rose
    "risk":  "#d4ac3a",   # Risk Dashboard: gold
    "bess":  "#2dd4bf",   # BESS Optimiser: teal
    "globe": "#f0883e",   # Energy Globe: orange
}

# Semantic tones shared by chips, callouts and pills.
TONES: dict[str, str] = {
    "neutral": COLORS["muted"],
    "info":    COLORS["info"],
    "ok":      COLORS["info"],
    "up":      COLORS["up"],
    "good":    COLORS["up"],
    "down":    COLORS["down"],
    "critical": COLORS["down"],
    "warn":    COLORS["warn"],
    "accent":  COLORS["accent"],
    "power":   COLORS["accent_2"],
}

FONT_SANS = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
FONT_MONO = "'JetBrains Mono', 'SFMono-Regular', Menlo, Consolas, 'Liberation Mono', monospace"
GOOGLE_FONTS_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap"
)

RADII: dict[str, str] = {"sm": "6px", "md": "10px", "lg": "14px", "pill": "999px"}
SPACE: dict[str, str] = {"xs": "0.25rem", "sm": "0.5rem", "md": "1rem", "lg": "1.5rem", "xl": "2.5rem"}

# Categorical palette for charts that do not set explicit series colours.
COLORWAY: list[str] = [
    "#38bdf8", "#f5b301", "#3fb950", "#a78bfa", "#fb7185",
    "#2dd4bf", "#d4ac3a", "#58a6ff", "#f0883e", "#8b98a9",
]

# Chart chrome. Every Plotly chart sits on a surface_1 panel (see the stPlotlyChart
# rule in the stylesheet): the paper is transparent and the plot area uses the same
# opaque surface, which also keeps Plotly's hover spike lines dark instead of white.
CHART: dict[str, str] = {
    "bg":                "rgba(0,0,0,0)",
    "panel_bg":          COLORS["surface_1"],
    "grid":              "rgba(148,163,184,0.09)",
    "zero":              "rgba(148,163,184,0.30)",
    "axis":              "rgba(148,163,184,0.22)",
    "text":              COLORS["text_2"],
    "muted":             COLORS["muted"],
    "faint":             COLORS["faint"],
    "annotation_bg":     COLORS["surface_2"],
    "annotation_border": COLORS["border_strong"],
    "edge":              COLORS["surface_1"],   # separator stroke between adjacent bars
}

PLOTLY_TEMPLATE = "ptp"

STANDARD_DISCLAIMER = "For informational purposes only. Not financial advice."
AUTHOR = "Tobias Gudbjerg"


def _esc(text: object) -> str:
    """HTML-escape plain text (quotes left intact for readability)."""
    return _html.escape("" if text is None else str(text), quote=False)


def _attr(text: object) -> str:
    """Escape a value for use inside a double-quoted HTML attribute."""
    return _html.escape("" if text is None else str(text), quote=True)


def _rgba(hex_color: str, alpha: float) -> str:
    """Convert '#rrggbb' to an rgba() string. Non-hex inputs are returned unchanged."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return hex_color
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _tone(tone: str | None) -> str:
    return TONES.get(tone or "neutral", tone if (tone or "").startswith("#") else COLORS["muted"])


# ══════════════════════════════════════════════════════════════════════════════
# Plotly template
# ══════════════════════════════════════════════════════════════════════════════

def _register_plotly_template() -> None:
    """Register the 'ptp' template (plotly_dark base + tokens) and make it the default."""
    template = go.layout.Template(pio.templates["plotly_dark"])
    axis = dict(
        gridcolor=CHART["grid"],
        zerolinecolor=CHART["zero"],
        linecolor=CHART["axis"],
        tickcolor=CHART["axis"],
        tickfont=dict(color=CHART["muted"], size=11),
        title=dict(font=dict(color=CHART["muted"], size=11)),
        showline=False,
        ticks="",
        automargin=True,
        spikecolor="rgba(148,163,184,0.55)",
        spikethickness=1,
        spikedash="solid",
    )
    template.layout.update(
        font=dict(family=FONT_SANS, size=12, color=CHART["text"]),
        paper_bgcolor=CHART["bg"],
        plot_bgcolor=CHART["panel_bg"],
        colorway=COLORWAY,
        title=dict(font=dict(size=13, color=COLORS["text"]), x=0, xanchor="left"),
        xaxis=axis,
        yaxis=axis,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=CHART["text"])),
        hoverlabel=dict(
            bgcolor=COLORS["surface_2"],
            bordercolor=COLORS["border_strong"],
            font=dict(family=FONT_SANS, size=12, color=COLORS["text"]),
        ),
        modebar=dict(bgcolor="rgba(0,0,0,0)", color=COLORS["faint"], activecolor=COLORS["accent"]),
        margin=dict(l=48, r=16, t=36, b=40),
    )
    pio.templates[PLOTLY_TEMPLATE] = template
    pio.templates.default = PLOTLY_TEMPLATE


_register_plotly_template()


# ══════════════════════════════════════════════════════════════════════════════
# Global stylesheet
# ══════════════════════════════════════════════════════════════════════════════

def _root_variables() -> str:
    tokens = {
        **{f"--ptp-{k.replace('_', '-')}": v for k, v in COLORS.items()},
        **{f"--ptp-{k}": v for k, v in ACCENTS.items()},
        "--ptp-font": FONT_SANS,
        "--ptp-mono": FONT_MONO,
        "--ptp-radius-sm": RADII["sm"],
        "--ptp-radius": RADII["md"],
        "--ptp-radius-lg": RADII["lg"],
        **{f"--ptp-space-{k}": v for k, v in SPACE.items()},
        "--ptp-shadow": "0 1px 2px rgba(0,0,0,0.35), 0 12px 32px -18px rgba(0,0,0,0.65)",
        "--ptp-highlight": "inset 0 1px 0 rgba(255,255,255,0.035)",
    }
    return ":root{" + "".join(f"{k}:{v};" for k, v in tokens.items()) + "}"


_CSS = r"""
/* ── Base ─────────────────────────────────────────────────────────────── */
html, body, .stApp { -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }
.stApp {
  background:
    radial-gradient(1100px 520px at 12% -12%, rgba(56,189,248,0.055), transparent 62%),
    radial-gradient(900px 480px at 92% -18%, rgba(245,179,1,0.045), transparent 60%),
    var(--ptp-bg);
  color: var(--ptp-text);
  font-family: var(--ptp-font);
}
::selection { background: rgba(245,179,1,0.28); color: #fff; }
* { scrollbar-width: thin; scrollbar-color: rgba(148,163,184,0.22) transparent; }
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(148,163,184,0.20); border-radius: 10px; border: 2px solid transparent; background-clip: content-box; }
::-webkit-scrollbar-thumb:hover { background: rgba(148,163,184,0.34); background-clip: content-box; }
a:focus-visible, button:focus-visible, [tabindex]:focus-visible { outline: 2px solid rgba(245,179,1,0.7); outline-offset: 2px; }

/* ── Streamlit chrome ─────────────────────────────────────────────────── */
[data-testid="stAppDeployButton"], [data-testid="stDecoration"], footer { display: none !important; }
header[data-testid="stHeader"] {
  background: rgba(10,14,20,0.80);
  backdrop-filter: saturate(150%) blur(14px);
  -webkit-backdrop-filter: saturate(150%) blur(14px);
  border-bottom: 1px solid var(--ptp-border);
  box-shadow: 0 10px 30px -22px rgba(0,0,0,0.9);
}
[data-testid="stToolbar"] { gap: 0.25rem; }
[data-testid="stMainMenuButton"], [data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapseButton"] button { color: var(--ptp-muted); }
[data-testid="stMainMenuButton"]:hover, [data-testid="stExpandSidebarButton"]:hover { color: var(--ptp-text); }
[data-testid="stStatusWidget"] { color: var(--ptp-muted); }
[data-testid="stHeaderLogo"], [data-testid="stSidebarLogo"] { height: 1.75rem; max-height: 1.75rem; }

/* Top navigation */
[data-testid="stTopNavLink"], [data-testid="stTopNavSection"] {
  border-radius: 8px;
  padding: 0.3rem 0.7rem;
  transition: background-color .15s ease, color .15s ease, box-shadow .15s ease;
}
[data-testid="stTopNavLink"] p, [data-testid="stTopNavSection"] p {
  font-size: 0.85rem; font-weight: 500; color: var(--ptp-muted); letter-spacing: 0.005em;
}
[data-testid="stTopNavLink"]:hover, [data-testid="stTopNavSection"]:hover { background: rgba(148,163,184,0.08); }
[data-testid="stTopNavLink"]:hover p, [data-testid="stTopNavSection"]:hover p { color: var(--ptp-text); }
[data-testid="stTopNavSection"] svg { color: var(--ptp-faint); }
[data-testid="stTopNavLink"][aria-current="page"] {
  background: rgba(245,179,1,0.10);
  box-shadow: inset 0 0 0 1px rgba(245,179,1,0.30);
}
[data-testid="stTopNavLink"][aria-current="page"] p { color: var(--ptp-text); font-weight: 600; }
.rc-overflow-item:has([data-testid="stTopNavDropdownLink"][aria-current="page"]) [data-testid="stTopNavSection"] {
  background: rgba(245,179,1,0.10);
  box-shadow: inset 0 0 0 1px rgba(245,179,1,0.30);
}
.rc-overflow-item:has([data-testid="stTopNavDropdownLink"][aria-current="page"]) [data-testid="stTopNavSection"] p { color: var(--ptp-text); font-weight: 600; }
[data-testid="stTopNavPopover"] {
  background: var(--ptp-surface-2);
  border: 1px solid var(--ptp-border-strong);
  border-radius: 12px;
  box-shadow: 0 18px 40px -16px rgba(0,0,0,0.85);
  padding: 0.3rem;
  min-width: 13.5rem;
}
[data-testid="stTopNavDropdownLink"] { border-radius: 8px; padding: 0.42rem 0.65rem; }
[data-testid="stTopNavDropdownLink"] p { font-size: 0.85rem; color: var(--ptp-text-2); }
[data-testid="stTopNavDropdownLink"]:hover { background: rgba(148,163,184,0.10); }
[data-testid="stTopNavDropdownLink"][aria-current="page"] { background: rgba(245,179,1,0.10); }
[data-testid="stTopNavDropdownLink"][aria-current="page"] p { color: var(--ptp-text); font-weight: 600; }

/* Sidebar */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0d121a 0%, #0a0e14 100%);
  border-right: 1px solid var(--ptp-border);
}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-top: 0.5rem; }
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] h4 {
  font-size: 0.78rem !important; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.1em; color: var(--ptp-muted) !important;
}
[data-testid="stSidebarNavLink"] { border-radius: 8px; }
[data-testid="stSidebarNavLink"][aria-current="page"] { background: rgba(245,179,1,0.10); }
[data-testid="stNavSectionHeader"] { color: var(--ptp-faint); text-transform: uppercase; letter-spacing: 0.1em; font-size: 0.7rem; }

/* ── Layout ───────────────────────────────────────────────────────────── */
[data-testid="stMainBlockContainer"] {
  max-width: 1480px;
  padding: 5.25rem 2.75rem 3rem;
}
@media (max-width: 1100px) { [data-testid="stMainBlockContainer"] { padding: 5rem 1.75rem 2.5rem; } }
@media (max-width: 640px)  { [data-testid="stMainBlockContainer"] { padding: 4.5rem 1rem 2rem; } }

/* ── Typography ───────────────────────────────────────────────────────── */
[data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3, [data-testid="stMarkdownContainer"] h4,
[data-testid="stMarkdownContainer"] h5, [data-testid="stMarkdownContainer"] h6,
[data-testid="stHeading"] h1, [data-testid="stHeading"] h2, [data-testid="stHeading"] h3 {
  font-family: var(--ptp-font); color: var(--ptp-text); letter-spacing: -0.015em;
}
[data-testid="stMarkdownContainer"] h1 { font-size: 1.9rem; font-weight: 650; letter-spacing: -0.025em; }
[data-testid="stMarkdownContainer"] h2 { font-size: 1.4rem; font-weight: 600; letter-spacing: -0.02em; }
[data-testid="stMarkdownContainer"] h3, [data-testid="stHeading"] h3 {
  font-size: 1.08rem; font-weight: 600; padding-top: 0.6rem; padding-bottom: 0.35rem;
}
[data-testid="stMarkdownContainer"] h4 { font-size: 0.98rem; font-weight: 600; padding-top: 0.5rem; }
[data-testid="stMarkdownContainer"] h5 { font-size: 0.86rem; font-weight: 600; color: var(--ptp-text-2); }
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { color: var(--ptp-text-2); line-height: 1.62; }
[data-testid="stMarkdownContainer"] strong { color: var(--ptp-text); font-weight: 600; }
[data-testid="stMarkdownContainer"] a { color: #7dd3fc; text-decoration: none; }
[data-testid="stMarkdownContainer"] a:hover { color: #bae6fd; text-decoration: underline; text-underline-offset: 3px; }
[data-testid="stMarkdownContainer"] code {
  font-family: var(--ptp-mono); font-size: 0.82em; color: #e7c66b;
  background: rgba(148,163,184,0.10); border: 1px solid var(--ptp-border-soft);
  padding: 0.08em 0.38em; border-radius: 5px;
}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
  color: var(--ptp-muted) !important; font-size: 0.8rem !important; line-height: 1.55;
}
[data-testid="stMarkdownContainer"] hr, hr {
  border: none !important; border-top: 1px solid var(--ptp-border) !important;
  margin: 1.1rem 0 !important; background: transparent !important; height: 0 !important;
}

/* Markdown tables */
[data-testid="stMarkdownContainer"] table {
  border-collapse: separate; border-spacing: 0; margin: 0.4rem 0 1rem;
  border: 1px solid var(--ptp-border); border-radius: var(--ptp-radius); overflow: hidden;
  font-size: 0.85rem;
}
[data-testid="stMarkdownContainer"] thead th {
  background: var(--ptp-surface-2); color: var(--ptp-muted);
  font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em;
  text-align: left; padding: 0.55rem 0.8rem; border: none; border-bottom: 1px solid var(--ptp-border);
}
[data-testid="stMarkdownContainer"] tbody td {
  padding: 0.5rem 0.8rem; border: none; border-bottom: 1px solid var(--ptp-border-soft);
  color: var(--ptp-text-2); vertical-align: top;
}
[data-testid="stMarkdownContainer"] tbody tr:last-child td { border-bottom: none; }
[data-testid="stMarkdownContainer"] tbody tr:hover td { background: rgba(148,163,184,0.035); }
@media (max-width: 640px) {
  [data-testid="stMarkdownContainer"] table { display: block; max-width: 100%; overflow-x: auto; }
}

/* ── Tabs: segmented control ──────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  gap: 2px; padding: 4px; flex-wrap: wrap;
  width: fit-content; max-width: 100%;
  background: var(--ptp-surface-1);
  border: 1px solid var(--ptp-border);
  border-radius: 12px;
  box-shadow: var(--ptp-highlight);
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none !important; }
.stTabs button[data-baseweb="tab"] {
  height: auto; margin: 0; padding: 0.42rem 0.85rem;
  background: transparent; border: none; border-radius: 8px;
  white-space: nowrap; flex: 0 0 auto;
  transition: background-color .15s ease, color .15s ease, box-shadow .15s ease;
}
.stTabs button[data-baseweb="tab"] p { font-size: 0.83rem; font-weight: 500; color: var(--ptp-muted); }
.stTabs button[data-baseweb="tab"]:hover { background: rgba(148,163,184,0.07); }
.stTabs button[data-baseweb="tab"]:hover p { color: var(--ptp-text); }
.stTabs button[data-baseweb="tab"][aria-selected="true"] {
  background: var(--ptp-surface-3);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.05), 0 0 0 1px rgba(148,163,184,0.20), 0 1px 3px rgba(0,0,0,0.45);
}
.stTabs button[data-baseweb="tab"][aria-selected="true"] p { color: var(--ptp-text); font-weight: 600; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.1rem; }
@media (max-width: 768px) {
  .stTabs [data-baseweb="tab-list"] {
    flex-wrap: nowrap; overflow-x: auto; width: 100%;
    scrollbar-width: none; -webkit-overflow-scrolling: touch;
  }
  .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
}

/* ── Expanders ────────────────────────────────────────────────────────── */
[data-testid="stExpander"] details {
  background: var(--ptp-surface-1);
  border: 1px solid var(--ptp-border);
  border-radius: 12px;
  box-shadow: var(--ptp-highlight);
  overflow: hidden;
}
[data-testid="stExpander"] summary { padding: 0.65rem 1rem; transition: background-color .15s ease; }
[data-testid="stExpander"] summary:hover { background: rgba(148,163,184,0.045); }
[data-testid="stExpander"] summary p { font-size: 0.88rem; font-weight: 500; color: var(--ptp-text-2); }
[data-testid="stExpander"] summary:hover p { color: var(--ptp-text); }
[data-testid="stExpander"] summary [data-testid="stIconMaterial"] { color: var(--ptp-faint); }
[data-testid="stExpander"] details[open] > summary { border-bottom: 1px solid var(--ptp-border-soft); }
[data-testid="stExpanderDetails"] { padding: 0.9rem 1.1rem 1rem; }

/* ── Buttons and page links ───────────────────────────────────────────── */
[data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-tertiary"],
[data-testid="stDownloadButton"] button, [data-testid="stLinkButton"] a {
  background: var(--ptp-surface-2);
  border: 1px solid var(--ptp-border-strong);
  border-radius: 9px;
  color: var(--ptp-text);
  box-shadow: var(--ptp-highlight);
  transition: border-color .15s ease, background-color .15s ease, transform .15s ease;
}
[data-testid="stBaseButton-secondary"]:hover, [data-testid="stDownloadButton"] button:hover,
[data-testid="stLinkButton"] a:hover {
  border-color: rgba(245,179,1,0.55); background: var(--ptp-surface-3); color: #fff;
}
[data-testid="stBaseButton-secondary"] p, [data-testid="stLinkButton"] a p { font-size: 0.86rem; font-weight: 500; }
[data-testid="stBaseButton-primary"] {
  background: var(--ptp-accent); border: 1px solid var(--ptp-accent); color: #14100a; border-radius: 9px;
}
[data-testid="stBaseButton-primary"]:hover { background: #ffc21f; border-color: #ffc21f; color: #14100a; }
[data-testid="stBaseButton-primary"] p, [data-testid="stBaseButton-primary"] [data-testid="stIconMaterial"] { color: #14100a !important; font-weight: 600; }
[data-testid="stPageLink-NavLink"] {
  background: var(--ptp-surface-2);
  border: 1px solid var(--ptp-border-strong);
  border-radius: 9px;
  padding: 0.4rem 0.9rem;
  box-shadow: var(--ptp-highlight);
  transition: border-color .15s ease, background-color .15s ease;
}
[data-testid="stPageLink-NavLink"]:hover { border-color: rgba(245,179,1,0.55); background: var(--ptp-surface-3); }
[data-testid="stPageLink-NavLink"] p { font-size: 0.86rem; font-weight: 500; color: var(--ptp-text) !important; }
[data-testid="stPageLink-NavLink"] [data-testid="stIconMaterial"] { color: var(--ptp-muted); }

/* ── Data frames and alerts ───────────────────────────────────────────── */
[data-testid="stDataFrame"], [data-testid="stTable"] { border-radius: var(--ptp-radius); }
[data-testid="stAlertContainer"] {
  border-radius: var(--ptp-radius);
  border: 1px solid var(--ptp-border);
  background: var(--ptp-surface-1);
  padding: 0.7rem 0.95rem;
}
[data-testid="stAlertContainer"] p { font-size: 0.86rem; line-height: 1.55; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) {
  background: rgba(88,166,255,0.07); border-color: rgba(88,166,255,0.24); }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) p { color: #b8d6ff; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {
  background: rgba(210,153,34,0.08); border-color: rgba(210,153,34,0.28); }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) p { color: #ecc871; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) {
  background: rgba(248,81,73,0.08); border-color: rgba(248,81,73,0.28); }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) p { color: #ffb1ab; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {
  background: rgba(63,185,80,0.08); border-color: rgba(63,185,80,0.28); }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) p { color: #9be7a7; }

/* ── Metrics ──────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
  background: var(--ptp-surface-1);
  border: 1px solid var(--ptp-border);
  border-radius: 12px;
  padding: 0.8rem 1rem;
  box-shadow: var(--ptp-highlight);
}
[data-testid="stMetricLabel"] p {
  font-size: 0.68rem !important; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.08em; color: var(--ptp-muted) !important;
}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] p {
  font-family: var(--ptp-mono); font-size: 1.35rem !important; font-weight: 600;
  letter-spacing: -0.02em; color: var(--ptp-text);
}

/* ── Charts: every Plotly figure sits on a panel ─────────────────────── */
[data-testid="stPlotlyChart"] {
  background: var(--ptp-surface-1);
  border: 1px solid var(--ptp-border);
  border-radius: 12px;
  box-shadow: var(--ptp-highlight);
  overflow: hidden;
}
.js-plotly-plot .plotly .main-svg:first-of-type { background: transparent !important; }
.js-plotly-plot .plotly .modebar { background: transparent !important; }
.js-plotly-plot .plotly .modebar-btn path { fill: var(--ptp-faint) !important; }
.js-plotly-plot .plotly .modebar-btn:hover path, .js-plotly-plot .plotly .modebar-btn.active path { fill: var(--ptp-accent) !important; }

/* ── Inputs ───────────────────────────────────────────────────────────── */
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="base-input"] { border-radius: 9px !important; }
[data-baseweb="tag"] {
  background: rgba(245,179,1,0.12) !important; border: 1px solid rgba(245,179,1,0.35);
  border-radius: 7px !important;
}
[data-baseweb="tag"] span, [data-baseweb="tag"] [role="presentation"] { color: #f7c948 !important; }
[data-baseweb="tag"] svg { color: rgba(247,201,72,0.75); }
[data-testid="stWidgetLabel"] p { color: var(--ptp-text-2); font-size: 0.84rem; }

/* ══ Components ══════════════════════════════════════════════════════════ */
.ptp-mono { font-family: var(--ptp-mono); font-variant-numeric: tabular-nums; }
.ptp-up { color: var(--ptp-up); } .ptp-down { color: var(--ptp-down); }
.ptp-warn { color: var(--ptp-warn); } .ptp-info { color: var(--ptp-info); }
.ptp-muted { color: var(--ptp-muted); } .ptp-faint { color: var(--ptp-faint); }

/* Page header */
.ptp-page-header { margin: 0 0 1.35rem; }
.ptp-kicker {
  display: inline-flex; align-items: center; gap: 0.55rem;
  font-size: 0.7rem; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--ph-accent, var(--ptp-accent));
}
.ptp-kicker::before {
  content: ""; width: 18px; height: 2px; border-radius: 2px;
  background: var(--ph-accent, var(--ptp-accent));
  box-shadow: 0 0 12px var(--ph-accent, var(--ptp-accent));
}
.ptp-page-header__title {
  margin: 0.45rem 0 0; font-size: 2rem; line-height: 1.15; font-weight: 650;
  letter-spacing: -0.028em; color: var(--ptp-text);
}
.ptp-page-header__subtitle {
  margin-top: 0.5rem; max-width: 78ch; font-size: 0.95rem; line-height: 1.6; color: var(--ptp-muted);
}
.ptp-page-header__sources { margin-top: 0.8rem; display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem; }
.ptp-page-header__sources-label {
  font-size: 0.66rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--ptp-faint); margin-right: 0.15rem;
}
@media (max-width: 640px) { .ptp-page-header__title { font-size: 1.55rem; } }

/* Section header */
.ptp-section-header { margin: 1.4rem 0 0.75rem; }
.ptp-section-header__title {
  font-size: 1.02rem; font-weight: 600; letter-spacing: -0.01em; color: var(--ptp-text);
  display: flex; align-items: center; gap: 0.6rem;
}
.ptp-section-header__title::after { content: ""; flex: 1; height: 1px; background: var(--ptp-border); }
.ptp-section-header__caption { margin-top: 0.3rem; font-size: 0.84rem; color: var(--ptp-muted); max-width: 90ch; }

/* Chips */
.ptp-chip {
  display: inline-flex; align-items: center; gap: 0.4rem; white-space: nowrap;
  font-size: 0.74rem; font-weight: 500; line-height: 1.2; padding: 0.24rem 0.6rem;
  color: var(--chip-fg); background: var(--chip-bg); border: 1px solid var(--chip-bd);
  border-radius: 999px;
}
.ptp-chip__dot { width: 6px; height: 6px; border-radius: 50%; background: var(--chip-dot); box-shadow: 0 0 8px var(--chip-dot); }

/* KPI card (also styles the legacy .kpi-card markup) */
.ptp-kpi, .kpi-card {
  position: relative; overflow: hidden;
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0) 45%), var(--ptp-surface-1);
  border: 1px solid var(--ptp-border);
  border-radius: 12px;
  padding: 0.85rem 1rem 0.8rem;
  min-height: 6.6rem;
  margin-bottom: 0.25rem;
  box-shadow: var(--ptp-highlight), var(--ptp-shadow);
}
.ptp-kpi::before {
  content: ""; position: absolute; left: 0; top: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, var(--kpi-accent, var(--ptp-page-accent, transparent)), transparent 85%);
  opacity: 0.85;
}
.ptp-kpi__label, .kpi-label {
  font-size: 0.68rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--ptp-muted); line-height: 1.35;
}
.ptp-kpi__body { display: flex; align-items: flex-end; justify-content: space-between; gap: 0.75rem; margin-top: 0.4rem; }
.ptp-kpi__value, .kpi-value {
  font-family: var(--ptp-mono); font-variant-numeric: tabular-nums;
  font-size: 1.42rem; font-weight: 600; letter-spacing: -0.03em; line-height: 1.15;
  color: var(--ptp-text); white-space: nowrap;
}
.ptp-kpi__spark { flex: 0 0 auto; opacity: 0.95; }
.ptp-kpi__delta, .kpi-delta { font-size: 0.75rem; line-height: 1.4; margin-top: 0.35rem; color: var(--ptp-muted); }
.delta-red   { color: #ff7b72; }
.delta-amber { color: #e3b341; }
.delta-green { color: #56d364; }
.delta-blue  { color: #79c0ff; }
@media (max-width: 640px) { .ptp-kpi__value, .kpi-value { font-size: 1.25rem; } .ptp-kpi, .kpi-card { min-height: 0; } }

/* Status pills (legacy pill() markup) */
.pill-critical, .pill-warn, .pill-ok, .pill-good {
  display: inline-block; border-radius: 6px; padding: 0.16rem 0.5rem;
  font-size: 0.66rem; font-weight: 700; letter-spacing: 0.08em; vertical-align: 0.08em;
}
.pill-critical { color: #ff7b72; background: rgba(248,81,73,0.12); border: 1px solid rgba(248,81,73,0.38); }
.pill-warn     { color: #e3b341; background: rgba(210,153,34,0.12); border: 1px solid rgba(210,153,34,0.38); }
.pill-ok       { color: #79c0ff; background: rgba(88,166,255,0.12); border: 1px solid rgba(88,166,255,0.38); }
.pill-good     { color: #56d364; background: rgba(63,185,80,0.12); border: 1px solid rgba(63,185,80,0.38); }

/* Callout (also styles the legacy .commentary markup) */
.ptp-callout, .commentary {
  position: relative;
  background: var(--co-bg, var(--ptp-surface-1));
  border: 1px solid var(--co-bd, var(--ptp-border));
  border-left: 3px solid var(--co-accent, var(--ptp-info));
  border-radius: 10px;
  padding: 0.75rem 1rem;
  margin: 0.4rem 0 1rem;
  color: var(--ptp-text-2);
  font-size: 0.88rem; line-height: 1.62;
}
.ptp-callout strong, .commentary strong { color: var(--ptp-text); }
.ptp-callout__title { font-weight: 600; color: var(--ptp-text); margin-bottom: 0.2rem; }
.commentary-critical { --co-accent: var(--ptp-down); }
.commentary-warn     { --co-accent: var(--ptp-warn); }
.commentary-good     { --co-accent: var(--ptp-up); }

/* Ticker strip */
.ptp-ticker {
  display: flex; flex-wrap: wrap; gap: 1px;
  background: var(--ptp-border); border: 1px solid var(--ptp-border);
  border-radius: 12px; overflow: hidden; box-shadow: var(--ptp-shadow);
}
.ptp-ticker__item { flex: 1 1 150px; min-width: 0; background: var(--ptp-surface-1); padding: 0.7rem 0.9rem 0.65rem; }
.ptp-ticker__head { display: flex; align-items: baseline; justify-content: space-between; gap: 0.5rem; }
.ptp-ticker__sym { font-family: var(--ptp-mono); font-size: 0.74rem; font-weight: 600; color: var(--ptp-text); letter-spacing: 0.02em; }
.ptp-ticker__sub { font-size: 0.66rem; color: var(--ptp-faint); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.ptp-ticker__row { display: flex; align-items: flex-end; justify-content: space-between; gap: 0.5rem; margin-top: 0.3rem; }
.ptp-ticker__px { font-family: var(--ptp-mono); font-variant-numeric: tabular-nums; font-size: 1.12rem; font-weight: 600; letter-spacing: -0.02em; color: var(--ptp-text); line-height: 1.2; }
.ptp-ticker__unit { font-size: 0.64rem; color: var(--ptp-faint); margin-top: 0.1rem; }
.ptp-ticker__chg { font-family: var(--ptp-mono); font-size: 0.74rem; font-weight: 500; margin-top: 0.15rem; white-space: nowrap; }
.ptp-ticker__na { color: var(--ptp-faint); }
.ptp-ticker-note { margin-top: 0.45rem; font-size: 0.74rem; color: var(--ptp-muted); }

/* Layer cards (landing page) */
.ptp-layer-card { display: flex; flex-direction: column; gap: 0.15rem; }
.ptp-layer-card__head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.65rem; }
.ptp-layer-card__icon {
  width: 38px; height: 38px; border-radius: 10px; display: grid; place-items: center;
  color: var(--lc-accent); background: var(--lc-tint); border: 1px solid var(--lc-border);
}
.ptp-layer-card__icon svg { width: 20px; height: 20px; }
.ptp-layer-card__stat {
  font-family: var(--ptp-mono); font-size: 0.72rem; color: var(--ptp-muted);
  background: rgba(148,163,184,0.06); border: 1px solid var(--ptp-border-soft);
  border-radius: 999px; padding: 0.18rem 0.55rem; white-space: nowrap;
}
.ptp-layer-card__kicker { font-size: 0.66rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: var(--lc-accent); }
.ptp-layer-card__title { font-size: 1.05rem; font-weight: 600; letter-spacing: -0.01em; color: var(--ptp-text); margin-top: 0.1rem; }
.ptp-layer-card__desc { font-size: 0.85rem; line-height: 1.6; color: var(--ptp-muted); margin-top: 0.35rem; }

/* Footer */
.ptp-footer {
  margin-top: 2.5rem; padding-top: 1.1rem; border-top: 1px solid var(--ptp-border);
  font-size: 0.74rem; line-height: 1.75; color: var(--ptp-faint);
}
.ptp-footer a { color: var(--ptp-muted) !important; text-decoration: none; border-bottom: 1px dotted rgba(148,163,184,0.35); }
.ptp-footer a:hover { color: var(--ptp-text) !important; border-bottom-color: var(--ptp-text); }
.ptp-footer__sources { display: flex; flex-wrap: wrap; gap: 0.15rem 0.9rem; }
.ptp-footer__label { color: var(--ptp-faint); }
.ptp-footer__brand { color: var(--ptp-muted); font-weight: 500; }
.ptp-footer__note { margin-top: 0.3rem; }
.ptp-footer__disclaimer { margin-top: 0.3rem; color: var(--ptp-muted); }

/* ══ Landing page ════════════════════════════════════════════════════════ */
.st-key-ptp_hero {
  position: relative; overflow: hidden;
  background:
    radial-gradient(720px 320px at 0% 0%, rgba(245,179,1,0.085), transparent 70%),
    radial-gradient(640px 320px at 100% 100%, rgba(56,189,248,0.065), transparent 70%),
    linear-gradient(180deg, var(--ptp-surface-1), rgba(15,20,28,0.55));
  border: 1px solid var(--ptp-border);
  border-radius: 18px;
  padding: 2.2rem 2.3rem 2rem;
  box-shadow: var(--ptp-shadow), var(--ptp-highlight);
}
.ptp-hero__title {
  margin-top: 0.8rem; font-size: 2.55rem; line-height: 1.08; font-weight: 700;
  letter-spacing: -0.035em; color: var(--ptp-text);
}
.ptp-hero__title span { display: block; color: var(--ptp-muted); font-weight: 600; }
.ptp-hero__lead { margin-top: 1.05rem; font-size: 1.06rem; line-height: 1.6; color: var(--ptp-text-2); max-width: 62ch; }
.ptp-hero__body { margin-top: 0.45rem; font-size: 0.9rem; line-height: 1.65; color: var(--ptp-muted); max-width: 74ch; }
.st-key-ptp_cta { margin-top: 0.85rem; }
.st-key-ptp_cta_primary [data-testid="stPageLink-NavLink"] {
  background: var(--ptp-accent); border-color: var(--ptp-accent);
  box-shadow: 0 6px 22px -10px rgba(245,179,1,0.7);
}
.st-key-ptp_cta_primary [data-testid="stPageLink-NavLink"]:hover { background: #ffc21f; border-color: #ffc21f; }
.st-key-ptp_cta_primary [data-testid="stPageLink-NavLink"] p { color: #14100a !important; font-weight: 600; }
.st-key-ptp_cta_primary [data-testid="stPageLink-NavLink"] [data-testid="stIconMaterial"] { color: #14100a; }
.ptp-hero-stats {
  display: grid; grid-template-columns: 1fr 1fr; gap: 1px;
  background: var(--ptp-border); border: 1px solid var(--ptp-border);
  border-radius: 14px; overflow: hidden;
}
.ptp-hero-stat { background: rgba(10,14,20,0.6); padding: 1.05rem 1.15rem 1rem; }
.ptp-hero-stat__value {
  font-family: var(--ptp-mono); font-size: 1.95rem; font-weight: 600; line-height: 1.1;
  letter-spacing: -0.04em; color: var(--ptp-text);
}
.ptp-hero-stat__label {
  margin-top: 0.35rem; font-size: 0.66rem; font-weight: 600; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--ptp-muted);
}
.ptp-strip-head {
  display: flex; align-items: baseline; justify-content: space-between; flex-wrap: wrap;
  gap: 0.25rem 1rem; margin: 0.9rem 0 0;
}
.ptp-strip-head__title { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase; color: var(--ptp-muted); }
.ptp-strip-head__meta { font-size: 0.74rem; color: var(--ptp-muted); }
.ptp-status-row { display: flex; flex-wrap: wrap; align-items: center; gap: 0.45rem; margin-top: 0.4rem; }
.ptp-status-row__label {
  font-size: 0.66rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--ptp-faint); margin-right: 0.3rem;
}
[class*="st-key-ptp_layer_"] {
  position: relative; height: 100%;
  background: linear-gradient(180deg, rgba(255,255,255,0.022), rgba(255,255,255,0) 42%), var(--ptp-surface-1);
  border: 1px solid var(--ptp-border);
  border-radius: 14px;
  padding: 1.1rem 1.15rem 0.85rem;
  box-shadow: var(--ptp-highlight);
  transition: border-color .2s ease, transform .2s ease, box-shadow .2s ease;
}
[class*="st-key-ptp_layer_"]:hover {
  border-color: rgba(148,163,184,0.30); transform: translateY(-2px);
  box-shadow: var(--ptp-shadow), var(--ptp-highlight);
}
[data-testid="stLayoutWrapper"]:has(> [class*="st-key-ptp_layer_"]) { height: 100%; }
[class*="st-key-ptp_layer_"] > [data-testid="stElementContainer"]:has([data-testid="stPageLink"]) { margin-top: auto; }
[class*="st-key-ptp_layer_"] [data-testid="stPageLink-NavLink"] {
  background: transparent; border: none; box-shadow: none; padding: 0.15rem 0;
}
[class*="st-key-ptp_layer_"] [data-testid="stPageLink-NavLink"]::after {
  content: ""; position: absolute; inset: 0; border-radius: 14px;
}
[class*="st-key-ptp_layer_"] [data-testid="stPageLink-NavLink"] p { color: var(--ptp-text-2) !important; font-size: 0.82rem; }
[class*="st-key-ptp_layer_"]:hover [data-testid="stPageLink-NavLink"] p,
[class*="st-key-ptp_layer_"]:hover [data-testid="stPageLink-NavLink"] [data-testid="stIconMaterial"] { color: var(--ptp-accent) !important; }
.ptp-about { color: var(--ptp-text-2); font-size: 0.9rem; line-height: 1.85; max-width: 820px; }
.ptp-about-stack { color: var(--ptp-muted); font-size: 0.82rem; line-height: 1.9; margin-top: 0.5rem; }
.ptp-about-stack__title {
  color: var(--ptp-text-2); font-size: 0.68rem; font-weight: 600; letter-spacing: 0.12em;
  text-transform: uppercase; margin-bottom: 0.15rem;
}
@media (max-width: 640px) {
  .st-key-ptp_hero { padding: 1.4rem 1.15rem 1.25rem; border-radius: 14px; }
  .ptp-hero__title { font-size: 1.85rem; }
  .ptp-hero__lead { font-size: 0.98rem; }
  .ptp-hero-stat__value { font-size: 1.55rem; }
}
"""


_RUN_KEY = "_ptp_run_token"
_THEME_KEY = "_ptp_theme_token"


def begin_run() -> None:
    """Mark the start of a script run. Called once by the router before rendering."""
    st.session_state[_RUN_KEY] = st.session_state.get(_RUN_KEY, 0) + 1




def _minify(css: str) -> str:
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    return re.sub(r"\s+", " ", css).strip()


_STYLESHEET: str | None = None


def stylesheet() -> str:
    """Return the full global stylesheet (fonts, tokens and component rules), minified."""
    global _STYLESHEET
    if _STYLESHEET is None:
        _STYLESHEET = _minify(f"@import url('{GOOGLE_FONTS_URL}');\n{_root_variables()}\n{_CSS}")
    return _STYLESHEET


def inject_theme() -> None:
    """Inject the global stylesheet once per script run.

    Style-only ``st.html`` content is routed to Streamlit's event container, so the
    stylesheet occupies no layout space. When the router has marked the run (see
    ``begin_run``), repeated calls from page modules within the same run are no-ops.
    """
    run = st.session_state.get(_RUN_KEY)
    if run is not None and st.session_state.get(_THEME_KEY) == run:
        return
    st.html(f"<style>{stylesheet()}</style>")
    st.session_state[_THEME_KEY] = run


# ══════════════════════════════════════════════════════════════════════════════
# Brand
# ══════════════════════════════════════════════════════════════════════════════

_LOGO_MARK = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'>"
    "<defs><linearGradient id='g' x1='0' y1='0' x2='0' y2='1'>"
    "<stop offset='0' stop-color='#ffd257'/><stop offset='1' stop-color='#f59e0b'/></linearGradient></defs>"
    "<rect x='0.5' y='0.5' width='31' height='31' rx='8' fill='#121924' stroke='rgba(245,179,1,0.55)'/>"
    "<path d='M18.2 5.5 9.6 17.6h5.6l-1.6 8.9 8.8-12.4h-5.7z' fill='url(#g)'/>"
    "</svg>"
)


def logo_svg() -> str:
    """Square brand mark (amber bolt on a dark tile), usable as an inline SVG or image."""
    return _LOGO_MARK


def render_logo() -> None:
    """Show the brand mark in the header and sidebar via ``st.logo``."""
    st.logo(_LOGO_MARK, size="large")


# ══════════════════════════════════════════════════════════════════════════════
# Inline components (return HTML strings)
# ══════════════════════════════════════════════════════════════════════════════

def render(html_str: str) -> None:
    """Render an HTML string produced by this module."""
    st.markdown(html_str, unsafe_allow_html=True)


def chip(text: str, tone: str = "neutral", dot: bool = False, title: str | None = None) -> str:
    """Small rounded label. ``tone`` is a TONES key or a hex colour."""
    color = _tone(tone)
    fg = COLORS["text_2"] if tone == "neutral" else color
    style = (
        f"--chip-fg:{fg};--chip-bg:{_rgba(color, 0.10) if tone != 'neutral' else 'rgba(148,163,184,0.06)'};"
        f"--chip-bd:{_rgba(color, 0.32) if tone != 'neutral' else COLORS['border']};--chip-dot:{color};"
    )
    dot_html = '<span class="ptp-chip__dot"></span>' if dot else ""
    title_attr = f' title="{_attr(title)}"' if title else ""
    return f'<span class="ptp-chip" style="{style}"{title_attr}>{dot_html}{_esc(text)}</span>'


def sparkline(values: Sequence[float] | None, color: str | None = None,
              width: int = 88, height: int = 28, fill: bool = True) -> str:
    """Inline SVG sparkline. Non-finite values are skipped; fewer than two points yield ''."""
    pts = [float(v) for v in (values or []) if v is not None and math.isfinite(float(v))]
    if len(pts) < 2:
        return ""
    if color is None:
        color = COLORS["up"] if pts[-1] >= pts[0] else COLORS["down"]
    lo, hi = min(pts), max(pts)
    span = (hi - lo) or 1.0
    pad = 2.0
    step = (width - 2 * pad) / (len(pts) - 1)
    coords = [
        (pad + i * step, pad + (height - 2 * pad) * (1 - (v - lo) / span))
        for i, v in enumerate(pts)
    ]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
    area = ""
    if fill:
        area_pts = f"{coords[0][0]:.1f},{height - pad:.1f} {line} {coords[-1][0]:.1f},{height - pad:.1f}"
        area = f'<polygon points="{area_pts}" fill="{_rgba(color, 0.12)}" stroke="none"/>'
    end_x, end_y = coords[-1]
    return (
        f'<svg class="ptp-spark" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'aria-hidden="true" focusable="false">{area}'
        f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="1.5" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
        f'<circle cx="{end_x:.1f}" cy="{end_y:.1f}" r="2" fill="{color}"/></svg>'
    )


def kpi_card(label: str, value: str, delta_html: str = "", accent: str | None = None,
             spark: Sequence[float] | None = None, spark_color: str | None = None) -> str:
    """KPI tile: small-caps label, monospaced value, optional delta line and sparkline.

    ``delta_html`` is inserted as HTML (typically from ``delta_span``). ``accent`` draws
    a thin top rule in that colour. ``spark`` is an optional sequence of numbers.
    """
    style = f' style="--kpi-accent:{_attr(accent)}"' if accent else ""
    spark_html = sparkline(spark, color=spark_color or accent) if spark else ""
    spark_block = f'<div class="ptp-kpi__spark">{spark_html}</div>' if spark_html else ""
    delta_block = f'<div class="ptp-kpi__delta">{delta_html}</div>' if delta_html else '<div class="ptp-kpi__delta">&nbsp;</div>'
    return (
        f'<div class="ptp-kpi"{style}>'
        f'<div class="ptp-kpi__label">{_esc(label)}</div>'
        f'<div class="ptp-kpi__body"><div class="ptp-kpi__value">{_esc(value)}</div>{spark_block}</div>'
        f"{delta_block}</div>"
    )


_CALLOUT_TONES = {
    "info": COLORS["info"], "ok": COLORS["info"], "neutral": COLORS["muted"],
    "warn": COLORS["warn"], "critical": COLORS["down"], "down": COLORS["down"],
    "good": COLORS["up"], "up": COLORS["up"], "accent": COLORS["accent"],
}


def callout(text: str, tone: str = "info", title: str | None = None) -> str:
    """Bordered note with a coloured left rule. ``text`` may contain inline HTML."""
    color = _CALLOUT_TONES.get(tone, _tone(tone))
    style = (
        f"--co-accent:{color};--co-bg:{_rgba(color, 0.055)};--co-bd:{_rgba(color, 0.20)};"
    )
    title_html = f'<div class="ptp-callout__title">{_esc(title)}</div>' if title else ""
    return f'<div class="ptp-callout" style="{style}">{title_html}<div>{text}</div></div>'


def ticker_strip(items: Iterable[dict], note: str | None = None) -> str:
    """Row of market quotes.

    Each item is a dict with keys: ``symbol``, ``sub`` (descriptor), ``value`` (float or
    None), ``decimals``, ``unit``, ``change`` (float or None), ``change_mode`` ('pct' or
    'abs'), optional ``spark`` (sequence of floats) and optional ``title`` (tooltip).
    Missing values render as a middle dot.
    """
    cells = []
    for it in items:
        value = it.get("value")
        dec = int(it.get("decimals", 2))
        has_value = isinstance(value, (int, float)) and math.isfinite(value)
        px = f"{value:,.{dec}f}" if has_value else "·"
        chg = it.get("change")
        if has_value and isinstance(chg, (int, float)) and math.isfinite(chg):
            arrow = "▲" if chg > 0 else ("▼" if chg < 0 else "■")
            cls = "ptp-up" if chg > 0 else ("ptp-down" if chg < 0 else "ptp-muted")
            if it.get("change_mode") == "abs":
                chg_txt = f"{arrow} {chg:+,.2f}"
            else:
                chg_txt = f"{arrow} {chg:+.2f}%"
            chg_html = f'<div class="ptp-ticker__chg {cls}">{chg_txt}</div>'
        else:
            chg_html = '<div class="ptp-ticker__chg ptp-ticker__na">·</div>'
        spark_html = ""
        if has_value and it.get("spark"):
            spark_html = sparkline(it["spark"], width=56, height=22, fill=False)
        title_attr = f' title="{_attr(it["title"])}"' if it.get("title") else ""
        px_cls = "ptp-ticker__px" if has_value else "ptp-ticker__px ptp-ticker__na"
        cells.append(
            f'<div class="ptp-ticker__item"{title_attr}>'
            f'<div class="ptp-ticker__head"><span class="ptp-ticker__sym">{_esc(it.get("symbol", ""))}</span>'
            f'<span class="ptp-ticker__sub">{_esc(it.get("sub", ""))}</span></div>'
            f'<div class="ptp-ticker__row"><div><div class="{px_cls}">{px}</div>'
            f'<div class="ptp-ticker__unit">{_esc(it.get("unit", "")) or "&nbsp;"}</div></div>'
            f'<div style="text-align:right">{spark_html}{chg_html}</div></div>'
            f"</div>"
        )
    note_html = f'<div class="ptp-ticker-note">{_esc(note)}</div>' if note else ""
    return f'<div class="ptp-ticker">{"".join(cells)}</div>{note_html}'


ICONS: dict[str, str] = {
    "pulse":   '<path d="M3 12h4l2.5-6.5 4 13 2.6-6.5H21"/>',
    "chart":   '<path d="M4 4v16h16"/><path d="m7 14 3.5-3.5 3 3L19 8"/><circle cx="19" cy="8" r="1.2" fill="currentColor"/>',
    "globe":   '<circle cx="12" cy="12" r="8.5"/><path d="M3.5 12h17"/><path d="M12 3.5c2.6 2.4 3.8 5.2 3.8 8.5s-1.2 6.1-3.8 8.5c-2.6-2.4-3.8-5.2-3.8-8.5s1.2-6.1 3.8-8.5z"/>',
    "cpu":     '<rect x="7" y="7" width="10" height="10" rx="2"/><path d="M10 10h4v4h-4z"/><path d="M9.5 3.5v3M14.5 3.5v3M9.5 17.5v3M14.5 17.5v3M3.5 9.5h3M3.5 14.5h3M17.5 9.5h3M17.5 14.5h3"/>',
    "target":  '<circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="4.5"/><circle cx="12" cy="12" r="1" fill="currentColor"/>',
    "shield":  '<path d="M12 3.2 19 6v5.2c0 4.4-2.9 8.2-7 9.6-4.1-1.4-7-5.2-7-9.6V6z"/><path d="m9 12 2.1 2.1L15.3 10"/>',
    "battery": '<rect x="2.8" y="7" width="16.4" height="10" rx="2.2"/><path d="M21.2 10.5v3"/><path d="m11.6 9-2.1 3.2h3l-2.1 3.3"/>',
    "orbit":   '<circle cx="12" cy="12" r="6"/><path d="M6.6 9.6c-3.3 1.6-4.9 3.4-4.2 4.6 1 1.7 6.1.8 11.5-2.3s9.1-6.8 8.1-8.5c-.7-1.2-3.1-1-6.3.2"/><circle cx="19.2" cy="5.6" r="1" fill="currentColor"/>',
}


def icon_svg(name: str, size: int = 20) -> str:
    """Line icon from ICONS as an inline SVG using ``currentColor``."""
    body = ICONS.get(name, ICONS["pulse"])
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        f'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{body}</svg>'
    )


def layer_card(kicker: str, title: str, description: str, stat: str | None = None,
               icon: str = "pulse", accent: str | None = None) -> str:
    """Landing-page card body: icon, key stat, kicker, title and a short description."""
    color = accent or COLORS["accent"]
    style = f"--lc-accent:{color};--lc-tint:{_rgba(color, 0.10)};--lc-border:{_rgba(color, 0.28)};"
    stat_html = f'<span class="ptp-layer-card__stat">{_esc(stat)}</span>' if stat else ""
    return (
        f'<div class="ptp-layer-card" style="{style}">'
        f'<div class="ptp-layer-card__head"><div class="ptp-layer-card__icon">{icon_svg(icon)}</div>{stat_html}</div>'
        f'<div class="ptp-layer-card__kicker">{_esc(kicker)}</div>'
        f'<div class="ptp-layer-card__title">{_esc(title)}</div>'
        f'<div class="ptp-layer-card__desc">{_esc(description)}</div>'
        f"</div>"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Page structure (render directly)
# ══════════════════════════════════════════════════════════════════════════════

def page_header(kicker: str, title: str, subtitle: str | None = None,
                sources: Sequence[str] | None = None, accent: str | None = None) -> str:
    """Render the standard page header and return its HTML.

    ``kicker`` is the small-caps line above the title (for example 'Layer 1'),
    ``sources`` an optional list of data-source names shown as chips, ``accent`` a
    colour (typically from ``ACCENTS``).
    """
    style = f' style="--ph-accent:{_attr(accent)}"' if accent else ""
    if accent:
        # Page-level accent picked up by KPI cards on the same page.
        st.html(f"<style>:root{{--ptp-page-accent:{_attr(accent)};}}</style>")
    sub_html = f'<div class="ptp-page-header__subtitle">{_esc(subtitle)}</div>' if subtitle else ""
    src_html = ""
    if sources:
        chips = "".join(chip(s) for s in sources)
        src_html = (
            '<div class="ptp-page-header__sources">'
            f'<span class="ptp-page-header__sources-label">Sources</span>{chips}</div>'
        )
    out = (
        f'<div class="ptp-page-header"{style}>'
        f'<div class="ptp-kicker">{_esc(kicker)}</div>'
        f'<div class="ptp-page-header__title" role="heading" aria-level="1">{_esc(title)}</div>'
        f"{sub_html}{src_html}</div>"
    )
    render(out)
    return out


def section_header(title: str, caption: str | None = None) -> str:
    """Render a section title with a hairline rule and optional caption; return its HTML."""
    cap = f'<div class="ptp-section-header__caption">{_esc(caption)}</div>' if caption else ""
    out = (
        f'<div class="ptp-section-header"><div class="ptp-section-header__title" role="heading" '
        f'aria-level="2">{_esc(title)}</div>{cap}</div>'
    )
    render(out)
    return out


def page_footer(sources: Sequence | None = None, note: str | None = None) -> str:
    """Render the standard footer and return its HTML.

    ``sources`` items are ``(label, name, url)`` / ``(label, name)`` tuples or plain
    strings. ``note`` is an optional page-specific line shown above the standard
    disclaimer ('For informational purposes only. Not financial advice.').
    """
    parts = [f'<span class="ptp-footer__brand">Built by {_esc(AUTHOR)}</span>']
    for src in sources or []:
        if isinstance(src, str):
            parts.append(f"<span>{_esc(src)}</span>")
            continue
        label, name = src[0], src[1]
        url = src[2] if len(src) > 2 else None
        name_html = (
            f'<a href="{_attr(url)}" target="_blank" rel="noopener noreferrer">{_esc(name)}</a>'
            if url else _esc(name)
        )
        label_html = f'<span class="ptp-footer__label">{_esc(label)}:</span> ' if label else ""
        parts.append(f"<span>{label_html}{name_html}</span>")
    note_html = f'<div class="ptp-footer__note">{_esc(note)}</div>' if note else ""
    out = (
        f'<div class="ptp-footer"><div class="ptp-footer__sources">{"".join(parts)}</div>'
        f'{note_html}<div class="ptp-footer__disclaimer">{_esc(STANDARD_DISCLAIMER)}</div></div>'
    )
    render(out)
    return out
