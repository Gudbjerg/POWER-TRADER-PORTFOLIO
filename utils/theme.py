"""
Design system for the European Gas & Power Market Intelligence Platform.

Visual concept: a survey sheet from a grid operator's drawing office. Pages sit on cool
engineering paper with a faint measuring grid; structure is drawn the way a single-line
diagram draws a power system: a busbar under every page title, meters instead of cards,
and nodes that mark where a layer taps into the bus. Two colours carry the subject: fjord
teal for the Nordic hydro system and cobalt for the Continental gas-fired system.

One module owns the visual language of the app:

* Design tokens (colours, typography, per-layer accents).
* ``inject_theme()``: the global stylesheet (fonts, app chrome, top navigation, tabs,
  expanders, buttons, page links, tables, alerts, metrics, sidebar).
* HTML components returned as strings for ``st.markdown(..., unsafe_allow_html=True)``:
  ``chip``, ``kpi_card``, ``sparkline``, ``callout``, ``ticker_strip``, ``interconnector``.
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
    # Surfaces
    "bg":            "#e7ece8",   # survey paper
    "surface_1":     "#f5f7f4",   # drawing sheet (panels, charts)
    "surface_2":     "#edf1ee",   # recessed areas (table heads, hover)
    "surface_3":     "#dfe6e1",
    # Lines
    "border":        "rgba(19,47,56,0.16)",
    "border_soft":   "rgba(19,47,56,0.09)",
    "border_strong": "rgba(19,47,56,0.30)",
    # Text
    "text":          "#132f38",   # petrol ink
    "text_2":        "#3a5560",
    "muted":         "#667b83",
    "faint":         "#93a3a8",
    # Brand
    "accent":        "#0e7a7e",   # fjord: Nordic hydro
    "accent_2":      "#2f45c6",   # cobalt: Continental gas
    # Semantic
    "up":            "#2f7a47",
    "down":          "#c2391f",
    "warn":          "#a86a12",
    "info":          "#2f45c6",
}

# Per-layer accents: the colour of the node where each page taps into the busbar.
ACCENTS: dict[str, str] = {
    "home":  COLORS["accent"],
    "l1":    "#0e7a7e",   # Live Monitor: fjord
    "l2":    "#2f45c6",   # Quant Analysis: cobalt
    "l3":    "#a86a12",   # Macro Signals: ochre
    "l4":    "#6e4a9e",   # ML Models: plum
    "l5":    "#c2391f",   # Mispricing Dashboard: signal red
    "risk":  "#8f5a1c",   # Risk Dashboard: umber
    "bess":  "#2f7a47",   # BESS Optimiser: moss
    "globe": "#1c8fb0",   # Energy Globe: sea
}

# Semantic tones shared by chips, callouts and status marks.
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

FONT_SANS = "'Archivo', 'Helvetica Neue', Arial, sans-serif"
FONT_DISPLAY = "'Big Shoulders Display', 'Archivo Narrow', 'Arial Narrow', sans-serif"
FONT_MONO = "'JetBrains Mono', 'SFMono-Regular', Menlo, Consolas, monospace"
GOOGLE_FONTS_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=Archivo:wght@400;500;600;700"
    "&family=Big+Shoulders+Display:wght@600;700;800"
    "&family=JetBrains+Mono:wght@400;500&display=swap"
)

RADII: dict[str, str] = {"sm": "2px", "md": "3px", "lg": "4px", "pill": "999px"}
SPACE: dict[str, str] = {"xs": "0.25rem", "sm": "0.5rem", "md": "1rem", "lg": "1.5rem", "xl": "2.5rem"}

# Categorical palette for charts that do not set explicit series colours.
COLORWAY: list[str] = [
    "#0e7a7e", "#2f45c6", "#a86a12", "#c2391f", "#6e4a9e",
    "#2f7a47", "#1c8fb0", "#c2611a", "#8f8424", "#667b83",
]

# Chart chrome. Every Plotly chart sits on a drawing-sheet panel (see the stPlotlyChart
# rule in the stylesheet): the paper is transparent and the plot area is the sheet.
CHART: dict[str, str] = {
    "bg":                "rgba(0,0,0,0)",
    "panel_bg":          COLORS["surface_1"],
    "grid":              "rgba(19,47,56,0.085)",
    "zero":              "rgba(19,47,56,0.40)",
    "axis":              "rgba(19,47,56,0.35)",
    "text":              COLORS["text_2"],
    "muted":             COLORS["muted"],
    "faint":             COLORS["faint"],
    "annotation_bg":     COLORS["surface_1"],
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
    """Register the 'ptp' template (plotly_white base + tokens) and make it the default."""
    template = go.layout.Template(pio.templates["plotly_white"])
    axis = dict(
        gridcolor=CHART["grid"],
        zerolinecolor=CHART["zero"],
        linecolor=CHART["axis"],
        tickcolor=CHART["axis"],
        tickfont=dict(color=CHART["muted"], size=11),
        title=dict(font=dict(color=CHART["muted"], size=11)),
        showline=True,
        linewidth=1,
        ticks="outside",
        ticklen=4,
        automargin=True,
        spikecolor="rgba(19,47,56,0.55)",
        spikethickness=1,
        spikedash="dot",
    )
    template.layout.update(
        font=dict(family=FONT_SANS, size=12, color=CHART["text"]),
        paper_bgcolor=CHART["bg"],
        plot_bgcolor=CHART["panel_bg"],
        colorway=COLORWAY,
        title=dict(font=dict(family=FONT_SANS, size=13, color=COLORS["text"]), x=0, xanchor="left"),
        xaxis=axis,
        yaxis=axis,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=CHART["text"])),
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor=COLORS["text"],
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
        "--ptp-display": FONT_DISPLAY,
        "--ptp-mono": FONT_MONO,
        "--ptp-radius-sm": RADII["sm"],
        "--ptp-radius": RADII["md"],
        "--ptp-radius-lg": RADII["lg"],
        **{f"--ptp-space-{k}": v for k, v in SPACE.items()},
    }
    return ":root{" + "".join(f"{k}:{v};" for k, v in tokens.items()) + "}"


_CSS = r"""
/* ── Base: survey paper with a measuring grid ─────────────────────────── */
html, body, .stApp { -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }
.stApp {
  background-color: var(--ptp-bg);
  background-image:
    linear-gradient(rgba(19,47,56,0.055) 1px, transparent 1px),
    linear-gradient(90deg, rgba(19,47,56,0.055) 1px, transparent 1px),
    linear-gradient(rgba(19,47,56,0.028) 1px, transparent 1px),
    linear-gradient(90deg, rgba(19,47,56,0.028) 1px, transparent 1px);
  background-size: 120px 120px, 120px 120px, 24px 24px, 24px 24px;
  background-position: -1px -1px;
  color: var(--ptp-text);
  font-family: var(--ptp-font);
  font-variant-numeric: tabular-nums;
}
::selection { background: rgba(14,122,126,0.22); color: var(--ptp-text); }
* { scrollbar-width: thin; scrollbar-color: rgba(19,47,56,0.25) transparent; }
a:focus-visible, button:focus-visible, [tabindex]:focus-visible, summary:focus-visible {
  outline: 2px solid var(--ptp-accent-2); outline-offset: 2px;
}
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation: none !important; transition: none !important; } }

/* ── Streamlit chrome ─────────────────────────────────────────────────── */
[data-testid="stAppDeployButton"], [data-testid="stDecoration"], footer { display: none !important; }
header[data-testid="stHeader"] {
  background: rgba(231,236,232,0.94);
  backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);
  border-bottom: 2px solid var(--ptp-text);
}
[data-testid="stMainMenuButton"], [data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapseButton"] button { color: var(--ptp-text-2); }
[data-testid="stStatusWidget"] { color: var(--ptp-muted); }
[data-testid="stHeaderLogo"], [data-testid="stSidebarLogo"] { height: 1.9rem; max-height: 1.9rem; }

/* Top navigation: plain ink links, the current page sits on a fjord rule */
[data-testid="stTopNavLink"], [data-testid="stTopNavSection"] {
  border-radius: 0; padding: 0.35rem 0.75rem; transition: color .12s ease, box-shadow .12s ease;
}
[data-testid="stTopNavLink"] p, [data-testid="stTopNavSection"] p { font-size: 0.9rem; font-weight: 500; color: var(--ptp-text-2); }
[data-testid="stTopNavLink"]:hover, [data-testid="stTopNavSection"]:hover { background: transparent; box-shadow: inset 0 -2px 0 var(--ptp-border-strong); }
[data-testid="stTopNavLink"]:hover p, [data-testid="stTopNavSection"]:hover p { color: var(--ptp-text); }
[data-testid="stTopNavSection"] svg { color: var(--ptp-muted); }
[data-testid="stTopNavLink"][aria-current="page"],
.rc-overflow-item:has([data-testid="stTopNavDropdownLink"][aria-current="page"]) [data-testid="stTopNavSection"] {
  background: transparent; box-shadow: inset 0 -3px 0 var(--ptp-accent);
}
[data-testid="stTopNavLink"][aria-current="page"] p,
.rc-overflow-item:has([data-testid="stTopNavDropdownLink"][aria-current="page"]) [data-testid="stTopNavSection"] p { color: var(--ptp-text); font-weight: 650; }
[data-testid="stTopNavPopover"] {
  background: var(--ptp-surface-1); border: 1px solid var(--ptp-text); border-radius: 2px;
  box-shadow: 4px 4px 0 rgba(19,47,56,0.12); padding: 0.25rem; min-width: 13.5rem;
}
[data-testid="stTopNavDropdownLink"] { border-radius: 2px; padding: 0.42rem 0.65rem; }
[data-testid="stTopNavDropdownLink"] p { font-size: 0.88rem; color: var(--ptp-text-2); }
[data-testid="stTopNavDropdownLink"]:hover { background: var(--ptp-surface-2); }
[data-testid="stTopNavDropdownLink"][aria-current="page"] { background: rgba(14,122,126,0.10); box-shadow: inset 3px 0 0 var(--ptp-accent); }
[data-testid="stTopNavDropdownLink"][aria-current="page"] p { color: var(--ptp-text); font-weight: 600; }

/* Sidebar: a margin strip of the same sheet */
section[data-testid="stSidebar"] { background: var(--ptp-surface-2); border-right: 2px solid var(--ptp-text); }
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-top: 0.5rem; }
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] h4 {
  font-family: var(--ptp-display) !important; font-size: 1.25rem !important; font-weight: 700;
  color: var(--ptp-text) !important; letter-spacing: 0.01em;
}
[data-testid="stSidebarNavLink"] { border-radius: 2px; }
[data-testid="stSidebarNavLink"][aria-current="page"] { background: rgba(14,122,126,0.10); box-shadow: inset 3px 0 0 var(--ptp-accent); }
[data-testid="stNavSectionHeader"] { color: var(--ptp-muted); font-size: 0.78rem; font-weight: 600; }

/* ── Layout ───────────────────────────────────────────────────────────── */
[data-testid="stMainBlockContainer"] { max-width: 1440px; padding: 5.25rem 2.75rem 3rem; }
@media (max-width: 1100px) { [data-testid="stMainBlockContainer"] { padding: 5rem 1.75rem 2.5rem; } }
@media (max-width: 640px)  { [data-testid="stMainBlockContainer"] { padding: 4.5rem 1rem 2rem; } }

/* ── Typography ───────────────────────────────────────────────────────── */
[data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2,
[data-testid="stHeading"] h1, [data-testid="stHeading"] h2 {
  font-family: var(--ptp-display); color: var(--ptp-text); font-weight: 750; letter-spacing: 0.005em; line-height: 1.02;
}
[data-testid="stMarkdownContainer"] h3, [data-testid="stMarkdownContainer"] h4,
[data-testid="stMarkdownContainer"] h5, [data-testid="stMarkdownContainer"] h6,
[data-testid="stHeading"] h3, [data-testid="stHeading"] h4 {
  font-family: var(--ptp-font); color: var(--ptp-text); letter-spacing: -0.01em;
}
[data-testid="stMarkdownContainer"] h1 { font-size: 2.6rem; }
[data-testid="stMarkdownContainer"] h2 { font-size: 2rem; }
[data-testid="stMarkdownContainer"] h3, [data-testid="stHeading"] h3 {
  font-size: 1.12rem; font-weight: 650; padding-top: 0.9rem; padding-bottom: 0.35rem;
}
[data-testid="stMarkdownContainer"] h4 { font-size: 1rem; font-weight: 650; padding-top: 0.6rem; }
[data-testid="stMarkdownContainer"] h5 { font-size: 0.9rem; font-weight: 600; color: var(--ptp-text-2); }
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { color: var(--ptp-text-2); line-height: 1.6; }
[data-testid="stMarkdownContainer"] strong { color: var(--ptp-text); font-weight: 650; }
[data-testid="stMarkdownContainer"] a { color: var(--ptp-accent-2); text-decoration: underline; text-decoration-color: rgba(47,69,198,0.35); text-underline-offset: 3px; }
[data-testid="stMarkdownContainer"] a:hover { text-decoration-color: var(--ptp-accent-2); }
[data-testid="stMarkdownContainer"] code {
  font-family: var(--ptp-mono); font-size: 0.82em; color: var(--ptp-text);
  background: var(--ptp-surface-2); border: 1px solid var(--ptp-border-soft); padding: 0.06em 0.34em; border-radius: 2px;
}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
  color: #4f6670 !important; font-size: 0.84rem !important; line-height: 1.55;
}
[data-testid="stMarkdownContainer"] hr, hr {
  border: none !important; border-top: 1px solid var(--ptp-border-strong) !important;
  margin: 1.2rem 0 !important; background: transparent !important; height: 0 !important;
}

/* Markdown tables: ruled like a schedule on a drawing */
[data-testid="stMarkdownContainer"] table {
  border-collapse: collapse; margin: 0.4rem 0 1rem; font-size: 0.86rem;
  background: var(--ptp-surface-1); border-top: 2px solid var(--ptp-text); border-bottom: 1px solid var(--ptp-text);
}
[data-testid="stMarkdownContainer"] thead th {
  background: transparent; color: var(--ptp-text); font-size: 0.8rem; font-weight: 650;
  text-align: left; padding: 0.5rem 0.8rem; border: none; border-bottom: 1px solid var(--ptp-text);
}
[data-testid="stMarkdownContainer"] tbody td {
  padding: 0.45rem 0.8rem; border: none; border-bottom: 1px solid var(--ptp-border-soft);
  color: var(--ptp-text-2); vertical-align: top;
}
[data-testid="stMarkdownContainer"] tbody tr:last-child td { border-bottom: none; }
@media (max-width: 640px) { [data-testid="stMarkdownContainer"] table { display: block; max-width: 100%; overflow-x: auto; } }

/* ── Tabs: drawing-register tabs on a rule ────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  gap: 0; padding: 0; flex-wrap: wrap; width: 100%;
  background: transparent; border: none; border-bottom: 2px solid var(--ptp-text); border-radius: 0;
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none !important; }
.stTabs button[data-baseweb="tab"] {
  height: auto; margin: 0 0 -2px; padding: 0.5rem 0.9rem 0.45rem;
  background: transparent; border: 1px solid transparent; border-bottom: none; border-radius: 2px 2px 0 0;
  white-space: nowrap; flex: 0 0 auto;
}
.stTabs button[data-baseweb="tab"] p { font-size: 0.88rem; font-weight: 500; color: var(--ptp-muted); }
.stTabs button[data-baseweb="tab"]:hover p { color: var(--ptp-text); }
.stTabs button[data-baseweb="tab"][aria-selected="true"] {
  background: var(--ptp-surface-1); border-color: var(--ptp-text); border-bottom: 2px solid var(--ptp-surface-1);
}
.stTabs button[data-baseweb="tab"][aria-selected="true"] p { color: var(--ptp-text); font-weight: 650; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.2rem; }
@media (max-width: 768px) {
  .stTabs [data-baseweb="tab-list"] { flex-wrap: nowrap; overflow-x: auto; scrollbar-width: none; }
  .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
}

/* ── Expanders: folded notes on the sheet ─────────────────────────────── */
[data-testid="stExpander"] details {
  background: var(--ptp-surface-1); border: 1px solid var(--ptp-border); border-left: 3px solid var(--ptp-text);
  border-radius: 2px; overflow: hidden;
}
[data-testid="stExpander"] summary { padding: 0.6rem 1rem; }
[data-testid="stExpander"] summary:hover { background: var(--ptp-surface-2); }
[data-testid="stExpander"] summary p { font-size: 0.92rem; font-weight: 600; color: var(--ptp-text); }
[data-testid="stExpander"] summary [data-testid="stIconMaterial"] { color: var(--ptp-muted); }
[data-testid="stExpander"] details[open] > summary { border-bottom: 1px solid var(--ptp-border-soft); }
[data-testid="stExpanderDetails"] { padding: 0.9rem 1.1rem 1rem; }

/* ── Buttons and page links ───────────────────────────────────────────── */
[data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-tertiary"],
[data-testid="stDownloadButton"] button, [data-testid="stLinkButton"] a, [data-testid="stPageLink-NavLink"] {
  background: var(--ptp-surface-1); border: 1px solid var(--ptp-text); border-radius: 2px;
  color: var(--ptp-text); box-shadow: 2px 2px 0 rgba(19,47,56,0.16);
  transition: transform .1s ease, box-shadow .1s ease;
}
[data-testid="stBaseButton-secondary"]:hover, [data-testid="stDownloadButton"] button:hover,
[data-testid="stLinkButton"] a:hover, [data-testid="stPageLink-NavLink"]:hover {
  background: var(--ptp-surface-1); border-color: var(--ptp-text); color: var(--ptp-text);
  transform: translate(-1px, -1px); box-shadow: 3px 3px 0 rgba(19,47,56,0.22);
}
[data-testid="stBaseButton-secondary"] p, [data-testid="stLinkButton"] a p { font-size: 0.88rem; font-weight: 600; }
[data-testid="stPageLink-NavLink"] { padding: 0.4rem 0.95rem; }
[data-testid="stPageLink-NavLink"] p { font-size: 0.88rem; font-weight: 600; color: var(--ptp-text) !important; }
[data-testid="stPageLink-NavLink"] [data-testid="stIconMaterial"] { color: var(--ptp-text-2); }
[data-testid="stBaseButton-primary"] {
  background: var(--ptp-accent); border: 1px solid var(--ptp-text); color: #fff; border-radius: 2px;
  box-shadow: 2px 2px 0 rgba(19,47,56,0.25);
}
[data-testid="stBaseButton-primary"]:hover { background: #0b6669; border-color: var(--ptp-text); color: #fff; }
[data-testid="stBaseButton-primary"] p, [data-testid="stBaseButton-primary"] [data-testid="stIconMaterial"] { color: #fff !important; font-weight: 600; }

/* ── Data frames and alerts ───────────────────────────────────────────── */
[data-testid="stDataFrame"], [data-testid="stTable"] { border-radius: 2px; }
[data-testid="stAlertContainer"] {
  border-radius: 2px; border: 1px solid var(--ptp-border); border-left: 4px solid var(--ptp-muted);
  background: var(--ptp-surface-1); padding: 0.7rem 0.95rem;
}
[data-testid="stAlertContainer"] p { font-size: 0.88rem; line-height: 1.55; color: var(--ptp-text) !important; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"])    { border-left-color: var(--ptp-accent-2); background: #eef0fa; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) { border-left-color: var(--ptp-warn); background: #f7efe0; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"])   { border-left-color: var(--ptp-down); background: #f8e7e3; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) { border-left-color: var(--ptp-up); background: #e7f1ea; }

/* ── Metrics: meters, not cards ───────────────────────────────────────── */
[data-testid="stMetric"] { background: transparent; border: none; border-top: 2px solid var(--ptp-text); border-radius: 0; padding: 0.5rem 0 0.2rem; }
[data-testid="stMetricLabel"] p { font-size: 0.84rem !important; font-weight: 600; color: var(--ptp-text-2) !important; }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] p {
  font-family: var(--ptp-display); font-size: 2rem !important; font-weight: 700; letter-spacing: 0.01em; color: var(--ptp-text);
}

/* ── Charts: every Plotly figure is drawn on a sheet ─────────────────── */
[data-testid="stPlotlyChart"] {
  background: var(--ptp-surface-1); border: 1px solid var(--ptp-border-strong); border-radius: 2px; overflow: hidden;
}
.js-plotly-plot .plotly .main-svg:first-of-type { background: transparent !important; }
.js-plotly-plot .plotly .modebar { background: transparent !important; }
.js-plotly-plot .plotly .modebar-btn path { fill: var(--ptp-faint) !important; }
.js-plotly-plot .plotly .modebar-btn:hover path, .js-plotly-plot .plotly .modebar-btn.active path { fill: var(--ptp-accent) !important; }

/* ── Inputs ───────────────────────────────────────────────────────────── */
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="base-input"] { border-radius: 2px !important; }
[data-baseweb="tag"] { background: rgba(14,122,126,0.12) !important; border: 1px solid rgba(14,122,126,0.45); border-radius: 2px !important; }
[data-baseweb="tag"] span, [data-baseweb="tag"] [role="presentation"] { color: var(--ptp-text) !important; }
[data-testid="stWidgetLabel"] p { color: var(--ptp-text-2); font-size: 0.86rem; font-weight: 500; }

/* ══ Components ══════════════════════════════════════════════════════════ */
.ptp-num { font-family: var(--ptp-display); font-variant-numeric: tabular-nums lining-nums; }
.ptp-mono { font-variant-numeric: tabular-nums; }
.ptp-up { color: var(--ptp-up); } .ptp-down { color: var(--ptp-down); }
.ptp-warn { color: var(--ptp-warn); } .ptp-info { color: var(--ptp-info); }
.ptp-muted { color: var(--ptp-muted); } .ptp-faint { color: var(--ptp-faint); }

/* Page header: the busbar */
.ptp-ph { margin: 0 0 1.6rem; }
.ptp-ph__bus { display: flex; align-items: center; gap: 0.6rem; }
.ptp-ph__node {
  flex: 0 0 auto; width: 15px; height: 15px; border-radius: 50%;
  background: var(--ph-accent, var(--ptp-accent)); box-shadow: 0 0 0 3px var(--ptp-bg), 0 0 0 5px var(--ptp-text);
}
.ptp-ph__layer { flex: 0 0 auto; font-size: 0.9rem; font-weight: 600; color: var(--ptp-text-2); }
.ptp-ph__rule { flex: 1 1 auto; height: 4px; background: var(--ptp-text); }
.ptp-ph__title {
  margin: 0.55rem 0 0; font-family: var(--ptp-display); font-size: 3.6rem; line-height: 0.95;
  font-weight: 800; letter-spacing: 0.005em; color: var(--ptp-text);
}
.ptp-ph__sub { margin-top: 0.7rem; max-width: 72ch; font-size: 1rem; line-height: 1.6; color: var(--ptp-text-2); }
.ptp-ph__sources { margin-top: 0.55rem; font-size: 0.84rem; color: var(--ptp-muted); }
.ptp-ph__sources strong { color: var(--ptp-text-2); font-weight: 600; }
@media (max-width: 640px) { .ptp-ph__title { font-size: 2.5rem; } }

/* Section header */
.ptp-section-header { margin: 2rem 0 0.8rem; }
.ptp-section-header__title {
  font-family: var(--ptp-display); font-size: 1.85rem; font-weight: 750; line-height: 1; color: var(--ptp-text);
  display: flex; align-items: center; gap: 0.75rem;
}
.ptp-section-header__title::after { content: ""; flex: 1; height: 2px; background: var(--ptp-text); }
.ptp-section-header__caption { margin-top: 0.45rem; font-size: 0.92rem; color: var(--ptp-text-2); max-width: 80ch; }

/* Tags (formerly chips): a square terminal and plain text */
.ptp-chip { display: inline-flex; align-items: center; gap: 0.4rem; font-size: 0.84rem; color: var(--ptp-text-2); margin-right: 0.9rem; }
.ptp-chip__dot { width: 8px; height: 8px; background: var(--chip-dot); outline: 1px solid var(--ptp-text); outline-offset: 1px; }

/* Meter (KPI): reads like an instrument on a panel; also styles legacy .kpi-card markup */
.ptp-kpi, .kpi-card {
  position: relative; border-top: 2px solid var(--ptp-text); padding: 0.55rem 0 0.4rem; margin-bottom: 0.6rem; min-height: 6.4rem;
}
.ptp-kpi::before {
  content: ""; position: absolute; left: 0; top: -2px; width: 34px; height: 5px;
  background: var(--kpi-accent, var(--ptp-page-accent, var(--ptp-accent)));
}
.ptp-kpi__label, .kpi-label { font-size: 0.86rem; font-weight: 600; color: var(--ptp-text-2); line-height: 1.3; }
.ptp-kpi__body { display: flex; align-items: flex-end; justify-content: space-between; gap: 0.75rem; margin-top: 0.15rem; }
.ptp-kpi__value, .kpi-value {
  font-family: var(--ptp-display); font-variant-numeric: tabular-nums lining-nums;
  font-size: 2.35rem; font-weight: 700; letter-spacing: 0.005em; line-height: 1; color: var(--ptp-text); white-space: nowrap;
}
.ptp-kpi__spark { flex: 0 0 auto; }
.ptp-kpi__delta, .kpi-delta { font-size: 0.82rem; line-height: 1.4; margin-top: 0.35rem; color: var(--ptp-muted); }
.delta-red   { color: var(--ptp-down); font-weight: 500; }
.delta-amber { color: var(--ptp-warn); font-weight: 500; }
.delta-green { color: var(--ptp-up); font-weight: 500; }
.delta-blue  { color: var(--ptp-text-2); }
@media (max-width: 640px) { .ptp-kpi__value, .kpi-value { font-size: 1.9rem; } .ptp-kpi, .kpi-card { min-height: 0; } }

/* Status marks (legacy pill() markup): a filled square and the word */
.pill-critical, .pill-warn, .pill-ok, .pill-good {
  display: inline-flex; align-items: center; gap: 0.4rem; font-size: 0.86rem; font-weight: 650; vertical-align: 0.02em;
}
.pill-critical::before, .pill-warn::before, .pill-ok::before, .pill-good::before { content: ""; width: 10px; height: 10px; background: currentColor; }
.pill-critical { color: var(--ptp-down); } .pill-warn { color: var(--ptp-warn); }
.pill-ok { color: var(--ptp-text-2); } .pill-good { color: var(--ptp-up); }

/* Callout: a note pinned to the sheet; also styles legacy .commentary markup */
.ptp-callout, .commentary {
  background: var(--ptp-surface-1); border: 1px solid var(--ptp-border);
  border-left: 4px solid var(--co-accent, var(--ptp-text-2)); border-radius: 2px;
  padding: 0.75rem 1rem; margin: 0.4rem 0 1rem; color: var(--ptp-text); font-size: 0.92rem; line-height: 1.6;
}
.ptp-callout strong, .commentary strong { color: var(--ptp-text); }
.ptp-callout__title { font-weight: 650; color: var(--ptp-text); margin-bottom: 0.2rem; }
.commentary-critical { --co-accent: var(--ptp-down); }
.commentary-warn     { --co-accent: var(--ptp-warn); }
.commentary-good     { --co-accent: var(--ptp-up); }

/* Meter strip (market snapshot) */
.ptp-ticker { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); border-top: 2px solid var(--ptp-text); border-bottom: 1px solid var(--ptp-text); background: var(--ptp-surface-1); }
.ptp-ticker__item { min-width: 0; padding: 0.6rem 0.9rem 0.55rem; border-left: 1px solid var(--ptp-border); }
.ptp-ticker__item:first-child { border-left: none; }
.ptp-ticker__head { display: flex; align-items: baseline; justify-content: space-between; gap: 0.5rem; }
.ptp-ticker__sym { font-size: 0.88rem; font-weight: 700; color: var(--ptp-text); }
.ptp-ticker__sub { font-size: 0.74rem; color: var(--ptp-muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.ptp-ticker__row { display: flex; align-items: flex-end; justify-content: space-between; gap: 0.5rem; margin-top: 0.2rem; }
.ptp-ticker__px { font-family: var(--ptp-display); font-variant-numeric: tabular-nums lining-nums; font-size: 1.9rem; font-weight: 700; color: var(--ptp-text); line-height: 1; }
.ptp-ticker__unit { font-size: 0.72rem; color: var(--ptp-muted); margin-top: 0.15rem; }
.ptp-ticker__chg { font-size: 0.8rem; font-weight: 600; margin-top: 0.15rem; white-space: nowrap; }
.ptp-ticker__na { color: var(--ptp-faint); }
.ptp-ticker-note { margin-top: 0.45rem; font-size: 0.8rem; color: var(--ptp-muted); }

/* Interconnector (landing page hero) */
.ptp-ic { margin-top: 1.6rem; }
.ptp-ic svg { width: 100%; height: auto; display: block; overflow: visible; }
.ptp-ic__flow { stroke-dasharray: 14 10; animation: ptp-flow 1.6s linear infinite; }
.ptp-ic__flow--rev { animation-direction: reverse; }
@keyframes ptp-flow { to { stroke-dashoffset: -24; } }
.ptp-ic__caption { margin-top: 0.4rem; font-size: 0.84rem; color: var(--ptp-muted); }

/* Single-line diagram of the platform (landing page) */
.ptp-sld-group { font-family: var(--ptp-display); font-size: 1.35rem; font-weight: 700; color: var(--ptp-text); margin: 0.9rem 0 0.3rem; }
[class*="st-key-ptp_bus_"] { border-left: 4px solid var(--ptp-text); margin-left: 0.5rem; padding: 0.2rem 0 0.4rem; gap: 1.4rem; }
[class*="st-key-ptp_tap_"] { position: relative; padding: 0 0 0 3rem; }
[class*="st-key-ptp_tap_"]::before {
  content: ""; position: absolute; left: 0; top: 1rem; width: 2rem; height: 2px; background: var(--ptp-text);
}
.ptp-tap__node {
  position: absolute; left: calc(-1rem - 7px); top: calc(1rem - 6px); width: 14px; height: 14px; border-radius: 50%;
  background: var(--tap-accent); box-shadow: 0 0 0 2px var(--ptp-text);
}
.ptp-tap { position: relative; }
.ptp-tap__head { display: flex; align-items: baseline; flex-wrap: wrap; gap: 0.2rem 0.9rem; }
.ptp-tap__title { font-family: var(--ptp-display); font-size: 1.7rem; font-weight: 750; line-height: 1.05; color: var(--ptp-text); }
.ptp-tap__stat { font-size: 0.84rem; font-weight: 600; color: var(--tap-accent); }
.ptp-tap__desc { margin-top: 0.25rem; max-width: 76ch; font-size: 0.93rem; line-height: 1.6; color: var(--ptp-text-2); }
[class*="st-key-ptp_tap_"] [data-testid="stPageLink-NavLink"] { margin-top: 0.35rem; padding: 0.25rem 0.75rem; }

/* Footer */
.ptp-footer { margin-top: 2.8rem; padding-top: 0.9rem; border-top: 2px solid var(--ptp-text); font-size: 0.8rem; line-height: 1.75; color: var(--ptp-muted); }
.ptp-footer a { color: var(--ptp-text-2) !important; text-decoration: underline; text-decoration-color: rgba(19,47,56,0.3); text-underline-offset: 3px; }
.ptp-footer a:hover { color: var(--ptp-text) !important; text-decoration-color: var(--ptp-text); }
.ptp-footer__sources { display: flex; flex-wrap: wrap; gap: 0.15rem 1.1rem; }
.ptp-footer__label { color: var(--ptp-muted); }
.ptp-footer__brand { color: var(--ptp-text); font-weight: 650; }
.ptp-footer__note, .ptp-footer__disclaimer { margin-top: 0.3rem; }

/* ══ Landing page ════════════════════════════════════════════════════════ */
.st-key-ptp_hero { padding: 0.4rem 0 0.6rem; }
.ptp-hero__title {
  margin-top: 0.9rem; font-family: var(--ptp-display); font-size: 5.2rem; line-height: 0.88; font-weight: 800;
  letter-spacing: 0.002em; color: var(--ptp-text); max-width: 22ch;
}
.ptp-hero__lead { margin-top: 1.1rem; font-size: 1.12rem; line-height: 1.55; color: var(--ptp-text); max-width: 60ch; }
.ptp-hero__body { margin-top: 0.5rem; font-size: 0.95rem; line-height: 1.65; color: var(--ptp-text-2); max-width: 70ch; }
.st-key-ptp_cta { margin-top: 1rem; }
.st-key-ptp_cta_primary [data-testid="stPageLink-NavLink"] { background: var(--ptp-accent); }
.st-key-ptp_cta_primary [data-testid="stPageLink-NavLink"] p,
.st-key-ptp_cta_primary [data-testid="stPageLink-NavLink"] [data-testid="stIconMaterial"] { color: #fff !important; }
.ptp-strip-head { display: flex; align-items: baseline; justify-content: space-between; flex-wrap: wrap; gap: 0.25rem 1rem; margin: 2rem 0 0.35rem; }
.ptp-strip-head__title { font-family: var(--ptp-display); font-size: 1.6rem; font-weight: 750; color: var(--ptp-text); }
.ptp-strip-head__meta { font-size: 0.82rem; color: var(--ptp-muted); }
.ptp-status-row { margin-top: 0.6rem; font-size: 0.84rem; color: var(--ptp-muted); }
.ptp-status-row__label { color: var(--ptp-text-2); font-weight: 600; margin-right: 0.6rem; }
.ptp-about { color: var(--ptp-text-2); font-size: 0.95rem; line-height: 1.8; max-width: 80ch; }
.ptp-about-stack { color: var(--ptp-text-2); font-size: 0.88rem; line-height: 1.85; margin-top: 0.5rem; }
.ptp-about-stack__title { color: var(--ptp-text); font-size: 0.9rem; font-weight: 650; margin-bottom: 0.1rem; }
@media (max-width: 900px) { .ptp-hero__title { font-size: 3.6rem; } }
@media (max-width: 640px) { .ptp-hero__title { font-size: 2.9rem; } .ptp-hero__lead { font-size: 1rem; } }
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
    "<rect x='1' y='1' width='30' height='30' rx='2' fill='#f5f7f4' stroke='#132f38' stroke-width='2'/>"
    "<path d='M8 21 C13 21 13 11 18 11 L24 11' fill='none' stroke='#132f38' stroke-width='2.4'/>"
    "<circle cx='8' cy='21' r='4' fill='#0e7a7e' stroke='#132f38' stroke-width='1.6'/>"
    "<circle cx='24' cy='11' r='4' fill='#2f45c6' stroke='#132f38' stroke-width='1.6'/>"
    "</svg>"
)


def logo_svg() -> str:
    """Square brand mark (a Nordic and a Continental node joined by a cable)."""
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
    """Inline tag: an optional square terminal in the tone colour followed by plain text."""
    color = _tone(tone)
    dot_html = f'<span class="ptp-chip__dot" style="--chip-dot:{color}"></span>' if dot else ""
    title_attr = f' title="{_attr(title)}"' if title else ""
    return f'<span class="ptp-chip"{title_attr}>{dot_html}{_esc(text)}</span>'


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
    """Meter: label, display-face value, optional delta line and sparkline.

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
    style = f"--co-accent:{color};"
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


def tap(title: str, description: str, stat: str | None = None, accent: str | None = None) -> str:
    """One page on the landing page's single-line diagram: node, title, key figure, text."""
    color = accent or COLORS["accent"]
    stat_html = f'<span class="ptp-tap__stat">{_esc(stat)}</span>' if stat else ""
    return (
        f'<div class="ptp-tap" style="--tap-accent:{color}"><span class="ptp-tap__node"></span>'
        f'<div class="ptp-tap__head"><span class="ptp-tap__title" role="heading" aria-level="3">{_esc(title)}</span>'
        f'{stat_html}</div><div class="ptp-tap__desc">{_esc(description)}</div></div>'
    )


def interconnector(left: dict, right: dict, link: str, capacity: str | None = None,
                   as_of: str | None = None) -> str:
    """Two bidding zones joined by a cable, with the price at each end.

    ``left`` and ``right`` are dicts with ``zone``, ``place`` and ``price`` (EUR/MWh or
    None). Power is drawn flowing from the cheaper zone to the dearer one; without both
    prices the cable is drawn still.
    """
    lp, rp = left.get("price"), right.get("price")
    have = all(isinstance(v, (int, float)) and math.isfinite(v) for v in (lp, rp))
    spread = (rp - lp) if have else None
    if spread is None or abs(spread) < 0.5:
        flow_cls, direction = "", "Prices are level; no flow incentive."
    elif spread > 0:
        flow_cls = "ptp-ic__flow"
        direction = f"{right['place']} pays €{spread:,.0f}/MWh more: the incentive is to export from {left['place']}."
    else:
        flow_cls = "ptp-ic__flow ptp-ic__flow--rev"
        direction = f"{left['place']} pays €{-spread:,.0f}/MWh more: the incentive is to import into {left['place']}."

    def _px(v):
        return f"{v:,.0f}" if isinstance(v, (int, float)) and math.isfinite(v) else "n/a"

    ink, fjord, cobalt = COLORS["text"], ACCENTS["l1"], COLORS["accent_2"]
    flow = (f'<path class="{flow_cls}" d="M120 70 L880 70" stroke="{COLORS["surface_1"]}" stroke-width="4" fill="none"/>'
            if flow_cls else "")
    cap = f", {capacity}" if capacity else ""
    svg = (
        '<svg viewBox="0 0 1000 150" role="img" '
        f'aria-label="{_attr(left["zone"])} {_px(lp)} EUR per MWh, {_attr(right["zone"])} {_px(rp)} EUR per MWh">'
        f'<path d="M120 70 L880 70" stroke="{ink}" stroke-width="10" fill="none"/>{flow}'
        f'<circle cx="80" cy="70" r="40" fill="{fjord}" stroke="{ink}" stroke-width="4"/>'
        f'<circle cx="920" cy="70" r="40" fill="{cobalt}" stroke="{ink}" stroke-width="4"/>'
        f'<text x="80" y="78" text-anchor="middle" font-family="{_attr(FONT_DISPLAY)}" font-weight="800" font-size="24" fill="#fff">{_esc(left["zone"])}</text>'
        f'<text x="920" y="78" text-anchor="middle" font-family="{_attr(FONT_DISPLAY)}" font-weight="800" font-size="24" fill="#fff">{_esc(right["zone"])}</text>'
        f'<text x="140" y="44" font-family="{_attr(FONT_DISPLAY)}" font-weight="800" font-size="42" fill="{ink}">€{_px(lp)}</text>'
        f'<text x="860" y="44" text-anchor="end" font-family="{_attr(FONT_DISPLAY)}" font-weight="800" font-size="42" fill="{ink}">€{_px(rp)}</text>'
        f'<text x="140" y="112" font-family="{_attr(FONT_SANS)}" font-size="15" fill="{COLORS["text_2"]}">{_esc(left["place"])}, day-ahead average</text>'
        f'<text x="860" y="112" text-anchor="end" font-family="{_attr(FONT_SANS)}" font-size="15" fill="{COLORS["text_2"]}">{_esc(right["place"])}, day-ahead average</text>'
        f'<text x="500" y="112" text-anchor="middle" font-family="{_attr(FONT_SANS)}" font-weight="600" font-size="15" fill="{ink}">{_esc(link)}{_esc(cap)}</text>'
        "</svg>"
    )
    when = f" Delivery day {as_of}." if as_of else ""
    return f'<div class="ptp-ic">{svg}<div class="ptp-ic__caption">{_esc(direction)}{_esc(when)}</div></div>'


# ══════════════════════════════════════════════════════════════════════════════
# Page structure (render directly)
# ══════════════════════════════════════════════════════════════════════════════

def page_header(kicker: str, title: str, subtitle: str | None = None,
                sources: Sequence[str] | None = None, accent: str | None = None) -> str:
    """Render the standard page header (the busbar) and return its HTML.

    ``kicker`` names the layer (for example 'Layer 1'), drawn beside the node where the
    page taps into the bus; ``sources`` lists data-source names; ``accent`` colours the
    node (typically from ``ACCENTS``).
    """
    style = f' style="--ph-accent:{_attr(accent)}"' if accent else ""
    if accent:
        # Page-level accent picked up by the meters on the same page.
        st.html(f"<style>:root{{--ptp-page-accent:{_attr(accent)};}}</style>")
    sub_html = f'<div class="ptp-ph__sub">{_esc(subtitle)}</div>' if subtitle else ""
    src_html = ""
    if sources:
        names = ", ".join(_esc(s) for s in sources)
        src_html = f'<div class="ptp-ph__sources"><strong>Sources</strong> {names}</div>'
    out = (
        f'<div class="ptp-ph"{style}>'
        f'<div class="ptp-ph__bus"><span class="ptp-ph__node"></span>'
        f'<span class="ptp-ph__layer">{_esc(kicker)}</span><span class="ptp-ph__rule"></span></div>'
        f'<div class="ptp-ph__title" role="heading" aria-level="1">{_esc(title)}</div>'
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
