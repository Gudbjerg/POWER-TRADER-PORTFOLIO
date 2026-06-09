"""
Phase E — BESS Optimiser
Battery Energy Storage System multi-market revenue stack and financial model for Nordic power markets.
Markets: DAM arbitrage, FCR-N (Frequency Containment Reserve — Normal), FCR-D (Disturbed), intraday.
All revenue inputs derived from the existing NO2 feature matrix plus hardcoded Nordic FCR price averages.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="BESS Optimiser",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.helpers import apply_dark_theme, kpi_card, delta_span
from models.feature_assembly import assemble_features
from config.settings import (
    BESS_FCR_N_EUR_PER_MW_YEAR,
    BESS_FCR_D_EUR_PER_MW_YEAR,
    BESS_FCR_D_AVAIL_DISC,
    BESS_DAM_VIABLE_DAYS_PCT,
    BESS_DAM_DISPATCH_HOURS,
    BESS_OPEX_PCT_CAPEX,
    BESS_DEFAULT_DISCOUNT_PCT,
)

apply_dark_theme()

# ── Colour palette (consistent with other pages) ──────────────────────────────
_C = {
    "dam":      "#58a6ff",
    "fcr_n":    "#3fb950",
    "fcr_d":    "#d4ac3a",
    "intraday": "#d2a8ff",
    "opex":     "#f85149",
    "depr":     "#484f58",
}


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def _dam_spread_from_features(feat: pd.DataFrame) -> dict:
    """
    Estimate achievable DAM daily spread from NO2 day-ahead prices.
    Uses rolling 5-day range as a proxy for daily peak/off-peak spread opportunity.
    Returns stats dict; falls back to Nordic historical averages if data is unavailable.
    """
    _fallback = {
        "achievable": 22.0, "p80": 28.0, "median": 14.0,
        "n_obs": 0, "fallback": True,
        "date_min": "n/a", "date_max": "n/a",
    }
    if feat.empty or "no2" not in feat.columns:
        return _fallback
    no2 = feat["no2"].dropna()
    if len(no2) < 30:
        return {**_fallback, "n_obs": len(no2)}
    roll = (no2.rolling(5).max() - no2.rolling(5).min()).dropna()
    p80  = float(roll.quantile(0.80))
    return {
        "achievable": float(roll[roll >= p80].mean()),
        "p80":        p80,
        "median":     float(roll.median()),
        "n_obs":      len(roll),
        "fallback":   False,
        "date_min":   str(feat["date"].min().date()),
        "date_max":   str(feat["date"].max().date()),
    }


def _irr(cash_flows: list[float]) -> float:
    """Bisection IRR over [0%, 300%]. Returns NaN if no real solution."""
    def _npv(r: float) -> float:
        return sum(c / (1.0 + r) ** t for t, c in enumerate(cash_flows))

    if _npv(0.0) <= 0:
        return float("nan")
    hi = 3.0
    if _npv(hi) >= 0:
        return float("nan")
    lo = 0.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if _npv(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-7:
            break
    return (lo + hi) / 2.0 * 100.0   # %


def _run_model(
    capacity_mwh: float,
    power_mw: float,
    eff_pct: float,
    deg_pct: float,
    capex_per_kwh: float,
    disc_pct: float,
    lifetime: int,
    markets: dict,
    spread: dict,
) -> dict:
    """
    Full BESS financial model. Accounts for annual capacity degradation.
    Returns dict with year-by-year data and summary metrics.
    """
    eff  = eff_pct / 100.0
    deg  = deg_pct / 100.0
    disc = disc_pct / 100.0

    capex      = capacity_mwh * 1_000.0 * capex_per_kwh      # €
    opex_yr    = capex * BESS_OPEX_PCT_CAPEX                  # € / year (fixed)
    depr_yr    = capex / lifetime                             # straight-line, non-cash

    # Year-1 revenue components
    energy_per_cycle = min(power_mw * BESS_DAM_DISPATCH_HOURS, capacity_mwh)
    viable_days      = 365.0 * BESS_DAM_VIABLE_DAYS_PCT

    dam_y1 = (spread["achievable"] * energy_per_cycle * eff * viable_days
               if markets["dam"] else 0.0)
    fcr_n_y1 = (power_mw * BESS_FCR_N_EUR_PER_MW_YEAR if markets["fcr_n"] else 0.0)
    fcr_d_y1 = (power_mw * BESS_FCR_D_EUR_PER_MW_YEAR * (1.0 - BESS_FCR_D_AVAIL_DISC)
                if markets["fcr_d"] else 0.0)
    id_y1    = dam_y1 * 0.50 if markets["intraday"] else 0.0
    total_y1 = dam_y1 + fcr_n_y1 + fcr_d_y1 + id_y1

    # Per-year data (with degradation on energy-related revenues)
    rows      = []
    cash_flows = [-capex]
    cumul_ud   = -capex    # undiscounted
    cumul_npv  = -capex    # discounted
    breakeven_yr: float | None = None

    for yr in range(1, lifetime + 1):
        fac = (1.0 - deg) ** (yr - 1)
        dam_yr    = dam_y1  * fac
        fcr_n_yr  = fcr_n_y1                 # availability-based, no degradation factor
        fcr_d_yr  = fcr_d_y1                 # same
        id_yr     = id_y1   * fac
        rev_yr    = dam_yr + fcr_n_yr + fcr_d_yr + id_yr
        net_cf    = rev_yr - opex_yr
        disc_cf   = net_cf / (1.0 + disc) ** yr

        cash_flows.append(net_cf)
        cumul_ud  += net_cf
        cumul_npv += disc_cf
        if breakeven_yr is None and cumul_ud >= 0:
            # Linear interpolation within the year
            prev_cumul = cumul_ud - net_cf
            frac = -prev_cumul / net_cf if net_cf > 0 else 0.0
            breakeven_yr = yr - 1 + frac

        rows.append({
            "year":       yr,
            "dam":        dam_yr,
            "fcr_n":      fcr_n_yr,
            "fcr_d":      fcr_d_yr,
            "intraday":   id_yr,
            "revenue":    rev_yr,
            "opex":       opex_yr,
            "depr":       depr_yr,
            "net_cf":     net_cf,
            "cumul_ud":   cumul_ud,
            "cumul_npv":  cumul_npv,
            "disc_cf":    disc_cf,
        })

    npv_total = sum(cf / (1.0 + disc) ** t for t, cf in enumerate(cash_flows))
    irr_val   = _irr(cash_flows)
    payback_simple = (capex / (total_y1 - opex_yr)
                      if (total_y1 - opex_yr) > 0 else float("inf"))

    return {
        "capex":           capex,
        "opex_yr":         opex_yr,
        "depr_yr":         depr_yr,
        "dam_y1":          dam_y1,
        "fcr_n_y1":        fcr_n_y1,
        "fcr_d_y1":        fcr_d_y1,
        "id_y1":           id_y1,
        "total_y1":        total_y1,
        "rows":            rows,
        "cash_flows":      cash_flows,
        "npv":             npv_total,
        "irr":             irr_val,
        "payback_simple":  payback_simple,
        "breakeven_yr":    breakeven_yr,
    }


def _npv_only(
    capacity_mwh, power_mw, eff_pct, deg_pct, capex_per_kwh,
    disc_pct, lifetime, markets, spread,
    capex_mult=1.0, fcr_mult=1.0, spread_mult=1.0,
) -> float:
    """NPV for sensitivity analysis. Applies multiplicative adjustments."""
    adj_spread = {**spread, "achievable": spread["achievable"] * spread_mult}
    s = _run_model(
        capacity_mwh, power_mw, eff_pct, deg_pct,
        capex_per_kwh * capex_mult, disc_pct, lifetime, markets, adj_spread,
    )
    # FCR revenues are capacity-based (not degraded), so the FCR sensitivity delta
    # is a constant annual increment discounted over the full lifetime.
    if fcr_mult != 1.0:
        fcr_adj = (s["fcr_n_y1"] + s["fcr_d_y1"]) * (fcr_mult - 1.0)
        disc = disc_pct / 100.0
        fcr_npv_adj = sum(
            fcr_adj / (1.0 + disc) ** yr
            for yr in range(1, lifetime + 1)
        )
        return s["npv"] + fcr_npv_adj
    return s["npv"]


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("## Phase E · BESS Optimiser")
st.caption(
    "Battery Energy Storage System multi-market revenue stack for Nordic power markets. "
    "Configure asset parameters and select markets to compute financial projections."
)

st.markdown(
    """<div style="background:#1f1108;border-left:3px solid #d4ac3a;padding:10px 14px;
    border-radius:4px;color:#d4ac3a;font-size:0.84rem;margin-bottom:14px;">
    <strong>Revenue estimates are simplified deterministic projections</strong> based on historical
    average market prices and are not guaranteed. Actual returns depend on contracted FCR volumes,
    intraday market conditions, battery degradation, and grid connection costs.
    FCR prices are market averages, not contracted rates. Not financial advice.
    </div>""",
    unsafe_allow_html=True,
)

# ── Feature matrix (for DAM spread) ──────────────────────────────────────────
with st.spinner("Loading market data for DAM spread estimation…"):
    _feat = assemble_features(years=3)

_spread = _dam_spread_from_features(_feat)
if _spread["fallback"]:
    st.caption(
        "DAM spread estimated from Nordic historical averages (€22/MWh achievable, top-20% filter). "
        "Feature matrix unavailable. Visit Layer 2 first to populate. FCR revenues are unaffected."
    )
else:
    st.caption(
        f"DAM spread estimated from NO2 prices {_spread['date_min']} → {_spread['date_max']} "
        f"({_spread['n_obs']:,} obs). "
        f"Median 5-day range: €{_spread['median']:.1f}/MWh · "
        f"Top-20% achievable: €{_spread['achievable']:.1f}/MWh."
    )

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR: ASSET CONFIGURATOR
# ═══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("### Asset Configurator")
st.sidebar.divider()

st.sidebar.markdown("**Battery specifications**")
_cap_mwh = float(st.sidebar.slider("Capacity (MWh)", 1, 500, 100, step=5,
                                    key="bess_cap"))
_pow_mw   = float(st.sidebar.slider("Power rating (MW)", 1, 200, 50, step=5,
                                     key="bess_pow"))
_eff_pct  = float(st.sidebar.slider("Round-trip efficiency (%)", 70, 95, 85,
                                     key="bess_eff"))
st.sidebar.caption(f"C-rate: {_pow_mw / _cap_mwh:.2f}  ·  Duration: {_cap_mwh / _pow_mw:.1f} h")

st.sidebar.divider()
st.sidebar.markdown("**Financial parameters**")
_capex_kwh  = float(st.sidebar.slider("CAPEX (€/kWh)", 100, 600, 250, step=10,
                                        key="bess_capex"))
_deg_pct    = float(st.sidebar.slider("Degradation (%/year)", 0, 5, 3, step=1,
                                        key="bess_deg"))
_lifetime   = int(st.sidebar.slider("Lifetime (years)", 5, 25, 15,
                                     key="bess_life"))
_disc_pct   = float(st.sidebar.slider("Discount rate (%)", 4, 15,
                                        int(BESS_DEFAULT_DISCOUNT_PCT),
                                        key="bess_disc"))

_capex_total = _cap_mwh * 1_000.0 * _capex_kwh
st.sidebar.caption(
    f"Total CAPEX: **€{_capex_total / 1e6:.2f}M**  ·  "
    f"Annual OPEX: €{_capex_total * BESS_OPEX_PCT_CAPEX / 1e3:,.0f}k"
)

st.sidebar.divider()
st.sidebar.markdown("**Market participation**")
_do_dam = st.sidebar.checkbox("DAM arbitrage (day-ahead)", value=True, key="bess_dam")
_do_fcr_n = st.sidebar.checkbox(
    f"FCR-N (Normal, ≈€{BESS_FCR_N_EUR_PER_MW_YEAR/1000:.0f}k/MW/yr)", value=True, key="bess_fcrn")
_do_fcr_d = st.sidebar.checkbox(
    f"FCR-D (Disturbed, ≈€{BESS_FCR_D_EUR_PER_MW_YEAR/1000:.0f}k/MW/yr, −30% avail. disc.)",
    value=True, key="bess_fcrd")
_do_id = st.sidebar.checkbox("Intraday arbitrage (50% of DAM proxy)", value=True, key="bess_id")

_markets = {"dam": _do_dam, "fcr_n": _do_fcr_n, "fcr_d": _do_fcr_d, "intraday": _do_id}

# ── Run model ─────────────────────────────────────────────────────────────────
_m = _run_model(
    _cap_mwh, _pow_mw, _eff_pct, _deg_pct,
    _capex_kwh, _disc_pct, _lifetime, _markets, _spread,
)

# ═══════════════════════════════════════════════════════════════════════════════
# KPI CARDS
# ═══════════════════════════════════════════════════════════════════════════════
_k1, _k2, _k3, _k4 = st.columns(4)

with _k1:
    st.markdown(
        kpi_card(
            "Year-1 revenue",
            f"€{_m['total_y1'] / 1e3:,.0f}k",
            delta_span(f"OPEX: €{_m['opex_yr'] / 1e3:,.0f}k/yr", "blue"),
        ),
        unsafe_allow_html=True,
    )
with _k2:
    _pb = _m["payback_simple"]
    _pb_str = f"{_pb:.1f} yrs" if np.isfinite(_pb) else "Never"
    _pb_col = ("green" if _pb < 10 else "amber" if _pb < _lifetime else "red")
    st.markdown(
        kpi_card(
            "Simple payback",
            _pb_str,
            delta_span(f"CAPEX €{_m['capex'] / 1e6:.2f}M", _pb_col),
        ),
        unsafe_allow_html=True,
    )
with _k3:
    _npv_val = _m["npv"]
    _npv_col = "green" if _npv_val > 0 else "red"
    st.markdown(
        kpi_card(
            f"NPV @ {_disc_pct:.0f}%",
            f"€{_npv_val / 1e6:+.2f}M",
            delta_span(f"{_lifetime}-year lifetime", _npv_col),
        ),
        unsafe_allow_html=True,
    )
with _k4:
    _irr_val = _m["irr"]
    if np.isnan(_irr_val):
        _irr_str, _irr_col = "< 0%", "red"
    else:
        _irr_str = f"{_irr_val:.1f}%"
        _irr_col = "green" if _irr_val > _disc_pct else ("amber" if _irr_val > 0 else "red")
    st.markdown(
        kpi_card("IRR", _irr_str,
                 delta_span(f"hurdle: {_disc_pct:.0f}%", _irr_col)),
        unsafe_allow_html=True,
    )

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 1: REVENUE BREAKDOWN STACKED BAR
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("#### Revenue Breakdown: Annual by Market (EUR)")

_yrs    = [r["year"]     for r in _m["rows"]]
_dam_v  = [r["dam"]      for r in _m["rows"]]
_fcrn_v = [r["fcr_n"]    for r in _m["rows"]]
_fcrd_v = [r["fcr_d"]    for r in _m["rows"]]
_id_v   = [r["intraday"] for r in _m["rows"]]
_opex_v = [-r["opex"]    for r in _m["rows"]]

_fig_bar = go.Figure()
if _do_dam:
    _fig_bar.add_trace(go.Bar(
        x=_yrs, y=_dam_v, name="DAM arbitrage",
        marker_color=_C["dam"],
        hovertemplate="Year %{x} DAM: €%{y:,.0f}<extra></extra>",
    ))
if _do_fcr_n:
    _fig_bar.add_trace(go.Bar(
        x=_yrs, y=_fcrn_v, name="FCR-N",
        marker_color=_C["fcr_n"],
        hovertemplate="Year %{x} FCR-N: €%{y:,.0f}<extra></extra>",
    ))
if _do_fcr_d:
    _fig_bar.add_trace(go.Bar(
        x=_yrs, y=_fcrd_v, name="FCR-D",
        marker_color=_C["fcr_d"],
        hovertemplate="Year %{x} FCR-D: €%{y:,.0f}<extra></extra>",
    ))
if _do_id:
    _fig_bar.add_trace(go.Bar(
        x=_yrs, y=_id_v, name="Intraday",
        marker_color=_C["intraday"],
        hovertemplate="Year %{x} Intraday: €%{y:,.0f}<extra></extra>",
    ))
_fig_bar.add_trace(go.Bar(
    x=_yrs, y=_opex_v, name="OPEX",
    marker_color=_C["opex"], opacity=0.7,
    hovertemplate="Year %{x} OPEX: €%{y:,.0f}<extra></extra>",
))
_fig_bar.update_layout(
    barmode="relative",
    template="plotly_dark",
    paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
    height=280,
    margin=dict(l=60, r=20, t=10, b=35),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(size=10)),
    xaxis=dict(title="Year", gridcolor="rgba(255,255,255,0.06)",
               tickfont=dict(size=10, color="#8b949e"), dtick=1),
    yaxis=dict(title="EUR / year", gridcolor="rgba(255,255,255,0.06)",
               tickfont=dict(size=10, color="#8b949e"), zeroline=True,
               zerolinecolor="rgba(255,255,255,0.2)"),
    hovermode="x unified",
)
st.plotly_chart(_fig_bar, use_container_width=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 2: CASH FLOW WATERFALL (YEAR 1)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("#### Year-1 Cash Flow Waterfall")

_wf_labels  = ["Revenue"] + (["DAM"] if _do_dam else [])
_wf_labels += (["FCR-N"] if _do_fcr_n else [])
_wf_labels += (["FCR-D"] if _do_fcr_d else [])
_wf_labels += (["Intraday"] if _do_id else [])
_wf_labels += ["OPEX", "Depreciation (non-cash)", "Net cash flow"]

_wf_vals = [_m["total_y1"]]
if _do_dam:    _wf_vals.append(_m["dam_y1"])
if _do_fcr_n:  _wf_vals.append(_m["fcr_n_y1"])
if _do_fcr_d:  _wf_vals.append(_m["fcr_d_y1"])
if _do_id:     _wf_vals.append(_m["id_y1"])
_wf_vals += [-_m["opex_yr"], -_m["depr_yr"],
              _m["total_y1"] - _m["opex_yr"]]

_wf_measure = ["absolute"]
_wf_measure += ["relative"] * (len(_wf_labels) - 3)
_wf_measure += ["relative", "relative", "total"]

_wf_colors = ["#3fb950"]  # Revenue
for _lab in _wf_labels[1:-3]:
    _wf_colors.append(_C.get(_lab.lower().replace(" ", "_").replace("-", "_"), "#58a6ff"))
_wf_colors += [_C["opex"], _C["depr"], "#ffffff"]

_fig_wf = go.Figure(go.Waterfall(
    orientation="v",
    measure=_wf_measure,
    x=_wf_labels,
    y=_wf_vals,
    connector=dict(line=dict(color="rgba(255,255,255,0.15)", width=1)),
    increasing=dict(marker=dict(color="#3fb950")),
    decreasing=dict(marker=dict(color="#f85149")),
    totals=dict(marker=dict(color="#58a6ff")),
    texttemplate="€%{y:,.0f}",
    textposition="outside",
    textfont=dict(size=9, color="#c9d1d9"),
    hovertemplate="%{x}: €%{y:,.0f}<extra></extra>",
))
_fig_wf.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
    height=300,
    margin=dict(l=60, r=20, t=30, b=50),
    xaxis=dict(tickfont=dict(size=10, color="#8b949e")),
    yaxis=dict(title="EUR", gridcolor="rgba(255,255,255,0.06)",
               tickfont=dict(size=10, color="#8b949e")),
    showlegend=False,
)
st.plotly_chart(_fig_wf, use_container_width=True)
st.caption("Depreciation shown for reference. Non-cash item; does not affect cash flow or NPV calculation.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 3: CUMULATIVE P&L / NPV CURVE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("#### Cumulative P&L over Asset Lifetime")

_yr_axis = [0] + _yrs
_cumud   = [-_m["capex"]] + [r["cumul_ud"]  for r in _m["rows"]]
_cumnpv  = [-_m["capex"]] + [r["cumul_npv"] for r in _m["rows"]]

_fig_cum = go.Figure()
_fig_cum.add_trace(go.Scatter(
    x=_yr_axis, y=_cumud,
    name="Cumulative cash flow (undiscounted)",
    line=dict(color="#58a6ff", width=2),
    hovertemplate="Year %{x}: €%{y:,.0f}<extra></extra>",
))
_fig_cum.add_trace(go.Scatter(
    x=_yr_axis, y=_cumnpv,
    name=f"Cumulative NPV (@ {_disc_pct:.0f}% discount)",
    line=dict(color="#d4ac3a", width=2, dash="dot"),
    hovertemplate="Year %{x} NPV: €%{y:,.0f}<extra></extra>",
))
_fig_cum.add_hline(y=0, line=dict(color="rgba(255,255,255,0.25)", width=1.2))

# Breakeven annotation
_be = _m.get("breakeven_yr")
if _be is not None and _be <= _lifetime:
    _fig_cum.add_vline(
        x=_be, line=dict(color="#3fb950", width=1.5, dash="dot"),
        annotation_text=f"  Breakeven yr {_be:.1f}",
        annotation_font=dict(color="#3fb950", size=10),
        annotation_position="top right",
    )

_fig_cum.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
    height=300,
    margin=dict(l=60, r=20, t=10, b=35),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(size=10)),
    xaxis=dict(title="Year", gridcolor="rgba(255,255,255,0.06)",
               tickfont=dict(size=10, color="#8b949e"), dtick=1),
    yaxis=dict(title="Cumulative EUR", gridcolor="rgba(255,255,255,0.06)",
               tickfont=dict(size=10, color="#8b949e"), zeroline=False),
    hovermode="x unified",
)
st.plotly_chart(_fig_cum, use_container_width=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SENSITIVITY TABLE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("#### NPV Sensitivity Analysis (±20%)")
st.caption(
    "Each row varies one input ±20%, holding all others constant at the configured values."
)

_base_npv = _m["npv"]
_sens_rows = []
for _var, _label, _kwarg in [
    ("CAPEX",      "CAPEX (€/kWh)",         "capex_mult"),
    ("FCR prices", "FCR prices (N + D)",     "fcr_mult"),
    ("DAM spread", "Energy spread (€/MWh)",  "spread_mult"),
]:
    _lo = _npv_only(
        _cap_mwh, _pow_mw, _eff_pct, _deg_pct, _capex_kwh, _disc_pct, _lifetime,
        _markets, _spread, **{_kwarg: 0.80},
    )
    _hi = _npv_only(
        _cap_mwh, _pow_mw, _eff_pct, _deg_pct, _capex_kwh, _disc_pct, _lifetime,
        _markets, _spread, **{_kwarg: 1.20},
    )
    _sens_rows.append({
        "Variable":    _label,
        "−20%":        f"€{_lo / 1e6:+.2f}M",
        "Base":        f"€{_base_npv / 1e6:+.2f}M",
        "+20%":        f"€{_hi / 1e6:+.2f}M",
        "NPV range":   f"€{(_hi - _lo) / 1e6:.2f}M",
    })

st.dataframe(pd.DataFrame(_sens_rows), use_container_width=True, hide_index=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# METHODOLOGY
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("Model methodology and assumptions", expanded=False):
    st.markdown(f"""
    **DAM arbitrage revenue:**
    Estimated from historical NO2 day-ahead prices. A rolling 5-day price range is used as a proxy
    for the intraday peak/off-peak spread available each day. The top {BESS_DAM_VIABLE_DAYS_PCT*100:.0f}%
    of spread days are treated as "achievable" ({BESS_DAM_VIABLE_DAYS_PCT*365:.0f} days/year).
    Revenue per viable day = achievable spread × min(power × {BESS_DAM_DISPATCH_HOURS:.0f}h, capacity) × efficiency.
    Intraday proxy = 50% of DAM (intraday spreads smaller but more frequent; see Löhndorf & Wozabal 2024).

    **FCR-N (Frequency Containment Reserve, Normal):**
    Always-on symmetric response at ±0.1 Hz. Revenue = power rating × €{BESS_FCR_N_EUR_PER_MW_YEAR:,.0f}/MW/year
    (approximate Nordic/NordREG market average). This is a capacity payment, not degraded by battery age.

    **FCR-D (Frequency Containment Reserve, Disturbed):**
    Asymmetric, activated on frequency deviation > 0.5 Hz. Revenue = power × €{BESS_FCR_D_EUR_PER_MW_YEAR:,.0f}/MW/year
    × (1 − {BESS_FCR_D_AVAIL_DISC:.0%} availability discount). The discount reflects the constraint that
    the battery must maintain ≈50% SoC to participate symmetrically in upward and downward regulation.

    **Degradation:**
    Energy-dependent revenues (DAM, intraday) decline annually at the configured degradation rate.
    FCR revenues (capacity-based) are not degraded. A more accurate model would use a cycle-count
    degradation curve (e.g., NMC: 0.002% per equivalent full cycle); linear approximation is used here.

    **Financial model:**
    NPV = −CAPEX + Σ(t=1..N) \[CF_t / (1 + r)^t\], where CF_t = Revenue_t − OPEX.
    OPEX is {BESS_OPEX_PCT_CAPEX*100:.0f}% of CAPEX per year (fixed). Depreciation (straight-line)
    is shown in the waterfall but does not enter the cash flow or NPV calculation.
    IRR is solved by bisection. Simple payback = CAPEX / (Year-1 Revenue − OPEX), undiscounted.

    **Sensitivity analysis:** Each input varied ±20% individually, other inputs held constant.
    FCR price sensitivity scales both FCR-N and FCR-D revenues proportionally.

    **Key references:**
    - Hameed, Z. & Træholt, C. (2025). *Multi-market revenue stacking for grid-scale BESS in Nordic power systems.*
      Applied Energy. (Framework for FCR + DAM + intraday joint optimisation.)
    - Löhndorf, N. & Wozabal, D. (2024). *Value of information in intraday electricity markets.*
      Operations Research. (Intraday arbitrage as a complement to DAM strategies.)

    **Limitations:** This is a simplified deterministic estimate. A rigorous analysis would use stochastic
    dynamic programming over hourly price paths (Hameed & Træholt), SoC constraints, grid connection
    costs, and contracted vs spot FCR exposure. FCR price averages mask significant auction-by-auction
    and seasonal variation. Results should be treated as directional, not as contracted cash flow projections.
    """)

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown(
    f"""<div style="color:#484f58;font-size:0.72rem;line-height:1.8;">
    Built by Tobias Gudbjerg &nbsp;|&nbsp;
    DAM data: ENTSO-E A44 / Nord Pool (NO2) &nbsp;|&nbsp;
    FCR prices: Nordic/NordREG market averages (FCR-N ≈ €{BESS_FCR_N_EUR_PER_MW_YEAR:,.0f}/MW/yr,
    FCR-D ≈ €{BESS_FCR_D_EUR_PER_MW_YEAR:,.0f}/MW/yr), approximate, varies by auction.<br>
    For analytical and educational purposes only. Not financial advice.
    Consult project-specific grid connection studies, PPA terms, and contracted FCR volumes
    for investment-grade projections.
    </div>""",
    unsafe_allow_html=True,
)
