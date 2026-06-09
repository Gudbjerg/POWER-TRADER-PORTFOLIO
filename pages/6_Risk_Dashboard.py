"""
D4 — Risk Dashboard
Portfolio VaR / stress-scenario simulator built on the D3 signal suite.
All signals are derived from the same feature matrix and logic used in Layer 2 / Mispricing Dashboard.
No new data sources required.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Risk Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.helpers import apply_dark_theme, kpi_card, delta_span
from models.feature_assembly import assemble_features

apply_dark_theme()

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

_ROLL_WINDOW = 252    # 1-year rolling percentile window
_MIN_PERIODS = 30     # minimum observations before signal fires
_ENTRY_Z     = 1.0    # z-score threshold for spread signal
_PCTILE_LO   = 25.0   # lower bound → long
_PCTILE_HI   = 75.0   # upper bound → short
_VaR_CONF    = 0.95
_N_BOOT      = 5000   # bootstrap resamples for VaR CI

# ── Signal catalogue ──────────────────────────────────────────────────────────
_SIGNALS = [
    {
        "id":         "ttf_seasonal",
        "label":      "TTF Seasonal Position",
        "category":   "Gas",
        "underlying": "ttf",
        "color":      "#e07b39",
        "source":     "D3 sig 1 / Tab 11",
        "description": "Long TTF when rolling 1-yr percentile < 25 (historically cheap); short when > 75.",
    },
    {
        "id":         "storage_ttf",
        "label":      "EU Storage → TTF Bias",
        "category":   "Storage / Gas",
        "underlying": "ttf",
        "color":      "#3fb950",
        "source":     "D3 sig 2 / Tab 1",
        "description": "Low EU storage fill (< 25th pctile) → long TTF; high fill → short TTF.",
    },
    {
        "id":         "no2_nl_zscore",
        "label":      "NO2/NL Spread Z-Score",
        "category":   "Power Spread",
        "underlying": "no_nl_spread",
        "color":      "#f85149",
        "source":     "D3 sig 3 / Tab 12–13",
        "description": "Long NO2/Short NL when expanding OLS residual z < −1; reverse when z > +1.",
    },
    {
        "id":         "no2_ttf_resid",
        "label":      "NO2 vs TTF OLS Residual",
        "category":   "Power / Gas",
        "underlying": "no2",
        "color":      "#d2a8ff",
        "source":     "D3 sig 4 / Tab 2",
        "description": "Long NO2 when rolling-pctile of gas-power OLS residual < 25.",
    },
    {
        "id":         "hydro_ttf",
        "label":      "Norwegian Hydro → Power",
        "category":   "Supply",
        "underlying": "ttf",
        "color":      "#79c0ff",
        "source":     "D3 sig 6 / Tab 10",
        "description": "Low hydro reservoir (< 25th pctile) → long TTF/power; high → short.",
    },
    {
        "id":         "ttf_storage_resid",
        "label":      "TTF vs Storage OLS Residual",
        "category":   "Gas / Storage",
        "underlying": "ttf",
        "color":      "#d4ac3a",
        "source":     "D3 sig 7 / Tab 9",
        "description": "Long TTF when storage-OLS residual < 25th pctile (TTF cheap vs storage).",
    },
]

_SIG_BY_ID = {s["id"]: s for s in _SIGNALS}

# ── Stress scenarios ──────────────────────────────────────────────────────────
# Price impacts (EUR/MWh) for the named underlying in each scenario.
_STRESS = {
    "Cold snap": {
        "label":       "Cold snap — NO2 +€30/MWh, TTF +€10/MWh",
        "ttf":          +10.0,
        "no2":          +30.0,
        "no_nl_spread": +30.0,
    },
    "Norwegian outage": {
        "label":       "Norwegian outage — NO2 +€25/MWh, TTF +€5/MWh",
        "ttf":          +5.0,
        "no2":          +25.0,
        "no_nl_spread": +25.0,
    },
    "Hormuz extension": {
        "label":       "Hormuz extension — TTF +€15/MWh, NO2 +€10/MWh",
        "ttf":          +15.0,
        "no2":          +10.0,
        "no_nl_spread": +5.0,
    },
    "EUR/USD −10%": {
        "label":       "EUR/USD −10% — TTF −€8/MWh, NO2 −€5/MWh",
        "ttf":          -8.0,
        "no2":          -5.0,
        "no_nl_spread": -3.0,
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def _roll_pctile(series: pd.Series) -> pd.Series:
    """Empirical percentile of the current observation within a rolling 1-yr window."""
    return series.rolling(_ROLL_WINDOW, min_periods=_MIN_PERIODS).apply(
        lambda x: float((x < x[-1]).sum()) / len(x) * 100.0,
        raw=True,
    )


def _pctile_direction(pctile: pd.Series) -> pd.Series:
    """
    +1 when indicator below lower threshold, −1 when above upper, else 0.
    Lagged 1 day so yesterday's signal generates today's trade (no look-ahead).
    """
    d = pd.Series(0.0, index=pctile.index)
    d[pctile < _PCTILE_LO] = +1.0
    d[pctile > _PCTILE_HI] = -1.0
    return d.shift(1)


@st.cache_data(ttl=3600, show_spinner=False)
def _compute_risk_signals(feat: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Derive direction signals and underlying daily returns from the feature matrix.
    Returns (dirs, rets) with the same integer index as feat.
    Directions are lagged 1 day; returns are same-day absolute differences.
    """
    feat = feat.copy().reset_index(drop=True)

    # ── Underlying absolute returns ───────────────────────────────────────
    rets = pd.DataFrame({"date": feat["date"]})
    for col in ("ttf", "no2", "no_nl_spread"):
        if col in feat.columns:
            rets[col] = feat[col].diff()

    # ── Signal directions ─────────────────────────────────────────────────
    dirs = pd.DataFrame({"date": feat["date"]})

    # 1. TTF Seasonal — low TTF historically → buy (mean reversion)
    if "ttf" in feat.columns:
        dirs["ttf_seasonal"] = _pctile_direction(_roll_pctile(feat["ttf"]))

    # 2. Storage → TTF — low storage fill = tight market = buy TTF
    if "storage_fill" in feat.columns:
        dirs["storage_ttf"] = _pctile_direction(_roll_pctile(feat["storage_fill"]))

    # 3. NO2/NL Spread — expanding-window OLS z-score (same as D3 signal 3)
    if "no2" in feat.columns and "nl" in feat.columns:
        try:
            import statsmodels.api as _sm
            _f3 = feat[["no2", "nl"]].dropna()
            if len(_f3) >= 100:
                _ols3 = _sm.OLS(_f3["no2"].values, _sm.add_constant(_f3["nl"].values)).fit()
                _r3 = pd.Series(
                    np.asarray(_ols3.resid, dtype=float).copy(), index=_f3.index
                ).reindex(feat.index)
                _em3 = _r3.expanding(30).mean()
                _es3 = _r3.expanding(30).std()
                _z3  = (_r3 - _em3) / (_es3 + 1e-9)
                _d3  = pd.Series(0.0, index=feat.index)
                _d3[_z3 < -_ENTRY_Z] = +1.0
                _d3[_z3 > +_ENTRY_Z] = -1.0
                dirs["no2_nl_zscore"] = _d3.shift(1)
        except Exception:
            pass

    # 4. NO2 vs TTF OLS residual
    if "no2" in feat.columns and "ttf" in feat.columns:
        try:
            import statsmodels.api as _sm
            _f4 = feat[["no2", "ttf"]].dropna()
            if len(_f4) >= 60:
                _ols4 = _sm.OLS(_f4["no2"].values, _sm.add_constant(_f4["ttf"].values)).fit()
                _r4 = pd.Series(
                    np.asarray(_ols4.resid, dtype=float).copy(), index=_f4.index
                ).reindex(feat.index)
                dirs["no2_ttf_resid"] = _pctile_direction(_roll_pctile(_r4))
        except Exception:
            pass

    # 5. Hydro → TTF — low hydro = less supply = buy TTF
    if "hydro_pct" in feat.columns:
        dirs["hydro_ttf"] = _pctile_direction(_roll_pctile(feat["hydro_pct"]))

    # 6. TTF vs Storage OLS residual
    if "ttf" in feat.columns and "storage_fill" in feat.columns:
        try:
            import statsmodels.api as _sm
            _f6 = feat[["ttf", "storage_fill"]].dropna()
            if len(_f6) >= 60:
                _ols6 = _sm.OLS(_f6["ttf"].values, _sm.add_constant(_f6["storage_fill"].values)).fit()
                _r6 = pd.Series(
                    np.asarray(_ols6.resid, dtype=float).copy(), index=_f6.index
                ).reindex(feat.index)
                dirs["ttf_storage_resid"] = _pctile_direction(_roll_pctile(_r6))
        except Exception:
            pass

    return dirs, rets


def _portfolio_pnl(
    feat: pd.DataFrame,
    dirs: pd.DataFrame,
    rets: pd.DataFrame,
    notionals: dict,
) -> pd.DataFrame:
    """Daily P&L per signal and combined. Returns DataFrame aligned with feat."""
    pnl = pd.DataFrame({"date": feat["date"]})
    for sig in _SIGNALS:
        sid   = sig["id"]
        n     = notionals.get(sid, 0.0)
        undrl = sig["underlying"]
        if n == 0 or sid not in dirs.columns or undrl not in rets.columns:
            pnl[sid] = 0.0
        else:
            pnl[sid] = dirs[sid].fillna(0.0) * rets[undrl].fillna(0.0) * n
    pnl["combined"] = pnl[[s["id"] for s in _SIGNALS]].sum(axis=1)
    return pnl


def _var_es(series: pd.Series, conf: float = _VaR_CONF) -> tuple[float, float]:
    """Historical-simulation VaR and ES (non-parametric)."""
    arr = series.dropna().values
    if len(arr) < 20:
        return float("nan"), float("nan")
    cutoff = np.percentile(arr, (1.0 - conf) * 100)
    var = -cutoff
    tail = arr[arr < cutoff]
    es = -float(tail.mean()) if len(tail) > 0 else var
    return var, es


def _bootstrap_var_ci(series: pd.Series, conf: float = _VaR_CONF) -> tuple[float, float]:
    """95% CI for VaR via bootstrap resampling."""
    arr = series.dropna().values
    if len(arr) < 30:
        return float("nan"), float("nan")
    rng   = np.random.default_rng(42)
    boots = np.array([
        -np.percentile(rng.choice(arr, size=len(arr), replace=True), (1.0 - conf) * 100)
        for _ in range(_N_BOOT)
    ])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("## D4 · Risk Dashboard")
st.caption(
    "Portfolio simulator built on the D3 signal suite. "
    "Select signals and assign notional sizes to compute historical P&L, VaR/ES, and stress exposure."
)

st.markdown(
    """<div style="background:#1f1108;border-left:3px solid #d4ac3a;padding:10px 14px;
    border-radius:4px;color:#d4ac3a;font-size:0.84rem;margin-bottom:14px;">
    ⚠️ <strong>For analytical purposes only.</strong> Position P&amp;L is simulated from historical
    signal performance. Signals are derived from statistical models applied to public market data.
    Not financial advice. Simulated past performance is not indicative of future results.
    </div>""",
    unsafe_allow_html=True,
)

# ── Feature matrix ────────────────────────────────────────────────────────────
with st.spinner("Assembling feature matrix…"):
    _feat = assemble_features(years=3)

if _feat.empty or len(_feat) < 60:
    st.error(
        "Feature matrix unavailable or too short (< 60 rows). "
        "Visit Layer 2 (Quant Analysis) first to populate the feature cache, "
        "or configure ENTSO-E / AGSI API keys."
    )
    st.stop()

_feat = _feat.copy().reset_index(drop=True)

with st.spinner("Computing signal directions…"):
    _dirs, _rets = _compute_risk_signals(_feat)

_available_sigs = [s for s in _SIGNALS if s["id"] in _dirs.columns]
_unavailable    = [s["label"] for s in _SIGNALS if s["id"] not in _dirs.columns]

_feat_date_min = str(_feat["date"].min().date())
_feat_date_max = str(_feat["date"].max().date())
st.caption(
    f"Feature matrix: {len(_feat):,} trading days  ({_feat_date_min} → {_feat_date_max})  ·  "
    f"{len(_available_sigs)} of {len(_SIGNALS)} signals computable"
)

# ── Sidebar: portfolio builder ────────────────────────────────────────────────
st.sidebar.markdown("### Portfolio Builder")
st.sidebar.caption(
    "Notional = EUR gain per €1/MWh move in the underlying. "
    "All defaults are 0 (opt-in). Signals that lack the required feature data are hidden."
)

_notionals: dict[str, float] = {}
_active_sids: list[str]      = []

for _sig in _available_sigs:
    _sid = _sig["id"]
    st.sidebar.markdown(f"**{_sig['label']}**")
    st.sidebar.caption(_sig["description"])
    _n = st.sidebar.slider(
        f"Notional {_sid}",
        min_value=0, max_value=5000, step=100, value=0,
        label_visibility="collapsed",
        key=f"n_{_sid}",
    )
    _notionals[_sid] = float(_n)
    if _n > 0:
        _active_sids.append(_sid)
    st.sidebar.divider()

if _unavailable:
    st.sidebar.caption("Unavailable (missing data): " + ", ".join(_unavailable))

# Compute portfolio P&L
_pnl_df  = _portfolio_pnl(_feat, _dirs, _rets, _notionals)
_combined = _pnl_df["combined"].dropna()
_n_active = len(_active_sids)
_total_n  = sum(_notionals.values())

# ── Portfolio KPIs ────────────────────────────────────────────────────────────
_k1, _k2, _k3, _k4 = st.columns(4)
with _k1:
    st.markdown(
        kpi_card("Active signals", str(_n_active),
                 delta_span(f"{len(_available_sigs)} available", "green" if _n_active > 0 else "blue")),
        unsafe_allow_html=True,
    )
with _k2:
    st.markdown(
        kpi_card("Total notional", f"€{_total_n:,.0f}/MWh",
                 delta_span("EUR per €1/MWh move", "blue")),
        unsafe_allow_html=True,
    )
with _k3:
    if _n_active > 0 and len(_combined) > 0:
        _cum = float(_combined.sum())
        st.markdown(
            kpi_card("Simulated cum. P&L", f"€{_cum:+,.0f}",
                     delta_span("full history", "green" if _cum > 0 else "red")),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            kpi_card("Simulated cum. P&L", "—",
                     delta_span("set notionals to activate", "blue")),
            unsafe_allow_html=True,
        )
with _k4:
    if _n_active > 0 and len(_combined) >= 20:
        _var_kpi, _es_kpi = _var_es(_combined)
        if not np.isnan(_var_kpi):
            st.markdown(
                kpi_card("1-day 95% VaR", f"€{_var_kpi:,.0f}",
                         delta_span(f"ES: €{_es_kpi:,.0f}", "red")),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(kpi_card("1-day 95% VaR", "n/a", delta_span("insufficient history", "blue")),
                        unsafe_allow_html=True)
    else:
        st.markdown(kpi_card("1-day 95% VaR", "—", delta_span("set notionals to activate", "blue")),
                    unsafe_allow_html=True)

st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 1 — EQUITY CURVES
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("#### Equity Curves — Simulated Cumulative P&L")

if _n_active == 0:
    st.info("Set signal notionals in the sidebar to activate the portfolio simulator.", icon="ℹ️")
else:
    _fig_eq = go.Figure()

    for _sid in _active_sids:
        _sm = _SIG_BY_ID[_sid]
        _cum_sig = _pnl_df[_sid].fillna(0).cumsum()
        _fig_eq.add_trace(go.Scatter(
            x=_feat["date"], y=_cum_sig,
            name=_sm["label"],
            line=dict(color=_sm["color"], width=1.2, dash="dot"),
            opacity=0.7,
            hovertemplate=f"{_sm['label']}: €%{{y:,.0f}}<extra></extra>",
        ))

    _fig_eq.add_trace(go.Scatter(
        x=_feat["date"], y=_pnl_df["combined"].fillna(0).cumsum(),
        name="Combined portfolio",
        line=dict(color="#ffffff", width=2.5),
        hovertemplate="Combined: €%{y:,.0f}<extra></extra>",
    ))
    _fig_eq.add_hline(y=0, line=dict(color="rgba(255,255,255,0.2)", width=1))
    _fig_eq.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        height=320,
        margin=dict(l=60, r=20, t=10, b=35),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=10)),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(size=10, color="#8b949e")),
        yaxis=dict(title="Cum. P&L (EUR)", gridcolor="rgba(255,255,255,0.06)",
                   tickfont=dict(size=10, color="#8b949e"), zeroline=False),
        hovermode="x unified",
    )
    st.plotly_chart(_fig_eq, use_container_width=True)

    # Per-signal stats table
    _rows = []
    for _sid in _active_sids:
        _sp = _pnl_df[_sid].dropna()
        if len(_sp) < 10:
            continue
        _std = float(_sp.std())
        _sv, _se = _var_es(_sp)
        _sharpe = float(_sp.mean() / _std) * np.sqrt(252) if _std > 0 else 0.0
        _cum_v   = _sp.cumsum()
        _dd      = float((_cum_v - _cum_v.expanding().max()).min())
        _rows.append({
            "Signal":       _SIG_BY_ID[_sid]["label"],
            "Cum. P&L":     f"€{float(_sp.sum()):+,.0f}",
            "Daily σ":      f"€{_std:,.0f}",
            "Ann. Sharpe":  f"{_sharpe:+.2f}",
            "95% VaR":      f"€{_sv:,.0f}" if not np.isnan(_sv) else "n/a",
            "Max Drawdown": f"€{_dd:,.0f}",
        })
    if _rows:
        st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)

st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 — VaR / ES
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("#### Value at Risk / Expected Shortfall — Combined Portfolio")

if _n_active == 0 or len(_combined) < 30:
    st.caption("Activate at least one signal with notional > 0 (and ≥ 30 days of P&L history) to compute risk metrics.")
else:
    _v95, _e95 = _var_es(_combined, 0.95)
    _v99, _e99 = _var_es(_combined, 0.99)
    _vlo, _vhi = _bootstrap_var_ci(_combined, 0.95)

    _r1, _r2, _r3, _r4 = st.columns(4)
    with _r1:
        _ci_str = f"bootstrap CI: €{_vlo:,.0f}–€{_vhi:,.0f}" if not np.isnan(_vlo) else "bootstrap CI n/a"
        st.markdown(kpi_card("1-day VaR (95%)", f"€{_v95:,.0f}", delta_span(_ci_str, "red")),
                    unsafe_allow_html=True)
    with _r2:
        st.markdown(kpi_card("1-day ES (95%)", f"€{_e95:,.0f}",
                             delta_span("mean tail loss beyond VaR", "red")),
                    unsafe_allow_html=True)
    with _r3:
        st.markdown(kpi_card("1-day VaR (99%)", f"€{_v99:,.0f}",
                             delta_span("99th-pctile loss", "amber")),
                    unsafe_allow_html=True)
    with _r4:
        st.markdown(kpi_card("1-day ES (99%)", f"€{_e99:,.0f}",
                             delta_span("mean tail loss at 99%", "amber")),
                    unsafe_allow_html=True)

    # P&L distribution
    _arr = _combined.dropna().values
    _fig_dist = go.Figure()
    _fig_dist.add_trace(go.Histogram(
        x=_arr, nbinsx=60,
        marker_color="#58a6ff", marker_line_color="#0d1117", marker_line_width=0.4,
        opacity=0.75,
        hovertemplate="Range: %{x:.0f}<br>Count: %{y}<extra></extra>",
    ))
    _fig_dist.add_vline(x=-_v95, line=dict(color="#f85149", width=1.5, dash="dot"),
                        annotation_text=f"  VaR95 −€{_v95:,.0f}",
                        annotation_font=dict(color="#f85149", size=9),
                        annotation_position="top right")
    _fig_dist.add_vline(x=-_v99, line=dict(color="#d4ac3a", width=1.5, dash="dot"),
                        annotation_text=f"  VaR99 −€{_v99:,.0f}",
                        annotation_font=dict(color="#d4ac3a", size=9),
                        annotation_position="top right")
    _fig_dist.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        height=200,
        margin=dict(l=50, r=20, t=10, b=35),
        xaxis=dict(title="Daily P&L (EUR)", gridcolor="rgba(255,255,255,0.06)",
                   tickfont=dict(size=10, color="#8b949e")),
        yaxis=dict(title="Count", gridcolor="rgba(255,255,255,0.06)",
                   tickfont=dict(size=10, color="#8b949e")),
        showlegend=False, bargap=0.02,
    )
    st.plotly_chart(_fig_dist, use_container_width=True)

    _skew = float(pd.Series(_arr).skew())
    _kurt = float(pd.Series(_arr).kurtosis())
    st.caption(
        f"{len(_arr):,} daily observations · "
        f"Mean: €{float(_arr.mean()):+,.0f} · Std: €{float(_arr.std()):,.0f} · "
        f"Skew: {_skew:.2f} · Excess kurtosis: {_kurt:.2f}"
        + (" — fat tails present; parametric VaR would understate risk" if _kurt > 1.0 else "")
    )

st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3 — STRESS SCENARIOS
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("#### Stress Scenarios — Single-Day Impact Estimates")
st.caption(
    "Stressed P&L = current signal direction × scenario price move × notional. "
    "Price moves are calibrated from historical analogues. Current direction = last day in lagged series."
)

if _n_active == 0:
    st.caption("Activate signals in the sidebar to compute stress exposure.")
else:
    # Current directions (last row of lagged direction series)
    _cur: dict[str, float] = {}
    for _sid in _active_sids:
        _dseries = _dirs[_sid].dropna() if _sid in _dirs.columns else pd.Series(dtype=float)
        _cur[_sid] = float(_dseries.iloc[-1]) if len(_dseries) > 0 else 0.0

    _sc_cols = st.columns(2)
    for _ci, (_sc_name, _sc) in enumerate(_STRESS.items()):
        _sc_total = 0.0
        _breakdown = []
        for _sid in _active_sids:
            _undrl  = _SIG_BY_ID[_sid]["underlying"]
            _impact = _sc.get(_undrl, 0.0)
            _n      = _notionals.get(_sid, 0.0)
            _d      = _cur.get(_sid, 0.0)
            _pl     = _d * _impact * _n
            _sc_total += _pl
            if _n > 0 and abs(_pl) > 0.01:
                _breakdown.append(f"{_SIG_BY_ID[_sid]['label']}: €{_pl:+,.0f}")

        with _sc_cols[_ci % 2]:
            _arrow = "▼" if _sc_total < 0 else ("▲" if _sc_total > 0 else "—")
            _col   = "red" if _sc_total < 0 else ("green" if _sc_total > 0 else "blue")
            st.markdown(
                kpi_card(_sc_name, f"{_arrow} €{_sc_total:+,.0f}",
                         delta_span(_sc["label"], _col)),
                unsafe_allow_html=True,
            )
            if _breakdown:
                st.caption(" · ".join(_breakdown))

    with st.expander("Scenario price-impact assumptions", expanded=False):
        st.markdown("""
        | Scenario | Underlying moves | Historical analogue |
        |---|---|---|
        | Cold snap | NO2 +€30, TTF +€10 | February 2021 Beast-from-the-East II; NO2 intraday peak +€35 |
        | Norwegian outage | NO2 +€25, TTF +€5 | Unplanned Norwegian hydro transmission events 2022–23 |
        | Hormuz extension | TTF +€15, NO2 +€10 | March 2022 LNG disruption analogue; Hormuz closure risk premium on global LNG |
        | EUR/USD −10% | TTF −€8, NO2 −€5 | 2022 EUR depreciation 0.96–1.06; weaker EUR raises USD-hedger exit levels, softens quoted EUR TTF |

        Stress P&L is a static one-day estimate, not a path-dependent multi-period scenario.
        Signal direction used is the most recent day's lagged position.
        """)

st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4 — SIGNAL CORRELATION MATRIX
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("#### Signal Return Correlation Matrix (most recent 30 days)")
st.caption(
    "Pairwise Pearson correlation of unit-notional daily P&L contributions for all available signals. "
    "High correlation (|ρ| ≥ 0.6) reduces diversification benefit — two correlated signals "
    "increase nominal notional without proportionally increasing portfolio variance."
)

# Compute unit-notional returns for ALL available signals (not just active)
_unit = pd.DataFrame()
for _sig in _available_sigs:
    _sid   = _sig["id"]
    _undrl = _sig["underlying"]
    if _sid in _dirs.columns and _undrl in _rets.columns:
        _raw = (_dirs[_sid].fillna(0.0) * _rets[_undrl].fillna(0.0)).values
        _unit[_sig["label"]] = _raw

if _unit.empty or _unit.shape[1] < 2:
    st.caption("Requires ≥ 2 computable signals for the correlation matrix.")
else:
    _window_data = _unit.dropna()
    _window_data = _window_data.iloc[-30:] if len(_window_data) >= 30 else _window_data
    _corr = _window_data.corr()
    _labels = list(_corr.columns)

    _fig_hm = go.Figure(go.Heatmap(
        z=_corr.values,
        x=_labels, y=_labels,
        colorscale=[
            [0.0, "#1c4a8a"], [0.35, "#1c3060"],
            [0.5, "#161b22"],
            [0.65, "#5a1c1c"], [1.0, "#8a1c1c"],
        ],
        zmin=-1, zmax=1,
        text=np.round(_corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=10, color="#e6edf3"),
        hovertemplate="%{x} / %{y}: %{z:.2f}<extra></extra>",
        showscale=True,
        colorbar=dict(
            title=dict(text="Pearson ρ", font=dict(size=10, color="#8b949e")),
            tickfont=dict(size=9, color="#8b949e"),
            thickness=12,
        ),
    ))
    _fig_hm.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        height=max(260, 65 * len(_labels)),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(tickfont=dict(size=9, color="#8b949e"), tickangle=-30),
        yaxis=dict(tickfont=dict(size=9, color="#8b949e"), autorange="reversed"),
    )
    st.plotly_chart(_fig_hm, use_container_width=True)

    # Highlight highly correlated pairs
    _high: list[str] = []
    for _i in range(len(_labels)):
        for _j in range(_i + 1, len(_labels)):
            _c = float(_corr.values[_i][_j])
            if abs(_c) >= 0.6:
                _high.append(f"{_labels[_i]} / {_labels[_j]}: ρ = {_c:+.2f}")
    if _high:
        st.warning(
            "High correlations (|ρ| ≥ 0.6): " + " · ".join(_high)
            + ". Stacking these positions concentrates rather than diversifies risk.",
            icon="⚠️",
        )
    else:
        st.caption("No signal pairs with |ρ| ≥ 0.6 in the last 30 days.")

st.divider()

# ── Methodology ───────────────────────────────────────────────────────────────
with st.expander("Methodology", expanded=False):
    st.markdown(f"""
    **Signal direction:** Each signal generates a daily +1 (long), 0 (flat), or −1 (short) flag
    based on a rolling {_ROLL_WINDOW}-day ({_ROLL_WINDOW//21}-month) empirical percentile of the
    underlying indicator. Below {_PCTILE_LO:.0f}th percentile → long; above {_PCTILE_HI:.0f}th → short.
    The NO2/NL Spread uses an expanding-window OLS z-score (|z| > {_ENTRY_Z}) instead of percentile.
    All directions are **lagged 1 trading day** — yesterday's signal drives today's position (no look-ahead bias).

    **Notional:** EUR gain per €1/MWh move in the signal's underlying price.
    P&L[t] = direction[t−1] × ΔPrice[t] × notional. Absolute EUR/MWh differences are used, not log-returns.
    Slippage, transaction costs, and financing costs are not modelled.

    **VaR / ES:** Historical simulation (non-parametric, no distribution assumption).
    1-day {_VaR_CONF*100:.0f}% VaR = −(5th percentile of empirical daily P&L distribution).
    Expected Shortfall (ES / CVaR) = mean of all observations below the VaR cutoff.
    Bootstrap 95% CI uses {_N_BOOT:,} resamples. Fat-tailed distributions (excess kurtosis > 0)
    cause parametric (Gaussian) VaR to understate tail risk — this is why we use non-parametric estimation.

    **Stress scenarios:** Static one-day price impacts applied to current signal directions.
    Not path-dependent. Magnitude calibrated from historical intraday/daily moves during analogous events.

    **Correlation matrix:** 30-day rolling Pearson on unit-notional (notional=1) daily P&L contributions.
    Reflects recent co-movement — correlations are not constant.

    **Coverage gap:** Clean Spark Spread (D3 sig 5) and Marginal Fuel (D3 sig 8) are not backtested
    here — they require historical EUA price series and live ENTSO-E load data respectively.
    """)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    """<div style="color:#484f58;font-size:0.72rem;line-height:1.8;">
    Built by Tobias Gudbjerg &nbsp;|&nbsp;
    Feature data: ENTSO-E A44 (NO2, NL), ICE/Yahoo Finance (TTF), GIE AGSI+ (EU storage), ENTSO-E B31 (hydro)<br>
    For analytical purposes only. Simulated results do not represent live trading. Not financial advice.
    </div>""",
    unsafe_allow_html=True,
)
