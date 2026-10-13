"""
Layer 4: Machine Learning and Advanced Models
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="ML Models",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from utils.helpers import ordinal, apply_dark_theme, commentary, kpi_card, delta_span, has_torch
from utils.theme import ACCENTS, CHART, PLOTLY_TEMPLATE, page_header, page_footer
from data.sentiment import get_sentiment_data, FEED_SOURCES_TEXT
from data.prices import get_ttf_data

apply_dark_theme()

page_header(
    "Layer 4", "Machine Learning",
    "Machine learning applied to European power and gas market forecasting and regime classification.",
    accent=ACCENTS["l4"],
)

_TORCH_AVAILABLE = has_torch()
if not _TORCH_AVAILABLE:
    st.info(
        "PyTorch and Transformers are not installed in this environment, so the LSTM forecaster "
        "and the FinBERT sentiment model cannot run here. The regime model (HMM) runs without them. "
        "Architecture, feature specifications and training setup are shown below."
    )

from models.feature_assembly import (
    load_feature_matrix, get_feature_meta, get_available_feature_sets,
    get_lstm_feature_cols, get_hmm_feature_cols,
)
from models.lstm_model import (
    is_trained as lstm_is_trained, load_meta as lstm_load_meta,
    train_lstm_full, predict_next, load_test_results,
)
from models.hmm_model import (
    is_trained as hmm_is_trained, load_meta as hmm_load_meta,
    train_hmm, predict_regime, get_regime_history, REGIME_COLORS,
)
st.divider()

# ── Load feature matrix (cached) ─────────────────────────────────────────────
def _get_features():
    return load_feature_matrix(years=3)

import time as _time
_t0 = _time.perf_counter()
with st.spinner("Loading feature matrix…"):
    _features_df = _get_features()
_load_ms = (_time.perf_counter() - _t0) * 1000

_avail = get_available_feature_sets(_features_df)
_n_rows = len(_features_df)
_feat_meta = get_feature_meta()

# Top status banner
_status_parts = []
for label, key in [("Prices", "core"), ("Storage", "storage"), ("Hydro", "hydro"), ("Wind", "wind")]:
    status = "available" if _avail.get(key) else "unavailable"
    _status_parts.append(f"{label}: {status}")
_ready = _avail.get("core") and _n_rows >= 50
_load_tag = "served from cache" if _load_ms < 500 else "assembled now"
_span = (
    f", {_features_df['date'].min():%d %b %Y} to {_features_df['date'].max():%d %b %Y}"
    if _n_rows else ""
)

st.markdown(
    commentary(
        f"Feature matrix: {_n_rows} trading days{_span} ({_load_tag}, {_load_ms/1000:.1f}s). "
        f"{' | '.join(_status_parts)}. "
        + ("The models below are trained on this matrix; the Retrain buttons fit them again on the latest data."
           if _ready else
           "Too little data to train: day-ahead prices and TTF are required as a minimum."),
        "ok" if _ready else "warn",
    ),
    unsafe_allow_html=True,
)

st.divider()

# ── Model cards ───────────────────────────────────────────────────────────────

st.markdown("### Model 1: LSTM Day-Ahead Price Forecaster")

c1, c2 = st.columns([3, 2])
with c1:
    st.markdown("""
**Objective:** Predict next-day average day-ahead power price for NO2 (Kristiansand) and NL (Netherlands).

**Architecture:**
- Two-layer LSTM: 64 units → 32 units, with 0.2 dropout between layers
- Input window: 21 days of daily features (rolling)
- Output: scalar next-day average price (regression)
- Framework: PyTorch

**Input features (per day):**
| Feature | Source | Rationale |
|---|---|---|
| Day-ahead price (target zone) | Nord Pool | Autoregressive signal |
| TTF gas price | Yahoo Finance | Marginal cost driver |
| Nordic-Continental spread | Nord Pool | Cross-border arbitrage signal |
| Norwegian hydro reservoir level | ENTSO-E B31 | Supply capacity |
| Wind generation (DE + NO) | ENTSO-E B16 | Displacement of gas plant |
| Gas storage fill (EU) | GIE AGSI+ | Supply buffer tightness |
| Calendar features | Derived | Seasonality, weekday, holiday |

**Training setup:**
- Sample: 3 years of overlapping daily data (approximately 1,000 observations)
- Train/validation/test split: 70% / 15% / 15% (time-ordered, no shuffling)
- Loss: mean absolute error (MAE), robust to outliers vs MSE
- Optimiser: Adam, learning rate 1e-3 with ReduceLROnPlateau scheduler

**Evaluation:**
- Baseline: naive persistence forecast (yesterday's price = tomorrow's price)
- Target: MAE below naive baseline on out-of-sample test set
- Secondary: directional accuracy (did the model predict the correct sign of next-day change?)
    """)

with c2:
    if not _TORCH_AVAILABLE:
        _lstm_status_color, _lstm_status_label = "#8b949e", "Not available here (PyTorch not installed)"
    elif lstm_is_trained():
        _lstm_status_color = "#3fb950"
        _lstm_status_label = f"Trained {lstm_load_meta().get('trained_at', '')[:10]}".strip()
    elif _ready:
        _lstm_status_color, _lstm_status_label = "#d29922", "Not trained yet"
    else:
        _lstm_status_color, _lstm_status_label = "#f85149", "Awaiting data"
    _lstm_feat_list = get_lstm_feature_cols(_features_df) if not _features_df.empty else []
    _wind_note = "incl. wind" if _avail.get("wind") else "wind unavailable"
    st.markdown(
        f"""
<div style="background:#161b22;border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:16px;margin-bottom:8px;">
  <div style="color:#58a6ff;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;font-weight:600;margin-bottom:8px;">Status</div>
  <div style="color:{_lstm_status_color};font-size:0.88rem;margin-bottom:12px;">{_lstm_status_label}</div>
  <div style="color:#8b949e;font-size:0.82rem;line-height:1.7;">
    Training rows: {_n_rows}<br>
    Features: {len(_lstm_feat_list)} ({_wind_note})<br>
    Hydro (ENTSO-E B31): {"available" if _avail.get("hydro") else "unavailable"}<br>
    Wind (ENTSO-E B18/B19): {"available" if _avail.get("wind") else "unavailable"}
  </div>
</div>
<div style="background:#161b22;border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:16px;">
  <div style="color:#58a6ff;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;font-weight:600;margin-bottom:8px;">Why This Matters</div>
  <div style="color:#8b949e;font-size:0.82rem;line-height:1.7;">
    A trader with a directionally accurate price signal (even just 55% correct)
    can size positions on intraday vs day-ahead spreads. The model is not intended to
    replace fundamental analysis but to quantify the marginal information content
    of the input features above a naive benchmark.
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

# ── LSTM live section ─────────────────────────────────────────────────────────
with st.expander("Live Model 1: Train and Predict", expanded=lstm_is_trained() and _TORCH_AVAILABLE):
    if not _TORCH_AVAILABLE:
        st.caption("The LSTM needs PyTorch, which is not installed in this environment.")
    elif not _ready:
        st.warning("Insufficient data to train. Need at least Nord Pool prices + TTF (50+ days).")
    else:
        _lstm_feat_cols = get_lstm_feature_cols(_features_df)
        st.caption(
            f"Training features ({len(_lstm_feat_cols)}): {', '.join(_lstm_feat_cols)}. "
            f"Data: {_n_rows} days. "
            + ("Wind generation excluded (no wind data available)."
               if not _avail.get("wind") else "Wind generation included.")
        )

        if lstm_is_trained():
            _lm = lstm_load_meta()
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.markdown(kpi_card("NO2 test MAE", f"€{_lm.get('test_mae_no2', '?')}/MWh",
                                     delta_span(f"naive: €{_lm.get('naive_mae_no2','?')}/MWh",
                                                "green" if _lm.get('test_mae_no2',99) < _lm.get('naive_mae_no2',0) else "amber")),
                             unsafe_allow_html=True)
            with col_b:
                st.markdown(kpi_card("NL test MAE", f"€{_lm.get('test_mae_nl','?')}/MWh",
                                     delta_span(f"naive: €{_lm.get('naive_mae_nl','?')}/MWh",
                                                "green" if _lm.get('test_mae_nl',99) < _lm.get('naive_mae_nl',0) else "amber")),
                             unsafe_allow_html=True)
            with col_c:
                st.markdown(kpi_card("Training observations", str(_lm.get("n_train","?")),
                                     delta_span(f"val: {_lm.get('n_val','?')} · test: {_lm.get('n_test','?')}", "blue")),
                             unsafe_allow_html=True)
            with col_d:
                trained_at = _lm.get("trained_at", "")[:10]
                st.markdown(kpi_card("Trained", trained_at,
                                     delta_span(f"{len(_lm.get('feature_cols',[]))} features", "blue")),
                             unsafe_allow_html=True)

            # Honesty banner: warn when LSTM underperforms naive baseline
            _no2_mae   = _lm.get("test_mae_no2")
            _no2_naive = _lm.get("naive_mae_no2")
            if _no2_mae is not None and _no2_naive is not None:
                try:
                    _mae_f, _naive_f = float(_no2_mae), float(_no2_naive)
                    if _mae_f > _naive_f:
                        _tw = load_test_results()
                        _tw_text = (
                            f" over the test window {_tw['date'].min():%d %b %Y} to {_tw['date'].max():%d %b %Y}"
                            if not _tw.empty else ""
                        )
                        st.warning(
                            f"**Model limitation:** the LSTM's NO2 test MAE (€{_mae_f:.2f}/MWh) is worse than "
                            f"the naive persistence baseline (€{_naive_f:.2f}/MWh){_tw_text}. Its predictions "
                            "revert toward the mean and should not be used for trading decisions. Candidate "
                            "improvements: a return target instead of the price level, a quantile loss and "
                            "walk-forward validation.",
                            icon="⚠️",
                        )
                except (TypeError, ValueError):
                    pass

            # Live prediction
            _pred = predict_next(_features_df)
            if _pred and "no2_pred" in _pred and _pred["no2_pred"] is not None:
                if "note" not in _pred:
                    st.markdown(
                        commentary(
                            f"Next-day forecast (as of {_pred['last_date']}): "
                            f"NO2: **€{_pred['no2_pred']:.1f}/MWh** | "
                            f"NL: **€{_pred['nl_pred']:.1f}/MWh**. "
                            "Directional signal only; not a substitute for fundamental analysis.",
                            "ok",
                        ),
                        unsafe_allow_html=True,
                    )
                else:
                    st.info(_pred.get("note", "Retrain model for live predictions."))

        # Actual vs predicted chart
        _test_df = load_test_results()
        if not _test_df.empty:
            _fig_pred = go.Figure()
            _fig_pred.add_trace(go.Scatter(
                x=_test_df["date"], y=_test_df["no2_actual"],
                name="NO2 actual", line=dict(color="#4cc9f0", width=1.5),
                hovertemplate="NO2 actual: €%{y:.1f}<extra></extra>",
            ))
            _fig_pred.add_trace(go.Scatter(
                x=_test_df["date"], y=_test_df["no2_pred"],
                name="NO2 predicted", line=dict(color="#4cc9f0", width=1.5, dash="dot"),
                hovertemplate="NO2 predicted: €%{y:.1f}<extra></extra>",
            ))
            _fig_pred.add_trace(go.Scatter(
                x=_test_df["date"], y=_test_df["nl_actual"],
                name="NL actual", line=dict(color="#f4a261", width=1.5),
                hovertemplate="NL actual: €%{y:.1f}<extra></extra>",
            ))
            _fig_pred.add_trace(go.Scatter(
                x=_test_df["date"], y=_test_df["nl_pred"],
                name="NL predicted", line=dict(color="#f4a261", width=1.5, dash="dot"),
                hovertemplate="NL predicted: €%{y:.1f}<extra></extra>",
            ))
            _fig_pred.update_layout(
                template=PLOTLY_TEMPLATE, paper_bgcolor=CHART["bg"], plot_bgcolor=CHART["panel_bg"],
                font=dict(color=CHART["text"], size=12),
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(showgrid=True, gridcolor=CHART["grid"]),
                yaxis=dict(title="EUR/MWh", showgrid=True, gridcolor=CHART["grid"]),
                hovermode="x unified",
            )
            st.markdown("**Test set: actual vs predicted**")
            st.plotly_chart(_fig_pred, width="stretch")
            st.caption(
                f"Solid lines: actual day-ahead prices. Dotted lines: LSTM predictions. "
                f"Test period: {_test_df['date'].min():%d %b %Y} to {_test_df['date'].max():%d %b %Y} "
                f"({len(_test_df)} trading days, out of sample). Large spikes are systematically "
                f"underestimated, a known limitation of sequence models trained on mean absolute error."
            )

        # Train button
        _btn_label = "Retrain Model 1 (LSTM)" if lstm_is_trained() else "Train Model 1 (LSTM)"
        if _TORCH_AVAILABLE and st.button(_btn_label, key="train_lstm"):
            progress_bar = st.progress(0)
            status_text  = st.empty()
            def _lstm_cb(epoch, total, tl, vl):
                pct = int(epoch / total * 100)
                progress_bar.progress(pct)
                status_text.caption(f"Epoch {epoch}/{total}: train MAE {tl:.4f}, val MAE {vl:.4f}")
            with st.spinner("Training LSTM… (this takes 1-3 minutes on CPU)"):
                result = train_lstm_full(_features_df, _lstm_feat_cols, progress_cb=_lstm_cb)
            progress_bar.empty()
            status_text.empty()
            if "error" in result:
                st.error(f"Training failed: {result['error']}")
            else:
                st.success(
                    f"Training complete. Test MAE: NO2 €{result['test_mae_no2']}/MWh "
                    f"(naive: €{result['naive_mae_no2']}/MWh) · "
                    f"NL: €{result['test_mae_nl']}/MWh (naive: €{result['naive_mae_nl']}/MWh). "
                    f"Trained on {result['n_train']} sequences."
                )
                st.cache_data.clear()
                st.rerun()

st.divider()

st.markdown("### Model 2: Hidden Markov Model Market Regime Classifier")

c3, c4 = st.columns([3, 2])
with c3:
    st.markdown("""
**Objective:** Classify the European power market into one of four structural regimes, updated daily.

**Regimes:**
| Regime | Characteristics | Typical price signal |
|---|---|---|
| Hydro-driven | High Norwegian reservoir levels, strong exports, low Nordic prices | NO2 at significant discount to NL |
| Gas-driven | Gas prices dominant, thermal plant on the margin, moderate hydro | NL closely tracks TTF (high R²) |
| Wind-driven | High wind penetration DE/SE, renewables suppressing midday prices | Intraday price spread very wide; solar cannibalisation effect strong |
| Geopolitical stress | Low storage, tight LNG supply, supply chain disruption | All zones elevated; TTF spike correlation high |

**Architecture:**
- Gaussian HMM with 4 hidden states (hmmlearn)
- Observation sequence: daily multivariate feature vector
- Emission model: multivariate Gaussian per state

**Input features (observation vector):**
- Price level (NO2 and NL, z-scored relative to 90-day mean)
- Price volatility (5-day rolling standard deviation)
- TTF gas price (z-scored)
- Gas storage deficit (current vs 5-year average for date, pp)
- Norwegian hydro reservoir percentile (0-100)
- Wind generation as % of load (Germany)
- Nordic-Continental spread (EUR/MWh)

**Validation:**
- State labels are inferred (unsupervised). Post-hoc regime labelling via inspection of state centroids.
- Stability check: expected regime duration (inverse of self-transition probability) should exceed 5 trading days
- Sanity check: Hydro-driven regime should concentrate in late spring / early summer; Gas-driven in autumn/winter
    """)

with c4:
    if hmm_is_trained():
        _hmm_status_color = "#3fb950"
        _hmm_status_label = f"Trained {hmm_load_meta().get('trained_at', '')[:10]}".strip()
    elif _ready:
        _hmm_status_color, _hmm_status_label = "#d29922", "Not trained yet"
    else:
        _hmm_status_color, _hmm_status_label = "#f85149", "Awaiting data"
    _hmm_feat_list = get_hmm_feature_cols(_features_df) if not _features_df.empty else []
    st.markdown(
        f"""
<div style="background:#161b22;border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:16px;margin-bottom:8px;">
  <div style="color:#58a6ff;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;font-weight:600;margin-bottom:8px;">Status</div>
  <div style="color:{_hmm_status_color};font-size:0.88rem;margin-bottom:12px;">{_hmm_status_label}</div>
  <div style="color:#8b949e;font-size:0.82rem;line-height:1.7;">
    Training rows: {_n_rows}<br>
    HMM features: {len(_hmm_feat_list)}<br>
    Hydro percentile: {"available" if _avail.get("hydro") else "unavailable"}<br>
    Wind z-score: {"available" if _avail.get("wind") else "unavailable"}
  </div>
</div>
<div style="background:#161b22;border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:16px;">
  <div style="color:#58a6ff;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;font-weight:600;margin-bottom:8px;">Why This Matters</div>
  <div style="color:#8b949e;font-size:0.82rem;line-height:1.7;">
    Physical traders need to understand which fundamental factor is driving the market
    on any given day. A regime signal compresses multiple data sources into a single
    actionable label, helping a trader decide whether to prioritise hydro, gas,
    or weather signals when constructing a view.
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

# ── HMM live section ──────────────────────────────────────────────────────────
with st.expander("Live Model 2: Train and Classify", expanded=hmm_is_trained()):
    if not _ready:
        st.warning("Insufficient data. Need at least Nord Pool prices + TTF (100+ days).")
    else:
        _hmm_feat_cols = get_hmm_feature_cols(_features_df)
        st.caption(
            f"HMM features ({len(_hmm_feat_cols)}): {', '.join(_hmm_feat_cols)}. "
            f"Data: {_n_rows} days."
        )

        if hmm_is_trained():
            _hm = hmm_load_meta()

            # Current regime
            _reg = predict_regime(_features_df)
            if _reg:
                reg_color = _reg["regime_color"]
                st.markdown(
                    f"""<div style="background:#161b22;border:2px solid {reg_color};border-radius:8px;
                    padding:14px 18px;margin-bottom:12px;display:inline-block;">
                    <span style="color:{reg_color};font-size:1.1rem;font-weight:700;">{_reg["regime"]}</span>
                    <span style="color:#8b949e;font-size:0.82rem;margin-left:12px;">
                    {_reg["confidence"]:.0f}% confidence · {_reg["latest_date"]}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

                # Regime history chart
                _hist_df = get_regime_history(_features_df)
                if not _hist_df.empty:
                    _C = {"bg": CHART["bg"], "panel_bg": CHART["panel_bg"], "grid": CHART["grid"], "text": CHART["text"]}
                    _fig_r = go.Figure()
                    for _regime, _grp in _hist_df.groupby("regime"):
                        _color = REGIME_COLORS.get(_regime, "#8b949e")
                        _fig_r.add_trace(go.Scatter(
                            x=_grp["date"], y=[_regime] * len(_grp),
                            mode="markers",
                            marker=dict(color=_color, size=6, symbol="square"),
                            name=_regime,
                            hovertemplate=f"<b>{_regime}</b><br>%{{x|%Y-%m-%d}}<extra></extra>",
                        ))
                    _fig_r.update_layout(
                        template=PLOTLY_TEMPLATE, paper_bgcolor=_C["bg"], plot_bgcolor=_C["panel_bg"],
                        font=dict(color=_C["text"], size=11),
                        margin=dict(l=10, r=10, t=10, b=10), height=200,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        xaxis=dict(showgrid=True, gridcolor=_C["grid"], title=None),
                        yaxis=dict(showgrid=False, title=None),
                        hovermode="x unified",
                    )
                    st.plotly_chart(_fig_r, width="stretch")

                    # ── 3D: realised-volatility overlay on the regime timeline (roadmap 3D) ──
                    # Pure post-processing in models/regime_volatility.py; uses the decoded
                    # history (_hist_df) and the NO2 column of the feature matrix.
                    from models.regime_volatility import (
                        realised_volatility as _rv_calc,
                        volatility_by_regime as _rv_by_regime,
                        transition_spike_proximity as _rv_proximity,
                    )
                    from config.settings import (
                        REGIME_VOL_WINDOW as _RV_WIN,
                        REGIME_VOL_SPIKE_PCTL as _RV_PCTL,
                        REGIME_VOL_PROXIMITY_DAYS as _RV_PROX,
                        REGIME_VOL_MIN_DAYS as _RV_MIN,
                    )

                    st.markdown("#### Realised Volatility Overlay: NO2 Day-Ahead")
                    st.caption(
                        f"Rolling {_RV_WIN}-trading-day realised volatility of NO2 day-ahead prices (standard "
                        "deviation of daily price changes, EUR/MWh, right axis) over the decoded regime timeline. "
                        f"Diamonds mark regime transitions; the dotted line is the {ordinal(_RV_PCTL)}-percentile "
                        "spike threshold."
                    )
                    _rv_has_no2 = (
                        "no2" in _features_df.columns
                        and int(_features_df["no2"].notna().sum()) > _RV_WIN
                    )
                    if not _rv_has_no2:
                        st.caption(
                            "Volatility overlay unavailable: the feature matrix holds no NO2 day-ahead prices "
                            f"(or fewer than {_RV_WIN + 1} days of them)."
                        )
                    else:
                        _rv_src = _features_df[["date", "no2"]].dropna().copy()
                        _rv_src["date"] = pd.to_datetime(_rv_src["date"]).dt.normalize()
                        _rv_df = _rv_calc(_rv_src.set_index("date")["no2"], window=_RV_WIN, method="abs")
                        _rv_hist = _hist_df[["date", "regime"]].copy()
                        _rv_hist["date"] = pd.to_datetime(_rv_hist["date"]).dt.normalize()
                        _rv_stats = _rv_proximity(_rv_hist, _rv_df, spike_pctl=_RV_PCTL, window=_RV_PROX)
                        _rv_tbl = _rv_by_regime(_rv_hist, _rv_df)

                        if not _rv_stats or _rv_stats["n_days"] < _RV_MIN or _rv_tbl.empty:
                            st.caption(
                                "Volatility overlay unavailable: fewer than "
                                f"{_RV_MIN} regime days with a defined {_RV_WIN}-day realised volatility."
                            )
                        else:
                            _rv_plot = _rv_hist.merge(_rv_df[["date", "rv"]], on="date", how="left")
                            _rv_valid = _rv_plot["rv"].dropna()
                            _rv_now = float(_rv_valid.iloc[-1])
                            _rv_now_pct = float((_rv_valid <= _rv_now).mean() * 100)
                            _rv_thr = _rv_stats["threshold"]

                            _fig_rv = go.Figure()
                            _rv_order = [r for r in REGIME_COLORS if r in set(_rv_plot["regime"])]
                            for _regime in _rv_order:
                                _grp = _rv_plot[_rv_plot["regime"] == _regime]
                                _fig_rv.add_trace(go.Scatter(
                                    x=_grp["date"], y=[_regime] * len(_grp),
                                    mode="markers", name=_regime,
                                    marker=dict(color=REGIME_COLORS.get(_regime, "#8b949e"), size=6, symbol="square"),
                                    hovertemplate=f"<b>{_regime}</b><extra></extra>",
                                ))
                            _fig_rv.add_trace(go.Scatter(
                                x=_rv_plot["date"], y=_rv_plot["rv"], yaxis="y2", mode="lines",
                                name=f"{_RV_WIN}-day realised vol",
                                line=dict(color="#e6edf3", width=1.4),
                                hovertemplate="Realised vol: €%{y:.2f}/MWh<extra></extra>",
                            ))
                            _fig_rv.add_trace(go.Scatter(
                                x=[_rv_plot["date"].min(), _rv_plot["date"].max()], y=[_rv_thr, _rv_thr],
                                yaxis="y2", mode="lines", name=f"P{_RV_PCTL:.0f} spike threshold",
                                line=dict(color="rgba(255,255,255,0.45)", width=1, dash="dot"),
                                hoverinfo="skip",
                            ))
                            _rv_tr = _rv_plot[_rv_plot["date"].isin(_rv_stats["transition_dates"])]
                            if not _rv_tr.empty:
                                _fig_rv.add_trace(go.Scatter(
                                    x=_rv_tr["date"], y=_rv_tr["rv"], yaxis="y2", mode="markers",
                                    name="Regime transition",
                                    marker=dict(
                                        symbol="diamond", size=9,
                                        color=[REGIME_COLORS.get(r, "#8b949e") for r in _rv_tr["regime"]],
                                        line=dict(color="#ffffff", width=1),
                                    ),
                                    customdata=_rv_tr[["regime"]].to_numpy(),
                                    hovertemplate="Transition to %{customdata[0]}<extra></extra>",
                                ))
                            _fig_rv.update_layout(
                                template="plotly_dark", paper_bgcolor=_C["bg"], plot_bgcolor=_C["panel_bg"],
                                font=dict(color=_C["text"], size=11),
                                margin=dict(l=10, r=10, t=10, b=10), height=330,
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                xaxis=dict(showgrid=True, gridcolor=_C["grid"], title=None),
                                yaxis=dict(showgrid=False, title=None,
                                           categoryorder="array", categoryarray=_rv_order),
                                yaxis2=dict(title="Realised vol (EUR/MWh)", overlaying="y", side="right",
                                            showgrid=True, gridcolor=_C["grid"], rangemode="tozero"),
                                hovermode="x unified",
                            )
                            st.plotly_chart(_fig_rv, width="stretch")

                            # KPI row
                            _rvk1, _rvk2, _rvk3, _rvk4 = st.columns(4)
                            with _rvk1:
                                st.markdown(kpi_card(
                                    f"{_RV_WIN}-day realised vol (latest)", f"€{_rv_now:.2f}/MWh",
                                    delta_span(f"{ordinal(_rv_now_pct)} percentile of sample · spike above €{_rv_thr:.2f}",
                                               "red" if _rv_now > _rv_thr else "blue"),
                                ), unsafe_allow_html=True)
                            _rv_n_tr = _rv_stats["n_transitions"]
                            with _rvk2:
                                st.markdown(kpi_card(
                                    "Transitions near a vol spike",
                                    f"{_rv_stats['n_near']} of {_rv_n_tr}" if _rv_n_tr else "n/a",
                                    delta_span(
                                        f"{_rv_stats['observed_share']:.0%} within ±{_RV_PROX} trading days"
                                        if _rv_n_tr else "no transitions in the scored window", "blue"),
                                ), unsafe_allow_html=True)
                            with _rvk3:
                                st.markdown(kpi_card(
                                    "Chance baseline", f"{_rv_stats['baseline_share']:.0%}",
                                    delta_span(f"share of days within ±{_RV_PROX}d of a spike", "blue"),
                                ), unsafe_allow_html=True)
                            with _rvk4:
                                _rv_p = _rv_stats["p_value"]
                                _rv_lift_ok = _rv_n_tr > 0 and np.isfinite(_rv_stats["lift"])
                                st.markdown(kpi_card(
                                    "Lift vs chance",
                                    f"{_rv_stats['lift']:.2f}×" if _rv_lift_ok else "n/a",
                                    delta_span(
                                        f"circular-shift p = {_rv_p:.3f}" if np.isfinite(_rv_p) else "p-value n/a",
                                        "amber" if (np.isfinite(_rv_p) and _rv_p < 0.05) else "blue"),
                                ), unsafe_allow_html=True)

                            # Per-regime table
                            _rv_rows = "".join(
                                "<tr style='border-bottom:1px solid #21262d'>"
                                f"<td style='padding:5px 10px;font-weight:600;color:{REGIME_COLORS.get(_r.regime, '#8b949e')}'>{_r.regime}</td>"
                                f"<td style='padding:5px 10px;text-align:right;color:#8b949e'>{int(_r.days):,}</td>"
                                f"<td style='padding:5px 10px;text-align:right;color:#8b949e'>{_r.share:.0%}</td>"
                                f"<td style='padding:5px 10px;text-align:right;color:#e6edf3'>€{_r.mean_rv:.2f}</td>"
                                f"<td style='padding:5px 10px;text-align:right;color:#e6edf3'>€{_r.median_rv:.2f}</td>"
                                f"<td style='padding:5px 10px;text-align:right;color:#8b949e'>€{_r.mean_price:.1f}</td>"
                                "</tr>"
                                for _r in _rv_tbl.itertuples()
                            )
                            st.markdown(
                                "<table style='width:100%;border-collapse:collapse;font-size:0.85rem;margin-top:6px'>"
                                "<thead><tr style='border-bottom:2px solid #30363d;color:#8b949e;text-align:left'>"
                                "<th style='padding:6px 10px'>Regime</th>"
                                "<th style='padding:6px 10px;text-align:right'>Days</th>"
                                "<th style='padding:6px 10px;text-align:right'>Share</th>"
                                f"<th style='padding:6px 10px;text-align:right'>Mean {_RV_WIN}d vol (EUR/MWh)</th>"
                                f"<th style='padding:6px 10px;text-align:right'>Median {_RV_WIN}d vol (EUR/MWh)</th>"
                                "<th style='padding:6px 10px;text-align:right'>Mean NO2 (EUR/MWh)</th>"
                                f"</tr></thead><tbody>{_rv_rows}</tbody></table>",
                                unsafe_allow_html=True,
                            )

                            # Commentary: computed from the overlay statistics
                            _rv_top, _rv_bot = _rv_tbl.iloc[0], _rv_tbl.iloc[-1]
                            if len(_rv_tbl) > 1:
                                _rv_text = (
                                    f"Across {_rv_stats['n_days']} trading days with a defined {_RV_WIN}-day "
                                    f"realised volatility, the {_rv_top.regime} regime was the most volatile "
                                    f"(mean €{_rv_top.mean_rv:.2f}/MWh, median €{_rv_top.median_rv:.2f}/MWh) and "
                                    f"{_rv_bot.regime} the least (mean €{_rv_bot.mean_rv:.2f}/MWh). "
                                )
                            else:
                                _rv_text = (
                                    f"All {_rv_stats['n_days']} scored trading days fall in the {_rv_top.regime} "
                                    f"regime (mean realised volatility €{_rv_top.mean_rv:.2f}/MWh). "
                                )
                            if _rv_n_tr == 0:
                                _rv_text += "No regime transitions occur on the scored days."
                            else:
                                _rv_text += (
                                    f"Of {_rv_n_tr} regime transitions, {_rv_stats['n_near']} "
                                    f"({_rv_stats['observed_share']:.0%}) fell within ±{_RV_PROX} trading days of a "
                                    f"volatility spike (realised volatility above its {ordinal(_RV_PCTL)} percentile, "
                                    f"€{_rv_thr:.2f}/MWh), against a chance baseline of {_rv_stats['baseline_share']:.0%}. "
                                )
                                if np.isfinite(_rv_p) and _rv_p < 0.05:
                                    _rv_text += (
                                        f"The excess is statistically significant (circular-shift p = {_rv_p:.3f}): "
                                        "regime changes tend to coincide with volatility shocks."
                                    )
                                elif _rv_stats["observed_share"] > _rv_stats["baseline_share"]:
                                    _rv_text += (
                                        "The excess over chance is not statistically significant "
                                        f"(circular-shift p = {_rv_p:.3f})."
                                    )
                                else:
                                    _rv_text += (
                                        "Transitions are no closer to volatility spikes than chance would imply "
                                        f"(circular-shift p = {_rv_p:.3f})."
                                    )
                            st.markdown(
                                commentary(_rv_text, "warn" if (np.isfinite(_rv_p) and _rv_p < 0.05) else "ok"),
                                unsafe_allow_html=True,
                            )

                            with st.expander("Volatility overlay: methodology", expanded=False):
                                st.markdown(f"""
                                **Realised volatility:** rolling sample standard deviation (ddof = 1) of daily NO2
                                day-ahead price changes, P(t) minus P(t-1) in EUR/MWh, over the last {_RV_WIN}
                                trading days (a full window is required). The feature matrix is indexed on TTF trading
                                days, so a Monday change spans the weekend auctions.

                                **Why absolute rather than log changes:** Nordic day-ahead prices can approach zero and
                                occasionally turn negative, which makes log changes undefined or explosive (a move from
                                €1 to €5/MWh is a 161% log change). The HMM itself measures volatility in EUR/MWh
                                (5-day standard deviation of NO2), so the overlay is on the same scale. Absolute
                                volatility scales with the price level, hence the mean NO2 price column in the table.
                                A log-change variant with non-positive prices excluded is implemented in
                                `models/regime_volatility.py` (method "log").

                                **Spikes and transitions:** a spike is a day with realised volatility above its
                                in-sample {ordinal(_RV_PCTL)} percentile (€{_rv_thr:.2f}/MWh; {_rv_stats['n_spike_days']} days).
                                A transition is a day whose Viterbi-decoded regime differs from the previous day's.
                                Only days with a defined volatility are scored.

                                **Chance baseline:** the share of scored days lying within ±{_RV_PROX} trading days of
                                at least one spike day, i.e. the probability that a transition placed on a random scored
                                day would count as near a spike. Lift = observed share divided by baseline.

                                **Significance:** exact one-sided circular-shift test. The transition sequence is shifted
                                cyclically against the fixed spike sequence by every lag of at least {2 * _RV_PROX + 1}
                                days (shorter lags leave a transition inside its own neighbourhood), and
                                p = (1 + shifts with at least the observed number of near-spike transitions) / (1 + shifts).
                                Cyclic shifting preserves the clustering of both sequences.

                                **Limitations:** the HMM observation vector includes a 5-day NO2 volatility z-score, so
                                some association between regimes and volatility levels is built in; the transition test
                                concerns timing. Viterbi decoding uses the full sample, so regime boundaries are smoothed
                                with hindsight, and the spike threshold is in-sample. The statistics are descriptive and
                                are not a trading signal.
                                """)
                    # ── end 3D: realised-volatility overlay ─────────────────────────────────

            # ── 3D: explicit notice when the trained HMM cannot be applied ──────
            if not _reg:
                st.caption(
                    "Regime classification unavailable: the saved model could not be applied to the "
                    "current feature matrix (it needs "
                    f"{', '.join(_hm.get('feature_cols', [])) or 'its training features'}). Retrain the "
                    "model to classify the latest data."
                )
            # ── end 3D notice ───────────────────────────────────────────────────

            # Regime statistics
            if _hm.get("state_pcts"):
                st.caption("Historical regime distribution: " +
                           " · ".join(f"{k}: {v}%" for k, v in _hm["state_pcts"].items()))
            if _hm.get("state_durations"):
                st.caption("Expected regime duration (trading days): " +
                           " · ".join(f"{k}: {v:.0f}d" for k, v in _hm["state_durations"].items()))

        # Train button
        _btn_label2 = "Retrain Model 2 (HMM)" if hmm_is_trained() else "Train Model 2 (HMM)"
        if st.button(_btn_label2, key="train_hmm"):
            with st.spinner("Fitting HMM (typically <30 seconds)…"):
                result = train_hmm(_features_df, _hmm_feat_cols)
            if "error" in result:
                st.error(f"Training failed: {result['error']}")
            else:
                regimes_str = " · ".join(f"{v}: {result['state_pcts'].get(v,'?')}%" for v in result.get("state_pcts", {}).keys())
                st.success(
                    f"HMM trained on {result['n_obs']} observations. "
                    f"Regime distribution: {regimes_str}."
                )
                st.cache_data.clear()
                st.rerun()

st.divider()

st.markdown("### Model 3: NLP Energy News Sentiment Signal")

# Exploratory badge
st.markdown(
    """<div style="display:inline-block;background:rgba(210,153,34,0.15);border:1px solid rgba(210,153,34,0.4);
    border-radius:6px;padding:6px 14px;margin-bottom:14px;">
    <span style="color:#d29922;font-size:0.78rem;font-weight:600;text-transform:uppercase;letter-spacing:1px;">
    Exploratory: Live Prototype</span>
    <span style="color:#8b949e;font-size:0.78rem;margin-left:10px;">
    Pipeline is running. Scoring builds up over time as headlines accumulate.</span>
    </div>""",
    unsafe_allow_html=True,
)

c5, c6 = st.columns([3, 2])
with c5:
    st.markdown("""
**Objective:** Generate a daily sentiment score for European energy market news and measure its
predictive power for next-day TTF price changes.

**Architecture:**
- **Model:** FinBERT (`ProsusAI/finbert`): BERT fine-tuned on financial text, classifies
  each headline as positive / negative / neutral with a confidence score
- **Input:** Energy-relevant headlines filtered from six RSS feeds (BBC Business, Guardian Energy,
  LNG World News, Energy Monitor, Power Technology, Offshore Technology); the archives of the last
  three are paged back 60 days when the stored history is too short
- **Processing:** Each headline is scored. Daily aggregation:
  net sentiment = mean(positive conf.) − mean(negative conf.) ∈ [−1, +1]

**Pipeline:**
1. Fetch RSS headlines (6 feeds, cached 6h)
2. Filter by energy keywords: LNG, TTF, Hormuz, Norwegian gas, pipeline, carbon…
3. Classify each headline with FinBERT (CPU inference, batches of 16)
4. Aggregate to daily net sentiment score
5. Compute rolling 3d and 7d moving averages
6. Compute rolling cross-correlation with next-day TTF % change (requires ≥21 days)

**Validation target (once history builds):**
- Granger causality: does lagged sentiment Granger-cause TTF daily returns?
- IC: rolling 30-day Spearman correlation of sentiment with next-day return
- Expected IC range: 0.05–0.15 (small but meaningful in geopolitical stress regimes)

**Limitations:**
- FinBERT is fine-tuned on Reuters/Bloomberg equity and macro news, not energy-specific text.
  Classification accuracy on energy-technical headlines (LNG terminal outages, pipeline curtailments)
  is likely lower than on general financial news.
- Net sentiment aggregates all energy-related headlines equally; a pipeline disruption story and
  a solar capacity announcement receive equal weight.
    """)

with c6:
    _finbert_line = "installed, running live" if _TORCH_AVAILABLE else "not installed in this environment"
    _sent_status = ("Active (experimental)", "#3fb950") if _TORCH_AVAILABLE else ("Not available here", "#8b949e")
    st.markdown(
        f"""
<div style="background:#161b22;border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:16px;margin-bottom:8px;">
  <div style="color:#58a6ff;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;font-weight:600;margin-bottom:8px;">Status</div>
  <div style="color:{_sent_status[1]};font-size:0.88rem;margin-bottom:12px;">{_sent_status[0]}</div>
  <div style="color:#8b949e;font-size:0.82rem;line-height:1.7;">
    FinBERT: {_finbert_line}<br>
    RSS feeds: 6 sources, 6h cache<br>
    History: archive backfill when shorter than 21 days<br>
    Granger / IC test: needs ≥21 days history<br>
    ENTSO-E: not required
  </div>
</div>
<div style="background:#161b22;border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:16px;">
  <div style="color:#58a6ff;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;font-weight:600;margin-bottom:8px;">Why This Matters</div>
  <div style="color:#8b949e;font-size:0.82rem;line-height:1.7;">
    Energy prices react to news faster than fundamentals update. The 2026 Hormuz
    escalation (a 40% TTF spike in three days) would have been detectable as a
    sharp negative sentiment shift before prices fully adjusted.
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

# ── Live sentiment panel ──────────────────────────────────────────────────────
st.markdown("#### Live Sentiment Feed")

with st.spinner("Fetching headlines and running FinBERT…"):
    sent = get_sentiment_data()
    ttf  = get_ttf_data()

if sent["error"]:
    st.info(sent["error"])
else:
    daily  = sent["daily"]
    scored = sent["scored"]

    C = {
        "bg": CHART["bg"], "panel_bg": CHART["panel_bg"],
        "grid": CHART["grid"], "text": CHART["text"],
        "pos": "#3fb950", "neg": "#f85149",
        "ma":  "#58a6ff", "zero": CHART["zero"],
    }

    # KPI row
    n_headlines  = len(sent["headlines"])
    latest_score = float(daily["net_sentiment"].iloc[-1]) if not daily.empty else 0.0
    latest_date  = daily["date"].iloc[-1].strftime("%Y-%m-%d") if not daily.empty else "n/a"
    sig_label    = "Bearish" if latest_score < -0.1 else ("Bullish" if latest_score > 0.1 else "Neutral")
    sig_color    = "red" if latest_score < -0.1 else ("green" if latest_score > 0.1 else "blue")

    n_days = len(daily)
    # Signal confidence: low with <14 days of data
    low_confidence = n_days < 14
    sig_color_adj = "amber" if low_confidence else sig_color

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(kpi_card("Headlines scored", str(n_headlines),
                             delta_span(f"{len(scored)} in stored history · 6h refresh", "blue")), unsafe_allow_html=True)
    with k2:
        st.markdown(kpi_card("Latest net sentiment", f"{latest_score:+.3f}",
                             delta_span(f"{latest_date}", sig_color_adj)), unsafe_allow_html=True)
    with k3:
        conf_note = "low confidence (<14d)" if low_confidence else "based on latest daily avg"
        st.markdown(kpi_card("Signal", sig_label,
                             delta_span(conf_note, sig_color_adj)), unsafe_allow_html=True)
    with k4:
        days_needed = max(0, 21 - n_days)
        granger_label = "Ready" if days_needed == 0 else f"{days_needed} more days"
        st.markdown(kpi_card("Granger test", granger_label,
                             delta_span(f"{n_days} days of history", "green" if days_needed == 0 else "amber")),
                    unsafe_allow_html=True)

    if low_confidence:
        st.caption(
            f"Sentiment signal based on {n_days} days of history, which is insufficient for statistical confidence. "
            "Signal validity increases as history accumulates beyond 14 days. "
            "FinBERT is fine-tuned on equity and macro news rather than energy-specific text; the directional signal should be treated as indicative only."
        )

    # Daily sentiment chart (if we have data)
    if not daily.empty:
        bar_colors = [C["pos"] if v >= 0 else C["neg"] for v in daily["net_sentiment"]]

        fig_s = go.Figure()
        fig_s.add_trace(go.Bar(
            x=daily["date"], y=daily["net_sentiment"],
            marker_color=bar_colors,
            name="Daily net sentiment",
            hovertemplate="<b>%{x}</b><br>Net sentiment: %{y:+.3f}<extra></extra>",
        ))
        if len(daily) >= 3:
            fig_s.add_trace(go.Scatter(
                x=daily["date"], y=daily["ma7"],
                name="7-day MA",
                line=dict(color=C["ma"], width=1.8, dash="dot"),
                hovertemplate="7d MA: %{y:+.3f}<extra></extra>",
            ))
        fig_s.add_hline(y=0, line_dash="solid", line_color=C["zero"], line_width=0.8)
        fig_s.update_layout(
            title=dict(text="Daily Energy News Sentiment (FinBERT, net = positive − negative)", font=dict(size=12)),
            template=PLOTLY_TEMPLATE,
            paper_bgcolor=C["bg"], plot_bgcolor=C["panel_bg"],
            font=dict(color=C["text"], size=12),
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(showgrid=True, gridcolor=C["grid"], title=None),
            yaxis=dict(title="Net sentiment (-1 to +1)", showgrid=True, gridcolor=C["grid"]),
            hovermode="x unified",
            height=260,
        )
        st.plotly_chart(fig_s, width="stretch")

    # Granger causality (requires ≥21 days)
    if not daily.empty and not ttf["prices"].empty and len(daily) >= 21:
        ttf_df = ttf["prices"][["date", "price"]].copy()
        ttf_df["date"] = pd.to_datetime(ttf_df["date"])
        ttf_df["return"] = ttf_df["price"].pct_change() * 100

        merged = daily[["date", "net_sentiment"]].merge(
            ttf_df[["date", "return"]], on="date", how="inner"
        ).dropna()

        if len(merged) >= 21:
            try:
                from statsmodels.tsa.stattools import grangercausalitytests
                gc = grangercausalitytests(merged[["return", "net_sentiment"]], maxlag=2, verbose=False)
                p_lag1 = gc[1][0]["ssr_ftest"][1]
                p_lag2 = gc[2][0]["ssr_ftest"][1]
                gc_result = f"Lag 1: p={p_lag1:.3f}  Lag 2: p={p_lag2:.3f}"
                gc_sig = "warn" if min(p_lag1, p_lag2) < 0.1 else "ok"
                st.markdown(
                    commentary(
                        f"Granger causality test (sentiment predicting TTF returns): {gc_result}. "
                        + ("p < 0.10: sentiment has marginal predictive power for next-day TTF moves."
                           if min(p_lag1, p_lag2) < 0.1 else
                           "No significant Granger causality at p < 0.10 with current history."),
                        gc_sig,
                    ),
                    unsafe_allow_html=True,
                )
            except Exception:
                pass
    elif len(daily) < 21:
        st.caption(
            f"Granger causality and rolling IC tests require ≥21 days of history. "
            f"Currently have {len(daily)} day(s); older headlines are being collected from the news archives."
        )

    # Recent scored headlines
    with st.expander(f"Recent scored headlines ({len(scored)} total)", expanded=True):
        if not scored.empty:
            display = scored.sort_values("date", ascending=False).head(20)[
                ["date", "source", "net_sentiment", "title"]
            ].copy()
            display["net_sentiment"] = display["net_sentiment"].round(3)
            display.columns = ["Date", "Source", "Net sentiment", "Headline"]
            st.dataframe(display.set_index("Date"), width="stretch")

    if sent["fetched_at"]:
        st.caption(f"Last updated: {sent['fetched_at'].strftime('%Y-%m-%d %H:%M UTC')} · "
                   f"Sources: {FEED_SOURCES_TEXT}")

st.divider()

# ── Data dependency map ───────────────────────────────────────────────────────
st.markdown("### Data Dependencies")
with st.expander("Feature pipeline and ENTSO-E data status", expanded=True):
    _hydro_st  = "Live" if _avail.get("hydro") else "Unavailable"
    _wind_st   = "Live" if _avail.get("wind")  else "Unavailable"
    _stor_st   = "Live" if _avail.get("storage") else "Unavailable"
    st.markdown(f"""
    | Feature | Source | Models | Status |
    |---|---|---|---|
    | NO2, NL day-ahead price | ENTSO-E A44 (Nord Pool for the latest days) | LSTM target + input, HMM | {"Live" if _avail.get("core") else "Unavailable"} |
    | TTF gas price | Yahoo Finance (TTF=F) | LSTM input, HMM | Live |
    | EU gas storage fill % | GIE AGSI+ | LSTM + HMM | {_stor_st} |
    | Norwegian hydro (TWh) | ENTSO-E B31 | LSTM + HMM | {_hydro_st} |
    | German wind generation | ENTSO-E B18/B19, energy-charts.info fallback | LSTM + HMM (optional) | {_wind_st} |

    **Current feature matrix:** {_n_rows} trading days, {len(get_lstm_feature_cols(_features_df))} LSTM features,
    {len(get_hmm_feature_cols(_features_df))} HMM features.
    The models use wind and hydro whenever those sources are available. Prices are daily averages
    over each delivery day (CET), and wind is energy per day from hourly mean power.
    """)

page_footer([
    ("Frameworks", "PyTorch · hmmlearn · Hugging Face Transformers (FinBERT)"),
])
