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
    page_icon="L",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from utils.helpers import apply_dark_theme, commentary, kpi_card, delta_span, has_torch
from data.sentiment import get_sentiment_data
from data.prices import get_ttf_data

apply_dark_theme()

st.markdown("## Layer 4: Machine Learning and Advanced Models")
st.caption(
    "Machine learning applied to European power and gas market forecasting and regime classification."
)

_TORCH_AVAILABLE = has_torch()
if not _TORCH_AVAILABLE:
    st.info(
        "Layer 4 ML models require PyTorch and Transformers, which are excluded from this "
        "deployment due to memory constraints on the free hosting tier. "
        "The full pipeline runs locally and on GPU-enabled platforms (e.g. Hugging Face Spaces). "
        "Architecture, feature specifications, and training setup are shown below."
    )

from models.feature_assembly import (
    assemble_features, get_feature_meta, get_available_feature_sets,
    get_lstm_feature_cols, get_hmm_feature_cols,
)
from models.lstm_model import (
    is_trained as lstm_is_trained, load_meta as lstm_load_meta,
    train_lstm_full, predict_next,
)
from models.hmm_model import (
    is_trained as hmm_is_trained, load_meta as hmm_load_meta,
    train_hmm, predict_regime, get_regime_history, REGIME_COLORS,
)
st.divider()

# ── Load feature matrix (cached) ─────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def _get_features():
    return assemble_features(years=3)

with st.spinner("Loading feature matrix…"):
    _features_df = _get_features()

_avail = get_available_feature_sets(_features_df)
_n_rows = len(_features_df)
_feat_meta = get_feature_meta()

# Top status banner
_status_parts = []
for label, key in [("Prices", "core"), ("Storage", "storage"), ("Hydro", "hydro"), ("Wind", "wind")]:
    status = "available" if _avail.get(key) else "pending"
    _status_parts.append(f"{label}: {status}")
_ready = _avail.get("core") and _n_rows >= 50

st.markdown(
    commentary(
        f"Feature matrix: {_n_rows} trading days assembled. "
        f"{' | '.join(_status_parts)}. "
        + ("Models can be trained: use the Train buttons in each model section below."
           if _ready else
           "Insufficient data to train. Nord Pool prices and TTF are required as a minimum."),
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
    _lstm_status_color = "#3fb950" if lstm_is_trained() else ("#d29922" if _ready else "#f85149")
    _lstm_status_label = "Trained (live)" if lstm_is_trained() else ("Ready to train" if _ready else "Awaiting data")
    _lstm_feat_list = get_lstm_feature_cols(_features_df) if not _features_df.empty else []
    _wind_note = "incl. wind" if _avail.get("wind") else "wind pending (B16)"
    st.markdown(
        f"""
<div style="background:#161b22;border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:16px;margin-bottom:8px;">
  <div style="color:#58a6ff;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;font-weight:600;margin-bottom:8px;">Status</div>
  <div style="color:{_lstm_status_color};font-size:0.88rem;margin-bottom:12px;">{_lstm_status_label}</div>
  <div style="color:#8b949e;font-size:0.82rem;line-height:1.7;">
    Training rows: {_n_rows}<br>
    Features: {len(_lstm_feat_list)} ({_wind_note})<br>
    Hydro (B31): {"available" if _avail.get("hydro") else "pending"}<br>
    Wind (B16/B19): {"available" if _avail.get("wind") else "pending (Fraunhofer ISE fallback active)"}
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
with st.expander("Live Model 1: Train and Predict", expanded=lstm_is_trained()):
    if not _ready:
        st.warning("Insufficient data to train. Need at least Nord Pool prices + TTF (50+ days).")
    else:
        _lstm_feat_cols = get_lstm_feature_cols(_features_df)
        st.caption(
            f"Training features ({len(_lstm_feat_cols)}): {', '.join(_lstm_feat_cols)}. "
            f"Data: {_n_rows} days. "
            + ("Wind generation not yet included; it will be added automatically once B16 data is available."
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

        # Train button
        _btn_label = "Retrain Model 1 (LSTM)" if lstm_is_trained() else "Train Model 1 (LSTM)"
        if st.button(_btn_label, key="train_lstm"):
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
    _hmm_status_color = "#3fb950" if hmm_is_trained() else ("#d29922" if _ready else "#f85149")
    _hmm_status_label = "Trained (live)" if hmm_is_trained() else ("Ready to train" if _ready else "Awaiting data")
    _hmm_feat_list = get_hmm_feature_cols(_features_df) if not _features_df.empty else []
    st.markdown(
        f"""
<div style="background:#161b22;border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:16px;margin-bottom:8px;">
  <div style="color:#58a6ff;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;font-weight:600;margin-bottom:8px;">Status</div>
  <div style="color:{_hmm_status_color};font-size:0.88rem;margin-bottom:12px;">{_hmm_status_label}</div>
  <div style="color:#8b949e;font-size:0.82rem;line-height:1.7;">
    Training rows: {_n_rows}<br>
    HMM features: {len(_hmm_feat_list)}<br>
    Hydro percentile: {"available" if _avail.get("hydro") else "pending"}<br>
    Wind z-score: {"available" if _avail.get("wind") else "pending"}
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
                    _C = {"bg": "#0d1117", "panel_bg": "#161b22", "grid": "rgba(255,255,255,0.06)", "text": "#c9d1d9"}
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
                        template="plotly_dark", paper_bgcolor=_C["bg"], plot_bgcolor=_C["panel_bg"],
                        font=dict(color=_C["text"], size=11),
                        margin=dict(l=10, r=10, t=10, b=10), height=200,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        xaxis=dict(showgrid=True, gridcolor=_C["grid"], title=None),
                        yaxis=dict(showgrid=False, title=None),
                        hovermode="x unified",
                    )
                    st.plotly_chart(_fig_r, width="stretch")

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
- **Input:** Energy-relevant headlines filtered from BBC Business, Guardian Energy,
  LNG World News, and Energy Monitor RSS feeds
- **Processing:** Each headline is scored. Daily aggregation:
  net sentiment = mean(positive conf.) − mean(negative conf.) ∈ [−1, +1]

**Pipeline:**
1. Fetch RSS headlines (4 feeds, cached 6h)
2. Filter by energy keywords: LNG, TTF, Hormuz, Norwegian gas, pipeline, carbon…
3. Classify each headline with FinBERT (CPU inference, ~1s/headline)
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
    st.markdown(
        """
<div style="background:#161b22;border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:16px;margin-bottom:8px;">
  <div style="color:#58a6ff;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;font-weight:600;margin-bottom:8px;">Status</div>
  <div style="color:#3fb950;font-size:0.88rem;margin-bottom:12px;">Active (experimental)</div>
  <div style="color:#8b949e;font-size:0.82rem;line-height:1.7;">
    FinBERT: installed, running live<br>
    RSS feeds: 4 sources, 6h cache<br>
    Sentiment scoring: active<br>
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
    st.warning(f"Sentiment pipeline: {sent['error']}")
else:
    daily  = sent["daily"]
    scored = sent["scored"]

    C = {
        "bg": "#0d1117", "panel_bg": "#161b22",
        "grid": "rgba(255,255,255,0.06)", "text": "#c9d1d9",
        "pos": "#3fb950", "neg": "#f85149",
        "ma":  "#58a6ff", "zero": "rgba(255,255,255,0.2)",
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
                             delta_span("last 30 days · 6h refresh", "blue")), unsafe_allow_html=True)
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
            template="plotly_dark",
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
            f"Currently have {len(daily)} day(s). Signal will self-validate as data accumulates."
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
                   f"Sources: BBC Business, Guardian Energy, LNG World News, Energy Monitor")

st.divider()

# ── Data dependency map ───────────────────────────────────────────────────────
st.markdown("### Data Dependencies")
with st.expander("Feature pipeline and ENTSO-E data status", expanded=True):
    _hydro_st  = "Live" if _avail.get("hydro") else "Pending (ENTSO-E B31)"
    _wind_st   = "Live" if _avail.get("wind")  else "Pending (ENTSO-E B16/B19); Fraunhofer ISE fallback active"
    _stor_st   = "Live" if _avail.get("storage") else "Requires AGSI_API_KEY"
    st.markdown(f"""
    | Feature | Source | Models | Status |
    |---|---|---|---|
    | NO2, NL day-ahead price | Nord Pool Data Portal | LSTM target + input, HMM | Live |
    | TTF gas price | Yahoo Finance (TTF=F) | LSTM input, HMM | Live |
    | EU gas storage fill % | GIE AGSI+ | LSTM + HMM | {_stor_st} |
    | Norwegian hydro (TWh) | ENTSO-E B31 | LSTM + HMM | {_hydro_st} |
    | German wind generation | ENTSO-E B16/B18/B19 + Fraunhofer ISE fallback | LSTM + HMM (optional) | {_wind_st} |

    **Current feature matrix:** {_n_rows} trading days, {len(get_lstm_feature_cols(_features_df))} LSTM features,
    {len(get_hmm_feature_cols(_features_df))} HMM features.
    Models can be trained now with available data and will automatically upgrade
    to include wind and hydro when those sources return.
    """)

st.divider()
st.markdown(
    """<div style="color:#484f58;font-size:0.72rem;line-height:1.8;">
    Built by Tobias Gudbjerg &nbsp;|&nbsp;
    Frameworks: PyTorch &nbsp;·&nbsp; hmmlearn &nbsp;·&nbsp; Hugging Face Transformers (FinBERT)<br>
    For informational purposes only. Not financial advice.
    </div>""",
    unsafe_allow_html=True,
)
