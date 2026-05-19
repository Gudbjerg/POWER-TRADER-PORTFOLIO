"""
Layer 2: Quantitative Analysis
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time as _time
from dotenv import load_dotenv
from datetime import date as _date

load_dotenv()

_PAGE_T0 = _time.perf_counter()

st.set_page_config(
    page_title="Quantitative Analysis",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from utils.helpers import apply_dark_theme, kpi_card, delta_span, commentary
from config.settings import (
    SCORE_RISK_PREMIUM_EUR, SCORE_SEASONAL_HI_PCT, SCORE_SEASONAL_LO_PCT,
    COINT_ENTRY_Z, COINT_MIN_OBS, COINT_MIN_MATCH, HYDRO_MAX_LAG,
    PCA_STORAGE_ALPHA, PCA_STORAGE_DECAY_TAU, PCA_ENTRY_Z, PCA_LOOKBACK_DAYS,
)
from data.gas_storage import get_storage_data
from data.prices import get_ttf_data
from data.spot_prices import fetch_spot_prices
from models.storage_monte_carlo import compute_monthly_stats, run_monte_carlo
from models.gas_power_regression import prepare_data, run_full_ols
from models.spike_detector import compute_zscores, latest_signals, ALERT_Z, WARNING_Z
from models.ttf_backtest import fetch_ttf_history, compute_seasonal_strategy, compute_strategy_stats
from models.supply_stack import build_stack, identify_marginal_fuel, make_merit_order_figure, fetch_eua_price
from models.nordic_decomp import (
    run_rolling_decomposition, current_contributions, dominant_driver,
    make_beta_chart, make_contribution_bar, FACTOR_LABELS,
)
from models.feature_assembly import assemble_features, get_available_feature_sets
from models.wind_forecast_error import (
    compute_errors, compute_rolling_rmse, compute_price_correlation,
    make_error_bar_chart, make_rmse_chart, make_correlation_scatter,
    ZONE_LABELS as WIND_ZONE_LABELS,
)
from data.wind import fetch_wind_daily, fetch_wind_forecast
from data.sentiment import get_sentiment_data

apply_dark_theme()

# ── Load data ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False, persist="disk")
def _get_features_l2():
    return assemble_features(years=3)

@st.cache_data(ttl=3600, show_spinner=False, persist="disk")
def _run_decomp_cached(df: pd.DataFrame, window: int = 90):
    return run_rolling_decomposition(df, window)

@st.cache_data(ttl=3600, show_spinner=False, persist="disk")
def _run_ols_cached(reg_df: pd.DataFrame):
    return run_full_ols(reg_df)

@st.cache_data(ttl=21600, show_spinner=False)
def _get_sentiment_l2():
    return get_sentiment_data()

@st.cache_data(ttl=3600, show_spinner=False, persist="disk")
def _get_eua():
    return fetch_eua_price()

@st.cache_data(ttl=3600, show_spinner=False, persist="disk")
def _get_ttf_history_norm():
    return fetch_ttf_history(years=7)

@st.cache_data(show_spinner=False)
def _run_sc_ols_cached(sc_m_df: pd.DataFrame) -> float:
    import statsmodels.api as _sm_ols
    _X   = _sm_ols.add_constant(sc_m_df["storage_pct"])
    _fit = _sm_ols.OLS(sc_m_df["price"], _X).fit()
    _row = sc_m_df.iloc[-1]
    return float(_row["price"] - (_fit.params[0] + _fit.params[1] * float(_row["storage_pct"])))

@st.cache_data(show_spinner=False)
def _run_sc_granger_cached(gc_df: pd.DataFrame) -> tuple:
    from statsmodels.tsa.stattools import grangercausalitytests as _gct
    _gc    = _gct(gc_df[["return", "net_sentiment"]], maxlag=2, verbose=False)
    _min_p = min(_gc[1][0]["ssr_ftest"][1], _gc[2][0]["ssr_ftest"][1])
    return _min_p, float(gc_df["net_sentiment"].iloc[-7:].mean())


# ── Forward Curve PCA helpers (D2) ───────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def _build_ttf_pca_panel(eu_storage: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Synthetic TTF M+1–M+18 forward-curve panel.
    F(t,m) = spot_t × seasonal_m × (1 + α × storage_z_t × exp(−m/τ))
    Returns (panel_df, status_msg). panel_df: index=date, cols=M+1..M+18.
    """
    from data.forward_curve import _fetch_ttf_long

    _spot_df = _fetch_ttf_long()
    if _spot_df.empty or len(_spot_df) < 200:
        return pd.DataFrame(), "TTF spot history unavailable"
    _spot_df = _spot_df.copy()
    _spot_df["date"]  = pd.to_datetime(_spot_df["date"])
    _spot_df["month"] = _spot_df["date"].dt.month
    _spot_df["year"]  = _spot_df["date"].dt.year

    # Seasonal index: monthly mean / annual mean, median across years
    _ratios: list[dict] = []
    for _yr, _grp in _spot_df.groupby("year"):
        if len(_grp) < 200:
            continue
        _ann = _grp["price"].mean()
        if _ann <= 0:
            continue
        for _mo, _mg in _grp.groupby("month"):
            _ratios.append({"month": int(_mo), "ratio": float(_mg["price"].mean() / _ann)})
    if _ratios:
        _si = pd.DataFrame(_ratios).groupby("month")["ratio"].median().to_dict()
    else:
        _si = {m: 1.0 for m in range(1, 13)}

    # Storage z-score vs same-DOY 5-year mean
    _stor_z = pd.Series(dtype=float)
    if (not eu_storage.empty
            and "gasDayStart" in eu_storage.columns
            and "full" in eu_storage.columns):
        _st2 = eu_storage.copy()
        _st2["date"] = pd.to_datetime(_st2["gasDayStart"]).dt.normalize()
        _st2["doy"]  = _st2["date"].dt.dayofyear
        _st2["year"] = _st2["date"].dt.year
        _cur_yr = pd.Timestamp.now().year
        _doy_stats = (
            _st2[_st2["year"] < _cur_yr]
            .groupby("doy")["full"]
            .agg(["mean", "std"])
            .rename(columns={"mean": "mu", "std": "sd"})
        )
        _st2 = _st2.merge(_doy_stats, on="doy", how="left")
        _st2["storage_z"] = ((_st2["full"] - _st2["mu"])
                             / _st2["sd"].clip(lower=0.5))
        _stor_z = _st2.set_index("date")["storage_z"].dropna()

    # Merge spot + storage z, forward-fill storage gaps
    _cutoff = pd.Timestamp.now() - pd.Timedelta(days=PCA_LOOKBACK_DAYS)
    _merged = (_spot_df[_spot_df["date"] >= _cutoff]
               .set_index("date")[["price"]]
               .copy())
    if not _stor_z.empty:
        _merged = _merged.join(_stor_z.rename("sz"), how="left")
        _merged["sz"] = _merged["sz"].ffill().fillna(0.0)
    else:
        _merged["sz"] = 0.0

    if len(_merged) < 100:
        return pd.DataFrame(), f"Insufficient history after cutoff: {len(_merged)} obs"

    # Build 18-tenor panel vectorised
    _decay = np.array([np.exp(-m / PCA_STORAGE_DECAY_TAU) for m in range(1, 19)])
    _panel_cols: dict[str, np.ndarray] = {}
    for _m in range(1, 19):
        _fwd_months = np.array(
            [((_dt.month - 1 + _m) % 12) + 1 for _dt in _merged.index]
        )
        _seas = np.array([_si.get(int(mo), 1.0) for mo in _fwd_months])
        _carry = 1.0 + PCA_STORAGE_ALPHA * _merged["sz"].values * _decay[_m - 1]
        _panel_cols[f"M+{_m}"] = _merged["price"].values * _seas * _carry

    _panel = pd.DataFrame(_panel_cols, index=_merged.index)
    _panel.index.name = "date"
    _stor_note = " · storage-carry applied" if not _stor_z.empty else " · no storage data (flat carry)"
    return _panel, f"{len(_panel)} obs | M+1–M+18{_stor_note}"


@st.cache_data(show_spinner=False)
def _run_pca(panel: pd.DataFrame) -> dict | None:
    """PCA on de-meaned panel. Returns dict or None on failure."""
    try:
        from sklearn.decomposition import PCA as _PCA
        _X   = panel.values.astype(float)
        _Xdm = _X - _X.mean(axis=0)
        _pca = _PCA(n_components=3)
        _sc  = _pca.fit_transform(_Xdm)
        _scores = pd.DataFrame(_sc, index=panel.index, columns=["PC1", "PC2", "PC3"])
        # Expanding z-score (no look-ahead bias)
        for _pc in ("PC1", "PC2", "PC3"):
            _mu = _scores[_pc].expanding(min_periods=30).mean()
            _sd = _scores[_pc].expanding(min_periods=30).std().clip(lower=1e-9)
            _scores[f"{_pc}_z"] = (_scores[_pc] - _mu) / _sd
        return {
            "ev_ratio":   _pca.explained_variance_ratio_.tolist(),
            "loadings":   pd.DataFrame(
                _pca.components_.T,
                index=panel.columns,
                columns=["PC1", "PC2", "PC3"],
            ),
            "scores":     _scores,
        }
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False, persist="disk")
def _get_scanner_prices(years: int = 3) -> pd.DataFrame:
    """Wide-format daily prices for cointegration scanner: NO2, NL, TTF + DE, FR (ENTSO-E A44) + NBP (yfinance)."""
    _base = _get_features_l2()
    if _base.empty:
        return pd.DataFrame()
    _out = _base[["date", "no2", "nl", "ttf"]].dropna(subset=["no2", "nl", "ttf"]).copy()
    _out["date"] = pd.to_datetime(_out["date"])
    _key = os.getenv("ENTSOE_API_KEY", "")
    if _key:
        try:
            from entsoe import EntsoePandasClient as _EPC
            _cli = _EPC(api_key=_key)
            _end = pd.Timestamp.now(tz="UTC").normalize()
            _st0 = _end - pd.Timedelta(days=int(years * 365))
            for _lbl, _eic in [("de", "10Y1001A1001A63L"), ("fr", "10YFR-RTE------C")]:
                _chunks: list = []
                _cs = _st0
                while _cs < _end:
                    _ce = min(_cs + pd.Timedelta(days=365), _end)
                    try:
                        _s = _cli.query_day_ahead_prices(_eic, start=_cs, end=_ce)
                        if _s is not None and not _s.empty:
                            _chunks.append(_s)
                    except Exception:
                        pass
                    _cs = _ce
                if _chunks:
                    _comb  = pd.concat(_chunks)
                    _comb  = _comb[~_comb.index.duplicated(keep="last")]
                    _daily = _comb.tz_convert("UTC").resample("D").mean()
                    _daily.index = pd.to_datetime(_daily.index).tz_localize(None).normalize()
                    _zdf   = pd.DataFrame({"date": _daily.index, _lbl: _daily.values})
                    _out   = _out.merge(_zdf, on="date", how="left")
        except Exception:
            pass
    try:
        import yfinance as _yf
        _nbp_raw = _yf.Ticker("NBP=F").history(
            start=(pd.Timestamp.now() - pd.Timedelta(days=int(years * 365))).strftime("%Y-%m-%d"),
            end=pd.Timestamp.now().strftime("%Y-%m-%d"),
        )
        if not _nbp_raw.empty:
            _nbp_idx = pd.to_datetime(_nbp_raw.index).tz_localize(None).normalize()
            _nbp_df  = pd.DataFrame({"date": _nbp_idx, "nbp": _nbp_raw["Close"].values})
            _out = _out.merge(_nbp_df, on="date", how="left")
    except Exception:
        pass
    return _out.sort_values("date").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _run_coint_scanner(price_df: pd.DataFrame) -> list[dict]:
    """Engle-Granger cointegration for all available pair combinations. Returns list of result dicts sorted by p-value."""
    try:
        from statsmodels.tsa.stattools import coint as _coint_fn
        import statsmodels.api as _sm_sc
    except ImportError:
        return []
    _pcols   = [c for c in price_df.columns if c != "date"]
    _results = []
    for _i, _a in enumerate(_pcols):
        for _b in _pcols[_i + 1:]:
            _pf = price_df[["date", _a, _b]].dropna()
            _n  = len(_pf)
            if _n < COINT_MIN_OBS:
                continue
            _ya = _pf[_a].values.astype(float)
            _yb = _pf[_b].values.astype(float)
            try:
                _, _pab, _ = _coint_fn(_ya, _yb)
                _, _pba, _ = _coint_fn(_yb, _ya)
            except Exception:
                continue
            if _pab <= _pba:
                _pval, _dep, _indep, _dn, _in_ = _pab, _ya, _yb, _a, _b
            else:
                _pval, _dep, _indep, _dn, _in_ = _pba, _yb, _ya, _b, _a
            _X_sc    = _sm_sc.add_constant(_indep)
            _ols_sc  = _sm_sc.OLS(_dep, _X_sc).fit()
            _beta_sc = float(_ols_sc.params[1])
            _resid_sc = np.asarray(_ols_sc.resid, dtype=float).copy()
            _rl  = _resid_sc[:-1]
            _rd  = np.diff(_resid_sc)
            _ou  = _sm_sc.OLS(_rd, _sm_sc.add_constant(_rl)).fit()
            _lam = float(_ou.params[1])
            _hl  = (-np.log(2) / _lam) if _lam < 0 else float("inf")
            _rs  = pd.Series(_resid_sc, dtype=float)
            _em  = _rs.expanding(min_periods=30).mean()
            _es  = _rs.expanding(min_periods=30).std()
            _zs  = ((_rs - _em) / (_es + 1e-9)).values
            _zv  = _zs[~np.isnan(_zs)]
            _znow = float(_zv[-1]) if len(_zv) else 0.0
            _mh  = min(60, max(5, int(_hl * 3))) if np.isfinite(_hl) else 21
            _hits, _tot, _bdays = 0, 0, []
            for _ix in range(len(_zs) - _mh):
                _z = _zs[_ix]
                if not np.isnan(_z) and abs(_z) > COINT_ENTRY_Z:
                    _tot += 1
                    _tsgn = -np.sign(_z)
                    for _d in range(1, _mh + 1):
                        _zf = _zs[_ix + _d]
                        if not np.isnan(_zf) and (np.sign(_zf) == _tsgn or abs(_zf) < 0.5):
                            _hits += 1
                            _bdays.append(_d)
                            break
            _hit_rate = _hits / _tot * 100 if _tot > 0 else float("nan")
            _avg_d    = float(np.mean(_bdays)) if _bdays else float("nan")
            _status   = "Not cointegrated"
            if _pval < 0.05:
                _status = "Actionable" if abs(_znow) > COINT_ENTRY_Z else "Wait"
            _results.append({
                "pair":      f"{_dn.upper()}/{_in_.upper()}",
                "a":         _dn,
                "b":         _in_,
                "n_obs":     _n,
                "pval":      round(float(_pval), 4),
                "half_life": float(_hl),
                "z_now":     round(float(_znow), 3),
                "hit_rate":  float(_hit_rate),
                "n_signals": _tot,
                "avg_days":  float(_avg_d),
                "max_hold":  _mh,
                "beta":      round(float(_beta_sc), 4),
                "status":    _status,
                "z_series":  _zs,
                "dates":     _pf["date"].values,
                "ya":        _dep,
                "yb":        _indep,
            })
    _results.sort(key=lambda r: r["pval"])
    return _results


@st.cache_data(ttl=1800, show_spinner=False, persist="disk")
def _get_de_live_load_gw() -> tuple:
    """
    Fetch today's German total load from ENTSO-E A65.
    Returns (load_gw: float | None, source_label: str).
    Falls back to yesterday if today's data not yet published.
    """
    _key = os.getenv("ENTSOE_API_KEY", "")
    if not _key:
        return None, "ENTSO-E key required"
    try:
        from entsoe import EntsoePandasClient
        import pytz
        from datetime import datetime as _dt
        _tz  = pytz.timezone("Europe/Berlin")
        _now = _dt.now(_tz)
        _start = pd.Timestamp(_now.date(), tz="Europe/Berlin")
        _end   = _start + pd.Timedelta(days=1)
        _client_a65 = EntsoePandasClient(api_key=_key)
        _ser = _client_a65.query_load("10Y1001A1001A83F", start=_start, end=_end)
        if _ser is None or (hasattr(_ser, "empty") and _ser.empty):
            _start -= pd.Timedelta(days=1)
            _end   -= pd.Timedelta(days=1)
            _ser    = _client_a65.query_load("10Y1001A1001A83F", start=_start, end=_end)
        if _ser is not None and not (hasattr(_ser, "empty") and _ser.empty):
            _gw = float(_ser.mean()) / 1000.0
            return _gw, f"ENTSO-E A65 ({_start.date()})"
        return None, "No load data returned"
    except Exception as _exc:
        return None, f"A65 error: {str(_exc)[:60]}"

with st.spinner(""):
    storage = get_storage_data()
    ttf     = get_ttf_data()
    spot_df = fetch_spot_prices(days=150)   # extended history for regression
    # features_df is lazy-loaded inside the tabs that need it (Nordic Decomp, Wind)
    # so the default tab (Storage MC) renders without waiting for ENTSO-E feature assembly

eu_df = storage["europe"]

current_pct: float | None = None
if not eu_df.empty and "full" in eu_df.columns:
    current_pct = float(eu_df["full"].iloc[-1])

monthly_stats = compute_monthly_stats(eu_df)
has_empirical = bool(monthly_stats)

# ── Sidebar: data freshness ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("#### Data freshness")
    _fmt = "%H:%M UTC"
    _st_at  = storage.get("fetched_at")
    _ttf_at = ttf.get("fetched_at")
    st.caption(f"Storage: {_st_at.strftime(_fmt) if _st_at else 'n/a'}")
    st.caption(f"TTF prices: {_ttf_at.strftime(_fmt) if _ttf_at else 'n/a'}")
    if not spot_df.empty:
        _spot_latest = pd.to_datetime(spot_df["date"]).max().strftime("%Y-%m-%d")
        st.caption(f"Spot prices: through {_spot_latest}")
    else:
        st.caption("Spot prices: unavailable")
    _fm = st.session_state.get("features_l2")
    if _fm is not None and not _fm.empty:
        _fm_latest = pd.to_datetime(_fm["date"]).max().strftime("%Y-%m-%d")
        st.caption(f"Feature matrix: {len(_fm)} rows through {_fm_latest}")
    else:
        st.caption("Feature matrix: loads on first visit to Nordic Decomp or Wind tab")
    st.divider()
    st.divider()

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("## Layer 2: Quantitative Analysis")
st.caption("Quantitative models for European power and gas market analysis.")
st.divider()

with st.expander("Model overview", expanded=False):
    st.markdown("""
    | Model | Status |
    |-------|--------|
    | Storage Refill Monte Carlo | Active |
    | Gas-to-Power OLS Regression (NL proxy) | Active |
    | Price Spike Detector | Active |
    | TTF Seasonal Injection-Withdrawal Backtest | Active |
    | Nordic price decomposition | Active |
    | Wind forecast error | Active |
    | Merit order / supply stack | Active |
    | Sentiment → TTF Granger Causality | Active |
    | Storage–Price OLS Regression | Active |
    | Hydro Reservoir Lead/Lag Analysis | Active |
    | TTF Seasonal Norm Tracker | Active |
    | NO2/NL Cointegration & Spread Mean-Reversion | Active |
    | Multi-Pair Cointegration Scanner (D1) | Active |
    | Quant Signal Scorecard (5-signal aggregate) | Active |
    """)

st.divider()

# ── Quant Signal Scorecard ────────────────────────────────────────────────────
_PILL = {
    "up":      "<span style='background:#3d1515;color:#f85149;border-radius:4px;"
               "padding:2px 7px;font-size:0.78rem;font-weight:600;'>↑ UPSIDE</span>",
    "down":    "<span style='background:#0f2a1a;color:#3fb950;border-radius:4px;"
               "padding:2px 7px;font-size:0.78rem;font-weight:600;'>↓ DOWNSIDE</span>",
    "neutral": "<span style='background:#1a2233;color:#58a6ff;border-radius:4px;"
               "padding:2px 7px;font-size:0.78rem;font-weight:600;'>→ NEUTRAL</span>",
    "na":      "<span style='background:#1a1a1a;color:#6e7681;border-radius:4px;"
               "padding:2px 7px;font-size:0.78rem;font-weight:600;'>⊘ N/A</span>",
}

def _score_card_html(name: str, value: str, direction: str, note: str) -> str:
    pill = _PILL[direction]
    return (
        f"<div style='background:#161b22;border:1px solid #30363d;border-radius:8px;"
        f"padding:14px 16px;height:100%;'>"
        f"<div style='color:#8b949e;font-size:0.74rem;margin-bottom:4px;'>{name}</div>"
        f"<div style='font-size:1.05rem;font-weight:700;color:#e6edf3;"
        f"margin-bottom:6px;'>{value}</div>"
        f"{pill}"
        f"<div style='color:#8b949e;font-size:0.74rem;margin-top:6px;line-height:1.3;'>{note}</div>"
        f"</div>"
    )

_sc_signals: list[dict] = []

# Signal 1: Storage risk premium (OLS residual)
try:
    _sc_eu = storage.get("europe", pd.DataFrame())
    _sc_ttf = ttf.get("prices", pd.DataFrame()).copy()
    if not _sc_eu.empty and not _sc_ttf.empty and "full" in _sc_eu.columns:
        _sc_str = _sc_eu[["gasDayStart", "full"]].dropna().rename(
            columns={"gasDayStart": "date", "full": "storage_pct"}
        )
        _sc_str["date"] = pd.to_datetime(_sc_str["date"])
        _sc_ttf["date"] = pd.to_datetime(_sc_ttf["date"])
        _sc_m = _sc_str.merge(_sc_ttf[["date", "price"]], on="date", how="inner").dropna()
        if len(_sc_m) >= COINT_MIN_MATCH:
            _sc_prem = _run_sc_ols_cached(_sc_m)
            _sc_dir = "up" if _sc_prem > SCORE_RISK_PREMIUM_EUR else ("down" if _sc_prem < -SCORE_RISK_PREMIUM_EUR else "neutral")
            _sc_note = ("TTF elevated above storage-implied value" if _sc_prem > SCORE_RISK_PREMIUM_EUR
                        else ("TTF below storage-implied value" if _sc_prem < -SCORE_RISK_PREMIUM_EUR
                              else "TTF near storage-implied fair value"))
            _sc_signals.append({"name": "Supply-Risk Premium", "value": f"{_sc_prem:+.1f} EUR/MWh",
                                 "direction": _sc_dir, "note": _sc_note})
        else:
            _sc_signals.append({"name": "Supply-Risk Premium", "value": "n/a",
                                 "direction": "na", "note": "Need AGSI key + 60+ obs"})
    else:
        _sc_signals.append({"name": "Supply-Risk Premium", "value": "n/a",
                             "direction": "na", "note": "Storage data unavailable"})
except ImportError:
    _sc_signals.append({"name": "Supply-Risk Premium", "value": "n/a",
                         "direction": "na", "note": "statsmodels not available"})
except Exception:
    _sc_signals.append({"name": "Supply-Risk Premium", "value": "n/a",
                         "direction": "na", "note": "Computation error"})

# Signal 2: TTF seasonal percentile rank
try:
    _sc_hist = _get_ttf_history_norm()
    if not _sc_hist.empty:
        _sc_hist = _sc_hist.copy()
        _sc_hist["date"]  = pd.to_datetime(_sc_hist["date"])
        _sc_hist["year"]  = _sc_hist["date"].dt.year
        _sc_hist["md"]    = _sc_hist["date"].dt.month * 100 + _sc_hist["date"].dt.day
        _sc_cur_yr  = _sc_hist["year"].max()
        _sc_latest_ttf_val  = float(_sc_hist.sort_values("date")["price"].iloc[-1])
        _sc_latest_md       = int(_sc_hist.sort_values("date")["md"].iloc[-1])
        _sc_day_hist = _sc_hist[(_sc_hist["year"] < _sc_cur_yr) & (_sc_hist["md"] == _sc_latest_md)]["price"]
        if len(_sc_day_hist) >= 2:
            _sc_pct_rank = float((_sc_day_hist < _sc_latest_ttf_val).mean() * 100)
            _sc_dir = "up" if _sc_pct_rank > SCORE_SEASONAL_HI_PCT else ("down" if _sc_pct_rank < SCORE_SEASONAL_LO_PCT else "neutral")
            _sc_note = (f"Historically elevated for this time of year" if _sc_pct_rank > SCORE_SEASONAL_HI_PCT
                        else (f"Historically cheap for this time of year" if _sc_pct_rank < SCORE_SEASONAL_LO_PCT
                              else "Within typical seasonal range"))
            _sc_signals.append({"name": "TTF Seasonal Rank", "value": f"{_sc_pct_rank:.0f}th pct",
                                 "direction": _sc_dir, "note": _sc_note})
        else:
            _sc_signals.append({"name": "TTF Seasonal Rank", "value": "n/a",
                                 "direction": "na", "note": "Insufficient history"})
    else:
        _sc_signals.append({"name": "TTF Seasonal Rank", "value": "n/a",
                             "direction": "na", "note": "yfinance unavailable"})
except Exception:
    _sc_signals.append({"name": "TTF Seasonal Rank", "value": "n/a",
                         "direction": "na", "note": "Computation error"})

# Signal 3: Marginal fuel (supply stack at live demand if available, else 60 GW)
try:
    _sc_ttf_px = ttf.get("prices", pd.DataFrame())
    _sc_ttf_latest = float(_sc_ttf_px["price"].iloc[-1]) if not _sc_ttf_px.empty else 50.0
    _sc_eua, _ = _get_eua()
    _sc_live_load, _sc_load_src = _get_de_live_load_gw()
    _sc_ref_demand = (
        float(np.clip(round(_sc_live_load), 30, 90))
        if _sc_live_load is not None else 60.0
    )
    _sc_stack  = build_stack(_sc_ttf_latest, _sc_eua)
    _sc_marg   = identify_marginal_fuel(_sc_stack, _sc_ref_demand)
    _sc_fuel   = _sc_marg["fuel"]
    _sc_dir = ("up" if _sc_fuel in ("gas", "oil")
               else ("down" if _sc_fuel in ("wind", "solar", "hydro")
                     else "neutral"))
    _sc_fuel_note = {
        "gas":   "Gas-marginal regime — TTF drives power prices",
        "oil":   "Oil/peaker marginal — demand near capacity ceiling",
        "coal":  "Coal-marginal — mid-stack clearing",
        "wind":  "Renewable-marginal — structurally low-cost clearing",
        "solar": "Renewable-marginal — structurally low-cost clearing",
        "hydro": "Hydro-marginal — low-cost supply clearing",
        "lignite": "Lignite-marginal — mid-stack clearing",
        "biomass": "Biomass-marginal — mid-stack clearing",
    }.get(_sc_fuel, f"{_sc_fuel} marginal")
    if _sc_live_load is None:
        _sc_fuel_note += f" · A65 n/a: {_sc_load_src}"
    _sc_demand_label = (
        f"{_sc_ref_demand:.0f} GW live"
        if _sc_live_load is not None
        else f"{_sc_ref_demand:.0f} GW ref"
    )
    _sc_signals.append({"name": f"Marginal Fuel ({_sc_demand_label})",
                         "value": _sc_marg["label"].split("/")[0].strip(),
                         "direction": _sc_dir, "note": _sc_fuel_note})
except Exception:
    _sc_signals.append({"name": "Marginal Fuel", "value": "n/a",
                         "direction": "na", "note": "Computation error"})

# Signal 4: Hydro level vs 90-day mean (session_state — only after Nordic Decomp/Wind visit)
try:
    _sc_fm = st.session_state.get("features_l2")
    if _sc_fm is not None and not _sc_fm.empty and "hydro_pct" in _sc_fm.columns:
        _sc_hydro_ser = _sc_fm["hydro_pct"].dropna()
        if len(_sc_hydro_ser) >= 30:
            _sc_hydro_cur  = float(_sc_hydro_ser.iloc[-1])
            _sc_hydro_mean = float(_sc_hydro_ser.iloc[-90:].mean())
            _sc_hydro_dev  = _sc_hydro_cur - _sc_hydro_mean
            _sc_dir = "down" if _sc_hydro_dev > 5 else ("up" if _sc_hydro_dev < -5 else "neutral")
            _sc_note = (f"Above 90d mean ({_sc_hydro_mean:.0f}%) — supply comfortable" if _sc_hydro_dev > 5
                        else (f"Below 90d mean ({_sc_hydro_mean:.0f}%) — hydro tightening" if _sc_hydro_dev < -5
                              else f"Near 90d mean ({_sc_hydro_mean:.0f}%)"))
            _sc_signals.append({"name": "Hydro Level", "value": f"{_sc_hydro_cur:.0f}%",
                                 "direction": _sc_dir, "note": _sc_note})
        else:
            _sc_signals.append({"name": "Hydro Level", "value": "n/a",
                                 "direction": "na", "note": "Visit Nordic Decomp tab to load"})
    else:
        _sc_signals.append({"name": "Hydro Level", "value": "n/a",
                             "direction": "na", "note": "Visit Nordic Decomp tab to load"})
except Exception:
    _sc_signals.append({"name": "Hydro Level", "value": "n/a",
                         "direction": "na", "note": "Computation error"})

# Signal 5: Granger sentiment
try:
    _sc_sent = _get_sentiment_l2()
    _sc_daily_sent = _sc_sent.get("daily", pd.DataFrame())
    _sc_ttf_px2 = ttf.get("prices", pd.DataFrame()).copy()
    if not _sc_daily_sent.empty and not _sc_ttf_px2.empty:
        _sc_daily_sent = _sc_daily_sent.copy()
        _sc_daily_sent["date"] = pd.to_datetime(_sc_daily_sent["date"])
        _sc_ttf_px2["date"]   = pd.to_datetime(_sc_ttf_px2["date"])
        _sc_ttf_px2["return"] = _sc_ttf_px2["price"].pct_change() * 100
        _sc_gc_merged = _sc_daily_sent[["date", "net_sentiment"]].merge(
            _sc_ttf_px2[["date", "return"]], on="date", how="inner"
        ).dropna().sort_values("date")
        if len(_sc_gc_merged) >= 21:
            _sc_min_p, _sc_recent_sent = _run_sc_granger_cached(_sc_gc_merged)
            if _sc_min_p < 0.10:
                _sc_dir = "up" if _sc_recent_sent < 0 else "down"
                _sc_note = (f"Significant (p={_sc_min_p:.3f}) + negative news flow → upside pressure"
                            if _sc_recent_sent < 0
                            else f"Significant (p={_sc_min_p:.3f}) + positive news flow → downside pressure")
            else:
                _sc_dir = "neutral"
                _sc_note = f"Not significant (p={_sc_min_p:.3f}) — no news-driven price signal"
            _sc_signals.append({"name": "Sentiment → TTF", "value": f"p={_sc_min_p:.3f}",
                                 "direction": _sc_dir, "note": _sc_note})
        else:
            _sc_signals.append({"name": "Sentiment → TTF", "value": "n/a",
                                 "direction": "na", "note": f"Need 21 aligned days ({len(_sc_gc_merged)} so far)"})
    else:
        _sc_signals.append({"name": "Sentiment → TTF", "value": "n/a",
                             "direction": "na", "note": "Sentiment pipeline unavailable"})
except ImportError:
    _sc_signals.append({"name": "Sentiment → TTF", "value": "n/a",
                         "direction": "na", "note": "statsmodels not available"})
except Exception:
    _sc_signals.append({"name": "Sentiment → TTF", "value": "n/a",
                         "direction": "na", "note": "Computation error"})

# Render scorecard
_n_up   = sum(1 for s in _sc_signals if s["direction"] == "up")
_n_down = sum(1 for s in _sc_signals if s["direction"] == "down")
_n_neu  = sum(1 for s in _sc_signals if s["direction"] == "neutral")
_agg_color = "#f85149" if _n_up > _n_down + _n_neu else ("#3fb950" if _n_down > _n_up + _n_neu else "#58a6ff")
_agg_label = (f"<span style='color:{_agg_color};font-weight:700;'>"
              f"{_n_up} upside &nbsp;/&nbsp; {_n_down} downside &nbsp;/&nbsp; {_n_neu} neutral</span>")

with st.expander("Quant Signal Scorecard", expanded=True):
    st.markdown(
        f"<div style='margin-bottom:10px;font-size:0.85rem;color:#8b949e;'>"
        f"Aggregate: {_agg_label}"
        f"<span style='color:#484f58;margin-left:12px;font-size:0.72rem;'>"
        f"Hydro loads after first Nordic Decomp visit · Sentiment accumulates over time</span></div>",
        unsafe_allow_html=True,
    )
    _sc_cols = st.columns(5)
    for _i, _sig in enumerate(_sc_signals):
        with _sc_cols[_i]:
            st.markdown(
                _score_card_html(_sig["name"], _sig["value"], _sig["direction"], _sig["note"]),
                unsafe_allow_html=True,
            )
    st.markdown("<div style='margin-bottom:4px'></div>", unsafe_allow_html=True)

st.divider()

# ── Model tabs ───────────────────────────────────────────────────────────────
tab_mc, tab_reg, tab_spike, tab_bt, tab_stack, tab_decomp, tab_wind, tab_granger, tab_storage_reg, tab_hydro_lag, tab_ttf_norm, tab_coint, tab_scanner, tab_pca = st.tabs([
    "Storage Monte Carlo",
    "Gas-to-Power Regression",
    "Price Spike Detector",
    "TTF Seasonal Strategy",
    "Supply Stack",
    "Nordic Decomposition",
    "Wind Forecast Error",
    "Granger Causality",
    "Storage–Price Regression",
    "Hydro Lead/Lag",
    "TTF Seasonal Norm",
    "NO2/NL Cointegration",
    "Cointegration Scanner",
    "Forward Curve PCA",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1: STORAGE REFILL MONTE CARLO
# ════════════════════════════════════════════════════════════════════════════
with tab_mc:
    st.markdown("### Storage Refill Monte Carlo Simulation")
    st.caption(
        "Empirical bootstrap simulation of 1,000 storage fill paths from the current "
        "EU aggregate level to November 1. Daily injection rates are sampled from the "
        "historical AGSI distribution by calendar month."
    )

    if current_pct is None:
        st.warning(
            "EU gas storage data unavailable. Add AGSI_API_KEY to .env to enable this model. "
            "Register free at agsi.gie.eu."
        )
    else:
        col_ctrl, col_spacer = st.columns([1, 3])
        with col_ctrl:
            rate_multiplier = st.slider(
                "Injection rate multiplier",
                min_value=0.4,
                max_value=1.5,
                value=1.0,
                step=0.05,
                help=(
                    "1.0x = historical average pace. "
                    "0.6x simulates a slow refill season (supply disruption, low LNG). "
                    "1.3x simulates an accelerated injection scenario."
                ),
            )

        paths, sim_dates = run_monte_carlo(
            current_pct=current_pct,
            monthly_stats=monthly_stats,
            n_paths=1000,
            rate_multiplier=rate_multiplier,
        )

        terminal    = paths[:, -1]
        p_reach_80  = float(np.mean(terminal >= 80.0)) * 100
        p_reach_90  = float(np.mean(terminal >= 90.0)) * 100
        p50_end     = float(np.percentile(terminal, 50))
        pct_dates   = pd.to_datetime(sim_dates)

        p10 = np.percentile(paths, 10, axis=0)
        p25 = np.percentile(paths, 25, axis=0)
        p50 = np.percentile(paths, 50, axis=0)
        p75 = np.percentile(paths, 75, axis=0)
        p90 = np.percentile(paths, 90, axis=0)

        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            color = "green" if p_reach_90 >= 70 else ("amber" if p_reach_90 >= 40 else "red")
            st.markdown(
                kpi_card("Prob(reach 90%, EU mandate)", f"{p_reach_90:.0f}%",
                         delta_span(f"at {rate_multiplier:.2f}x injection rate", color)),
                unsafe_allow_html=True,
            )
        with k2:
            color = "green" if p_reach_80 >= 80 else ("amber" if p_reach_80 >= 55 else "red")
            st.markdown(
                kpi_card("Prob(reach 80%, min threshold)", f"{p_reach_80:.0f}%",
                         delta_span("original 2022 emergency target", color)),
                unsafe_allow_html=True,
            )
        with k3:
            color = "green" if p50_end >= 90 else ("amber" if p50_end >= 75 else "red")
            st.markdown(
                kpi_card("P50 level on Nov 1", f"{p50_end:.1f}%",
                         delta_span("median scenario", color)),
                unsafe_allow_html=True,
            )
        with k4:
            st.markdown(
                kpi_card("Current EU Storage", f"{current_pct:.1f}%",
                         delta_span("current fill level", "blue")),
                unsafe_allow_html=True,
            )

        # Fan chart
        C = {
            "bg": "#0d1117", "panel_bg": "#161b22",
            "grid": "rgba(255,255,255,0.06)", "text": "#c9d1d9",
            "band_out": "rgba(88,166,255,0.12)", "band_in": "rgba(88,166,255,0.27)",
            "median": "#58a6ff", "hist": "#8b949e",
            "mandate": "#d4ac3a", "target": "#3fb950",
        }
        fig = go.Figure()

        if not eu_df.empty and "gasDayStart" in eu_df.columns:
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=90)
            hp = eu_df[eu_df["gasDayStart"] >= cutoff]
            fig.add_trace(go.Scatter(
                x=hp["gasDayStart"], y=hp["full"],
                name="EU storage (actual)",
                line=dict(color=C["hist"], width=1.8),
                hovertemplate="Actual: %{y:.1f}%<extra></extra>",
            ))

        fig.add_trace(go.Scatter(x=pct_dates, y=p90, line=dict(color="rgba(0,0,0,0)", width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=pct_dates, y=p10, name="P10-P90 range",
                                 fill="tonexty", fillcolor=C["band_out"],
                                 line=dict(color="rgba(0,0,0,0)", width=0), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=pct_dates, y=p75, line=dict(color="rgba(0,0,0,0)", width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=pct_dates, y=p25, name="P25-P75 range",
                                 fill="tonexty", fillcolor=C["band_in"],
                                 line=dict(color="rgba(0,0,0,0)", width=0), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=pct_dates, y=p50, name="P50 (median)",
                                 line=dict(color=C["median"], width=2.2),
                                 hovertemplate="P50: %{y:.1f}%<extra></extra>"))
        fig.add_hline(y=90, line_dash="dash", line_color=C["target"], line_width=1.5,
                      annotation_text="90% EU mandate (Nov 1)", annotation_position="right",
                      annotation_font=dict(color=C["target"], size=11))
        fig.add_hline(y=80, line_dash="dot", line_color=C["mandate"], line_width=1.0,
                      annotation_text="80% (2022 original)", annotation_position="right",
                      annotation_font=dict(color=C["mandate"], size=10))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor=C["bg"], plot_bgcolor=C["panel_bg"],
            font=dict(color=C["text"], size=12),
            margin=dict(l=10, r=90, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(showgrid=True, gridcolor=C["grid"], title=None),
            yaxis=dict(title="Storage fill (%)", showgrid=True, gridcolor=C["grid"],
                       range=[max(0, current_pct - 5), 102]),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        if not has_empirical:
            st.info(
                "Simulation is using fallback injection rates. "
                "Add AGSI_API_KEY to .env for empirically calibrated results."
            )

        with st.expander("Methodology", expanded=True):
            n_hist = int((~eu_df["gasDayStart"].isna()).sum()) if not eu_df.empty else 0
            st.markdown(f"""
            **Method:** Empirical bootstrap Monte Carlo with 1,000 paths.

            **Injection rate sampling:** For each simulated calendar day, a daily injection increment
            (percentage points of working gas volume per day) is drawn with replacement from the
            empirical distribution observed in that calendar month, excluding the current year
            to avoid look-ahead bias. Calibrated from {n_hist:,} AGSI EU aggregate observations
            (approximately 5 years of history).

            **Injection season:** April 1 through October 31. Storage is held flat outside this window.

            **Rate multiplier:** Scales all sampled daily increments uniformly. Allows stress-testing
            of below-average (supply disruption, low LNG diversion) or above-average injection seasons.
            The random seed is fixed per multiplier value for reproducible scenario comparisons.

            **Reference levels:**
            - **90% (green dashed):** Current EU mandate. The original 2022 emergency regulation (EU 2022/1032)
              set 80%, but subsequent extensions raised the target to 90% by November 1 for the 2023–2025+
              period. In 2025, EU storage reached 83% by October 1 and met the 90% mandate by November 1.
            - **80% (gold dotted):** The original 2022 emergency target under EU Regulation 2022/1032.
              Shown for historical reference.

            **Percentile bands:** Outer band (P10-P90) contains 80% of paths. Inner band (P25-P75) is the
            interquartile range. The blue line is the P50 median path.

            **Limitations:** Model assumes injection rates are IID within a given month across years.
            Does not capture serial correlation, geopolitical supply disruptions, or demand-side shocks.
            Use alongside live market signals on the Monitor page.
            """)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2: GAS-TO-POWER REGRESSION
# ════════════════════════════════════════════════════════════════════════════
with tab_reg:
    st.markdown("### Gas-to-Power OLS Regression")
    st.caption(
        "OLS regression of Netherlands day-ahead electricity price on TTF front-month gas price. "
        "NL is the most direct pairing with TTF (both Netherlands-based markets). "
        "Residuals identify dates where power is priced above or below what gas fundamentals imply."
    )

    POWER_ZONE = "NL"
    ttf_df = ttf["prices"]
    reg_df = prepare_data(ttf_df, spot_df, power_zone=POWER_ZONE)
    ols    = _run_ols_cached(reg_df)

    if not ols:
        st.warning(
            "Insufficient data for regression. Requires at least 20 days of overlapping "
            "TTF and NL day-ahead prices. Spot price data may still be loading."
        )
    else:
        n_obs       = ols["n_obs"]
        slope       = ols["slope"]
        intercept   = ols["intercept"]
        r2          = ols["r2"]
        residuals   = ols["residual"]
        z_scores    = ols["residual_zscore"]
        fitted_arr  = ols["fitted"]

        latest_z     = float(z_scores[-1])
        latest_resid = float(residuals[-1])
        latest_act   = float(reg_df["power_price"].iloc[-1])
        latest_fit   = float(fitted_arr[-1])

        z_color  = "red" if abs(latest_z) > 2 else ("amber" if abs(latest_z) > 1.5 else "green")
        # Low R² is expected in a renewables-dominated regime; signal value is in residuals, not fit
        r2_regime = "renewables-dominated (residual signal valid)" if r2 < 0.3 else (
            "moderate gas-power coupling" if r2 < 0.6 else "strong gas-marginal regime"
        )
        r2_color = "amber"  # neutral: low R² is regime information, not a model failure

        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(
                kpi_card("Residual z-score (latest)", f"{latest_z:+.2f}",
                         delta_span("above" if latest_z > 0 else "below gas-model price", z_color)),
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                kpi_card("R-squared (full sample)", f"{r2:.2f}",
                         delta_span(r2_regime, r2_color)),
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown(
                kpi_card("Beta (TTF to NL power)", f"{slope:.2f}",
                         delta_span("EUR/MWh power per EUR/MWh gas", "blue")),
                unsafe_allow_html=True,
            )
        with k4:
            st.markdown(
                kpi_card("Residual (actual - fitted)", f"{latest_resid:+.1f} EUR/MWh",
                         delta_span(f"actual {latest_act:.1f}, model {latest_fit:.1f}", z_color)),
                unsafe_allow_html=True,
            )

        C = {
            "bg": "#0d1117", "panel_bg": "#161b22",
            "grid": "rgba(255,255,255,0.06)", "text": "#c9d1d9",
            "scatter": "#58a6ff", "regline": "#d4ac3a",
            "actual": "#c9d1d9", "fitted": "#58a6ff",
            "resid_pos": "#f85149", "resid_neg": "#3fb950",
            "zero": "rgba(255,255,255,0.3)",
        }

        # Two-column layout: scatter (left), residuals time series (right)
        col_scat, col_resid = st.columns([1, 1])

        # Scatter plot
        with col_scat:
            x_range = [float(reg_df["ttf_price"].min()) * 0.97,
                       float(reg_df["ttf_price"].max()) * 1.03]
            y_line  = [slope * x + intercept for x in x_range]

            fig_sc = go.Figure()
            fig_sc.add_trace(go.Scatter(
                x=reg_df["ttf_price"], y=reg_df["power_price"],
                mode="markers",
                marker=dict(color=C["scatter"], size=5, opacity=0.7),
                name="Daily observations",
                hovertemplate="TTF: €%{x:.1f}, NL: €%{y:.1f}<extra></extra>",
            ))
            # Latest point highlighted
            fig_sc.add_trace(go.Scatter(
                x=[float(reg_df["ttf_price"].iloc[-1])],
                y=[latest_act],
                mode="markers",
                marker=dict(color="#f85149", size=9, symbol="circle",
                            line=dict(color="white", width=1)),
                name="Latest",
                hovertemplate=f"Latest: TTF €{reg_df['ttf_price'].iloc[-1]:.1f}, NL €{latest_act:.1f}<extra></extra>",
            ))
            fig_sc.add_trace(go.Scatter(
                x=x_range, y=y_line,
                mode="lines",
                line=dict(color=C["regline"], width=1.5, dash="dot"),
                name=f"OLS fit (R²={r2:.2f}, β={slope:.2f})",
                hoverinfo="skip",
            ))
            fig_sc.update_layout(
                title=dict(text="NL Power vs. TTF Gas Price (EUR/MWh)", font=dict(size=12)),
                template="plotly_dark", paper_bgcolor=C["bg"], plot_bgcolor=C["panel_bg"],
                font=dict(color=C["text"], size=11),
                margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(title="TTF (EUR/MWh)", showgrid=True, gridcolor=C["grid"]),
                yaxis=dict(title="NL day-ahead (EUR/MWh)", showgrid=True, gridcolor=C["grid"]),
                height=340,
            )
            st.plotly_chart(fig_sc, use_container_width=True)

        # Residuals time series
        with col_resid:
            z_vals  = z_scores
            z_dates = reg_df["date"].values
            colors  = [C["resid_pos"] if z > 0 else C["resid_neg"] for z in z_vals]

            fig_res = go.Figure()
            fig_res.add_trace(go.Bar(
                x=z_dates, y=z_vals,
                marker_color=colors,
                name="Residual z-score",
                hovertemplate="z: %{y:.2f}<extra></extra>",
            ))
            for level, label in [(2.0, "+2\u03c3"), (-2.0, "\u22122\u03c3")]:
                fig_res.add_hline(y=level, line_dash="dash",
                                  line_color="rgba(248,81,73,0.55)", line_width=1)
            fig_res.add_hline(y=0, line_dash="solid",
                              line_color=C["zero"], line_width=0.8)
            fig_res.update_layout(
                title=dict(text="Standardised Residual (z-score)", font=dict(size=12)),
                template="plotly_dark", paper_bgcolor=C["bg"], plot_bgcolor=C["panel_bg"],
                font=dict(color=C["text"], size=11),
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis=dict(showgrid=True, gridcolor=C["grid"], title=None),
                yaxis=dict(showgrid=True, gridcolor=C["grid"], title="z-score"),
                showlegend=False,
                hovermode="x unified",
                height=340,
            )
            st.plotly_chart(fig_res, use_container_width=True)

        # Signal interpretation
        if abs(latest_z) >= 2.0:
            direction = "above" if latest_z > 0 else "below"
            implication = (
                "NL power is unusually expensive relative to the gas cost curve. "
                "Possible drivers: North Sea wind drought, cold-snap demand spike, or interconnector constraints. "
                if latest_z > 0 else
                "NL power is unusually cheap relative to the gas cost curve. "
                "Possible drivers: North Sea wind surplus, weak industrial demand, "
                "or strong NorNed imports from Norway."
            )
            st.markdown(
                commentary(
                    f"Signal: NL day-ahead at {latest_act:.1f} EUR/MWh is {abs(latest_resid):.1f} EUR/MWh "
                    f"{direction} the TTF gas-model prediction ({latest_fit:.1f} EUR/MWh), "
                    f"a {abs(latest_z):.1f} standard deviation residual over the {n_obs}-day sample. "
                    + implication,
                    "critical" if abs(latest_z) > 2.5 else "warn",
                ),
                unsafe_allow_html=True,
            )
        else:
            st.caption(
                f"Latest NL day-ahead ({latest_act:.1f} EUR/MWh) is within normal range relative to "
                f"the TTF gas-model ({latest_fit:.1f} EUR/MWh). Residual z-score: {latest_z:+.2f}."
            )

        with st.expander("Methodology", expanded=True):
            st.markdown(f"""
            **Model:** OLS regression of Netherlands day-ahead electricity price on TTF front-month gas
            price, estimated over the full available sample of {n_obs} overlapping trading days.

            **Estimated coefficients:** Intercept {intercept:.1f} EUR/MWh, slope {slope:.2f}
            (EUR/MWh power per EUR/MWh gas), R-squared {r2:.2f}.

            **Why R²={r2:.2f} is not a model failure:**
            In a gas-marginal market (2021–2022), gas explained ~60–70% of power price variation.
            The current sample period reflects a renewables-dominated regime: on many days the marginal
            generator is wind or solar (near-zero marginal cost), so TTF has weak explanatory power
            for the absolute price level. R²={r2:.2f} confirms this: it is regime information, not a
            coding error. The model's value is not prediction accuracy; it is the **residual z-score**:
            on days when gas should explain power but the residual is extreme (|z|>2), there is an
            identifiable fundamental dislocation (wind drought, demand shock, interconnector constraint).

            **Data sources:**
            - NL day-ahead price: Nord Pool Data Portal (daily average, EUR/MWh)
            - TTF price: ICE TTF front-month futures via Yahoo Finance (EUR/MWh, approximately 120 calendar days / ~85 trading days)

            **Why NL rather than DE:** Nord Pool's public data portal returns null prices for DE-LU, which
            falls under separate EPEX Spot data licensing. NL is an appropriate proxy: it is directly
            connected to Germany and Belgium and is priced within the same Central West European (CWE)
            market coupling zone. The TTF gas hub is located in the Netherlands, making the NL power-gas
            relationship especially direct.

            **Residual z-score:** The raw residual (actual minus fitted, EUR/MWh) is normalised by the
            standard deviation of residuals over the sample. A z-score exceeding +2 indicates NL power
            is materially expensive relative to what the gas price implies; below -2 indicates cheapness.
            The z-score is the primary signal; the R² is secondary context.

            **Interpretation of beta ({slope:.2f}):**
            In a gas-marginal power market, the theoretical beta equals 1 divided by the CCGT heat rate
            efficiency (approximately 0.55-0.60), implying beta in the range 1.6-1.8. A beta below this
            range confirms renewables are partially displacing gas at the margin in the current sample.

            **Limitations:** The model is bivariate and omits coal price, carbon (EU ETS) price,
            wind and solar output, and hydro availability. It identifies deviations from the historical
            gas-power relationship but does not attribute them to specific fundamental drivers.
            """)

        st.caption("Sources: Nord Pool Data Portal (NL day-ahead) | ICE/Yahoo Finance (TTF)")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3: PRICE SPIKE DETECTOR
# ════════════════════════════════════════════════════════════════════════════
with tab_spike:
    st.markdown("### Day-Ahead Price Spike Detector")
    st.caption(
        f"Rolling 30-day z-score on daily day-ahead prices for each bidding zone. "
        f"Alert threshold: |z| > {ALERT_Z}. Warning threshold: |z| > {WARNING_Z}. "
        "Identifies zones with anomalous price levels relative to recent history."
    )

    spike_df  = spot_df   # already fetched at page load (days=150, more than sufficient)
    z_df      = compute_zscores(spike_df, window=30)
    latest_z  = latest_signals(z_df)

    if z_df.empty:
        st.warning("Spot price data unavailable for spike detection.")
    else:
        # Alert summary KPIs
        n_alert = int((latest_z["signal"] == "alert").sum()) if not latest_z.empty else 0
        n_warn  = int((latest_z["signal"] == "warn").sum())  if not latest_z.empty else 0
        n_norm  = int((latest_z["signal"] == "normal").sum()) if not latest_z.empty else 0

        ka, kw, kn, kd = st.columns(4)
        with ka:
            color = "red" if n_alert > 0 else "green"
            st.markdown(kpi_card("Zones in alert", str(n_alert),
                                 delta_span(f"|z| > {ALERT_Z}", color)), unsafe_allow_html=True)
        with kw:
            color = "amber" if n_warn > 0 else "green"
            st.markdown(kpi_card("Zones in warning", str(n_warn),
                                 delta_span(f"|z| > {WARNING_Z}", color)), unsafe_allow_html=True)
        with kn:
            st.markdown(kpi_card("Zones normal", str(n_norm),
                                 delta_span("within range", "green")), unsafe_allow_html=True)
        with kd:
            latest_date = z_df["date"].max() if not z_df.empty else "n/a"
            st.markdown(kpi_card("Latest data", str(latest_date),
                                 delta_span("30-day rolling window", "blue")), unsafe_allow_html=True)

        # Zone signal table
        if not latest_z.empty:
            st.markdown("#### Current Signal by Zone")
            ZONE_LABELS = {
                "NO1": "NO1 (Oslo)", "NO2": "NO2 (Kristiansand)",
                "SE3": "SE3 (Stockholm)", "NL": "NL (Netherlands)", "FI": "FI (Finland)",
            }
            SIG_PILL = {
                "alert":  '<span class="pill-critical">ALERT</span>',
                "warn":   '<span class="pill-warn">WARN</span>',
                "normal": '<span class="pill-ok">NORMAL</span>',
            }
            for _, row in latest_z.iterrows():
                zone   = row["zone"]
                price  = row["price_eur_mwh"]
                z      = row["z_score"]
                mean   = row["rolling_mean"]
                sig    = row["signal"]
                label  = ZONE_LABELS.get(zone, zone)
                pill   = SIG_PILL.get(sig, "")
                z_str  = f"{z:+.2f}" if not pd.isna(z) else "n/a"
                mean_s = f"{mean:.1f}" if not pd.isna(mean) else "n/a"
                st.markdown(
                    f"{pill} &nbsp; <strong>{label}</strong> &nbsp; "
                    f"€{price:.1f}/MWh &nbsp; z = {z_str} &nbsp; "
                    f"<span style='color:#8b949e;font-size:0.82rem;'>(30d avg: €{mean_s})</span>",
                    unsafe_allow_html=True,
                )
            st.markdown("<div style='margin-bottom:12px'></div>", unsafe_allow_html=True)

        # Z-score time series chart
        C = {
            "bg": "#0d1117", "panel_bg": "#161b22",
            "grid": "rgba(255,255,255,0.06)", "text": "#c9d1d9",
        }
        ZONE_COLORS = {
            "NO1": "#4caf8f", "NO2": "#2980b9", "SE3": "#7ec8e3",
            "NL":  "#c0392b", "FI":  "#8e6bbf",
        }

        fig_z = go.Figure()
        for zone in sorted(z_df["zone"].unique()):
            sub = z_df[z_df["zone"] == zone].dropna(subset=["z_score"]).sort_values("date")
            if sub.empty:
                continue
            label = ZONE_LABELS.get(zone, zone)
            fig_z.add_trace(go.Scatter(
                x=sub["date"], y=sub["z_score"],
                name=label,
                line=dict(color=ZONE_COLORS.get(zone, "#888"), width=1.5),
                hovertemplate=f"{label}: z=%{{y:.2f}}<extra></extra>",
            ))

        for level, color, dash in [
            (ALERT_Z,  "rgba(248,81,73,0.7)",  "dash"),
            (-ALERT_Z, "rgba(248,81,73,0.7)",  "dash"),
            (WARNING_Z, "rgba(210,153,34,0.5)", "dot"),
            (-WARNING_Z,"rgba(210,153,34,0.5)", "dot"),
            (0,         "rgba(255,255,255,0.2)", "solid"),
        ]:
            fig_z.add_hline(y=level, line_dash=dash, line_color=color, line_width=1)

        fig_z.update_layout(
            template="plotly_dark",
            paper_bgcolor=C["bg"], plot_bgcolor=C["panel_bg"],
            font=dict(color=C["text"], size=12),
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(showgrid=True, gridcolor=C["grid"], title=None),
            yaxis=dict(title="z-score (30-day rolling)", showgrid=True, gridcolor=C["grid"]),
            hovermode="x unified",
        )
        st.plotly_chart(fig_z, use_container_width=True)

        with st.expander("Methodology", expanded=False):
            st.markdown(f"""
            **Method:** For each bidding zone, the rolling 30-day mean and standard deviation of daily
            average day-ahead prices are computed. The z-score is defined as:
            `z = (price - rolling_mean) / rolling_std`.

            **Thresholds:**
            - |z| > {ALERT_Z}: Alert. The current price is {ALERT_Z} standard deviations above or below
              the 30-day historical average. This is anomalous relative to recent price history.
            - |z| > {WARNING_Z}: Warning. Elevated deviation; monitor for further moves.
            - |z| < {WARNING_Z}: Normal range.

            **Positive z:** Current price is above recent average. Possible causes: cold snap, wind drought,
            outage, or supply disruption.

            **Negative z:** Current price is below recent average. Possible causes: renewable surplus,
            weak demand, strong hydro inflow, or excess interconnector imports.

            **Limitations:** The z-score is relative to recent history (30 days), not long-run levels.
            A zone can show z = 0 while trading at historically elevated absolute prices if it has been
            consistently high for the past 30 days. Always compare against the absolute price level.

            Data source: Nord Pool Data Portal, day-ahead market prices.
            """)

        st.caption("Source: Nord Pool Data Portal, Day-Ahead Market Prices | nordpoolgroup.com")


# ════════════════════════════════════════════════════════════════════════════
# TAB 4: TTF SEASONAL INJECTION-WITHDRAWAL BACKTEST
# ════════════════════════════════════════════════════════════════════════════
with tab_bt:
    st.markdown("### TTF Seasonal Injection-Withdrawal Strategy Backtest")
    st.caption(
        "Rules-based strategy: buy summer TTF (Apr-Sep average), sell winter TTF (Oct-Mar average) "
        "when the seasonal spread exceeds round-trip storage cost. "
        "P&L is expressed per MWh of storage capacity deployed."
    )

    col_ctrl, col_spacer = st.columns([1, 3])
    with col_ctrl:
        storage_cost = st.slider(
            "Storage cost (EUR/MWh round-trip)",
            min_value=1.0,
            max_value=10.0,
            value=4.0,
            step=0.5,
            help=(
                "Round-trip cost of holding gas in storage: injection fee + withdrawal fee + "
                "fuel shrinkage, typically €3-5/MWh for European underground storage. "
                "Slide up to stress-test the strategy against higher costs."
            ),
        )

    ttf_hist = fetch_ttf_history(years=7)
    bt = compute_seasonal_strategy(ttf_hist, storage_cost=storage_cost)
    stats = compute_strategy_stats(bt)

    if bt.empty:
        st.warning(
            "TTF historical data unavailable. "
            "Requires yfinance and internet access (ticker: TTF=F)."
        )
    else:
        # ── KPI row ──────────────────────────────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            hit_color = "green" if stats["hit_rate"] >= 60 else ("amber" if stats["hit_rate"] >= 40 else "red")
            st.markdown(
                kpi_card("Hit rate (full sample)", f"{stats['hit_rate']:.0f}%",
                         delta_span(f"profitable in {int(stats['hit_rate']*stats['n_years']/100)}/{stats['n_years']} gas years", hit_color)),
                unsafe_allow_html=True,
            )
        with k2:
            pnl_color = "green" if stats["avg_pnl"] > 0 else "red"
            st.markdown(
                kpi_card("Avg P&L (full sample)", f"€{stats['avg_pnl']:.1f}/MWh",
                         delta_span("includes 2021-22 energy crisis", pnl_color)),
                unsafe_allow_html=True,
            )
        with k3:
            exc = stats.get("avg_pnl_ex_crisis")
            if exc is not None:
                exc_color = "green" if exc > 0 else ("amber" if exc > -2 else "red")
                exc_n     = stats.get("n_ex_crisis", 0)
                st.markdown(
                    kpi_card("Avg P&L (ex-crisis)", f"€{exc:.1f}/MWh",
                             delta_span(f"excl. GY2021-22 · {exc_n} years", exc_color)),
                    unsafe_allow_html=True,
                )
            else:
                sharpe_color = "green" if stats["sharpe"] >= 0.5 else ("amber" if stats["sharpe"] >= 0 else "red")
                st.markdown(
                    kpi_card("Sharpe ratio", f"{stats['sharpe']:.2f}",
                             delta_span("annual P&L / std dev", sharpe_color)),
                    unsafe_allow_html=True,
                )
        with k4:
            st.markdown(
                kpi_card("Avg summer-winter spread", f"€{stats['avg_spread']:.1f}/MWh",
                         delta_span("before storage cost", "blue")),
                unsafe_allow_html=True,
            )

        C = {
            "bg": "#0d1117", "panel_bg": "#161b22",
            "grid": "rgba(255,255,255,0.06)", "text": "#c9d1d9",
            "pos": "#3fb950", "neg": "#f85149",
            "cum": "#58a6ff", "zero": "rgba(255,255,255,0.25)",
            "cost": "rgba(210,180,40,0.55)",
        }

        # ── Annual P&L bar chart ─────────────────────────────────────────────
        bar_colors = [C["pos"] if v >= 0 else C["neg"] for v in bt["pnl"]]

        fig_bt = go.Figure()
        fig_bt.add_trace(go.Bar(
            x=bt["label"],
            y=bt["pnl"],
            marker_color=bar_colors,
            name="Annual P&L",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "P&L: €%{y:.1f}/MWh<br>"
                "<extra></extra>"
            ),
        ))
        fig_bt.add_hline(
            y=0,
            line_dash="solid",
            line_color=C["zero"],
            line_width=0.8,
        )
        fig_bt.update_layout(
            title=dict(text="Annual P&L per MWh (net of storage cost)", font=dict(size=12)),
            template="plotly_dark",
            paper_bgcolor=C["bg"], plot_bgcolor=C["panel_bg"],
            font=dict(color=C["text"], size=12),
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(showgrid=False, title=None),
            yaxis=dict(title="EUR/MWh", showgrid=True, gridcolor=C["grid"]),
            hovermode="x unified",
            height=300,
        )
        st.plotly_chart(fig_bt, use_container_width=True)

        # ── Cumulative P&L line ──────────────────────────────────────────────
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=bt["label"],
            y=bt["cum_pnl"],
            mode="lines+markers",
            line=dict(color=C["cum"], width=2.2),
            marker=dict(size=6, color=C["cum"]),
            name="Cumulative P&L",
            hovertemplate="Cumulative: €%{y:.1f}/MWh<extra></extra>",
        ))
        fig_cum.add_hline(y=0, line_dash="solid", line_color=C["zero"], line_width=0.8)
        fig_cum.update_layout(
            title=dict(text="Cumulative P&L per MWh", font=dict(size=12)),
            template="plotly_dark",
            paper_bgcolor=C["bg"], plot_bgcolor=C["panel_bg"],
            font=dict(color=C["text"], size=12),
            margin=dict(l=10, r=10, t=40, b=10),
            showlegend=False,
            xaxis=dict(showgrid=False, title=None),
            yaxis=dict(title="EUR/MWh", showgrid=True, gridcolor=C["grid"]),
            hovermode="x",
            height=220,
        )
        st.plotly_chart(fig_cum, use_container_width=True)

        # ── Crisis context note ──────────────────────────────────────────────
        exc = stats.get("avg_pnl_ex_crisis")
        if exc is not None and stats["n_years"] > 0:
            crisis_rows = bt[bt["gas_year"].astype(int).isin([2021, 2022])]
            if not crisis_rows.empty:
                crisis_pnl_str = ", ".join(
                    f"{row['label']}: €{row['pnl']:.0f}/MWh"
                    for _, row in crisis_rows.iterrows()
                )
                ex_n = stats.get("n_ex_crisis", 0)
                st.caption(
                    f"Note: GY2021 and GY2022 reflect the European energy crisis, with geopolitical-shock spreads "
                    f"that dominate the full-sample average ({crisis_pnl_str}). "
                    f"Excluding these years, the average P&L across {ex_n} normal gas years is "
                    f"€{exc:.1f}/MWh, which is a more representative picture of steady-state storage economics."
                )

        # ── Results table ────────────────────────────────────────────────────
        with st.expander("Per-year results table", expanded=False):
            display = bt[["label", "summer_avg", "winter_avg", "spread", "pnl"]].copy()
            display.columns = ["Gas Year", "Summer avg (€/MWh)", "Winter avg (€/MWh)",
                                "Spread (€/MWh)", f"P&L net of €{storage_cost:.1f} cost (€/MWh)"]
            st.dataframe(display.set_index("Gas Year"), use_container_width=True)

        # ── Signal: current gas year injection economics ──────────────────────
        if not ttf_hist.empty:
            ttf_hist_dated = ttf_hist.copy()
            ttf_hist_dated["date"] = pd.to_datetime(ttf_hist_dated["date"])
            ttf_hist_dated["month"] = ttf_hist_dated["date"].dt.month
            ttf_hist_dated["year"]  = ttf_hist_dated["date"].dt.year

            today_ts = pd.Timestamp.now()
            # Gas year label = calendar year of summer half (Apr onwards)
            gy_start = today_ts.year if today_ts.month >= 4 else today_ts.year - 1

            curr_summer = ttf_hist_dated[
                (ttf_hist_dated["year"] == gy_start) &
                (ttf_hist_dated["month"].between(4, 9))
            ]
            curr_price = float(ttf_hist_dated["price"].iloc[-1])

            if not curr_summer.empty:
                curr_summer_avg = float(curr_summer["price"].mean())
                n_summer_days   = len(curr_summer)
                breakeven_winter = round(curr_summer_avg + storage_cost, 1)
                st.markdown(
                    commentary(
                        f"GY {gy_start}/{str(gy_start + 1)[-2:]}: summer injection season underway "
                        f"({n_summer_days} trading days, average summer price so far: "
                        f"€{curr_summer_avg:.1f}/MWh). "
                        f"Break-even winter average (summer avg + €{storage_cost:.1f}/MWh storage cost): "
                        f"€{breakeven_winter:.1f}/MWh. "
                        f"Winter forwards above this level imply a profitable storage trade.",
                        "ok",
                    ),
                    unsafe_allow_html=True,
                )

        with st.expander("Strategy methodology", expanded=True):
            st.markdown(f"""
            **Strategy:** Seasonal carry trade modelled from the perspective of a physical storage
            operator (always-in). For each gas year, gas is injected at summer prices (Apr-Sep) and
            withdrawn at winter prices (Oct-Mar). The net P&L per MWh is the realised
            winter-minus-summer average price spread, minus the round-trip cost of storage.
            A financial trader would only enter when the spread exceeds storage cost; a physical
            operator runs the position every year regardless.

            **Gas year definition:** April 1 of year Y to March 31 of year Y+1.
            - Summer half: April–September (injection season)
            - Winter half: October–March (withdrawal season)

            **P&L calculation:**
            `P&L = mean(Winter prices) − mean(Summer prices) − storage_cost`

            **Storage cost (€{storage_cost:.1f}/MWh):** Set via the slider above.
            Round-trip European underground storage typically costs €3-5/MWh, covering injection
            and withdrawal tariffs, fuel shrinkage, and financing costs. Higher for LNG re-gassing.

            **Proxy:** TTF front-month continuous futures (Yahoo Finance TTF=F). The front-month
            price is used as a proxy for the average injection/withdrawal price. A more precise
            backtest would use the specific summer and winter futures contracts (e.g., ICE TTF
            Q3 and Q1/Q2 strips), but front-month provides a directionally accurate proxy.

            **Sharpe ratio:** Computed as mean(annual P&L) / std(annual P&L), treating each
            gas year as one independent observation. Not annualised further.

            **Limitations:** Does not model:
            - Cost of capital (margin requirements on ICE TTF futures)
            - Basis risk between front-month and actual strip prices
            - Storage capacity constraints or fill-level path dependency
            - Transaction costs beyond the storage cost assumption
            Use as a directional indicator of seasonal spread dynamics, not as a precise
            trading P&L estimate.

            **Data:** ICE TTF front-month continuous futures via Yahoo Finance (TTF=F),
            approximately {len(ttf_hist):,} trading days of history.
            """)

        st.caption("Source: ICE/Yahoo Finance (TTF=F) | agsi.gie.eu")


# ════════════════════════════════════════════════════════════════════════════
# TAB 5: MERIT ORDER / SUPPLY STACK
# ════════════════════════════════════════════════════════════════════════════
with tab_stack:
    st.markdown("### German Merit Order — Supply Stack")
    st.caption(
        "German generation fleet ordered by short-run marginal cost (SRMC). "
        "Demand line shows where on the merit order current consumption clears. "
        "The intersecting fuel sets the theoretical clearing price. "
        "Capacities from Bundesnetzagentur 2024 survey; gas SRMC is live TTF-derived."
    )

    # ── Inputs ───────────────────────────────────────────────────────────────
    ttf_prices = ttf.get("prices", pd.DataFrame())
    ttf_latest = float(ttf_prices["price"].iloc[-1]) if not ttf_prices.empty else 50.0

    _live_load_gw, _load_src = _get_de_live_load_gw()
    _default_demand = (
        float(np.clip(round(_live_load_gw), 30, 90))
        if _live_load_gw is not None
        else 60.0
    )

    col_left, col_right = st.columns([2, 1])
    with col_left:
        demand_gw = st.slider(
            "Modelled demand (GW)",
            min_value=30.0, max_value=90.0, value=_default_demand, step=1.0,
            help=(
                "Default set to today's ENTSO-E A65 German actual load average. "
                "German grid load varies from ~35 GW (summer weekend night) to ~85 GW "
                "(cold winter peak). Slide to override."
            ),
        )
    with col_right:
        _load_badge = (
            f"Live demand: {_live_load_gw:.1f} GW ({_load_src})"
            if _live_load_gw is not None
            else f"Live demand unavailable — {_load_src}. Using 60 GW default."
        )
        st.markdown(
            commentary(
                f"TTF input: €{ttf_latest:.1f}/MWh (live). "
                f"EUA carbon price fetched below. {_load_badge}",
                "ok" if _live_load_gw is not None else "warn",
            ),
            unsafe_allow_html=True,
        )

    # ── Fetch EUA, build stack ────────────────────────────────────────────────
    eua_price, eua_source = _get_eua()
    stack_df  = build_stack(ttf_latest, eua_price)
    marginal  = identify_marginal_fuel(stack_df, demand_gw)

    # ── KPI row ──────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)

    nl_latest: float | None = None
    if not spot_df.empty:
        nl_row = spot_df[spot_df["zone"] == "NL"].sort_values("date")
        if not nl_row.empty:
            nl_latest = float(nl_row["price_eur_mwh"].iloc[-1])

    gas_srmc = ttf_latest / 0.55 + eua_price * 0.37
    coal_srmc = 11.0 / 0.40 + eua_price * 0.85

    with k1:
        st.markdown(
            kpi_card("TTF gas", f"€{ttf_latest:.1f}/MWh",
                     delta_span("live front-month", "blue")),
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            kpi_card("EUA carbon", f"€{eua_price:.0f}/t CO₂",
                     delta_span(eua_source, "blue")),
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            kpi_card("Gas CCGT SRMC", f"€{gas_srmc:.1f}/MWh",
                     delta_span("TTF÷0.55 + EUA×0.37", "amber")),
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            kpi_card("Marginal fuel", marginal["label"].split("/")[0].strip(),
                     delta_span(f"€{marginal['marginal_cost']:.1f}/MWh implied", "amber")),
            unsafe_allow_html=True,
        )
    with k5:
        if nl_latest is not None:
            premium = nl_latest - marginal["marginal_cost"]
            # Sign-aware interpretation: depends on which fuel is marginal
            _zero_cost_fuels = {"wind", "solar", "hydro"}
            if marginal["fuel"] in _zero_cost_fuels:
                # NL > €0 implied is NOT scarcity — it's the renewable-surplus disconnect
                label = "above renewable clearing" if premium > 0 else "below renewable clearing"
            elif premium > 10:
                label = "above thermal cost floor"
            elif premium < -10:
                label = "below thermal cost floor"
            else:
                label = "near implied"
            color = "red" if premium > 10 else ("green" if premium < -10 else "blue")
            st.markdown(
                kpi_card("NL vs implied", f"€{nl_latest:.1f}/MWh",
                         delta_span(f"{premium:+.1f} EUR/MWh — {label}", color)),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                kpi_card("NL actual", "n/a",
                         delta_span("spot data unavailable", "amber")),
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Merit order chart ─────────────────────────────────────────────────────
    fig_stack = make_merit_order_figure(
        stack_df, demand_gw, marginal, actual_power_price=nl_latest,
    )
    st.plotly_chart(fig_stack, use_container_width=True)

    # ── Commentary ───────────────────────────────────────────────────────────
    if marginal["fuel"] == "gas":
        prose = (
            f"At {demand_gw:.0f} GW demand, the marginal fuel is gas CCGT at an implied clearing price of "
            f"€{marginal['marginal_cost']:.1f}/MWh (TTF €{ttf_latest:.1f}÷0.55 + EUA €{eua_price:.0f}×0.37). "
            f"Gas CCGT fleet is approximately {marginal['utilisation_pct']:.0f}% utilised within its capacity block. "
            "This is the gas-marginal regime: power prices and TTF are highly correlated, and the clean-spark "
            "spread is the key signal for generation economics."
        )
        status = "warn"
    elif marginal["fuel"] == "coal":
        prose = (
            f"At {demand_gw:.0f} GW demand, the marginal fuel is hard coal at an implied clearing price of "
            f"€{marginal['marginal_cost']:.1f}/MWh. Gas CCGT SRMC (€{gas_srmc:.1f}/MWh) is above coal, "
            "meaning gas plant is off the margin — a coal-marginal regime. The clean-dark spread is the "
            "relevant signal for generation economics in this demand scenario."
        )
        status = "warn"
    elif marginal["fuel"] in ("wind", "solar"):
        _gap_note = ""
        if nl_latest is not None:
            _gap = nl_latest - marginal["marginal_cost"]
            _gap_note = (
                f" NL day-ahead at €{nl_latest:.1f}/MWh is €{_gap:.1f}/MWh above the structural cost "
                "floor of €0/MWh — this is not a scarcity premium. It reflects grid constraints, "
                "reserve capacity costs, import flows (NordLink, NorNed), or real-time demand that "
                "the simplified merit order does not capture."
            )
        prose = (
            f"At {demand_gw:.0f} GW demand, the marginal fuel is renewable (zero SRMC). "
            "Negative or near-zero wholesale prices are possible when renewable output is high and "
            f"demand is low.{_gap_note}"
        )
        status = "ok"
    else:
        prose = (
            f"At {demand_gw:.0f} GW demand, the marginal fuel is {marginal['label']} at "
            f"€{marginal['marginal_cost']:.1f}/MWh."
        )
        status = "ok"

    st.markdown(commentary(prose, status), unsafe_allow_html=True)

    # ── Methodology ──────────────────────────────────────────────────────────
    with st.expander("Methodology and assumptions", expanded=False):
        st.markdown(f"""
        **Model:** Static merit order. Generation sources are ranked by short-run marginal cost (SRMC)
        and stacked left-to-right. The demand line intersects the stack at the marginal fuel, whose
        SRMC sets the theoretical day-ahead clearing price.

        **Dynamic inputs:**
        - Gas CCGT SRMC = TTF ÷ 0.55 + EUA × 0.37 (live TTF: **€{ttf_latest:.1f}/MWh**, EUA: **€{eua_price:.0f}/t**)
        - Hard coal SRMC = 11 EUR/MWh_th ÷ 0.40 + EUA × 0.85 ≈ **€{coal_srmc:.1f}/MWh** at current EUA
        - Demand default: **{_default_demand:.0f} GW** — {"ENTSO-E A65 today's average (" + _load_src + ")" if _live_load_gw is not None else "static 60 GW fallback (" + _load_src + ")"}

        **Static inputs (Bundesnetzagentur 2024):**
        - Wind: 66 GW installed | Solar: 73 GW | Hydro: 4.5 GW | Biomass: 8.5 GW
        - Lignite: 17 GW (€20/MWh) | Hard coal: 23 GW | Gas CCGT/OCGT: 30 GW | Oil peakers: 2.5 GW
        - Nuclear: 0 GW (all plants retired April 2023)

        **EUA source:** {eua_source}

        **Key simplifications:**
        - This model shows installed capacity, not available capacity. Typical thermal availability
          is 85-90%; renewables depend on weather conditions.
        - Wind and solar produce at weather-dependent output levels — their effective capacity at any
          given hour is not 66/73 GW. The stack shows the structural position, not real-time dispatch.
        - No cross-border imports or exports. Germany regularly imports from France and Scandinavia,
          which can suppress the clearing price below the domestic marginal fuel SRMC.
        - No must-run obligations, ramp constraints, or reserve requirements.
        - Hard coal fuel cost is a market-average estimate (~API2 reference). Actual bilateral
          contract prices may differ.

        **Scarcity premium:** Difference between NL actual day-ahead price and the model-implied
        clearing price. A persistent positive premium indicates factors not captured by the pure
        merit order (scarcity, congestion, forward risk premium). A persistent negative premium
        indicates renewable surplus, strong imports, or weak demand relative to installed capacity.

        **Data sources:** Bundesnetzagentur Kraftwerksliste 2024 | ICE/Yahoo Finance (TTF=F, CO2.L) |
        Nord Pool Data Portal (NL day-ahead)
        """)

    st.caption(
        "Sources: Bundesnetzagentur 2024 | ICE/Yahoo Finance (TTF, EUA) | Nord Pool (NL day-ahead)"
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 6: NORDIC PRICE DECOMPOSITION
# ════════════════════════════════════════════════════════════════════════════
with tab_decomp:
    st.markdown("### Nordic Price Decomposition")
    st.caption(
        "Rolling 90-day multivariate OLS decomposes the NO2 day-ahead price into contributions "
        "from continental price (NL), gas price (TTF), and — where available — Norwegian hydro "
        "reservoir level and German wind output. Regressors are standardised so beta magnitudes "
        "are directly comparable across drivers."
    )

    # Lazy-load feature matrix: only assembled when this tab is first visited
    if "features_l2" not in st.session_state:
        with st.spinner("Assembling feature matrix (first visit — subsequent loads are instant)…"):
            st.session_state["features_l2"] = _get_features_l2()
    features_df = st.session_state["features_l2"]

    _avail_l2 = get_available_feature_sets(features_df)
    _n_rows_l2 = len(features_df)

    if features_df.empty or _n_rows_l2 < 100:
        st.warning(
            "Insufficient feature data for decomposition. "
            "Requires at least 100 trading days of NO2, NL, and TTF prices."
        )
    else:
        # Run rolling decomposition (cached — fast on subsequent slider interactions)
        _t_decomp0 = _time.perf_counter()
        decomp_results, feat_cols, model_label = _run_decomp_cached(features_df, window=90)
        _decomp_ms = (_time.perf_counter() - _t_decomp0) * 1000
        _decomp_tag = "disk cache" if _decomp_ms < 500 else "computed"

        # Model mode banner
        if len(feat_cols) < 4:
            st.info(
                f"Active: **{model_label}**. "
                + ("Hydro and wind data unavailable — ENTSO-E key required for full 4-factor model."
                   if len(feat_cols) == 2 else
                   "Wind data unavailable — ENTSO-E key or Fraunhofer fallback required for 4-factor model.")
                + f" | Rolling OLS: {_decomp_tag}, {_decomp_ms/1000:.1f}s",
                icon="ℹ️",
            )
        else:
            st.success(
                f"Active: **{model_label}**. | Rolling OLS: {_decomp_tag}, {_decomp_ms/1000:.1f}s",
                icon="✅",
            )

        if decomp_results.empty:
            st.warning("Rolling OLS could not converge. Check that statsmodels is installed.")
        else:
            # ── KPI row ──────────────────────────────────────────────────────
            driver_key, driver_beta = dominant_driver(decomp_results, feat_cols)
            contribs   = current_contributions(features_df, decomp_results, feat_cols)

            latest_no2 = float(features_df.sort_values("date")["no2"].dropna().iloc[-1])
            latest_r2  = float(decomp_results["r2"].dropna().iloc[-1])

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.markdown(
                    kpi_card("NO2 latest", f"€{latest_no2:.1f}/MWh",
                             delta_span("day-ahead price", "blue")),
                    unsafe_allow_html=True,
                )
            with k2:
                st.markdown(
                    kpi_card("Model R²", f"{latest_r2:.2f}",
                             delta_span(f"90-day rolling window", "blue")),
                    unsafe_allow_html=True,
                )
            with k3:
                driver_label = FACTOR_LABELS.get(driver_key, driver_key)
                st.markdown(
                    kpi_card("Dominant driver", driver_label.split("(")[0].strip(),
                             delta_span(f"β = {driver_beta:.2f} EUR/MWh per σ", "amber")),
                    unsafe_allow_html=True,
                )
            with k4:
                total_explained = sum(abs(v) for v in contribs.values())
                st.markdown(
                    kpi_card("Total driver signal", f"€{total_explained:.1f}/MWh",
                             delta_span("sum of |contributions|", "blue")),
                    unsafe_allow_html=True,
                )

            st.divider()

            # ── Beta time series chart ────────────────────────────────────────
            st.markdown("#### Rolling 90-Day Beta Coefficients")
            st.caption(
                "Each line shows how strongly that factor is driving NO2 over the trailing 90-day window. "
                "Positive β: factor pushing prices up. Negative β: factor suppressing prices. "
                "Unit: EUR/MWh per one standard deviation of the factor."
            )
            fig_beta = make_beta_chart(decomp_results, feat_cols, window=90)
            st.plotly_chart(fig_beta, use_container_width=True)

            # ── Today's contribution bar ──────────────────────────────────────
            if contribs:
                st.markdown("#### Today's Factor Contributions")
                st.caption(
                    "Each bar shows β_i × z_i — how much that driver is currently adding or subtracting "
                    "from the model-implied NO2 price, in EUR/MWh."
                )
                fig_contrib = make_contribution_bar(contribs, no2_actual=latest_no2)
                st.plotly_chart(fig_contrib, use_container_width=True)

            # ── Interpretation ────────────────────────────────────────────────
            if driver_key in contribs:
                driver_contrib = contribs[driver_key]
                direction = "upward" if driver_contrib > 0 else "downward"
                st.markdown(
                    commentary(
                        f"Current dominant driver: {FACTOR_LABELS.get(driver_key, driver_key)}. "
                        f"Over the trailing 90-day window, a one-standard-deviation move in this factor "
                        f"corresponds to a {driver_beta:.2f} EUR/MWh move in NO2. "
                        f"At current levels, it is exerting {abs(driver_contrib):.1f} EUR/MWh of "
                        f"{direction} price pressure. "
                        f"Rolling R² = {latest_r2:.2f} — the {model_label} explains "
                        f"{latest_r2*100:.0f}% of NO2 variance over the trailing window.",
                        "ok",
                    ),
                    unsafe_allow_html=True,
                )

            # ── Methodology ───────────────────────────────────────────────────
            with st.expander("Methodology", expanded=False):
                factor_list = " | ".join(
                    f"`{c}` ({FACTOR_LABELS.get(c, c)})" for c in feat_cols
                )
                st.markdown(f"""
                **Model:** Rolling ordinary least squares (OLS), 90-day window.

                `NO2 = α + β₁·NL + β₂·TTF{' + β₃·hydro_pct' if 'hydro_pct' in feat_cols else ''}{' + β₄·de_wind_gwh' if 'de_wind_gwh' in feat_cols else ''} + ε`

                **Active factors:** {factor_list}

                **Standardisation:** All regressors are globally standardised (mean 0, std 1 over
                the full sample) before fitting, so beta magnitudes represent EUR/MWh per one-
                standard-deviation change. This makes the coefficients directly comparable across
                factors with different units (EUR/MWh for NL/TTF, % for hydro, GWh for wind).

                **Rolling window:** 90 trading days (~4.5 months). Shorter windows capture recent
                regime shifts; longer windows are more stable. The window was chosen to balance
                responsiveness with statistical stability. Minimum 45 observations required.

                **Interpretation of betas:**
                - NL β > 0: When continental European prices are high, Nordic prices track higher
                  (price coupling via NordLink, NorNed, Viking Link).
                - TTF β > 0: Gas-price pass-through to Nordic prices (gas-fired peakers set the
                  marginal price when hydro is tight or demand is high).
                - hydro β < 0: Higher reservoir levels → more hydro dispatch → lower prices.
                  Negative beta is expected and confirms hydro as a supply-side depressant.
                - wind β < 0: Higher German wind → lower continental prices via spillover →
                  lower NO2 via interconnector coupling.

                **Factor contributions:** β_i × z_i, where z_i = (x_i − μ_i) / σ_i at today's
                value. Represents the model-implied price contribution of each driver given current
                conditions. Contributions do not sum to the actual price level (the intercept term
                and unexplained residual account for the remainder).

                **Feature data:** {_n_rows_l2} trading days assembled.
                Feature availability: {"hydro ✅" if _avail_l2.get("hydro") else "hydro ❌"} |
                {"wind ✅" if _avail_l2.get("wind") else "wind ❌ (ENTSO-E key required)"}.

                **Forward link — Phase D cointegration scanner:** The same rolling regression
                mechanism, extended with Engle-Granger cointegration tests, will underpin the
                NO2–NL pair-trading signal (expected spread mean-reversion speed and entry
                z-score thresholds).

                **Data sources:** ENTSO-E A44 (NO2, NL day-ahead prices) | ICE/Yahoo Finance (TTF) |
                ENTSO-E B31 (hydro reservoirs) | Fraunhofer ISE energy-charts.info (DE wind)
                """)

            st.caption(
                "Sources: ENTSO-E A44 (day-ahead prices) | ICE/Yahoo Finance (TTF) | "
                "ENTSO-E B31 (hydro) | Fraunhofer ISE (wind)"
            )


# ════════════════════════════════════════════════════════════════════════════
# TAB 7: WIND FORECAST ERROR TRACKER
# ════════════════════════════════════════════════════════════════════════════
with tab_wind:
    st.markdown("### Wind Forecast Error Tracker")
    st.caption(
        "ENTSO-E A69 day-ahead wind generation forecast versus actual generation. "
        "Tracks how well TSO wind forecasts perform and whether large forecast misses "
        "coincide with elevated intraday price volatility."
    )

    # ── Load feature matrix (lazy — only when this tab is visited) ───────────
    if "features_l2" not in st.session_state:
        with st.spinner("Assembling feature matrix (first visit — subsequent loads are instant)…"):
            st.session_state["features_l2"] = _get_features_l2()
    features_df = st.session_state["features_l2"]

    # ── Load wind data ────────────────────────────────────────────────────────
    @st.cache_data(ttl=3600, show_spinner=False, persist="disk")
    def _get_wind_forecast():
        return fetch_wind_forecast(days=90)

    @st.cache_data(ttl=3600, show_spinner=False, persist="disk")
    def _get_wind_actual():
        return fetch_wind_daily(days=90)

    with st.spinner("Loading wind forecast and actual data…"):
        fc_data  = _get_wind_forecast()
        act_data = _get_wind_actual()

    fc_df     = fc_data.get("forecast", pd.DataFrame())
    fc_zones  = fc_data.get("zones", [])
    fc_source = fc_data.get("source", "unavailable")
    act_df    = act_data.get("daily", pd.DataFrame())

    # ── Source banners ────────────────────────────────────────────────────────
    if fc_source == "unavailable":
        st.warning(
            "Wind forecast data (ENTSO-E A69) is unavailable. "
            "Possible causes: no ENTSOE_API_KEY set, or ENTSO-E server instability "
            "(A69 wind forecast endpoint may be affected by the ongoing platform issues "
            "also affecting B16/B19 actual generation). Check API key in settings.",
            icon="⚠️",
        )
    else:
        _zones_str = ", ".join(WIND_ZONE_LABELS.get(z, z) for z in fc_zones)
        st.success(
            f"Forecast data loaded from {fc_source}. "
            f"Zones with data: {_zones_str}.",
            icon="✅",
        )

    if fc_df.empty or act_df.empty or not fc_zones:
        st.info(
            "Both forecast (ENTSO-E A69) and actual (ENTSO-E B18/B19 or Fraunhofer ISE) "
            "data are required for error computation. Actual DE wind is available via "
            "Fraunhofer ISE fallback — forecast data requires an active ENTSO-E API key.",
            icon="ℹ️",
        )
    else:
        # ── Compute errors ────────────────────────────────────────────────────
        error_df = compute_errors(fc_df, act_df, fc_zones)
        rmse_df  = compute_rolling_rmse(error_df, fc_zones, window=7)

        if error_df.empty:
            st.warning(
                "Could not compute forecast errors — no overlapping dates between "
                "forecast and actual data. This typically means the zones with "
                "forecast data (e.g. DK1, NO, GB) do not have corresponding actual "
                "generation data available via the current fallback chain."
            )
        else:
            # ── KPI row ───────────────────────────────────────────────────────
            kpi_cols = st.columns(len(fc_zones) + 1)
            with kpi_cols[0]:
                latest_date = error_df["date"].max()
                st.markdown(
                    kpi_card("Latest data", str(latest_date),
                             delta_span(f"{len(error_df)} days", "blue")),
                    unsafe_allow_html=True,
                )
            for i, zone in enumerate(fc_zones):
                col = f"{zone}_error_gwh"
                if col not in error_df.columns:
                    continue
                latest_err = float(error_df[col].dropna().iloc[-1]) if not error_df[col].dropna().empty else 0.0
                mean_abs   = float(error_df[col].abs().mean()) if not error_df[col].dropna().empty else 0.0
                color = "red" if abs(latest_err) > mean_abs * 1.5 else "blue"
                with kpi_cols[i + 1]:
                    st.markdown(
                        kpi_card(
                            WIND_ZONE_LABELS.get(zone, zone),
                            f"{latest_err:+.1f} GWh",
                            delta_span(f"mean |err| {mean_abs:.1f} GWh", color),
                        ),
                        unsafe_allow_html=True,
                    )

            st.divider()

            # ── Error bar chart ───────────────────────────────────────────────
            st.markdown("#### Daily Forecast Error (last 60 days)")
            st.caption("Red = over-forecast (forecast > actual). Green = under-forecast.")
            fig_err = make_error_bar_chart(error_df, fc_zones, last_n_days=60)
            st.plotly_chart(fig_err, use_container_width=True)

            # ── RMSE chart ────────────────────────────────────────────────────
            if not rmse_df.empty:
                st.markdown("#### Rolling 7-Day RMSE by Country")
                st.caption(
                    "Lower RMSE = more accurate wind forecast. "
                    "RMSE spikes typically coincide with rapid weather pattern changes."
                )
                fig_rmse = make_rmse_chart(rmse_df, fc_zones)
                st.plotly_chart(fig_rmse, use_container_width=True)

            # ── Correlation scatter ───────────────────────────────────────────
            primary_zone = fc_zones[0] if fc_zones else "DE"
            corr_df = compute_price_correlation(error_df, features_df, zone=primary_zone, window=30)

            if not corr_df.empty:
                st.markdown(f"#### Forecast Error vs Price Volatility — {WIND_ZONE_LABELS.get(primary_zone, primary_zone)}")
                st.caption(
                    "Tests whether large wind forecast misses coincide with large next-day price moves. "
                    "Rolling 30-day Pearson correlation annotated. Significant positive correlation "
                    "implies wind forecast error is a viable intraday volatility predictor."
                )
                fig_corr = make_correlation_scatter(corr_df, zone=primary_zone)
                st.plotly_chart(fig_corr, use_container_width=True)

                latest_rho = corr_df["rolling_corr"].dropna().iloc[-1] if not corr_df["rolling_corr"].dropna().empty else float("nan")
                if not np.isnan(latest_rho):
                    if abs(latest_rho) > 0.3:
                        interpretation = (
                            f"Rolling 30-day correlation ρ = {latest_rho:.2f} — statistically meaningful. "
                            "Large wind forecast misses are co-moving with elevated power price volatility in "
                            "the current period. This is consistent with the theory that wind forecast errors "
                            "create imbalances that require expensive real-time correction."
                        )
                        status = "warn" if latest_rho > 0.3 else "ok"
                    else:
                        interpretation = (
                            f"Rolling 30-day correlation ρ = {latest_rho:.2f} — weak relationship. "
                            "Wind forecast errors are not strongly co-moving with price volatility in the "
                            "current period. Other drivers (storage levels, gas price, continental flows) "
                            "are likely dominating price formation."
                        )
                        status = "ok"
                    st.markdown(commentary(interpretation, status), unsafe_allow_html=True)

    # ── Methodology ───────────────────────────────────────────────────────────
    with st.expander("Methodology", expanded=False):
        st.markdown("""
        **Wind forecast data:** ENTSO-E A69 (Day-Ahead Aggregated Generation — Wind and Solar),
        queried via `query_wind_and_solar_forecast()` in `entsoe-py`. Onshore (B19) + offshore
        (B18) PSR types are summed to daily GWh.

        **Actual wind data:** ENTSO-E B18/B19 for non-DE zones. Germany (DE) actual wind is
        sourced from Fraunhofer ISE energy-charts.info as primary (more reliable than ENTSO-E
        B18/B19 which are subject to ongoing server instability).

        **Error definition:** `forecast_error = forecast_gwh − actual_gwh`
        - Positive error = TSO over-predicted wind (more dispatch available than expected)
        - Negative error = TSO under-predicted wind (less dispatch than expected)

        **Error percentage:** `forecast_error / actual_gwh × 100`. Undefined when actual = 0 (shown as n/a).

        **Rolling RMSE:** `sqrt( mean( error² ) )` over a 7-day trailing window.

        **Price correlation:** Pearson correlation between `|forecast_error_gwh|` and
        `|day-over-day NO2 price change (EUR/MWh)|` over a 30-day trailing window.
        The NO2 price change is used as a proxy for intraday/next-day market volatility —
        a direct intraday continuous price series requires REMIT / Epex intraday data.

        **ENTSO-E A69 vs B16/B19 availability:** A69 (forecasts) and B16/B19 (actuals) are
        separate document types that may be served from different ENTSO-E platform components.
        If A69 is unavailable, the tab shows a clear error banner rather than silent null data.

        **Countries:** DE (Fraunhofer actuals, A69 forecast), DK1 (ENTSO-E both), NO (ENTSO-E both),
        GB (ENTSO-E both — availability may vary post-Brexit).
        """)

    st.caption(
        "Sources: ENTSO-E A69 (wind forecast) | ENTSO-E B18/B19 (actuals) | "
        "Fraunhofer ISE energy-charts.info (DE actuals fallback)"
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 8: GRANGER CAUSALITY — SENTIMENT → TTF
# ════════════════════════════════════════════════════════════════════════════
with tab_granger:
    st.markdown("### Sentiment → TTF Granger Causality")
    st.caption(
        "Tests whether lagged energy news sentiment Granger-causes next-day TTF price returns. "
        "A statistically significant result (p < 0.05) means past sentiment adds predictive power "
        "beyond TTF's own history — a potential leading indicator for gas price direction."
    )

    with st.spinner("Loading sentiment history…"):
        _sent = _get_sentiment_l2()

    _daily_sent = _sent.get("daily", pd.DataFrame())
    _sent_error = _sent.get("error")

    if _sent_error and _daily_sent.empty:
        st.warning(
            f"Sentiment pipeline unavailable: {_sent_error} "
            "This tab requires the FinBERT model (HuggingFace Spaces deployment) and "
            "at least 21 days of scored headlines.",
            icon="⚠️",
        )
    else:
        _ttf_prices_df = ttf.get("prices", pd.DataFrame())

        if _ttf_prices_df.empty:
            st.warning("TTF price data unavailable — cannot compute Granger test.", icon="⚠️")
        elif _daily_sent.empty:
            st.info(
                "No sentiment history yet. Granger test requires at least 21 days of scored headlines. "
                "History accumulates automatically on each page load.",
                icon="ℹ️",
            )
        else:
            # ── Build merged series ───────────────────────────────────────────
            _ttf_df = _ttf_prices_df[["date", "price"]].copy()
            _ttf_df["date"] = pd.to_datetime(_ttf_df["date"])
            _ttf_df["return"] = _ttf_df["price"].pct_change() * 100

            _daily_sent = _daily_sent.copy()
            _daily_sent["date"] = pd.to_datetime(_daily_sent["date"])

            _merged = _daily_sent[["date", "net_sentiment"]].merge(
                _ttf_df[["date", "return"]], on="date", how="inner"
            ).dropna().sort_values("date")

            _MIN_DAYS = 21
            _n_merged = len(_merged)

            if _n_merged < _MIN_DAYS:
                _days_needed = _MIN_DAYS - _n_merged
                st.info(
                    f"Granger test requires {_MIN_DAYS} aligned trading days with both sentiment "
                    f"and TTF return data. Currently have **{_n_merged}** — "
                    f"**{_days_needed} more** needed before the test can run. "
                    "Sentiment history accumulates automatically.",
                    icon="ℹ️",
                )
                # Show what we have so far
                if _n_merged > 0:
                    st.markdown("**Available data preview:**")
                    st.dataframe(
                        _merged.tail(10)[["date", "net_sentiment", "return"]].rename(columns={
                            "date": "Date", "net_sentiment": "Net Sentiment", "return": "TTF Return (%)"
                        }),
                        use_container_width=True, hide_index=True,
                    )
            else:
                # ── Run Granger test ──────────────────────────────────────────
                _maxlag = min(7, _n_merged // 7)
                try:
                    from statsmodels.tsa.stattools import grangercausalitytests
                    _gc = grangercausalitytests(
                        _merged[["return", "net_sentiment"]], maxlag=_maxlag, verbose=False
                    )
                    _p_vals = {
                        lag: _gc[lag][0]["ssr_ftest"][1]
                        for lag in range(1, _maxlag + 1)
                    }
                    _f_vals = {
                        lag: _gc[lag][0]["ssr_ftest"][0]
                        for lag in range(1, _maxlag + 1)
                    }
                    _min_p    = min(_p_vals.values())
                    _best_lag = min(_p_vals, key=_p_vals.get)

                    # ── KPI row ───────────────────────────────────────────────
                    _k1, _k2, _k3, _k4 = st.columns(4)
                    with _k1:
                        st.markdown(
                            kpi_card("Best p-value", f"{_min_p:.3f}",
                                     delta_span(f"at lag {_best_lag}d", "blue")),
                            unsafe_allow_html=True,
                        )
                    with _k2:
                        _sig_label = "Significant (p<0.05)" if _min_p < 0.05 else (
                            "Borderline (p<0.10)" if _min_p < 0.10 else "Not significant"
                        )
                        _sig_color = "red" if _min_p < 0.05 else ("amber" if _min_p < 0.10 else "blue")
                        st.markdown(
                            kpi_card("Verdict", _sig_label,
                                     delta_span("SSR F-test", _sig_color)),
                            unsafe_allow_html=True,
                        )
                    with _k3:
                        _f_best = _f_vals[_best_lag]
                        st.markdown(
                            kpi_card("F-statistic", f"{_f_best:.2f}",
                                     delta_span(f"lag {_best_lag}d", "blue")),
                            unsafe_allow_html=True,
                        )
                    with _k4:
                        st.markdown(
                            kpi_card("Sample size", f"{_n_merged} days",
                                     delta_span(f"max lag: {_maxlag}d", "blue")),
                            unsafe_allow_html=True,
                        )

                    st.divider()

                    # ── P-value bar chart ─────────────────────────────────────
                    st.markdown("#### Granger Test P-Values by Lag")
                    st.caption(
                        "Each bar shows the SSR F-test p-value for whether net sentiment at that lag "
                        "has incremental predictive power for next-day TTF returns. "
                        "Bars below the red line (p=0.05) indicate statistical significance."
                    )

                    _lags  = list(_p_vals.keys())
                    _pvals = list(_p_vals.values())
                    _bar_colors = [
                        "#f85149" if p < 0.05 else ("#e07b39" if p < 0.10 else "#58a6ff")
                        for p in _pvals
                    ]

                    _fig_gc = go.Figure()
                    _fig_gc.add_trace(go.Bar(
                        x=[f"Lag {lag}d" for lag in _lags],
                        y=_pvals,
                        marker_color=_bar_colors,
                        text=[f"p={p:.3f}" for p in _pvals],
                        textposition="outside",
                        hovertemplate="Lag %{x}: p=%{y:.4f}<extra></extra>",
                    ))
                    _fig_gc.add_hline(
                        y=0.05, line=dict(color="rgba(248,81,73,0.8)", width=1.5, dash="dash"),
                        annotation_text="p=0.05 significance threshold",
                        annotation_font=dict(color="rgba(248,81,73,0.8)", size=10),
                        annotation_position="top right",
                    )
                    _fig_gc.add_hline(
                        y=0.10, line=dict(color="rgba(210,153,34,0.5)", width=1, dash="dot"),
                        annotation_text="p=0.10 borderline",
                        annotation_font=dict(color="rgba(210,153,34,0.5)", size=10),
                        annotation_position="bottom right",
                    )
                    _fig_gc.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="#0d1117",
                        plot_bgcolor="#161b22",
                        xaxis=dict(title="Lag", gridcolor="rgba(255,255,255,0.06)"),
                        yaxis=dict(
                            title="p-value (SSR F-test)",
                            range=[0, min(1.05, max(_pvals) * 1.2)],
                            gridcolor="rgba(255,255,255,0.06)",
                        ),
                        margin=dict(l=60, r=20, t=30, b=50),
                        height=360,
                        showlegend=False,
                    )
                    st.plotly_chart(_fig_gc, use_container_width=True)

                    # ── Sentiment + TTF return overlay ────────────────────────
                    st.markdown("#### Net Sentiment vs TTF Daily Return")
                    st.caption(
                        "Visual check: are large sentiment moves followed by TTF price moves? "
                        "Secondary axis (right) shows TTF daily return."
                    )
                    _fig_ts = go.Figure()
                    _fig_ts.add_trace(go.Scatter(
                        x=_merged["date"], y=_merged["net_sentiment"],
                        name="Net Sentiment",
                        line=dict(color="#58a6ff", width=1.5),
                        hovertemplate="Sentiment: %{y:.2f}<extra></extra>",
                    ))
                    _fig_ts.add_trace(go.Scatter(
                        x=_merged["date"], y=_merged["return"],
                        name="TTF Return (%)",
                        line=dict(color="#e07b39", width=1.2),
                        yaxis="y2",
                        hovertemplate="TTF return: %{y:.2f}%<extra></extra>",
                    ))
                    _fig_ts.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="#0d1117",
                        plot_bgcolor="#161b22",
                        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                        yaxis=dict(
                            title="Net sentiment",
                            gridcolor="rgba(255,255,255,0.06)",
                            side="left",
                        ),
                        yaxis2=dict(
                            title="TTF daily return (%)",
                            overlaying="y", side="right",
                            showgrid=False,
                        ),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                        margin=dict(l=60, r=60, t=30, b=50),
                        height=320,
                        hovermode="x unified",
                    )
                    st.plotly_chart(_fig_ts, use_container_width=True)

                    # ── Commentary ────────────────────────────────────────────
                    if _min_p < 0.05:
                        _prose = (
                            f"Granger causality is statistically significant at lag {_best_lag}d "
                            f"(p={_min_p:.3f}, F={_f_best:.2f}). "
                            "This means that past energy news sentiment contains incremental predictive "
                            "information for next-day TTF returns, beyond TTF's own lagged values. "
                            "In practical terms: when sentiment deteriorates sharply, a TTF upside move "
                            f"tends to follow within {_best_lag} trading day(s). "
                            "Note: Granger causality does not imply structural causality — "
                            "both may respond to a common unobserved shock (e.g. a supply disruption "
                            "hits news feeds before it is fully priced in)."
                        )
                        _status = "warn"
                    elif _min_p < 0.10:
                        _prose = (
                            f"Borderline Granger causality at lag {_best_lag}d (p={_min_p:.3f}). "
                            "Sentiment shows weak predictive value for TTF returns — directionally "
                            "interesting but not robust enough to trade on without additional confirmation. "
                            f"Sample size: {_n_merged} days."
                        )
                        _status = "ok"
                    else:
                        _prose = (
                            f"No statistically significant Granger causality detected at any lag up to {_maxlag}d "
                            f"(best p={_min_p:.3f}). "
                            "Sentiment does not add predictive power for TTF daily returns beyond TTF's own history "
                            "in the current sample. This can reflect: insufficient history, high noise in the sentiment "
                            "series, or a regime where macro/supply factors dominate over news flow."
                        )
                        _status = "ok"

                    st.markdown(commentary(_prose, _status), unsafe_allow_html=True)

                    # ── Methodology ───────────────────────────────────────────
                    with st.expander("Methodology", expanded=False):
                        st.markdown(f"""
                        **Granger causality test (Granger 1969):** A variable X is said to Granger-cause Y
                        if lagged values of X add statistically significant predictive power for Y in an
                        OLS regression that already includes lagged values of Y.

                        **Test setup:**
                        - Y: TTF daily return (%) = `(price_t − price_{{t−1}}) / price_{{t−1}} × 100`
                        - X: Net sentiment score = (positive − negative) FinBERT classifications per day,
                          aggregated across all scored energy headlines
                        - Test: SSR F-test at each lag 1…{_maxlag}
                        - Null hypothesis H₀: sentiment at lag k does not Granger-cause TTF return

                        **Significance levels:**
                        - p < 0.05: reject H₀ — sentiment is a statistically significant predictor
                        - p < 0.10: borderline — directionally suggestive, not robust
                        - p ≥ 0.10: fail to reject H₀ — no predictive evidence in current sample

                        **Sentiment source:** ProsusAI/FinBERT applied to filtered energy RSS headlines
                        (BBC Business, Guardian Energy, LNG World News, Energy Monitor).
                        Daily net sentiment = Σ(positive − negative) scores across all day's headlines.

                        **Limitations:**
                        - Small sample: RSS feeds carry 2–7 days of articles per fetch; history grows over time.
                        - FinBERT was trained on financial news, not specifically energy commodities.
                        - Granger causality ≠ structural causality. Both series may respond to common
                          unobserved shocks (e.g. geopolitical event).
                        - The test uses daily returns; intraday or hourly sentiment would be stronger.

                        **Sample:** {_n_merged} aligned trading days, max lag {_maxlag}d.
                        """)

                    st.caption("Sources: ProsusAI/FinBERT | BBC/Guardian/LNG World News RSS | ICE/Yahoo Finance (TTF=F)")

                except ImportError:
                    st.warning(
                        "statsmodels is required for the Granger causality test. "
                        "Run `pip install statsmodels` to enable this tab.",
                        icon="⚠️",
                    )
                except Exception as _e:
                    st.error(f"Granger test failed: {_e}", icon="🚨")


# ════════════════════════════════════════════════════════════════════════════
# TAB 9: STORAGE–PRICE OLS REGRESSION
# ════════════════════════════════════════════════════════════════════════════
with tab_storage_reg:
    st.markdown("### EU Storage → TTF Price Regression")
    st.caption(
        "OLS regression of TTF front-month price on EU aggregate storage fill level. "
        "The residual — actual TTF minus the storage-implied fair value — is the "
        "supply-risk premium: a positive residual means the market prices in additional "
        "fear (geopolitical, supply disruption) beyond what storage fundamentals explain."
    )

    _eu_reg = storage.get("europe", pd.DataFrame())
    _ttf_reg = ttf.get("prices", pd.DataFrame())

    if _eu_reg.empty:
        st.warning(
            "EU gas storage data unavailable. Add AGSI_API_KEY to .env to enable this model.",
            icon="⚠️",
        )
    elif _ttf_reg.empty:
        st.warning("TTF price data unavailable.", icon="⚠️")
    else:
        # ── Build merged series ───────────────────────────────────────────────
        _str_df = _eu_reg[["gasDayStart", "full"]].copy().dropna()
        _str_df = _str_df.rename(columns={"gasDayStart": "date", "full": "storage_pct"})
        _str_df["date"] = pd.to_datetime(_str_df["date"])

        _ttf_r = _ttf_reg[["date", "price"]].copy()
        _ttf_r["date"] = pd.to_datetime(_ttf_r["date"])

        _sreg = _str_df.merge(_ttf_r, on="date", how="inner").dropna().sort_values("date")

        _MIN_OBS = COINT_MIN_MATCH
        if len(_sreg) < _MIN_OBS:
            st.info(
                f"Storage regression requires {_MIN_OBS} overlapping observations. "
                f"Currently have {len(_sreg)}. "
                "Ensure AGSI_API_KEY is set and TTF history is available.",
                icon="ℹ️",
            )
        else:
            try:
                import statsmodels.api as _sm

                _X = _sm.add_constant(_sreg["storage_pct"])
                _y = _sreg["price"]
                _ols_res = _sm.OLS(_y, _X).fit()

                _alpha   = float(_ols_res.params["const"])
                _beta    = float(_ols_res.params["storage_pct"])
                _r2      = float(_ols_res.rsquared)
                _p_beta  = float(_ols_res.pvalues["storage_pct"])

                _sreg = _sreg.copy()
                _sreg["fitted"]   = _ols_res.fittedvalues.values
                _sreg["residual"] = (_sreg["price"] - _sreg["fitted"])

                _latest_row     = _sreg.iloc[-1]
                _latest_storage = float(_latest_row["storage_pct"])
                _latest_ttf     = float(_latest_row["price"])
                _latest_fitted  = float(_latest_row["fitted"])
                _latest_resid   = float(_latest_row["residual"])

                # ── KPI row ───────────────────────────────────────────────────
                _k1, _k2, _k3, _k4, _k5 = st.columns(5)
                with _k1:
                    st.markdown(
                        kpi_card("EU storage", f"{_latest_storage:.1f}%",
                                 delta_span("latest fill level", "blue")),
                        unsafe_allow_html=True,
                    )
                with _k2:
                    st.markdown(
                        kpi_card("TTF actual", f"€{_latest_ttf:.1f}/MWh",
                                 delta_span("front-month", "blue")),
                        unsafe_allow_html=True,
                    )
                with _k3:
                    st.markdown(
                        kpi_card("Storage-implied", f"€{_latest_fitted:.1f}/MWh",
                                 delta_span(f"β={_beta:.2f} EUR/pp", "amber")),
                        unsafe_allow_html=True,
                    )
                with _k4:
                    _prem_color = "red" if _latest_resid > 5 else ("green" if _latest_resid < -5 else "blue")
                    _prem_label = "supply fear premium" if _latest_resid > 5 else (
                        "below fundamental value" if _latest_resid < -5 else "near fair value"
                    )
                    st.markdown(
                        kpi_card("Risk premium", f"{_latest_resid:+.1f} EUR/MWh",
                                 delta_span(_prem_label, _prem_color)),
                        unsafe_allow_html=True,
                    )
                with _k5:
                    st.markdown(
                        kpi_card("R² (full sample)", f"{_r2:.2f}",
                                 delta_span(f"p={_p_beta:.3f} for β", "blue")),
                        unsafe_allow_html=True,
                    )

                st.divider()

                # ── Scatter + OLS fit ─────────────────────────────────────────
                st.markdown("#### Storage Fill % vs TTF Price — Scatter & OLS Fit")
                st.caption(
                    "Each dot is one trading day. Colour indicates year. "
                    "Red dot = current observation. Regression line shows the storage-implied fair value."
                )

                _years = _sreg["date"].dt.year.unique()
                _palette = ["#58a6ff", "#3fb950", "#f0e040", "#e07b39", "#f85149",
                            "#a371f7", "#7ec8e3", "#ffa657"]
                _fig_scatter = go.Figure()
                for _i, _yr in enumerate(sorted(_years)):
                    _sub = _sreg[_sreg["date"].dt.year == _yr]
                    _fig_scatter.add_trace(go.Scatter(
                        x=_sub["storage_pct"], y=_sub["price"],
                        mode="markers",
                        name=str(_yr),
                        marker=dict(color=_palette[_i % len(_palette)], size=5, opacity=0.6),
                        hovertemplate=f"{_yr}: storage=%{{x:.1f}}%, TTF=€%{{y:.1f}}/MWh<extra></extra>",
                    ))

                # OLS regression line
                _x_line = np.linspace(float(_sreg["storage_pct"].min()),
                                      float(_sreg["storage_pct"].max()), 100)
                _y_line = _alpha + _beta * _x_line
                _fig_scatter.add_trace(go.Scatter(
                    x=_x_line, y=_y_line,
                    mode="lines",
                    name=f"OLS fit (R²={_r2:.2f})",
                    line=dict(color="rgba(255,255,255,0.6)", width=2, dash="dash"),
                    hoverinfo="skip",
                ))

                # Current obs highlight
                _fig_scatter.add_trace(go.Scatter(
                    x=[_latest_storage], y=[_latest_ttf],
                    mode="markers",
                    name="Current",
                    marker=dict(color="#f85149", size=12, symbol="star",
                                line=dict(color="white", width=1)),
                    hovertemplate=f"Now: storage={_latest_storage:.1f}%, TTF=€{_latest_ttf:.1f}/MWh<extra></extra>",
                ))

                _fig_scatter.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0d1117",
                    plot_bgcolor="#161b22",
                    xaxis=dict(title="EU storage fill level (%)",
                               gridcolor="rgba(255,255,255,0.06)"),
                    yaxis=dict(title="TTF front-month (EUR/MWh)",
                               gridcolor="rgba(255,255,255,0.06)"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                                font=dict(size=10)),
                    margin=dict(l=60, r=20, t=30, b=50),
                    height=420,
                )
                st.plotly_chart(_fig_scatter, use_container_width=True)

                # ── Residual time series ──────────────────────────────────────
                st.markdown("#### Supply-Risk Premium (OLS Residual) Over Time")
                st.caption(
                    "Residual = actual TTF − storage-implied TTF. "
                    "Positive: market prices in supply risk beyond storage fundamentals. "
                    "Negative: TTF trading below what storage fill alone would suggest."
                )

                _resid_color = [
                    "#f85149" if r > 0 else "#3fb950"
                    for r in _sreg["residual"]
                ]
                _fig_resid = go.Figure()
                _fig_resid.add_trace(go.Bar(
                    x=_sreg["date"], y=_sreg["residual"],
                    name="Risk premium",
                    marker_color=_resid_color,
                    hovertemplate="Date: %{x|%Y-%m-%d}<br>Premium: %{y:+.1f} EUR/MWh<extra></extra>",
                ))
                _fig_resid.add_hline(y=0, line=dict(color="rgba(255,255,255,0.3)", width=1))
                _fig_resid.add_hline(
                    y=10, line=dict(color="rgba(248,81,73,0.4)", width=1, dash="dot"),
                    annotation_text="+10 EUR/MWh",
                    annotation_font=dict(color="rgba(248,81,73,0.6)", size=9),
                    annotation_position="top right",
                )
                _fig_resid.add_hline(
                    y=-10, line=dict(color="rgba(63,185,80,0.4)", width=1, dash="dot"),
                    annotation_text="−10 EUR/MWh",
                    annotation_font=dict(color="rgba(63,185,80,0.6)", size=9),
                    annotation_position="bottom right",
                )
                _fig_resid.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0d1117",
                    plot_bgcolor="#161b22",
                    xaxis=dict(title=None, gridcolor="rgba(255,255,255,0.06)"),
                    yaxis=dict(title="Residual (EUR/MWh)",
                               gridcolor="rgba(255,255,255,0.06)"),
                    margin=dict(l=60, r=20, t=20, b=50),
                    height=320,
                    showlegend=False,
                )
                st.plotly_chart(_fig_resid, use_container_width=True)

                # ── Commentary ────────────────────────────────────────────────
                _eq_str = f"TTF = {_alpha:.1f} + {_beta:.2f} × storage_pct"
                if _latest_resid > 5:
                    _prose = (
                        f"At {_latest_storage:.1f}% EU storage fill, the OLS model implies a fair-value "
                        f"TTF of €{_latest_fitted:.1f}/MWh ({_eq_str}). "
                        f"Actual TTF (€{_latest_ttf:.1f}/MWh) is €{_latest_resid:.1f}/MWh above this — "
                        "a supply-fear premium. The market is pricing in risk (geopolitical, supply disruption, "
                        "demand shock) that the storage fill level alone does not justify."
                    )
                    _prem_status = "warn"
                elif _latest_resid < -5:
                    _prose = (
                        f"At {_latest_storage:.1f}% EU storage fill, the OLS model implies a fair-value "
                        f"TTF of €{_latest_fitted:.1f}/MWh ({_eq_str}). "
                        f"Actual TTF (€{_latest_ttf:.1f}/MWh) is €{abs(_latest_resid):.1f}/MWh below this — "
                        "TTF is trading at a discount to storage-implied fundamental value. "
                        "Possible explanations: strong LNG imports, subdued industrial demand, or a risk-off "
                        "market environment compressing the forward risk premium."
                    )
                    _prem_status = "ok"
                else:
                    _prose = (
                        f"At {_latest_storage:.1f}% EU storage fill, the OLS model implies a fair-value "
                        f"TTF of €{_latest_fitted:.1f}/MWh ({_eq_str}). "
                        f"Actual TTF (€{_latest_ttf:.1f}/MWh) is {_latest_resid:+.1f} EUR/MWh — "
                        "near the storage-implied fundamental level. "
                        "The market is pricing storage fundamentals without a significant risk premium or discount."
                    )
                    _prem_status = "ok"

                st.markdown(commentary(_prose, _prem_status), unsafe_allow_html=True)

                # ── Methodology ───────────────────────────────────────────────
                with st.expander("Methodology", expanded=False):
                    st.markdown(f"""
                    **Model:** Simple OLS regression over the full available history:

                    `TTF (EUR/MWh) = α + β × EU_storage_fill (%) + ε`

                    **Estimated coefficients (full sample, N={len(_sreg)}):**
                    - Intercept α = {_alpha:.1f} EUR/MWh
                    - Storage slope β = {_beta:.2f} EUR/MWh per percentage point
                    - R² = {_r2:.3f} &nbsp; | &nbsp; p-value for β = {_p_beta:.4f}

                    **Interpretation of β:** A one percentage-point higher storage fill is associated
                    with a {abs(_beta):.2f} EUR/MWh {"lower" if _beta < 0 else "higher"} TTF price on average.
                    The negative sign (if β < 0) reflects the fundamental inverse relationship: high storage
                    reduces scarcity risk and therefore gas prices.

                    **Supply-risk premium (residual):** The OLS residual captures the component of TTF that
                    is not explained by storage fill alone. Persistent positive residuals indicate that the
                    market is pricing in supply risk beyond what the current fill level justifies — a fear
                    premium driven by geopolitical events, pipeline disruptions, or LNG market tightness.

                    **Limitations:**
                    - OLS assumes a linear, time-stationary relationship between storage and price.
                      In practice, the relationship is non-linear (scarcity accelerates at low fill levels)
                      and regime-dependent (pre-2022 vs post-2022 markets behave differently).
                    - The model does not control for season, demand, LNG supply, or weather.
                    - Residual should be interpreted as a signal for further investigation, not a trading signal.

                    **Data sources:** GIE AGSI+ (EU aggregate storage) | ICE/Yahoo Finance (TTF=F)
                    """)

                st.caption("Sources: GIE AGSI+ (EU storage) | ICE/Yahoo Finance (TTF=F)")

            except ImportError:
                st.warning(
                    "statsmodels is required for OLS regression. Run `pip install statsmodels`.",
                    icon="⚠️",
                )
            except Exception as _e:
                st.error(f"Storage regression failed: {_e}", icon="🚨")


# ════════════════════════════════════════════════════════════════════════════
# TAB 10: HYDRO RESERVOIR LEAD/LAG ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
with tab_hydro_lag:
    st.markdown("### Hydro Reservoir Lead/Lag Analysis")
    st.caption(
        "Cross-correlation of Norwegian hydro reservoir fill level (% of historical max) "
        "with NO2 day-ahead price at lags 0–21 days. A negative correlation at lag k means "
        "high reservoir fill k days ago is associated with lower prices today — quantifying "
        "how far ahead hydro fundamentals lead the spot market."
    )

    # Lazy-load feature matrix
    if "features_l2" not in st.session_state:
        with st.spinner("Assembling feature matrix (first visit — subsequent loads are instant)…"):
            st.session_state["features_l2"] = _get_features_l2()
    _hydro_feat_df = st.session_state["features_l2"]

    if _hydro_feat_df.empty:
        st.warning("Feature matrix unavailable. Check ENTSO-E API key.", icon="⚠️")
    elif "hydro_pct" not in _hydro_feat_df.columns or _hydro_feat_df["hydro_pct"].notna().sum() < 60:
        st.info(
            "Norwegian hydro reservoir data is not available in the current feature set. "
            "This requires an active ENTSO-E API key (ENTSOE_API_KEY in .env). "
            "Without hydro data, the lead/lag analysis cannot run.",
            icon="ℹ️",
        )
    else:
        _MAX_LAG = HYDRO_MAX_LAG

        _hdf = _hydro_feat_df[["date", "no2", "hydro_pct"]].dropna().copy()
        _hdf["date"] = pd.to_datetime(_hdf["date"])
        _hdf = _hdf.sort_values("date").reset_index(drop=True)

        if len(_hdf) < _MAX_LAG + 30:
            st.warning(
                f"Insufficient overlapping hydro + NO2 data ({len(_hdf)} rows). "
                f"Need at least {_MAX_LAG + 30}.",
                icon="⚠️",
            )
        else:
            # ── Compute cross-correlations ────────────────────────────────────
            _lags  = list(range(0, _MAX_LAG + 1))
            _corrs = []
            for _lag in _lags:
                _shifted = _hdf["hydro_pct"].shift(_lag)
                _r = _shifted.corr(_hdf["no2"])
                _corrs.append(float(_r) if not pd.isna(_r) else 0.0)

            _abs_corrs  = [abs(c) for c in _corrs]
            _peak_lag   = int(_lags[int(np.argmax(_abs_corrs))])
            _peak_corr  = _corrs[_peak_lag]
            _lag0_corr  = _corrs[0]

            # ── KPI row ───────────────────────────────────────────────────────
            _h1, _h2, _h3, _h4 = st.columns(4)
            with _h1:
                st.markdown(
                    kpi_card("Peak lag", f"{_peak_lag} days",
                             delta_span("strongest hydro → price signal", "blue")),
                    unsafe_allow_html=True,
                )
            with _h2:
                _corr_color = "red" if abs(_peak_corr) > 0.4 else ("amber" if abs(_peak_corr) > 0.2 else "blue")
                st.markdown(
                    kpi_card("Peak correlation", f"{_peak_corr:.3f}",
                             delta_span("ρ(hydro[t−k], NO2[t])", _corr_color)),
                    unsafe_allow_html=True,
                )
            with _h3:
                st.markdown(
                    kpi_card("Lag-0 correlation", f"{_lag0_corr:.3f}",
                             delta_span("contemporaneous", "blue")),
                    unsafe_allow_html=True,
                )
            with _h4:
                st.markdown(
                    kpi_card("Sample", f"{len(_hdf)} days",
                             delta_span(f"max lag: {_MAX_LAG}d", "blue")),
                    unsafe_allow_html=True,
                )

            st.divider()

            # ── Cross-correlation bar chart ────────────────────────────────────
            st.markdown("#### Cross-Correlation ρ(hydro_pct[t−k], NO2[t]) by Lag")
            st.caption(
                "Negative correlation (blue) means higher hydro fill at lag k is associated with lower "
                "NO2 price today — the expected direction. The lag with the largest |ρ| is the point "
                "where hydro has the most predictive power for NO2."
            )

            _bar_cols = [
                "#f85149" if c > 0 else "#58a6ff"
                for c in _corrs
            ]
            _fig_cc = go.Figure()
            _fig_cc.add_trace(go.Bar(
                x=_lags,
                y=_corrs,
                marker_color=_bar_cols,
                text=[f"{c:.2f}" for c in _corrs],
                textposition="outside",
                hovertemplate="Lag %{x}d: ρ=%{y:.3f}<extra></extra>",
            ))
            # Highlight peak lag
            _fig_cc.add_vline(
                x=_peak_lag,
                line=dict(color="rgba(240,224,64,0.6)", width=2, dash="dash"),
                annotation_text=f"  Peak lag: {_peak_lag}d",
                annotation_font=dict(color="#f0e040", size=10),
                annotation_position="top right",
            )
            _fig_cc.add_hline(y=0, line=dict(color="rgba(255,255,255,0.2)", width=1))
            _fig_cc.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0d1117",
                plot_bgcolor="#161b22",
                xaxis=dict(title="Lag (days)", dtick=1,
                           gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(title="Pearson correlation ρ",
                           gridcolor="rgba(255,255,255,0.06)",
                           range=[min(_corrs) * 1.2 - 0.05, max(_corrs) * 1.2 + 0.05]),
                margin=dict(l=60, r=20, t=30, b=50),
                height=380,
                showlegend=False,
            )
            st.plotly_chart(_fig_cc, use_container_width=True)

            # ── Rolling correlation time series ───────────────────────────────
            _roll_win = 90
            st.markdown(f"#### Rolling {_roll_win}-Day Correlation at Peak Lag ({_peak_lag}d)")
            st.caption(
                f"How has the hydro→NO2 lead relationship at lag {_peak_lag}d evolved over time? "
                "Periods of near-zero rolling correlation indicate other drivers dominated."
            )

            _hdf_lag = _hdf.copy()
            _hdf_lag["hydro_lagged"] = _hdf_lag["hydro_pct"].shift(_peak_lag)
            _hdf_lag = _hdf_lag.dropna(subset=["hydro_lagged", "no2"])
            _hdf_lag["rolling_corr"] = (
                _hdf_lag["hydro_lagged"]
                .rolling(_roll_win)
                .corr(_hdf_lag["no2"])
            )

            _fig_roll = go.Figure()
            _fig_roll.add_trace(go.Scatter(
                x=_hdf_lag["date"],
                y=_hdf_lag["rolling_corr"],
                mode="lines",
                name=f"Rolling {_roll_win}d ρ",
                line=dict(color="#58a6ff", width=1.5),
                hovertemplate="%{x|%Y-%m-%d}: ρ=%{y:.3f}<extra></extra>",
            ))
            _fig_roll.add_hline(y=0, line=dict(color="rgba(255,255,255,0.2)", width=1))
            _fig_roll.add_hline(
                y=-0.4, line=dict(color="rgba(248,81,73,0.3)", width=1, dash="dot"),
                annotation_text="ρ=−0.4",
                annotation_font=dict(color="rgba(248,81,73,0.5)", size=9),
                annotation_position="bottom right",
            )
            _fig_roll.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0d1117",
                plot_bgcolor="#161b22",
                xaxis=dict(title=None, gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(title="Rolling Pearson ρ",
                           gridcolor="rgba(255,255,255,0.06)"),
                margin=dict(l=60, r=20, t=20, b=50),
                height=300,
                showlegend=False,
            )
            st.plotly_chart(_fig_roll, use_container_width=True)

            # ── Commentary ────────────────────────────────────────────────────
            if abs(_peak_corr) > 0.4:
                _h_prose = (
                    f"Strong hydro→NO2 lead relationship: ρ = {_peak_corr:.3f} at lag {_peak_lag}d. "
                    f"{'Higher' if _peak_corr < 0 else 'Lower'} Norwegian hydro reservoir fill {_peak_lag} days ago "
                    f"is associated with {'lower' if _peak_corr < 0 else 'higher'} NO2 day-ahead prices today. "
                    "This confirms Norwegian hydro as a leading fundamental driver of Nordic spot prices. "
                    "Hydro reservoir changes — from precipitation, snowmelt, or drawdown — propagate into "
                    f"spot market pricing with a delay of roughly {_peak_lag} trading day(s)."
                )
                _h_status = "warn" if _peak_corr > 0 else "ok"
            elif abs(_peak_corr) > 0.2:
                _h_prose = (
                    f"Moderate hydro→NO2 correlation at lag {_peak_lag}d (ρ = {_peak_corr:.3f}). "
                    "The hydro signal is present but not dominant — other factors (continental prices, TTF, "
                    "grid constraints) are likely exerting stronger influence on NO2 in the current sample."
                )
                _h_status = "ok"
            else:
                _h_prose = (
                    f"Weak hydro→NO2 correlation across all lags 0–{_MAX_LAG}d (peak ρ = {_peak_corr:.3f}). "
                    "In the current sample, hydro reservoir level does not show a meaningful predictive "
                    "relationship with NO2 prices. Continental price spillover (NL via NordLink) or gas price "
                    "pass-through may be dominating the price formation mechanism."
                )
                _h_status = "ok"

            st.markdown(commentary(_h_prose, _h_status), unsafe_allow_html=True)

            # ── Methodology ───────────────────────────────────────────────────
            with st.expander("Methodology", expanded=False):
                st.markdown(f"""
                **Cross-correlation:** For each lag k ∈ [0, {_MAX_LAG}], compute the Pearson correlation
                coefficient between `hydro_pct[t−k]` and `NO2[t]`:

                `ρ(k) = corr(hydro_pct[t−k], NO2[t])`

                **Expected sign:** Negative. Higher hydro reservoir fill → more hydro generation capacity
                available → lower marginal cost of supply → lower spot prices. Negative ρ confirms the
                fundamental hydro supply relationship.

                **Lag interpretation:** The peak lag k* is the number of days by which changes in hydro
                reservoir fill level lead changes in NO2 spot prices. This reflects how quickly new
                hydro fundamentals information propagates into market prices.

                **Rolling correlation:** {_roll_win}-day trailing window applied to the lagged pair.
                Periods of low rolling correlation indicate that other drivers temporarily dominated
                price formation (e.g. cold snap, gas price shock, continental congestion).

                **Data:** Norwegian hydro reservoir filling level (ENTSO-E B31, TWh), normalised to
                % of expanding 98th percentile to create a stationary fill-rate signal.

                **Limitations:**
                - Weekly ENTSO-E hydro data is forward-filled to daily — this reduces the effective
                  degrees of freedom and may understate lags shorter than 7 days.
                - Correlation does not imply causality: hydro and prices may both respond to common
                  factors (temperature, precipitation forecasts).
                - The relationship may be non-linear at extreme fill levels (very low = scarcity,
                  very high = spill risk / negative prices).

                **Sample:** {len(_hdf)} overlapping days of hydro_pct and NO2.
                """)

            st.caption("Sources: ENTSO-E B31 (hydro reservoirs) | ENTSO-E A44 (NO2 day-ahead prices)")


# ════════════════════════════════════════════════════════════════════════════
# TAB 11: TTF SEASONAL NORM TRACKER
# ════════════════════════════════════════════════════════════════════════════
with tab_ttf_norm:
    st.markdown("### TTF Seasonal Norm Tracker")
    st.caption(
        "Current TTF prices overlaid on the historical seasonal distribution (prior years). "
        "Shaded bands show the 10th–90th and 25th–75th percentile range by calendar day. "
        "Prices trading above the 90th percentile signal historically extreme levels; "
        "below the 10th percentile signal historically cheap gas."
    )

    with st.spinner("Loading TTF price history…"):
        _ttf_norm_df = _get_ttf_history_norm()

    if _ttf_norm_df.empty:
        st.warning(
            "TTF price history unavailable. yfinance must be installed and TTF=F must be accessible.",
            icon="⚠️",
        )
    else:
        _ttf_norm_df = _ttf_norm_df.copy()
        _ttf_norm_df["date"] = pd.to_datetime(_ttf_norm_df["date"])
        _ttf_norm_df["year"]  = _ttf_norm_df["date"].dt.year
        _ttf_norm_df["month"] = _ttf_norm_df["date"].dt.month
        _ttf_norm_df["day"]   = _ttf_norm_df["date"].dt.day
        _ttf_norm_df["md"]    = _ttf_norm_df["month"] * 100 + _ttf_norm_df["day"]  # MMDD key

        _current_year = _ttf_norm_df["date"].dt.year.max()
        _hist_df  = _ttf_norm_df[_ttf_norm_df["year"] < _current_year]
        _curr_df  = _ttf_norm_df[_ttf_norm_df["year"] == _current_year].copy()

        if _hist_df.empty or len(_hist_df["year"].unique()) < 2:
            st.info("Need at least 2 prior years of history to compute seasonal bands.", icon="ℹ️")
        else:
            # ── Compute seasonal bands by calendar day ────────────────────────
            _bands = (
                _hist_df.groupby("md")["price"]
                .agg(
                    p10=lambda x: x.quantile(0.10),
                    p25=lambda x: x.quantile(0.25),
                    p50=lambda x: x.quantile(0.50),
                    p75=lambda x: x.quantile(0.75),
                    p90=lambda x: x.quantile(0.90),
                )
                .reset_index()
            )

            # Build a reference date axis using current year (or 2024 as leap-year base)
            _ref_year = 2024
            def _md_to_date(md: int) -> pd.Timestamp:
                m, d = divmod(md, 100)
                try:
                    return pd.Timestamp(year=_ref_year, month=m, day=d)
                except ValueError:
                    return pd.NaT

            _bands["ref_date"] = _bands["md"].apply(_md_to_date)
            _bands = _bands.dropna(subset=["ref_date"]).sort_values("ref_date")

            # Map current year prices to ref_date axis for overlay
            _curr_df = _curr_df.copy()
            _curr_df["ref_date"] = _curr_df.apply(
                lambda r: _md_to_date(int(r["md"])), axis=1
            )
            _curr_df = _curr_df.dropna(subset=["ref_date"]).sort_values("ref_date")

            # ── Current position in seasonal distribution ─────────────────────
            _latest_ttf_norm = float(_ttf_norm_df.sort_values("date")["price"].iloc[-1])
            _latest_md       = int(_ttf_norm_df.sort_values("date")["md"].iloc[-1])
            _band_row        = _bands[_bands["md"] == _latest_md]

            _pct_rank: float | None = None
            if not _band_row.empty:
                _p10 = float(_band_row["p10"].iloc[0])
                _p90 = float(_band_row["p90"].iloc[0])
                _p50 = float(_band_row["p50"].iloc[0])
                # Percentile rank in historical distribution for this calendar day
                _day_hist = _hist_df[_hist_df["md"] == _latest_md]["price"]
                if len(_day_hist) >= 2:
                    _pct_rank = float((_day_hist < _latest_ttf_norm).mean() * 100)

            # ── KPI row ───────────────────────────────────────────────────────
            _n1, _n2, _n3, _n4 = st.columns(4)
            with _n1:
                st.markdown(
                    kpi_card("TTF current", f"€{_latest_ttf_norm:.1f}/MWh",
                             delta_span("latest front-month", "blue")),
                    unsafe_allow_html=True,
                )
            if not _band_row.empty:
                with _n2:
                    _vs_median = _latest_ttf_norm - _p50
                    _med_color = "red" if _vs_median > 5 else ("green" if _vs_median < -5 else "blue")
                    st.markdown(
                        kpi_card("vs seasonal median", f"{_vs_median:+.1f} EUR/MWh",
                                 delta_span(f"median: €{_p50:.1f}/MWh", _med_color)),
                        unsafe_allow_html=True,
                    )
                with _n3:
                    if _pct_rank is not None:
                        _rank_label = (
                            "above 90th pct" if _pct_rank > 90 else
                            "below 10th pct" if _pct_rank < 10 else
                            f"{_pct_rank:.0f}th percentile"
                        )
                        _rank_color = "red" if _pct_rank > 90 else ("green" if _pct_rank < 10 else "blue")
                        st.markdown(
                            kpi_card("Seasonal rank", f"{_pct_rank:.0f}th pct",
                                     delta_span(_rank_label, _rank_color)),
                            unsafe_allow_html=True,
                        )
                with _n4:
                    _hist_years = sorted(_hist_df["year"].unique())
                    st.markdown(
                        kpi_card("History", f"{len(_hist_years)} years",
                                 delta_span(f"{_hist_years[0]}–{_hist_years[-1]}", "blue")),
                        unsafe_allow_html=True,
                    )

            st.divider()

            # ── Seasonal norm chart ───────────────────────────────────────────
            st.markdown(f"#### TTF {_current_year} vs Seasonal Distribution ({_hist_years[0]}–{_hist_years[-1]})")
            st.caption(
                "Dark shaded band = 25th–75th percentile (typical range). "
                "Light shaded band = 10th–90th percentile (historical extremes). "
                f"Solid line = {_current_year} actual prices."
            )

            _fig_norm = go.Figure()

            # 10th–90th outer band
            _fig_norm.add_trace(go.Scatter(
                x=list(_bands["ref_date"]) + list(_bands["ref_date"])[::-1],
                y=list(_bands["p90"]) + list(_bands["p10"])[::-1],
                fill="toself",
                fillcolor="rgba(88,166,255,0.10)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name="10th–90th pct (historical)",
                showlegend=True,
            ))

            # 25th–75th inner band
            _fig_norm.add_trace(go.Scatter(
                x=list(_bands["ref_date"]) + list(_bands["ref_date"])[::-1],
                y=list(_bands["p75"]) + list(_bands["p25"])[::-1],
                fill="toself",
                fillcolor="rgba(88,166,255,0.22)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name="25th–75th pct (typical)",
                showlegend=True,
            ))

            # Median
            _fig_norm.add_trace(go.Scatter(
                x=_bands["ref_date"], y=_bands["p50"],
                mode="lines",
                name="Seasonal median",
                line=dict(color="rgba(88,166,255,0.5)", width=1.5, dash="dot"),
                hovertemplate="Median: €%{y:.1f}/MWh<extra></extra>",
            ))

            # Current year actual
            if not _curr_df.empty:
                _fig_norm.add_trace(go.Scatter(
                    x=_curr_df["ref_date"], y=_curr_df["price"],
                    mode="lines",
                    name=f"{_current_year} actual",
                    line=dict(color="#f0e040", width=2),
                    hovertemplate=f"{_current_year}: €%{{y:.1f}}/MWh<extra></extra>",
                ))

            _fig_norm.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0d1117",
                plot_bgcolor="#161b22",
                xaxis=dict(
                    title="Calendar month",
                    tickformat="%b",
                    dtick="M1",
                    gridcolor="rgba(255,255,255,0.06)",
                ),
                yaxis=dict(
                    title="TTF front-month (EUR/MWh)",
                    gridcolor="rgba(255,255,255,0.06)",
                    rangemode="tozero",
                ),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                            font=dict(size=10)),
                margin=dict(l=60, r=20, t=30, b=50),
                height=420,
            )
            st.plotly_chart(_fig_norm, use_container_width=True)

            # ── Commentary ────────────────────────────────────────────────────
            if _pct_rank is not None:
                if _pct_rank > 90:
                    _norm_prose = (
                        f"TTF at €{_latest_ttf_norm:.1f}/MWh is at the {_pct_rank:.0f}th percentile of "
                        f"the historical seasonal distribution for this time of year — "
                        f"€{_latest_ttf_norm - _p50:.1f}/MWh above the seasonal median. "
                        "This is historically elevated. Such levels typically reflect supply risk "
                        "(geopolitical disruption, LNG tightness) or cold-weather demand pressure "
                        "rather than pure storage fundamentals."
                    )
                    _norm_status = "warn"
                elif _pct_rank < 10:
                    _norm_prose = (
                        f"TTF at €{_latest_ttf_norm:.1f}/MWh is at the {_pct_rank:.0f}th percentile of "
                        f"the historical seasonal distribution — "
                        f"€{abs(_latest_ttf_norm - _p50):.1f}/MWh below the seasonal median. "
                        "Historically cheap. Possible explanations: strong LNG imports, full storage, "
                        "mild demand, or a post-shock normalization from prior elevated prices."
                    )
                    _norm_status = "ok"
                else:
                    _norm_prose = (
                        f"TTF at €{_latest_ttf_norm:.1f}/MWh sits at the {_pct_rank:.0f}th percentile "
                        f"of the historical seasonal range — {abs(_latest_ttf_norm - _p50):.1f} EUR/MWh "
                        f"{'above' if _latest_ttf_norm > _p50 else 'below'} the seasonal median. "
                        "Within normal seasonal bounds for this time of year."
                    )
                    _norm_status = "ok"
                st.markdown(commentary(_norm_prose, _norm_status), unsafe_allow_html=True)

            # ── Methodology ───────────────────────────────────────────────────
            with st.expander("Methodology", expanded=False):
                st.markdown(f"""
                **Seasonal distribution:** For each calendar day (month-day pair), the prior-year
                closing prices of TTF front-month futures (TTF=F via Yahoo Finance/ICE) are collected
                across all available history ({', '.join(str(y) for y in _hist_years)}).
                The 10th, 25th, 50th, 75th, and 90th percentiles define the seasonal bands.

                **Current year overlay:** {_current_year} prices are plotted on the same calendar axis
                (Jan 1 → Dec 31) to show where the current year stands relative to prior-year seasonal
                patterns.

                **Percentile rank:** The current TTF price is ranked against the same calendar-day
                distribution. A 90th-percentile reading means that for this day of the year, TTF has
                historically been cheaper in 90% of prior years.

                **Interpretation:**
                - Above 90th percentile: historically extreme. Indicates supply risk, cold-weather demand,
                  or market stress beyond seasonal norms.
                - 25th–75th band: typical seasonal range. No unusual signal.
                - Below 10th percentile: historically cheap. Watch for demand recovery or LNG supply tightening.

                **Data:** ICE TTF front-month (TTF=F), {len(_hist_years)} years of history,
                fetched via yfinance. Note: TTF=F is a continuous front-month contract; roll effects
                may introduce noise around contract expiry dates.
                """)

            st.caption("Source: ICE/Yahoo Finance (TTF=F)")


# ════════════════════════════════════════════════════════════════════════════
# TAB 12: NO2/NL COINTEGRATION & SPREAD MEAN-REVERSION
# ════════════════════════════════════════════════════════════════════════════
with tab_coint:
    st.markdown("### NO2/NL Cointegration & Spread Mean-Reversion")
    st.caption(
        "Engle-Granger cointegration test on the NO2–NL day-ahead price pair. "
        "If cointegrated, the spread has a long-run equilibrium and mean-reverts. "
        "The Ornstein-Uhlenbeck half-life quantifies how quickly deviations correct. "
        "Entry/exit z-score thresholds define a rules-based spread trading signal."
    )

    # Lazy-load feature matrix
    if "features_l2" not in st.session_state:
        with st.spinner("Assembling feature matrix (first visit — subsequent loads are instant)…"):
            st.session_state["features_l2"] = _get_features_l2()
    _coint_feat = st.session_state["features_l2"]

    if _coint_feat.empty:
        st.warning("Feature matrix unavailable. Check ENTSO-E API key.", icon="⚠️")
    elif "no2" not in _coint_feat.columns or "nl" not in _coint_feat.columns:
        st.warning("NO2 and NL price series required. Check ENTSO-E A44 API key.", icon="⚠️")
    else:
        _ci_df = _coint_feat[["date", "no2", "nl"]].dropna().copy()
        _ci_df["date"] = pd.to_datetime(_ci_df["date"])
        _ci_df = _ci_df.sort_values("date").reset_index(drop=True)
        _ci_df["spread"] = _ci_df["no2"] - _ci_df["nl"]

        _MIN_COINT = COINT_MIN_OBS
        if len(_ci_df) < _MIN_COINT:
            st.info(
                f"Cointegration test requires at least {_MIN_COINT} observations. "
                f"Currently have {len(_ci_df)}.",
                icon="ℹ️",
            )
        else:
            try:
                from statsmodels.tsa.stattools import coint as _eg_coint
                import statsmodels.api as _sm_ci

                # ── Engle-Granger cointegration test ─────────────────────────
                _ci_score, _ci_pval, _ci_crit = _eg_coint(_ci_df["no2"], _ci_df["nl"])

                # OLS to get residual (spread series for OU fit)
                _ci_X   = _sm_ci.add_constant(_ci_df["nl"])
                _ci_ols = _sm_ci.OLS(_ci_df["no2"], _ci_X).fit()
                _ci_beta_hedge = float(_ci_ols.params["nl"])
                _ci_resid = _ci_ols.resid.values.copy()

                # ── Ornstein-Uhlenbeck half-life via OLS on lagged residual ──
                _ci_resid_lag = _ci_resid[:-1]
                _ci_resid_d   = np.diff(_ci_resid)
                _ci_ou_X = _sm_ci.add_constant(_ci_resid_lag)
                _ci_ou   = _sm_ci.OLS(_ci_resid_d, _ci_ou_X).fit()
                _ci_lambda = float(_ci_ou.params[1])
                _ci_halflife = (-np.log(2) / _ci_lambda) if _ci_lambda < 0 else float("inf")

                # ── Spread z-score (expanding window — no look-ahead bias) ───
                _ci_resid_s  = pd.Series(_ci_resid, dtype=float)
                _ci_exp_mean = _ci_resid_s.expanding(min_periods=30).mean()
                _ci_exp_std  = _ci_resid_s.expanding(min_periods=30).std()
                _ci_z_series = ((_ci_resid_s - _ci_exp_mean) / (_ci_exp_std + 1e-9)).values
                _ci_z_valid  = _ci_z_series[~np.isnan(_ci_z_series)]
                _ci_z_now    = float(_ci_z_valid[-1]) if len(_ci_z_valid) else 0.0
                _ci_spread_now  = float(_ci_df["spread"].iloc[-1])

                # ── Backtest: reversion hit-rate at ±1σ entry ─────────────────
                _ENTRY_Z = COINT_ENTRY_Z
                _MAX_HOLD = min(60, max(5, int(_ci_halflife * 3))) if np.isfinite(_ci_halflife) else 21
                _bt_hits, _bt_total, _bt_mean_days = 0, 0, []
                for _idx in range(len(_ci_z_series) - _MAX_HOLD):
                    _z = _ci_z_series[_idx]
                    if not np.isnan(_z) and abs(_z) > _ENTRY_Z:
                        _bt_total += 1
                        _target_sign = -np.sign(_z)
                        _revert_days = None
                        for _d in range(1, _MAX_HOLD + 1):
                            _zf = _ci_z_series[_idx + _d]
                            if not np.isnan(_zf) and (np.sign(_zf) == _target_sign or abs(_zf) < 0.5):
                                _revert_days = _d
                                break
                        if _revert_days is not None:
                            _bt_hits += 1
                            _bt_mean_days.append(_revert_days)
                _bt_hit_rate = (_bt_hits / _bt_total * 100) if _bt_total > 0 else 0.0
                _bt_avg_days = float(np.mean(_bt_mean_days)) if _bt_mean_days else float("nan")
                _ci_hl_prose = f"{_ci_halflife:.1f}d" if np.isfinite(_ci_halflife) else "∞"
                _bt_avg_str  = f"{_bt_avg_days:.1f}d" if not np.isnan(_bt_avg_days) else "n/a"

                # ── KPI row ───────────────────────────────────────────────────
                _co1, _co2, _co3, _co4, _co5 = st.columns(5)
                with _co1:
                    _coint_verdict = "Cointegrated" if _ci_pval < 0.05 else (
                        "Borderline" if _ci_pval < 0.10 else "Not cointegrated"
                    )
                    _coint_color = "red" if _ci_pval < 0.05 else ("amber" if _ci_pval < 0.10 else "blue")
                    st.markdown(
                        kpi_card("Cointegration", _coint_verdict,
                                 delta_span(f"E-G p={_ci_pval:.3f}", _coint_color)),
                        unsafe_allow_html=True,
                    )
                with _co2:
                    _hl_str = f"{_ci_halflife:.1f}d" if np.isfinite(_ci_halflife) else "∞ (no reversion)"
                    st.markdown(
                        kpi_card("OU Half-life", _hl_str,
                                 delta_span("spread mean-reversion speed", "blue")),
                        unsafe_allow_html=True,
                    )
                with _co3:
                    _z_color = "red" if abs(_ci_z_now) > 2 else ("amber" if abs(_ci_z_now) > 1 else "blue")
                    st.markdown(
                        kpi_card("Spread z-score", f"{_ci_z_now:+.2f}σ",
                                 delta_span(f"raw: {_ci_spread_now:+.1f} EUR/MWh", _z_color)),
                        unsafe_allow_html=True,
                    )
                with _co4:
                    st.markdown(
                        kpi_card("Hedge ratio β", f"{_ci_beta_hedge:.3f}",
                                 delta_span("NO2 = α + β·NL + ε", "blue")),
                        unsafe_allow_html=True,
                    )
                with _co5:
                    st.markdown(
                        kpi_card("±1σ Reversion rate", f"{_bt_hit_rate:.0f}%",
                                 delta_span(f"avg {_bt_avg_days:.1f}d | {_bt_total} signals" if not np.isnan(_bt_avg_days) else f"{_bt_total} signals", "blue")),
                        unsafe_allow_html=True,
                    )

                st.divider()

                # ── Spread z-score time series ────────────────────────────────
                st.markdown("#### NO2–NL Spread Z-Score (OLS Residual, Full Sample)")
                st.caption(
                    "Z-score of the OLS residual from regressing NO2 on NL. "
                    "Values beyond ±1σ are potential entry points for mean-reversion trades. "
                    "Red dot = current position."
                )

                _ci_dates = _ci_df["date"].values
                _fig_z_ci = go.Figure()
                _fig_z_ci.add_trace(go.Scatter(
                    x=_ci_dates, y=_ci_z_series,
                    mode="lines",
                    name="Spread z-score",
                    line=dict(color="#58a6ff", width=1.2),
                    hovertemplate="%{x|%Y-%m-%d}: z=%{y:.2f}<extra></extra>",
                ))
                # Entry/exit bands
                for _lv, _col, _lbl in [
                    (2.0,  "rgba(248,81,73,0.5)",   "±2σ (aggressive entry)"),
                    (1.0,  "rgba(224,123,57,0.5)",   "±1σ (base entry)"),
                    (-1.0, "rgba(224,123,57,0.5)",   ""),
                    (-2.0, "rgba(248,81,73,0.5)",    ""),
                    (0.0,  "rgba(255,255,255,0.2)",  "Mean"),
                ]:
                    _fig_z_ci.add_hline(
                        y=_lv, line=dict(color=_col, width=1, dash="dot"),
                        annotation_text=_lbl if _lbl else "",
                        annotation_font=dict(color=_col, size=9),
                        annotation_position="top right" if _lv > 0 else "bottom right",
                    )
                # Current position
                _fig_z_ci.add_trace(go.Scatter(
                    x=[_ci_dates[-1]], y=[_ci_z_now],
                    mode="markers",
                    name="Current",
                    marker=dict(color="#f85149", size=10, symbol="circle",
                                line=dict(color="white", width=1)),
                    hovertemplate=f"Now: z={_ci_z_now:+.2f}<extra></extra>",
                ))
                _fig_z_ci.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                    xaxis=dict(title=None, gridcolor="rgba(255,255,255,0.06)"),
                    yaxis=dict(title="Z-score (σ)", gridcolor="rgba(255,255,255,0.06)"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                    margin=dict(l=60, r=20, t=30, b=50),
                    height=360,
                    hovermode="x unified",
                )
                st.plotly_chart(_fig_z_ci, use_container_width=True)

                # ── Raw spread + NO2 / NL price overlay ──────────────────────
                st.markdown("#### NO2 vs NL Day-Ahead Prices")
                st.caption(
                    "Visual cointegration check: both series should share a common long-run trend "
                    "if the pair is cointegrated."
                )
                _fig_px = go.Figure()
                _fig_px.add_trace(go.Scatter(
                    x=_ci_df["date"], y=_ci_df["no2"], name="NO2",
                    line=dict(color="#58a6ff", width=1.5),
                    hovertemplate="NO2: €%{y:.1f}/MWh<extra></extra>",
                ))
                _fig_px.add_trace(go.Scatter(
                    x=_ci_df["date"], y=_ci_df["nl"], name="NL",
                    line=dict(color="#e07b39", width=1.5),
                    hovertemplate="NL: €%{y:.1f}/MWh<extra></extra>",
                ))
                _fig_px.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                    xaxis=dict(title=None, gridcolor="rgba(255,255,255,0.06)"),
                    yaxis=dict(title="EUR/MWh", gridcolor="rgba(255,255,255,0.06)"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                    margin=dict(l=60, r=20, t=30, b=50),
                    height=300,
                    hovermode="x unified",
                )
                st.plotly_chart(_fig_px, use_container_width=True)

                # ── Commentary ────────────────────────────────────────────────
                if _ci_pval < 0.05:
                    if abs(_ci_z_now) > 1.0:
                        _side = "long NO2 / short NL" if _ci_z_now < -1.0 else "short NO2 / long NL"
                        _ci_prose = (
                            f"NO2 and NL are statistically cointegrated (E-G p={_ci_pval:.3f}). "
                            f"The spread is mean-reverting with an OU half-life of {_ci_hl_prose}. "
                            f"Current z-score = {_ci_z_now:+.2f}σ — beyond the ±1σ entry threshold. "
                            f"Signal: {_side}. "
                            f"Historical ±1σ reversion rate: {_bt_hit_rate:.0f}% within {int(_MAX_HOLD)} days "
                            f"(avg {_bt_avg_str} to revert, {_bt_total} signals in sample)."
                        )
                        _ci_status = "warn"
                    else:
                        _ci_prose = (
                            f"NO2 and NL are statistically cointegrated (E-G p={_ci_pval:.3f}). "
                            f"OU half-life: {_ci_hl_prose}. "
                            f"Current z-score = {_ci_z_now:+.2f}σ — within the neutral zone (|z| < 1σ). "
                            "No active spread trade signal."
                        )
                        _ci_status = "ok"
                else:
                    _ci_prose = (
                        f"Engle-Granger test does not reject the null of no cointegration "
                        f"(p={_ci_pval:.3f} > 0.05). "
                        "In the current sample, NO2 and NL do not share a statistically confirmed "
                        "long-run equilibrium. Spread trades carry higher risk of trending rather than reverting. "
                        f"OU half-life estimate ({_ci_hl_prose}) should be treated with caution when "
                        "cointegration is not confirmed."
                    )
                    _ci_status = "ok"

                st.markdown(commentary(_ci_prose, _ci_status), unsafe_allow_html=True)

                # ── Methodology ───────────────────────────────────────────────
                with st.expander("Methodology", expanded=False):
                    st.markdown(f"""
                    **Engle-Granger cointegration test (Engle & Granger 1987):**
                    Two non-stationary series X and Y are cointegrated if there exists a linear combination
                    `ε = Y − α − β·X` that is stationary (I(0)). This means deviations from the long-run
                    equilibrium are transient — the spread mean-reverts.

                    **Step 1 — OLS regression:** `NO2 = α + β·NL + ε`
                    - Estimated β (hedge ratio) = {_ci_beta_hedge:.3f}
                    - This means 1 MWh NO2 is hedged by {_ci_beta_hedge:.3f} MWh NL exposure.

                    **Step 2 — Residual stationarity (ADF test on ε):**
                    - Test statistic = {_ci_score:.3f}
                    - p-value = {_ci_pval:.4f}
                    - Critical values: 1% = {_ci_crit[0]:.3f}, 5% = {_ci_crit[1]:.3f}, 10% = {_ci_crit[2]:.3f}
                    - Conclusion: {"residual is stationary → cointegrated" if _ci_pval < 0.05 else "residual is non-stationary → cointegration not confirmed"}

                    **Ornstein-Uhlenbeck half-life:**
                    Estimated by regressing `Δε_t = λ·ε_{{t−1}} + const + u_t`.
                    Half-life = −ln(2) / λ = **{_ci_hl_prose}**.
                    Interpretation: the spread takes ~{_ci_hl_prose} to halve a deviation on average.
                    Typical Nordic/Continental half-lives: 3–15 days (faster when interconnectors are uncongested).

                    **Z-score:** `z_t = (ε_t − μ_ε(t)) / σ_ε(t)` where μ_ε(t) and σ_ε(t) are
                    expanding-window moments (min 30 obs, no look-ahead). Entry threshold: |z| > 1σ.
                    Exit: |z| < 0.5σ (or sign flip).

                    **Backtest (simple):** For each historical ±1σ entry, checks whether the z-score
                    reverted to <0.5σ (or opposite sign) within {_MAX_HOLD} trading days.
                    Hit rate = {_bt_hit_rate:.0f}% ({_bt_hits}/{_bt_total} signals).

                    **Important limitations:**
                    - Cointegration is a full-sample result — regime breaks (e.g. the 2021/22 energy crisis)
                      can cause temporary breakdown of the relationship.
                    - Hedge ratio β is estimated unconditionally; rolling β from the Nordic Decomp tab
                      provides a more responsive estimate.
                    - Transaction costs, bid-ask spreads, and settlement mechanics are not modelled.
                    - The spread is wholesale day-ahead; intraday or balancing spreads differ.

                    **Data:** ENTSO-E A44 (NO2 and NL day-ahead prices), {len(_ci_df)} trading days.
                    """)

                st.caption("Sources: ENTSO-E A44 (NO2, NL day-ahead prices)")

            except ImportError:
                st.warning("statsmodels is required. Run `pip install statsmodels`.", icon="⚠️")
            except Exception as _e:
                st.error(f"Cointegration analysis failed: {_e}", icon="🚨")


# ════════════════════════════════════════════════════════════════════════════
# TAB 13: MULTI-PAIR COINTEGRATION SCANNER
# ════════════════════════════════════════════════════════════════════════════
with tab_scanner:
    st.markdown("### Multi-Pair Cointegration Scanner")
    st.caption(
        "Engle-Granger cointegration tests across the full European gas-hub and power-zone universe. "
        "For each cointegrated pair the Ornstein-Uhlenbeck half-life and current z-score determine "
        "whether a spread trade is actionable today. "
        "Universe: NO2, NL, TTF (base) + DE, FR (ENTSO-E A44) + NBP (yfinance, when available)."
    )

    _SCANNER_YEARS = 3

    if "scanner_prices" not in st.session_state:
        with st.spinner("Fetching extended universe prices (DE, FR via ENTSO-E A44)…"):
            st.session_state["scanner_prices"] = _get_scanner_prices(years=_SCANNER_YEARS)
    _scan_px = st.session_state["scanner_prices"]

    _scan_cols = [c for c in _scan_px.columns if c != "date"] if not _scan_px.empty else []

    if _scan_px.empty or len(_scan_cols) < 2:
        st.warning(
            "Scanner requires at least 2 price series. Check ENTSO-E API key for DE and FR data.",
            icon="⚠️",
        )
    else:
        _scan_universe = " · ".join(c.upper() for c in _scan_cols)
        st.caption(f"Universe loaded: **{_scan_universe}** — {len(_scan_px):,} daily obs (longest common window)")
        _expected_extra = {"de", "fr", "nbp"}
        _missing_extra  = _expected_extra - set(_scan_cols)
        if _missing_extra:
            st.caption(
                "Not loaded: " + ", ".join(m.upper() for m in sorted(_missing_extra))
                + " — ENTSO-E A44 key required for DE/FR; NBP requires yfinance (ICE/Yahoo Finance)."
            )

        _scan_err_msg = None
        try:
            with st.spinner("Running Engle-Granger tests across all pairs…"):
                _scan_res = _run_coint_scanner(_scan_px)
        except Exception as _scan_ex:
            _scan_err_msg = f"{type(_scan_ex).__name__}: {_scan_ex}"
            _scan_res = []

        if _scan_err_msg:
            st.error(f"Scanner failed — {_scan_err_msg}", icon="🚨")
        elif not _scan_res:
            st.info(
                "No pairs tested — no pair had sufficient overlapping data "
                f"(≥{COINT_MIN_OBS} obs required per pair).",
                icon="ℹ️",
            )
        else:
            # ── Summary KPI row ───────────────────────────────────────────────────
            _sc_n_coint = sum(1 for r in _scan_res if r["pval"] < 0.05)
            _sc_n_act   = sum(1 for r in _scan_res if r["status"] == "Actionable")
            _sc_top     = _scan_res[0]

            _sck1, _sck2, _sck3 = st.columns(3)
            with _sck1:
                st.markdown(
                    kpi_card("Cointegrated pairs",
                             f"{_sc_n_coint} / {len(_scan_res)}",
                             delta_span("E-G p < 0.05", "blue")),
                    unsafe_allow_html=True,
                )
            with _sck2:
                st.markdown(
                    kpi_card("Actionable signals",
                             str(_sc_n_act),
                             delta_span(f"|z| > {COINT_ENTRY_Z}σ today",
                                        "red" if _sc_n_act > 0 else "blue")),
                    unsafe_allow_html=True,
                )
            with _sck3:
                _top_col = "red" if _sc_top["pval"] < 0.05 else ("amber" if _sc_top["pval"] < 0.10 else "blue")
                st.markdown(
                    kpi_card("Strongest pair",
                             _sc_top["pair"],
                             delta_span(f"E-G p={_sc_top['pval']:.3f}", _top_col)),
                    unsafe_allow_html=True,
                )

            st.divider()

            # ── Ranked table ──────────────────────────────────────────────────────
            st.markdown("#### All Pairs — Ranked by Cointegration Strength")

            _STATUS_BADGE = {
                "Actionable": ("<span style='background:#3d1515;color:#f85149;border-radius:4px;"
                               "padding:2px 8px;font-size:0.78rem;font-weight:600'>Actionable</span>"),
                "Wait":       ("<span style='background:#2d2008;color:#d4ac3a;border-radius:4px;"
                               "padding:2px 8px;font-size:0.78rem;font-weight:600'>Wait</span>"),
                "Not cointegrated": ("<span style='background:#1c1c1c;color:#484f58;border-radius:4px;"
                                     "padding:2px 8px;font-size:0.78rem;font-weight:600'>Not cointegrated</span>"),
            }

            _tbl_rows_html = []
            for _r in _scan_res:
                _hl_s = f"{_r['half_life']:.1f}d" if np.isfinite(_r["half_life"]) else "∞"
                _hr_s = f"{_r['hit_rate']:.0f}%" if not np.isnan(_r["hit_rate"]) else "n/a"
                _zc   = "#f85149" if abs(_r["z_now"]) > 2 else ("#d4ac3a" if abs(_r["z_now"]) > 1 else "#8b949e")
                _pc   = "#3fb950" if _r["pval"] < 0.05 else ("#d4ac3a" if _r["pval"] < 0.10 else "#484f58")
                _tbl_rows_html.append(
                    f"<tr style='border-bottom:1px solid #21262d'>"
                    f"<td style='padding:5px 10px;font-weight:600;color:#e6edf3'>{_r['pair']}</td>"
                    f"<td style='padding:5px 10px;text-align:right;color:#8b949e'>{_r['n_obs']:,}</td>"
                    f"<td style='padding:5px 10px;text-align:right;color:{_pc}'>{_r['pval']:.3f}</td>"
                    f"<td style='padding:5px 10px;text-align:right;color:#8b949e'>{_hl_s}</td>"
                    f"<td style='padding:5px 10px;text-align:right;color:{_zc};font-weight:600'>{_r['z_now']:+.2f}σ</td>"
                    f"<td style='padding:5px 10px;text-align:right;color:#8b949e'>{_hr_s}</td>"
                    f"<td style='padding:5px 10px'>{_STATUS_BADGE.get(_r['status'], _r['status'])}</td>"
                    f"</tr>"
                )

            _tbl_header = (
                "<table style='width:100%;border-collapse:collapse;font-size:0.88rem;margin-top:4px'>"
                "<thead><tr style='border-bottom:2px solid #30363d;color:#8b949e;text-align:left'>"
                "<th style='padding:6px 10px'>Pair</th>"
                "<th style='padding:6px 10px;text-align:right'>Obs</th>"
                "<th style='padding:6px 10px;text-align:right'>E-G p-value</th>"
                "<th style='padding:6px 10px;text-align:right'>OU Half-life</th>"
                "<th style='padding:6px 10px;text-align:right'>z now</th>"
                "<th style='padding:6px 10px;text-align:right'>±1σ hit rate</th>"
                "<th style='padding:6px 10px'>Status</th>"
                "</tr></thead><tbody>"
            )
            st.markdown(
                _tbl_header + "".join(_tbl_rows_html) + "</tbody></table>",
                unsafe_allow_html=True,
            )

            # ── Pair detail charts ────────────────────────────────────────────────
            _coint_pairs = [r for r in _scan_res if r["pval"] < 0.05]
            if _coint_pairs:
                st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)
                st.markdown("#### Cointegrated Pair Detail")
                st.caption(
                    "Expand any cointegrated pair to view its spread z-score history and "
                    "raw price overlay. Actionable pairs are expanded by default."
                )

                for _cp in _coint_pairs:
                    _cp_is_act = _cp["status"] == "Actionable"
                    _cp_hl_s   = f"{_cp['half_life']:.1f}d" if np.isfinite(_cp["half_life"]) else "∞"
                    _exp_lbl   = (f"{_cp['pair']}  ·  {_cp['status']}  ·  "
                                  f"z = {_cp['z_now']:+.2f}σ  ·  OU ½-life {_cp_hl_s}")

                    with st.expander(_exp_lbl, expanded=_cp_is_act):
                        # KPI row
                        _dk1, _dk2, _dk3, _dk4 = st.columns(4)
                        with _dk1:
                            _dvc = "red" if _cp["pval"] < 0.05 else "amber"
                            st.markdown(
                                kpi_card("Cointegration",
                                         "Confirmed" if _cp["pval"] < 0.05 else "Borderline",
                                         delta_span(f"E-G p={_cp['pval']:.3f}", _dvc)),
                                unsafe_allow_html=True,
                            )
                        with _dk2:
                            st.markdown(
                                kpi_card("OU Half-life", _cp_hl_s,
                                         delta_span("mean-reversion speed", "blue")),
                                unsafe_allow_html=True,
                            )
                        with _dk3:
                            _dzc = "red" if abs(_cp["z_now"]) > 2 else ("amber" if abs(_cp["z_now"]) > 1 else "blue")
                            st.markdown(
                                kpi_card("Spread z-score", f"{_cp['z_now']:+.2f}σ",
                                         delta_span(_cp["status"], _dzc)),
                                unsafe_allow_html=True,
                            )
                        with _dk4:
                            _dhr_s = f"{_cp['hit_rate']:.0f}%" if not np.isnan(_cp["hit_rate"]) else "n/a"
                            _davg_s = (f"avg {_cp['avg_days']:.1f}d | {_cp['n_signals']} signals"
                                       if not np.isnan(_cp["avg_days"])
                                       else f"{_cp['n_signals']} signals")
                            st.markdown(
                                kpi_card("±1σ Hit rate", _dhr_s,
                                         delta_span(_davg_s, "blue")),
                                unsafe_allow_html=True,
                            )

                        # Z-score time series
                        _dfig_z = go.Figure()
                        _dfig_z.add_trace(go.Scatter(
                            x=_cp["dates"], y=_cp["z_series"],
                            mode="lines", name="Z-score",
                            line=dict(color="#58a6ff", width=1.2),
                            hovertemplate="%{x|%Y-%m-%d}: z=%{y:.2f}<extra></extra>",
                        ))
                        for _dlv, _dcol, _dlbl in [
                            ( 2.0, "rgba(248,81,73,0.5)",   "±2σ"),
                            ( 1.0, "rgba(224,123,57,0.5)",  "±1σ"),
                            (-1.0, "rgba(224,123,57,0.5)",  ""),
                            (-2.0, "rgba(248,81,73,0.5)",   ""),
                            ( 0.0, "rgba(255,255,255,0.2)", "Mean"),
                        ]:
                            _dfig_z.add_hline(
                                y=_dlv,
                                line=dict(color=_dcol, width=1, dash="dot"),
                                annotation_text=_dlbl if _dlbl else "",
                                annotation_font=dict(color=_dcol, size=9),
                                annotation_position="top right" if _dlv >= 0 else "bottom right",
                            )
                        _dfig_z.add_trace(go.Scatter(
                            x=[_cp["dates"][-1]], y=[_cp["z_now"]],
                            mode="markers", name="Current",
                            marker=dict(color="#f85149", size=9, symbol="circle",
                                        line=dict(color="white", width=1)),
                            hovertemplate=f"Now: z={_cp['z_now']:+.2f}<extra></extra>",
                        ))
                        _dfig_z.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                            xaxis=dict(title=None, gridcolor="rgba(255,255,255,0.06)"),
                            yaxis=dict(title="Z-score (σ)", gridcolor="rgba(255,255,255,0.06)"),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                            margin=dict(l=60, r=20, t=30, b=50),
                            height=300,
                            hovermode="x unified",
                        )
                        _cp_key = _cp["pair"].replace("/", "_")
                        st.plotly_chart(_dfig_z, use_container_width=True, key=f"sc_z_{_cp_key}")

                        # Price overlay
                        _dfig_px = go.Figure()
                        _dfig_px.add_trace(go.Scatter(
                            x=_cp["dates"], y=_cp["ya"],
                            name=_cp["a"].upper(),
                            line=dict(color="#58a6ff", width=1.3),
                            hovertemplate=f"{_cp['a'].upper()}: €%{{y:.1f}}/MWh<extra></extra>",
                        ))
                        _dfig_px.add_trace(go.Scatter(
                            x=_cp["dates"], y=_cp["yb"],
                            name=_cp["b"].upper(),
                            line=dict(color="#e07b39", width=1.3),
                            hovertemplate=f"{_cp['b'].upper()}: €%{{y:.1f}}/MWh<extra></extra>",
                        ))
                        _dfig_px.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                            xaxis=dict(title=None, gridcolor="rgba(255,255,255,0.06)"),
                            yaxis=dict(title="EUR/MWh", gridcolor="rgba(255,255,255,0.06)"),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                            margin=dict(l=60, r=20, t=20, b=50),
                            height=260,
                            hovermode="x unified",
                        )
                        st.plotly_chart(_dfig_px, use_container_width=True, key=f"sc_px_{_cp_key}")

                        # Trade signal commentary
                        if _cp["status"] == "Actionable":
                            _dside = (f"long {_cp['a'].upper()} / short {_cp['b'].upper()}"
                                      if _cp["z_now"] < -COINT_ENTRY_Z
                                      else f"short {_cp['a'].upper()} / long {_cp['b'].upper()}")
                            _dpr = (
                                f"{_cp['pair']} is statistically cointegrated (E-G p={_cp['pval']:.3f}). "
                                f"Spread mean-reverts with OU half-life {_cp_hl_s}. "
                                f"Current z-score = {_cp['z_now']:+.2f}σ — beyond the "
                                f"±{COINT_ENTRY_Z}σ entry threshold. "
                                f"Signal: {_dside}. "
                                f"Historical ±{COINT_ENTRY_Z}σ reversion rate: {_dhr_s} within "
                                f"{_cp['max_hold']}d ({_cp['n_signals']} signals in sample)."
                            )
                            st.markdown(commentary(_dpr, "warn"), unsafe_allow_html=True)
                        else:
                            _dpr = (
                                f"{_cp['pair']} is cointegrated (E-G p={_cp['pval']:.3f}), "
                                f"OU half-life {_cp_hl_s}. "
                                f"Current z-score = {_cp['z_now']:+.2f}σ — within the neutral band. "
                                "No active spread entry signal."
                            )
                            st.markdown(commentary(_dpr, "ok"), unsafe_allow_html=True)

            # ── Methodology ───────────────────────────────────────────────────────
            with st.expander("Methodology", expanded=False):
                st.markdown(f"""
                **Universe:** Day-ahead prices for NO2 and NL (ENTSO-E A44), TTF front-month
                (ICE/yfinance), DE and FR day-ahead prices (ENTSO-E A44, when available),
                NBP (ICE/yfinance, when available). All power prices in EUR/MWh; gas prices
                converted where possible. NBP from yfinance may include FX/unit differences
                relative to EUR/MWh — treat gas-to-gas pairs with additional caution.

                **Pair selection:** All unique combinations of available series with
                ≥{COINT_MIN_OBS} overlapping daily observations are tested.
                Results are ranked by E-G p-value (most cointegrated first).

                **Test direction:** Engle-Granger is run in both directions (Y on X and X on Y);
                the direction with the lower p-value is retained. OLS regression
                `Y = α + β·X + ε` defines the spread residual series.

                **Z-score:** `z_t = (ε_t − μ_ε(t)) / σ_ε(t)` — expanding-window moments
                (min 30 obs, no look-ahead bias). Entry threshold: |z| > {COINT_ENTRY_Z}σ.
                Exit: sign flip or |z| < 0.5σ.

                **Status logic:**
                - *Actionable* — E-G p < 0.05 AND |z_now| > {COINT_ENTRY_Z}σ
                - *Wait* — E-G p < 0.05 but |z_now| ≤ {COINT_ENTRY_Z}σ (spread in neutral zone)
                - *Not cointegrated* — E-G p ≥ 0.05

                **OU half-life:** Estimated via `Δε_t = λ·ε_{{t−1}} + const`; half-life = −ln(2)/λ.
                Capped maximum hold at {min(60, max(5, 21))}–60 days for backtest integrity.

                **Limitations:** All Tab 12 caveats apply. Gas-to-power pairs may show spurious
                cointegration during structural breaks. Rolling-window cointegration (Phase D2)
                will provide more robust entry timing signals.

                **Data:** ENTSO-E A44 (power zone day-ahead prices) | ICE/Yahoo Finance (TTF, NBP).
                History: {_SCANNER_YEARS} years requested per series.
                """)

            st.caption(
                "Sources: ENTSO-E A44 (NO2, NL, DE, FR day-ahead) | "
                "ICE/Yahoo Finance (TTF, NBP)"
            )


# ════════════════════════════════════════════════════════════════════════════
# TAB 14: FORWARD CURVE PCA
# ════════════════════════════════════════════════════════════════════════════
with tab_pca:
    st.markdown("### Forward Curve PCA — TTF M+1–M+18 Structural Decomposition")
    st.caption(
        "PCA on a model-derived TTF forward strip. "
        "PC1 = level shift · PC2 = slope (steepening/flattening) · PC3 = curvature (butterfly). "
        "Trade signals: expanding z-score of PC2 and PC3 vs ±2σ thresholds."
    )

    with st.spinner("Building TTF forward-curve panel…"):
        _pca_panel, _pca_msg = _build_ttf_pca_panel(eu_df)

    _pca_result = _run_pca(_pca_panel) if not _pca_panel.empty else None

    if _pca_result is None:
        st.warning(f"Forward curve PCA unavailable — {_pca_msg}")
    else:
        _ev       = _pca_result["ev_ratio"]
        _pc1_pct  = round(_ev[0] * 100, 1)
        _pc2_pct  = round(_ev[1] * 100, 1)
        _pc3_pct  = round(_ev[2] * 100, 1)
        _ld       = _pca_result["loadings"]
        _sc_df    = _pca_result["scores"]

        # Sanity check
        if _pc1_pct < 70.0 or _pc2_pct > 25.0:
            st.warning(
                f"Sanity check: PC1={_pc1_pct:.1f}% (expect ≥80%), "
                f"PC2={_pc2_pct:.1f}% (expect ≤15%). "
                "Panel decomposition may not be structurally sound."
            )

        # ── KPI row ──────────────────────────────────────────────────────────
        _pk1, _pk2, _pk3, _pk4 = st.columns(4)
        with _pk1:
            st.markdown(
                kpi_card("PC1 explained variance", f"{_pc1_pct:.1f}%",
                         delta_span("Level shift", "blue")),
                unsafe_allow_html=True,
            )
        with _pk2:
            st.markdown(
                kpi_card("PC2 explained variance", f"{_pc2_pct:.1f}%",
                         delta_span("Slope (carry)", "green" if _pc2_pct <= 15 else "amber")),
                unsafe_allow_html=True,
            )
        with _pk3:
            st.markdown(
                kpi_card("PC3 explained variance", f"{_pc3_pct:.1f}%",
                         delta_span("Curvature (butterfly)", "blue")),
                unsafe_allow_html=True,
            )
        with _pk4:
            st.markdown(
                kpi_card("Panel type", "Model-derived",
                         delta_span("Storage-carry adjusted", "amber")),
                unsafe_allow_html=True,
            )
        st.caption(_pca_msg)

        # ── Explained variance bar chart ──────────────────────────────────────
        st.markdown("#### Explained Variance by Component")
        _fig_ev = go.Figure(go.Bar(
            x=["PC1 (Level)", "PC2 (Slope)", "PC3 (Curvature)"],
            y=[_pc1_pct, _pc2_pct, _pc3_pct],
            marker_color=["#58a6ff", "#3fb950", "#d4ac3a"],
            text=[f"{v:.1f}%" for v in [_pc1_pct, _pc2_pct, _pc3_pct]],
            textposition="outside",
        ))
        _fig_ev.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font=dict(color="#c9d1d9", size=11),
            margin=dict(l=10, r=10, t=10, b=10), height=220,
            yaxis=dict(title="Variance explained (%)", range=[0, 115],
                       gridcolor="rgba(255,255,255,0.06)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            showlegend=False,
        )
        st.plotly_chart(_fig_ev, use_container_width=True)

        # ── Factor loadings ───────────────────────────────────────────────────
        st.markdown("#### Factor Loadings (M+1 to M+18)")
        st.caption(
            "PC1 ≈ flat across tenors (parallel shift) · "
            "PC2 ≈ monotonic (steepening/flattening) · "
            "PC3 ≈ U-shaped (butterfly)"
        )
        _fig_ld = go.Figure()
        for _pc, _col in [("PC1", "#58a6ff"), ("PC2", "#3fb950"), ("PC3", "#d4ac3a")]:
            _fig_ld.add_trace(go.Scatter(
                x=list(_ld.index), y=_ld[_pc].tolist(),
                name=_pc, line=dict(color=_col, width=2),
                hovertemplate=f"{_pc}: %{{y:.4f}}<extra></extra>",
            ))
        _fig_ld.add_hline(y=0, line=dict(color="rgba(255,255,255,0.2)", width=1))
        _fig_ld.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font=dict(color="#c9d1d9", size=11),
            margin=dict(l=10, r=10, t=10, b=50), height=280,
            yaxis=dict(title="Loading", gridcolor="rgba(255,255,255,0.06)", zeroline=False),
            xaxis=dict(title="Tenor", gridcolor="rgba(255,255,255,0.06)",
                       tickangle=45, tickfont=dict(size=9)),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            hovermode="x unified",
        )
        st.plotly_chart(_fig_ld, use_container_width=True)

        # ── Factor scores time series ─────────────────────────────────────────
        st.markdown("#### Factor Scores — Full Panel History")
        _fig_sc = go.Figure()
        for _pc, _col, _lbl in [
            ("PC1", "#58a6ff", "PC1 (Level)"),
            ("PC2", "#3fb950", "PC2 (Slope)"),
            ("PC3", "#d4ac3a", "PC3 (Curvature)"),
        ]:
            _fig_sc.add_trace(go.Scatter(
                x=_sc_df.index, y=_sc_df[_pc],
                name=_lbl, line=dict(color=_col, width=1.5),
                hovertemplate=f"{_lbl}: %{{y:.2f}}<extra></extra>",
            ))
        _fig_sc.add_hline(y=0, line=dict(color="rgba(255,255,255,0.2)", width=1))
        _fig_sc.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font=dict(color="#c9d1d9", size=11),
            margin=dict(l=10, r=10, t=10, b=10), height=300,
            yaxis=dict(title="Score", gridcolor="rgba(255,255,255,0.06)", zeroline=False),
            xaxis=dict(title=None, gridcolor="rgba(255,255,255,0.06)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            hovermode="x unified",
        )
        st.plotly_chart(_fig_sc, use_container_width=True)

        # ── Trade signals ─────────────────────────────────────────────────────
        st.markdown("#### Trade Signals — PC2 and PC3 Z-Scores")
        st.caption(
            f"Expanding z-score (no look-ahead). "
            f"PC2 z > +{PCA_ENTRY_Z:.0f}σ: curve unusually steep → short front / long back. "
            f"PC3 z > +{PCA_ENTRY_Z:.0f}σ: butterfly wide → sell wings (M+1, M+18), buy belly (M+9)."
        )
        _fig_sig = go.Figure()
        for _pcz, _col, _lbl in [
            ("PC2_z", "#3fb950", "PC2 z-score (Slope)"),
            ("PC3_z", "#d4ac3a", "PC3 z-score (Curvature)"),
        ]:
            if _pcz in _sc_df.columns:
                _fig_sig.add_trace(go.Scatter(
                    x=_sc_df.index, y=_sc_df[_pcz],
                    name=_lbl, line=dict(color=_col, width=1.5),
                    hovertemplate=f"{_lbl}: %{{y:+.2f}}σ<extra></extra>",
                ))
        for _lvl, _ann_c in [
            (PCA_ENTRY_Z,  "#f85149"),
            (-PCA_ENTRY_Z, "#3fb950"),
        ]:
            _fig_sig.add_hline(
                y=_lvl,
                line=dict(color=_ann_c, width=1, dash="dot"),
                annotation_text=f"{_lvl:+.0f}σ",
                annotation_font=dict(size=9, color="#8b949e"),
            )
        _fig_sig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.2)", width=1))
        _fig_sig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font=dict(color="#c9d1d9", size=11),
            margin=dict(l=10, r=10, t=10, b=10), height=280,
            yaxis=dict(title="Z-score (σ)", gridcolor="rgba(255,255,255,0.06)", zeroline=False),
            xaxis=dict(title=None, gridcolor="rgba(255,255,255,0.06)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            hovermode="x unified",
        )
        st.plotly_chart(_fig_sig, use_container_width=True)

        # Current signal status boxes
        _sc_last = _sc_df.iloc[-1]
        _pca_signals = [
            (
                "PC2_z", "PC2 (Slope)",
                f"Curve unusually steep (+{PCA_ENTRY_Z:.0f}σ) — Calendar spread: short front, long back",
                f"Curve unusually flat (−{PCA_ENTRY_Z:.0f}σ) — Reverse calendar: long front, short back",
            ),
            (
                "PC3_z", "PC3 (Curvature)",
                f"Butterfly wide (+{PCA_ENTRY_Z:.0f}σ) — Sell wings (M+1, M+18), buy belly (M+9)",
                f"Butterfly compressed (−{PCA_ENTRY_Z:.0f}σ) — Buy wings (M+1, M+18), sell belly (M+9)",
            ),
        ]
        for _pcz, _lbl, _bull_msg, _bear_msg in _pca_signals:
            if _pcz not in _sc_last.index:
                continue
            _z = float(_sc_last[_pcz])
            if _z >= PCA_ENTRY_Z:
                _msg, _border = _bull_msg, "#f85149"
            elif _z <= -PCA_ENTRY_Z:
                _msg, _border = _bear_msg, "#3fb950"
            else:
                _msg, _border = "In neutral zone — no trade signal", "#58a6ff"
            st.markdown(
                f"<div class='commentary' style='border-left-color:{_border};'>"
                f"<strong>{_lbl}</strong>&nbsp;&nbsp;"
                f"<span style='color:{_border};'>z = {_z:+.2f}σ</span><br>"
                f"<span style='color:#c9d1d9;font-size:0.85rem;'>{_msg}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # ── Methodology expander (mandatory) ──────────────────────────────────
        with st.expander("Methodology and data notes", expanded=False):
            st.markdown(
                "> **Model-derived panel disclosure**  \n"
                "> This analysis uses a model-derived forward curve panel. The panel is constructed "
                "from daily TTF spot prices and EU storage z-scores using a storage-carry model; "
                "it does not use observed futures contract quotes. Factor loadings and trade signals "
                "are illustrative of structural methodology and should not be used for mark-to-market "
                "or live trading decisions."
            )
            st.markdown(f"""
**Panel construction**

For each historical date t, the M+m implied forward price is:

```
F(t, m) = spot_t × seasonal_m × (1 + α × storage_z_t × exp(−m / τ))
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| `spot_t`  | TTF=F close | ICE TTF front-month daily close (Yahoo Finance) |
| `seasonal_m` | computed | Median monthly/annual ratio from 3yr TTF spot history |
| `storage_z_t` | computed | EU aggregate storage fill vs 5-yr same-DOY mean (GIE AGSI+) |
| α | {PCA_STORAGE_ALPHA} | Carry sensitivity: storage deviation → curve shape adjustment |
| τ | {PCA_STORAGE_DECAY_TAU} months | Decay: storage signal fades on far tenors (exp(−m/τ)) |

The storage carry term introduces a second source of genuine time-variation: elevated storage (positive z) compresses near-month carry (curve flattens); depleted storage (negative z) widens carry (curve steepens). This is consistent with the Schwartz-Smith two-factor commodity model and standard gas storage valuation theory.

**PCA methodology**

The panel is de-meaned column-wise (each tenor centred on its historical mean). PCA is applied via singular value decomposition on the de-meaned matrix.

- **PC1 (Level):** approximately flat loadings across tenors — represents parallel curve shifts driven primarily by spot.
- **PC2 (Slope):** monotonic loadings — represents steepening/flattening, driven by storage z-score variation propagated through the carry term.
- **PC3 (Curvature):** U-shaped or ∩-shaped loadings — residual mid-curve distortion relative to wings.

Expected variance decomposition for energy forward curves (Litterman & Scheinkman 1991 applied to interest rates; Cortazar & Schwartz 1994 extended to commodities): PC1 80–95%, PC2 4–15%, PC3 1–5%.

**Trade signal thresholds**

Z-scores use an expanding window from the start of the panel (no look-ahead bias). Entry threshold: ±{PCA_ENTRY_Z:.0f}σ.

- PC2 z > +{PCA_ENTRY_Z:.0f}σ: curve unusually steep → calendar spread: short front, long back
- PC2 z < −{PCA_ENTRY_Z:.0f}σ: curve unusually flat → reverse calendar spread
- PC3 z > +{PCA_ENTRY_Z:.0f}σ: butterfly wide → sell wings (M+1, M+18), buy belly (M+9)
- PC3 z < −{PCA_ENTRY_Z:.0f}σ: butterfly compressed → buy wings, sell belly

**Limitations**

PCA assumes a stationary covariance structure — this breaks during major regime shifts (Russian supply cut-off 2022, extreme cold snaps, Hormuz disruptions). The model-derived panel means factor loadings reflect the two-factor construction (spot + storage carry), not an observed futures term structure. A production-grade implementation would use live ICE TTF strip quotes from an EEX/Trayport feed.
""")

        st.caption(
            f"Sources: ICE/Yahoo Finance (TTF=F) | GIE AGSI+ (EU storage) | "
            f"{PCA_LOOKBACK_DAYS}-day panel | sklearn PCA | Updated on page load"
        )


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    """<div style="color:#484f58;font-size:0.72rem;line-height:1.8;">
    Built by Tobias Gudbjerg &nbsp;|&nbsp;
    Storage data: <a href="https://agsi.gie.eu" style="color:#484f58;">GIE AGSI+</a> &nbsp;|&nbsp;
    Gas price: <a href="https://finance.yahoo.com" style="color:#484f58;">ICE/Yahoo Finance (TTF)</a> &nbsp;|&nbsp;
    Power price: <a href="https://www.nordpoolgroup.com" style="color:#484f58;">Nord Pool</a><br>
    For informational purposes only. Not financial advice.
    </div>""",
    unsafe_allow_html=True,
)
