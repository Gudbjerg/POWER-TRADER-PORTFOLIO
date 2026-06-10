"""
Layer 5: Mispricing Dashboard
Composite rich/cheap signal aggregator across gas, power, storage, and spreads.
Every signal carries a historical percentile and confidence rating derived from
the same underlying data and models as Layers 1–4.
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
import time as _time
from dotenv import load_dotenv

load_dotenv()

_PAGE_T0 = _time.perf_counter()

st.set_page_config(
    page_title="Mispricing Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from utils.helpers import apply_dark_theme, kpi_card, delta_span
from config.settings import (
    COINT_ENTRY_Z, COINT_MIN_OBS,
)
from data.gas_storage import get_storage_data
from data.prices import get_ttf_data
from models.feature_assembly import assemble_features
from models.ttf_backtest import fetch_ttf_history
from models.supply_stack import build_stack, identify_marginal_fuel, fetch_eua_price

apply_dark_theme()

# ── Constants ────────────────────────────────────────────────────────────────
_CCGT_HEAT_RATE = 7.5 / 3.6   # MWh gas per MWh power (~49% efficiency CCGT)
_CCGT_EMISSION  = 0.202        # tCO2/MWh power (CCGT)
_HIST_WINDOW    = 504          # ~2yr of trading days for rolling percentile
_HIST_WINDOW_3Y = 756          # ~3yr

# ── Helpers ──────────────────────────────────────────────────────────────────

def _pctile(series: pd.Series | np.ndarray, value: float) -> float:
    arr = np.asarray(series, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 10:
        return float("nan")
    return float((arr < value).sum() / len(arr) * 100)


def _confidence(pctile: float) -> tuple[str, str]:
    if np.isnan(pctile):
        return "low", "Low"
    dist = abs(pctile - 50)
    if dist >= 35:
        return "high", "High"
    if dist >= 20:
        return "medium", "Medium"
    return "low", "Low"


def _dir_from_pctile(pctile: float, low_is_bullish: bool = False) -> tuple[str, str]:
    """(direction, label) from percentile. low_is_bullish inverts convention."""
    if np.isnan(pctile):
        return "neutral", "Neutral"
    if low_is_bullish:
        if pctile < 25:
            return "up", "Bullish"
        if pctile > 75:
            return "down", "Bearish"
        return "neutral", "Neutral"
    else:
        if pctile > 75:
            return "up", "Elevated"
        if pctile < 25:
            return "down", "Depressed"
        return "neutral", "Neutral"


# ── Signal computation ────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _get_mispricing_signals() -> list[dict]:
    """Compute all mispricing signals. Returns list sorted by extremity (most off-centre first)."""
    signals: list[dict] = []

    feat     = assemble_features(years=3)
    ttf_hist = fetch_ttf_history(years=7)
    storage  = get_storage_data()

    # ── 1. TTF Seasonal Position ─────────────────────────────────────────────
    try:
        if not ttf_hist.empty and "price" in ttf_hist.columns:
            _th = ttf_hist.copy()
            _th["date"] = pd.to_datetime(_th["date"])
            _th["doy"]  = _th["date"].dt.dayofyear
            _latest     = _th.sort_values("date").iloc[-1]
            _cur        = float(_latest["price"])
            _cur_doy    = int(_latest["doy"])
            _doys       = set(range(_cur_doy - 10, _cur_doy + 11))
            _hist       = _th[_th["doy"].isin(_doys)]["price"].dropna()
            _p          = _pctile(_hist, _cur)
            _d, _dl     = _dir_from_pctile(_p)
            _conf, _clbl = _confidence(_p)
            signals.append({
                "name":     "TTF Seasonal Position",
                "category": "Gas",
                "current":  f"€{_cur:.1f}/MWh",
                "pctile":   _p,
                "conf":     _conf,
                "clabel":   _clbl,
                "direction": _d,
                "dlabel":   _dl,
                "note":     f"vs same-DOY ±10d window, 7yr history ({len(_hist)} obs)",
                "source":   "Tab 11",
            })
    except Exception:
        pass

    # ── 2. EU Storage Fill vs 5yr Seasonal ───────────────────────────────────
    try:
        _eu = storage.get("europe", pd.DataFrame())
        if not _eu.empty and "full" in _eu.columns:
            _eu = _eu.copy()
            _eu["date"] = pd.to_datetime(_eu["gasDayStart"])
            _eu["fill"] = pd.to_numeric(_eu["full"], errors="coerce")
            _eu = _eu.dropna(subset=["fill"]).sort_values("date").reset_index(drop=True)
            _eu["doy"]  = _eu["date"].dt.dayofyear
            _cur        = float(_eu["fill"].iloc[-1])
            _cur_doy    = int(_eu["doy"].iloc[-1])
            _doys       = set(range(_cur_doy - 7, _cur_doy + 8))
            _hist       = _eu[_eu["doy"].isin(_doys)]["fill"].dropna()
            _seas_mean  = float(_hist.mean()) if len(_hist) >= 5 else float("nan")
            _dev        = _cur - _seas_mean
            _p          = _pctile(_hist, _cur)
            _d, _dl     = _dir_from_pctile(_p, low_is_bullish=True)
            _conf, _clbl = _confidence(_p)
            _cur_str    = (f"{_cur:.1f}%  ({_dev:+.1f}pp vs seasonal)"
                           if not np.isnan(_seas_mean) else f"{_cur:.1f}%")
            signals.append({
                "name":     "EU Storage Fill",
                "category": "Storage",
                "current":  _cur_str,
                "pctile":   _p,
                "conf":     _conf,
                "clabel":   _clbl,
                "direction": _d,
                "dlabel":   _dl,
                "note":     (f"vs 5yr same-DOY ±7d mean ({_seas_mean:.1f}%)"
                             if not np.isnan(_seas_mean) else "vs seasonal mean"),
                "source":   "Tab 1",
            })
    except Exception:
        pass

    # ── 3. NO2/NL Spread Z-Score (Cointegration) ─────────────────────────────
    try:
        if not feat.empty and "no2" in feat.columns and "nl" in feat.columns:
            import statsmodels.api as _sm3
            _f3 = feat[["date", "no2", "nl"]].dropna()
            if len(_f3) >= COINT_MIN_OBS:
                _ya = _f3["no2"].values.astype(float)
                _yb = _f3["nl"].values.astype(float)
                _r3 = _sm3.OLS(_ya, _sm3.add_constant(_yb)).fit()
                _res = np.asarray(_r3.resid, dtype=float)
                _rs  = pd.Series(_res)
                _em  = _rs.expanding(min_periods=30).mean()
                _es  = _rs.expanding(min_periods=30).std()
                _zs  = ((_rs - _em) / (_es + 1e-9)).values
                _zv  = _zs[~np.isnan(_zs)]
                _znow = float(_zv[-1]) if len(_zv) else 0.0
                # Percentile of z-score in last 1yr
                _z_hist = pd.Series(_zs).dropna().iloc[-252:]
                _p      = _pctile(_z_hist, _znow)
                if abs(_znow) > COINT_ENTRY_Z:
                    _d  = "up" if _znow < -COINT_ENTRY_Z else "down"
                    _dl = ("Long NO2 / Short NL" if _znow < -COINT_ENTRY_Z
                           else "Short NO2 / Long NL")
                else:
                    _d, _dl = "neutral", "Neutral"
                _conf, _clbl = _confidence(_p)
                signals.append({
                    "name":     "NO2/NL Spread Z-Score",
                    "category": "Power Spread",
                    "current":  f"{_znow:+.2f}σ",
                    "pctile":   _p,
                    "conf":     _conf,
                    "clabel":   _clbl,
                    "direction": _d,
                    "dlabel":   _dl,
                    "note":     f"Expanding-window OU residual; entry at |z| > {COINT_ENTRY_Z}σ",
                    "source":   "Tab 12 / Tab 13",
                })
    except Exception:
        pass

    # ── 4. NO2 vs TTF (Gas-to-Power OLS Residual) ───────────────────────────
    try:
        if not feat.empty and "no2" in feat.columns and "ttf" in feat.columns:
            import statsmodels.api as _sm4
            _f4 = feat[["date", "no2", "ttf"]].dropna().iloc[-_HIST_WINDOW:]
            if len(_f4) >= 60:
                _r4  = _sm4.OLS(_f4["no2"].values, _sm4.add_constant(_f4["ttf"].values)).fit()
                _res = np.asarray(_r4.resid, dtype=float)
                _cur = float(_res[-1])
                _p   = _pctile(pd.Series(_res), _cur)
                _d, _dl = _dir_from_pctile(_p)
                _conf, _clbl = _confidence(_p)
                signals.append({
                    "name":     "NO2 vs TTF (OLS Residual)",
                    "category": "Power / Gas",
                    "current":  f"{_cur:+.1f} EUR/MWh",
                    "pctile":   _p,
                    "conf":     _conf,
                    "clabel":   _clbl,
                    "direction": _d,
                    "dlabel":   ("NO2 Rich" if _d == "up" else
                                 ("NO2 Cheap" if _d == "down" else "Fair Value")),
                    "note":     "NO2 residual from NO2 ~ α + β·TTF (2yr rolling OLS)",
                    "source":   "Tab 2",
                })
    except Exception:
        pass

    # ── 5. Clean Spark Spread ────────────────────────────────────────────────
    # Uses historical EUA prices (CO2.L from yfinance) so the backtest percentile
    # reflects genuine historical spread distribution, not snapshot EUA applied to history.
    try:
        if not feat.empty and "no2" in feat.columns and "ttf" in feat.columns:
            import yfinance as _yf
            _eua_hist = pd.DataFrame()
            try:
                _raw_eua = _yf.Ticker("CO2.L").history(period="3y")
                if not _raw_eua.empty and "Close" in _raw_eua.columns:
                    _eua_hist = _raw_eua[["Close"]].copy()
                    _eua_hist.index = pd.to_datetime(_eua_hist.index).tz_localize(None).normalize()
                    _eua_hist.index.name = "date"
                    _eua_hist = _eua_hist.rename(columns={"Close": "eua"}).reset_index()
            except Exception:
                pass

            if _eua_hist.empty:
                # No historical EUA available: drop signal entirely rather than use snapshot value.
                pass
            else:
                _f5 = (
                    feat[["date", "no2", "ttf"]].dropna()
                    .merge(_eua_hist, on="date", how="inner")
                    .iloc[-_HIST_WINDOW:]
                )
                if len(_f5) >= 30:
                    _css = (
                        _f5["no2"].values
                        - _f5["ttf"].values * _CCGT_HEAT_RATE
                        - _f5["eua"].values * _CCGT_EMISSION
                    )
                    _cur    = float(_css[-1])
                    _cur_eua = float(_f5["eua"].iloc[-1])
                    _p      = _pctile(pd.Series(_css), _cur)
                    _d      = "up" if _p > 75 else ("down" if _p < 25 else "neutral")
                    _dl     = ("Running" if _d == "up" else
                               ("Below Cost" if _d == "down" else "Marginal"))
                    _conf, _clbl = _confidence(_p)
                    signals.append({
                        "name":     "Clean Spark Spread",
                        "category": "Spread",
                        "current":  f"{_cur:+.1f} EUR/MWh",
                        "pctile":   _p,
                        "conf":     _conf,
                        "clabel":   _clbl,
                        "direction": _d,
                        "dlabel":   _dl,
                        "note":     (f"NO2 - TTF*{_CCGT_HEAT_RATE:.2f} - EUA*{_CCGT_EMISSION:.3f}"
                                     f"  (CCGT ~49% eff., EUA hist. avg. ~{_cur_eua:.0f}/t)"),
                        "source":   "Computed",
                    })
    except Exception:
        pass

    # ── 6. Norwegian Hydro Level ─────────────────────────────────────────────
    try:
        if not feat.empty and "hydro_pct" in feat.columns:
            _f6  = feat[["date", "hydro_pct"]].dropna().iloc[-_HIST_WINDOW_3Y:]
            _cur = float(_f6["hydro_pct"].iloc[-1])
            _p   = _pctile(_f6["hydro_pct"], _cur)
            _d, _dl = _dir_from_pctile(_p, low_is_bullish=True)
            _conf, _clbl = _confidence(_p)
            signals.append({
                "name":     "Norwegian Hydro Level",
                "category": "Supply",
                "current":  f"{_cur:.1f}% of max",
                "pctile":   _p,
                "conf":     _conf,
                "clabel":   _clbl,
                "direction": _d,
                "dlabel":   _dl,
                "note":     "Reservoir as % of expanding max (3yr); low → bullish gas/power",
                "source":   "Tab 10",
            })
    except Exception:
        pass

    # ── 7. TTF vs Storage (Regression Residual) ──────────────────────────────
    try:
        if not feat.empty and "storage_fill" in feat.columns and "ttf" in feat.columns:
            import statsmodels.api as _sm7
            _f7 = feat[["date", "ttf", "storage_fill"]].dropna().iloc[-_HIST_WINDOW:]
            if len(_f7) >= 60:
                _r7  = _sm7.OLS(_f7["ttf"].values,
                                _sm7.add_constant(_f7["storage_fill"].values)).fit()
                _res = np.asarray(_r7.resid, dtype=float)
                _cur = float(_res[-1])
                _p   = _pctile(pd.Series(_res), _cur)
                _d, _dl = _dir_from_pctile(_p)
                _conf, _clbl = _confidence(_p)
                signals.append({
                    "name":     "TTF vs Storage (OLS Residual)",
                    "category": "Gas / Storage",
                    "current":  f"{_cur:+.1f} EUR/MWh",
                    "pctile":   _p,
                    "conf":     _conf,
                    "clabel":   _clbl,
                    "direction": _d,
                    "dlabel":   ("TTF Rich" if _d == "up" else
                                 ("TTF Cheap" if _d == "down" else "Fair Value")),
                    "note":     "TTF residual from TTF ~ α + β·StorageFill (2yr rolling OLS)",
                    "source":   "Tab 9",
                })
    except Exception:
        pass

    # ── 8. Marginal Fuel Regime ──────────────────────────────────────────────
    try:
        _ttf_px  = get_ttf_data().get("prices", pd.DataFrame())
        _ttf_now = float(_ttf_px["price"].iloc[-1]) if not _ttf_px.empty else 50.0
        _eua_now, _ = fetch_eua_price()
        if _eua_now is not None:
            _demand = 60.0
            try:
                from entsoe import EntsoePandasClient as _EP8
                import pytz as _tz8
                from datetime import datetime as _dt8
                _key8 = os.getenv("ENTSOE_API_KEY", "")
                if _key8:
                    _now8 = _dt8.now(_tz8.timezone("Europe/Berlin"))
                    _s8   = pd.Timestamp(_now8.date(), tz="Europe/Berlin")
                    _ser8 = _EP8(api_key=_key8).query_load(
                        "10Y1001A1001A83F", start=_s8, end=_s8 + pd.Timedelta(days=1)
                    )
                    if _ser8 is not None and not (hasattr(_ser8, "empty") and _ser8.empty):
                        _demand = float(np.clip(round(float(_ser8.mean()) / 1000.0), 30, 90))
            except Exception:
                pass
            _stack = build_stack(_ttf_now, _eua_now)
            _marg  = identify_marginal_fuel(_stack, _demand)
            _fuel  = _marg["fuel"]
            _d8    = ("up" if _fuel in ("gas", "oil")
                      else ("down" if _fuel in ("wind", "solar", "hydro") else "neutral"))
            _dl8   = {
                "gas": "Gas-Marginal", "coal": "Coal-Marginal", "oil": "Oil-Marginal",
                "wind": "Wind-Marginal", "solar": "Solar-Marginal", "hydro": "Hydro-Marginal",
                "lignite": "Lignite-Marginal", "biomass": "Biomass-Marginal",
            }.get(_fuel, _fuel.capitalize())
            signals.append({
                "name":     "Marginal Fuel Regime",
                "category": "Demand",
                "current":  f"{_fuel.capitalize()} @ {_demand:.0f} GW",
                "pctile":   float("nan"),
                "conf":     "medium",
                "clabel":   "Medium",
                "direction": _d8,
                "dlabel":   _dl8,
                "note":     f"Merit-order stack at {_demand:.0f} GW DE demand",
                "source":   "Tab 5",
            })
    except Exception:
        pass

    # Sort by extremity (most off-centre first), NaN pctile last
    signals.sort(key=lambda s: -(abs(s["pctile"] - 50)
                                  if not np.isnan(s["pctile"]) else -1))
    return signals


# ── Page ─────────────────────────────────────────────────────────────────────

st.markdown("## Layer 5 · Mispricing Dashboard")
st.caption(
    "Composite rich/cheap signal aggregator across gas, power, storage, and spread markets. "
    "Each signal shows its current value and historical percentile within a 2–7 year rolling window. "
    "Signals are sorted by extremity. The most off-centre appear first."
)

with st.spinner("Computing signals…"):
    _signals = _get_mispricing_signals()

# Feature coverage line
_feat_cov = assemble_features(years=3)
_coverage = _feat_cov.attrs.get("coverage_report") or {}
_missing_feat = [k for k, v in _coverage.items() if not v]
if _missing_feat:
    st.caption(
        f"Feature groups unavailable: {', '.join(_missing_feat)}. "
        "Some signals may be omitted. Check ENTSO-E key (hydro/wind) and AGSI key (storage)."
    )

if not _signals:
    st.warning(
        "No signals computed. Check ENTSO-E and AGSI API keys in the Space secrets.",
        icon="⚠️",
    )
    st.stop()

# ── Summary KPI row ───────────────────────────────────────────────────────────
_n_up   = sum(1 for s in _signals if s["direction"] == "up")
_n_down = sum(1 for s in _signals if s["direction"] == "down")
_n_neut = sum(1 for s in _signals if s["direction"] == "neutral")
_n_high = sum(1 for s in _signals if s["conf"] == "high")

_kc1, _kc2, _kc3, _kc4 = st.columns(4)
with _kc1:
    st.markdown(
        kpi_card("Bullish signals", str(_n_up),
                 delta_span("↑ elevated / low-is-bullish below threshold", "red" if _n_up > 0 else "blue")),
        unsafe_allow_html=True,
    )
with _kc2:
    st.markdown(
        kpi_card("Bearish signals", str(_n_down),
                 delta_span("↓ depressed / high-is-bearish above threshold", "red" if _n_down > 0 else "blue")),
        unsafe_allow_html=True,
    )
with _kc3:
    st.markdown(
        kpi_card("Neutral signals", str(_n_neut),
                 delta_span("within ±25th–75th pctile band", "blue")),
        unsafe_allow_html=True,
    )
with _kc4:
    st.markdown(
        kpi_card("High-confidence", str(_n_high),
                 delta_span("signal in pctile < 15 or > 85", "blue")),
        unsafe_allow_html=True,
    )

st.divider()

# ── Direction pill helpers ────────────────────────────────────────────────────
_UP_PILL  = ("<span style='background:#0f2a1a;color:#3fb950;border-radius:4px;"
             "padding:2px 8px;font-size:0.78rem;font-weight:600'>{}</span>")
_DN_PILL  = ("<span style='background:#3d1515;color:#f85149;border-radius:4px;"
             "padding:2px 8px;font-size:0.78rem;font-weight:600'>{}</span>")
_NE_PILL  = ("<span style='background:#1c1c28;color:#8b949e;border-radius:4px;"
             "padding:2px 8px;font-size:0.78rem;font-weight:600'>{}</span>")
_D_PILLS  = {"up": _UP_PILL, "down": _DN_PILL, "neutral": _NE_PILL}

_CONF_PILL = {
    "high":   "<span style='color:#d4ac3a;font-size:0.76rem;font-weight:600'>High</span>",
    "medium": "<span style='color:#58a6ff;font-size:0.76rem;font-weight:600'>Medium</span>",
    "low":    "<span style='color:#484f58;font-size:0.76rem;font-weight:600'>Low</span>",
}

_CAT_COLORS = {
    "Gas":          ("#1c2230", "#58a6ff"),
    "Storage":      ("#1c2820", "#3fb950"),
    "Power Spread": ("#2d1c1c", "#f85149"),
    "Power / Gas":  ("#2a1c2d", "#d2a8ff"),
    "Spread":       ("#28200e", "#d4ac3a"),
    "Supply":       ("#1c2820", "#3fb950"),
    "Gas / Storage":("#1a1c28", "#79c0ff"),
    "Demand":       ("#2a1e10", "#e07b39"),
}

# ── Signal table ──────────────────────────────────────────────────────────────
st.markdown("#### Signal Table")

_rows_html = []
for _s in _signals:
    _bg, _fc = _CAT_COLORS.get(_s["category"], ("#1c1c1c", "#8b949e"))
    _cat_html = (f"<span style='background:{_bg};color:{_fc};border-radius:4px;"
                 f"padding:1px 6px;font-size:0.76rem;font-weight:500'>{_s['category']}</span>")

    _dp = _s["pctile"]
    if np.isnan(_dp):
        _bar_html = "<span style='color:#484f58;font-size:0.80rem'>—</span>"
    else:
        _bar_clr = ({"up": "#3fb950", "down": "#f85149", "neutral": "#58a6ff"}
                    .get(_s["direction"], "#58a6ff"))
        _bar_html = (
            f"<div style='display:flex;align-items:center;gap:6px'>"
            f"<div style='flex:0 0 110px;height:6px;background:#21262d;border-radius:3px;overflow:hidden'>"
            f"<div style='width:{_dp:.1f}%;height:100%;background:{_bar_clr};border-radius:3px'></div>"
            f"</div>"
            f"<span style='color:#8b949e;font-size:0.80rem;white-space:nowrap'>{_dp:.0f}th</span>"
            f"</div>"
        )

    _d_html   = _D_PILLS.get(_s["direction"], _NE_PILL).format(_s["dlabel"])
    _cf_html  = _CONF_PILL.get(_s["conf"], _CONF_PILL["low"])

    _rows_html.append(
        f"<tr style='border-bottom:1px solid #21262d'>"
        f"<td style='padding:9px 10px;vertical-align:top'>"
        f"  <div style='font-weight:600;color:#e6edf3;font-size:0.92rem'>{_s['name']}</div>"
        f"  <div style='color:#8b949e;font-size:0.78rem;margin-top:3px'>{_s['note']}</div>"
        f"  <div style='margin-top:4px;color:#484f58;font-size:0.74rem'>source: {_s['source']}</div>"
        f"</td>"
        f"<td style='padding:9px 10px;vertical-align:top;white-space:nowrap'>{_cat_html}</td>"
        f"<td style='padding:9px 10px;vertical-align:top;color:#e6edf3;font-weight:500;"
        f"    white-space:nowrap;font-size:0.92rem'>{_s['current']}</td>"
        f"<td style='padding:9px 10px;vertical-align:top'>{_bar_html}</td>"
        f"<td style='padding:9px 10px;vertical-align:top'>{_d_html}</td>"
        f"<td style='padding:9px 10px;vertical-align:top'>{_cf_html}</td>"
        f"</tr>"
    )

_tbl_header = (
    "<table style='width:100%;border-collapse:collapse;font-size:0.88rem;margin-top:4px'>"
    "<thead><tr style='border-bottom:2px solid #30363d;color:#8b949e;text-align:left'>"
    "<th style='padding:6px 10px'>Signal</th>"
    "<th style='padding:6px 10px'>Category</th>"
    "<th style='padding:6px 10px'>Current value</th>"
    "<th style='padding:6px 10px'>Percentile (hist)</th>"
    "<th style='padding:6px 10px'>Direction</th>"
    "<th style='padding:6px 10px'>Confidence</th>"
    "</tr></thead><tbody>"
)
st.markdown(_tbl_header + "".join(_rows_html) + "</tbody></table>",
            unsafe_allow_html=True)

st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

# ── Composite commentary ──────────────────────────────────────────────────────
st.divider()
_sig_up_names   = [s["name"] for s in _signals if s["direction"] == "up"]
_sig_down_names = [s["name"] for s in _signals if s["direction"] == "down"]

if _n_up > _n_down + 1:
    _comp_prose = (
        f"{_n_up} of {len(_signals)} signals are bullish: "
        + ", ".join(_sig_up_names[:3])
        + (f" and {_n_up - 3} more" if _n_up > 3 else "")
        + f". {_n_down} bearish, {_n_neut} neutral. "
        "Overall positioning skews toward upside risk in gas/power markets."
    )
    _comp_status = "warn"
elif _n_down > _n_up + 1:
    _comp_prose = (
        f"{_n_down} of {len(_signals)} signals are bearish: "
        + ", ".join(_sig_down_names[:3])
        + (f" and {_n_down - 3} more" if _n_down > 3 else "")
        + f". {_n_up} bullish, {_n_neut} neutral. "
        "Overall positioning skews toward downside risk / cheap market conditions."
    )
    _comp_status = "ok"
else:
    _comp_prose = (
        f"Mixed signals: {_n_up} bullish, {_n_down} bearish, {_n_neut} neutral across "
        f"{len(_signals)} indicators. No dominant directional bias. "
        "Monitor high-confidence signals for emerging conviction."
    )
    _comp_status = "ok"

from utils.helpers import commentary as _commentary
st.markdown(_commentary(_comp_prose, _comp_status), unsafe_allow_html=True)

# ── Methodology ──────────────────────────────────────────────────────────────
with st.expander("Methodology", expanded=False):
    st.markdown(f"""
    **Percentile calculation:** For each signal, the current value is ranked within its
    historical distribution (rolling 2–7 year window, same day-of-year ±10d for seasonal
    signals). Percentile = fraction of historical observations below the current value × 100.
    Requires ≥10 historical observations; shown as "—" if insufficient data.

    **Confidence levels:**
    - *High* — signal percentile is below 15th or above 85th (strong departure from historical norm)
    - *Medium* — percentile below 30th or above 70th
    - *Low* — percentile within 30th–70th range (signal near historical median)

    **Direction convention:**
    - *Elevated / Bullish* — signal is historically high. For storage and hydro (low-is-bullish),
      low values yield "Bullish"; for all others, high values yield "Elevated".
    - Spread z-scores: direction = trade signal side (long/short) when |z| > {COINT_ENTRY_Z}σ.
    - Marginal Fuel: gas/oil-marginal → "up"; renewable-marginal → "down".

    **Composite commentary** is based on a simple signal count (majority direction wins).
    It is not probability-weighted or correlation-adjusted.

    **Clean Spark Spread** formula: NO2 − TTF × {_CCGT_HEAT_RATE:.2f} − EUA × {_CCGT_EMISSION:.3f}
    (CCGT assumption: 49% efficiency, {_CCGT_EMISSION:.3f} tCO2/MWh).

    **Data sources:** ENTSO-E A44 (NO2, NL day-ahead), ICE/yfinance (TTF, EUA),
    GIE AGSI+ (EU storage), ENTSO-E B31 (hydro), ENTSO-E A65 (DE load).
    Signals with missing data sources are silently omitted from the table.
    """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    """<div style="color:#484f58;font-size:0.72rem;line-height:1.8;">
    Built by Tobias Gudbjerg &nbsp;|&nbsp;
    Gas price: <a href="https://finance.yahoo.com" style="color:#484f58;">ICE/Yahoo Finance (TTF)</a> &nbsp;|&nbsp;
    Power price: <a href="https://transparency.entsoe.eu" style="color:#484f58;">ENTSO-E A44</a> &nbsp;|&nbsp;
    Storage: <a href="https://agsi.gie.eu" style="color:#484f58;">GIE AGSI+</a><br>
    For informational purposes only. Not financial advice.
    </div>""",
    unsafe_allow_html=True,
)
