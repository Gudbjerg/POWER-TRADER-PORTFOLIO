---
title: European Power and Gas Analysis Platform
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.38.0"
app_file: app.py
pinned: false
---

# ⚡ European Gas & Power Market Intelligence Platform

**A four-layer market intelligence platform built to mirror the analytical workflow of a European gas and power trader.** Live fundamental surveillance, quantitative signal generation, cross-commodity macro context, and machine learning — integrated across five pages and 19 analytical modules.

**→ Live demo: [huggingface.co/spaces/TobGud/power-trader-portfolio](https://huggingface.co/spaces/TobGud/power-trader-portfolio)**

---

## What it does

The platform covers the full analytical stack of a physical gas and power trading desk. All data is sourced from public APIs (GIE AGSI+, ENTSO-E Transparency, Nord Pool, ICE/Yahoo Finance). All models are implemented from scratch in Python.

### Layer 1 — Live Market Monitor

Real-time fundamentals refreshed hourly across ten panels:

| Panel | Source | Signal |
|---|---|---|
| EU gas storage vs 5-year band | GIE AGSI+ | Fill % vs historical min/mean/max |
| German gas storage | GIE AGSI+ | Country-level fill, ~25% of EU working volume |
| NW EU LNG terminal sendout | GIE ALSI | Daily sendout: Zeebrugge, Gate, South Hook, Dunkirk |
| TTF gas price | ICE / Yahoo Finance | Spot, 30/90d MA, spike alerts at ±2σ |
| TTF seasonal forward strip | Yahoo Finance | M+1–M+18 implied curve, Winter/Summer spread, Cal 26/27 |
| Day-ahead spot prices | Nord Pool | NO1, NO2, SE3, NL, FI — Nordic–Continental spread + 365-day history |
| Norwegian hydro reservoirs | ENTSO-E B31 | Weekly fill vs P10/P50/P90 percentile bands |
| Nordic cross-border flows | ENTSO-E B09 | NordLink, NorNed, NSN, Skagerrak + interconnector utilisation % |
| German solar duck curve | ENTSO-E / Fraunhofer ISE | Intraday price vs solar output, 14-day average |
| German coal generation | ENTSO-E A75 | Quarterly hard coal + lignite dispatch, fuel-switching indicator |

### Layer 2 — Quantitative Analysis (14 models)

| Model | Method | Output |
|---|---|---|
| Storage Refill Monte Carlo | Empirical bootstrap, 1,000 paths | Probability of reaching EU 90% mandate by Nov 1 |
| Gas-to-Power OLS Regression | TTF→NL day-ahead, rolling window | Residual z-score tracker |
| Price Spike Detector | Rolling 30-day z-score | Active alerts across all Nord Pool zones |
| TTF Seasonal Backtest | Rules-based injection-withdrawal strategy | Annual P&L, Sharpe ratio, ex-crisis stats |
| German Supply Stack | Merit order, dynamic gas/coal SRMC | Marginal fuel ID, live ENTSO-E demand override |
| Nordic Price Decomposition | Rolling 90-day multivariate OLS | Beta time series: NL, TTF, hydro, wind → NO2 |
| Wind Forecast Error Tracker | ENTSO-E A69 vs B18/B19 actuals | Daily error (GWh/%), 7-day RMSE, price correlation |
| Granger Causality (Sentiment→TTF) | SSR F-test, lags 1–7d | p-value, significance verdict, sentiment/return overlay |
| Storage–Price OLS Regression | Linear OLS, full history | Supply-risk premium: actual TTF − storage-implied fair value |
| Hydro Lead/Lag | Pearson cross-correlation, lags 0–21d | Peak lag, rolling 90d correlation |
| TTF Seasonal Norm Tracker | 5-year historical percentile bands | Current TTF vs 10th/25th/50th/75th/90th pct |
| NO2/NL Cointegration & Spread | Engle-Granger, OU half-life | Expanding z-score (no look-ahead), hedge ratio, backtest hit-rate |
| Multi-Pair Cointegration Scanner | 6-asset universe (NO2/NL/TTF/DE/FR/NBP) | All pairs ranked by E-G p-value, OU half-life, entry signal |
| Forward Curve PCA | sklearn PCA on model-derived M+1–M+18 panel | Level/slope/curvature factors, PC2/PC3 z-score trade signals |

### Layer 3 — Geopolitical & Macro Signals (4 panels)

- **Geopolitical overlay:** TTF price history with 19 annotated supply disruptions (Iran/Hormuz conflict cluster, Ras Laffan strike, US LNG tariff threat, Norwegian outages, Russian cut-off, Qatar diversions) plus 5/20/60-day momentum ribbon
- **EU gas supply mix 2020–2026:** Russia collapse from 40% → 3%, US LNG rise from 7% → 29%, Algerian variability — structural shift in European supply security
- **Cross-commodity spillover chain:** TTF → European fertiliser (Haber-Bosch economics) → global wheat/corn, with rolling 90-day Pearson correlations
- **7×7 Correlation Grid:** 90-day rolling Pearson heatmap across TTF, Brent, API2 coal, EUA, copper, aluminium, and Baltic Dry — with lead-lag analysis (±10-day lag sweep, top 15 pairs ranked by |ρ|)

### Layer 4 — ML Models

- **LSTM price forecaster:** 2-layer PyTorch LSTM, 64→32 units, dropout 0.2. Features: NO2/NL prices, TTF, spread, volatility, hydro, calendar. Honest baseline comparison — model flagged if MAE exceeds naïve benchmark.
- **HMM regime classifier:** 4-state Gaussian HMM (hmmlearn). Regimes: hydro-driven, gas-driven, renewables-driven, geopolitical stress. Current regime badge with confidence and 30-day history.
- **FinBERT sentiment signal:** Live scoring of energy headlines from Reuters, Montel, and Recharge RSS. Daily net sentiment, Granger causality test vs TTF returns.

### Layer 5 — Mispricing Dashboard

Eight rich/cheap signals aggregated into a single ranked view: historical percentile context, Bullish/Bearish/Neutral badge, High/Medium/Low confidence, and composite commentary. Signals sorted by extremity — most off-centre first. Covers TTF seasonal position, EU storage fill, NO2/NL spread z-score, NO2 vs TTF residual, Clean Spark Spread, Norwegian hydro level, TTF vs storage residual, and marginal fuel regime.

---

## Running locally

**Prerequisites:** Python 3.11+. Free API keys:
- [GIE AGSI+](https://agsi.gie.eu) — gas storage and LNG sendout
- [ENTSO-E Transparency](https://transparency.entsoe.eu) — power prices, hydro, generation, flows

```bash
git clone https://github.com/Gudbjerg/POWER-TRADER-PORTFOLIO.git
cd POWER-TRADER-PORTFOLIO
pip install -r requirements.txt
```

Create `.env` in the project root:

```env
AGSI_API_KEY=your_gie_key_here
ENTSOE_API_KEY=your_entsoe_key_here
```

```bash
streamlit run app.py
```

All panels display a clear status message when API keys are absent. The platform degrades gracefully — public data (TTF, Nord Pool spot, yfinance) loads without any keys.

---

## Stack

| Layer | Libraries |
|---|---|
| Dashboard | Streamlit 1.38+, Plotly |
| Data | Pandas, NumPy, yfinance, entsoe-py, requests, feedparser |
| Quant models | scikit-learn, statsmodels |
| ML | PyTorch, hmmlearn, HuggingFace Transformers (FinBERT) |
| Deploy | HuggingFace Spaces |

---

## Key design decisions

**No look-ahead in quantitative models.** Rolling z-scores are computed backward-looking. Expanding-window z-scores (cointegration, PCA factor scores) use only data available at each historical date.

**All fallbacks are visible.** Every panel that switches to a fallback source (e.g. Fraunhofer ISE when ENTSO-E B16 is unavailable) shows a banner identifying the alternative. No silent degradation.

**Honest ML benchmarks.** The LSTM forecaster displays a banner when its MAE exceeds the naïve persistence baseline — distinguishing "model trains" from "model adds value."

**Model-derived data disclosed.** The Forward Curve PCA panel uses a storage-carry model to construct a synthetic TTF strip. The methodology expander and a prominent KPI card both state clearly that this is model-derived, not observed futures quotes.

---

*Built by Tobias Gudbjerg. For informational purposes only. Not financial advice.*
