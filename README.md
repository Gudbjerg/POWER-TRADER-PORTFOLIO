# European Power and Gas Analysis Platform

A four-layer market intelligence platform for European power and gas markets, built to mirror the analytical workflow of a physical energy trader. All data from public APIs. All models built from scratch in Python.

**Live demo:** [power-trader-portfolio.streamlit.app](https://power-trader-portfolio.streamlit.app)

---

## Platform Overview

### Layer 1: Live Market Monitor

Real-time fundamentals refreshed hourly across 9 panels:

| Panel | Data source | What it shows |
|---|---|---|
| EU gas storage vs 5-year band | GIE AGSI+ | Current fill % against historical min/mean/max range |
| German gas storage | GIE AGSI+ | Germany-specific storage, 25% of annual consumption |
| NW EU LNG sendout | GIE ALSI | Daily sendout from Zeebrugge, Gate, South Hook, Dunkirk |
| TTF gas price | ICE via Yahoo Finance | Spot price, 30/90-day MA, spike alerts |
| TTF seasonal forward curve | Yahoo Finance | 18-month implied strip, Winter/Summer spread, Cal 26/27 |
| Day-ahead spot prices | Nord Pool | NO1, NO2, SE3, NL, FI with Nordic-Continental spread |
| Norwegian hydro reservoirs | ENTSO-E B31 | Weekly fill level vs P10/P50/P90 historical percentiles |
| Nordic cross-border flows | ENTSO-E | NordLink, NorNed, NSN, Skagerrak + NO2-GB trade balance |
| Solar duck curve | ENTSO-E / Fraunhofer ISE | Germany intraday price vs solar output, 14-day average |
| German coal generation | ENTSO-E A75 | Quarterly hard coal + lignite dispatch, fuel switching indicator |

### Layer 2: Quantitative Analysis

| Model | Method | Key output |
|---|---|---|
| Storage Refill Monte Carlo | Empirical bootstrap, 1,000 paths | Probability of reaching EU 90% mandate by Nov 1 |
| Gas-to-Power OLS Regression | TTF to NL day-ahead, rolling window | Residual z-score tracker, regime interpretation |
| Price Spike Detector | Rolling 30-day z-score | Active spike alerts across all Nord Pool bidding zones |
| TTF Seasonal Backtest | Rules-based injection-withdrawal strategy | Annual P&L, Sharpe ratio, ex-crisis statistics |

### Layer 3: Geopolitical and Macro Signals

- TTF price history with 12 annotated supply event overlays: Iran conflict, US LNG tariff threats, Norwegian outages, Russian supply cuts, Qatar diversions
- EU gas supply mix by origin, 2020 to present: Russia collapse, US LNG rise, Algerian variability
- Cross-commodity spillover chain: TTF to European fertiliser to wheat/corn, with rolling 90-day correlations

### Layer 4: ML Models

- **LSTM price forecaster** (Model 1): 2-layer PyTorch LSTM, 64-32 units, dropout 0.2. Features: NO2 and NL prices, TTF, spread, volatility, hydro, calendar. Train/predict UI with progress callback. Requires local training before first use.
- **HMM regime classifier** (Model 2): 4-state Gaussian HMM via hmmlearn. Regimes: hydro-driven, gas-driven, renewables-driven, geopolitical stress. Current regime badge with confidence and 30-day history.
- **FinBERT sentiment signal** (Model 3): Live scoring of energy headlines from Reuters, Montel, and Recharge RSS feeds. Daily net sentiment score. Granger causality test unlocks after 21 days of history accumulation.

---

## Running Locally

**Prerequisites:** Python 3.11+. Free API keys from:
- [GIE AGSI+](https://agsi.gie.eu): gas storage and LNG
- [ENTSO-E Transparency Platform](https://transparency.entsoe.eu): power, hydro, solar, generation

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

Opens at `http://localhost:8501`. All panels show a clear status message when API keys are absent.

---

## Deploying to Streamlit Community Cloud

1. Fork or clone this repository to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Set the main file to `app.py`
4. Under **Settings > Secrets**, add:

```toml
AGSI_API_KEY = "your_gie_key_here"
ENTSOE_API_KEY = "your_entsoe_key_here"
```

The app reads from `st.secrets` on Cloud and `.env` locally. No code changes needed.

**Note on ML models:** LSTM and HMM weights are not included in the repository. Train locally first (click Train in Layer 4), then commit the files in `models/weights/` before deploying. FinBERT sentiment history accumulates on disk automatically after deployment.

---

## Data Source Status

| Source | Status | Notes |
|---|---|---|
| GIE AGSI+ (storage) | Live | Updated daily at 19:30 CET |
| GIE ALSI (LNG) | Live | Daily sendout per terminal |
| Yahoo Finance / TTF=F | Live | Front-month continuous |
| Nord Pool (spot prices) | Live | DE-LU returns null due to licensing; NL used as proxy |
| ENTSO-E B31 (hydro) | Live | Weekly filling levels, Norway aggregate |
| ENTSO-E A75 (generation) | Live | German coal by production type |
| ENTSO-E A44 (day-ahead prices) | Live | Used for ML training data (NO2, NL, 3-year history) |
| ENTSO-E B09/B10 (flows) | Degraded | Server issue since March 2026; panel built and waiting |
| ENTSO-E B16/B19 (wind/solar) | Degraded | Same server issue; Fraunhofer ISE fallback active for solar |
| Fraunhofer ISE / energy-charts.info | Fallback | Activates automatically when ENTSO-E B16 is unavailable |

All panels display a source note when operating on fallback data.

---

## Architecture

```
app.py                          # Landing page with layer cards and data source status
pages/
  1_Live_Monitor.py             # Layer 1: 8-tab live dashboard
  2_Quant_Analysis.py           # Layer 2: 4-tab quantitative models
  3_Macro_Signals.py            # Layer 3: geopolitical and macro panels
  4_ML_Models.py                # Layer 4: ML model specs and live inference
data/
  gas_storage.py                # GIE AGSI+ (EU + DE storage, historical bands)
  lng_terminals.py              # GIE ALSI (NW EU LNG sendout)
  prices.py                     # TTF front-month via yfinance
  forward_curve.py              # TTF seasonal forward strip construction
  spot_prices.py                # Nord Pool day-ahead prices (public endpoint)
  power_flows.py                # ENTSO-E cross-border physical flows
  solar.py                      # ENTSO-E B16 solar + Fraunhofer ISE fallback
  hydro.py                      # ENTSO-E B31 hydro reservoir levels
  generation.py                 # ENTSO-E A75 German generation by fuel type
  wind.py                       # ENTSO-E B18/B19 wind + Fraunhofer ISE fallback
  commodities.py                # Cross-commodity prices via yfinance
  events.py / events.json       # Geopolitical event catalogue (manually curated)
  sentiment.py                  # FinBERT pipeline with 60-day disk persistence
components/
  storage_chart.py              # Gas storage vs historical band
  lng_chart.py                  # LNG sendout stacked area
  prices_chart.py               # TTF spot with event overlays
  forward_curve_chart.py        # Seasonal forward curve bar chart
  spot_prices_chart.py          # Multi-zone day-ahead price comparison
  flows_chart.py                # Cross-border flows + NO2-GB trade balance
  solar_chart.py                # Intraday solar cannibalisation (duck curve)
  hydro_chart.py                # Hydro reservoir vs percentile bands
  coal_chart.py                 # German coal generation fuel switching chart
models/
  storage_monte_carlo.py        # Bootstrap refill simulation, 1,000 paths
  gas_power_regression.py       # OLS regression with rolling residuals
  spike_detector.py             # Rolling z-score across all bidding zones
  ttf_backtest.py               # Seasonal injection-withdrawal backtest
  feature_assembly.py           # Daily feature matrix from all sources
  lstm_model.py                 # PyTorch LSTM definition, training, inference
  hmm_model.py                  # hmmlearn HMM, regime labelling, inference
utils/
  helpers.py                    # CSS theme, HTML KPI components, API key checks
  scenarios.py                  # Signal interpretation (status, headline, detail text)
config/
  settings.py                   # ENTSO-E EIC codes, alert thresholds
```

---

## Stack

**Language:** Python 3.11+

**Dashboard:** Streamlit, Plotly

**Data:** Pandas, NumPy, yfinance, entsoe-py, requests, feedparser

**Models:** scikit-learn, statsmodels, PyTorch, hmmlearn, transformers (FinBERT)

**Deploy:** Streamlit Community Cloud (GitHub integration, auto-redeploy on push)

---

## Key Design Decisions

**No lookahead in ML pipeline.** The scaler is fit on training data only. Rolling z-scores are computed backward-looking. Hydro percentile uses an expanding quantile, not a full-sample quantile.

**All fallbacks are visible.** Every panel that switches to a fallback data source shows an `st.info` banner identifying the alternative source and explaining the switch. No silent degradation.

**Ex-crisis statistics.** The TTF backtest reports both full-sample and ex-crisis averages. GY2021 and GY2022 are flagged as energy crisis outliers that dominate the unconditional mean.

**EU storage mandate is 90%, not 80%.** The original 2022 emergency regulation (EU 2022/1032) set an 80% target. Extended regulations raised this to 90% by November 1. All references use 90%.

---

*Built by Tobias Gudbjerg. For informational purposes only. Not financial advice.*
