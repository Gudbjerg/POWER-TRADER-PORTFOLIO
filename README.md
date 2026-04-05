# European Power and Gas Analysis Platform

A four-layer market intelligence platform for European power and gas markets, built to mirror the analytical workflow of a physical energy trader. Integrates live fundamental data, quantitative supply-demand models, geopolitical context, and market analytics, all from public APIs, all built from scratch.

**Live demo:** *(deploy link here once on Streamlit Community Cloud)*

---

## What It Does

### Layer 1: Live Market Monitor
Real-time fundamentals refreshed hourly:

| Panel | Data source | Key signal |
|---|---|---|
| EU + German gas storage | GIE AGSI+ | vs 5-year historical band |
| NW EU LNG sendout | GIE ALSI | week-on-week change, 4 countries |
| TTF gas price | ICE via Yahoo Finance | 30/90-day MA, spike alerts |
| TTF seasonal forward curve | Yahoo Finance (TTF=F history) | Winter/Summer spread, Cal 26/27 |
| Day-ahead spot prices | Nord Pool | NO1, NO2, SE3, NL, FI |
| Norwegian hydro reservoirs | ENTSO-E B31 | vs historical P10/P50/P90 |
| Nordic cross-border flows | ENTSO-E | NordLink, NorNed, NSN, Skagerrak |
| Solar duck curve | ENTSO-E | Germany intraday cannibalisation |

### Layer 2: Quantitative Analysis

| Model | Method |
|---|---|
| Storage Refill Monte Carlo | Empirical bootstrap, 1,000 paths, fan chart |
| Gas-to-Power OLS Regression | TTF to NL day-ahead with rolling z-score residuals |
| Price Spike Detector | Rolling 30-day z-score across all Nord Pool zones |
| TTF Seasonal Backtest | Rules-based injection-withdrawal strategy, 2019-2025 |

### Layer 3: Geopolitical and Macro Signals
- TTF price history with annotated supply event overlays (12 events: Iran, US LNG tariff threats, Norwegian outages, Russian supply cuts)
- EU gas supply mix by origin, 2020 to present (Russia collapse, US LNG rise, Algerian variability)
- Cross-commodity spillover: TTF to fertiliser to wheat/corn, with rolling 90-day correlations

### Layer 4: ML Models
- LSTM price forecaster and Hidden Markov Model regime classifier: specified, blocked on ENTSO-E B16/B31 data
- FinBERT NLP sentiment signal: live (experimental) - scores energy headlines from 4 RSS feeds in real time

---

## Running Locally

### Prerequisites
- Python 3.11+
- API keys (free registration):
  - [GIE AGSI+](https://agsi.gie.eu): gas storage and LNG data
  - [ENTSO-E Transparency Platform](https://transparency.entsoe.eu): power flows, hydro, solar

### Setup

```bash
git clone <repo-url>
cd power-trader-portfolio
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
AGSI_API_KEY=your_gie_key_here
ENTSOE_API_KEY=your_entsoe_key_here
```

Run the app:

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`. Panels that require API keys display a clear status message when keys are absent.

---

## Deploying to Streamlit Community Cloud

1. Push this repository to GitHub (ensure `.env` is in `.gitignore` - it is)
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub repo
3. Set the main file to `app.py`
4. Add your API keys under **Settings > Secrets** in the format:

```toml
AGSI_API_KEY = "your_gie_key_here"
ENTSOE_API_KEY = "your_entsoe_key_here"
```

The app reads from `st.secrets` on Cloud and falls back to `.env` locally. No code changes needed.

---

## Architecture

```
app.py                          # Landing page
pages/
  1_Live_Monitor.py             # Layer 1: live data dashboard
  2_Quant_Analysis.py           # Layer 2: quantitative models
  3_Macro_Signals.py            # Layer 3: geopolitical / macro
  4_ML_Models.py                # Layer 4: ML models
data/
  gas_storage.py                # GIE AGSI+ fetcher (EU + DE storage)
  lng_terminals.py              # GIE ALSI fetcher (NW EU LNG)
  prices.py                     # TTF front-month via yfinance
  forward_curve.py              # TTF seasonal forward strip
  spot_prices.py                # Nord Pool day-ahead prices
  power_flows.py                # ENTSO-E cross-border flows
  solar.py                      # ENTSO-E solar generation
  hydro.py                      # ENTSO-E hydro reservoir storage
  commodities.py                # Cross-commodity prices via yfinance
  events.py / events.json       # Geopolitical event catalogue
  sentiment.py                  # FinBERT energy news sentiment pipeline
components/
  storage_chart.py              # Gas storage vs historical band
  lng_chart.py                  # LNG sendout stacked bar
  prices_chart.py               # TTF spot with MA and overlays
  forward_curve_chart.py        # Seasonal forward curve
  spot_prices_chart.py          # Day-ahead prices by zone
  flows_chart.py                # Cross-border flow heatmap
  solar_chart.py                # Solar duck curve
  hydro_chart.py                # Hydro reservoir vs percentiles
models/
  storage_monte_carlo.py        # Refill path simulation
  gas_power_regression.py       # OLS regression + residuals
  spike_detector.py             # Rolling z-score spike detector
  ttf_backtest.py               # Seasonal injection-withdrawal backtest
utils/
  helpers.py                    # API key checks, CSS theme, HTML components
  scenarios.py                  # Signal interpretation (status, headline, detail)
config/
  settings.py                   # EIC codes, thresholds
```

---

## Data Sources

| Source | Data | Licence |
|---|---|---|
| [GIE AGSI+](https://agsi.gie.eu) | EU and German gas storage | Free API, registration required |
| [GIE ALSI](https://alsi.gie.eu) | NW EU LNG terminal sendout | Free API, same key as AGSI+ |
| [ENTSO-E Transparency](https://transparency.entsoe.eu) | Power flows, hydro, solar, generation | Free API, registration required |
| [Nord Pool](https://www.nordpoolgroup.com) | Day-ahead spot prices | Public endpoint, no key required |
| [Yahoo Finance / ICE](https://finance.yahoo.com) | TTF natural gas futures | Public, via yfinance |

---

## Stack

- **Python 3.11+**: Streamlit, Plotly, Pandas, NumPy
- **Data**: yfinance, entsoe-py, requests, feedparser
- **Models**: scikit-learn, statsmodels, transformers, torch
- **Deploy**: Streamlit Community Cloud

---

*For informational purposes only. Not financial advice.*
