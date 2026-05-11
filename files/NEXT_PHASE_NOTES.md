# Next Phase Notes — Research-Driven Additions

**Compiled:** 4 May 2026
**Purpose:** New ideas backed by external research from LinkedIn, academic papers, practitioner sources. Each idea is sized, scoped, and connected to existing platform components.

---

## Industry Context Refresh

### Practitioner Sources Discovered

| Source | Why it matters |
|---|---|
| **DXT Commodities (Switzerland)** | Named in October 2025 ECMWF coverage as one of the most aggressive AI-weather adopters in European power trading. They run hundreds of computer models to predict weather-driven shifts in supply/demand. Sets the bar for what state-of-the-art looks like. |
| **Statkraft Cross-Commodity Trading desk** (Düsseldorf, Oslo) | Active hiring for "Power Strategist/Trader — Cross Commodity Trading". Trades power, gas, oil, coal, EUA, dry bulk, metals jointly. Statkraft Markets has Laxman Pararasasingam in options/systematic trading and a dedicated cross-commodity team. Job descriptions confirm what cross-commodity workflow looks like at the desk level. |
| **Nordic Power Trading (Copenhagen)** | Investment company that trades Nordic power on Nasdaq OMX. Worth following for Nordic-specific commentary. |
| **PowerBot** (intraday algo platform) | Now a certified ISV for Nord Pool day-ahead and intraday auctions. Used by many Nordic + DE power traders. Their public posts give insight into what intraday automation actually does. |
| **Norlys Energy Trading (Denmark)** | First to trade EEX Zonal Future product. Specialise in cPPAs and physical balancing. Active LinkedIn presence with technical posts on Nordic decoupling and zonal hedging. |
| **Northpool B.V.** (Netherlands, Vancouver) | European power and gas trader, public job postings give insight into team structure. |
| **Uniper Global Commodities (Düsseldorf)** | Their Quantitative Risk Modelling and Analytics (RQPR) team published the "Cointegrating Jumps" paper (Sabino, Uniper) — exact Margrabe spread option methodology applied to power/gas spread valuation. Direct relevance to Phase D PCA + cointegration work. |

### Academic Papers Worth Pulling Methodology From

| Paper | Method | Application to dashboard |
|---|---|---|
| Cufaro Petroni & Sabino (Uniper) — "Cointegrating Jumps: an Application to Energy Facilities" | Self-decomposable random variables, dependent Poisson processes, Margrabe spread option formula. Power and gas dynamics with mean-reverting OU + compound Poisson jumps. | Phase D spread option valuation. The Margrabe closed-form is exactly what should price the gas-to-power spread option in C9. |
| Anonymous arXiv 2603.04260 — "State-dependent marginal emission factors with autoregressive components" | MS-ARX (Markov-switching autoregressive with exogenous regressors) for marginal emission factors. Detects regime instability via robust structural break tests. Key finding: linear models overestimate abatement potential by masking gas-driven vs coal-driven dichotomy. | Validates the existing HMM 4-state design. Their MSDR (Markov switching dynamic regression) finding of three regimes (low-emission renewable / high-emission fossil / volatile transition) maps directly to the existing Renewables-driven / Gas-driven / Hydro-driven / Geopolitical Stress regimes. Worth citing as academic backing for the Layer 4 model design. |
| arXiv 2412.00062 — "Deep Learning-Based Electricity Price Forecast for Virtual Bidding in Wholesale Electricity Market" (ERCOT) | Transformer-based price spread forecast (real-time vs day-ahead), walk-forward validation, peak-hour-only trading achieves >50% precision and consistent profit. | Direct template for Phase E intraday/day-ahead spread strategies. The "trade only at peak hour" insight is non-obvious and worth testing on Nord Pool data. |
| arXiv 2406.13851 — "Optimizing Quantile-based Trading Strategies in Electricity Arbitrage" (Belgian BM) | State-aware models (SARMA, ARM) outperform pure time series (ARMA, ARX) when contextual information is added. Bunn et al. (2020) cited showing balancing market prices show predictable behaviour contrary to efficiency conjectures. | Quantile-loss approach (Phase C10 LSTM hardening) is academically backed. The "context features improve forecasts" finding validates the Phase A2 multivariate decomposition. |
| arXiv 2410.07224 — "Detecting Structural Breakpoints in natural gas and electricity wholesale prices via Bayesian ensemble approach" (Eurobank, 10 European markets) | Bayesian ensemble breakpoint detection + partial mutual information for causality. Tests on 10 EU markets including TTF and major power hubs. | Methodology for a possible Phase D add-on: structural break detector for TTF + power that flags regime changes in real time, complementing the HMM. The 2022 break detected by their model is the validation set. |

### ECMWF AI Forecasts (Operational Since Feb 2025)

ECMWF's **AIFS** machine-learning weather forecast became operational 25 February 2025. Outperforms physics-based models on tropical cyclone tracks (20% better) and is computationally ~1000x cheaper. **AIFS ENS** (ensemble) became operational July 2025. Open-source via the **Anemoi framework** (github, permissive licence).

What this means for the dashboard: a meaningful cross-commodity edge is increasingly weather-led. DXT Commodities runs hundreds of weather models. The dashboard's Layer 1 / Layer 2 should at least reference (or ideally consume) AIFS forecast data for German / Nordic wind + solar over the next 1–7 days. This sits squarely in the Wind Forecast Error tab's natural extension path.

---

## New Ideas (NEW1 to NEW5)

### NEW1 — Copper-Power Link Panel (Layer 3 extension)

**Why:** Copper is the canonical "Dr Copper" growth indicator. Two reasons it's directly relevant for European power: (a) industrial activity drives baseload demand, and (b) the energy transition is structurally copper-intensive (every km of transmission, every transformer, every EV). When LME copper rises sharply, German industrial demand for power follows with a lag. When copper falls, smelter curtailment risk rises.

**Implementation:**
- Add `data/commodities.py` LME copper fetcher (yfinance ticker `HG=F` or `LMCADS03 LME` via Yahoo).
- New panel on Layer 3 Cross-Commodity Spillover tab: copper price overlay on German baseload power price, rolling 90-day correlation.
- KPI: 6-month change in LME copper as a forward indicator for industrial power demand.
- Commentary block explaining the linkage and the current correlation reading.

**Effort:** ~45 min. Single fetcher addition + chart + commentary.

**Connections:** Pairs with the existing TTF→fertiliser→food chain. Both are cross-commodity transmission stories; copper is the "input side" (industrial activity → power demand), fertiliser is the "output side" (gas price → food prices).

### NEW2 — Aluminium Smelter Stress Indicator (Layer 3)

**Why:** ~15% of European aluminium smelter operating cost is electricity. When power prices spike, smelters curtail. When they curtail, baseload demand falls, which feeds back into power prices — a stabilising negative feedback. Historical precedent: 2022 European energy crisis saw Aluminium Dunkerque, Norsk Hydro, Speira, Trimet announce curtailments as power surged. Aluminium price + announced curtailment news is a real-time stress indicator for the European power complex.

**Implementation:**
- LME aluminium price via yfinance (`ALI=F` or LME aluminium ticker).
- Power-cost-as-percent-of-aluminium-cost time series: `(DE_baseload * 14 MWh per tonne) / aluminium_price`. When this exceeds ~50%, smelters lose money.
- Simple curtailment risk gauge: "Power cost share of aluminium cost: 38% (NORMAL)" / "47% (ELEVATED)" / "55% (CRITICAL — curtailment risk)".

**Effort:** ~30 min. Pure computation on existing data + new aluminium fetcher.

**Connections:** Validates the gas-to-power regression by giving a downstream demand-destruction signal.

### NEW3 — Nordic Spread Cointegration Test (Layer 2 extension)

**Why:** The Nordic Decomposition tab from Phase A2 establishes that NL is the dominant driver of NO2 (β=23.83, R²=0.79). The natural follow-up is: are these two series cointegrated? If yes, the spread is mean-reverting and tradeable as a pair. This is the foundation for Phase D pair-trading scanner. Engle-Granger cointegration test is in `statsmodels.tsa.stattools.coint`.

**Implementation:**
- New sub-panel inside the Nordic Decomposition tab.
- ADF unit root test on NO2, on NL, and on the spread (NO2 - NL).
- If both series are I(1) and the spread is I(0) → cointegrated. Print the cointegration coefficient and a half-life of mean reversion.
- Plot NO2-NL spread with z-score thresholds (entry at ±2, exit at 0).
- Stub backtest: enter when |z|>2, exit at z=0, plot equity curve.

**Effort:** ~75 min. Pure additional analytics on existing data.

**Connections:** Direct prelude to Phase D cointegration scanner. Same machinery, scaled from one pair to the universe.

### NEW4 — AIFS Wind/Solar Forecast Integration (Phase A3 extension)

**Why:** The Wind Forecast Error tab shipped in Phase A3 uses ENTSO-E A69 (TSO day-ahead wind forecasts). The next step is to compare TSO forecasts against an independent AI-based forecast — specifically ECMWF's open-source AIFS — and surface the divergence. When AIFS predicts more wind than the TSO does, German power prices may be lower than the day-ahead market is pricing in. Tradeable.

**Implementation (longer-horizon):**
- Use the open-source `anemoi-inference` package (github, permissive licence) to load a pretrained AIFS model.
- Run AIFS for the next 24h forecast at German wind farm locations.
- Aggregate to total expected GWh.
- Compare to ENTSO-E A69 TSO forecast. Plot the divergence as a new sub-panel in the Wind Forecast Error tab.
- Backtest: when AIFS - TSO > +X GWh, did NL/DE day-ahead prices end up lower than the day-ahead market closed at?

**Effort:** This is a multi-session build (model loading, inference pipeline, backtest harness). **Defer to a dedicated session.** But scope it now in `FUTURE_PLANS.md` as a Phase D candidate.

**Connections:** This is the "DXT Commodities-style" edge that an actual desk would build. Including it (even partially) signals practitioner-grade thinking to anyone reading the dashboard.

### NEW5 — Statkraft-Style Cross-Commodity Panel (Layer 3 architectural addition)

**Why:** Statkraft's public-facing description of their trading desk says explicitly: "Besides power, our traders buy and sell global gas, oil, oil products, metals and dry bulk and analyse cross-commodity influences on the power market." That's a 7-commodity cross-trading desk. The dashboard's Layer 3 currently has 3 commodities (TTF, wheat, corn). Match the Statkraft framing by expanding to a 7-commodity correlation grid.

**Implementation:**
- Layer 3 new tab: "Cross-Commodity Grid".
- 7×7 rolling 90-day correlation heatmap: TTF, Brent, API2 coal, EUA, LME copper, LME aluminium, Baltic Dry Index.
- Lead-lag table: which commodity leads which by N days, with N tested from -10 to +10 and best-lag picked by max correlation.
- Identify the strongest cross-commodity signal active right now ("Copper leading German baseload by 5 days, ρ=0.62") and surface it as a KPI.

**Effort:** ~90 min, mostly UI + correlation maths on data already fetched.

**Connections:** This becomes the centrepiece of Layer 3 and demonstrates exactly the analytical workflow Statkraft public materials describe. Direct positioning value for any Statkraft-related outreach.

---

## Connection to Marius Slette / Hafslund

Marius Slette focuses on Norwegian physical power, hydro reservoir dispatch, and the Norgespris debate. The dashboard is well-positioned for him because:
- Layer 1 Hydro tab + Phase A2 Nordic Decomposition + B2 hydro-price lead/lag = a complete Norwegian-hydro-dispatch picture.
- 19-event timeline including Hormuz cluster makes Layer 3 directly relevant to gas-driven NO2 spikes.
- BESS optimisation (Phase E) connects to flexibility trading at Hafslund's parent group.

What's still missing for him specifically: a **Norwegian system-vs-zonal price decomposition panel** showing NO1/NO2/NO3/NO4/NO5 against the system price. This is the Norgespris policy debate visualised. Cheap to build (data already in feature matrix or one Nord Pool API call away). Add it as a sub-panel on Layer 1 Prices tab when convenient.

---

## Forum and Public Source Mining (Lower Priority)

- **r/algotrading** and **r/quantfinance** posts on commodity strategies — generally low signal but occasional gems on backtest pitfalls.
- **EnergyCharts.info** (Fraunhofer ISE) — best EU power data, openly licensed, already used as fallback in `data/solar.py`. Worth checking if their wind forecasts are also exposed.
- **EMBER climate** — open dataset on European generation mix, useful for backfilling historical generation if needed.
- **Open Power System Data** — backfilled long-history datasets for European power markets.

---

## Books to Mine for Phase D Methodology

- *Energy Trading and Risk Management* — Iris Mack (introductory, broad)
- *Modeling and Pricing in Financial Markets for Weather Derivatives* — Benth & Saltyte-Benth
- *Stochastic Modelling of Electricity and Related Markets* — Benth, Saltyte-Benth, Koekebakker (mathematical, foundational for any spread-option work)
- *Commodities and Commodity Derivatives* — Helyette Geman (industry standard)

---

## Summary of Concrete Next-Session Deliverables

| Item | Effort | Priority |
|---|---|---|
| Bug 3: Slider re-fetch audit + caching fix (CRITICAL UX) | 60 min | P0 |
| Bug 4: Pre-seeded disk cache + lazy-load tabs + sidebar refresh | 90 min | P0 |
| Bug 1: Supply Stack chart missing zero-cost fuels | 30 min | P0 |
| Bug 2: "Scarcity premium" KPI sign-aware label | 15 min | P0 |
| Bug 5: LSTM honesty banner ("naive baseline beats this in current regime") | 20 min | P0 |
| B6: Nordic spread history sub-panel | 30 min | P1 |
| B5: Interconnector utilisation % | 45 min | P1 |
| B4: Granger surface in Layer 2 | 20 min | P1 |
| B1: Storage-to-price regression tab | 60 min | P2 |
| B2: Hydro-price lead/lag | 60 min | P2 |
| B3: TTF vs seasonal norm tracker | 30 min | P2 |
| NEW1: Copper-power link panel | 45 min | P3 |
| NEW2: Aluminium smelter stress indicator | 30 min | P3 |
| NEW3: Nordic spread cointegration test | 75 min | P3 |

P0 = bugs blocking UX. P1 = highest-value Phase B. P2 = remaining Phase B. P3 = research-driven add-ons. NEW4 (AIFS) and NEW5 (cross-commodity grid) deferred to dedicated sessions due to scope.

**Recommended session ordering:** P0 first (all 5 bugs, ~3.5 hrs). If time remains, P1 (~95 min). Defer P2/P3 to a follow-up session.
