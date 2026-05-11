# Power Trader Portfolio — Integrated Roadmap & Implementation Brief

**Target:** Claude Code task brief for the next development cycle.
**Source review date:** 3 May 2026.
**Authoritative reference files (already on local repo):** `project_dashboard.md`, `POWER TRADER PORTFOLIO.md`, `FUTURE_PLANS.md`, `1st_prompt.md`, `extra_context.md`.
**Key inputs to this brief:** Project owner's per-layer feedback, screenshots of Layer 2 Storage MC and Layer 4 panels, `app.py` source on Hugging Face, the existing `FUTURE_PLANS.md` drafted in the previous Claude Code session, and external research on (a) the 2026 Iran/Hormuz macro environment, (b) ENTSO-E platform stability, (c) BESS multi-market optimisation literature, (d) Nord Pool 15-minute MTU transition, (e) cointegration/PCA in commodity markets, and (f) Marius Slette's Hafslund context via the *Tid er penger* podcast.

This document supersedes earlier drafts. It harmonises with `FUTURE_PLANS.md` rather than duplicating it — the Priority 1 models drafted in that file are accepted as-is and incorporated below.

---

## Section 0 — Strategic Framing

The current state is a high-quality multi-layer dashboard. The next state should be a **decision-support and signal-generation platform**: every panel either feeds an explicit signal, an explicit trade idea, or an explicit risk metric. The four layers should stop functioning as parallel silos and start composing — Layer 1 fundamentals feed Layer 2 quant signals, which feed Layer 3 macro overlays, which feed Layer 4 ML, which feed a new **Layer 5 Strategy Book** and **Layer 6 BESS Optimisation**.

The unifying problem to solve: *"Given everything observable today, what is mispriced, by how much, with what confidence, and how would a physical power trader express the trade?"*

---

## Section 0.5 — Live Walkthrough Findings (Direct Observation)

The live HuggingFace deployment was walked through directly before this brief was finalised. Material findings versus the prior screenshot-based assumptions:

**Cold start is the dominant user experience problem.** From cold the dashboard takes approximately 60 seconds to render Layer 1, with the loading screen showing serial calls progressing one at a time: `fetch_all_flows()` → `fetch_hydro_reservoirs()` → others. Each blocks for 10-30 seconds. This is the user's actual first-impression experience. Layer 4 then re-blocks on a separate "Loading feature matrix…" call that does not share cache with Layer 1, so navigating between layers compounds the wait. Async fetching and persistent disk cache (Section 5 C1) are higher-priority than originally framed; this is the fix that most changes the perceived quality of the platform.

**Storage Monte Carlo is more polished than the screenshot suggested.** The previous screenshot critique was based on out-of-context labelling. The live panel correctly displays the 90% EU mandate as the primary KPI, with the 80% threshold properly secondary and labelled "MIN THRESHOLD / original 2022 emergency target". The injection-rate multiplier (0.40 to 1.50) is already an exposed slider, not a fixed parameter. The methodology expander is already detailed. The remaining work for this panel is narrower than originally scoped: add the bootstrap-window UI control (all years / ex-crisis / post-2022), add a narrative subtitle that explains *why* the 0% headline is contextually accurate at current fill, and add the Hormuz scenario MC. Section 5 C9 is updated accordingly.

**Gas-to-Power Regression already produces tradeable commentary.** The residual interpretation prose ("NL day-ahead at 1.2 EUR/MWh is 92.2 EUR/MWh below the TTF gas-model prediction (93.4 EUR/MWh), a 3.1 standard deviation residual over the 44-day sample. NL power is unusually cheap relative to the gas cost curve. Possible drivers: North Sea wind surplus, weak industrial demand, or strong NorNed imports from Norway.") is already written in the right voice. R²=0.03 is properly contextualised in the methodology expander as a renewables-regime artefact rather than a model failure. The productionisation work in Section 5 C9 should extend this voice, not replace it.

**Layer 1 sub-tab structure works well.** Eight sub-tabs (Overview, Gas Storage, LNG Terminals, Prices, Power Flows, Generation, Solar, Hydro Reservoirs) are a lot but they navigate fluidly. No redesign needed. The Overview tab in particular uses ELEVATED / NORMAL signal badges which are exactly the right design pattern for the Mispricing Dashboard in Phase D — visual language should match.

**Silent partial-data nulls observed in Layer 1 LNG.** "No sendout data available for: Great Britain" appears as an inline text caption rather than a clearly-flagged data-source banner. This is a candidate for the Section 5 C3 fallback registry pattern — GB LNG sendout should fall back to Elexon Insights or a National Grid feed rather than silently render no data.

**Layer 1 KPI cards are well-coloured.** The N-C SPREAD card uses red text "Nordic premium, import signal", TTF uses red "elevated" framing, gas storage uses green "below 5yr avg, min 30.2%". The semantic colour layer is doing useful work and should not be flattened in any future redesign pass.

**Threshold annotations on the TTF chart are well-executed.** Dashed horizontal lines at €30 (Pre-crisis avg), €35 (Elevated), €50 (High risk) with right-aligned labels. This is the design pattern the cointegration scanner spread charts in Section 8 D1 should reuse for entry/exit thresholds.

**Layer 4 was not directly observable** — the feature matrix load did not complete during the walkthrough window, which is itself the most important Layer 4 finding. The LSTM, HMM, and FinBERT critiques remain based on screenshot evidence and prior-session context. The blocking feature-matrix load is added to the bug catalogue.

---

## Section 1 — The 2026 Macro Reality (Critical Context)

The single most important context update versus the dashboard's last meaningful refresh. Several panels need re-anchoring against this reality.

**The Iran war and Strait of Hormuz crisis** dominates everything. Operation Epic Fury launched 28 February 2026; the Strait was effectively closed from 4 March; Ras Laffan suffered Iranian drone strikes that took roughly 17% of QatarEnergy's 77 Mt/yr LNG export capacity offline; QatarEnergy declared force majeure; the strait has cycled between closure, partial opening, US naval blockade, and a fragile ceasefire since.

Key facts for market structure:
- TTF doubled from ~€32/MWh on 27 February to over €60/MWh in mid-March, peaked above €70/MWh on confirmation of Ras Laffan damage, sat near €50/MWh by late April, and rose again as Hormuz re-closed in late April.
- European gas storage entered the refill season at roughly 30% — its lowest level in years following a harsh 2025–2026 winter.
- QatarEnergy's confirmed multi-year repair timeline implies structural supply loss of approximately 20 Mt/yr through 2027–2028 versus pre-war expectations.
- Goldman Sachs and the IEA have both flagged this as the largest oil-market supply disruption in history and the largest gas-market disruption since the 2022 Russia shock.
- ECB postponed planned rate cuts on 19 March, raised 2026 inflation projections, cut growth projections.

**Implications for the dashboard:**
1. The Layer 3 geopolitical event catalogue (currently 12 events) must add an Iran/Hormuz cluster — see Section 6.
2. The Layer 4 LSTM test window (Q1–Q2 2026) coincides with the Hormuz crisis, which is structurally unlike any training data; the test MAE worse than naive baseline is partly a regime-mismatch artefact, not pure model failure. See Section 7.
3. The Layer 2 Storage MC headline of "0% probability of reaching 90% mandate at 1.00x injection rate" is contextually accurate given current fill and Hormuz-disrupted LNG inflows — the panel should display a narrative subtitle that explains *why*.
4. The cross-commodity spillover panel must capture the gas → fertiliser → food chain that is currently playing out in real time.

The current `app.py` Research Pipeline section already lists "Qatar LNG Diversion: Event Study" and references the "March 2026 Hormuz escalation". This work is now operational rather than aspirational.

---

## Section 2 — Current State (Reconciled With Claude Code's Status Check)

Reproducing and integrating the status check from the previous Claude Code session, with corrections from the most recent screenshots and source reads.

### Layer 1 — Live Market Monitor

| Panel | Spec | Status | Notes |
|---|---|---|---|
| EU gas storage vs 5-year band | ✅ | Live | EU + DE, historical min/mean/max |
| German gas storage | ✅ | Live | Country-specific |
| NW EU LNG sendout | ✅ | Live | Zeebrugge, Gate, South Hook, Dunkirk + WoW alert |
| TTF gas price + MA + spike alerts | extra | Live | Not in original spec |
| TTF seasonal forward curve | extra | Live | 18-month implied strip |
| Day-ahead spot prices (NO1, NO2, SE3, NL, FI) | ✅ | Live | DE-LU null → NL proxy |
| Norwegian hydro reservoirs | ✅ | Live | P10/P50/P90 bands |
| Nordic cross-border flows | ✅ | Built, degraded | ENTSO-E B09/B10 server down since March 2026 |
| Solar duck curve | ✅ | Live (fallback) | ENTSO-E B16 down → Fraunhofer ISE active |
| German coal generation | extra | Live | Fuel switching indicator |

**Summary:** 7/7 spec panels live, 3 extras added, 1 panel awaiting ENTSO-E recovery. Per the wider research, the ENTSO-E B09/B10/B16/B19 outage is structural per the active `entsoe-py` issue tracker (multiple open issues #502, #510, #511, #512 from late 2025/early 2026). Treat the fallback architecture as permanent, not transitional.

### Layer 2 — Quantitative Analysis

| Model | Spec | Status | Notes |
|---|---|---|---|
| Storage Refill Monte Carlo | ✅ | Live | 1000 paths, 90% mandate line, rate slider |
| Gas-to-Power OLS Regression | ✅ | Live | NL proxy for DE-LU, rolling residual z-score |
| Nordic Price Decomposition | ✅ | **Not built** | Spec'd in `FUTURE_PLANS.md` Priority 1A |
| Merit Order / Supply Stack | ✅ | **Not built** | Spec'd in `FUTURE_PLANS.md` Priority 1B |
| Price Spike Detector | ✅ | Live | Rolling 30d z-score all zones |
| Wind Forecast Error Tracker | ✅ | **Not built** | Spec'd in `FUTURE_PLANS.md` Priority 1C |
| TTF Seasonal Backtest | extra | Live | Annual P&L + ex-crisis stats |

**Summary:** 3/6 spec models live, 3 spec'd-and-pending in `FUTURE_PLANS.md`, 1 extra. **Priority 1 is to complete the three pending models.**

### Layer 3 — Macro Signals

| Panel | Spec | Status | Notes |
|---|---|---|---|
| Geopolitical event overlay | ✅ | Live | 12 events in `events.json` |
| EU gas supply mix | ✅ | Live | Hardcoded 2020–2026, no free API |
| Cross-commodity spillover chain | ✅ | Live | TTF + wheat + corn, rolling 90d correlations |

**Summary:** 3/3 spec panels live. Major gap: no Iran/Hormuz cluster yet — see Section 6.

### Layer 4 — ML Models

| Model | Spec | Status | Notes |
|---|---|---|---|
| LSTM price forecaster | ✅ | Live | 751 days, 14 features, 133KB weights in main |
| HMM regime classifier | ✅ | Live | 4-state, **confidence + label bugs already fixed** in earlier Claude Code session |
| FinBERT sentiment signal | ✅ | Live (HF only) | Accumulating; Granger test unlock ~27 April 2026 (now passed) |

**Summary:** 3/3 spec models built. The "100% Renewables-driven" issue I previously flagged was from a pre-fix screenshot — the regime label and confidence bugs are reportedly resolved. The deeper model-quality issues (LSTM regressing to mean, HMM feature dominance under stress) remain open and are addressed in Section 7.

### Deployment

| Platform | URL | Status |
|---|---|---|
| Streamlit Cloud | power-trader-portfolio.streamlit.app | Live |
| Hugging Face Spaces | huggingface.co/spaces/TobGud/power-trader-portfolio | Live |

The repo uses a documented dual-branch architecture: `main` for Streamlit Cloud (no torch), `hf-deploy` for HuggingFace (with torch + cached features). The hard rule never to run `git checkout main -- requirements.txt` on `hf-deploy` is preserved in `project_dashboard.md` and must be respected by any future work.

---

## Section 3 — Phase A: Complete the Three Pending Layer 2 Models (Priority 1)

These three models are already drafted in `FUTURE_PLANS.md`. The specs there are good; this section accepts them and adds nothing material. Build order per the previous Claude Code session:

1. **Supply Stack** first — least data-dependent, no new fetchers needed, standalone logic.
2. **Nordic Price Decomposition** — uses existing feature matrix, just needs `statsmodels` multivariate OLS.
3. **Wind Forecast Error Tracker** — needs new A69 fetcher in `data/wind.py`, most moving parts.

### Cross-Reference With This Brief

The three models connect to wider architecture as follows:

- **Nordic Price Decomposition** is the analytical foundation for the Layer 5 cointegration scanner (Section 8). The same regression machinery, extended with cointegration testing, gives a pair-trading signal.
- **Supply Stack** is the analytical foundation for the clean-spark and clean-dark spread panels added in Section 5, and for the LSTM feature enrichment in Section 7. The marginal-fuel identification logic is reused.
- **Wind Forecast Error Tracker** outputs are a direct feature for the LSTM in Section 7 — a good wind forecast is the single strongest predictor of next-day spot, and the *error* is a vol indicator.

Build all three with the existing dual-branch architecture in mind. Any new file in `data/` or `models/` should respect the cache seeding rules in `project_dashboard.md`.

---

## Section 4 — Phase B: Zero-New-API Quick Wins (Priority 2)

`FUTURE_PLANS.md` lists six ideas that use only data already fetched. All are accepted into this roadmap as Phase B. Restated here for completeness with light enrichment.

### B1. Storage-to-Price Regression
Scatter EU storage fill % vs TTF, coloured by season. OLS fit + confidence band. "At current storage, TTF historically trades at X±Y EUR/MWh." Rolling residual = supply-risk premium signal. **Connection to Section 8:** the residual is one of the inputs to the Mispricing Dashboard.

### B2. Hydro-Price Lead/Lag Analysis
Cross-correlation of `hydro_pct` and NO2 at lags 0–21 days. Plot CCF vs lag. Annotate peak predictive lag. **Connection to Section 7:** the peak lag identified here is the rationale for the lookback window in any Norway-zone-conditional LSTM feature.

### B3. TTF vs Seasonal Norm Tracker
TTF current vs historical seasonal range, same band methodology as the storage chart. "TTF is currently X% above/below seasonal average." Pairs naturally with the storage chart on Layer 1.

### B4. Granger Causality Surface in Layer 2
The Layer 4 Granger test should also surface in Layer 2 as a quant signal. "Energy news sentiment is/is not a statistically significant predictor of next-day TTF moves (p=X)." History unlock ~27 April 2026 has now passed; the test should be runnable.

### B5. Interconnector Utilisation %
Add capacity constants to `config/settings.py` and divide actual flow by max capacity. Show as KPI cards beside the existing flows chart. Activates when ENTSO-E B09/B10 recover, but the static capacity-constants work can land now. **Connection to Section 5:** this becomes the foundation for the per-interconnector dashboard expansion.

### B6. Nordic Spread History
NO2–NL spread (already in feature matrix as `no_nl_spread`) gets a dedicated 1-year history sub-chart in the Prices tab. Threshold annotation: "When spread > +€20 NL premium, NordLink historically runs at >90% capacity." **Connection to Section 8:** this is the canonical Nordic-Continental cointegration pair; the cointegration scanner should include it as a test case.

---

## Section 5 — Phase C: Architectural Additions (My Layer)

This is where this brief adds material that is not in `FUTURE_PLANS.md`. Each item below is a strategic addition that builds on top of the existing platform.

### C1. Hosting Migration — Off Hugging Face Spaces

Hugging Face Spaces is a poor fit for a real-time analytics dashboard. Cold starts are slow, the wrapper UI is heavy, the free tier is contended, and the "Fetching metadata from the HF Docker repository..." chrome at the top of every page is visible to anyone visiting the site. The ephemeral filesystem is also the root cause of the FinBERT sentiment-history persistence weakness.

| Platform | Cost | Fit | Cold start | Custom domain | Verdict |
|---|---|---|---|---|---|
| **Render** | $7/mo | Excellent | Fast | Yes, free TLS | **Recommended primary** |
| **Fly.io** | ~$5–10/mo | Good | Very fast | Yes | If EU residency or low-latency to ENTSO-E EU servers matters |
| **Railway** | $5/mo + usage | Good | Fast | Yes | Simpler than Fly |
| **Streamlit Cloud** | Free | Native | Medium | Subdomain only | Keep as backup mirror |
| **HuggingFace Spaces** | Free | Acceptable | Slow | No | **Deprecate** |

**Architecture:**
1. Primary on Render with custom domain (e.g. `tobiasgudbjerg.com/power`).
2. Streamlit Cloud kept as backup mirror — the dual-branch architecture continues to apply.
3. Hugging Face Spaces deprecated. Keep the HF account for **model artefacts in a model repo, not a Space.** The Streamlit app pulls weights from the model repo at startup with `huggingface_hub.snapshot_download`, cached locally via `@st.cache_resource`.
4. **Persist FinBERT sentiment history to a free Postgres** (Render Postgres or Supabase free tier). Solves the ephemeral-filesystem problem permanently.

**Branch architecture under Render:** the new platform target replaces `hf-deploy`. The dual-branch system can become `main` (Streamlit Cloud) and `render-deploy` (Render with full ML stack). Migrate the `hf-deploy` patterns over.

### C2. 15-Minute MTU Resolution

Since 30 September 2025, the Single Day-Ahead Coupling (SDAC) auction across the EU operates on 96 fifteen-minute Market Time Units per day rather than 24 one-hour units. Any panel that currently aggregates day-ahead prices to hourly is showing a smoothed, lower-resolution version of the actual market. The intraday continuous market also operates at fifteen-minute resolution.

Implementation:
- Migrate `data/spot_prices.py` and `data/prices.py` to natively fetch 15-minute MTUs.
- Add a UI toggle for resolution: 15-min (native), hourly (mean), daily (mean). Default to 15-min for date ranges under 7 days, hourly for 7–60 days, daily beyond.
- Add a 15-minute volatility KPI ("intra-day price range") which is the variable a working intraday trader cares about.

This is a hard requirement, not a polish item — it directly affects the credibility of the dashboard with any practitioner.

### C3. Permanent Fallback Registry Pattern

The ENTSO-E B16/B19 outage is structural per the open `entsoe-py` issue tracker. Stop treating Fraunhofer ISE as a fallback and treat it as a primary source for Germany. Build a registry abstraction:

```python
# data/sources/registry.py
class DataSourceRegistry:
    def __init__(self, sources: list[Callable], name: str):
        self.sources = sources  # ordered list of provider callables
        self.name = name

    def fetch(self, **kwargs) -> tuple[pd.DataFrame, str]:
        """Returns (data, provider_name_used). Tries each in order."""
        errors = []
        for src in self.sources:
            try:
                df = src(**kwargs)
                if df is not None and len(df) > 0:
                    return df, src.__name__
            except Exception as e:
                errors.append((src.__name__, str(e)))
        raise AllSourcesFailedError(self.name, errors)
```

Registry order per data type:
- **DE wind/solar:** Energy-Charts.info (primary), ENTSO-E B16/B19 (when up), SMARD.de (Bundesnetzagentur), Open Power System Data (historical backfill).
- **GB wind/solar:** Elexon Insights API (primary), ENTSO-E (fallback).
- **ES wind/solar:** Red Eléctrica de España (REE) ESIOS API (primary), ENTSO-E (fallback).
- **IT wind/solar:** Terna API (primary), ENTSO-E (fallback).
- **Nordic flows:** Nord Pool web data (primary), ENTSO-E B09/B10 (when up), Statnett/Svenska Kraftnät direct (fallback).

Each panel surfaces a small badge identifying the active provider — already implemented as `st.info` banners in many places, standardise it across all panels.

### C4. Iran/Hormuz Event Cluster

Add to `data/events.json` with at minimum these markers, each with date, label, description, category (geopolitical/supply/policy), impact (critical/warn):

- 28 February 2026 — Operation Epic Fury, US/Israel strikes on Iran, Khamenei killed.
- 2 March 2026 — Iranian drone strikes on Qatari gas facilities, QatarEnergy halts production.
- 4 March 2026 — Strait of Hormuz officially closed.
- ~20 March 2026 — Ras Laffan damage confirmed (17% of capacity, 3-5 year repair).
- 8 April 2026 — first ceasefire announced.
- 13 April 2026 — US naval blockade of Iranian ports (dual blockade).
- 17 April 2026 — Iran reopens Strait under truce.
- ~19 April 2026 — Strait re-closed.

Each event should carry an "impact score" so the chart can render visual weights, and a TTF Δ% in the 24h after the event for empirical magnitude calibration.

### C5. Cross-Commodity Spillover Expansion

The current TTF → fertiliser → wheat/corn chain is well-conceived but narrow. Add:

- **Copper as Dr Copper macro indicator** — copper is structurally tied to grid investment (every kilometre of transmission upgrade requires copper). Plot LME copper vs DE industrial production index, and against rolling 90-day correlation with German baseload.
- **Aluminium** — ~15% of European smelter cost is electricity. When power spikes, smelters curtail; when they curtail, demand falls. Aluminium price + smelter curtailment news at Trimet, Aluminium Dunkerque, Norsk Hydro, Speira.
- **EUA carbon price** dedicated panel — spot, MSR impact, auction calendar, EEX positioning.
- **Crude oil (Brent)** + refined products (gasoil) — oil-indexed gas contracts still exist in long-term LNG agreements.
- **Shipping freight (Baltic Dirty/Clean Tanker Index)** — Hormuz insurance premiums jumped from 0.125% to 0.2-0.4% per transit during the crisis.
- **Urea / ammonia** — gas is ~70% of urea production cost; reshape the existing fertiliser panel to show this explicitly.
- **EUR/USD** — US LNG priced in USD; drives the JKM-TTF arbitrage.
- **EUR/NOK** — Norwegian power exports settle across this rate.

### C6. Power Flows — Major Expansion (Top Priority Within Layer 1)

The project owner identified this as the highest-leverage Layer 1 area; agreed. Concrete additions:

- **Per-interconnector dashboard.** NordLink (DE-NO2), NorNed (NL-NO2), NSL (GB-NO2), Skagerrak (DK1-NO2), SydVästlänken (NO-SE), Cobra (DK-NL), IFA / IFA2 / ELink (FR-GB), BritNed (NL-GB), Viking Link (DK-GB). Each with: scheduled flow, day-ahead price spread, congestion rent (= flow × spread), implied utilisation %.
- **Implicit auction efficiency** — when two zones have a price difference > 0 but the interconnector is not flowing in the high-price direction, that is congestion or curtailment — flag it.
- **Flow-based vs ATC capacity comparison** for Continental Europe (ENTSO-E B17 + Nordic flow-based market coupling).
- **Trade-balance accounting per zone.** Net imports/exports in TWh and EUR/MWh-weighted. Norway as a battery for Europe — quantify it.
- **Cross-border price convergence index.** Rolling correlation of NO2 and NL day-ahead, FR and DE day-ahead, NO2 and GB. Drops in correlation = constrained interconnector = arbitrage opportunity.

### C7. LNG Terminal Depth

- **Per-terminal utilisation rate** (capacity ÷ sendout).
- **Vessel arrivals layer** via free/cheap AIS feed (Datalastic, MarineTraffic API low volume).
- **JKM-TTF spread panel** with shipping cost overlaid. Relevant given Hormuz dynamics. JKM data sources: EIA weekly, S&P Global Platts limited free quotes, JKM=F via CME if Yahoo Finance carries it, or the LNG ETF as proxy. **Connection to `FUTURE_PLANS.md` 3B** — that idea is incorporated here.
- **Force majeure tracker** — manual research artefact, table of currently force-majeure'd LNG facilities with lost capacity and expected restart.
- **Sendout vs TTF correlation panel.**

### C8. Clean-Spark and Clean-Dark Spreads

Currently absent and is a glaring omission for a power-trading dashboard.

```
clean_spark = power_price - (gas_price / efficiency_ccgt) - (eua_price * emission_factor_gas)
clean_dark  = power_price - (coal_price / efficiency_coal) - (eua_price * emission_factor_coal)
```

Plot the rolling spread, the spread between spark and dark (which determines the merit-order switching point), and a heat-map of the spread by hour of day. **Connection to Phase A:** this builds on the Supply Stack model from Priority 1B.

### C9. Layer 2 Productionisation

Beyond completing the three pending models, the existing Layer 2 models need productionisation passes:

**Storage Refill Monte Carlo:**
- *Live observation: most of the bugs flagged from the screenshot are already correctly handled in production.* The 90% EU mandate is the primary KPI, the 80% threshold is correctly secondary, the injection-rate multiplier is already an exposed slider, and the methodology expander is already detailed.
- Remaining work is narrower than originally scoped:
- Add a narrative subtitle that explains *why* the 0% headline is contextually accurate at current fill (33.1%, post-Hormuz LNG disruption, historical injection rates not sufficient).
- Add bootstrap-window UI control: "all years", "ex-crisis", "post-2022", "this-year-equivalent". Report side-by-side.
- Add a Hormuz scenario: "Strait closed for X more weeks" → simulate TTF and storage outcomes.

**Gas-to-Power OLS Regression:**
- Productionise to full structural model: `power = β0 + β1·gas + β2·coal + β3·EUA + β4·wind_de + β5·solar_de + β6·hydro_no + β7·load_residual + β8·dummy_weekend + β9·dummy_winter + ε`.
- Report t-stats, R², residual diagnostics, Durbin-Watson, Breusch-Pagan, Ramsey RESET.
- Implied heat rate tracker (slope on TTF over time).
- Carbon-adjusted regression with EUA pass-through coefficient.
- Residual z-score as alpha signal — backtest it.
- Rolling-window regression with regime dummies (per HMM state).
- Gas-to-power spread option valuation.
- Error correction model (ECM) if cointegration with TTF holds.

**Day-Ahead Spike Detector:**
- Spike conditional probability model — logistic regression with calibrated probability output.
- Cost-weighted spike accounting (total EUR cost of all spikes YTD per zone).
- Intraday spike persistence — Markov transition matrix across zones.
- 15-minute spike detection (with the SDAC transition).

**TTF Seasonal Backtest:**
- Already live with annual P&L + ex-crisis stats per the status check, but expand to a strategy book:
  1. Seasonal calendar spread (current).
  2. Storage arbitrage intrinsic.
  3. Storage extrinsic (real option) — switching strategy.
  4. Cal-spread mean reversion.
  5. TTF-NBP basis trade (cointegrated).
  6. Cross-fuel switching (clean spark vs clean dark).
  7. **Hormuz event-window strategy** — rules-based: when an LNG-relevant geopolitical event flag fires, go long TTF M+1 with stop at -5% / target at +15%. Backtest against the existing 12-event catalogue plus the new Hormuz cluster.

Each strategy reported with equity curve, Sharpe, Sortino, max DD, hit rate, ex-crisis stats, transaction-cost sensitivity.

### C10. Layer 4 ML Hardening

**LSTM:**
- Predict log-returns, not levels. Prices are non-stationary; returns are roughly stationary.
- Quantile loss (pinball) instead of MSE — gives a forecast interval, more useful and more honest about uncertainty.
- Add exogenous forecast features: ENTSO-E A69 wind/solar forecasts (the Wind Forecast Error Tracker from Phase A produces these directly).
- Walk-forward validation, not single train-test split.
- Naive baseline comparison surfaced in the UI: "this model beats naive baseline X% of the time on Y metric". If it does not beat naive, say so explicitly.
- "Model is not useful in this regime" indicator — when conditional volatility exceeds threshold or HMM flags Geopolitical Stress, surface a warning rather than a misleading point forecast.
- A/B test against a DLinear or NHITS baseline (recent literature shows these beat LSTM on financial time series with limited data).

**HMM:**
- Confidence + label bugs already fixed per the previous session. The remaining work is feature-side:
- Audit per-state mean vectors so dominant features per regime are visible.
- Lengthen the TTF z-score normalisation window to 5 years (so 2026 stress shows up as a genuine outlier).
- Geopolitical Stress override rule: if news sentiment z < -2 *and* TTF z > +1.5 over a 5-day window, force-flag Geopolitical Stress regardless of HMM output.
- Transition matrix visualisation (4×4 heatmap).
- Regime-conditional return statistics.
- Out-of-sample regime forecasting.

**FinBERT:**
- Migrate sentiment history off the HF ephemeral filesystem to HF dataset repo or external Postgres (covered by C1).
- Topic-disaggregated sentiment via NER tagging.
- Source weighting (Reuters vs Montel vs Recharge — empirically derive credibility weights from Granger significance).
- Sentiment as LSTM feature (rerun and report whether MAE improves).
- Sentiment-spike trading rule backtest.

---

## Section 6 — Phase D: Layer 5 Strategy Book and Cross-Layer Synthesis

This is the layer that ties everything together and answers the "everything is separated" critique. Net-new content not in `FUTURE_PLANS.md`.

### D1. Cointegration & Pair-Trading Scanner

Engle-Granger and Johansen tests across the full universe of tradeable contracts. Output: a live table of cointegrated pairs ranked by current spread z-score. Cell click → detailed view with equity curve of a 2σ-entry / 0.5σ-exit strategy.

Coverage:
- TTF vs NBP (UK gas, the canonical European gas hub pair)
- TTF vs PSV (Italy)
- TTF vs PEG (France)
- TTF vs CEGH (Austria)
- DE base vs FR base (power)
- NO2 vs NL (Nordic-Continental power spread — direct extension of B6)
- Clean-spark spread vs clean-dark spread
- Coal vs gas (cross-fuel switching)
- TTF vs JKM (LNG arbitrage spread)

Implementation: pre-test for unit roots with ADF; if both series are I(1), test for cointegration; if cointegrated, estimate the cointegrating vector; the residual (the spread) is mean-reverting and tradeable. Standard pairs-trading entry/exit rules. Academic reference: Engle-Granger (1987), Johansen (1988, 1995). European energy markets specifically: convergence-trading research has documented economically and statistically significant risk-adjusted excess returns.

### D2. PCA on the Forward Curve

Apply PCA to the TTF and power forward curves. Standard finding: first three principal components explain ~99% of variance, corresponding to **level, slope, and curvature** (same as fixed-income yield curve decomposition). Applications:
- Identify when the curve shape is anomalous vs the historical PCA-reconstructed shape — that residual is a curve-shape arbitrage signal.
- Hedge a complex position with three liquid forward contracts (level + slope + curvature replication).
- Cross-curve PCA: joint PCA on TTF and power forwards captures co-movement; deviations are mispricing candidates.

This connects directly to the *Commodity Derivatives* and *Quantitative Methods* modules in the Bayes coursework.

### D3. Mispricing Dashboard

Single page that aggregates every "rich/cheap" signal generated elsewhere:
- Gas-to-power regression residual z (from Section 5 C9)
- Storage Monte Carlo P50 vs implied forward curve
- Cointegration spread z-scores (from D1)
- PCA forward-curve residuals (from D2)
- Storage-to-price residual (from B1)
- Sentiment-conditional valuation gaps
- HMM regime-conditional expected return vs current spot
- Clean-spark vs clean-dark relative value
- TTF-JKM arbitrage gap (from C7)

Each row: signal, current value, historical percentile, suggested direction, confidence. This is the page that makes the dashboard a tool rather than a viewer.

### D4. Risk Dashboard

- **Portfolio simulator.** Define a notional portfolio (long DE base Cal+1, short TTF Cal+1, long EUA, etc). Compute VaR (parametric, historical, MC), expected shortfall, P&L attribution by factor.
- **Stress scenarios.** "Russia cuts remaining gas", "Norwegian outage 5 GW for 14 days", "Cold snap -10°C", "Hormuz stays closed through Q3 2026", "EUR/USD breaks 0.95". Re-run portfolio P&L under each.

### D5. Signal Backtester

Single backtest engine that takes any boolean signal series and a holding period, returns the standard performance dashboard. Lets every signal generated in Layers 1–4 be evaluated immediately.

---

## Section 7 — Phase E: Layer 6 BESS Optimisation

Net-new content. The area of European power markets with the most rapid commercial growth and the most direct alignment with the project owner's career trajectory.

### Why This Layer Matters

Equinor approved its first US battery storage projects in 2024 with Danske Commodities handling commercialisation. DC, Centrica Energy, Eku Energy, and others have built audited multi-market BESS optimisation portfolios. Statkraft is expanding flexibility trading. The Welkin Mill battery (35 MW/70 MWh, Manchester) signed by DC is one of dozens of similar deals signed in 2024-2025.

The technical literature is mature: stochastic mixed-integer optimisation co-optimising day-ahead, intraday continuous, and FCR markets with state-of-charge, inverter loss, and pay-as-bid vs pay-as-clear rules. Recent work (arXiv 2504.06932, 2506.02837) shows yearly revenue improvements of 14-58% from higher trading frequency and proper multi-market value stacking.

### Implementation

A "BESS Optimiser" page with:

- **Asset configurator.** User defines power capacity (MW), energy capacity (MWh), round-trip efficiency, max cycles per day, country/zone.
- **Multi-market revenue projection** for the configured asset:
  - Day-ahead arbitrage (charge low / discharge high across 96 MTUs).
  - Intraday continuous arbitrage (high-frequency LOB trading).
  - FCR-N (frequency containment, normal — Nordics).
  - FCR-D (frequency containment, disturbance).
  - aFRR / mFRR (automatic and manual frequency restoration reserves).
  - Capacity market payments where applicable (GB).
- **Backtest with realistic constraints.** Run the optimiser against the last 365 days of zone-specific data with state-of-charge constraints and realistic efficiency losses.
- **Value stacking visualisation** — most BESS revenue in the Nordics now comes from FCR-N + FCR-D + intraday, not day-ahead arbitrage alone.
- **Cycle / degradation overlay** — total cycles used, implied calendar life consumed, dollar cost of degradation per MWh traded.

### Reference Implementations

- Hameed & Træholt (2025) — "Optimal BESS Scheduling for Multi-Market Participation in the Nordics" — methodology for FCR-N, FCR-D, spot co-optimisation in DK, FI, NO.
- Löhndorf, Wozabal et al. (2024) — "Maximizing Battery Storage Profits via High-Frequency Intraday Trading" — order-by-order LOB methodology, German intraday.
- "Continuous Intraday Trading: An Open-Source Multi-Market Bidding Framework for Energy Storage Systems" (ACM e-Energy 2025).
- Centrica Energy public materials on TBM (Transaction-Based Model) and audited 670 MW BESS portfolio.

### Why This Lands With the Right Audience

The Hafslund / Statkraft / Danske Commodities outreach context sits inside a market where physical power trading is increasingly dominated by flexibility optimisation: hydro reservoir dispatch, BESS, demand-response, district heating CHP. A dashboard that can simulate a BESS asset on real Nordic data — with FCR-N/FCR-D mechanics correctly implemented — is a direct demonstration of the analytical capability a flex-trading desk uses daily. It is a much more concrete demonstration than another regression panel.

---

## Section 8 — UX / Visual / Branding Polish

The `app.py` design language is already strong (dark theme, layer cards, Research Pipeline section, typography). The following refinements push from "good Streamlit dashboard" to "looks like a Bloomberg terminal panel".

### Navigation
- Replace the default Streamlit page selector with a persistent custom sidebar via `st.navigation` (Streamlit ≥1.36) or `streamlit-option-menu`.
- Breadcrumbs on every page.
- URL state. Sync active tab and key controls (date range, regime filter, injection-rate multiplier) to query parameters via `st.query_params`. Means individual charts can be shared as direct links.

### Interactivity
- Cross-filter panels on the same page via `streamlit-plotly-events`.
- Hover synchronisation — broadcast a vertical rule across every chart on the page.
- Tooltips with sourcing — `(i)` icon on every KPI card with calculation, source, last-refresh time.

### Branding
- Custom domain + favicon + Open Graph image. The HF wrapper is what makes the link previews on LinkedIn look generic.
- Footer on every page with the existing landing-page footer pattern.
- Dark/light theme toggle (light variant for screenshare presentations).

### Pitch Mode
- `?mode=pitch` URL parameter that strips sidebar and chrome, leaving the chart at full width with a clean caption. This is what gets screenshotted.

---

## Section 9 — Connection to Bayes Coursework and ABG Sundal Collier

A trader interviewing the project owner should see immediately how the dashboard connects to the stated CV. The platform has the foundations but does not currently surface the connection.

| Module | Where it should be visible |
|---|---|
| Commodity Derivatives | TTF forward curve PCA, calendar spreads, gas-to-power spread option, storage real option (existing TTF backtest + D2 + C9 expansion) |
| Applied Machine Learning | LSTM with quantile loss, walk-forward validation, HMM regime classifier, FinBERT (existing Layer 4 + C10 hardening) |
| FX Trading | EUR/USD and EUR/NOK panels in Layer 3 (C5), EUR/USD as feature in JKM-TTF arbitrage |
| Market Microstructure | 15-minute MTU resolution (C2), intraday LOB analysis in BESS layer (Phase E), flow-based market coupling in C6 |
| Quantitative Methods | OLS structural model with full diagnostics (C9), cointegration testing (D1), Engle-Granger, Johansen, Granger causality, ECM |

ABG Sundal Collier connections:
- KAXCAP rebalancing methodology maps onto the cointegration/relative-value scanner in D1.
- Python data engineering pipelines map onto the async data layer, registry pattern (C3), persistent caching architecture.
- FactSet and Bloomberg fluency map onto the data-source-agnostic abstraction and Bloomberg-terminal idioms (tooltips, sourcing, last-refresh badges).

Surface this on the landing page or a dedicated "About" page that explicitly maps the dashboard to the CV.

---

## Section 10 — Outreach Context: Marius Slette and Hafslund

Marius Slette is a physical power trader at Hafslund. He featured on the *Tid er penger* podcast in October 2025 covering Norwegian power market mechanics, hydro reservoir dispatch logic, the Norgespris policy debate, and the practical day-to-day of physical power trading. The podcast is the closest thing to ground truth on how a Hafslund trader thinks. Listening to it is high-leverage preparation for any cover letter or networking conversation.

Specific themes the dashboard can speak to:
- Norwegian hydro reservoir mechanics — Layer 1 hydro panel is already strong; sharpen with explicit "water value" overlays.
- Norgespris pricing structure debate — surface the difference between system price and zonal price as a panel.
- Cross-border flow economics from NO to GB and DE — covered by C6.
- 15-minute MTU transition — physical power traders deal with this daily; the dashboard adapting is a direct signal of practitioner-grade thinking (covered by C2).
- BESS / flexibility — Hafslund's parent group is involved in flex assets; covered by Phase E.

### Other LinkedIn / Public Sources to Monitor

- **Danske Commodities** company page — intraday trading, BESS optimisation, Nordic flexibility, district heating CHP.
- **Statkraft, Equinor, Vattenfall, RWE Supply & Trading, Centrica Energy, Axpo** — corporate accounts.
- **Aurora Energy Research, Wood Mackenzie, ICIS (Ethan Tillcock UK & European Gas), Argus Media, Montel** — analytical firms; ICIS currently most cited on Hormuz fallout.
- **Energy Flux (energyflux.news)** — independent analyst commentary on TTF / LNG / Hormuz.
- **Bayes alumni in energy/commodity trading.**

### Academic and Research Sources

- EPRG (Cambridge) — European hub integration working papers.
- Oxford Institute for Energy Studies — open-access EU gas / LNG / hub pricing.
- ScienceDirect Energy Economics — convergence trading, cointegration, PCA on forward curves.
- arXiv quant-finance — pairs trading, regime models, term-structure modelling, BESS.

### Books

- *Energy Trading and Risk Management* — Iris Mack
- *Modeling and Pricing in Financial Markets for Weather Derivatives* — Benth & Saltyte-Benth
- *Stochastic Modelling of Electricity and Related Markets* — Benth, Saltyte-Benth, Koekebakker
- *Commodities and Commodity Derivatives* — Helyette Geman

---

## Section 11 — Open Issues and Bugs Catalogue

| Severity | Component | Issue | Hypothesised Cause | Fix | Status |
|---|---|---|---|---|---|
| HIGH | `app.py` cold-start data layer | Cold start ~60s, serial blocking calls (fetch_all_flows → fetch_hydro_reservoirs → ...) | No async parallelism, no persistent disk cache | Async fetching + `@st.cache_data(ttl=3600, persist="disk")` (covered by C1) | Open, **observed live** |
| HIGH | `pages/4_ML_Models.py` feature matrix load | Blocks for 30s+ on Layer 4 entry, does not share cache with Layer 1 | Separate cache key from Layer 1, eager evaluation | Share `assemble_features()` result via `st.session_state` across pages | Open, **observed live** |
| MED | `data/lng.py` GB sendout | "No sendout data available for: Great Britain" rendered as inline caption, not as a data-source banner | GIE ALSI does not return GB data | Add Elexon Insights / National Grid fallback to registry (covered by C3) | Open, **observed live** |
| RESOLVED | `pages/4_ML_Models.py` (HMM) | Confidence + regime label bugs | — | — | ✅ Fixed in earlier session |
| HIGH | `pages/4_ML_Models.py` (LSTM) | Test MAE €37.31/MWh worse than naive €13.96/MWh | Level prediction with MSE collapses to mean under high vol; Q1 2026 test window is Hormuz crisis | Switch to log-return prediction with quantile loss; walk-forward validation; "not useful in this regime" indicator | Open |
| HIGH | `data/sentiment.py` / Granger test | Persistence weak; Granger test gating may need surface fix | HF ephemeral filesystem; live writes lost on restart | Move history to HF dataset repo or external Postgres (covered by C1) | Open |
| HIGH | `data/solar.py`, `data/wind.py`, `data/power_flows.py` | ENTSO-E B16/B19/B09/B10 down 21+ days | Platform-wide ENTSO-E instability per active `entsoe-py` issues | Implement registry pattern with Energy-Charts as primary for DE, ESIOS for ES, Terna for IT, Elexon for GB (covered by C3) | Open |
| MED | All Layer 1 charts | Hourly resolution despite SDAC moving to 15-min on 30 Sep 2025 | Stale assumption | Migrate to 15-min native, allow hourly aggregate as toggle (covered by C2) | Open |
| MED | Landing page | "Fetching metadata from the HF Docker repository..." chrome | HF Spaces wrapper | Migrate to Render (covered by C1) | Open |
| MED | TTF event annotations | Hormuz cluster missing | Pre-dates the event | Add 8 sub-events from late Feb-April 2026 (covered by C4) | Open |
| LOW | `app.py` Plotly defaults | Plotly logo and full toolbar visible everywhere | Default config | Centralise theme override in `utils/plot_theme.py` | Open |

---

## Section 12 — Execution Sequencing

Ordered by dependency and impact-per-effort.

### Phase A — Complete Layer 2 Spec (Weeks 1–2)
**Source: `FUTURE_PLANS.md` Priority 1.**
A1. `models/supply_stack.py` + Supply Stack tab.
A2. `models/nordic_decomp.py` + Nordic Price Decomposition tab.
A3. `data/wind.py` A69 fetcher + `models/wind_forecast_error.py` + Wind Forecast Error tab.
A4. Update Layer 2 model overview expander to "Active" for the three.

### Phase B — Zero-New-API Quick Wins (Week 3)
**Source: `FUTURE_PLANS.md` Priority 2.**
B1. Storage-to-price regression.
B2. Hydro-price lead/lag.
B3. TTF vs seasonal norm.
B4. Granger surface in Layer 2.
B5. Interconnector utilisation (capacity constants now, full activation when B09/B10 recover).
B6. Nordic spread history.

### Phase C — Architectural Foundation (Weeks 4–6)
**Source: this brief.**
C1. Hosting migration to Render with custom domain. Move model artefacts to HF model repo. Persist FinBERT history to Render Postgres.
C2. 15-minute MTU migration in `data/spot_prices.py` and `data/prices.py`.
C3. Implement DataSourceRegistry pattern; add Energy-Charts, ESIOS, Terna, Elexon as primary providers per zone.
C4. Iran/Hormuz event cluster in `events.json`.
C5. Cross-commodity expansion (copper, aluminium, EUA, Brent, freight, urea, FX).
C6. Per-interconnector dashboard + flow analytics.
C7. LNG depth (utilisation, vessel arrivals, JKM-TTF, force-majeure tracker).
C8. Clean-spark / clean-dark spreads (extends Phase A Supply Stack).
C9. Layer 2 productionisation (storage MC narrative, full regression structural form, spike conditional probability, strategy book).
C10. Layer 4 ML hardening (LSTM log-returns + quantile, HMM feature audit + override, FinBERT topic disaggregation).

### Phase D — Strategy Book and Synthesis (Weeks 7–8)
**Source: this brief.**
D1. Cointegration scanner.
D2. PCA forward curve.
D3. Mispricing dashboard.
D4. Risk dashboard.
D5. Signal backtester.

### Phase E — BESS Optimisation Layer (Weeks 9–10)
**Source: this brief.**
E1. Asset configurator.
E2. Multi-market revenue projection.
E3. Backtest engine with SoC and degradation.
E4. Value-stacking visualisation.

### Phase F — UX and Branding (Week 11)
**Source: this brief Section 8.**
F1. Custom navigation, breadcrumbs, URL state.
F2. Cross-filter and hover-sync.
F3. Tooltips with sourcing.
F4. Pitch mode.
F5. OG image, footer with CV link, favicon.
F6. About / Bayes mapping page.

---

## Section 13 — Acceptance Criteria

The platform is ready for Hafslund / Statkraft / Danske Commodities outreach when:

- The site loads in under 3 seconds from cold for a first-time visitor.
- All charts render at 15-minute MTU resolution natively where applicable.
- Every panel has a visible source, last-refresh timestamp, and methodology tooltip.
- No panel silently fails — every fallback is labelled.
- The HMM regime classifier correctly identifies the current regime as Geopolitical Stress (validated against the Iran/Hormuz event cluster).
- The LSTM either beats naive baseline on at least one zone, or honestly displays "naive baseline not beaten in current regime".
- The Mispricing Dashboard surfaces at least 8 live signals with backtested historical hit rates.
- The Strategy Book backtests show ex-crisis Sharpe ratios of at least 0.8 on at least three strategies.
- The Cointegration Scanner produces at least 3 currently-divergent pairs at any given time, with full historical statistics.
- The BESS Optimiser produces a credible annual revenue projection for a configured Nordic BESS asset, broken down across DAM / Intraday / FCR-N / FCR-D.
- The custom domain resolves; favicon, OG image, and footer with CV link are in place.
- The connection to specific Bayes modules is visible from the landing page or About page.

---

*End of report. Use the prompt in the companion file `CLAUDE_CODE_PROMPT.md` to begin implementation.*
