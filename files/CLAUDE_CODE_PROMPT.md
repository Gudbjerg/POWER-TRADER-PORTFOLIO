# Claude Code Continuation Prompt — Power Trader Portfolio

Copy everything below the divider and paste as the opening message of the next Claude Code session.

---

## CONTEXT

The European Power and Gas Analysis Platform is the multi-layer energy market intelligence dashboard at `power-trader-portfolio.streamlit.app` and `huggingface.co/spaces/TobGud/power-trader-portfolio`. The previous Claude Code session left the project in a stable state with three Layer 2 spec models pending and a deeper architectural roadmap drafted in a new external review.

The next development cycle has two synchronised inputs:

1. **`FUTURE_PLANS.md`** — drafted by Claude Code in the previous session. Lists the three pending Layer 2 spec models (Nordic Price Decomposition, Supply Stack, Wind Forecast Error Tracker) plus six zero-new-API quick wins and longer-term ideas.

2. **`feedback_report.md`** (newly added) — a 13-section integrated roadmap covering hosting migration, 15-minute MTU adoption, ENTSO-E fallback registry, Iran/Hormuz event cluster, BESS optimisation layer, cointegration / PCA / mispricing dashboard, Layer 4 ML hardening, and execution sequencing across Phases A–F. Section 0.5 contains direct live-walkthrough findings — read this first as it corrects several earlier screenshot-based assumptions.

The two documents are harmonised: `feedback_report.md` accepts the `FUTURE_PLANS.md` Priority 1 and 2 work as Phases A and B, then layers Phases C–F on top. Both should be treated as authoritative.

## PRE-FLIGHT — DO THIS FIRST

Before writing any code, read in full and confirm understanding of:

1. `project_dashboard.md` — branch architecture, deploy commands, sync workflow, hard rules.
2. `POWER TRADER PORTFOLIO.md` — original project specification.
3. `FUTURE_PLANS.md` — the previous session's roadmap.
4. `feedback_report.md` — the integrated roadmap.
5. `1st_prompt.md` and `extra_context.md` — earlier session context.
6. `app.py`, `pages/2_Quant_Analysis.py`, `pages/4_ML_Models.py`, `models/feature_assembly.py`, `data/spot_prices.py`, `data/prices.py`, `data/wind.py`, `data/solar.py`, `data/sentiment.py`, `data/events.json`, `config/settings.py`.

After reading, produce a structured response with:
- Confirmation of which authoritative files have been read.
- One paragraph summarising current state (Layer 1, 2, 3, 4 status).
- One paragraph identifying any conflicts, gaps, or stale assumptions in the documents.
- A proposed execution order for this session, broken down by deliverable.
- Any clarifying questions before starting.

Do not begin coding until I confirm.

## BRANCH AND DEPLOY HARD RULES

Per `project_dashboard.md`:
- Never run `git checkout main -- requirements.txt` on `hf-deploy`.
- Never auto-commit or auto-push without explicit instruction.
- Cache seeding files (`features_cache.csv`, `.sentiment_history.csv`) live on `hf-deploy` only.
- All Python source changes flow `main` → `hf-deploy` via the documented sync workflow.
- The `hf-deploy` branch carries the torch/transformers requirements; `main` does not.

If any planned change violates a rule above, surface the conflict and ask before proceeding.

## SCOPE FOR THIS SESSION

Phase A from `feedback_report.md`. Specifically the three pending Layer 2 spec models in build order:

**A1. Supply Stack (`models/supply_stack.py`)**
- New tab in `pages/2_Quant_Analysis.py`.
- German merit order chart per the spec in `FUTURE_PLANS.md` 1B.
- Dynamic gas marginal cost from existing TTF fetcher; static capacities for other fuels with disclaimer in UI.
- EUA carbon price: try yfinance `CO2.L` or `EUAc1=F`, fallback to hardcoded ~65 EUR/t.
- Output: stacked bar chart (capacity x cost), demand line, marginal-fuel annotation, scarcity-premium KPI vs NL actual.
- Foundation for the clean-spark / clean-dark spreads that follow in Phase C8.

**A2. Nordic Price Decomposition (`models/nordic_decomp.py`)**
- New tab in `pages/2_Quant_Analysis.py`.
- Multivariate OLS via `statsmodels.OLS`: `NO2 ~ hydro_pct + de_wind_gwh + nl + ttf`.
- Standardise regressors before fitting.
- Rolling 90-day window.
- Graceful degradation: 4-factor when full data, 3-factor when wind missing, 2-factor when wind+hydro missing — banner identifies active mode.
- Output: rolling beta time series (one line per factor), dominant-driver badge, today's contribution stacked bar.
- Reuses the existing `assemble_features()` call at the top of `pages/2_Quant_Analysis.py`.
- Connects forward to Phase D1 cointegration scanner.

**A3. Wind Forecast Error Tracker (`models/wind_forecast_error.py` + extension to `data/wind.py`)**
- New tab in `pages/2_Quant_Analysis.py`.
- Add `fetch_wind_forecast()` to `data/wind.py` using `entsoe-py` `query_wind_and_solar_forecast()` (A69 document type — confirm whether this server is also affected by the wider B09/B10/B16/B19 outage; if it is, surface a banner and degrade gracefully).
- Compute daily forecast error (GWh and %) per country: DE, DK1, NO, GB.
- Rolling 7-day RMSE per country.
- Correlation with price volatility proxy (|day-over-day NO2 price change|) — scatter plot with rolling 30-day correlation annotation.
- Output: daily error bar chart, RMSE line chart, scatter with correlation annotation.

For each model:
- Update the model overview expander in `pages/2_Quant_Analysis.py` from "Not built" to "Active".
- Update `README.md` after implementation, per the standing instruction in `FUTURE_PLANS.md` ("remember to update readme after implementing each feature").
- Respect the `width="stretch"` rule from `project_dashboard.md`.
- Cache fetchers with `@st.cache_data(ttl=3600, persist="disk")` per existing pattern.

## VOICE AND STYLE

- Use formal, impersonal tone in any prose written into the codebase or UI.
- No oversimplification — production-quality logic with full edge-case handling.
- Quantify outputs commercially where relevant.
- Match the existing `app.py` design language (dark theme, Bloomberg-style typography, the existing CSS in `utils/helpers.py`).

## OUT OF SCOPE FOR THIS SESSION

The following items in `feedback_report.md` are not for this session — note them as deferred and do not begin work on them:

- Phase B (zero-new-API quick wins) — next session.
- Phase C (hosting migration to Render, 15-min MTU, fallback registry, Hormuz event cluster, etc.) — multi-session, requires explicit go-ahead.
- Phase D (cointegration scanner, PCA, mispricing, risk dashboard, signal backtester) — after Phase C.
- Phase E (BESS optimisation layer) — after Phase D.
- Phase F (UX polish, custom domain, OG image, etc.) — final phase.

## DELIVERABLES FOR THIS SESSION

End-of-session checklist:
1. `models/supply_stack.py` — committed, tested locally, integrated as a new tab.
2. `models/nordic_decomp.py` — committed, tested locally, integrated as a new tab.
3. `models/wind_forecast_error.py` + `data/wind.py` extension — committed, tested locally, integrated as a new tab.
4. `pages/2_Quant_Analysis.py` — updated tabs list, model overview expander updated.
5. `README.md` — updated to reflect the three new models.
6. Sync to `hf-deploy` per the documented workflow.
7. Status summary at end of session: what was built, what was tested, what is pending for next session.

Begin with the pre-flight read and the proposed execution order. Wait for confirmation before coding.
