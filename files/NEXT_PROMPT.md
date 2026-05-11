# Next Claude Code Prompt

Paste everything below the divider into Claude Code as the opening message of the next session.

---

Last session shipped Phase 0A, 0B, A1, A2, A3 and the README update. Verified live by visual walkthrough on 4 May 2026. **Three new findings, two bugs, and a list of Phase B ideas with new research backing.** Read the additions file `files/NEXT_PHASE_NOTES.md` first along with the existing context files (`project_dashboard.md`, `FUTURE_PLANS.md`, `power_trader_feedback_report.md`, `CLAUDE_CODE_PROMPT.md`).

## Verified Live (4 May 2026)

- ✅ Phase 0A disk cache fix: landing page loads in ~10s (was ~70s) **on warm container**. ⚠️ Layer 2 Quant Analysis still ~60s on first cold load (cache empty). User-reported: "still takes ages to load" — see Bug 4 below.
- ✅ Phase 0B Hormuz cluster: all 8 events visible on TTF chart in Layer 3.
- ✅ Phase A1 Supply Stack: tab live, demand slider working visually, dynamic gas SRMC at €109/MWh, EUA at €70/tCO2. ⚠️ Slider drag triggers full data re-fetch — see Bug 3.
- ✅ Phase A2 Nordic Decomposition: tab live, R²=0.79, 4-factor model active, dominant driver = Continental price (β=23.83).
- ✅ Phase A3 Wind Forecast Error: A69 forecast loaded for DE/DK1/NO, Germany mean |err| = 108.1 GWh.
- ⚠️ LSTM (Layer 4): predictions still hug the mean line — user-confirmed "model 1 still looks like an avg price". Methodology fix is Phase C10; near-term mitigation is an honesty banner — see Bug 5.

## Bugs / Regressions to Fix First (highest priority)

### Bug 1 — Supply Stack chart missing zero-cost fuels at start of merit order
The chart only shows the thermal plants (lignite, biomass, hard coal, gas, oil). The renewables and hydro that should appear as ~€0/MWh bars from 0 GW to ~143 GW (66 wind + 73 solar + 4.5 hydro) are not rendered. The bars likely have height=0 in Plotly which collapses them visually. Fix by either:
- Drawing zero-cost fuels as a thin coloured band along the x-axis (height ~2 px) so they remain visible, OR
- Switching from `go.Bar` to a stepped-line approach (scatter with shape="hv") which renders zero-cost steps correctly.

The chart is the most visually striking component and currently misleads viewers into thinking Germany has no renewables.

### Bug 2 — "Scarcity premium" KPI label is wrong sign
The Supply Stack KPI card "NL VS IMPLIED" shows "€127.6/MWh, +127.6 EUR/MWh scarcity premium" when implied is €0 (wind-marginal) and NL is €127.6. This is the **opposite** of scarcity — it's a *renewables-surplus disconnect*. The label should:
- Be neutral: "NL minus implied" or "Implied gap".
- Have a sign-aware interpretation block: positive (NL > implied) = power expensive vs fundamentals (could be import-driven, demand-driven, or model under-counting); negative (NL < implied) = power cheap vs fundamentals.
- Add a short commentary string that explains which side the gap sits on and why.

### Bug 3 — Quant Analysis sliders trigger full data re-fetch on every drag (CRITICAL UX)
User report: "things in the quant layer doesn't work when dragging the scaler". Confirmed pattern: Streamlit re-runs the page on every slider value change. If the feature matrix and all data fetchers run inside the page body (not inside `@st.cache_data` wrappers correctly keyed), every drag of the storage MC injection-rate slider, the supply stack demand slider, or the TTF backtest storage cost slider triggers a 60-second re-fetch.

Fix:
- Audit every slider-driven panel. Verify the slider value is NOT a parameter of any cached function call. Cached calls should be parameterised on date / source only, not on user inputs.
- Move the heavy computation (feature matrix assembly, MC bootstrap, OLS fit) into `@st.cache_data` calls keyed only on data inputs.
- Slider value should be applied as a final downstream transform (filter, lookup, pure pandas operation) on already-cached data.
- Test by dragging each slider and confirming the response time is sub-second.

This is likely THE reason the platform feels slow even after the disk cache fix landed. The cache works for first-page-load, but then every interaction destroys the gain.

### Bug 4 — Cold start STILL slow on Layer 2/4 despite Phase 0A
User confirmed: "still takes ages to load". Phase 0A `persist="disk"` only helps on the SECOND visit of a SAME container. HuggingFace ephemeral containers reset disk on restart, so the very first visitor after any restart still hits 60+ seconds. Root cause is two-layered:
1. Disk cache empty after every container restart (HF infrastructure choice).
2. The page body itself loads serially regardless of cache state.

Fix sequence:
1. **Pre-seed disk cache via committed parquet/CSV files.** Already partly done with `features_cache.csv` but extend to: `gas_storage_cache.parquet`, `lng_terminals_cache.parquet`, `hydro_cache.parquet`, `wind_cache.parquet`, `solar_cache.parquet`. On first load: read from committed cache (instant), then fire a background refresh that updates `@st.cache_data` for the next visitor. The committed cache is acceptably stale (~24h).
2. **Lazy-load per Layer 2 tab.** Currently the page top loads the full feature matrix for ALL tabs. Move the loader inside each tab function so only the active tab's data resolves. Storage MC tab does NOT need wind forecasts.
3. **Add a sidebar "Last refreshed" + manual refresh button.** Tells the user what's stale and gives them control rather than waiting on a hidden background job.

### Bug 5 — LSTM still hugs the mean (visual confirmation in current data)
User confirmed: "model 1 still looks like an avg price because the prices are so volatile". The MAE €37.31 vs naive €13.96 is now backed by the visual: predictions are a flat line through the actuals. The methodology is in `feedback_report.md` Section 7 — switch to log-return prediction, quantile (pinball) loss, walk-forward validation, add wind/solar forecast features (Phase A3 wind forecast tracker now provides A69 inputs that the LSTM should consume), regime-aware suppression banner ("LSTM is not informative under Geopolitical Stress regime").

This is bigger than a bug fix — it's a Phase C10 work item. **Do not attempt during this session unless Phase B is complete and there's time left.** Just ensure the panel surfaces an honest "naive baseline beats this model in current regime" banner so a viewer doesn't draw incorrect conclusions.

## Build Order for This Session — Reordered

The cold-start and slider-rerun bugs are blocking. They go first.

1. **Bug 3 fix — slider re-fetch audit.** Highest priority. Audit Storage MC, Supply Stack, TTF Seasonal Strategy, any other slider-driven panel. Move all heavy work into `@st.cache_data` calls keyed only on data inputs. Slider value applied as final downstream transform. Test by dragging each slider and confirming sub-second response. ~60 min.
2. **Bug 4 fix — pre-seeded disk cache + lazy-load tabs.** Commit cached parquet/CSV for the 5 slowest fetchers. Move feature matrix and per-tab data loaders inside tab functions. Add sidebar "Last refreshed" indicator + manual refresh button. ~90 min.
3. **Bug 1 + Bug 2 — Supply Stack chart + scarcity-premium label.** ~45 min combined.
4. **Bug 5 — LSTM regime-aware honesty banner.** Don't fix the model in this session; just add a banner: "LSTM MAE €37.31/MWh underperforms naive baseline (€13.96/MWh) in current Geopolitical Stress regime — predictions revert to mean and should not be used for trading decisions." Future session will switch to log-returns + quantile loss. ~20 min.

**Then if time remains:** Phase B from `FUTURE_PLANS.md`, in this order:

5. **B6 Nordic spread history** — easiest win, all data already in feature matrix. Add to Layer 1 Prices tab. ~30 min.
6. **B5 Interconnector utilisation** — capacity constants in `config/settings.py` + utilisation % in flows panel. ~45 min.
7. **B4 Granger surface in Layer 2** — surface the existing Layer 4 Granger test result as a Layer 2 quant signal. ~20 min.
8. **B1 Storage-to-price regression** — scatter EU storage % vs TTF, OLS fit, residual = supply-risk premium. New 8th Layer 2 tab. ~60 min.
9. **B2 Hydro-price lead/lag** — cross-correlation of hydro_pct and NO2 at lags 0–21 days. ~60 min.
10. **B3 TTF vs seasonal norm tracker** — TTF current vs historical seasonal range. Pair with the existing storage chart on Layer 1. ~30 min.

If everything in Phase B finishes, move to NEW1 Copper-power link panel (see `NEXT_PHASE_NOTES.md`).

## Hard Rules (unchanged)

- Dual-branch architecture: `main` (Streamlit Cloud) + `hf-deploy` (HuggingFace).
- Never `git checkout main -- requirements.txt` on `hf-deploy`.
- Phase commit messages with co-author tag.
- Update README after each feature.
- Sync to `hf-deploy` and push to HF before session end.
- Verifiable evidence required for performance fixes (timing banners, before/after measurements).

## Pre-flight

Before coding:
1. Read `NEXT_PHASE_NOTES.md` in full.
2. Confirm understanding of all 5 bugs and the cold-start architecture choice.
3. **Slider audit first.** Open `pages/2_Quant_Analysis.py` and identify every `st.slider` call. For each, trace the variables that depend on the slider value down to any `@st.cache_data`-decorated function. Report which sliders trigger which fetchers. This audit is the diagnostic for Bug 3 and must be completed before any fix is attempted.
4. Propose execution order — confirm with me before starting.
5. After Bug 3 is fixed, demonstrate it: paste a "before vs after" timing comparison (drag the Storage MC injection-rate slider 5 times — total time should drop from ~5×60s=5min to ~5×0.5s=2.5s).

Wait for confirmation before writing code.
