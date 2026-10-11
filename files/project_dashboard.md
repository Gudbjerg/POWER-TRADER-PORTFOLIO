# Project Dashboard — European Power & Gas Platform

## Close-of-session checklist

Run these commands **in order** at the end of every session. "Synced" means ALL three pushes have been confirmed — none alone counts.

```bash
# 1. Push code to GitHub main
git checkout main
git push origin main

# 2. Merge main into hf-deploy and push to GitHub
git checkout hf-deploy
git merge main --no-edit
git push origin hf-deploy

# 3. Push to HuggingFace Space (REQUIRED — this is the live deploy)
#    hf-deploy contains binary files excluded by HF's Xet policy, so we use
#    a squash branch that omits them. Run this every time there are new commits.
git checkout -b hf-squash-$(date +%Y%m%d) $(git ls-remote hf HEAD | cut -f1)
git merge --squash hf-deploy
git commit -m "Deploy <phase/description> to HuggingFace Space (squashed)"
git push hf HEAD:main
git checkout hf-deploy
# Optional cleanup:
git branch -d hf-squash-$(date +%Y%m%d) 2>/dev/null || true
```

**Verify the push landed:**
```bash
git ls-remote hf
# HEAD should match the squash commit you just created
```

---

## Why the squash is needed (HF Xet policy)

HuggingFace Spaces now requires binary files (`.pt`, `.joblib`) to go through their
Xet storage system. The `hf-deploy` branch history contains commit `82f9a59` which
adds model weights as Git LFS objects. HF's pre-receive hook scans every commit in
the push range and rejects any that contain LFS-tracked binaries, **even if a later
commit deletes them**.

The squash approach creates a single new commit whose net tree matches `hf-deploy`
tip but never passes through an intermediate state where the binaries are present.
The `.gitattributes` on `hf-deploy` already removes `*.pt`/`*.joblib` LFS tracking;
the `models/weights/*.pt` and `*.joblib` files are in `.gitignore` on that branch.

Layer 4 (LSTM/HMM) will show "Ready to train" on the live Space since no weights
are deployed. Layers 1–3 are fully unaffected.

---

## Branch map

| Branch | Remote | Purpose |
|--------|--------|---------|
| `main` | `origin/main` (GitHub) | Primary dev, Streamlit Community Cloud deploy |
| `hf-deploy` | `origin/hf-deploy` (GitHub) | Mirror of main + HF-specific config, no binary weights |
| `hf-squash-*` | `hf/main` (HuggingFace) | Ephemeral squash branch for each HF deploy |

---

## Cache refresh (monthly)

```bash
# Push updated disk cache files to hf-deploy
git checkout hf-deploy
git add data/generation_cache.csv  # or any other accumulated cache files
git commit -m "Refresh disk cache for HF cold-start"
# Then run the close-of-session checklist above
```

---

## HuggingFace Space

- URL: https://huggingface.co/spaces/TobGud/power-trader-portfolio
- Build: auto-triggers on push to `hf/main`; takes 2–5 min
- Secrets (set in Space Settings → Variables and secrets):
  - `AGSI_API_KEY`
  - `ENTSOE_API_KEY`

---

## GitHub repository

- URL: https://github.com/Gudbjerg/POWER-TRADER-PORTFOLIO
- Streamlit Community Cloud: auto-deploys from `origin/main`

---

## Outreach status

**Outreach-ready as of 2026-05-19.**

Lead link: https://huggingface.co/spaces/TobGud/power-trader-portfolio

Platform at outreach cut: 5 pages · 14 Layer 2 tabs · 4 Layer 3 tabs · Mispricing Dashboard (8 signals) · HF commit `0b9a1f6`

---

## Shipped features

Everything confirmed live on HuggingFace Space and `origin/main`, with these exceptions: rows labelled **Q4** and **v2** were merged into `origin/main` on 2026-10-11 from branch `feat/platform-v2` (the HF Space follows with the next close-of-session deploy), and the D4 and Phase E rows were moved here from the deferred table because both pages are on `origin/main` (their live status on the Space was not re-checked).

| Phase / Item | Layer | Description |
|---|---|---|
| Phase 0 | — | Project scaffold: Streamlit multi-page setup, dual-branch deploy (main / hf-deploy), .env secrets, dark theme |
| Phase A1 | Layer 1 | Live Monitor: EU/DE gas storage, TTF gas price + spike detector, LNG terminal sendout, Nord Pool day-ahead prices, ENTSO-E cross-border flows, solar output, Norwegian hydro reservoir levels |
| Phase A2 | Layer 2 | First quant tabs: Monte Carlo price simulation, Gas-Power OLS regression (TTF → DE baseload), spike detector, seasonal backtest |
| Phase A3 | Layer 2 | Additional tabs: Supply Stack model, Nordic hydro decomposition, wind forecast error tracker, Granger causality TTF↔Brent |
| Phase B | Layer 2 / Layer 3 | Storage-Price OLS, Hydro Lead/Lag (ENTSO-E + AGSI), TTF Seasonal Normalisation, NO2/NL Cointegration; Layer 3: Geopolitical Overlay, EU Gas Supply Mix, Cross-Commodity Spillover (gas→agriculture chain) |
| Phase C | Layer 1 / Layer 2 | Live Monitor polish, Mispricing Dashboard scorecard foundation (scorecard engine, 8 signals wired to live feeds) |
| Phase D1 | Layer 2 | Multi-Pair Cointegration Scanner (Tab 13): Engle-Granger on 6 country pairs, spread z-score signals, export/import regime context |
| Phase D2 | Layer 2 | Forward Curve PCA (Tab 14): synthetic TTF M+1–M+18 panel (storage-carry model), Litterman–Scheinkman level/slope/curvature decomposition, PC2/PC3 trade signals |
| Phase D3 | Layer 5 | Mispricing Dashboard (full page, 8 signals): TTF risk premium, seasonal percentile, storage gap, gas-power spread, flow capacity, carbon signal, hydro reservoir, live load A65 |
| Phase F | All pages | Presentation polish: landing page hero copy, ⚡ favicon across all 6 pages, debug sidebar sections removed, README outreach rewrite with HF live link |
| NEW5 | Layer 3 | 7×7 Cross-Commodity Correlation Grid (Tab 4): 90-day Pearson heatmap, lead-lag table (top 15 pairs, ±10d sweep), KPI cards |
| Norwegian zonal | Layer 1 | "Norgespris debate" sub-panel: NO1/NO2/NO3/NO4/NO5 vs system price (SYS) day-ahead, zone spread KPIs |
| NEW1 | Layer 3 | Copper–power grid investment linkage panel: LME copper vs German baseload proxy, rolling 90-day correlation, 6-month copper change as forward indicator |
| NEW2 | Layer 3 | Aluminium smelter stress indicator: power-cost-as-%-of-smelter-revenue time series, NORMAL / ELEVATED / CRITICAL gauge |
| D4 | Page 6 | Risk Dashboard: 6-signal backtestable portfolio, 1,000-path bootstrap 95%/99% 1-day VaR and Expected Shortfall, 4 named stress scenarios, 30-day rolling signal correlation matrix (commit `8b0fda1`; previously listed as deferred) |
| Phase E | Page 7 | BESS Optimiser: DAM arbitrage plus FCR-N/FCR-D revenue stack, NPV / IRR / payback, ±20% sensitivity table (commit `bf80c8e`; previously listed as deferred) |
| Q4 · 3A | Layer 1 | FI minus IT-North day-ahead spread (Prices tab): KPIs for latest spread, 30-day average and z-score vs 1-year; two-panel 1-year chart; computed commentary; methodology expander. IT-North from energy-charts.info (SMARD, CC BY 4.0) with ENTSO-E A44 fallback; FI from Nord Pool, extended with ENTSO-E A44 when a key is set (`data/italy_prices.py`, `models/fi_it_spread.py`) |
| Q4 · 3B | Layer 3 / Layer 1 | TTF vs JKM tab (Layer 3): JKM=F converted from USD/MMBtu to EUR/MWh with EURUSD=X, spread TTF minus JKM, rolling 1-year percentile, KPI row, three-panel chart, interpretation and data caveats; compact TTF minus JKM KPI card on the Layer 1 LNG Terminals tab (`models/lng_arbitrage.py`, `data/lng_prices.py`) |
| Q4 · 3D | Layer 4 | Realised-volatility overlay in the HMM section: rolling 30-trading-day NO2 volatility on a secondary axis over the regime timeline, per-regime mean/median volatility table, share of transitions within ±5 trading days of a volatility spike against a chance baseline with a circular-shift p-value; HMM inference now degrades cleanly when trained features are missing (`models/regime_volatility.py`) |
| Q4 · D2_alt | Layer 2 | Rolling cointegration stability in the Cointegration Scanner tab: Engle-Granger (scanner settings) on rolling 180-observation windows every 21 trading days, pairs x window-end p-value heatmap with a colour break at 0.05, per-pair stability table (share of cointegrated windows, current streak, last break), Stability column in the ranking table and stability context in the signal commentary; also fixes the scanner and NO2/NL tab OU estimates, which indexed numpy statsmodels params with `.iloc` (`models/coint_stability.py`) |
| Q4 · 3E | Layer 3 | Event timeline extended from 19 to 28 entries for 20 April to 10 October 2026 (Hormuz MoU, re-closure and attacks, Qatari LNG carrier strike, EU short-term Russian LNG ban, June heatwave, US LNG shift to Asia, Commission demand-curb letter); every new entry carries a `source` URL from a news agency, EU body or major outlet; no verifiable 2026 Norwegian outage was found for the window, so none was added |
| v2 · Interface | All pages | `app.py` router with grouped top navigation (`st.navigation`), page modules in `views/` with fixed URL paths, design system in `utils/theme.py` (tokens, global stylesheet, page header, KPI card, callout, chip, ticker strip, layer card, footer, Plotly template `ptp`), redesigned overview page with a delayed market snapshot strip and data-source status |
| v2 · Data fixes | Layers 1 to 5 | ENTSO-E clients time out after 30 s instead of hanging; DE-LU bidding zone for German day-ahead prices; no disk-persisted caches for market data; ALSI send-out labelled GWh/d; season-aware storage commentary and refill pace against the 5-year norm; NaN-safe ordinals and EUA handling in the Mispricing Dashboard |
| v2 · Energy Globe | Page 8 | CesiumJS 3D globe embedded in the platform (`views/8_Energy_Globe.py`): Norwegian shelf facilities and export pipelines (Sodir, NLOD), 38 European LNG import terminals (GEM, CC BY 4.0) with live GIE ALSI send-out, utilisation and tank stocks per gas day, 10 maritime chokepoints with EIA flows; curated views linked from the overview, Layer 1, Layer 3 and the Risk Dashboard (`data/energy_globe.py`, `components/energy_globe.py`, `data/globe/`) |

---

## Deferred work (post first-wave outreach)

| Item | Description |
|------|-------------|
| Phase C-arch | **Render migration: deferred indefinitely** — artifacts in repo (`Dockerfile`, `render.yaml`) for future self-host or paid migration. Current production: HF Spaces. 15-min MTU granularity and DataSourceRegistry abstraction also deferred. |
| D5 | Scope TBD |
