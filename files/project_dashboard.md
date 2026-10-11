# Project Dashboard: European Gas and Power Platform

## Close-of-session checklist

Run these commands **in order** at the end of every session. "Synced" means all three pushes have been confirmed; none counts alone.

```bash
# 1. Push code to GitHub main
git checkout main
git push origin main

# 2. Merge main into hf-deploy and push to GitHub
git checkout hf-deploy
git merge main --no-edit
git push origin hf-deploy

# 3. Deploy to the HuggingFace Space (required: this is the live deploy)
#    HF rejects binary files that are not stored through Xet, and hf-deploy carries
#    the README screenshots. Commit the hf-deploy tree without its binary files as one
#    new commit on top of the Space's current head. No checkout is needed.
git fetch origin hf-deploy
git fetch hf main                       # FETCH_HEAD is now the Space's head
HF_HEAD=$(git rev-parse FETCH_HEAD)
SRC=$(git rev-parse origin/hf-deploy)
IDX=$(mktemp)
GIT_INDEX_FILE=$IDX git read-tree "$SRC"
git diff --numstat 4b825dc642cb6eb9a060e54bf8d69288fbee4904 "$SRC" \
  | awk '$1 == "-" {print $3}' \
  | while read -r f; do GIT_INDEX_FILE=$IDX git rm --cached -q -- "$f"; done
TREE=$(GIT_INDEX_FILE=$IDX git write-tree); rm -f "$IDX"
C=$(git commit-tree "$TREE" -p "$HF_HEAD" -m "Deploy <description> to HuggingFace Space (squashed, no binaries)")
git branch "hf-squash-$(date +%Y%m%d%H%M)" "$C"
git push hf "${C}:refs/heads/main"      # braces matter in zsh: $C:r is a modifier
```

**Verify the push landed:**
```bash
git ls-remote hf refs/heads/main
# should print the commit you just created; the Space rebuilds within a few minutes
```

---

## Why the squash is needed (HF Xet policy)

HuggingFace Spaces require binary files to go through their Xet storage system, and
the pre-receive hook rejects a push if any commit in the range contains a binary file
that is not, even if a later commit deletes it. The `hf-deploy` history contains such
files: model weights (commit `82f9a59`) and the PNG screenshots the README shows on
GitHub.

The deploy therefore pushes one new commit whose tree is the `hf-deploy` tree minus
every binary file, on top of the Space's previous head, so no pushed commit ever
contains a binary. The Space needs none of them: the screenshots are for GitHub, and
the `models/weights/*.pt` and `*.joblib` files are in `.gitignore` on `hf-deploy`.

Layer 4 (LSTM/HMM) shows "Ready to train" on the live Space because no weights are
deployed. Layers 1 to 3 are unaffected.

---

## Branch map

| Branch | Remote | Purpose |
|--------|--------|---------|
| `main` | `origin/main` (GitHub) | Primary development; a Streamlit Community Cloud app also builds from it (sign-in required, not linked publicly) |
| `hf-deploy` | `origin/hf-deploy` (GitHub) | Mirror of `main` plus HF-only files: warm-start caches and HF config, no model weights |
| `hf-squash-*` | `hf/main` (HuggingFace) | Local record of each HF deploy commit |

---

## Cache refresh (monthly)

`hf-deploy` ships `data/generation_cache_v2.csv` (German coal generation since 2019) so a
fresh Space only fetches the days since the cache ends instead of the whole history. The
file is ignored on `main`, where the local app keeps it up to date. Commit it from a
separate worktree: checking out `main` again in the main checkout would delete the ignored
local copy, because `hf-deploy` tracks it.

```bash
git worktree add ../hf-deploy-wt hf-deploy
cp data/generation_cache_v2.csv ../hf-deploy-wt/data/
git -C ../hf-deploy-wt add data/generation_cache_v2.csv
git -C ../hf-deploy-wt commit -m "Refresh the bundled coal generation cache"
git -C ../hf-deploy-wt push origin hf-deploy
git worktree remove ../hf-deploy-wt
# Then run step 3 of the close-of-session checklist above
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
- Streamlit Community Cloud: builds from `origin/main` (sign-in required; the public link is the HF Space)
- CI: `.github/workflows/tests.yml` runs the offline tests on every push to `main`

---

## Shipped features

Everything confirmed live on HuggingFace Space and `origin/main`. Rows labelled **Q4** and **v2** were merged into `origin/main` from branch `feat/platform-v2` and deployed to the HF Space on 2026-10-11, and the **v2.1** rows followed the same day. All eight pages, including D4 (Page 6) and Phase E (Page 7), were opened on the Space on 2026-10-11 and rendered without errors.

| Phase / Item | Layer | Description |
|---|---|---|
| Phase 0 | All pages | Project scaffold: Streamlit multi-page setup, dual-branch deploy (main / hf-deploy), .env secrets, dark theme |
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
| v2.1 · Coal generation | Layer 1 | German hard coal and lignite are energy from hourly mean power: the 15-minute ENTSO-E A75 readings had been summed as if they were hourly MWh, which overstated every coal figure fourfold. The disk cache moved to `generation_cache_v2.csv` (version 1 is no longer read), fresh rows replace cached ones in the seven-day overlap, days still being published are left out, and `hf-deploy` ships the cache so the Live Monitor renders on a fresh Space without fetching the history since 2019 first |
| v2.1 · GIE resilience | Layer 1, 2, 4, 5 | AGSI+ and ALSI requests share `data/gie.py`: HTTP 429 (GIE allows 60 requests per minute per address, and the Space shares its egress address with other tenants) and gateway errors are retried with the server's Retry-After, a history that fails part-way raises instead of returning a partial series, and a failed fetch is remembered for five minutes instead of an hour. The Live Monitor no longer says "AGSI key required" when the key is set but GIE did not answer, and the NW EU LNG send-out card shows n/a, naming the missing country, instead of a partial total |

---

## Deferred work (post first-wave outreach)

| Item | Description |
|------|-------------|
| Phase C-arch | **Render migration: deferred indefinitely.** Artifacts in repo (`Dockerfile`, `render.yaml`) for future self-host or paid migration. Current production: HF Spaces. 15-min MTU granularity and DataSourceRegistry abstraction also deferred. |
| D5 | Scope TBD |
