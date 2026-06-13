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

Everything confirmed live on HuggingFace Space and `origin/main`.

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

---

## Deferred work (post first-wave outreach)

| Item | Description |
|------|-------------|
| Phase C-arch | **Render migration: deferred indefinitely** — artifacts in repo (`Dockerfile`, `render.yaml`) for future self-host or paid migration. Current production: HF Spaces. 15-min MTU granularity and DataSourceRegistry abstraction also deferred. |
| D2_alt | Rolling cointegration stability heatmap — which pairs consistently cointegrated vs episodic; feeds D3 confidence scores |
| D4 | Scope TBD |
| D5 | Scope TBD |
| Phase E (BESS) | Battery storage arbitrage optimiser — multi-session build, high positioning value for storage-focused roles |
