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

### Remaining deferred work (post first-wave outreach)

| Item | Description |
|------|-------------|
| Phase C-arch | Render migration, 15-min MTU granularity, DataSourceRegistry abstraction |
| D2_alt | Rolling cointegration stability heatmap — which pairs consistently cointegrated vs episodic; feeds D3 confidence scores |
| Phase E (BESS) | Battery storage arbitrage optimiser — multi-session build, high positioning value for storage-focused roles |
