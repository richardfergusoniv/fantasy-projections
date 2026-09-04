# H4 — Insufficient-history / rookie prior (research-only)

**Date:** 2026-09-04  
**Branch:** `cursor/qb-h4-insufficient-history-prior`  
**Base (H3):** `4d61c891b24e817995b82efa2f9ecaa09b067a92` on `cursor/qb-upstream-feature-allocation-59f0` (draft PR #8)  
**Model ID:** `h4_insufficient_history_prior` (isolated; H3 unchanged)  
**Verdict:** **NO-GO FOR H4**

Research-only. Does not merge to `master`. Does not change the sealed production release, active pointers, H3 artifacts, or application projection source.

---

## Predeclared decision (before final eval)

Committed first as `output/qb_h4/predeclared_decision_policy.json` (commit `a698d93`).

GO requires **all** frozen H3 gates vs sealed `model_points_end_to_end`, including:

- Latest chronological OOS (2025) overall ΔMAE within +2% of sealed MAE
- Primary-cohort paired bootstrap: **95% CI upper bound of (H4_MAE − sealed_MAE) must be strictly &lt; 0**
- Primary improves on ≥2 fit folds
- Top-12 non-inferiority (≤ +3%) on latest OOS
- Spearman drop ≤ 0.02 on latest OOS
- Established-veteran non-inferiority (≤ +2%) on latest OOS
- Zero conservation violations; availability applied once; zero non-QB changes

Do not weaken thresholds, tune on 2025, or substitute a point estimate for the CI gate.

---

## What H4 changed (narrow)

| Piece | Behavior |
|---|---|
| Experience taxonomy | `established_veteran` / `limited_history` / `rookie` / `insufficient_history` / `missing_identity` from preseason-only info |
| Rookie / thin-history priors | Role-conditioned empirical-Bayes shrink toward historically comparable QB active rates (peer seasons &lt; target) |
| Designed/scramble coverage | Portable fixture from `weekly_qb_repair_cache` PBP for **2022–2025**; **2018–2021 uncovered** (no PBP in repo — reported, not invented) |
| Null designed | Still never pocket; uncovered stays `insufficient_history` / carries-based dual-threat |
| Path | Same H3 infra: role starts → portable team reconcile → compose → unchanged ensemble weights |
| Production | No global default switch; sealed release untouched |

Limitation: college rushing / draft capital were **not** added (no leakage-safe committed source beyond `is_rookie` + depth tier + prior NFL starts).

---

## Results vs sealed and repaired H3

Δ = candidate − sealed (negative = better). 2025 = latest chronological OOS (not pristine holdout).

### Full eval universe (includes rookies H3 zeroed)

| Season | n | Sealed MAE | H3 MAE | H4 MAE | Δ vs sealed | Δ vs H3 |
|---:|---:|---:|---:|---:|---:|---:|
| 2023 | 49 | 79.95 | 78.99 | 73.82 | **−6.13** | −5.17 |
| 2024 | 48 | 75.61 | 78.47 | 75.52 | −0.09 | −2.95 |
| 2025 | 43 | 67.19 | 76.25 | 76.47 | **+9.28** | +0.22 |

### H3-comparable universe (nonzero H3; matches H3 report n / sealed MAE)

| Season | n | Sealed MAE | H3 MAE | H4 MAE | Δ vs sealed | Δ vs H3 |
|---:|---:|---:|---:|---:|---:|---:|
| 2023 | 42 | 80.97 | 72.96 | 69.04 | **−11.93** | −3.93 |
| 2024 | 43 | 71.75 | 60.07 | 61.28 | **−10.47** | +1.20 |
| 2025 | 37 | 66.82 | 67.88 | 69.50 | **+2.68** | +1.62 |

H3-comparable 2025 +2.68 exceeds the frozen +2% band (tol ≈ 1.34).

### Primary cohort (latest OOS 2025)

| | Value |
|---|---|
| n | 18 |
| Δ vs sealed | −3.40 (point improve) |
| Bootstrap 95% CI | [−23.8, **+16.2**] — **includes 0** |

### Experience cohorts (2025)

| Cohort | n | Δ vs sealed | Δ vs H3 |
|---|---:|---:|---:|
| established_veteran | 29 | −1.24 | +0.86 |
| limited_history | 8 | **+16.88** | +4.38 |
| rookie | 6 | **+49.96** | −8.45 |
| top-12 | 12 | −16.11 | −0.52 |

Rookie priors beat H3’s zeros but remain far worse than sealed. Limited-history regresses on 2025.

### Invariants

| Check | Result |
|---|---|
| Team pass / QB-rush conservation | 0 violations |
| Availability applied once | 0 double-avail |
| Reconciliation skipped | no (portable fixture; DB placeholder fail-fast) |
| Non-QB changes | 0 |
| H3 code/artifacts | unchanged |

---

## Decision

# **NO-GO FOR H4**

### Failing stage / cohorts

1. **2025 overall** ΔMAE **+9.28** vs sealed (full universe); **+2.68** on H3-comparable universe (still above +2% tol)
2. **2025 primary bootstrap CI** upper bound **+16.2** (must be &lt; 0)
3. **2025 Spearman** drop beyond tol
4. **Rookie / limited_history** on 2025 remain material failure modes for the H4 thesis

Stop. Do not weaken gates. Do not auto-start H5. An honest next hypothesis would need a leakage-safe acquisition/draft or college-rush feature source that this repo does not currently provide — or a different opportunity model for first-year QBs than peer active-rate shrinkage alone.

---

## Reproduce

```bash
# from H3 tip 4d61c89
git checkout cursor/qb-h4-insufficient-history-prior
python scripts/qb_h4_build_designed_coverage.py   # writes predeclared policy + coverage
python -m pytest tests/test_qb_h4.py tests/test_qb_h3_infra.py -q
python scripts/qb_h4_end_to_end_eval.py           # exit 2 = NO-GO FOR H4
```

## Production safety

| Item | Status |
|---|---|
| PR target | H3 branch (not `master`) |
| Draft | yes |
| `v2_baseline_20260830` / active pointer | Untouched |
| H3 results under `output/qb_h3/` | Untouched |
| App projection source | Untouched |
