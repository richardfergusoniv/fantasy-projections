# Phase 1 — Sealed-Baseline Verification (Frozen Candidate)

**Date:** 2026-09-03  
**Branch:** `cursor/qb-upstream-feature-allocation-59f0` (draft PR #8)  
**Configuration:** frozen (no threshold / archetype / pooling / allocation retune)  
**Verdict:** **NO-GO** for production candidate / manual promotion

## Comparator confirmation

| Question | Answer |
|---|---|
| What did the prior “GO” use? | **A — injury-diluted intermediate carry-forward** (`baseline_conflated` in `predict_player`) |
| Was that the sealed final? | **No** |
| What does this bakeoff use? | **B — sealed historical final fantasy points** = `fantasy_evaluation_*.csv::model_points_end_to_end` (end-to-end model stack after the historical pipeline). For 2024–2025, `incumbent_pred` from accuracy-first is reported as a supplement, not used to retune gates. |
| Compared quantity | **Final season fantasy points** and **PPG / points-per-active**, not only intermediate carries/attempts |

Declaration artifact: `output/qb_sealed_baseline_bakeoff/comparator_declaration.json`

## Out-of-sample cutoffs (enforced)

| Prediction season | Max train season allowed | Enforced |
|---:|---:|---|
| 2023 | 2022 | yes (`oos_max_train_season < season` assert) |
| 2024 | 2023 | yes |
| 2025 | 2024 | yes |

No refit or retune of the frozen configuration using 2025 after the prior holdout report.

## Frozen gates (unchanged)

Same `GateThresholds` as the active-archetype experiment (`overall` +2% non-inferiority, primary cohort improve on ≥2 fit folds, holdout primary improve, bootstrap CI upper < 0, top-12 +3% tol, Spearman drop ≤ 0.02). **Not edited after seeing sealed results.**

## Paired results vs sealed final (`model_points_end_to_end`)

Δ MAE = candidate − sealed (negative = candidate better).

### 2023 (fit)

| Cohort | n | Sealed pts MAE | Cand pts MAE | Δ MAE | Spearman sealed→cand |
|---|---:|---:|---:|---:|---|
| Overall | 42 | 80.97 | 86.08 | **+5.11** | 0.534→0.560 |
| Top-12 | 12 | 76.22 | 74.22 | −1.99 | 0.399→0.329 |
| Dual-threat | — | — | — | — | (none classified in fold) |
| Returning-injury | 11 | 85.28 | 92.20 | +6.92 | 0.409→0.355 |
| Pocket | 32 | 81.53 | 86.20 | +4.68 | 0.537→0.639 |
| Primary bootstrap CI | | | | | [−30.2, **+50.2**] |

### 2024 (fit)

| Cohort | n | Sealed pts MAE | Cand pts MAE | Δ MAE | Spearman sealed→cand |
|---|---:|---:|---:|---:|---|
| Overall | 43 | 71.75 | 70.76 | −0.99 | 0.749→0.651 |
| Top-12 | 12 | 68.80 | 84.82 | +16.02 | 0.441→0.245 |
| Dual-threat | 11 | 94.13 | 86.90 | −7.23 | 0.455→0.673 |
| Returning-injury | 13 | 62.60 | 60.52 | −2.07 | 0.797→0.808 |
| Pocket | 25 | 68.96 | 65.57 | −3.40 | 0.665→0.617 |
| Primary bootstrap CI | | | | | [−24.9, +12.2] |

### 2025 (holdout)

| Cohort | n | Sealed pts MAE | Cand pts MAE | Δ MAE | Spearman sealed→cand |
|---|---:|---:|---:|---:|---|
| Overall | 37 | 66.82 | 67.48 | +0.66 | 0.559→0.526 |
| Top-12 | 12 | 75.56 | 57.30 | −18.26 | −0.73→−0.26 |
| Dual-threat | 12 | 81.69 | 66.26 | −15.43 | 0.343→0.497 |
| Returning-injury | 11 | 82.50 | 74.25 | −8.25 | 0.609→0.464 |
| Pocket | 22 | 63.66 | 71.50 | +7.84 | 0.502→0.345 |
| Primary bootstrap CI | | | | | [−32.5, **+15.9**] |

Also reported on each cohort: attempts/active MAE, carries/active MAE, expected-starts MAE, sealed PPG MAE, candidate points-per-active MAE (see `fold_*_rows.json` / `selection_decision.json`).

## Gate outcomes vs sealed

| Gate | Result |
|---|---|
| Overall non-inferiority | **FAIL** (2023 +5.1% beyond +2% tol) |
| Primary cohort improve on ≥2 fit folds | **FAIL** (only 2024 improved; 2023 worsened) |
| Holdout primary improve | pass (point Δ −8.0) |
| Holdout primary bootstrap CI excludes 0 | **FAIL** (upper +15.9) |
| Holdout top-12 non-inferior | pass |
| Holdout Spearman | **FAIL** (0.559→0.526, drop > 0.02) |

**Experiment verdict vs sealed final: NO-GO**

## Burrow “30.1 attempts” units (explicit)

| Quantity | Value |
|---|---:|
| Attempts **per active start** | **≈ 37.27** |
| Expected starts | **≈ 13.06** |
| Expected **season** attempts | **≈ 486.6** (= 37.27 × 13.06) |
| Availability-adjusted attempts **per scheduled team game** (÷17) | **≈ 28.62** |
| Board composed attempts/g after starter-share floor (prior diagnostic) | **≈ 30.07** |

The earlier “30.1” figure was **availability-adjusted board attempts per scheduled team game**, not attempts per active start.

## Phases 2–5 status

| Phase | Status |
|---|---|
| 2 Immutable 2026 candidate namespace | **Not started** (blocked by Phase 1 NO-GO) |
| 3 Full 2026 QB review table for promotion | **Not started** |
| 4 Six-league shadow test | **Not started** |
| 5 Manual promotion | **NO-GO** — do not promote |

Sealed `v2_baseline_20260830` and `active_release_2026.json` remain untouched.

## Reproduce

```bash
uv sync --frozen --extra jobs --extra dev
uv run python scripts/qb_sealed_baseline_bakeoff.py
# exit code 2 = NO-GO
```

Artifacts: `output/qb_sealed_baseline_bakeoff/`

## Single next falsifiable hypothesis (do not auto-start)

**H3:** Active-start / archetype decomposition must be injected into the **sealed feature-construction + retrain + compose + ensemble** path and re-evaluated against `model_points_end_to_end` with that full stack. A post-hoc rate-replace candidate that beats injury-diluted carry-forward is not sufficient to beat the sealed end-to-end board on final fantasy points.

## PR handling

- PR #8 remains **draft**; this verification freezes the production-candidate decision at NO-GO.
- No separate production-candidate PR (Phases 2–5 blocked).
- PR #7 may still merge as diagnostic-only after human review (unchanged recommendation).
