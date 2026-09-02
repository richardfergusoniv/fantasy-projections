# Weekly event cohort and evaluation-integrity repair report

Generated: 2026-08-31 (local)

Follow-up to [WEEKLY_JOINT_USAGE_DRAWS_REPORT.md](WEEKLY_JOINT_USAGE_DRAWS_REPORT.md).
Prior experiment `output/weekly_v2/experiments/joint_usage_draws_20260831/` is preserved as
invalidated historical evidence.

Corrected namespace: `output/weekly_v2/experiments/event_cohort_eval_repair_20260831/`

## Executive go / no-go decisions (seven separate)

| Decision | Result | Evidence |
|---|---|---|
| Point-model classification | **GO** (with caveats) | Unchanged trained artifact; volume tune `promote=false`, `selected=null` |
| Event layer | **NO-GO** | Corrected OOF: `brier_beats_training_baseline=8/36` (requires ≥60% majority) |
| Joint-draw classification | **NO-GO** | Event gate failed; PPFD/K/DST/league/decision gates incomplete |
| Manual trained shadow publication | **NO-GO** | Joint + point dispersion gates not cleared |
| Automatic weekly publication | **NO-GO** | `auto_publish_allowed=false` |
| Start/sit use | **NO-GO** | Full OOF lineup/matchup backtest incomplete |
| Public-internet deployment | **NO-GO** | Unchanged external blockers |

**Valid completion outcome:** evaluation-integrity defects fixed; corrected cohort and
leakage-free wiring in place; honest training-only baselines; event layer still fails
predeclared gates; shared-latent tuning **not** authorized.

## 1. Prior metrics invalidated

| Prior metric | Why invalidated |
|---|---|
| Event `0/21` Brier wins vs baseline | Oracle baseline used test-fold prevalence (`y_true.mean()`); wrong event denominators |
| `is_active_label` from `play_prob`/`is_out` | Circular labeling; not observed roster active status |
| Joint CRPS `2.84` vs legacy `4.89` | Same-week actuals wired into `_synthetic_team_game_from_rows` (constants 0.85/0.70, shares, attempts) |
| Zero-mass gap `0.252` | Leaked partition inputs and point means = realized FP |
| Teammate corr `0.012` | Invalid evaluation path; not used for latent tuning |
| Conservation `ok OR violations < 50` | Arbitrary bypass removed |

## 2. Corrected cohort construction

Contract: `weekly_mixture_contract_v2` (`src/projection/weekly/draws/contracts_v2.py`).

Sources (2022–2025):

- Weekly nflverse roster membership (`data/raw/rosters_weekly_*`)
- REG schedules (`data/raw/schedules_*`)
- Left-joined `player_week_panel` outcomes/features

Artifact: `cohort/complete_roster_cohort.parquet`
Content hash (all rows): `d26a9a9118899c021559995b8ad4c10176f413e49f91f49c51b0c224d35ea70a`

| Metric | Corrected cohort (scheduled rows) |
|---|---:|
| Total cohort rows | 56,778 |
| Scheduled roster-week rows | 54,088 |
| Active denominator | 40,362 |
| Participation denominator (given active) | 30,062 |
| Positive-usage denominator (given participation) | 22,978 |
| Observed active rate | 74.5% |
| Participation \| active | 76.4% |
| Positive usage \| participation | 92.0% |
| Fantasy-point zero rate | **63.3%** |

Row-state counts (mutually auditable):

| State | Count |
|---|---:|
| `positive_usage` | 21,134 |
| `outcome_missing_or_source_incomplete` | 16,525 |
| `not_on_recoverable_roster_at_cutoff` | 13,726 |
| `bye_or_no_scheduled_game` | 2,690 |
| `participated_zero_positive_usage` | 1,844 |
| `active_no_offensive_participation` | 859 |

The prior ~52% zero figure came from the **skill-position stats panel** only. The corrected
full roster cohort shows **~63%** FP zeros on scheduled rows, driven by inactive/DNP,
missing outcomes, and true zero-usage classes—not a single conflated DNP bucket.

## 3. Label / denominator contracts

```text
active_label:          denominator = scheduled + rostered + observed roster status
participated_label:    denominator = active_label == true
positive_usage_label:  denominator = participated_label == true
```

- Active ground truth: weekly roster `ACT` vs `INA`/`RES` (never `play_prob`)
- Participation: `offense_snaps > 0` when snap source present; else unknown
- Positive usage: QB attempts/carries; RB carries/targets; WR/TE targets

## 4. Feature / target leakage controls

- `feature_outcome_split.py`: denylist + `split_prediction_outcome_frames`
- `prediction_inputs.py`: rejects/forbids same-week actual columns; requires fold event preds
- Poison tests in `tests/test_weekly_event_cohort_eval_repair.py`
- Outcomes joined only after partition generation for scoring

## 5. Training-only baselines

- `event_baselines.py`: constant training prevalence + depth/status logistic (+ play_prob heuristic for active)
- `evaluate_event_predictions()` **requires** explicit baseline; no `y_true.mean()` default
- Per-fold artifacts: `event_oof/baselines_train_end_*.json`

## 6. Event-model OOF results (corrected)

Outer folds: train ≤2022→test 2023; ≤2023→2024; ≤2024→2025.

- Unweighted logistic (`class_weight=None`); no balanced-weight default
- Comparator: strongest valid deployable baseline (depth/status logistic when fit, else constant prevalence)
- **Aggregate gate:** `8/36` cells beat training baseline Brier (< majority at 60%)
- Prior `0/21` retained under `joint_usage_draws_20260831/event_calibration.json` (not revised)

## 7. Leakage-free joint draw evidence

Wiring: fold-specific `p_participates` / `p_positive_usage` from fitted models/baselines;
`projection_rows_only` strips outcomes before `build_scheduled_game_from_predictions`;
point means from draw MC means (not realized FP).

OOF joint sample (120 games, 80 draws, 2023–2025 outer folds):

| Metric | Corrected |
|---|---:|
| CRPS (mean) | 1.07 |
| Zero-mass \|pred−actual\| | 0.013 |
| Players scored | 1,189 |
| Conservation violations | 0 |
| Teammate corr (mean) | NaN (sparse pairs in sample) |

Prior CRPS `2.84` comparison is **superseded** (leaked inputs).

## 8. Exact live six-league scoring

**Incomplete.** Live shadow DB / exact `LeagueRuleSnapshot` validation not run in this repair pass.
Fixture substitution remains blocked by readiness policy.

## 9. Shared-latent tuning

**Not reached.** Event gate failed (`shared_latent_tuning_authorized=false`).

## 10. Tests and commands

```text
uv run pytest tests/test_weekly_event_cohort_eval_repair.py tests/test_weekly_joint_usage_draws.py -q
uv run python scripts/weekly_v2_event_cohort_eval_repair.py --max-games-per-fold 40
```

Focused repair tests: **23 passed**.
Prior `joint_usage_draws_20260831` artifacts untouched.

## 11. Why publication / start-sit remain disabled

- Point-dispersion gate unchanged and failing (frozen volume-tune policy)
- Event layer does not beat training-only baselines at the predeclared majority threshold
- PPFD/K/DST/league/decision gates lack historical evidence artifacts
- `auto_publish_allowed=false` by `JointReadinessReport.recompute_decisions`

## 12. Files changed

| Area | Files |
|---|---|
| Cohort / contract v2 | `cohort_panel.py`, `contracts_v2.py` |
| Leakage / inputs | `feature_outcome_split.py`, `prediction_inputs.py` |
| Events / baselines | `event_models.py`, `event_baselines.py` |
| Readiness | `readiness.py` |
| Evaluation | `scripts/weekly_v2_event_cohort_eval_repair.py` |
| Tests | `tests/test_weekly_event_cohort_eval_repair.py`, `tests/test_weekly_joint_usage_draws.py` |
| Docs | this report; link added to joint usage report |
