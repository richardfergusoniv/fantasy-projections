# Weekly v2 volume tuning report

Generated: 2026-08-31 (local; experiment completed 2026-09-01 UTC)

Follow-up to [WEEKLY_V2_TRAINING_AND_PROMOTION_REPORT.md](WEEKLY_V2_TRAINING_AND_PROMOTION_REPORT.md).

Downstream architecture evaluation (separate from this grid): [WEEKLY_JOINT_USAGE_DRAWS_REPORT.md](WEEKLY_JOINT_USAGE_DRAWS_REPORT.md).

## Executive go / no-go decisions

| Decision | Result | Evidence |
|---|---|---|
| Trained artifact classification | **GO** (with caveats) | Existing `output/weekly_v2/models/season=2026/` manifest v2 still validates; no replacement candidate trained |
| Manual trained shadow publication | **NO-GO** | No grid candidate passes frozen promotion; 2023 dispersion remains below 0.70 for every candidate |
| Automatic weekly publication | **NO-GO** | `auto_publish_allowed` remains false; tuning selection `promote=false` |
| Use trained results for start/sit decisions | **NO-GO** | Dispersion gate failure; draw/conservation validation incomplete |
| Public-internet deployment | **NO-GO** | Unchanged external blockers (PostgreSQL, Docker, email, OpenAI, deployment) |

## 1. Reproduced baseline

**Command:**

```text
uv run python scripts/weekly_v2_evaluate.py --start 2022 --end 2025
```

**Exit code:** 2 (promotion failed, expected)

| Artifact | SHA256 |
|---|---|
| `output/weekly_v2/preseason_backtest.json` | `66c7a76517f8b13ce4339fb764e40821f4f7a0dda5531783e12ba55ff68682d7` |
| `data/processed/player_week_panel.parquet` | `b09a8ac0299d9bd18255c9a7e436a7982d34c9509f74a732ac79a7051afbae3c` |

**Calibrated dispersion (nested calibration, frozen WR/TE slope cap 1.52):**

| Season | Dispersion |
|---|---|
| 2023 | **0.6887** |
| 2024 | **0.6880** |
| 2025 | 0.8348 (passes) |

Policy minimum: **0.7000** (frozen; no tolerance applied).

## 2. Root cause of under-dispersion

Primary compression is **pre-calibration volume architecture**, concentrated in **WR/TE**:

| Segment | 2023 raw OOF dispersion | Pred SD | Actual SD |
|---|---|---|---|
| Overall | 0.5171 | 3.69 | 7.13 |
| WR | **0.3469** | 2.51 | 7.23 |
| TE | **0.3533** | 1.76 | 4.97 |
| QB | 0.7310 | 6.54 | 8.95 |
| RB | 0.4985 | 3.57 | 7.16 |

Diagnostic artifact: `output/weekly_v2/experiments/dispersion_baseline_20260831/dispersion_diagnostics.json`

### Attribution (quantitative)

1. **Two-stage participation × conditional share** — `P(usage) × E(share|usage)` shrinks cross-player spread; WR/TE conditional HGB predictions cluster (pred SD ~2.5 vs actual ~7.2).
2. **Lag-prior blending and depth floors** (`_blend_volume_with_lagged_priors`, `_seed_depth_share_anchors`) pull sticky elites and rookies toward priors, reducing tail separation.
3. **Team composition normalization** — `compose_team_volume_predictions` enforces simplex constraints; combined with accounting caps, room-level shares compress toward committee means.
4. **Cohort zero mass** — ~52% of roster-week actuals are DNP/zero; predicted zero fraction ~0%. Participation classifiers do not restore full cross-player variance after calibration.
5. **Calibration ceiling (not root cause)** — WR/TE slope cap 1.52 lifts overall dispersion from ~0.52 raw to ~0.689 calibrated but cannot close the remaining 1.6% gap without exceeding the frozen cap.

**Not primary:** availability double-counting on historical folds (most rows `play_prob=1.0` from nflverse). Injury haircut + participation feature overlap affects questionable/out players but does not explain healthy-cohort WR/TE compression.

Supporting evidence from the grid: `legacy_direct` (no two-stage multiplication) reaches 2024 **0.7026** but still fails 2023 at **0.6915**, confirming two-stage is contributory but not the sole compressor.

## 3. Probability / availability semantics

| Contract | Representation |
|---|---|
| Available for game | `play_prob`, `is_out` injury flags |
| Active, no usage | Low participation probability in two-stage model |
| Snap/dropback participation | Not separately modeled; implied via snap/dropback share targets |
| Positive target/carry/RZ usage | Participation classifier threshold on share targets |
| Conditional usage share | Conditional HGB/ridge on positive-usage rows |

**Double-counting finding:** `play_prob` enters the participation classifier as a feature **and** `apply_injury_haircut(mode="shares")` scales `pred_*_share` before accounting. For historical evaluation this is mostly inert (`play_prob≈1`). For live questionable players, shares are scaled twice in expectation — conservative, not the dispersion bottleneck. Hard `mode="stats"` zeroing is a separate downstream step and does not duplicate participation.

Tests: `tests/test_weekly_v2_probability_semantics.py`

## 4. Tuning harness defects fixed

| Defect | Fix |
|---|---|
| Tuner imported missing `scripts.preseason_eval` | Tuner uses `src/projection/weekly/evaluate/harness.py` + `nested_selection.py` |
| Dispersion disabled in candidate eligibility (`min=0, max=999`) | Lexicographic selection requires dispersion band before MAE gains |
| MAE-only ranking | Dispersion-aware rank key + full `promotion_gate` on final claim |
| Evaluator CLI ignored tuning selection | `--tuning-selection` / `--volume-options-json` on `weekly_v2_evaluate.py` |
| Train read mutable root `tuning_selection.json` + env hack | Explicit `--tuning-selection`; `set_registry_dir(output_dir)`; manifest embeds provenance |
| No experiment provenance | Namespace under `output/weekly_v2/experiments/<id>/` with protocol + hashes |
| Same seasons used for selection and holdout | Nested per-outer-fold inner selection; warm-up folds flagged |
| Redundant season re-evaluation | Per-candidate disk cache; one backtest per candidate |

Authoritative library: `src/projection/weekly/evaluate/harness.py`

Tests: `tests/test_weekly_v2_tuning_harness.py`

## 5. Frozen candidate grid and selection rule

Grid (`src/projection/weekly/models/volume_config.py` `DEFAULT_CANDIDATE_GRID`):

- `baseline_two_stage` (current)
- `two_stage_half_life_{6,4,2}`
- `two_stage_participation_ridge`
- `two_stage_conditional_ridge`
- `two_stage_conservative_hgb` (declared; identical hyperparameters to baseline in this pass)
- `legacy_direct`

**Selection rule (predeclared, lexicographic):**

1. Inner-fold selection gate (dispersion ∈ [0.70, 1.30], MAE/rank thresholds) with `min_seasons` adapted to available calibrated folds
2. MAE improvement vs baseline on inner folds
3. Rank correlation non-degradation
4. Dispersion distance to 1.0
5. **Train only if** full frozen `PromotionPolicy` (`min_seasons=3`) passes on all calibrated outer folds

## 6. Nested fold design

- **Outer targets:** 2023–2025 scored; 2022 calibration warm-up (not promotion-scored)
- **Inner selection for outer T:** calibrated reports from seasons strictly before T
- **Calibration:** nested leave-one-season-out within each candidate backtest
- **2026 training selection:** requires full-policy pass; none achieved

Experiment: `output/weekly_v2/experiments/volume_tune_20260831_v2/`
- `nested_selection.json`
- `tuning_selection.json` (`promote=false`, `selected=null`)
- `selection_protocol.json`
- `candidate_cache/*.json`

## 7. Candidate Pareto table (calibrated)

| Candidate | 2023 disp | 2024 disp | 2025 disp | MAE 2023 | Promote |
|---|---|---|---|---|---|
| baseline_two_stage | 0.6887 | 0.6880 | 0.8348 | 4.0042 | **No** |
| two_stage_half_life_6 | 0.6802 | 0.6913 | 0.8332 | 4.0196 | **No** |
| two_stage_half_life_4 | 0.6824 | 0.6909 | 0.8342 | 4.0281 | **No** |
| two_stage_half_life_2 | 0.6758 | 0.6912 | 0.8325 | 4.0795 | **No** |
| two_stage_participation_ridge | 0.6819 | 0.6897 | 0.8297 | 4.0172 | **No** |
| two_stage_conditional_ridge | 0.6469 | 0.6669 | 0.8300 | 4.1856 | **No** (+ rank fail 2024) |
| two_stage_conservative_hgb | 0.6887 | 0.6880 | 0.8348 | 4.0042 | **No** |
| legacy_direct | **0.6915** | **0.7026** | 0.8394 | 4.0079 | **No** (2023 only) |

No candidate may average away a failed 2023 or 2024 fold. Closest: `legacy_direct` clears 2024 but fails 2023 by ~0.0085.

## 8. Selected candidate

**None.** `tuning_selection.json`:

- `promote: false`
- `selected: null`
- `best_relative_candidate: legacy_direct` (relative rank only; not authorized for training)

Rationale: every predeclared candidate fails the frozen all-fold promotion policy. Retraining a 2026 candidate is not justified.

## 9. Comparison vs current model

| Metric | Baseline 2023 | Baseline 2024 | Best relative (`legacy_direct`) 2023 | 2024 |
|---|---|---|---|---|
| Calibrated dispersion | 0.6887 | 0.6880 | 0.6915 | 0.7026 |
| Calibrated MAE | 4.004 | 4.234 | 4.008 | 4.235 |
| Calibrated rank_corr | 0.458 | 0.442 | 0.461 | 0.443 |

No new 2026 candidate trained (existing artifact preserved).

## 10. Calibration ordering / rank interaction (1.52 vs 1.60)

Within-position calibration is a positive affine map `intercept + slope × x` with `slope ≤ cap`, which **preserves strict within-position ordering** (tests in `tests/test_weekly_v2_calibration_invariants.py`).

The earlier 1.60 cap rank regression is therefore **not** within-position rank destruction. Likely mechanisms:

- **Cross-position season aggregation** — season-level ranks mix positions after weekly calibration
- **Zero clipping** after calibration (`clip(0.0, None)`) creates ties at zero for low projections
- **WR/TE interval expansion** — wider floors/ceilings affect interval metrics, not within-position point order

Evaluation applies calibration **once** in the harness path. Do not resume slope-cap searching.

## 11. Artifact hashes / experiment outputs

| Artifact | Status |
|---|---|
| Existing 2026 manifest | Unchanged (`output/weekly_v2/models/season=2026/manifest.json`) |
| New tuned 2026 candidate | **Not produced** (valid no-go) |
| Dispersion diagnostics | `output/weekly_v2/experiments/dispersion_baseline_20260831/dispersion_diagnostics.json` |
| Nested selection | `output/weekly_v2/experiments/volume_tune_20260831_v2/nested_selection.json` |
| Tuning selection | `promote=false`; panel hash `b09a8ac0…` |

## 12. Verification

```text
uv run pytest tests/test_weekly_v2_tuning_harness.py tests/test_weekly_v2_probability_semantics.py tests/test_weekly_v2_calibration_invariants.py -q
uv run pytest -q
uv run python scripts/weekly_v2_evaluate.py --start 2022 --end 2025
uv run python scripts/weekly_v2_dispersion_diagnose.py --oof output/weekly_v2/preseason_oof.parquet
uv run python scripts/weekly_v2_tune_preseason.py --namespace volume_tune_20260831_v2 --seed 42
uv run python scripts/audit_weekly_features.py
uv run python scripts/verify_mvp.py
uv run python scripts/audit_blueprint_mvp.py
uv run python scripts/vertical_smoke.py
```

| Command | Outcome |
|---|---|
| Focused harness/probability/calibration tests | Passed |
| Full `pytest -q` | **788 passed, 1 skipped** |
| Baseline evaluate | Exit 2 (expected) |
| Nested tuner `volume_tune_20260831_v2` | Exit 2 (`promote=false`, expected no-go) |
| `audit_weekly_features` / `verify_mvp` / `audit_blueprint_mvp` (49/49) / `vertical_smoke` | All passed |

## 13. Remaining blockers

- PPFD first-down stats not modeled
- Teammate conservation gates for stat draws open
- K/DST weekly models not in readiness
- Start/sit draw distributions not fully validated
- PostgreSQL/Docker deployment unverified
- **Architectural:** WR/TE volume variance requires mixture/draw redesign, not calibration-cap or hyperparameter tuning

## Next architectural change (evidence-backed)

Replace or augment the two-stage **point-mass** share expectation with either:

1. Explicit **zero-inflated** mixture at the fantasy-point layer that respects cohort DNP mass (~52% zeros), or
2. **Draw-based** usage events (sample participation, then conditional share) for simulation while keeping calibrated means for point projections

`legacy_direct` improving 2024 past 0.70 without clearing 2023 shows architecture matters more than recency/classifier family knobs, but not enough alone.

Do **not** pursue further slope-cap or promotion-threshold adjustments.

See [WEEKLY_JOINT_USAGE_DRAWS_REPORT.md](WEEKLY_JOINT_USAGE_DRAWS_REPORT.md) for the
implemented mixture/joint-draw follow-up and its separate go/no-go gates.
