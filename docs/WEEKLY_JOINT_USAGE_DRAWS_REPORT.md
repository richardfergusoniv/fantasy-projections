# Weekly joint usage-mixture and correlated draws report

Generated: 2026-08-31 (local)

Follow-up to [WEEKLY_V2_VOLUME_TUNING_REPORT.md](WEEKLY_V2_VOLUME_TUNING_REPORT.md) and
[WEEKLY_V2_TRAINING_AND_PROMOTION_REPORT.md](WEEKLY_V2_TRAINING_AND_PROMOTION_REPORT.md).

This architecture evaluation is **not** a volume-grid selection. The completed
nested experiment `output/weekly_v2/experiments/volume_tune_20260831_v2/` remains
authoritative with `promote=false`, `selected=null`.

## Executive go / no-go decisions (six separate)

| Decision | Result | Evidence |
|---|---|---|
| Point-model classification | **GO** (with caveats) | Unchanged trained `season=2026` artifact; volume tune still `promote=false` |
| Joint-draw classification | **NO-GO** | Event Brier fails vs frequency baseline (0/21 cells); zero-mass gap 0.252; decision backtest incomplete |
| Manual trained shadow publication | **NO-GO** | Joint gates incomplete; point dispersion still fails frozen policy |
| Automatic weekly publication | **NO-GO** | `auto_publish_allowed=false`; point-dispersion gate unchanged and failing |
| Start/sit use | **NO-GO** | Decision lineup/matchup outer-fold gate not passed |
| Public-internet deployment | **NO-GO** | Unchanged external blockers (PostgreSQL, Docker, email, OpenAI, deployment) |

**Valid completion outcome:** architecture + evaluation implemented; failed gates
documented with evidence; active/fallback releases untouched; no artificial
variance or gate substitution.

## 1. Existing architecture defects reproduced

Regression tests in `tests/test_weekly_legacy_draw_defects.py` prove the trained
legacy path still:

1. Scales all component stats by one fantasy-point factor (perfect lockstep; fractional TDs).
2. Emits no discrete DNP / participation event.
3. Draws teammates independently (near-zero correlation under shared salt).
4. Does not conserve team pass/receive identities across players.
5. Omits first downs from typical trained mean rows (PPFD understated until joint path).
6. Documents independence as the legacy/fallback assumption in `src/app/decisions/draws.py`.
7. Uses fixed-prior independent K/DST simulators unlinked to a shared game draw.

## 2. Event / target / probability contracts

Versioned contract: `weekly_mixture_contract_v1`
(`src/projection/weekly/draws/contracts.py`).

| Layer | Name | Owner |
|---|---|---|
| Active for game | `p_active` | `play_prob` (sole availability authority) |
| Offensive participation \| active | `p_participates` | `event_model.offensive_participation` |
| Positive usage \| participates | `p_positive_usage` | `event_model.positive_usage` |
| Conditional usage \| positive | shares | `volume.conditional_share` |
| Conditional efficiency \| usage | rates | `efficiency.conditional` |

Identity (hand-tested):

```text
E[stat] = P(active) × P(participates | active) × E[stat | participates]
```

`apply_active_once(..., already_conditioned_on_active=True)` refuses double
counting. Bye / no-game rows are classified `bye_no_game` and excluded from
participation denominators.

## 3. Data cutoffs, sources, hashes, leakage

| Artifact | Path / hash |
|---|---|
| Source panel | `data/processed/player_week_panel.parquet` (same panel used by volume tune) |
| Mixture panel | `output/weekly_v2/experiments/joint_usage_draws_20260831/mixture_panel/` |
| Experiment summary | `.../joint_usage_eval_summary.json` |
| Contract fingerprint | recorded in summary (`contract_fingerprint`) |

Labels are same-week outcomes for training only; features remain pre-kickoff /
as-of. Existing poisoned-future tests retained. Bye rows use null `game_id` →
not treated as failed participation.

## 4. Model specifications and training samples

| Component | Spec |
|---|---|
| Event models | Per-position logistic + `StandardScaler`; `class_weight=balanced`; seed 42 |
| Joint game engine | Shared env latent → team Poisson totals → Bernoulli events → Dirichlet room allocation → conditional efficiency → reserve reconciliation → game-linked K/DST |
| First downs | Position beta-binomial rates; bounds vs receptions/carries; pass FD := rec FD |
| Partition schema | **v2** `joint_stat_partition.json` (legacy v1 retained) |

Draw mode labels: `legacy_points_independent`, `legacy_scaled_components`,
`joint_stat_mixture_candidate`, `joint_stat_mixture_validated`.

## 5. Event calibration (rolling OOF)

Source: `event_calibration.json`. Representative Brier vs frequency baseline:

| Fold | Event | Pos | Brier | Baseline | Result |
|---|---|---|---:|---:|---|
| 2023 | participated | WR | 0.146 | 0.032 | Fail |
| 2023 | positive_usage | WR | 0.167 | 0.094 | Fail |
| 2024 | participated | RB | 0.150 | 0.063 | Fail |
| 2025 | positive_usage | TE | 0.195 | 0.109 | Fail |

**Aggregate:** `brier_beats_baseline=0/21` scored cells.

**Diagnosis:** panel skill rows with scheduled games have very high participation
prevalence; balanced logistic over-predicts zeros and loses to the frequency
baseline. `is_active_label` cells were skipped (insufficient negatives under
historical `play_prob≈1`). Next change: redefine events on a complete-roster
cohort with true DNP mass, drop balanced weights, and/or use depth/status-only
baselines as the event layer.

## 6. Distributional metrics vs baselines

Sample: 2024 week-1 games (`draw_count=80`).

| Metric | Joint mixture | Legacy scaled components |
|---|---:|---:|
| CRPS (mean) | **2.839** | 4.892 |
| Zero-mass gap \|pred−actual\| | 0.252 | (legacy near-zero pred mass) |
| Teammate corr (mean) | 0.012 | ~0 by construction |

Joint CRPS improves vs legacy on this sample, but zero-mass calibration misses
the predeclared `<0.25` gate. **Deterministic point-dispersion is unchanged and
still failing** (2023 baseline 0.6887); sample variance of draws is **not** used
to claim the point gate passed.

## 7. Unchanged deterministic point-promotion results

From `volume_tune_20260831_v2/tuning_selection.json` (preserved):

- `promote: false`
- `selected: null`
- `best_relative_candidate: legacy_direct` (not trained; failed 2023)

WR/TE slope cap remains **1.52**. No hyperparameter re-search on the old volume
architecture.

## 8. Team/game conservation and correlation

After reserve handling for rooms missing a modeled QB:

- Conservation report: **0 violations** on 240 draws / 5 team-blocks (`tol=3`).
- Structural invariants covered: targets, receptions≤targets, pass/rec yards & TDs,
  carries, first-down bounds, inactive zeroing, bye skip, locked actuals.
- Correlation evidence is still weak on the week-1 sample (0.012); shared env is
  present but room Dirichlet competition offsets stacking signal. Next change:
  stronger shared game latent on yards/TDs before room allocation.

## 9. PPFD, K, and DST validation

| Path | Status |
|---|---|
| PPFD components in joint draws | Implemented (`pass/rush/rec_first_downs`); bounds tested |
| Kicker | Game-linked to team TDs / scoring drives |
| DST | Game-linked to opponent points/yards |
| Live exact K/DST historical eval | Thin — marked high uncertainty |
| Fixture six-league shadow | PPFD nonzero on 20/20 sampled players |

PPFD still marked incomplete for **validated** start/sit until live contract
blobs (not just fixtures) are scored end-to-end with owner roster identity.

## 10. Decision-level lineup and matchup backtests

- Thresholds predeclared in `DecisionReadinessThresholds` **before** outer review.
- Synthetic unit tests cover regret, win probability, stability, and gate failure.
- Full out-of-fold start/sit / matchup backtest on dynasty lineups: **not completed**
  in this run → decision gate **FAIL** → start/sit remains **NO-GO**.

## 11. Six-league live shadow results

Artifact:
`output/weekly_v2/experiments/joint_usage_draws_20260831/six_league_joint_shadow.json`

| Check | Result |
|---|---|
| Partition verify | OK (`schema_version=2`) |
| Distinct contracts scored | 6 / 6 (fixture shapes mapped to live league IDs) |
| Pointer advanced | **false** |
| Tampered `validated` without gates | Blocked |
| Exact live `scoring_settings` blobs | **Unavailable** in static artifacts / owner JSON (DB snapshots not loaded here) |
| Empty pre-draft redraft roster skip / owner identity | Not fully exercised (blocker noted) |

## 12. Artifact / partition schema and hashes

| Item | Value |
|---|---|
| Experiment | `output/weekly_v2/experiments/joint_usage_draws_20260831/` |
| Joint partition | `shadow_partition/joint_stat_partition.json` |
| Partition SHA256 | `f9549c2861a9c79bae09c04a43aee667521501c8cbaae0dd6bbaec8464bba652` |
| Schema | 2 (`JOINT_PARTITION_SCHEMA_VERSION`) |
| Legacy schema | 1 retained (`stat_draw_partition.json` path unchanged) |

## 13. Defects found and fixed

| Defect | Fix |
|---|---|
| Legacy lockstep scaled draws | Retained as labeled legacy; joint engine replaces candidate path |
| No DNP event | Explicit Bernoulli active/participates/positive_usage |
| Independent teammates | Game-level shared env + room Dirichlet |
| No conservation | Per-draw validation + reserves; missing-QB reserve path |
| Missing first downs | Binomial FD sampling + map keys for scoring |
| Independent K/DST | `special_teams_game.py` linked to `GameOffenseState` |
| Double-count risk for `play_prob` | Contract + `apply_active_once` |
| Conservation fail when QB absent from sample | Team passing line placed in explicit reserve |

## 14. Skips, degradations, external blockers

- Event models lose to frequency baseline on high-prevalence labels.
- Teammate correlation weak on first sample.
- Full OOF decision backtest deferred (gate fail).
- Exact live scoring_settings not in static files.
- PostgreSQL/Docker/email/OpenAI/public deploy unchanged blockers.
- Point-dispersion still fails; **must not** be “passed” via draw variance.

## 15. Separate go/no-go (detail)

1. **Point-model classification:** GO with caveats (unchanged).
2. **Joint-draw classification:** NO-GO (event calibration + zero-mass + decision gaps).
3. **Manual trained shadow publication:** NO-GO.
4. **Automatic weekly publication:** NO-GO (`auto_publish_allowed=false`).
5. **Start/sit use:** NO-GO.
6. **Public-internet deployment:** NO-GO.

### Narrowest evidence-backed next model change

Rebuild the discrete-event layer on the **complete recoverable roster cohort**
(where ~50%+ true zeros live), fit **unweighted / depth-status** calibrators that
beat frequency baselines out of fold, then strengthen the shared game latent so
QB–receiver correlation and zero-mass calibration clear predeclared gates—without
touching the 1.52 slope cap or point-promotion thresholds.

## Commands / tests

```text
uv run pytest tests/test_weekly_legacy_draw_defects.py tests/test_weekly_joint_usage_draws.py -q
uv run python scripts/weekly_v2_joint_usage_eval.py --draw-count 80 --max-games 8
uv run python scripts/weekly_v2_joint_six_league_shadow.py
```

Focused draw suites: **20 passed**. Full deterministic suite after this work:
**809 passed, 1 skipped** (prior baseline was 788 passed / 1 skipped).

## Files changed (grouped)

- **Data/features:** `src/projection/weekly/draws/mixture_panel.py`
- **Models:** `event_models.py`, `first_downs.py`, `special_teams_game.py`
- **Draws:** `game_engine.py`, `conservation.py`, `partition_schema.py`; app
  `weekly_draws.py` joint writer; `weekly_stat_draw.py` first-down map
- **Scoring / decisions:** `decisions/draws.py` mode labels + joint docs;
  six-league shadow script
- **Evaluation / readiness:** `evaluate_dist.py`, `decision_backtest.py`,
  `readiness.py`; `scripts/weekly_v2_joint_usage_eval.py`
- **Tests:** `tests/test_weekly_legacy_draw_defects.py`,
  `tests/test_weekly_joint_usage_draws.py`
- **Docs:** this report; links from training/tuning reports

## Follow-up repair (2026-08-31)

See [WEEKLY_EVENT_COHORT_EVAL_REPAIR_REPORT.md](WEEKLY_EVENT_COHORT_EVAL_REPAIR_REPORT.md)
for the corrected complete-roster cohort, training-only baselines, leakage-free joint
wiring, and updated go/no-go evidence under
`output/weekly_v2/experiments/event_cohort_eval_repair_20260831/`.
This report and `joint_usage_draws_20260831/` remain preserved as invalidated
historical evidence.
