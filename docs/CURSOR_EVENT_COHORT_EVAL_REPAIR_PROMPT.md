# Cursor full-roster event cohort and evaluation-integrity repair prompt

Copy everything below the divider into Cursor Agent mode from the repository
root after the joint usage-draw no-go implementation is present.

---

You are the senior ML validation engineer responsible for repairing the event
cohort, prediction wiring, and historical evaluation used by the new weekly
joint-draw architecture.

The previous pass correctly left every publication/start-sit decision at
NO-GO, but several implementation defects make its event and distributional
metrics unsuitable for model selection. Fix the evidence pipeline first. Then
fit and evaluate the simplest correctly specified event layer on the complete
recoverable roster cohort. Do not tune the shared game latent until event
probabilities and evaluation provenance are trustworthy.

This is an implementation and rerun task—not a documentation-only audit. Fix
the defects, add regression tests, rebuild all affected artifacts under a new
namespace, run rolling out-of-fold evaluation, and preserve the no-go if the
corrected model does not pass.

## Current decision state

Preserve these defaults:

- Point-model classification: GO with caveats.
- Joint-draw classification: NO-GO.
- Manual trained shadow publication: NO-GO.
- Automatic weekly publication: NO-GO.
- Start/sit use: NO-GO.
- Public-internet deployment: NO-GO.

The volume selection remains permanently recorded as:

```text
output/weekly_v2/experiments/volume_tune_20260831_v2/
promote=false
selected=null
```

Do not retrain `legacy_direct`, change the WR/TE slope cap of 1.52, weaken the
0.70 point-dispersion minimum, or substitute draw variance for point-forecast
dispersion.

The previous joint artifact/report remains historical evidence:

```text
output/weekly_v2/experiments/joint_usage_draws_20260831/
docs/WEEKLY_JOINT_USAGE_DRAWS_REPORT.md
```

Never overwrite it. The corrected experiment must use a new namespace such as:

```text
output/weekly_v2/experiments/event_cohort_eval_repair_20260831/
```

## Release-blocking defects already identified

Reproduce these with focused tests before or while fixing them.

### 1. Event probabilities are not wired into joint evaluation

In `scripts/weekly_v2_joint_usage_eval.py`, `_synthetic_team_game_from_rows()`
currently contains behavior equivalent to:

```text
p_participates = 0.85 if condition or True else 0.2
p_positive_usage = 0.7
```

The `or True` makes participation a constant 0.85. Positive usage is always
0.70. The fitted `EventModelBundle` is evaluated separately but its predictions
do not power the tested joint draws. Remove every constant/test shortcut from
the real evaluation and application candidate path.

### 2. Same-week outcomes leak into draw inputs

The joint sample currently builds predictive inputs from the target week's
realized values, including some or all of:

- `offense_snaps`
- `target_share`
- `carry_share`
- QB `attempts`
- realized `fantasy_points` assigned as `point_means`

Actual outcomes may be used only after prediction for scoring. They must never
enter `PlayerGameInput`, `TeamGameInput`, event probabilities, point means,
conditional shares, game environment, or draw calibration for that same target
week.

### 3. The frequency baseline uses target-fold outcomes

`evaluate_event_predictions()` defaults `baseline_rate` to `y_true.mean()`.
The rolling evaluator does not pass a training-fold rate, so its “frequency
baseline” uses the test fold's realized prevalence. That is an oracle baseline,
not a deployable OOF comparator. The reported `0/21` result cannot be treated as
a valid model-versus-baseline conclusion until corrected.

Use training-only baselines frozen before the test fold. Retain the old 0/21
artifact as invalidated historical evidence; do not erase or silently revise it.

### 4. Active target leakage / circular labeling

`build_mixture_panel()` defines `is_active_label` from `is_out` and
`play_prob`, while the active model includes `is_out` and `play_prob` as
features. This predicts a label from the same rule that created it rather than
predicting observed game availability. Build an outcome label from official
inactive/game participation/roster evidence when recoverable. If true active
status cannot be observed for a row, mark it unknown; do not synthesize it from
the prediction feature.

`play_prob` may remain an as-of feature and the live `p_active` authority, but it
cannot also define the historical ground truth used to validate itself.

### 5. Conditional denominators are wrong

The contract says:

```text
P(active)
P(participates | active)
P(positive_usage | participates)
```

Current model fitting filters only to scheduled games for every event. It fits
`positive_usage_label=False` on nonparticipants, estimating an unconditional
probability and then multiplies it by `P(participates)`, double-counting the
participation event. Fit/evaluate each event only on its declared denominator:

- Active: scheduled, roster-eligible rows with observed active outcome.
- Participation: rows observed active.
- Positive usage: rows observed participating.

Make these denominator counts explicit in metadata and tests.

### 6. The source panel is not the full roster-week cohort

Filtering the player-week statistics panel to skill positions selects players
represented in weekly stats/feature data. It does not prove that every rostered
player scheduled that week is present. Rebuild from the complete recoverable
historical roster × scheduled-week cohort, left-joining stats so true DNP/zero
rows remain.

Reuse the repository's existing cohort construction (including
`complete_roster_week_outcomes` / `roster_week_cohort`) when its grain and as-of
semantics fit. Do not create a second inconsistent zero-fill implementation.

### 7. Readiness gates overstate partial components

The current evaluation sets PPFD, kicker, and DST readiness true from the
presence of code while simultaneously admitting historical validation is thin.
League scoring completeness is fixture-mapped rather than based on exact live
rule snapshots. Conservation currently permits a fallback condition equivalent
to `ok OR violations < 50`.

Readiness must be evidence-based:

- Structural conservation requires the declared tolerance policy to pass; it
  cannot permit an arbitrary count of violations.
- PPFD/K/DST implementation presence is not validation readiness.
- Fixture contracts mapped to live IDs are not exact live contract validation.
- Artifact hashing alone does not establish correct prediction provenance.

## Read before editing

Read completely:

1. `docs/WEEKLY_JOINT_USAGE_DRAWS_REPORT.md`
2. Prior weekly training and volume-tuning reports
3. `scripts/weekly_v2_joint_usage_eval.py`
4. `scripts/weekly_v2_joint_six_league_shadow.py`
5. Everything under `src/projection/weekly/draws/`
6. `src/app/projections/weekly_draws.py`
7. `src/app/decisions/draws.py`
8. Weekly inference, feature, panel, roster, schedule, status, and accounting
   modules
9. `src/projection/weekly/evaluate/preseason.py` and the repaired harness
10. Live Sleeper shadow-sync code, models, rule snapshots, and reports
11. Scoring contracts/compiler and decision services
12. Every related joint/event/leakage/conservation test

Inspect `git status` and preserve all existing work, artifacts, reports, live
shadow databases, and active release pointers. Do not commit, deploy, publish,
write to Sleeper, or call paid services.

## Phase 1 — create a versioned complete-roster cohort

Build `weekly_mixture_contract_v2` from explicit source grains:

1. Historical weekly roster membership/eligibility as known before kickoff.
2. NFL schedule and kickoff times.
3. Official inactive/active evidence or the strongest recoverable public proxy.
4. On-field participation/snaps.
5. Weekly player component outcomes.
6. Strictly pre-kickoff/lagged predictive features.

Construct one row per recoverable rostered skill player × scheduled team-week.
Do not require a box-score row to enter the cohort. Left-join outcomes and make
zero/missing semantics explicit.

Classify rows into mutually auditable states:

- `bye_or_no_scheduled_game` — excluded from all game-event denominators.
- `not_on_recoverable_roster_at_cutoff` — excluded, not a DNP.
- `active_status_unknown` — retained for downstream labels when possible but
  excluded from active calibration.
- `inactive` — scheduled and observed inactive.
- `active_no_offensive_participation`.
- `participated_zero_positive_usage`.
- `positive_usage`.
- `outcome_missing_or_source_incomplete` — excluded from scoring, never zeroed
  silently.

For each season/position/depth band, report:

- Candidate roster-week rows.
- Included/excluded counts and reasons.
- Observed active rate.
- Participation rate conditional on active.
- Positive-usage rate conditional on participation.
- Fantasy-point zero rate.
- Missing outcome/status rates.

Reconcile the previous claim of approximately 52% zero outcomes. Explain which
zero classes produce it and whether it is stable across seasons. If the rate
changes materially under the corrected cohort, update the diagnosis rather than
forcing the old number.

Persist the cohort, schema, source hashes, code/config hash, cutoff policy, and
row-level exclusion reason under the new experiment namespace. The content hash
must cover all rows, not only a frame header and first 50 rows.

## Phase 2 — enforce feature/label isolation

Create typed prediction and outcome schemas. The model/inference frame must
contain only allowed pre-kickoff features. Outcome labels and same-week actuals
must live in a separate frame joined only inside evaluation.

Add denylist and lineage checks for same-week outcome fields, including:

- Actual snaps/routes/participation.
- Actual attempts/targets/carries/receptions.
- Actual shares.
- Actual yards/touchdowns/first downs.
- Actual fantasy points.
- Any column derived from them.

Lagged versions are allowed only when their source week is strictly earlier and
the feature contract proves the lag.

Add poisoned-target tests: changing target-week outcomes, actual shares, snaps,
or fantasy points must not change target-week input features or predictions.
Changing prior-week evidence may change them.

Add a test proving the active outcome label is independent of the `play_prob`
feature construction. A poisoned `play_prob` may affect prediction, but cannot
change the ground-truth label.

## Phase 3 — correct event targets and denominators

Define the v2 labels and denominators in code, metadata, and documentation:

```text
active_label:
  denominator = scheduled + rostered + observed active status

participated_label:
  denominator = active_label == true

positive_usage_label:
  denominator = participated_label == true
```

Determine participation from on-field offensive snaps/participation data, not
positive targets/carries alone where a snap source exists. Where the source is
missing, mark unknown rather than inventing a negative.

Determine positive usage from position-appropriate opportunities:

- QB: dropbacks/pass attempts and designed carries under a documented rule.
- RB: carries or targets.
- WR/TE: targets; optionally routes only for participation, not positive usage.

Do not mix event targets with conditional share regression targets.

Add tests using a miniature roster/game table containing every state above.
Verify exact denominator counts and that chained probabilities reconstruct
unconditional event rates without double counting.

## Phase 4 — establish honest deployable baselines

For every outer fold and event/position, compute baselines using training data
only. Include:

1. Training-fold constant prevalence.
2. A simple depth/status baseline fitted only on training rows.
3. The existing heuristic/live policy where it can be replayed historically
   without future information.

Persist baseline fit seasons, groups, smoothing priors, fallback behavior, and
hashes. Sparse group rates require training-only shrinkage; they must not peek at
the test fold.

`evaluate_event_predictions()` must require an explicit baseline vector or a
fitted baseline object. Remove the default that derives baseline prevalence from
`y_true`. Add a regression test that fails if test labels influence baseline
predictions.

Compare models primarily against the strongest valid deployable baseline, while
still reporting constant frequency for context. Do not choose an intentionally
weak comparator to obtain wins.

## Phase 5 — fit calibrated event models without posterior distortion

Start with simple per-position or hierarchical logistic models on the corrected
cohort. Remove `class_weight="balanced"` as the default because it changes the
posterior under high-prevalence labels and caused overprediction of rare zeros.
If weights are tested, correct/calibrate the resulting probabilities and compare
them honestly.

Candidate event approaches may include a small predeclared set:

- Training-frequency/depth-status baseline.
- Unweighted regularized logistic.
- Unweighted logistic plus training-only Platt or isotonic calibration when
  inner-fold sample size supports it.
- A conservative tree model only if it adds OOF value and remains calibratable.

Use nested expanding-window selection. Fit calibration on inner OOF predictions,
never the outer test fold. Report Brier, log loss, calibration error, reliability
bins, sharpness, prevalence, and sample counts.

Predeclare required event-cell and aggregate gates before viewing corrected
outer results. Preserve the previous majority-cell requirement unless a policy
change is independently justified before results; never weaken it after seeing
the rerun.

If an event model cannot beat the depth/status baseline, the baseline itself may
be selected as the production probability layer if it is calibrated and
properly versioned. Complexity is not a requirement.

## Phase 6 — wire fold-specific predictions into joint draws

Remove `_synthetic_team_game_from_rows()` from evaluation or convert it into a
strict prediction-input builder that accepts only:

- Fold-specific event predictions.
- Fold-specific point/team/volume/efficiency model predictions.
- Pre-kickoff schedule, roster, depth, status, odds/context, and identities.

It must reject actual/stat columns by schema. `PlayerGameInput.p_active`,
`p_participates`, and `p_positive_usage` must come from the exact fitted event
artifact/baseline selected for that fold. Team pass/rush totals and player shares
must come from out-of-fold model predictions, not actuals. `point_means` must be
the predicted mixture means, never realized fantasy points.

Actuals are joined only after the complete draw partition has been frozen and
hashed.

Add end-to-end tests with intentionally different event-model probabilities.
Prove the partition probabilities and zero mass change accordingly. Prove
constants 0.85/0.70, `or True`, or actual target-week stats cannot appear in the
candidate path.

## Phase 7 — rerun full rolling evaluation

Replace the single 2024 week-1 sample with every recoverable outer-fold game
within a bounded, documented evaluation population. Use enough draws for stable
metrics and report Monte Carlo error. A development sample may remain for fast
tests, but cannot power readiness.

For 2023–2025 outer folds:

1. Fit models/baselines on earlier seasons only.
2. Generate predictions using only pre-kickoff features.
3. Freeze and hash joint partitions.
4. Join actual outcomes after prediction.
5. Score event, component, distribution, conservation, correlation, and decision
   metrics.

Report event results by season/event/position/depth/status and aggregate. Report
zero-mass calibration by zero class, not only one global number.

Recalculate CRPS against the legacy baseline on the identical cohort and using
leakage-free inputs. The prior `2.84 vs 4.89` comparison is not valid evidence
because the joint path consumed same-week actuals; label it superseded rather
than repeating it.

Point-dispersion remains separately evaluated by the unchanged frozen point
policy. Do not derive a new point-gate verdict from joint draws.

## Phase 8 — readiness truthfulness

Harden `JointReadinessReport` so each gate requires its own evidence artifact
and provenance hash.

- Event gate: corrected OOF event metrics against training-only baselines.
- Distribution gate: leakage-free full OOF proper scores and zero-mass policy.
- Conservation gate: exact declared tolerance result; remove `OR < 50` bypass.
- PPFD gate: leakage-safe first-down historical evaluation plus exact live
  contract scoring.
- K/DST gates: historical baseline evaluation, not implementation presence.
- League scoring gate: exact live rule snapshots from the live shadow database,
  not fixture shapes relabeled with live IDs.
- Decision gate: completed full OOF lineup/matchup backtest.
- Artifact gate: schema, hash, input/model linkage, and no-outcome-at-prediction
  provenance.

A missing evidence artifact is a failed/incomplete gate. `auto_publish_allowed`
must remain false while the point gate or any required production gate fails.

## Phase 9 — exact six-league contract validation

Use the isolated live shadow database at the configured path—currently expected
under `output/live_shadow/`—to load the actual persisted `LeagueRuleSnapshot`
for all six Sleeper league IDs. Do not include manager/roster/private payloads in
committed reports.

Validate:

- Six exact distinct current rule snapshots and contract hashes.
- PPFD, bonuses, Superflex, K, and DEF rules from the real blobs.
- Four populated dynasty leagues through owner-roster decision APIs.
- Two pre-draft redraft leagues skipped for empty rosters.
- No fixture league or rule substitution.

If the shadow database is absent or stale, provide/run the existing explicit
GET-only shadow sync when authorized by its local config. If live access is not
available, keep exact-live scoring incomplete rather than mapping fixtures.

## Phase 10 — only then strengthen shared-game correlation

After the corrected event model and OOF wiring pass their own gates, measure
residual teammate/opponent dependence. The previous teammate correlation near
0.01 came from an evaluation path with invalid inputs and should not determine a
new latent strength directly.

Estimate leakage-safe residual correlation targets by game/relationship:

- QB–WR/TE.
- Competing same-team receivers/RBs.
- QB–opposing QB/pass catchers through game environment.
- Offense–kicker.
- Offense–opposing DST.

Tune one small, predeclared latent/residual structure on inner folds and validate
outer-fold correlation, CRPS, conservation, and decision metrics. Do not force a
positive teammate correlation if the empirical relationship is conditional on
position/game script. Never add arbitrary shared Gaussian noise solely to raise
correlation.

If event calibration still fails, stop with a valid no-go before latent tuning.

## Required tests

Add focused tests proving:

- Full roster rows survive when no box-score row exists.
- Bye/missing-source/roster-churn rows are not mislabeled DNP.
- Active outcomes do not derive from `play_prob`.
- Every event uses its correct conditional denominator.
- Test-fold prevalence cannot influence baseline predictions.
- Balanced weighting is not silently used.
- Actual same-week features are rejected from inference builders.
- Poisoning target-week actuals cannot change predictions or partitions.
- Fitted event probabilities—not constants—power game draws.
- Point means are predictions rather than actuals.
- Corrected partition hashes link to fold-specific inputs/models.
- Conservation has no arbitrary violation-count bypass.
- Live scoring validation cannot substitute fixture contracts.
- Missing validation evidence produces NO-GO.

Run focused tests, the full deterministic suite, feature/leakage audits,
blueprint audit, MVP verification, and vertical smoke. Keep full OOF evaluation
as a reproducible explicit workflow and use a small synthetic fold in CI.

## Required artifacts and report

Create:

- Versioned complete-roster cohort schema/artifact.
- Feature/outcome separation and lineage manifest.
- Training-only baseline artifacts per fold.
- Selected event-model/calibration artifacts per fold.
- Leakage-free OOF predictions and joint partitions.
- Corrected readiness evidence.
- `docs/WEEKLY_EVENT_COHORT_EVAL_REPAIR_REPORT.md`.

The report must include:

1. Each prior metric invalidated and why.
2. Corrected cohort construction and row-state counts.
3. Source completeness and limitations by season.
4. Label/denominator contracts.
5. Feature/target leakage tests.
6. Training-only baseline definitions and hashes.
7. Event-model OOF metrics versus both valid baselines.
8. Evidence that fitted probabilities power the draws.
9. Corrected leakage-free CRPS, zero mass, correlation, and conservation.
10. Exact live six-league scoring result or explicit blocker.
11. Whether shared-latent tuning was authorized by the event gate.
12. All commands/tests, skips, and failures.
13. Separate go/no-go decisions for point model, event layer, joint draws,
    manual shadow, automatic publication, start/sit, and internet deployment.

Link the new report from `WEEKLY_JOINT_USAGE_DRAWS_REPORT.md` without deleting or
rewriting the earlier no-go evidence.

## Valid completion outcomes

### Corrected event-layer GO

- Full roster-week cohort and outcomes are correct and leakage-safe.
- Training-only baselines are valid.
- Selected event probabilities beat the predeclared baseline gates OOF.
- Those exact probabilities power leakage-free joint draws.
- Zero-mass and proper-score evidence is recalculated on the full OOF cohort.
- Only then may shared-latent evaluation proceed.
- Publication/start-sit remain governed by every separate outstanding gate.

### Valid no-go

- All evaluation-integrity defects are fixed.
- The corrected simple event candidates fail the predeclared gates.
- No stronger game-latent tuning is attempted on an invalid event layer.
- Active releases and publication flags remain unchanged.
- The report identifies the narrowest remaining data/target/model limitation.

Do not optimize toward the old 0/21 or 0.252 numbers. Those came from an invalid
baseline and leaked draw inputs. First produce trustworthy evidence; improvement
is secondary.

## Final response

Lead with the seven separate go/no-go decisions. Then state:

1. Which prior metrics were invalidated.
2. How the full roster cohort changed zero/event prevalence.
3. How labels, denominators, and training-only baselines were corrected.
4. The selected event layer or evidence-backed no-selection.
5. Corrected outer-fold event, CRPS, zero-mass, correlation, conservation, and
   point metrics.
6. Whether shared-latent tuning was reached.
7. Exact live scoring validation status.
8. Tests and commands run, including external blockers.
9. Why automatic publication and start/sit remain enabled or disabled.
10. Files changed by cohort/data, models, draws, evaluation, readiness, tests,
    and documentation.

Never claim the event model improved based on a test-prevalence oracle baseline.
Never claim joint-draw improvement from a partition built with same-week actual
inputs.
