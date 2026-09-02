# Cursor weekly-v2 volume tuning and nested evaluation prompt

Copy everything below the divider into Cursor Agent mode from the repository
root after the weekly-v2 training/promotion pass and its latest calibration
evaluation are present.

---

You are the senior ML engineer responsible for diagnosing and correcting the
weekly-v2 model's remaining under-dispersion through leakage-safe volume-model
tuning and retraining.

This is an implementation, experimentation, and evaluation task. Do not weaken
the promotion gate, keep increasing calibration caps, or stop after producing a
tuning JSON file. Repair the tuning harness, determine the actual source of
compressed predictions, run an honest nested/rolling comparison, retrain the
2026 candidate with the selected configuration only if justified, and leave
automatic publication disabled unless *all* applicable gates pass.

## Current result and frozen policy

The latest calibrated strict evaluation exits with code 2. Promotion correctly
fails:

```text
2023 calibrated dispersion: 0.689
2024 calibrated dispersion: 0.688
minimum policy dispersion:   0.700
```

The WR/TE calibration slope cap is currently 1.52. A prior 1.60 cap caused
unacceptable rank behavior. Calibration-only adjustment is now frozen for this
pass. Do not change any of the following merely to make the candidate pass:

- `min_dispersion_ratio = 0.70`
- `max_dispersion_ratio = 1.30`
- `min_coverage = 0.95`
- `min_mae_improvement = 0.02`
- `min_rank_improvement = 0.0`
- `min_interval_coverage = 0.72`
- `max_interval_coverage = 0.90`
- WR/TE slope cap of 1.52

Do not add tolerance, rounding, alternate populations, selective season
exclusions, position exclusions, or post-hoc overrides around the 0.70 floor.
The fact that the miss is approximately 1.6% does not authorize a policy change.

The current five decisions remain:

- Trained artifact classification: GO with caveats.
- Manual trained shadow publication: NO-GO.
- Automatic weekly publication: NO-GO.
- Use for start/sit decisions: NO-GO.
- Public-internet deployment: NO-GO.

## Important defects in the current tuning path

Inspect and fix these before trusting any tuning result:

1. `scripts/weekly_v2_tune_preseason.py` imports
   `scripts.preseason_eval` / `preseason_eval`, but the current evaluation
   implementation is `scripts/weekly_v2_evaluate.py`. The tuner must use the
   exact same cohort, as-of rules, projection path, metrics, and calibration
   implementation as the production evaluation.
2. Candidate eligibility currently disables the dispersion bounds by passing
   `min_dispersion_ratio=0.0` and `max_dispersion_ratio=999.0`.
3. Candidate ranking currently optimizes only relative MAE gain. It can select
   a model that leaves the known dispersion problem unchanged or worse.
4. `weekly_v2_evaluate.py` accepts `volume_options` in `evaluate_season()` but
   its CLI/main path does not load or accept the tuning selection. A selection
   file can therefore exist without affecting the reported promotion result.
5. `weekly_v2_train.py` reads `MODELS_DIR/tuning_selection.json`, changes an
   environment variable after importing path constants, writes through global
   registry paths, and later moves matching files. This is fragile and can use
   a stale or wrong tuning artifact.
6. The tuner writes a single mutable `tuning_selection.json` without enough
   code/data/candidate/fold provenance to prove what it selected.
7. The same seasons must not both select a candidate and then masquerade as an
   untouched holdout evaluation of that selection.

Add regression tests that fail against each defective behavior before or while
fixing it.

## Authoritative files to read completely

Before editing, read:

1. `docs/WEEKLY_V2_TRAINING_AND_PROMOTION_REPORT.md`
2. `output/weekly_v2/preseason_backtest.json`
3. `output/weekly_v2/preseason_oof.parquet` through a schema/statistics audit
4. `scripts/weekly_v2_tune_preseason.py`
5. `scripts/weekly_v2_evaluate.py`
6. `scripts/weekly_v2_train.py`
7. `scripts/weekly_v2_fit_calibration.py`
8. `src/projection/weekly/models/volume.py`
9. `src/projection/weekly/models/base.py`
10. `src/projection/weekly/models/calibration.py`
11. `src/projection/weekly/evaluate/preseason.py`
12. Volume feature construction, accounting, availability, veteran, rookie,
    and season projection modules
13. Weekly inference, league scoring, draw, readiness, and promotion code under
    `src/app/projections/` and `src/app/releases/`
14. Relevant weekly, leakage, accounting, and false-green tests
15. The original tuner/evaluator and their tests in
    `../fantasy-projections-2`, only as provenance—not as runtime dependencies

Read current git status and preserve every existing tracked/untracked user
change. Do not mutate current trained artifacts, evaluation evidence, sealed
releases, active pointers, or live shadow database. Run experiments under a new
namespaced directory.

## Phase 1 — reproduce and decompose under-dispersion

Reproduce the exact latest 0.689/0.688 result before changing model code. Record
the command, hashes, configuration, calibration caps, and output path.

Then decompose dispersion and error for each outer season and position by:

- Weekly fantasy points before and after calibration.
- Season-aggregated fantasy points.
- Predicted versus actual standard deviation.
- Predicted and actual quantiles.
- Zero/DNP mass versus positive-usage distribution.
- Starter, rotational, deep-roster, and unlisted depth bands.
- Rookie versus veteran.
- Healthy/active versus availability-limited observations.
- Team and positional-room concentration.
- Each material component: attempts, carries, targets, snaps, receptions,
  yards, touchdowns, and shares.
- Participation probability and conditional positive-usage prediction.
- Before and after lag-prior blending.
- Before and after team composition/accounting/depth adjustments.

Attribute the 2023/2024 shortfall quantitatively. Identify whether compression
originates primarily in:

- Participation classification.
- Conditional volume regression.
- Probability × conditional-share multiplication.
- Recency weighting.
- Lagged-prior blending.
- Team composition normalization.
- Depth-share damping or caps.
- Team totals.
- Efficiency regression.
- Availability being counted twice.
- Cohort/evaluation construction.
- Position-level calibration.

Do not assume “volume model” means only the estimator hyperparameters. Trace one
row through every transformation.

Create a diagnostic artifact containing fold/position/depth-band metrics and
transformation-stage dispersion. Do not persist private live-roster data in the
artifact; historical public training/evaluation data is the intended input.

## Phase 2 — verify probability semantics and double-counting

The current `TwoStageVolumeModel` computes expected share approximately as:

```text
P(positive usage) × E(share | positive usage)
```

The broader pipeline separately carries injury/play probability and availability
logic. Determine exactly what each probability represents and whether player
non-participation is applied twice.

Define separate contracts for:

- Available for the game.
- Active but receives no usage.
- Participates in snaps/dropbacks.
- Receives positive target/carry/red-zone usage.
- Conditional usage share given the relevant participation event.

For expected fantasy points, probability mixtures may legitimately affect the
mean. For simulated draws, the event should usually be sampled rather than
flattened into every draw. Do not remove a probability solely to inflate
dispersion. Correct it only if the same event is duplicated or the target/event
definition is wrong.

Add hand-checked tests for a healthy starter, questionable starter, healthy
reserve, and inactive player. Ensure expected means and draw mixtures are
coherent and no availability fact is charged twice.

## Phase 3 — repair the experimental harness

Create one authoritative evaluation library used by tuning, standalone
evaluation, training selection, and final promotion. CLI modules should be thin
wrappers around it.

The harness must accept explicit:

- Data panel path/hash.
- Outer target seasons.
- Inner selection seasons/folds.
- Scoring configuration.
- Candidate specification file.
- Calibration policy and frozen caps.
- Random seed.
- Artifact namespace/output directory.
- Resume/cache behavior keyed by full input/config/code fingerprint.

It must persist each candidate's complete configuration, code/data hashes,
fold-specific training seasons, model hashes or cache keys, metrics, runtime,
warnings, and failures. Never overwrite a prior experiment.

Pass the selected volume configuration explicitly into final training. Do not
rely on process-global environment changes, implicit root files, or moving
artifacts after training. Update the model registry/training functions to accept
an explicit output directory or registry object throughout.

The final model manifest must embed the selected tuning artifact hash and exact
volume configuration. Manifest verification must fail if the tuning artifact is
missing, altered, selected on disallowed folds, or inconsistent with model
metadata.

Add tests proving:

- Tuner and evaluator call the same evaluation function.
- A selected option changes the actual trained model metadata and predictions.
- Stale root-level selection files are ignored.
- Explicit candidate output paths do not leak into one another.
- Changing a candidate configuration invalidates cached results.
- Dispersion is not disabled or omitted from selection evidence.

## Phase 4 — establish an honest selection protocol

Because 2022–2025 results and calibration behavior have already been inspected,
none of those seasons should now be described as a pristine untouched holdout.
Use nested expanding-window evaluation to obtain an honest generalization
estimate:

- Each outer target season is predicted using only strictly earlier seasons.
- Within an outer fold, select architecture/hyperparameters using only inner
  folds strictly earlier than the outer target.
- Calibration for the outer fold uses only predictions/residuals available from
  earlier folds.
- Record warm-up folds that cannot support inner selection; do not score them as
  if selected out of sample.
- Aggregate outer-fold metrics only after every candidate/selection rule is
  frozen.

For training the 2026 model, selection may use the full completed 2016–2025
history, but the claimed performance estimate must come from the nested outer
procedure—not from evaluating the final selected model on seasons that directly
selected it.

Predeclare the candidate grid and deterministic selection rule before reviewing
new candidate results. Keep the grid finite and technically motivated. Do not
launch an unbounded hyperparameter search against four visible seasons.

## Phase 5 — candidate families

Evaluate the current model as an immutable baseline, then a modest grid covering
the most plausible volume causes. Candidate dimensions may include:

- Two-stage participation/conditional architecture versus legacy direct only as
  a controlled comparison.
- Recency half-life: all-history and a small predeclared set such as 2, 4, and 6
  seasons.
- Conditional regressor family: current HGB, conservative alternate HGB
  regularization/depth, and ridge where justified.
- Participation classifier family: current HGB and logistic/ridge baseline.
- Participation threshold and minimum classifier rows only if diagnostics show
  current definitions are inappropriate.
- Lagged-prior blend weights for target/carry share using a small simplex grid.
- Current composition/accounting settings versus one or two diagnosed,
  conservation-safe alternatives.

Do not vary calibration caps in this search. Do not search dozens of highly
correlated knobs. Do not alter depth/accounting constants unless the stage
decomposition shows they materially create compression; if varied, preserve
team conservation and test all positions.

Expose HGB/regression parameters through typed model configuration rather than
hidden module constants. Persist them in model metadata. Use deterministic
random states.

## Phase 6 — multi-objective candidate selection

Candidate selection cannot optimize MAE alone. Use a predeclared, transparent
lexicographic or Pareto rule that considers:

1. Required cohort coverage and absence of leakage/schema failures.
2. MAE improvement versus the frozen baseline.
3. Rank correlation/non-degradation.
4. Dispersion proximity to 1.0 without exceeding the frozen `[0.70, 1.30]`
   policy range.
5. Interval coverage within `[0.72, 0.90]` after nested calibration.
6. No catastrophic regression by position or decision-relevant depth band.
7. Season-to-season stability and reasonable complexity.

Do not allow a candidate to compensate for a failed 2023 or 2024 dispersion
fold with a strong 2025 average. Promotion remains an all-required-fold gate.

Persist the full Pareto table, not only the winner. Report effect sizes and
uncertainty/bootstrap intervals where practical. If no candidate dominates the
current baseline and passes the full policy, select none and return a valid
no-go.

## Phase 7 — investigate calibration/rank interaction without retuning it

Keep the 1.52 caps fixed, but explain why the earlier 1.60 cap reportedly caused
rank blow-ups. A positive linear transformation within one position should
preserve within-position ordering. Determine whether the observed problem was:

- Cross-position overall ranking shifts.
- Position-dependent intercepts/slopes.
- Zero clipping creating ties.
- A metric/join/cohort bug.
- Calibration applied multiple times.
- Season-level aggregation after weekly calibration.
- Something else.

Add invariants that calibration is applied exactly once, preserves within-position
order except documented zero ties, and uses only earlier OOF residuals. This is
diagnostic hardening, not permission to resume slope-cap searching.

## Phase 8 — retrain and reevaluate the 2026 candidate

Only if nested selection chooses a candidate:

1. Train a new immutable 2026 candidate under a new directory using completed
   seasons through 2025.
2. Fit the 2026-safe calibration from strict prior OOF rows with frozen caps.
3. Write a new manifest containing candidate-selection provenance and hashes.
4. Validate clean-process loading and feature schema.
5. Generate new week-1 projections and draw partitions.
6. Re-run the strict promotion evaluation using the honest nested evidence.
7. Run inference-versus-fallback, league scoring, identity, conservation,
   partition, and failure-injection tests.

Do not replace the previous candidate or evidence. Compare them side by side.

Even if the dispersion gate passes, do not automatically set the recurring
publisher live. The prior report still lists unresolved readiness work:

- PPFD first-down statistics are not yet modeled.
- Teammate conservation gates for stat draws remain open.
- Kicker and DST weekly models are not integrated into readiness.
- Start/sit draw distributions are not fully validated.
- PostgreSQL/Docker deployment remains unverified.

The operations API may report the newly evaluated artifact state, but
`auto_publish_allowed` must represent all required gates, not dispersion alone.

## Required verification

Run all new focused tests plus the full deterministic suite:

```text
uv run pytest -q
uv run python scripts/audit_weekly_features.py
uv run python scripts/verify_mvp.py
uv run python scripts/audit_blueprint_mvp.py
uv run python scripts/vertical_smoke.py
```

Run the repaired tuner with explicit nested folds, candidate spec, seed, and
new output namespace. Then run training/evaluation/project commands with the
selected config only if selection succeeds.

Do not make CI execute the full expensive tuning search. Add a tiny synthetic
nested-CV fixture that proves selection, isolation, cache invalidation,
dispersion-aware ranking, and no-future-fold behavior.

## Required report

Create `docs/WEEKLY_V2_VOLUME_TUNING_REPORT.md` containing:

1. Reproduced baseline and exact hashes.
2. Transformation-stage diagnosis of under-dispersion.
3. Probability/availability semantics and any double-counting finding.
4. Tuning harness defects found and fixes made.
5. Frozen candidate grid and selection rule established before results.
6. Inner/outer fold design and leakage evidence.
7. Full candidate Pareto table by fold and position.
8. Selected candidate—or explicit no-selection—with rationale.
9. Comparison against the current model for MAE, rank, dispersion, interval
   coverage, depth bands, and decision metrics.
10. The 2023 and 2024 dispersion results to at least four decimal places.
11. Calibration ordering/invariant findings.
12. Final artifact/model/output hashes if retrained.
13. Separate go/no-go decisions for artifact classification, manual trained
    shadow publication, automatic publication, start/sit use, and internet
    deployment.
14. Remaining PPFD, draw-conservation, K/DST, infrastructure, and external
    blockers.

Update `docs/WEEKLY_V2_TRAINING_AND_PROMOTION_REPORT.md` only with a concise
linked follow-up status. Do not rewrite prior failed evidence as if it never
happened.

## Completion outcomes

### Valid tuned outcome

- The tuning/evaluation/training path is authoritative and reproducible.
- Nested selection chooses a model without future-fold leakage.
- New outer-fold results pass all frozen promotion thresholds, including every
  required season's dispersion.
- Accuracy, rank, calibration, coverage, and important segments do not regress
  beyond policy.
- The final 2026 candidate embeds the exact tuning provenance and generates real
  inference.
- Automatic publication remains disabled until the other open readiness gates
  are completed and explicitly reviewed.

### Valid no-go outcome

- The harness and selection methodology are corrected.
- Under-dispersion is quantitatively localized.
- Every reasonable predeclared candidate is evaluated honestly.
- No candidate passes the frozen policy without unacceptable regressions.
- The current active/fallback state is unchanged and publication remains
  blocked.
- The report identifies the next architectural change supported by evidence,
  rather than recommending another calibration-cap search.

Do not stop at “the tuner ran.” The task ends only with an honest selected
candidate plus nested evidence, or an evidence-backed no-selection.

## Final response

Lead with the five go/no-go decisions. Then report:

1. The root cause of under-dispersion.
2. The tuning harness defects fixed.
3. The selected candidate and exact configuration, or why none was selected.
4. Fold/position metrics versus baseline, especially 2023 and 2024 dispersion.
5. Whether a new 2026 candidate was trained and its manifest hash.
6. Tests and experiment commands run, including failures and runtime blockers.
7. Why automatic publication and start/sit are still enabled or disabled based
   on the complete gate set.
8. Files changed, grouped by diagnostics, tuning, model, evaluation, tests, and
   documentation.

Never describe a candidate as promoted merely because its overall average
improved or because it crossed 0.70 after seeing and tuning directly against the
same fold. Preserve the evidence trail and gate integrity.
