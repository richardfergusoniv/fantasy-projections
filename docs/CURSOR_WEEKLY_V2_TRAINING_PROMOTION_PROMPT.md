# Cursor weekly-v2 training, inference, and promotion prompt

Copy everything below the divider into Cursor Agent mode from the repository
root after the live Sleeper shadow-beta changes are present.

---

You are the senior ML/platform engineer responsible for replacing the app's
weekly fixture bridge with leakage-safe, trained, executable 2026 weekly
projections and an honest promotion gate.

This is an implementation and evaluation task, not a request for scaffolding or
a plan. Inspect the existing code and artifacts, port or rebuild the missing
training/evaluation entrypoints, train models in this repository when the data
supports it, run strict historical evaluation, wire actual model inference into
the application, and promote only if every gate passes. If the model fails a
gate, leave automatic publication disabled and deliver a precise no-go report.
Never lower thresholds, relabel a fallback, or manufacture an artifact merely
to obtain a green result.

## Current verified state

- Live read-only Sleeper shadow beta passed for the configured owner and six 2026
  leagues.
- Player identity reconciliation resolved 1,320 of 1,320 rostered IDs with no
  unresolved or ambiguous starters.
- All six live scoring contracts compile.
- The four in-season dynasty leagues exercise lineup, waiver, trade, and
  dynasty APIs.
- The two redraft leagues are currently `pre_draft` with empty rosters.
- Weekly readiness currently reports `weekly_v2_state: fixture` and
  `auto_publish_allowed: false`.
- PostgreSQL, Docker, email, OpenAI, and public deployment remain outside this
  task unless already locally available.

Owner-confirmed dynasty rules and live integration behavior must remain
unchanged.

## Critical false-green risk to fix first

Do not assume this task is complete when nine `.joblib` files appear.

`src/app/projections/weekly_v2_bridge.py` currently classifies a run as
`trained` primarily from the presence of these files plus a schema-compatible
manifest:

```text
team_totals.joblib
volume_QB.joblib
volume_RB.joblib
volume_WR.joblib
volume_TE.joblib
efficiency_QB.joblib
efficiency_RB.joblib
efficiency_WR.joblib
efficiency_TE.joblib
```

However, `src/app/projections/weekly_run.py` currently calls `_weekly_rows()`,
which scales sealed preseason mean/quantile points with a deterministic hash
factor. It does not invoke `project_veterans_week()` or
`project_week_with_rookies()` and does not consume a real weekly-v2 output.
Therefore the current gate could report “trained” while publishing derived
preseason values. Treat this as a release-blocking S1 defect.

Readiness must require compatible, provenance-verified artifacts *and* the
candidate must be constructed from the output of those exact artifacts.
Publication metadata and tests must prove that linkage.

## Authoritative context to read completely

Before editing, read:

1. `docs/LIVE_SLEEPER_BETA_REPORT.md`
2. `docs/PRODUCTION_READINESS_AUDIT.md`
3. `docs/APP_IMPLEMENTATION_BLUEPRINT.md`
4. `docs/APP_DATA_CONTRACTS.md`
5. `docs/APP_OPERATIONS_RUNBOOK.md`
6. `docs/WEEKLY_V2_PORT_PROVENANCE.md`
7. `STATE_OF_BUILD.md` and `docs/PIPELINE_MAP.md`
8. Existing projection decisions under `docs/decisions/`
9. `src/app/projections/weekly_v2_bridge.py`
10. `src/app/projections/weekly_run.py`
11. `src/app/releases/`, projection persistence models, and decision services
12. Everything under `src/projection/weekly/`
13. `src/projection/weekly_audit/` and relevant feature-contract tests
14. `src/projection/special_teams/` and the league scoring compiler
15. `pyproject.toml`, `uv.lock`, CI, and current tests

The sibling repository `../fantasy-projections-2` may be inspected as a source
of the original weekly-v2 implementation. In particular, inspect completely:

- `scripts/ingest_data.py`
- `scripts/build_features.py`
- `scripts/tune_preseason.py`
- `scripts/train.py`
- `scripts/preseason_eval.py`
- `scripts/walkforward_eval.py`
- `scripts/fit_calibration.py`
- `scripts/evaluate.py`
- `scripts/project.py`
- `scripts/project_season.py`
- `src/projections/evaluate/`
- `src/projections/models/selection.py`
- `models/training_manifest.json`
- `outputs/preseason_backtest.json`
- Relevant tests and model metadata

Port required code with provenance and import-path updates. The resulting app,
training pipeline, evaluation, and artifacts must not depend at runtime on the
sibling repository.

## Existing sibling artifacts

The sibling repository currently contains locally trained model files and a
manifest created on 2026-08-23. Its manifest claims training seasons 2016–2025
and fingerprints a roughly 49.8 MB player-week panel. It includes team totals,
four volume models, four efficiency models, four rookie models, calibration,
and tuning selection artifacts.

Treat these as candidate evidence, not automatically trusted production
artifacts:

- Verify every recorded hash against the actual file.
- Verify feature schemas, training cutoffs, target definitions, library
  compatibility, and evaluation provenance.
- Do not load untrusted pickle/joblib data. These local sibling artifacts may be
  inspected only after confirming their expected provenance; deployed code must
  never accept arbitrary uploaded joblib files.
- Joblib objects serialized under the old `projections.*` module namespace may
  not be portable to `src.projection.weekly.*`. Do not add the sibling package
  as a production dependency or retain a compatibility import solely to make an
  old pickle load.
- Prefer reproducibly retraining under the current namespace from verified
  inputs. A controlled artifact migration is acceptable only if equivalence and
  runtime independence are proven.
- Do not copy absolute paths from the sibling manifest into the new manifest.

## Working-tree and data safety

- Inspect `git status`, tracked diff, and untracked files first.
- Preserve all existing user work and model/release artifacts.
- Do not reset, revert, stash, or broadly delete files.
- Never modify a sealed historical release bundle or active pointer in place.
- Train and evaluate in a new namespaced candidate directory.
- Do not commit large raw datasets, trained weights, caches, or generated
  projections unless the repository's artifact policy explicitly requires it.
- Keep generated artifacts content-addressed or accompanied by cryptographic
  hashes and provenance manifests.
- Never train on 2026 regular-season outcomes for a 2026 preseason/week-1
  candidate. The current date is 2026-08-31; only information legitimately
  available at each historical prediction timestamp may enter its fold.
- Do not enable automatic publication until all gates below pass.
- Do not call paid services or require `CFBD_API_KEY` without explicit existing
  configuration. Record degraded rookie inputs instead of fabricating them.

## Phase 1 — establish artifact and pipeline inventory

Create a machine-readable inventory of:

- Every weekly-v2 model/output file currently in this repo and the sibling.
- File hash, size, model type, serialization namespace, metadata, target
  columns, feature columns, training seasons, and creation time.
- Required versus optional artifacts.
- Data inputs and cache fingerprints.
- Current training, calibration, evaluation, projection, and application
  publication entrypoints.
- Every place `weekly_v2_state`, `artifact_mode`, `model_version`, or
  `auto_publish_allowed` is computed or displayed.

Prove the exact reason the app reports nine missing files. Identify whether the
rookie, calibration, tuning, and market artifacts are required for weekly
publication, optional/degraded, or unrelated. Do not leave contradictory
definitions of readiness across API, jobs, UI, or scripts.

Add a regression test showing that merely placing nine dummy files and a
minimal manifest cannot make a candidate publishable or label hash-scaled
preseason rows as trained.

## Phase 2 — restore reproducible training and evaluation commands

Port or implement first-class current-repo commands for:

1. Historical public-data ingestion.
2. Leakage-safe player-week panel construction.
3. Rolling/expanding-window tuning.
4. Strict out-of-fold preseason and in-season evaluation.
5. Final training through the last completed season.
6. Point and interval calibration using earlier out-of-fold predictions only.
7. Weekly projection generation.
8. Season/ROS projection generation where weekly-v2 supports it.
9. Candidate manifest creation and validation.

Use clear command names such as `scripts/weekly_v2_ingest.py`,
`scripts/weekly_v2_build_features.py`, `scripts/weekly_v2_train.py`,
`scripts/weekly_v2_evaluate.py`, and `scripts/weekly_v2_project.py`, or integrate
equivalent subcommands into the existing CLI. Preserve helpful sibling behavior
without copying stale import paths or assumptions.

Resolve the `nflreadpy` versus `nfl-data-py` mismatch explicitly. Choose one
maintained, locked ingestion path compatible with this repository, or implement
a narrow adapter. Do not leave a hidden optional import that fails only during
production refresh. Maintain required source attribution for nflverse,
ffopportunity, and FTN-derived inputs.

All commands must:

- Accept explicit seasons, weeks, as-of timestamps, input directories, output
  directories, and random seeds.
- Print and persist the resolved configuration.
- Fail on missing required columns or incompatible schemas.
- Be deterministic within documented numerical tolerance.
- Never overwrite a promoted artifact directory.
- Write candidates to a temporary/new directory and finalize atomically.
- Return nonzero on validation failure.
- Produce useful `--help` and runbook examples.

## Phase 3 — leakage audit before training

Perform a field-level leakage audit of the actual panel used for every target.
Do not rely solely on function names or the existing audit script.

For a prediction at `(season, week, kickoff)`, prove:

- Same-week and future box-score outcomes are unavailable.
- Rolling features use shifted history.
- Injury, roster, depth, transactions, and player status use observed-at/as-of
  snapshots no later than the prediction cutoff.
- Vegas and market inputs use archived as-of snapshots, not closing or current
  values unavailable at prediction time.
- Season-level durability uses only prior seasons.
- Rookie college, combine, draft-capital, and landing-spot features were known
  by the historical cutoff.
- Calibration and hyperparameter selection use only prior folds.
- Player identity reconciliation cannot join a future team/role backward.
- Post-kickoff corrections cannot revise frozen pregame evaluation rows.
- Current 2026 injury/news data cannot contaminate historical training rows.

Add automated temporal assertions with deliberately poisoned future values.
They must prove outputs for an earlier cutoff do not change. Persist a feature
contract listing source, grain, observed-time column, allowed lag, target usage,
and licensing/attribution.

Any unresolved leakage on a material feature is a no-go for promotion.

## Phase 4 — train the 2026 candidate reproducibly

The intended 2026 candidate should use completed seasons 2016–2025 where the
source/feature is legitimately available. Document narrower availability by
feature and model. The target season 2026 must be excluded.

Train at minimum:

- Team opportunity/totals model.
- Position-specific volume models for QB, RB, WR, and TE.
- Position-specific efficiency models for QB, RB, WR, and TE.

Train rookie models for QB/RB/WR/TE when verified inputs support them. If CFBD
data is unavailable, distinguish among cached verified college inputs, a
documented no-CFBD model, and missing rookie readiness. Never silently use
today's prospect data in historical folds.

Set and persist deterministic seeds. Capture exact library versions, code
revision or working-tree hash, input hashes, feature lists/order/dtypes, target
definitions, missing-value policies, sample counts by season and position,
hyperparameters, training ranges, and model hashes.

The final manifest must be a new versioned schema, for example:

```text
output/weekly_v2/models/season=2026/manifest.json
```

It must use relative/content-addressed artifact references, not machine-specific
absolute paths. Include a `model_version` derived from a stable manifest/input
fingerprint. The manifest must fail validation if a model or metadata file is
missing, extra, changed, stale, trained through 2026, schema-incompatible, or
unloadable in a clean current-repo process.

Loading joblib artifacts is allowed only from the configured immutable model
store after manifest hash verification and safe path validation.

## Phase 5 — historical evaluation and promotion thresholds

Run strict expanding-window out-of-fold evaluation across every recoverable
target season, with emphasis on 2022–2025. Never evaluate on rows used to fit
that fold's model, calibration, or hyperparameters.

Report by season and position, plus overall:

- Recoverable cohort coverage including DNP zeros.
- MAE and RMSE for fantasy points and material component stats.
- Rank correlation.
- Top-12/top-24/top-36 identification or equivalent starter-relevant metrics.
- Calibration error and interval coverage for published quantiles.
- Bias by position, depth band, rookie/veteran, injury/availability band, and
  projection magnitude.
- Team-level conservation/accounting residuals.
- Stability under repeated seeded runs.
- Decision-relevant lineup regret and matchup probability calibration where
  feasible.

Compare against meaningful frozen baselines:

- Prior-season/per-game or rolling public baseline.
- Existing production v1/sealed preseason projection where temporally valid.
- A simple position/depth baseline.
- Market/ECR only when archived as-of snapshots provide a common leakage-safe
  cohort.

Reuse previously approved thresholds only if their definition and population
remain identical. Otherwise define thresholds before viewing final holdout
results and document the rationale. Promotion should require material coverage,
no catastrophic segment regression, acceptable calibration, conservation, and
improvement or justified non-inferiority on decision-relevant cohorts.

Do not select the final model on 2025 and then describe 2025 as untouched
holdout. If 2025 affects selection, label it validation and retain a genuinely
separate test mechanism or use nested/rolling selection honestly.

If the model fails, do not tune on the final holdout repeatedly. Produce a
no-go report with the failed segments and keep fixture/fallback state.

## Phase 6 — generate real weekly projections

Wire a service that executes weekly-v2 inference for a specified season, week,
and as-of timestamp using the verified manifest. It must:

- Build or load the correct as-of feature panel.
- Invoke the trained team-total, volume, efficiency, and applicable rookie
  models.
- Apply availability and depth-chart inputs known by the cutoff.
- Enforce team/player accounting constraints.
- Produce stable canonical player IDs compatible with live Sleeper rosters.
- Include opponents, kickoff times, byes, teams, positions, and availability.
- Emit component stat expectations, not only generic PPR fantasy points.
- Emit uncertainty/draw information sufficient for league scoring and matchup
  simulation.
- Persist an immutable output with input/model/feature hashes and observed time.
- Fail on feature drift, missing material teams/starters, or partial inputs.

The candidate app path must consume this real output. Delete or retain the hash
scaling function only as explicitly named test/fallback behavior; it may never
run under `artifact_mode=trained`.

Add an end-to-end test using tiny deterministic trained models whose prediction
is intentionally different from the preseason scaling result. Prove the
published candidate contains the model output, records the exact model/output
hashes, and changes when the test model changes. Also prove missing/corrupt
weights, feature-schema drift, or mismatched output provenance blocks promotion.

## Phase 7 — league-specific scoring and simulation

Do not publish one pre-scored PPR number to all six leagues. Weekly-v2 should
produce a football-stat distribution; the league scoring layer should apply
each compiled Sleeper scoring contract.

Verify or implement:

- Standard, half/full PPR, and all live offensive rules.
- Points per first down.
- Yardage thresholds/bonuses.
- Passing/rushing/receiving/fumble components.
- Superflex eligibility in lineup construction.
- Kicker and team-defense projections from the documented simplified models.
- Correct handling of negative and overlapping scoring rules.

Nonlinear rules such as bonuses and PPFD must be calculated at draw/event level,
not by applying thresholds only to the mean. If weekly-v2 does not directly
model first downs, introduce a documented, leakage-safe conditional rate model
with uncertainty, or keep PPFD publication blocked. Do not invent constant
points-per-first-down adjustments without evaluation.

DST and kicker models may remain simpler and more uncertain, but they must use
real current team/opponent inputs and be labeled separately from skill-position
weekly-v2 readiness.

For each of the six live league rule contracts, generate a scored shadow
candidate and verify that at least one intentionally different scoring rule
produces the expected different player distribution/ranking. Retain league
contract hashes in the run provenance.

## Phase 8 — draw quality and matchup probabilities

The application uses projections for start/sit and matchup win probability, so
means alone are insufficient. Produce or calibrate player/team draws that:

- Preserve relevant teammate/opponent correlation.
- Respect team opportunity/stat conservation.
- Represent availability as a separate event where appropriate.
- Do not independently sample mutually exclusive roles without reconciliation.
- Reproduce calibrated marginal intervals from out-of-fold residuals.
- Are deterministic for the same run seed and partition.
- Remain stable enough that recommendations do not flip excessively at the
  configured 10,000-draw count.

Run existing simulation partition, draw-count, calibration, conservation, and
decision-change gates. Do not reuse season-long uncertainty artifacts unless
their grain and target match weekly output. Store partition hashes and prove
that incomplete/mismatched partitions block promotion.

## Phase 9 — honest readiness and atomic promotion

Replace presence-based readiness with a verified contract. A weekly candidate
may be `trained` and automatically publishable only when all are true:

1. Manifest schema and target season are supported.
2. All required artifacts and metadata exist under safe paths.
3. Every hash matches.
4. Training cutoff is strictly before the target season/week as required.
5. Feature schema and runtime library compatibility match.
6. Models load in a clean process.
7. A real weekly output was generated by those exact model hashes.
8. Output provenance, coverage, freshness, identity, and league scoring pass.
9. Evaluation/promotion report has an explicit passing decision.
10. Draw partitions and quantile calibration pass.
11. No fallback or derived-preseason code path contributed rows.

The API, operations screen, worker, CLI, and publication service must use the
same readiness result. `auto_publish_allowed` must reflect the entire contract,
not merely artifact presence.

Publish through the existing immutable candidate, validation, pointer-swap, and
rollback workflow. First run an isolated shadow promotion. Inject failures
before output completion, after output completion, during gates, and before
pointer swap. Prove the active pointer never exposes a partial/failed candidate.

Do not enable the recurring production schedule simply because a manual shadow
candidate passes. Create a deliberate configuration flag whose safe default is
off, and document the exact final operator action. If all training/evaluation
and shadow gates pass, the report may recommend enabling it; do not silently
change the user's production environment.

## Phase 10 — application decision validation

Using the trained shadow candidate and live shadow database, exercise the four
in-season dynasty leagues:

- Legal lineup recommendations.
- Current-opponent and optimized-opponent win probability.
- Roster-aware waivers and FAAB where applicable.
- Both-side trade impact and fairness.
- Dynasty state and future-pick valuation.

Confirm the two pre-draft redraft leagues remain correctly skipped until
rosters populate. Do not seed fake live rosters to obtain a green result.

Compare fixture/fallback versus trained recommendations and report material
changes. Manually inspect high-impact anomalies: implausible leaders, inactive
players, wrong teams, missing starters, extreme DST/K values, zero projections,
and lineup choices dominated by scoring-contract errors.

The app must display model version, run freshness, trained/fallback state,
uncertainty, and citations/status evidence correctly. A trained candidate does
not validate live injury research, email, OpenAI, PostgreSQL, or deployment.

## Required tests and commands

Run all existing deterministic checks plus new weekly-v2 checks. At minimum:

```text
uv sync --frozen --all-extras --dev
uv run pytest -q
uv run python scripts/audit_weekly_features.py
uv run python scripts/verify_mvp.py
uv run python scripts/audit_blueprint_mvp.py
uv run python scripts/vertical_smoke.py
uv run python scripts/validate_compose_config.py
```

Run the restored training/evaluation/project commands with explicit 2026
candidate directories. Run the live shadow recommendation exercise without
writing to Sleeper or enabling production publication.

If Node/npm becomes available, rerun web tests/build/e2e. Otherwise preserve the
existing external-blocked label. PostgreSQL/Docker are not prerequisites for ML
evaluation, but they remain required before internet deployment.

Tests must cover:

- Manifest/path/hash tampering.
- Untrusted or incompatible joblib rejection.
- Training cutoff and poisoned-future leakage.
- Feature order/type drift.
- Clean-process model loading.
- Real-inference versus hash-fallback distinction.
- League-specific scoring at draw level.
- K/DST integration.
- Determinism and draw stability.
- Identity and coverage gates.
- Partial artifact/output failure.
- Atomic shadow promotion and rollback.
- One authoritative readiness decision across API, jobs, and UI.

Do not make required CI download live multi-gigabyte data or call external APIs.
Use small deterministic fixtures for CI and keep full training/evaluation as an
explicit reproducible workflow with cached, hashed public inputs.

## Required artifacts and report

Create or update:

- Reproducible weekly-v2 ingest/build/train/evaluate/project entrypoints.
- A versioned artifact manifest schema and validator.
- Leakage audit/feature contract artifacts.
- Out-of-fold evaluation and promotion decision artifacts.
- Immutable trained candidate models if training succeeds.
- Immutable 2026/week-1 shadow output and draw partitions if inference succeeds.
- `docs/WEEKLY_V2_TRAINING_AND_PROMOTION_REPORT.md`.
- `docs/WEEKLY_V2_PORT_PROVENANCE.md` with every newly ported file.
- Operations runbook commands and rollback instructions.

The report must state:

1. Exact data cutoffs and fingerprints.
2. Training seasons and samples by model/position.
3. Feature and target contracts.
4. Leakage audit evidence.
5. Out-of-fold metrics and baselines by season/position.
6. Calibration, conservation, coverage, and draw-stability results.
7. Model and output hashes.
8. Whether artifacts were retrained or migrated, with proof of compatibility.
9. Evidence that the app consumed real inference rather than preseason scaling.
10. League-specific shadow results for the six live contracts.
11. Every warning, degradation, skip, and external blocker.
12. Separate go/no-go decisions for:
    - trained artifact classification;
    - manual trained shadow publication;
    - automatic weekly publication;
    - use of trained results for start/sit decisions;
    - public-internet deployment.

## Completion criteria

The task is complete when one of these honest outcomes is reached:

### Passing outcome

- Reproducible 2026 models are trained through at most 2025.
- Leakage-safe historical evaluation passes predeclared gates.
- The manifest and all artifacts validate in a clean process.
- Real weekly inference produces complete canonical-ID stat distributions.
- All six league scoring contracts can score the output correctly at draw
  level, with the two empty redraft rosters skipped only at recommendation time.
- The application candidate is built from the verified output, never the hash
  scaling fallback.
- Shadow publication, failure injection, and rollback pass.
- Operations reports `trained` for the verified candidate and explains the
  exact model/output provenance.
- The report makes an evidence-backed automatic-publication recommendation.

### Valid no-go outcome

- Every locally achievable implementation, port, audit, and test is complete.
- The exact missing data, failed evaluation threshold, unsupported scoring
  target, or incompatibility is demonstrated with evidence.
- Fixture/fallback state and automatic-publication block remain intact.
- No fake/dummy/copied-incompatible artifact is accepted.
- The report gives the shortest technically sound remediation path.

Do not declare success just because nine files exist or 2026 projections were
written. Success means evaluated models actually generated the league-scored
candidate consumed by the app.

## Final response

Lead with the five separate go/no-go decisions from the report. Then provide:

1. Whether models were retrained or migrated and why.
2. The critical false-green defect and the exact fix proving real inference.
3. Evaluation results versus baselines and any failed segments.
4. League-specific scoring/draw validation.
5. Commands run and their outcomes, including blocked checks.
6. Artifact/model/output locations and hashes without embedding large binaries.
7. Remaining steps, grouped into model/data, operational infrastructure, and
   external credentials.
8. Files changed, grouped by training, evaluation, inference, application,
   tests, and documentation.

If automatic publication is still a no-go, say so plainly and leave it disabled.
If it is recommended, do not enable the user's recurring production schedule;
state the one explicit operator action still required after review.
