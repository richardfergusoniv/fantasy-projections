# Cursor weekly usage-mixture and correlated-draw architecture prompt

Copy everything below the divider into Cursor Agent mode from the repository
root after the completed nested volume-tuning experiment is present.

---

You are the senior probabilistic-modeling engineer responsible for the next
evidence-backed weekly-v2 architectural change: replace the compressed,
independent point-mass approximation with explicit availability/participation
mixtures and team-correlated component-stat draws suitable for start/sit and
matchup decisions.

This is an implementation and evaluation task. Do not resume hyperparameter
search against the same volume architecture, increase calibration slopes,
weaken the promotion policy, or claim victory because simulated outcomes are
more variable than point forecasts. Build the new distributional architecture,
evaluate it with proper probabilistic and decision metrics, integrate it into
the app's league-scoring path, and preserve every existing no-go until the
corresponding gate genuinely passes.

## Current evidence and decision state

The completed nested experiment is:

```text
output/weekly_v2/experiments/volume_tune_20260831_v2/
```

Its final selection is:

```text
promote: false
selected: null
best_relative_candidate: legacy_direct
```

All eight predeclared volume candidates failed the frozen all-fold policy.
Representative calibrated dispersion:

| Candidate | 2023 | 2024 | 2025 | Result |
| --- | ---: | ---: | ---: | --- |
| Baseline two-stage | 0.6887 | 0.6880 | 0.8348 | No-go |
| Legacy direct | 0.6915 | 0.7026 | 0.8394 | No-go; 2023 fails |

The quantitative diagnosis found:

- Raw 2023 WR dispersion `0.3469`, predicted SD `2.51`, actual SD `7.23`.
- Raw 2023 TE dispersion `0.3533`, predicted SD `1.76`, actual SD `4.97`.
- Approximately 52% of roster-week outcomes are DNP/zero while deterministic
  point projections produce approximately 0% exact zeros.
- Compression occurs before calibration through participation-probability ×
  conditional-share means, lag/depth priors, team composition, and accounting.
- Live questionable players can receive overlapping participation and injury
  haircuts, although this did not explain the historical healthy-cohort gap.
- Calibration at the frozen WR/TE cap of 1.52 cannot close the remaining gap.

Current decisions remain:

- Trained artifact classification: GO with caveats.
- Manual trained shadow publication: NO-GO.
- Automatic weekly publication: NO-GO.
- Start/sit use: NO-GO.
- Public-internet deployment: NO-GO.

The latest deterministic verification is 788 passed, 1 skipped;
`audit_blueprint_mvp` is 49/49 and `vertical_smoke` passed. Preserve those
results and the completed no-selection evidence.

## Non-negotiable anti-cheating rules

- Do not change the existing point-promotion thresholds or WR/TE 1.52 slope cap.
- Do not train `legacy_direct` as the 2026 selection; it failed 2023.
- Do not average away a failed season.
- Do not exclude DNPs, reserves, rookies, or difficult depth bands from the
  established recoverable cohort.
- Do not generate one random realization and label it a point forecast.
- Do not compare the standard deviation of stochastic draws with the standard
  deviation of deterministic means and call the point-dispersion gate passed.
- Do not add arbitrary noise after prediction solely to widen distributions.
- Do not calibrate or tune on a target fold's outcomes.
- Do not sample every player independently and call the result joint.
- Do not scale every component stat by one fantasy-point random multiplier.
- Do not allow team/player component totals to violate football accounting.
- Do not label PPFD, kicker, defense, or nonlinear bonus scoring exact unless
  the required component events exist in every draw.
- Do not enable automatic publication or the recurring production schedule in
  this task.

## Read completely before editing

1. `docs/WEEKLY_V2_VOLUME_TUNING_REPORT.md`
2. `docs/WEEKLY_V2_TRAINING_AND_PROMOTION_REPORT.md`
3. `output/weekly_v2/experiments/volume_tune_20260831_v2/nested_selection.json`
4. `output/weekly_v2/experiments/volume_tune_20260831_v2/tuning_selection.json`
5. The baseline dispersion diagnostic artifact referenced by the report
6. `src/projection/weekly/evaluate/harness.py`
7. `src/projection/weekly/evaluate/nested_selection.py`
8. `src/projection/weekly/evaluate/dispersion_diagnostics.py`
9. `src/projection/weekly/models/volume.py` and `volume_config.py`
10. Weekly availability, feature, accounting, team-total, efficiency, veteran,
    rookie, and season-projector modules
11. `src/app/projections/weekly_draws.py`
12. `src/app/projections/weekly_stat_draw.py`
13. `src/app/decisions/draws.py`
14. `src/app/projections/weekly_league_scoring.py`
15. `src/projection/special_teams/models.py`
16. The scoring compiler/contracts and all six live scoring-contract summaries
17. Weekly release/readiness/promotion and decision services
18. Existing simulation/draw-count/calibration/conservation evaluation code in
    this repository
19. Relevant tests, runbooks, data contracts, and modeling decisions

Inspect the dirty working tree first. Preserve every user change, current model
artifact, experiment result, shadow database, immutable release, and active
pointer. Work in new namespaced candidate directories. Never rewrite the failed
tuning evidence.

## Critical deficiencies in the current draw implementation

Explicitly reproduce and fix these shortcomings for the trained path:

1. `src/app/projections/weekly_draws.py` samples an independent split-normal
   fantasy-point value for each player and proportionally scales every component
   statistic. Passing attempts, touchdowns, yards, targets, and receptions
   therefore move in lockstep and can become fractional in implausible ways.
2. The same file has no discrete DNP/positive-usage event from the volume model.
3. Draws are independent across teammates and opponents, so they cannot express
   a shared game environment, QB/pass-catcher covariance, or competition within
   a position room.
4. Team passing totals need not equal player passing/receiving totals; target,
   completion, carry, yard, and touchdown accounting is not guaranteed per draw.
5. First downs are absent from weekly-v2 output, so PPFD is understated.
6. `src/app/decisions/draws.py` explicitly documents independent player draws.
   Preserve independence only for clearly labeled legacy/fallback mode.
7. The current K/DST simulator uses fixed/simple priors and independent random
   calls; it is not linked to schedule, opponent strength, scoring drives, or
   the shared game draw.

Add regression tests that demonstrate these defects against the current
trained-draw path before or while replacing it.

## Phase 1 — define forecast grains and probability contracts

Write a versioned data/model contract distinguishing:

- `P(active_for_game)` from status/injury evidence available by cutoff.
- `P(offensive_participation | active)`.
- `P(positive target/carry/dropback/red-zone usage | participates)`.
- Conditional usage distribution given the event.
- Conditional efficiency distribution given realized usage.
- Deterministic point forecast: expectation of the full mixture.
- Predictive component-stat distribution: joint draw set.
- League-scored predictive distribution: the scoring contract applied per draw.

Specify targets, grains, source timestamps, zero definitions, censoring, and
which event owns each probability. `play_prob` must have one authoritative
location. Injury availability may influence the active event; it must not also
silently multiply already unconditional shares unless mathematically required
and documented.

Use historical active/inactive status when recoverable. Distinguish true DNP,
active zero usage, bye/no-game, preseason roster churn, and missing data. Do not
treat a bye or absent schedule row as a failed participation event.

Add hand-calculated expectation tests proving:

```text
E[stat] = P(active) × P(participates | active) × E[stat | participates]
```

where appropriate, without applying either probability twice.

## Phase 2 — build an as-of mixture training panel

Extend the existing leakage-safe panel or create a derived training contract
containing event labels and conditional targets. For each player-week:

- Scheduled game and kickoff.
- Roster/active/inactive/bye state known at the evaluation cutoff.
- Position, team, opponent, and stable identity.
- Pre-kickoff injury/status and practice features.
- Lagged/depth/role/team context.
- Active/participated/positive-usage labels.
- Conditional target, carry, snap, route/dropback, red-zone, and air-yard shares.
- Component-stat outcomes and first-down outcomes.

Audit target construction carefully. A player with zero targets but positive
snaps is not the same event as an inactive player. A rostered player whose game
was never scheduled in the cohort must not become a zero label accidentally.

All features must pass existing as-of and poisoned-future tests. Persist schema,
cutoff, row counts, event rates by season/position/depth, and input hashes.

## Phase 3 — model discrete events separately

Implement and evaluate explicit probabilistic models for the necessary event
layers. Start with simple, calibrated, interpretable baselines before adding
complexity. Requirements:

- Probabilities are calibrated out of fold by position and important status/
  depth bands.
- Report Brier score, log loss, calibration curves/bins, sharpness, and event
  rates against frequency and simple depth/status baselines.
- Prevent probabilities of exactly zero or one except hard rules such as bye,
  already inactive, or locked actuals.
- Preserve deterministic seeds and explicit model configuration.
- Fail cleanly when a position lacks enough positive examples.
- Do not use current/live statuses in historical folds.

Determine through evaluation whether active, participates, and positive-usage
need separate models for each position/target or whether some layers can share a
well-defined event. Document the choice rather than multiplying a generic
classifier across every volume target.

## Phase 4 — team/game-level joint draw generator

Implement a draw engine whose fundamental unit is a scheduled game and team,
not an independent player.

For every simulation index:

1. Sample shared game environment from pre-kickoff team/opponent inputs.
2. Sample team plays, pass attempts, rush attempts, scoring opportunities, and
   touchdowns from calibrated count/rate distributions.
3. Sample player active and participation events using the explicit contracts.
4. Allocate targets, carries, dropbacks, snaps, red-zone work, and air yards
   jointly within each team/position room.
5. Sample completions, yards, touchdowns, interceptions, fumbles, and first
   downs conditionally on realized opportunities.
6. Reconcile all player components to team totals.
7. Generate correlated opposing-team and teammate outcomes.
8. Feed the same game draw into K and DST models.

Use a principled constrained composition such as a calibrated logistic-normal,
Dirichlet/Dirichlet-multinomial, empirical residual bootstrap, or another
documented approach. Select based on rolling out-of-fold evidence. Independent
Bernoulli events for every receiver are insufficient if they routinely create
impossible rooms; use conditional room logic or correlated latent variables.

Expected player means derived from draws should reconcile to the deterministic
mixture expectation within Monte Carlo tolerance. Do not force draws to match a
mis-specified point mean if that would destroy calibrated event behavior; expose
and resolve the inconsistency through gates.

## Phase 5 — per-draw football accounting

Add hard validation and, where appropriate, construction-by-design invariants:

- Team QB dropbacks/pass attempts reconcile with team totals.
- Player targets do not exceed team attempts and reconcile to the named plus
  explicitly reported replacement/other receiver reserve.
- Player receptions do not exceed targets.
- Team receptions/completions reconcile.
- Passing yards reconcile to receiving yards subject to documented sack/stat
  conventions.
- Passing TDs reconcile to receiving TDs.
- Player carries plus explicit QB/other reserve reconcile to team rush attempts.
- Rushing and receiving touchdowns do not exceed team offensive touchdowns.
- Shares remain within valid bounds.
- Inactive players have zero stats.
- Already-played players are locked to actuals and never resampled.
- Bye teams have zero player/K/DST game output.
- No NaN, infinity, negative count, or impossible fractional discrete-event
  semantics leak into league scoring.

Do not “repair” invalid independent draws only by proportional scaling if the
repair destroys the event mixture. Prefer generating valid draws structurally;
use reserves transparently for unmodeled players.

Persist conservation diagnostics by draw/team/stat and fail the candidate when
violations exceed exact or predeclared numerical tolerances.

## Phase 6 — points per first down

Model the component events required by live PPFD scoring:

- `pass_first_downs`
- `rush_first_downs`
- `rec_first_downs`

Use public, temporally valid play-level/weekly data already permitted by the
project. Fit conditional rates/distributions by position and context with
leakage-safe rolling evaluation. A beta-binomial/binomial or other count model
is acceptable when justified and calibrated.

Per draw:

- Receiving first downs cannot exceed receptions.
- Rushing first downs cannot exceed carries.
- Passing first downs should reconcile with team receiving first downs under
  the data provider's stat definition.
- First-down outcomes must co-vary with yards/opportunities rather than being an
  independent constant multiplier.

Evaluate count MAE/calibration and the resulting fantasy-point distribution in
the live PPFD league. Keep PPFD scoring marked incomplete if the model or source
definition cannot be validated.

## Phase 7 — kicker and defense integration

Replace fixed-context standalone special-team draws with game-linked inputs.
Keep these models intentionally simpler but evidence-based.

Kicker draws should use:

- Team scoring-drive/opportunity distribution from the shared game draw.
- Red-zone touchdown versus field-goal settlement probability.
- Kicker identity/status.
- Distance-bucket attempt and accuracy priors.
- Stadium/venue and weather when known by cutoff.
- Extra-point opportunities linked to team touchdowns.

DST draws should use public opponent-adjusted efficiency inputs such as EPA,
success rate, pressure/sack and turnover priors, pace, implied points, weather,
and home/away context. Do not require proprietary DVOA unless a licensed source
already exists. Generate the exact live scoring components: points/yards
allowed, sacks, interceptions, fumble recoveries, defensive/special-teams TDs,
safeties, blocked kicks, and any other supported nonzero rule.

Kicker, offense, opponent offense, and DST outcomes from one game must be
coherent. For example, scoring drives and points allowed cannot be sampled from
unrelated worlds.

Evaluate against simple historical baselines and report high uncertainty. K/DST
failure must block exact-league start/sit readiness only in leagues that require
the affected position, while remaining visible globally.

## Phase 8 — probabilistic evaluation

Use strict rolling/expanding out-of-fold evaluation. Reuse the repaired harness
and explicit namespaces; do not corrupt the completed volume experiment.

Evaluate discrete events with:

- Brier score and log loss.
- Reliability/calibration bins.
- Sharpness and prevalence by position/depth/status.

Evaluate component and fantasy-point distributions with proper scoring rules:

- CRPS or an equivalent sample-based proper score.
- Weighted interval score.
- P10/P50/P90 coverage and width.
- PIT/rank histograms where appropriate.
- Mean MAE/RMSE and rank correlation.
- Point-forecast dispersion under the existing unchanged policy.
- Tail and zero-mass calibration.
- Team/game conservation and correlation diagnostics.

Compare against:

- The current independent split-normal/scaled-component draw implementation.
- The current baseline two-stage deterministic model.
- Simple empirical/frequency mixture baselines.
- `legacy_direct` only as a non-selected diagnostic baseline.

Do not call the old point-dispersion gate passed because predictive samples
match actual variance. Report deterministic-mean dispersion and distributional
calibration separately. If deterministic mean dispersion remains below 0.70,
automatic point-publication remains a no-go even when the new draws are useful.

## Phase 9 — decision-level backtesting

Backtest the actual user-facing decisions on out-of-fold draws:

- Legal lineup selection and realized lineup regret.
- Probability calibration for head-to-head matchup wins.
- Current-opponent versus optimized-opponent modes.
- Recommendation stability at increasing draw counts.
- Sensitivity to uncertain/inactive players.
- League-scoring differences, especially PPFD and yardage bonuses.
- K/DST lineup effects in leagues that require them.

Use only information available before kickoff and freeze recommendations before
actuals. Compare against deterministic-mean lineups and a simple prior/market
baseline where leakage-safe snapshots exist.

Set decision-readiness thresholds before reviewing the final outer results.
Include Monte Carlo error. Start/sit may become a GO only if the new
distribution, scoring fidelity, conservation, and matchup calibration gates all
pass; the current no-go is the default.

## Phase 10 — application integration

Introduce a new versioned joint partition schema rather than silently changing
schema version 1. It must include or reference:

- Season, week, as-of cutoff, schedule/game IDs, and kickoff times.
- Model/manifest/feature/evaluation hashes.
- Global seed and partition identity.
- Per-game/team latent draw provenance.
- Player identity, position, team, opponent, availability and event-model
  versions.
- Component-stat draws or immutable partition references.
- K/DST/PPFD readiness.
- Conservation and probabilistic-gate results.

The decision layer must load aligned joint draws by simulation index. Do not
reseed independently per API request. Locked actuals must replace only the
relevant player/game outcomes without changing the frozen pregame evidence used
for evaluation.

Retain current points-only and independent-stat implementations only as explicit
legacy/fallback modes. Operations/API/UI must distinguish:

- `legacy_points_independent`
- `legacy_scaled_components`
- `joint_stat_mixture_candidate`
- `joint_stat_mixture_validated`

Never describe a mixed/fallback draw set as exact league scoring. Show source
freshness, partition hash, draw count, scoring fidelity, unavailable rules, and
Monte Carlo error.

## Phase 11 — six-league live shadow validation

After historical gates, generate an isolated 2026 week-1 shadow partition using
the existing trained candidate and live shadow league contracts. Do not promote
or overwrite active pointers.

For all six leagues:

- Apply the exact compiled scoring contract per draw.
- Verify distinct scoring produces expected differences.
- Exercise lineup and matchup endpoints for the four populated dynasty leagues.
- Correctly skip the two empty pre-draft redraft rosters.
- Verify owner roster identity and zero unresolved starters.
- Inspect high-impact anomalies and K/DST rankings.
- Verify PPFD contributions are nonzero and sourced from modeled first-down
  events where the league scores them.
- Verify repeated requests consume the same immutable partition.

Run failure injection and prove partial/corrupt/mismatched partitions cannot
advance any pointer or power a “validated” start/sit answer.

## Readiness policy

Create separate gates for:

1. Point-model promotion under the existing frozen policy.
2. Event-probability calibration.
3. Joint-draw proper scores and interval calibration.
4. Per-draw football conservation.
5. PPFD component readiness.
6. Kicker readiness.
7. DST readiness.
8. League scoring completeness.
9. Decision-level lineup/matchup validation.
10. Artifact integrity and atomic publication.

`auto_publish_allowed` must require every production-required gate. It must stay
false if the old point-dispersion gate still fails. Do not conflate the trained
point artifact classification with validated joint draws.

## Required tests

Add focused deterministic tests for:

- Probability expectation and no double counting.
- DNP, active-zero, positive-usage, bye, and locked-actual semantics.
- Correlated teammate/opponent behavior.
- Exact team/player accounting per draw.
- Component distributions not moving in perfect proportional lockstep.
- Stable seed/partition reproducibility and different-seed independence.
- First-down bounds and pass/receive reconciliation.
- Nonlinear bonus and PPFD scoring per draw.
- Kicker/DST shared-game coherence.
- Point versus draw gate separation.
- Schema/hash/path tampering and partial partitions.
- API mode/fidelity reporting.
- Decision stability and calibrated synthetic matchup examples.

Run the complete deterministic suite and all existing audits after focused
tests. Keep expensive full historical evaluation as an explicit reproducible
workflow, with small synthetic CI fixtures proving the algorithmic contracts.

## Required report

Create `docs/WEEKLY_JOINT_USAGE_DRAWS_REPORT.md` containing:

1. Existing architecture defects reproduced.
2. Event/target/probability contracts.
3. Data cutoffs, sources, hashes, and leakage evidence.
4. Model specifications and training samples.
5. Event calibration results by fold/position/depth/status.
6. Distributional metrics versus both baselines.
7. Unchanged deterministic point-promotion results.
8. Team/game conservation and correlation evidence.
9. PPFD, K, and DST validation.
10. Decision-level lineup and matchup backtests.
11. Six-league live shadow results.
12. Artifact/partition schema and hashes.
13. Defects found and fixed.
14. Every skip, degradation, and external blocker.
15. Separate go/no-go decisions for:
    - point-model classification;
    - joint-draw classification;
    - manual trained shadow publication;
    - automatic weekly publication;
    - start/sit use;
    - public-internet deployment.

Link this report from the prior training and tuning reports. Do not rewrite the
no-selection result or imply this architecture was chosen by that candidate
grid; it is a separately evaluated follow-up.

## Valid completion outcomes

### Passing distributional outcome

- Mixture events are explicit, leakage-safe, and calibrated.
- Joint stat draws are team/game-correlated and conserve football totals.
- PPFD/K/DST component paths pass for all applicable live contracts.
- Proper distribution and decision metrics pass predeclared outer-fold gates.
- Six-league shadow validation passes with immutable partitions.
- Start/sit may be recommended only if its complete gate passes.
- Automatic publication remains disabled unless the unchanged point gate and
  every other production gate also pass.

### Valid no-go outcome

- The architecture and evaluation are implemented correctly.
- The exact failed event, distribution, conservation, scoring, or decision gate
  is documented with evidence.
- Current active/fallback releases remain untouched.
- No artificial variance or gate substitution is used.
- The report identifies the narrowest evidence-backed next model change.

Do not stop after generating visually wider draws. The goal is calibrated,
coherent decision distributions—not variance for its own sake.

## Final response

Lead with the six separate go/no-go decisions. Then report:

1. What replaced the independent scaled-component draw path.
2. Availability/participation semantics and whether double counting was fixed.
3. Historical probabilistic and deterministic metrics.
4. Conservation, correlation, PPFD, K, and DST results.
5. Decision-level start/sit and matchup evidence.
6. Six-league shadow outcome.
7. Commands/tests run, failures, and blocked checks.
8. New model/partition/report paths and hashes.
9. Why automatic publication and start/sit remain enabled or disabled.
10. Files changed, grouped by data/features, models, draws, scoring, decisions,
    application integration, tests, and documentation.

Never claim the point-dispersion gate passed from the variance of stochastic
draws. Never enable recurring publication automatically.
