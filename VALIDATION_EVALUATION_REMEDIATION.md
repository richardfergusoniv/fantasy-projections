# Validation and Evaluation Remediation

This pass fixes validation leakage and makes the external Sleeper proxy
auditable. It does not claim byte-for-byte historical parity with the live 2026
projection path; the remaining interface limits are stated below.

## Causal validation

- Receiving-share composition and the returning-veteran coverage diagnostic now
  weight predicted rates with an availability model trained strictly on earlier
  folds. Held-out `games_played_to` is used only as an observed target, never as
  a predicted-transform input.
- Availability evaluation is rolling-origin and reports three separate scopes:
  all source-season players, players present in the target-season roster
  snapshot, and players absent from that snapshot as attrition.
- Across 2023-2025 folds, the availability model beats carry-forward in the
  all-player and roster-eligible aggregates at every position. Roster-eligible
  fold-mean MAE is QB 3.052, RB 3.593, WR 3.328, and TE 3.427, versus naive
  3.898, 4.499, 4.451, and 3.980.
- The old “coherence” backtest is relabeled returning-veteran receiving coverage.
  Its numerator omits rookies and players without both adjacent-season rates, so
  it is not the physical whole-team passing/receiving identity.

## Season-total parity

The 2025 season-total test now uses the production team-total-by-share receiving
composition wherever the current transition interface supplies a conditional
rate row. Coverage is explicit: 79.8% of roster-eligible WR rows and 88.8% of
roster-eligible TE rows use the composed path. Remaining rows use an explicitly
labeled independent-rate fallback.

The availability-composed season total beats carry-forward for each reported
position on the target-roster-eligible scope: WR receiving yards 140.5 vs 167.2
MAE, TE receiving yards 89.5 vs 106.8, RB rushing yards 173.4 vs 186.2, and QB
passing yards 658.5 vs 776.5.

The parity limit remains material: the historical path has no dated curated role
file, full rookie-room composition, or production QB/team reconciliation entry
point. A future `project_season_as_of` interface is needed to score the exact
preseason-eligible roster without conditioning row availability on the outcome.

## Prediction intervals

`models/interval_residuals.csv` is now calibrated from pooled, strictly-forward
rolling residuals over the 2023, 2024, and 2025 test seasons rather than the one
repeatedly inspected 2025 fold. Each row records fold count and calibration
basis.

`models/interval_forward_coverage.csv` evaluates untouched folds using residuals
from earlier test seasons only. Across 46 position/stat/fold evaluations, mean
coverage is 0.820 for an 0.800 target. Fold-mean coverage is 0.793 in 2024 and
0.847 in 2025. These are marginal stat intervals, not joint fantasy-point or
availability intervals.

## Sleeper proxy

- API payloads are persisted as content-addressed JSON snapshots with endpoint,
  UTC retrieval time, and SHA-256 metadata.
- Response types are validated before field access; requests use timeouts and
  HTTP status checks.
- Name fallback accepts only a unique name/position candidate, or a unique team
  match among ambiguous candidates. Sleeper ID, team, name, candidate counts,
  collision state, match method, and snapshot provenance remain in output.
- Reporting now separates all matched players, positive Sleeper projections, and
  players projected for at least 50 Sleeper points, preventing zero-only rows
  from dominating the headline.

## Downstream provenance

Fantasy and Sleeper deliverables retain team passing anchors when available,
explicit attempt/passing-yard/receiving-yard normalization scales, and aggregate
share-cap/normalization flags. The generic stat-specific scale is no longer
silently selected from an arbitrary long-format row.

## Verification

- `python -m unittest discover -v`: 49 tests pass.
- `python -m src.projection.backtest`: completes successfully on the live
  database and regenerates cross-fitted interval residuals.
- Targeted tests prove held-out games do not affect predicted participation
  transforms, availability scopes remain separate, ambiguous Sleeper names stay
  unmatched unless team resolves them, snapshot metadata is written, forward
  coverage uses only earlier folds, and normalization provenance survives fantasy
  aggregation.
