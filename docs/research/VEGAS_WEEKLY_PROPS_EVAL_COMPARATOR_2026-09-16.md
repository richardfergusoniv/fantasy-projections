# Vegas weekly props evaluation comparator (Role 2) — 2026-09-16

**Date:** 2026-09-16
**Scope:** research / shadow evaluation harness only. New package
`src/projection/weekly_eval/`, fixtures, tests, and a dry-run CLI.
**Does not:** promote or reseal, change `APP_PROJECTION_SOURCE`, wire the PWA,
train on props as targets, blend Vegas into the weekly model (Role 3), or edit
`src/projection/weekly_latent/` allocation code.

Locked decision:
[`docs/decisions/WEEKLY_MODEL_PROMOTION_AND_PROPS_ROLES_2026-09-15.md`](../decisions/WEEKLY_MODEL_PROMOTION_AND_PROPS_ROLES_2026-09-15.md).

## Role

| Role | This PR |
|---|---|
| **1. Production benchmark** | Untouched. The app still serves Vegas weekly props. |
| **2. Evaluation comparator** | **This work.** Timestamped snapshots; score a shadow board against market-implied distributions and realized outcomes. |
| **3. Optional ensemble** | **Forbidden.** `blend_weights` fail closed. No market-free challenger has been reported. |

Role 2 is measurement. It is not a live scrape treated as a historical as-of line, not a training target, and not a promotion argument.

## Inputs

Grain: one row per `(player_id, season, week, market)`.

1. **Prop snapshots** (`prop_snapshots.csv`)

   | Field | Required | Notes |
   |---|---|---|
   | `player_id` | yes | Join key |
   | `season`, `week` | yes | Join key |
   | `market` | yes | Canonical names (`rec_yards`, `pass_yards`, …) |
   | `line` or `implied_mean` | one of | Over/under line and/or market location |
   | `implied_p_over` | no | De-vig P(over) when available |
   | `as_of` | **yes** | Snapshot timestamp. Missing/blank → fail closed |
   | `kickoff_at` | **yes** | Cutoff. `as_of > kickoff_at` → fail closed (later-season leak) |
   | `player_name`, `source` | no | Provenance |

2. **Shadow weekly board** (player-week means CSV)

   Required: `player_id`, `season`, `week`, `market`, `model_mean`.
   Optional: `model_std`, `model_p_over`, identity fields.
   Same-week outcome columns (`targets`, `attempts`, `team_attempts`,
   `fantasy_points`, …) on this frame **fail closed**. Lagged `_l3` / `_l5`
   / `_prior` forms remain allowed.

3. **Realized outcomes** (optional)

   Required: `player_id`, `season`, `week`, `market`, `actual`.
   Labels only — not joined back as features.

Committed synthetic fixture:
`src/projection/weekly_eval/fixtures/` (four player-week rows).

## Outputs

`compare_shadow_to_vegas(...)` returns a JSON-ready dict. The CLI writes
`output/shadow_vegas_props_compare/summary.json`.

Always present:

- `role = "evaluation_comparator"`
- `role3_blend = false`
- `promoting = false`
- `gate_verdict = "not_promoting"`
- `n_board`, `n_snapshots`, `n_matched`, `n_with_actuals`
- `metrics`, `by_market`
- `leakage` flags documenting fail-closed gates

This summary is **not** a promotion artifact and is not an app pointer.

## Metrics chosen

The promotion gate asks for more than RMSE: pinball / CRPS / coverage /
calibration, and prop Brier where relevant. The synthetic fixture has means,
optional std, a line, implied P(over), and actuals — enough for:

| Metric | Why |
|---|---|
| **MAE / RMSE** (model vs actual, market vs actual) | Point loss vs realized box. Floor, not the achievement. |
| **MAE model vs market** | Diagnostic only. Agreement with the market is **not** a target. |
| **Brier** on over/under vs the snapshot line | Proper score for the binary event implied by a prop. |
| **Gaussian CRPS** | Proper score for a predictive distribution when we have (mean, std). Std comes from `model_std` or a documented CV fallback (`0.25`, min 1.0). Market std is inverted from `(mean, line, P(over))` when identified. |
| **Pinball at q=0.5** | Quantile / calibration-friendly; equals half of MAE when the quoted median is the mean. |
| **Coverage of Gaussian p10–p90** | Interval honesty check on the same Gaussian used for CRPS. |

What this fixture cannot yet do (and does not fake): sample-based CRPS from
Monte Carlo draws, empirical reliability diagrams with enough n, or de-vig
from raw American odds. Those belong in a later Role 2 pass once live
snapshots and model draws exist.

## Leakage

- **as_of required.** No snapshot row without a timestamp.
- **kickoff_at required.** A timestamp without a cutoff cannot prove the line
  was observable before the game.
- **as_of ≤ kickoff_at.** A November line must not sit on a week-1 row.
- Duplicate `(player, season, week, market)` snapshots: latest still-valid
  `as_of` wins.
- **Same-week outcomes are not features.** The comparator reuses
  `SAME_WEEK_OUTCOME_DENYLIST` / `is_allowed_prediction_column` from
  `src/projection/weekly/draws/feature_outcome_split.py` (read-only import).
  That is the existing inference denylist, including `team_attempts` /
  `team_air_yards`. Outcomes may appear only on the label frame.
- **No Role 3 blend.** Non-empty `blend_weights` raise
  `Role3BlendForbiddenError`.

## Out of scope

- Role 1 production scrape, consensus merge, pointer swap, freshness gates
- Role 3 ensemble / shrinkage toward the market
- Training that uses props as targets or features
- PWA / API / `APP_PROJECTION_SOURCE` changes
- `src/projection/weekly_latent/` allocation, conservation, M3 availability
- Claiming the weekly model beats Vegas from this synthetic dry-run
- Live `projections.db` or provider scrapes (this PR is fixture-only)

## How to run

```bash
uv run python scripts/compare_shadow_vegas_props.py --dry-run
uv run pytest tests/weekly_eval -q
```

Dry-run reads the committed fixtures and writes
`output/shadow_vegas_props_compare/summary.json`.

## Gate

Not promoting. A green unit test and a research merge are not the 6–8 live
shadow-week bar in the decision note. One synthetic fixture is a harness, not
evidence of weekly accuracy.
