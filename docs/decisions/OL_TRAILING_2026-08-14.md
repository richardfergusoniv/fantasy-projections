# OL trailing average (2026-08-14)

## Change

Live predict features only: optional snap-weighted trailing average of
exact-season OL quality scores over `OL_TRAILING_SEASONS` ending at the
source season. Historical/backtest feature rows stay exact-season.

- Kill-switch: `OL_TRAILING_SEASONS` in [`src/projection/contracts.py`](src/projection/contracts.py)
- Implementation: [`src/projection/ol_quality.py`](src/projection/ol_quality.py)
  (`trailing_for_seasons` on `team_season_ol_quality`)
- Wired from `build_player_season_features(..., ol_trailing_for_seasons={source})`
  and roster-move OL context for the live source season.

## Ablation

Predict next season's team OL scores from prior exact vs 3-season trailing
mean (2023–2025 holds):

| Score | MAE exact→next | MAE trail→next |
|---|---|---|
| ol_pass_protection_score | 0.00506 | **0.00482** |
| ol_run_blocking_score | 0.05897 | **0.05368** |

Trailing ≤ exact on 5/6 fold×metric cells; mean MAE improves or ties.

## Decision

**Shipped** `OL_TRAILING_SEASONS = 3`. Set to `0` to disable.
