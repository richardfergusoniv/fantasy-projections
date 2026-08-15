# OC inheritance weight fit (2026-08-14)

## Method

`loso_fit_inheritance_weights` in [`src/coordinator/inheritance.py`](src/coordinator/inheritance.py)
(also `scripts/fit_oc_inheritance_weights.py`):

- Grid team weight ∈ {0.3, 0.4, 0.5, 0.6, 0.7} with
  `internal_team_w >= outside_team_w`.
- Score first-year seats on SD-scaled tendency METRICS plus pass-mix MIX_COLS
  MAE vs observed season values (leave-one-season-out over assignment years).

## Results

| Candidate | Mean LOSO MAE |
|---|---|
| **Best: internal 0.6 / outside 0.6** | **0.419** |
| Judgment 0.7 / 0.3 | 0.430 |
| Team-only (1.0 / 1.0) | 0.437 |

`recommend_update=True` (beats judgment and team-only).

## Decision

**Shipped** `INHERITANCE_WEIGHTS` = 60% team / 40% OC for both `internal` and
`outside_hire`. Single source remains `inheritance.py` (consumed by
`oc_profiles`, pass mix, rush mix).
