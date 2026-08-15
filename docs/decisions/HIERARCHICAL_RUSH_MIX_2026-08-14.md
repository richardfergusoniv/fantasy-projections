# Hierarchical rush mix (2026-08-14)

Thin L2 rush layer mirroring pass mix: RB / QB / OTHER carry shares.
No TE/FB package splits.

## LOSO gate

`python -m src.projection.team_rush_mix`

| | MAE |
|---|---|
| scheme+lag | **0.0343** |
| prior-season | 0.0347 |
| league-mean | 0.0415 |

`beats_prior=True`, `beats_league=True` → treat as a **measured win**.

## Wiring

`project_season`: after usage priors / pass L2–L3, attach rush mix →
`apply_hierarchical_rush_distribution` → `normalize_team_rushing_volume`.
