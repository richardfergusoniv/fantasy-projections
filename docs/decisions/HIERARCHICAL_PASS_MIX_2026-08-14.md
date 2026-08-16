# Hierarchical pass distribution (2026-08-14)

> **RETIRED.** Hierarchical pass mix and volume composition were deleted from
> the shipped pipeline. Forecast `pred_pg` plus availability hygiene is the
> only board path. Kept as historical design notes.

Team volume → WR/TE/RB group shares → player shares within group.

## What changed (historical)

- **L2** `team_pass_mix.py`: observed team WR/TE/RB target shares; scheme
  model; OC inheritance.
- **L3**: `apply_hierarchical_pass_distribution` renormalized receiving
  volume within `(team, position)`.
- Coverage floors, usage-share priors, and formation-role L3 blend lived on
  this path and inflated boards (e.g. Pierce-class WR rates).

Modules and tests referenced by the original note no longer exist.
