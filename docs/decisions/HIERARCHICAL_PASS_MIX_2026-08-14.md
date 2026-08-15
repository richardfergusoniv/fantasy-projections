# Hierarchical pass distribution (2026-08-14)

Team volume → WR/TE/RB group shares → player shares within group.

## What changed

- **L2** [`src/projection/team_pass_mix.py`](src/projection/team_pass_mix.py):
  observed team WR/TE/RB target shares from free weekly usage; scheme model
  on personnel / PROE / pace / public FTN; OC inheritance via the same
  weights as tendency profiles.
- **L3**: `apply_hierarchical_pass_distribution` renormalizes receiving
  volume within `(team, position)` to `team_pass_attempts × group_share`.
- **Vacancy**: `team_vacated_opportunity` now emits
  `vacated_target_share_{wr,te,rb}`; arrivals/incumbents net within position.
- **Reconcilers**: when L2 columns are present, receiving yards / receptions /
  TDs scale **within position**, so they cannot wash out group mix.
- **Usage priors**: `USAGE_SHARE_BLEND_W` stays 0; reviewed curated priors
  are within-group overrides only. Chart slot defaults stay
  `usage_share_reviewed=False`; live injury renumber must not flip that.
- **WR formation roles**: curated `formation_role` (`LWR`/`RWR`/`SWR`) is a
  preseason role assignment (Ourlads columns), not observed snap alignment.
  L3 blends within-WR budget toward those columns (`FORMATION_ROLE_BLEND_W`);
  live WR prior refresh keys off `formation_role` so removing an LWR does not
  promote the RWR into the LWR prior.
- **Hunter**: roster-preferred position resolution so CB-master dual threats
  keep offensive weeks; charted as JAX WR2.

## What stayed league-wide on purpose

- Gate B rate ladder (availability calibration, not room ranking).
- `NAMED_REC_*_COVERAGE` floors (residuals after hierarchical composition).
- Rushing still team → RB (no TE/FB package split in v1).

## Validation

```bash
python -m src.projection.team_pass_mix
python -m unittest tests.test_team_pass_mix tests.test_data_prep_appearances
```

Ship the scheme mix only when LOSO MAE beats prior-season mix (`beats_prior`).
If it does not, `predict` still attaches a mix row (scheme or league-mean
fallback) so L3 composition has a budget; revisit scheme features before
treating L2 as a measured win.
