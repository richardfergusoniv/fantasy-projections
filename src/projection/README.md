# Projection pipeline stages

Opportunity-first NFL fantasy projections. Stages own one job;
[`predict.py`](predict.py) wires them and re-exports public names for tests.

## Flow

1. **Features** — [`features.py`](features.py), [`data_prep.py`](data_prep.py), [`ol_quality.py`](ol_quality.py): player-season opportunity and efficiency context.
2. **Train** — [`train.py`](train.py), [`transitions.py`](transitions.py): LightGBM rate/share models and Ridge team anchors.
3. **Veterans / rookies** — veteran path + [`rookies.py`](rookies.py) rule path (rookies always `low_confidence`).
4. **Depth / availability** — curated chart eligibility + Gate A games + Gate B rate ladder ([`depth_rates.py`](depth_rates.py), [`depth_gating.py`](depth_gating.py)).
5. **Compose** — availability hygiene, team-anchor metadata, counting-stat caps, season totals ([`composition.py`](composition.py), [`team_reconcile.py`](team_reconcile.py)). Does **not** invent or redistribute team volume onto players; shipped `pred_pg` is the forecast rate after Gate A/B.
6. **Fantasy / draft** — [`fantasy_points.py`](fantasy_points.py) and draft/team UIs.

Stage modules extracted from the former monolithic predict path: [`depth_rates.py`](depth_rates.py), [`artifacts.py`](artifacts.py), [`depth_gating.py`](depth_gating.py), [`roster_moves.py`](roster_moves.py), [`replacement.py`](replacement.py), [`veterans.py`](veterans.py), [`team_reconcile.py`](team_reconcile.py).

## Contracts

Shared paths and calibrated constants live in [`contracts.py`](contracts.py).
First-year OC inheritance weights live in [`src/coordinator/inheritance.py`](../coordinator/inheritance.py) (tendency profiles).

## Curated depth chart rule

The curated starters CSV is **eligibility / room membership / formation_role metadata**, not automatic usage order. Formation order is not usage. Chart `usage_share_prior` values are research display defaults for depth-chart tooling only; they do not move projections.
