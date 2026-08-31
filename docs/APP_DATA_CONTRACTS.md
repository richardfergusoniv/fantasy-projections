# Application data contracts

## Read response envelope

Authenticated reads include:

- `data_as_of` — ISO timestamp for the underlying snapshot
- `projection_run_id` — immutable forecast identity when applicable
- Snapshot IDs (`rule_snapshot_id`, `roster_snapshot_id`, etc.) where relevant

## Scoring contract

Compiled `ScoringContract` objects store:

- `linear_rules` — per-stat unit scoring (including receptions and first downs)
- `threshold_rules` — yardage bonuses evaluated per draw
- `dst_rules` — team defense keys
- `roster_slots` — legal lineup slots
- `unsupported_keys` — nonzero Sleeper keys without mappings; blocks publication
- `contract_hash` — deterministic hash of normalized contract

## Availability events

Lifecycle fields:

- `active_from`, `active_until`, `cleared_at`
- `source_snapshot_id`, `evidence_ids`, `policy_json`

Events clear only on healthy, complete source payloads.

## Projection runs

Modes: `preseason`, `weekly`, `ros`, `dynasty`.

Each run stores `as_of`, `model_version`, `input_hash`, manifest URI, and partitioned draw artifacts.

## Trade evaluation

Three independent outputs:

1. `objective` — roster utility per side
2. `fairness` — gap and uncertainty
3. `acceptance` — bounded probability (75–90% objective, 10–25% tendency)

## Assistant audit

Server stores hashed user ID, tools called, model, token usage, estimated cost, and latency. The assistant never invents projections when tools fail.
