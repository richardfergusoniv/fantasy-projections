# Draw-count rollout decision — 2026-08-28

## Context

Nested-prefix draw-stability evidence (7.5k / 10k / 15k vs 20k reference) is **decision-stable** but **numerically failing** at all sub-20k candidates. Production remains on the provisional **1k** publish profile while we run a **namespaced 10k release candidate (RC)** to collect measured operating data without mutating the live board.

## Frozen evidence

Immutable bundle (copy + hash manifest):

```text
output/model_v3/frozen/draw_stability_intermediate_v20k_2026/
  freeze_manifest.json
  draw_stability_intermediate_v20k_2026.json
  draw_count_decision.json
  decision_change_diagnostics_2026.json
  player_stability_diagnostics_2026.parquet
```

Contract anchors:

- `selected_board_hash`: `67f2c4b88ad370b15e2363d4f915e5ec915d1ea6280625df528edfbd75d41700`
- `canonical_projection_run_id`: `d494c516-f86a-4fc3-afdd-dd8635b72ec5`

## Rollout artifact

`output/model_v3/draw_count_rollout_decision.json` references `freeze_id` and production release pointers — not mutable live filenames.

## RC publish (non-public)

```bash
python -m src.projection.publish --season 2026 \
  --simulation-profile release_candidate \
  --artifact-namespace rc_10k_20260828 \
  --rollout-label "decision-stable_numerically-not-validated_rc"
```

Writes under:

```text
output/model_v3/release_candidates/season=2026/namespace=rc_10k_20260828/
```

**Does not modify:**

- `draft_assistant/data/players_2026.json`
- `output/model_v3/simulation_manifest_2026.json`
- `output/model_v3/release_report_2026.json`
- `output/model_v3/releases/release_2026_current.json`

RC board export: `players_2026_rc.json` in the namespace directory.

## Rollback vs 1k republish

| Action | Meaning |
|--------|---------|
| **Operational rollback** | Restore `output/model_v3/releases/release_2026_current.json` from `release_2026_previous_1k.json` (pointer only; no pipeline rerun) |
| **1k republish** | `python -m src.projection.publish --season 2026 --simulation-draws 1000` — new run under current inputs; **not** a rollback |

## Validation and comparison

```bash
python scripts/validate_release_candidate_publish.py \
  --artifact-namespace rc_10k_20260828

python scripts/compare_draw_profile_overlays.py \
  --rc-namespace rc_10k_20260828
```

Comparison emits `hold` / `board_or_contract_identity_mismatch` when board or contract hashes differ — not numerical draw-count deltas.

## Human policy decision (Phase 2 closed)

**Selected policy:** `maintain_1000_temporarily`  
**Production profile:** `provisional_current_configuration` (1,000 draws)

Human decision record: [`DRAW_COUNT_ROLLOUT_HUMAN_DECISION_2026-08-28.md`](DRAW_COUNT_ROLLOUT_HUMAN_DECISION_2026-08-28.md)

Closure artifact: `output/model_v3/draw_count_rollout_decision.json` (`draw_count_rollout_decision_v2`, `phase_2_status: closed`)

Measured RC runtime: **9,774s (~163 min)** — see `rc_experiment` block in closure artifact.

## Runtime estimate

Planning estimate in rollout artifact includes `runtime_estimate_basis`. The RC run replaces estimates with measured `runtime_seconds` on the RC simulation manifest.
