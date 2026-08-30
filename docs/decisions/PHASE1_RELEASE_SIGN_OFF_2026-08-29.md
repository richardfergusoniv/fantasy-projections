# Phase 1 release sign-off — `phase1_rehearsal_20260829`

**Date:** 2026-08-29  
**Decided by:** Richard (agent-assisted review)  
**Season:** 2026  
**Namespace:** `phase1_rehearsal_20260829`  
**Manifest SHA-256:** `a89865441775f67b46ceb6ce50898b859e4a4fc4295585a0b3d2b62bd121ab0c`  
**Release ID:** `c17b8f4c-d8bc-4a3f-8077-974eb533a03b`

## Decision

**Approve** `phase1_rehearsal_20260829` for active production promotion.

Phase 1 architecture is validated: one immutable, internally consistent board-and-simulation release becomes active through a single reversible pointer update. This sign-off covers the remaining **player-facing** gate before re-promotion.

## Hash alignment (selected board identity)

All references agree on the sealed board:

| Reference | `selected_board_file_hash` | `selected_points_vector_hash` |
|-----------|---------------------------|------------------------------|
| Sealed manifest | `476b2342…` | `88621002…` |
| Recomputed from `fantasy_points_2026.csv` | match | match |
| `simulation_manifest_2026.json` | match | — |
| `release_report_2026.json` provenance | match | — |
| Public browser manifest copy | manifest `a8986544…` | — |
| Overlay population (778 players) | — | `f50054b4…` match |

## Accuracy-first eligible RB/WR deltas (vs legacy `players_2026.json`)

93 players under `market_no_v3` selected arm (ADP ≤ 120):

| Position | Count | Mean Δ pts/game | Median Δ | Max \|Δ\| | Mean Δ VORP |
|----------|-------|-----------------|----------|-----------|-------------|
| RB | 40 | +0.10 | +0.13 | 3.62 | +1.49 |
| WR | 53 | +0.89 | +0.74 | 8.42 | −2.36 |

Largest moves are concentrated in the intended transformed population (e.g. Alec Pierce +8.4 pts from market-curve WR arm; Amon-Ra St. Brown +4.1 pts). Incumbent QB/TE arms unchanged by design.

685 players retain `incumbent` treatment; no `new_player_v1_only` classifications in this board.

## Propagated outputs

- **Tiers:** 763/778 changed vs legacy — expected: legacy board lacked 10k simulation overlay fields; tiers recomputed from new ranks/VORP.
- **Ranks:** 464 players moved >20 overall ranks — concentrated in RB/WR selected arm and deep sleepers gaining finish-probability context.
- **Replacement values:** mean shift +6.8 pts — consistent with accuracy-first level shift on skill positions.
- **Sleepers surface:** `deep_band_accuracy.json` unchanged hash; comparison artifact updated with new board means.

## Simulation overlay spot-check

All 778 exported players populated:

- `p_finish_top6` … `p_finish_top48` (monotonic: 0 violations)
- `sim_vorp_p10/p50/p90`, `p_vorp_positive`
- `expected_pos_rank`, `median_pos_rank`

By treatment: incumbent (685) and selected (93) both fully populated.

## Archived artifacts

Bundle archive index:  
`output/model_v3/release_archive/season=2026/namespace=phase1_rehearsal_20260829/release_archive_index.json`

| Artifact | Path |
|----------|------|
| Sealed manifest | `output/model_v3/release_bundles/season=2026/namespace=phase1_rehearsal_20260829/release_bundle_manifest.json` |
| Validation attestation | `.../release_bundle_validation.json` |
| Player review | `output/model_v3/release_archive/season=2026/namespace=phase1_rehearsal_20260829/phase1_player_review_20260829.json` |
| Production rehearsal report | `docs/decisions/PHASE1_PRODUCTION_REHEARSAL_2026-08-29.md` |
| This decision | `docs/decisions/PHASE1_RELEASE_SIGN_OFF_2026-08-29.md` |

## Promotion

```bash
python -m src.projection.promote_release --season 2026 --artifact-namespace phase1_rehearsal_20260829
python scripts/validate_release_bundle.py --season 2026 --artifact-namespace phase1_rehearsal_20260829 --require-active
```

Rollback: `python -m src.projection.promote_release --season 2026 --rollback` (previous: `phase1_rehearsal_prior`).
