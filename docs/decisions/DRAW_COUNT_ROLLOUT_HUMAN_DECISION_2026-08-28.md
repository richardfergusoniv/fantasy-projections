# Draw-count rollout — human decision record (Phase 2 closure)

**Date:** 2026-08-28  
**Decided by:** Richard  
**Season:** 2026  
**Artifact:** `output/model_v3/draw_count_rollout_decision.json` (schema `draw_count_rollout_decision_v2`)

## Current production label

```text
provisional_current_configuration
```

Production remains at **1,000 draws**. This is an explicitly provisional setting, not a validated default.

## Evidence summary

| Source | Result |
|--------|--------|
| Frozen nested-prefix sweep (7.5k / 10k / 15k vs 20k) | All sub-20k candidates **fail** strict numerical gate |
| Decision-change diagnostics (20k reference) | Provenance **ok**; material decision changes **survive** at 20k |
| RC publish (`rc_10k_20260828`) | **Pass** — 778 overlays, public artifacts unchanged |
| Measured RC runtime | **9,774s (~163 min)** |
| Overlay comparison (1k vs RC-10k) | **Hold** — `replacement_contract_hash` mismatch |

## Operational policy options considered

### 1. Strict numerical policy (`strict_numerical_policy_20000`)

Move production to **20,000 draws** after a namespaced, non-public RC validates runtime and artifact behavior.

**Not selected.** No 20k RC has been run. Extrapolated runtime from measured 10k RC (~163 min) suggests a 20k RC would be multi-hour and should precede any production promotion under this policy.

### 2. Decision-stable compromise (`decision_stable_compromise_10000`)

Move to **10,000 draws** with explicit sign-off that the setting is decision-stable but **not** numerically validated by the current strict gate.

**Not selected.** RC operational validation passed, but overlay comparison did not establish a pure draw-count effect (`board_or_contract_identity_mismatch` on `replacement_contract_hash`). Promoting 10k would conflate decision stability with numerical validation the gate explicitly rejects.

### 3. Maintain 1,000 temporarily (`maintain_1000_temporarily`) — **SELECTED**

Retain the current **1,000-draw** profile with a visible release-report risk flag until runtime capacity or a future sampling-design change supports a stronger setting.

**Selected.** Preserves the live board while documenting known draw-count limitations. Phase 2 RC data is archived under `output/model_v3/release_candidates/season=2026/namespace=rc_10k_20260828/` for future reassessment.

## Actions taken

1. `draw_count_rollout_decision.json` updated to `draw_count_rollout_decision_v2` with `phase_2_status: closed`
2. Release pointer profile set to `provisional_current_configuration`
3. Risk flag appended to `output/model_v3/release_report_2026.json` → `summary_risks`

## Rollback vs republish

| Action | Command / mechanism |
|--------|---------------------|
| **Operational rollback** | Restore `output/model_v3/releases/release_2026_previous_1k.json` → `release_2026_current.json` |
| **1k republish** | `python -m src.projection.publish --season 2026 --simulation-draws 1000` (new run; **not** a rollback) |

## Revisit triggers

- Successful **20k RC** with pass validation and acceptable measured runtime
- Sampling-design change that restores strict identity across production and RC board exports
- Explicit policy decision to accept **10k compromise** with documented numerical gate waiver

## Supersession (2026-08-29)

`maintain_1000_temporarily` was superseded by
[`DRAW_COUNT_ROLLOUT_HUMAN_DECISION_2026-08-29.md`](DRAW_COUNT_ROLLOUT_HUMAN_DECISION_2026-08-29.md)
(`decision_stable_compromise_10000`) after the overlay identity hold was resolved and the
3-hour / zero material-core promotion gate passed.
