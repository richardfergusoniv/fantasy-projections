# Draw-count rollout — human decision revisit (10k compromise)

**Date:** 2026-08-29  
**Decided by:** Richard  
**Season:** 2026  
**Artifact:** `output/model_v3/draw_count_rollout_decision.json` (schema `draw_count_rollout_decision_v2`)  
**Supersedes operational policy in:** [`DRAW_COUNT_ROLLOUT_HUMAN_DECISION_2026-08-28.md`](DRAW_COUNT_ROLLOUT_HUMAN_DECISION_2026-08-28.md)

## Current production label

```text
decision_stable_compromise_10000
```

Production moves to **10,000 draws** under an explicit decision-stable compromise (not strict numerical validation vs 20k).

## Revisit trigger

Phase 2 closed on `maintain_1000_temporarily` partly because the 1k↔RC-10k overlay comparison held on `replacement_contract_hash`. Root cause: `generated_at` was included in the replacement contract hash body, so every rebuild mismatched even when board/run identity matched.

After excluding wall-clock fields from the hash:

| Check | Result |
|-------|--------|
| Overlay comparison (1k vs RC-10k) | **compare** — identity aligned; metric deltas interpretable |
| Nested-prefix 10k vs 20k | `material_decision_events=0`, `core_adp_decision_events=0` |
| Measured RC-10k runtime | **9656s (~161 min)** ≤ **3h scheduled-run budget** |
| Strict numerical gate vs 20k | Still **fail** (unchanged; not waived as numerical pass) |

## Operational policy selected

### Decision-stable compromise (`decision_stable_compromise_10000`) — **SELECTED**

Promote to **10,000 draws** because the clean overlay compare restores a pure draw-count comparison, nested-prefix decision stability at 10k shows **zero material and zero core-player decision events**, and measured runtime stays within the **three-hour** budget.

**Not selected:** `strict_numerical_policy_20000` (no 20k RC; multi-hour extrapolated runtime).  
**Not selected:** `maintain_1000_temporarily` (revisit criteria met).  
**Deferred:** variance-reduction project (option C) — only if 10k/20k become impractical later.

## Actions

1. Update `draw_count_rollout_decision.json` with clean overlay comparison + promotion gate
2. Set release pointer profile to `decision_stable_compromise_10000`
3. Default production publish draws to **10000**
4. Production republish: `python -m src.projection.publish --season 2026 --simulation-draws 10000`
5. Replace provisional 1k risk flag with the 10k compromise risk flag on the release report

## Rollback

| Action | Command / mechanism |
|--------|---------------------|
| **Operational rollback** | Restore `output/model_v3/releases/release_2026_previous_1k.json` → `release_2026_current.json` |
| **1k republish** | `python -m src.projection.publish --season 2026 --simulation-draws 1000` (new run; **not** a rollback) |
