# Live shadow-week tracker (Role 2) — 2026-09-16

**Status:** research / shadow. Gate verdict: **not_promoting**.
**Starts at:** 0 credited weeks (of 6–8 required by the promotion note).
**Machine-readable:** [`output/shadow_weekly_schedule_m3/live_shadow_weeks.json`](../../output/shadow_weekly_schedule_m3/live_shadow_weeks.json)

Locked bar:
[`docs/decisions/WEEKLY_MODEL_PROMOTION_AND_PROPS_ROLES_2026-09-15.md`](../decisions/WEEKLY_MODEL_PROMOTION_AND_PROPS_ROLES_2026-09-15.md) §3.

## How a week is credited

A `(season, week)` is credited **only** when all of the following are true:

1. An M3 shadow Role 2 board exists for that week (player-week means; no same-week outcome columns).
2. Timestamped Vegas weekly-prop snapshots exist for that week with **`as_of` ≤ `kickoff_at`**.
3. `scripts/compare_shadow_vegas_props.py` (or `compare_m3_to_vegas_props`) runs with **`n_matched > 0`**.
4. The snapshot source is **`live_timestamped`** (allow-listed). `supplied`, fixture, synthetic, and dry-run labels never credit.

Fixture and M3 dry-run compares prove the harness (matching + fail-closed leakage). They **do not** increment `credited_weeks`.

Outcomes (`actual`) are required for vs-actual MAE / CRPS / Brier / coverage. A week may take a first credit from model-vs-market alone; attach outcomes after the week completes before treating the week as promotion-gate evidence.

Re-credit is idempotent per `(season, week)`. The gate stays `not_promoting` until the decision-note bar clears. One or two 2026 weeks update the model; they do not declare superiority.

## How to credit

Do not hand-edit the JSON to invent weeks. After a live compare:

```bash
# After a live Role 2 run that returned n_matched > 0
# use src.projection.weekly_eval.live_weeks.credit_live_week
# then write_tracker(...) — see docs/ops/ROLE2_WEEKLY_MEASURE_RUNBOOK.md
```
