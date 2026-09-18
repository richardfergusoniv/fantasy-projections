# Shadow Vegas weekly props comparator (Role 2)

Research/shadow only. This directory is **not** a sealed board, not a Vegas
replacement, not a PWA input, and not a Role 3 blend.

- Schema: `vegas_weekly_props_eval_v1`
- Role: evaluation comparator
- Gate verdict: `not_promoting`

Regenerate from committed synthetic fixtures:

```bash
uv run python scripts/compare_shadow_vegas_props.py --dry-run
uv run python scripts/compare_shadow_vegas_props.py --m3-dry-run
```

Live weeks: see [`docs/ops/ROLE2_WEEKLY_MEASURE_RUNBOOK.md`](../../docs/ops/ROLE2_WEEKLY_MEASURE_RUNBOOK.md).

```bash
uv run python scripts/export_role2_live_props.py --season 2026 --week 2
```

Do not credit fixture runs toward the 6–8 live-shadow bar.
