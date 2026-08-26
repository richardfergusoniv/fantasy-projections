# Player sentiment

Diagnostic, point-in-time player sentiment for every projected QB/RB/WR/TE.
The current `markdown_market_v1` snapshot uses the 32 reviewed summaries in
`perplexity research/` plus the contemporaneous ECR-versus-ADP gap in
`data/consensus/consensus_<season>.json`.

The score is **relative within position**, after removing observable depth and
availability structure. `+50` means roughly the 75th percentile of residual
sentiment for that position; it does not mean a 50% projection increase.
No evidence is stored as null/`coverage=none`, never as neutral sentiment.

## Commands

```bash
# Build the standalone snapshot and coverage report
python -m src.sentiment.snapshot --season 2026 --as-of 2026-08-24

# Refresh existing artifacts without changing a projected stat
python -m src.sentiment.refresh_outputs --season 2026 --as-of 2026-08-24
python -m src.draft_assistant.prepare --season 2026
python -m src.team_stats.prepare --season 2026

# Check whether enough historical snapshots exist to attempt an ablation
python -m src.sentiment.gate
```

`predict --as-of` also attaches the latest eligible sentiment directly. The
2026 markdown set has a shared cutoff of 2026-08-24 and is not used for an
earlier as-of date.

## Model gate

`models/sentiment_manifest.json` is the only activation surface. All positions
remain false because only one season exists. The score is visible in CSV/JSON
and the dashboards but does not alter rate, availability, rookie, team-anchor,
fantasy-point, VORP, tier, or ensemble calculations.

Before an end-to-end ablation is allowed, a position needs three distinct
preseason seasons, 200 non-null player-seasons, and 40% coverage. Passing the
data-prerequisite audit alone never activates the feature.
