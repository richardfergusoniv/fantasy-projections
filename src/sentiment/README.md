# Player sentiment

Diagnostic, point-in-time player sentiment for every projected QB/RB/WR/TE.
The current `markdown_market_v1` snapshot uses the 32 reviewed summaries in
`perplexity research/` plus the contemporaneous ECR-versus-ADP gap in
`data/consensus/consensus_<season>.json`.

Cross-team daily summaries for 2026-08-26 through 2026-08-29 are preserved in
`perplexity research/daily/` and imported separately into
`data/sentiment/ledger/legacy_daily_2026.jsonl`. They contain placeholder
citations rather than recoverable source URLs, so every row is permanently
`legacy_unverified` and `training_eligible=false`. They do not change the
active diagnostic snapshot until the daily set is complete and explicitly
promoted into the diagnostic aggregation.

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

# Import the dated legacy-unverified daily reports
python -m src.sentiment.ledger --season 2026 --daily
```

`predict --as-of` also attaches the latest eligible sentiment directly. The
2026 markdown set has a shared cutoff of 2026-08-24 and is not used for an
earlier as-of date.

## Snapshot history

`snapshot.py` writes a dated `data/sentiment/sentiment_<season>_<as-of>.csv`
alongside a coverage summary. **These files are tracked on purpose.** They are
the point-in-time record the model gate below is waiting on: a new as-of date
produces a new file, so re-running never destroys an earlier observation.

`/data/*` is ignored wholesale, so `data/sentiment/` needs the explicit negation
in `.gitignore` to survive. Until that negation existed, every run wrote to an
untracked path and the three-season history could not accumulate at all -- the
gate was waiting on evidence the repo was discarding.

## Model gate

`models/sentiment_manifest.json` is the only activation surface. All positions
remain false because only one season exists. The score is visible in CSV/JSON
and the dashboards but does not alter rate, availability, rookie, team-anchor,
fantasy-point, VORP, tier, or ensemble calculations.

Before an end-to-end ablation is allowed, a position needs three distinct
preseason seasons, 200 non-null player-seasons, and 40% coverage. Passing the
data-prerequisite audit alone never activates the feature.
