# 2024 WR Sentiment Feasibility Spike — Search Protocol

This protocol applies identically to all **105** frozen wide receivers in
`data/sentiment/spike/population_2024_wr.json`. Follow the shuffled
`collection_order` in that file so a partial run remains an unbiased sample.

## Cutoff

Evidence must be published **on or before** `2024-09-04T00:00:00Z` (day before
2024 Week 1 kickoff). This is an upper bound for the Week 1 roster population;
real drafts occur weeks earlier. Coverage measured here does not transfer to a
draft-time overlay without re-measurement at an earlier cutoff.

## Source classes (query in order)

1. Team beat writers / local newspapers (game-week and camp coverage)
2. National outlets with dated articles (ESPN, NFL.com, The Athletic if accessible)
3. Team official sites / press conferences with dated transcripts
4. Archived social posts only when the original URL and timestamp are recoverable

Stop after **three** distinct source classes yield no player-specific hit.

## Per-player workflow

1. Record `player_id`, `display_name`, and `band` from the frozen population.
2. Run the source-class queries above with the player name + team + role keywords.
3. On a hit, capture:
   - `source_url` (original or archived)
   - verbatim `excerpt` (player-specific, role/usage/development/coach sentiment)
   - `publication_timestamp` (ISO-8601, unambiguous)
   - `captured_content_hash` (SHA-256 of archived body text)
   - `reviewer` (initials)
   - `source_class` (beat / national / team_site / social)
4. Reject any excerpt containing 2024 outcome statistics or retrospective season language.
5. On a miss after the source-class budget, record a miss row in the attempts log:
   - `player_id`, `verified: false`, `miss_reason` (no_coverage | paywall | no_timestamp | off_topic | timebox)

## Verified claim schema

Extend the Phase B ledger row with:

```json
{
  "evidence_tier": "verified",
  "training_eligible": true,
  "source_url": "...",
  "excerpt": "...",
  "publication_timestamp": "...",
  "captured_content_hash": "...",
  "reviewer": "...",
  "source_class": "beat"
}
```

Validate with:

```bash
python scripts/validate_spike_claims.py data/sentiment/spike/claims_2024_wr.jsonl
```

## Timebox

Five working days or 40 reviewer hours, whichever comes first. Unattempted players
count as uncovered. A partial run can only fail the gate, never pass it.

## Stop rule

- **&lt; 40%** verified-player coverage (`verified_numerator / 105`): stop all backfill
  and overlay work; record `historical_collection_infeasible` in the manifest.
- **≥ 40%**: stop after the feasibility report; estimate full 2023–2025 RB/WR/TE
  labor from measured hours-per-covered-player and seek a fresh go/no-go.

Passing does **not** authorize backfill, modeling, ranking changes, or overlays.
