# Fantasy Tools

Local static app with two views powered by your projection pipeline:

- **Draft Assistant** — FantasyPros-style board from `output/fantasy_points_<season>.csv`
- **Team Projections** — ESPN-style team stats + depth chart from `output/projections_<season>.csv`

## Features

### Draft

- **VORP overall board** — All tab ranks by value over replacement for **1QB / 2RB / 3WR / 1TE / 1FLEX**, not raw PPG (so elite RBs/WRs outrank high-scoring QBs).
- **Player cards** — hover or click a name for fantasy drivers, volume scales, VORP, and context (same cards as Team Projections).
- **Tiered rankings** — All, position, and FLEX tiers use PPG VORP cliffs.
- **Draft checkboxes** — mark players drafted; state persists in your browser.
- **Snake draft tracking** — set league size, your draft slot, and current pick to see who is on the clock.
- **Suggested picks** — blends VORP, positional need, and tier.
- **Roster builder** — QB/RB/WR/TE/FLEX/BN slots fill as you draft yourself.

### Team Projections

- Team picker with Passing / Rushing / Receiving tables (Total or Per Game)
- ESPN-style depth chart (Starter / 2nd / 3rd / 4th)
- Player hover cards and fullscreen detail modal

## Quick start

```bash
# 1. Export JSON after projections exist
python -m src.draft_assistant.prepare --season 2026
python -m src.team_stats.prepare --season 2026

# 2. Serve the combined app
python -m src.draft_assistant.serve --open
```

Open http://127.0.0.1:8765/ (Draft) or http://127.0.0.1:8765/teams/ (Team Projections).

`python -m src.team_stats.serve` is an alias for the same combined server.

## Draft workflow

1. Set **Teams**, **Your slot**, and **Current pick** in the header (Teams also rescales VORP baselines).
2. Use **All** for positional-value order; QB/RB/WR/TE/FLEX for position PPG boards.
3. Check a player when they are drafted — your pick is tagged when you are on the clock.
4. Use **Suggested picks** for quick adds during your turn.
5. **Undo pick** / **Reset draft** as needed; progress saves to `localStorage`.

## VORP baselines

`VORP = max(0, fantasy_pts − replacement pts/game)`.

Replacement rank for an N-team league:

`floor(N × starters + N × flex_share) + 1`

| Position | Starters | FLEX share | Replacement @ 12 teams |
|----------|----------|------------|-------------------------|
| QB | 1 | 0.00 | QB13 |
| RB | 2 | 0.40 | RB29 |
| WR | 3 | 0.50 | WR43 |
| TE | 1 | 0.10 | TE14 |

Defaults live in `src/draft_assistant/vorp.py`. The browser recomputes when you change **Teams**.

## Tier thresholds

| Scope   | Rule                                      |
|---------|-------------------------------------------|
| Overall (VORP) | 0.75 PPG-VORP drop or 4% relative |
| QB (VORP) | 0.85 PPG-VORP or 3%                    |
| RB (VORP) | 0.75 PPG-VORP or 3%                    |
| WR (VORP) | 0.55 PPG-VORP or 3%                    |
| TE (VORP) | 0.50 PPG-VORP or 3%                    |
| FLEX (VORP) | 0.65 PPG-VORP or 3% (RB/WR/TE)       |

Adjust in `src/draft_assistant/tiers.py` / `vorp.py` and re-run `prepare`.

## Scoring

Half-PPR, 4-point passing TD — matches the projection pipeline in `src/projection/fantasy_points.py`.
