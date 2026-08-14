# Draft Assistant

FantasyPros Draft Wizard-style board powered by your `output/fantasy_points_<season>.csv` projections.

## Features

- **Tiered rankings** — players with similar projected points per game are grouped; tier breaks appear when projections drop by a position-specific threshold (or 4% for overall board).
- **Draft checkboxes** — mark players drafted; state persists in your browser.
- **Snake draft tracking** — set league size, your draft slot, and current pick to see who is on the clock.
- **Suggested picks** — blends projection value, positional need, and tier.
- **Roster builder** — QB/RB/WR/TE/FLEX/BN slots fill as you draft yourself.

## Quick start

```bash
# 1. Export projections to JSON (after fantasy_points CSV exists)
python -m src.draft_assistant.prepare --season 2026

# 2. Serve the app
python -m src.draft_assistant.serve --open
```

Open http://127.0.0.1:8765/ if the browser does not launch automatically.

## Workflow

1. Set **Teams**, **Your slot**, and **Current pick** in the header.
2. Use position tabs (QB/RB/WR/TE) for position-specific tiers and ranks.
3. Check a player when they are drafted — your pick is tagged when you are on the clock.
4. Use **Suggested picks** for quick adds during your turn.
5. **Undo pick** / **Reset draft** as needed; progress saves to `localStorage`.

## Tier thresholds

| Scope   | Rule                                      |
|---------|-------------------------------------------|
| Overall | 1.0 pt drop or 4% relative drop           |
| QB      | 0.85 pt or 3%                             |
| RB      | 0.75 pt or 3%                             |
| WR      | 0.55 pt or 3%                             |
| TE      | 0.50 pt or 3%                             |
| FLEX    | 0.65 pt or 3% (RB/WR/TE combined)         |

Adjust in `src/draft_assistant/tiers.py` and re-run `prepare`.

## Scoring

Half-PPR, 4-point passing TD — matches the projection pipeline in `src/projection/fantasy_points.py`.
