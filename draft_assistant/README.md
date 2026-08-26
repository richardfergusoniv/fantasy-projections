# Fantasy Tools

Player cards, the draft board, and the team view now display a
position-relative sentiment score with evidence confidence. It is labeled
**diagnostic** while `models/sentiment_manifest.json` has no active positions;
the score does not change VORP, tiers, suggestions, or projected fantasy
points. See `src/sentiment/README.md` for refresh and gate commands.

Local static app with views powered by **this repo’s rate-forecast (v1) pipeline**, with an optional **v1/v2 draft ensemble** post-process on the board:

- **Draft Assistant** — FantasyPros-style board from `output/fantasy_points_<season>.csv`, blended with archived v2 season points when `output/model_v2/` is present (see Ensemble below)
- **Team Projections** — ESPN-style team stats + depth chart from `output/projections_<season>.csv` (native v1; not blended)
- **Total Projections** — League-wide Passing / Rushing / Receiving leaders by position from the same team_stats JSON

The sibling repo `fantasy-projections-2` is a **different** model (team-first Ridge). Its boards live there (and as read-only copies under `output/model_v2/` when synced). Do not feed v2 CSVs into this app’s canonical `output/` projection paths — the draft blend reads `output/model_v2/` only as a post-process input.

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

### Total Projections

- League-wide Passing / Rushing / Receiving leaderboards (Total or Per Game)
- Position filter (All / QB / RB / WR / TE) plus player/team search
- Team column links into Team Projections; same player cards as that page

## Quick start

Projections come from **this repo** (LightGBM rate-forecast + gates).

```bash
# After projecting the season into output/fantasy_points_*.csv + projections_*.csv
# (and optionally syncing v2 into output/model_v2/):
python -m src.draft_assistant.prepare --season 2026
python -m src.team_stats.prepare --season 2026
python -m src.draft_assistant.compare_prepare --season 2026
python -m src.draft_assistant.serve --open
```

`prepare` **defaults to the v1/v2 draft ensemble** when both
`src/draft_assistant/ensemble_weights.json` and
`output/model_v2/fantasy_points_<season>.csv` exist. Pass `--no-ensemble` for a
native v1-only board. Weights were fit on 2023–2024 OOF with 2025 held out
([TEST_BEFORE_REWRITE_2026-08-24.md](../docs/decisions/TEST_BEFORE_REWRITE_2026-08-24.md));
they do **not** change `compose_board` or LightGBM.

Open http://127.0.0.1:8766/ (Draft), `/teams/` (Team Projections), `/totals/` (Total Projections), or `/compare/` (our ranks vs ECR/ADP).

| Port | Repo | Model |
|------|------|--------|
| **8766** | `fantasy-projections` | v1 rate-forecast |
| **8765** | `fantasy-projections-2` | v2 team-first |

### Optional: archive a v2 board here (comparison + draft ensemble input)

```bash
# Writes ONLY to output/model_v2/ — never overwrites native output/
python -m src.draft_assistant.from_v2 --season 2026 --no-run-project
```

Set `FANTASY_PROJECTIONS_V2` if the sibling repo is not at `../fantasy-projections-2`.
With `output/model_v2/fantasy_points_<season>.csv` present, draft `prepare`
blends v1 and v2 season points using `src/draft_assistant/ensemble_weights.json`.

### Rankings comparison

`/compare/` is a sortable table of our VORP board vs:

- **ECR** — FantasyPros PPR consensus (via nflverse / DynastyProcess)
- **ADP** — Fantasy Football Calculator half-PPR (attribution required)

`Δ ECR` / `Δ ADP` = our rank − market (negative means we like the player more).

```bash
python -m src.draft_assistant.compare_prepare --season 2026 --teams 12
```

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

Position and FLEX tiers are PPG VORP cliffs; overall tiers use overall VORP gaps. See `src/draft_assistant/tiers.py`.

## Scoring

Half-PPR, 4-point passing TD — from this repo’s `src/projection/fantasy_points.py`.
