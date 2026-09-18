# Phase 0b — DraftKings weekly TD coverage hole (DK + FD only)

**Date:** 2026-09-18
**Scope:** ops note for Role 1 weekly multi-book consensus. No third book,
no Role 3 blend, no promote / sealed-pointer change.

## What Role 1 does

For each `(gsis, season, week, market)`, the weekly_props path builds a
display line with `src.projection.market_quotes.robust_median` over accepted
DraftKings + FanDuel quotes (`src/projection/weekly_props/consensus.py`).

| Books present | Role 1 line | `line_basis` |
|---|---|---|
| DK + FD | median of the two | `robust_median` |
| One book only | that book’s line | `single_book` |
| Neither | baseline / omitted | — |

Per-book quotes stay on provider snapshots (and Role 2 `--mode single`
export). Role 2 compare must not book-shop MAE; the comparator collapses
multi-book duplicates with the same `robust_median`, not “latest as_of wins.”

## The hole

Live **DraftKings weekly** game-prop boards do **not** currently expose
`rush_tds` / `rec_tds` (see `WEEKLY_OU_BOARDS` in
`src/ingest/props/providers/draftkings_live.py` — yards / receptions / pass
TDs only). Those TD markets appear on DK **season futures**, not the weekly
O/U boards we scrape for Role 1.

Live **FanDuel** weekly maps include `RUSHING_TOUCHDOWNS` / `RECEIVING_TOUCHDOWNS`
(`fanduel_live.py`), so many player-weeks have FD-only rush/rec TD lines.

With `min_distinct_books_per_market = 1` (unchanged), an FD-only TD quote is
still a valid Role 1 line (`line_basis=single_book`, `book_count=1`). Calling
that “consensus” is accurate only in the weak sense that it is the only
eligible book — not a median of two.

## Why we do **not** raise `min_books` yet

Raising `min_distinct_books_per_market` to 2 would **drop** most weekly
rush/rec TD lines from the board until a third source (or DK weekly TD
coverage) appears. That is a coverage regression, not a quality upgrade.

Phase 0b documents the hole. Do **not** raise the global min until a third
eligible book is online. No BettingPros / OddsChecker / prediction markets
in this phase.

## Related

- Roles lock: [`docs/decisions/WEEKLY_MODEL_PROMOTION_AND_PROPS_ROLES_2026-09-15.md`](../decisions/WEEKLY_MODEL_PROMOTION_AND_PROPS_ROLES_2026-09-15.md)
- Role 2 export: [`docs/ops/ROLE2_WEEKLY_MEASURE_RUNBOOK.md`](ROLE2_WEEKLY_MEASURE_RUNBOOK.md)
