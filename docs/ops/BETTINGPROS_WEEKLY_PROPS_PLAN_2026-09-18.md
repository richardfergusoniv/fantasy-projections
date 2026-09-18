# Plan — BettingPros weekly props (Role 1, opt-in)

**Date:** 2026-09-18  
**Theme:** Eng — BettingPros for Role 1 `weekly_props`  
**Gate verdict target:** `not_promoting`

## Goal

Add a BettingPros live weekly game-prop scrape path that emits the same
`NormalizedQuote` / `ProviderSnapshot` contracts as DraftKings + FanDuel, and
can join Role 1 `robust_median` when explicitly enabled.

## Non-goals (hard)

- No promote / `APP_PROJECTION_SOURCE` flip / sealed League Value / Role 3 blend
- Do **not** raise `min_distinct_books_per_market` globally
- OddsChecker, third US book extraction, Kalshi/Polymarket — out of theme
- No invented credentials committed

## Approach (chosen)

**Browser/HTML scrape of BettingPros pages**, parsing the embedded Island
bootstrap JSON the site already SSR's into the document — same family of
evidence as the 2026-09-05 draft-assistant BP dump (which recorded both
`www.bettingpros.com/.../player-props/` and `api.bettingpros.com/v3/*`).

Why not API-only:

1. Direct `api.bettingpros.com/v3/offers` returns **403 Forbidden** from typical
   cloud egress even with Chrome TLS impersonation.
2. Market board HTML returns **200** and embeds offers + `offersPagination`
   next links into `/v3/offers`.
3. Per-player analyzer pages
   (`/nfl/props/{slug}/{market-slug}/`) embed `participantPropOffer` with
   BettingPros Consensus (`book_id=0`) lines — full coverage without the API.

Fallback: attempt `/v3/offers` pagination via existing `fetch_json` /
`curl_cffi` when the environment allows; soft-fail and continue.

## Market map (game period)

| Canonical | BP slug | BP `market_id` | Notes |
|---|---|---|---|
| `pass_yards` | `passing-yards` | 103 | |
| `pass_tds` | `passing-touchdowns` | 102 | |
| `pass_attempts` | `passing-attempts` | 333 | |
| `pass_completions` | `passing-completions` | 100 | |
| `rush_yards` | `rushing-yards` | 107 | |
| `rush_attempts` | `rushing-attempts` | 106 | |
| `rec_yards` | `receiving-yards` | 105 | |
| `receptions` | `receptions` | 104 | |
| `rush_tds` / `rec_tds` | — | — | **Not** in BP weekly O/U catalog (only any-TD / first-TD style markets). Do not claim TD-hole fill until fixtures prove otherwise. |

Quote book identity: `sportsbook="bettingpros"` from Consensus `book_id=0`
("BettingPros Consensus"). Nested DK/FD lines inside BP are **not** re-emitted
(would double-count Role 1 median).

## Enablement

- Default `WEEKLY_PROPS_PROVIDERS` stays `draftkings,fanduel`.
- Opt-in: append `bettingpros` (Richard fills any job-env change).
- Fail-closed **for BP only** if scrape fails; other books still succeed.
- Snapshot load sources include `bettingpros` when a catalog row exists.

## TDD / deliverables

1. Ops ToS/feasibility note under `docs/ops/`
2. `bettingpros_live.py` parsers + `LiveBettingProsProvider` + fixture JSON
3. Wire `LIVE_CAPABLE` / `build_providers` / persist load sources / runbook links
4. Draft PR, `gate_verdict=not_promoting`

## Risks

- Aggregator consensus partially correlates with DK/FD → median not independent;
  documented, not a blocker while `min_books=1` and BP is opt-in.
- HTML/API shape can change; board-level error isolation mirrors DK/FD.
- Participant fill is chatty; polite pacing + board-first path.
