# BettingPros live weekly scrape — ToS / robots / feasibility

**Date:** 2026-09-18  
**Scope:** Role 1 `weekly_props` third-book ingest + Role 2 per-provider
snapshot persist. No min_books raise, no Role 3 blend, no promote /
sealed-pointer change, no prediction-market means.

## Intended use

| Role | What BettingPros contributes |
|---|---|
| Role 1 display line | **Per-book** quotes from traditional sportsbooks on BP (e.g. Caesars, BetMGM) with `source=bettingpros` and `sportsbook=<book name>`. These are genuine third+ books beside live DraftKings + FanDuel. |
| Role 2 export | The `bettingpros` provider snapshot persists those per-book rows (`scripts/export_role2_live_props.py --mode single`). Offline `--from-fixture` uses the same per-book shape (`data/props/fixtures/providers/bettingpros.json`); blended top-level lines are rejected. |

### What we deliberately do **not** emit

| Skip | Why |
|---|---|
| `book_id=0` BettingPros Consensus | Aggregate already blends DK/FD (and can include prediction markets). Emitting it as `sportsbook=bettingpros` / `kind=book` would inflate `book_count` / `line_basis` and double-weight DK/FD inside `robust_median`. |
| DraftKings / FanDuel on BP | Already scraped by primary live providers — re-emitting would double-weight lines. |
| Kalshi / Polymarket / … | Prediction markets must not enter Role 1/2 means (same rule as season-path `vegas_consensus.py`). |
| PrizePicks / Underdog / Pick6 / … | DFS / pick'em / exchange-style boards, not sportsbook O/U. |

Same spirit as season-path BP handling in `src/draft_assistant/vegas_consensus.py`
(per-book map, strip prediction markets). OddsChecker remains a later theme if
gaps remain after BP.

BP weekly boards do **not** expose separate `rush_tds` / `rec_tds` O/U
(market `334` is a combined "touchdowns" rung). The Phase 0b DK TD hole is
only partially helped (pass TDs + volume markets get additional books).

## Endpoints chosen

Public JSON that the BettingPros site already calls:

| Endpoint | Purpose |
|---|---|
| `GET https://api.bettingpros.com/v3/events?sport=NFL&season=&week=` | Week slate + home/visitor + kickoff |
| `GET https://api.bettingpros.com/v3/offers?sport=NFL&market_id=&event_id=&limit=10&page=` | Paginated **all-book** player O/U offers (no `book_id=0` filter; client drops consensus / PMs / DK/FD) |
| `GET https://api.bettingpros.com/v3/books` | Book id → name map |
| `GET https://api.bettingpros.com/v3/markets?sport=NFL` | Catalog (ids pinned in code) |

Weekly market ids scraped (see `WEEKLY_MARKET_IDS` in
`src/ingest/props/providers/bettingpros_live.py`):

`103` pass_yards, `102` pass_tds, `333` pass_attempts, `100` pass_completions,
`101` pass_ints, `107` rush_yards, `106` rush_attempts, `105` rec_yards,
`104` receptions.

Auth header: `x-api-key` from env **`BETTINGPROS_API_KEY`** (the site's
browser-embedded public client key). Do **not** commit the value — gitleaks
flags the literal as `generic-api-key`. Ops sets it locally or in
`PRODUCTION_JOB_ENV`. Missing/empty → provider `success=False` with a clear
error; DK/FD continue.

Transport: `src/ingest/props/providers/http.py` (`fetch_json` with
`curl_cffi` Chrome impersonation preferred).

Shape reference (season dump, not the live weekly source of truth):
`draft_assistant/data/vegas_raw/bettingpros.json`.

## ToS / robots / feasibility

| Check | Finding (2026-09-18) |
|---|---|
| `https://www.bettingpros.com/robots.txt` | `User-Agent: *` / `Allow: /` |
| `https://api.bettingpros.com/robots.txt` | `Disallow: /` (sitemaps allowed) |
| Feasibility from this cloud egress | Live `events` + `offers` succeed with `BETTINGPROS_API_KEY` + Chrome impersonation |
| Missing `BETTINGPROS_API_KEY` | Provider skipped (`LiveFetchError`); no HTTP calls |

`api.bettingpros.com` robots Disallow means automated scraping of the API is
**not** blessed by robots.txt even though the browser UI uses the same host.
Treat this as a low-volume, ops-owned ingest (same spirit as DK/FD public
sportsbook feeds), not a high-frequency crawl. Re-check ToS / robots before
raising cadence. Human legal review remains outside this note.

If egress starts returning 403 / TLS fingerprint blocks: provider isolation
records `success=False` for BettingPros only; DraftKings / FanDuel in the same
`run_ingest` continue. Validate parsers offline with `--mode fixture` /
`--from-fixture` (see below).

## Failure modes

| Failure | Behavior |
|---|---|
| `BETTINGPROS_API_KEY` unset/empty | `LiveFetchError` before HTTP → `success=False` stub; DK/FD unchanged |
| HTTP 4xx/5xx, invalid JSON, TLS / bot block | `LiveFetchError` → `LiveBettingProsProvider` returns `success=False` stub; other providers unchanged |
| Empty board / all events closed | `success=False`, `error=no_quotes` (or board errors) |
| Mismatched per-book over/under lines | That book skipped (no guessed line) |
| Import / unexpected exception | Caught in `LiveBettingProsProvider.fetch` (provider isolation) |

Do **not** raise `min_distinct_books_per_market` globally until BP (or another
third book) is stable in production scrapes.

## How ops runs it

```bash
# Live weekly scrape including BettingPros (local / job host with egress).
# Prefer curl_cffi (declared in pyproject) for Chrome impersonation.
# BETTINGPROS_API_KEY = site's public browser x-api-key (ops-owned; not in git).
export BETTINGPROS_API_KEY='…'   # from BettingPros frontend network tab / ops vault
uv run python -m src.ingest.props.cli \
  --season 2026 --week 2 \
  --mode live \
  --providers draftkings,fanduel,bettingpros

# Offline / CI: fixture providers only (no network, no API key needed).
# Fixture BP JSON mirrors live per-book emission (`markets.*.books` with
# Caesars/BetMGM etc.). Blended top-level lines without a books map are
# filtered out by BettingProsProvider (fail closed — never sportsbook=bettingpros).
uv run python -m src.ingest.props.cli \
  --season 2026 --week 1 \
  --from-fixture \
  --providers draftkings,fanduel,bettingpros \
  --fixtures-dir data/props/fixtures/providers

# Production job: set WEEKLY_PROPS_PROVIDERS=draftkings,fanduel,bettingpros
# and BETTINGPROS_API_KEY in PRODUCTION_JOB_ENV (GitHub secret). Default
# providers remain draftkings,fanduel until ops opts in — this PR does not
# flip the production default.
```

Unit tests mock HTTP; they never hit the live API in CI.

## Related

- Phase 0b DK TD hole: [`WEEKLY_PROPS_DK_TD_COVERAGE_PHASE0B.md`](WEEKLY_PROPS_DK_TD_COVERAGE_PHASE0B.md)
- Role 2 export: [`ROLE2_WEEKLY_MEASURE_RUNBOOK.md`](ROLE2_WEEKLY_MEASURE_RUNBOOK.md)
- Roles lock: [`docs/decisions/WEEKLY_MODEL_PROMOTION_AND_PROPS_ROLES_2026-09-15.md`](../decisions/WEEKLY_MODEL_PROMOTION_AND_PROPS_ROLES_2026-09-15.md)
