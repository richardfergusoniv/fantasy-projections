# BettingPros weekly scrape — ToS / feasibility (Role 1)

**Date:** 2026-09-18  
**Scope:** ops / research note for adding BettingPros as an **opt-in** Role 1
weekly props provider. Not a promotion. Does not raise
`min_distinct_books_per_market`. OddsChecker / prediction markets remain out.

## What we are scraping

Public NFL **game** player-prop odds pages on BettingPros, for example:

- Board: `https://www.bettingpros.com/nfl/odds/player-props/{slug}/`
- Analyzer: `https://www.bettingpros.com/nfl/props/{player-slug}/{slug}/`

The site SSR-embeds a large Island bootstrap JSON blob (sports, markets,
events, `offers`, `offerCounts`, `books`, `offersPagination`) into the HTML.
That is the scrape surface — **browser/page style**, not a licensed odds API
we own credentials for.

The 2026-09-05 draft-assistant artifact
(`draft_assistant/data/vegas_raw/bettingpros.json`) already recorded both the
HTML player-props URLs and `https://api.bettingpros.com/v3/offers|markets`.
The live scrape code from that swarm was **not** committed; only the dump was.

## Why not API-only

| Probe (2026-09-18, cloud egress) | Result |
|---|---|
| `GET api.bettingpros.com/v3/markets?sport=NFL` | **403** `Forbidden` (plain HTTP and `curl_cffi` Chrome impersonation) |
| `GET api.bettingpros.com/v3/offers?...` | **403** even after visiting www and sending Referer/Origin |
| `GET www.bettingpros.com/nfl/odds/player-props/...` | **200**, ~3MB HTML with embedded offers (first page, `limit=5`) |
| `GET www.bettingpros.com/nfl/props/{slug}/{market}/` | **200**, embeds `participantPropOffer` with Consensus line |

So an API-only client fails closed on this egress class. HTML/bootstrap parse
is the feasible path that matches “browser/scrape style like the draft
assistant.” We still **attempt** `/v3/offers` pagination when the environment
allows (same TLS helper as DraftKings); failure is soft and does not abort DK/FD.

No API key / cookie secret is invented or committed. If Richard later adds a
job-env secret that unlocks the API, wire it through existing env patterns —
do not bake credentials into the repo.

## robots.txt / Terms of Use (honest read)

- `https://www.bettingpros.com/robots.txt` (fetched 2026-09-18):  
  `User-Agent: *` / `Allow: /` — no Disallow rules observed.
- Terms of Use (`/terms/`): public marketing/legal shell; this note does **not**
  claim a clean commercial-license grant to redistribute book lines at scale.
  Treat the integration as **internal Role 1 consensus input**, polite rate
  limits, fail-closed per provider, and reversible by removing `bettingpros`
  from `WEEKLY_PROPS_PROVIDERS`.
- Precedent in-repo: DK live already uses TLS impersonation against public
  sportsbook JSON; BP HTML parse is the analogous “public page → structured
  quotes” adapter, not a new sealed-board rewrite.

**Human judgment still owns** whether production job egress should hit BP.
This PR keeps BP **opt-in** (default providers remain `draftkings,fanduel`).

## Intended market map (game O/U)

Confirmed present on the weekly `player-props` catalog (`period=game`):

| Canonical market | BP slug | `market_id` |
|---|---|---|
| `pass_yards` | `passing-yards` | 103 |
| `pass_tds` | `passing-touchdowns` | 102 |
| `pass_attempts` | `passing-attempts` | 333 |
| `pass_completions` | `passing-completions` | 100 |
| `rush_yards` | `rushing-yards` | 107 |
| `rush_attempts` | `rushing-attempts` | 106 |
| `rec_yards` | `receiving-yards` | 105 |
| `receptions` | `receptions` | 104 |

### TD coverage hunch — **not claimed**

Live DraftKings weekly boards omit `rush_tds` / `rec_tds` (Phase 0b). BP’s
weekly catalog has **no** dedicated rushing/receiving TD O/U markets (slugs
like `rushing-touchdowns` / `receiving-touchdowns` return empty offers). It
has any-TD / first-TD / last-TD style props instead, which are **not** mapped
into Role 1 `rush_tds` / `rec_tds`. Season-long BP futures do have
`rush_tds` / `rec_tds`, but those are out of scope for weekly Role 1.

Do not advertise “BP fills the DK TD hole” until fixtures prove a weekly
rush/rec TD O/U mapping.

## Quote contract

- Emit `NormalizedQuote` with `source="bettingpros"`, `sportsbook="bettingpros"`,
  `period="game"`, Consensus `book_id=0` line + over/under American odds.
- Do **not** unpack nested DraftKings/FanDuel rows from BP into separate
  sportsbooks (would double-count Role 1 `robust_median`).
- BP is an **aggregator**; its consensus correlates with DK/FD. Opt-in +
  documented. Raising `min_distinct_books_per_market` still waits for a true
  independent third book if that is the goal.

## Failure modes

| Mode | Behavior |
|---|---|
| Board HTML fetch fails | Board error in metadata; try other markets |
| Embedded JSON missing / shape drift | Board error; provider `success=False` if zero quotes |
| `/v3/offers` 403 | Soft-skip pagination; use board page + analyzer fill |
| Analyzer fill partial | Quotes from successful pages only |
| BP omitted from `WEEKLY_PROPS_PROVIDERS` | Provider not constructed (default) |
| BP enabled but fails | `live_fetch_stub` / empty success; **DK/FD still ingest** |

## Enablement

```text
WEEKLY_PROPS_PROVIDERS=draftkings,fanduel,bettingpros
```

Scheduled jobs read `PRODUCTION_JOB_ENV` (not Vercel). Richard sets secrets /
job env; this repo only documents the knob.

## Related

- Plan: [`BETTINGPROS_WEEKLY_PROPS_PLAN_2026-09-18.md`](BETTINGPROS_WEEKLY_PROPS_PLAN_2026-09-18.md)
- Phase 0b DK TD hole: [`WEEKLY_PROPS_DK_TD_COVERAGE_PHASE0B.md`](WEEKLY_PROPS_DK_TD_COVERAGE_PHASE0B.md)
- Roles lock: [`docs/decisions/WEEKLY_MODEL_PROMOTION_AND_PROPS_ROLES_2026-09-15.md`](../decisions/WEEKLY_MODEL_PROMOTION_AND_PROPS_ROLES_2026-09-15.md)
- App ops: [`docs/APP_OPERATIONS_RUNBOOK.md`](../APP_OPERATIONS_RUNBOOK.md)
