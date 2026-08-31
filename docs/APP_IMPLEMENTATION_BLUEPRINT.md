# Fantasy Decision App — implementation blueprint

**Status:** Fixture-ready MVP (2026-08-31) — FastAPI API/worker, PostgreSQL migrations, React/Vite PWA (8 screens), scoring compiler, Sleeper fixture sync, availability lifecycle, decision engines, incremental affected-team simulation, projection rollback, simulation partition gates, weekly v2 fixture bridge, OpenAI tool gateway (optional key), Docker Compose (static-validated + healthchecks), CI (Python + PWA + `verify_mvp.py` + `audit_blueprint_mvp.py`). Run `uv run python scripts/audit_blueprint_mvp.py` for the full checklist. That
audit is an inventory check, not production validation: see
[`PRODUCTION_READINESS_AUDIT.md`](PRODUCTION_READINESS_AUDIT.md) for what has
been exercised on real running code versus what is still fixture-only or
blocked on an external credential or runtime.  
**Scope:** Private, mobile-first Sleeper companion for one user across two
redraft and four dynasty leagues.

This document turns the existing projection pipeline and draft assistant into
a hosted decision application. It is an implementation target, not a claim
that the described services already exist.

## 1. Product contract

The application must support:

- Draft recommendations and player comparisons.
- Weekly start/sit recommendations optimized for matchup win probability.
- Both current-lineup and optimized-opponent matchup views.
- Rest-of-season projections updated after each completed NFL week.
- Waiver recommendations and suggested FAAB bids on Tuesday morning.
- Redraft and dynasty trade analysis, including both managers' objective
  benefit, fairness, and estimated acceptance probability.
- Multi-year dynasty values and future rookie-pick distributions.
- League-specific scoring and lineup rules imported from Sleeper.
- Superflex, points per first down, yardage bonuses, kickers, and team defense.
- Automatic depth, availability, and projection publishing.
- Cited injury research that affects weekly and rest-of-season availability.
- A conversational assistant that explains deterministic recommendations.

The application will not submit lineups, waivers, drafts, or trades to
Sleeper. Sleeper is a read-only integration. Push notifications and multi-user
accounts are out of scope for the first release.

### Decision hierarchy

The system must preserve this ordering:

1. Versioned data and deterministic models produce forecasts.
2. League rules turn forecasted stat draws into fantasy-point draws.
3. Decision engines compute lineup, waiver, trade, and draft utility.
4. The AI assistant retrieves and explains those results.

The language model is never the source of record for projections, scores,
trade values, or promotion decisions.

## 2. Existing assets and required changes

### Reuse

- The current repo remains the canonical component-stat and probabilistic
  engine.
- The sibling `fantasy-projections-2` weekly, team-first code is consolidated
  into this repo instead of deployed as a second runtime dependency.
- The release-bundle manifest, validation gates, immutable namespaces, and
  active-pointer promotion pattern remain the publishing foundation.
- The current draft assistant supplies the initial design language, player
  cards, tier/VORP concepts, and projection surfaces.
- Sleeper player status ingestion and depth-chart event detection become the
  first source adapter for the availability service.

### Replace or extend

- Replace the fixed half-PPR scoring dictionary with a compiled,
  league-specific scoring contract.
- Replace permanent status overrides with lifecycle events that can be
  activated and cleared.
- Add weekly and rest-of-season projection releases alongside the preseason
  season board.
- Add first downs, fumbles, two-point conversions where required, team
  defense, and kicker forecast fields.
- Move browser persistence from `localStorage` to server-side league and draft
  state, retaining local storage only as an offline cache.
- Replace the static-only Python HTTP server with an API application and a
  separately hosted static PWA.

## 3. Target topology

```text
                         ┌─────────────────────────────┐
                         │ Mobile PWA                  │
                         │ Draft / Lineup / Waiver /   │
                         │ Trade / Chat                │
                         └──────────────┬──────────────┘
                                        │ HTTPS
                         ┌──────────────▼──────────────┐
                         │ Python API                  │
                         │ Auth, leagues, decisions,   │
                         │ assistant tool gateway      │
                         └───────┬───────────┬─────────┘
                                 │           │
                    ┌────────────▼───┐   ┌──▼────────────────┐
                    │ PostgreSQL     │   │ Object storage    │
                    │ App state,     │   │ Releases, draws,  │
                    │ snapshots/jobs │   │ source snapshots  │
                    └────────────┬───┘   └──▲────────────────┘
                                 │           │
                         ┌───────▼───────────┴─────────┐
                         │ Scheduled worker            │
                         │ Ingest → project → validate │
                         │ → simulate → publish        │
                         └──────────┬──────────────────┘
                                    │
                 ┌──────────────────▼───────────────────┐
                 │ Sleeper / nflverse / schedules /     │
                 │ weather / odds / cited news sources  │
                 └──────────────────────────────────────┘
```

### Deployment units

Use one OCI container image with separate commands:

- `app-api`: long-running API process.
- `app-worker`: on-demand scheduled job process.
- `app-migrate`: database migration process.

The single-user workload does not initially justify Redis, a dedicated queue,
or independent microservices. PostgreSQL stores jobs and uses an advisory lock
to prevent duplicate scheduled runs.

### Proposed application stack

- **API and jobs:** Python 3.14, FastAPI, Pydantic, SQLAlchemy, Alembic.
- **Database:** managed PostgreSQL.
- **Artifacts:** S3-compatible object storage with immutable object names.
- **Frontend:** TypeScript, React, Vite, service worker, installable PWA.
- **Authentication:** email magic link; one authorized user allowlist.
- **AI:** server-side OpenAI Responses API with typed function tools and web
  search. API credentials never reach the browser.

The existing vanilla-JavaScript draft board is ported incrementally rather
than discarded. The initial PWA may embed or wrap unchanged views while new
league, matchup, trade, and assistant screens use the new application shell.

## 4. Target repository layout

```text
src/
  app/
    api/                 # FastAPI routers and response contracts
    auth/                # session and email identity boundary
    jobs/                # scheduler entrypoints and job state
    league/              # Sleeper sync and normalized league rules
    scoring/             # scoring compiler and draw-level scorer
    decisions/           # lineup, waiver, trade, draft services
    assistant/           # Responses API gateway and tool definitions
    persistence/         # ORM models, repositories, migrations
    releases/            # projection pointer/read APIs
  projection/
    weekly/              # consolidated v2 weekly track
    ros/                 # rest-of-season aggregation
    availability/        # status/news evidence and lifecycle policy
    special_teams/       # DST and kicker models
web/
  src/
    app/
    draft/
    matchup/
    waivers/
    trades/
    dynasty/
    assistant/
migrations/
tests/
  app/
  scoring/
  decisions/
  integration/
```

Do not move the existing projection packages during the first phase. Introduce
the app boundary around them, add contract tests, and move code only when the
weekly consolidation makes ownership unambiguous.

## 5. Durable data model

All timestamps are stored as UTC. User-facing schedules use the IANA timezone
`America/Los_Angeles`, which handles PST/PDT automatically.

### Identity and league tables

| Table | Purpose | Important fields |
|---|---|---|
| `app_user` | Authorized application identity | `id`, `email`, `created_at` |
| `sleeper_account` | Stable Sleeper identity | `user_id`, `username`, `last_synced_at` |
| `league` | One Sleeper league/season | `league_id`, `season`, `name`, `type`, `status`, `previous_league_id` |
| `league_rule_snapshot` | Immutable raw + normalized settings | `league_id`, `fetched_at`, `raw_json`, `normalized_json`, `contract_hash` |
| `league_draft_rule` | Dynasty rookie-order policy | `league_id`, `rule` (`max_pf` or `reverse_standings`), `confirmed_at` |
| `league_member` | Manager identity in a league | `league_id`, `user_id`, `roster_id`, `display_name` |
| `roster_snapshot` | Point-in-time roster/start state | `league_id`, `week`, `roster_id`, `fetched_at`, `players`, `starters`, `reserve` |
| `matchup_snapshot` | Sleeper matchup state | `league_id`, `week`, `roster_id`, `matchup_id`, `fetched_at`, `points` |
| `league_transaction` | Completed waivers/trades | `league_id`, `transaction_id`, `type`, `status`, `created_at`, `payload` |
| `traded_pick` | Current future-pick ownership | `league_id`, `season`, `round`, `original_roster_id`, `owner_roster_id` |

`league_rule_snapshot.raw_json` is always retained. A changed Sleeper scoring
key must produce a new normalized contract and invalidate affected cached
decisions.

### Player, status, and evidence tables

| Table | Purpose | Important fields |
|---|---|---|
| `player_identity` | Sleeper/GSIS and display identity | `player_id`, `sleeper_id`, `gsis_id`, `name`, `position`, `team` |
| `player_status_snapshot` | Raw Sleeper player status | `player_id`, `fetched_at`, `status`, `injury_status`, `practice`, `raw_json` |
| `injury_evidence` | Cited report extracted from a source | `id`, `player_id`, `published_at`, `fetched_at`, `source_url`, `source_title`, `claim_json`, `confidence` |
| `availability_event` | Active/cleared availability policy | `id`, `player_id`, `event_type`, `active_from`, `active_until`, `cleared_at`, `source_snapshot_id`, `evidence_ids`, `policy_json` |
| `depth_snapshot` | Versioned derived depth chart | `season`, `as_of`, `artifact_uri`, `content_hash` |

Availability events replace append-only overrides. A valid healthy Sleeper
snapshot may close an active event; an unhealthy or incomplete source payload
must not clear the league's availability state.

### Projection and decision tables

| Table | Purpose | Important fields |
|---|---|---|
| `projection_run` | Immutable forecast identity | `id`, `mode`, `season`, `week`, `as_of`, `model_version`, `input_hash`, `status`, `manifest_uri` |
| `player_projection` | Queryable summary | `run_id`, `player_id`, `team`, `opponent`, `availability_probability`, `mean_json`, `quantiles_json` |
| `simulation_partition` | Draw artifact identity | `run_id`, `partition_key`, `uri`, `sha256`, `draw_count` |
| `active_projection_pointer` | Active run by mode | `mode`, `season`, `week`, `run_id`, `activated_at`, `previous_run_id` |
| `decision_snapshot` | Reproducible recommendation | `id`, `kind`, `league_id`, `week`, `projection_run_id`, `roster_snapshot_id`, `result_json`, `created_at` |
| `manager_state` | Contender/rebuilder inference | `league_id`, `roster_id`, `as_of`, `label`, `probabilities_json`, `features_json`, `overridden_label` |
| `trade_proposal` | Incoming/outgoing manually logged offer | `id`, `league_id`, `created_at`, `created_by_roster_id`, `sides_json`, `direction`, `status`, `countered_by_id` |
| `trade_evaluation` | Frozen trade result | `proposal_id`, `projection_run_id`, `objective_json`, `fairness_json`, `acceptance_json` |
| `manager_tendency` | Smoothed behavioral features | `league_id`, `roster_id`, `as_of`, `sample_size`, `features_json`, `model_version` |

### Job and audit tables

| Table | Purpose |
|---|---|
| `job_run` | Scheduled/manual job status, attempt count, timings, error, and correlation ID |
| `source_snapshot` | Endpoint, request parameters, fetched time, body hash, raw artifact URI, and source health verdict |
| `promotion_event` | Candidate validation, pointer swap, previous pointer, and rollback details |
| `assistant_audit` | Hashed user ID, request class, tools called, source IDs, token usage, latency, and cost estimate |

## 6. Projection contracts

### Forecast modes

| Mode | Horizon | Primary consumers |
|---|---|---|
| `preseason` | Full season before Week 1 | Draft board, initial dynasty values |
| `weekly` | One NFL week | Start/sit, matchup probability, DST/K |
| `ros` | Remaining scheduled weeks | Waivers, redraft trades, contender state |
| `dynasty` | Current year plus three future seasons | Dynasty trades and pick values |

Every output carries `as_of`, `projection_run_id`, model version, input
snapshot hashes, and release status.

### Canonical weekly offensive stat draw

The weekly simulator should produce, when applicable:

- Passing attempts, completions, yards, touchdowns, interceptions, and first
  downs.
- Carries, rushing yards, rushing touchdowns, and rushing first downs.
- Targets, receptions, receiving yards, receiving touchdowns, and receiving
  first downs.
- Fumbles lost and two-point conversions when a connected scoring contract
  requires them.
- Availability and active-game state.

First downs may begin as calibrated conditional-rate draws derived from
player/team opportunities. They must be evaluated with rolling-origin tests
before being used in the PPFD league.

### Team defense draw

The initial intentionally simple DST model uses:

- Opponent-adjusted offensive/defensive EPA per play.
- Sack and pressure probability.
- Turnover probability with strong regression.
- Opponent implied points, QB status, offensive-line context, venue, and
  weather.
- Draw-level points allowed, sacks, turnovers, and defensive touchdowns.

Use an internally calculated opponent-adjusted efficiency metric rather than a
proprietary DVOA feed.

### Kicker draw

The initial kicker model uses:

- Expected team drives and scoring opportunities.
- Red-zone touchdown versus field-goal tendency.
- Kicker accuracy regressed by attempt distance.
- Venue and weather.
- Draw-level field-goal attempts/makes by distance and extra points.

Kicker and DST projections influence weekly decisions but receive negligible
multi-year dynasty value.

## 7. League scoring compiler

The scoring service stores both the raw Sleeper scoring payload and a compiled
contract.

```python
class ScoringContract:
    linear_rules: list[LinearRule]
    threshold_rules: list[ThresholdRule]
    dst_rules: list[DefenseRule]
    roster_slots: list[RosterSlot]
    unsupported_keys: list[str]
    contract_hash: str
```

- Linear rules score each stat unit, including receptions and first downs.
- Threshold rules score yardage bonuses on each draw, never on the mean.
- Roster slots express QB/RB/WR/TE/FLEX/SUPER_FLEX/K/DEF eligibility.
- An unsupported nonzero scoring key blocks recommendation publication for
  that league until mapped or explicitly waived.
- Contract fixtures copied from all six connected leagues become permanent
  regression tests.

Football stat draws are universal. League scoring is a cheap, deterministic
post-process; do not train a separate football model per league.

## 8. Decision engines

### Matchup and start/sit

For each shared simulation draw:

1. Score every available player under the selected league contract.
2. Evaluate the user's current Sleeper starters.
3. Solve the legal maximum-score lineup for both rosters.
4. Compare current and optimized-opponent scenarios.
5. Return win, tie, and loss probabilities plus lineup deltas.

The UI toggle changes the opponent assumption:

- `current`: use the opponent's currently submitted starters.
- `optimized`: use the opponent's optimal legal lineup per recommendation
  policy.

Recommended starts maximize win probability. Expected points, median, floor,
ceiling, and swap regret remain visible diagnostics.

### Waivers

Rank available players by incremental roster utility, not universal rest-of-
season points. Inputs include:

- Weekly and ROS distributions.
- Probability of entering the user's starting lineup.
- Replacement level under the league's slots and scoring.
- Positional depth, bye weeks, injuries, and playoff schedule.
- Contender/rebuilder state in dynasty.
- Sleeper trending adds as a market/urgency signal, not a projection target.

Suggested FAAB is derived from marginal win/playoff utility, remaining budget,
league transaction behavior, and scarcity. It must include a range and
confidence, not false single-dollar precision.

### Dynasty state and future picks

Infer `contender`, `fringe`, `retooling`, or `rebuilding` probabilities from:

- Current and optimized lineup strength.
- ROS win and playoff probabilities.
- Age-adjusted three-year player value.
- Positional depth and scarcity.
- Owned future draft capital.

Allow a user override without deleting the inferred state.

For the upcoming rookie draft:

- `max_pf` leagues rank non-playoff teams using simulated optimal/potential
  points.
- `reverse_standings` leagues simulate record and the league's confirmed
  playoff/final-placement rule.

For later picks, infer early/mid/late probabilities from roster trajectory and
widen the distribution with each additional year.

### Trade evaluation

Keep three separate outputs:

1. **Objective utility:** lineup, ROS, playoff, dynasty, depth, and pick impact
   for each roster.
2. **Fairness:** relative objective gain/loss and uncertainty.
3. **Acceptance probability:** objective benefit plus bounded manager behavior.

Objective roster benefit supplies 75–90% of the acceptance signal. Manager
tendencies may supply 10–25% and may not turn a materially harmful trade into
a recommended offer.

Completed trades import from Sleeper. Pending/incoming proposals are manually
logged because the documented Sleeper API does not expose a proposal inbox.
The UI records offered, accepted, rejected, countered, and expired outcomes.

Manager tendency features include youth/pick preference, positional buying,
consolidation versus depth, package size, contender overpayment, trade timing,
and counter behavior. Use hierarchical shrinkage toward league-wide behavior
until a manager has enough observations.

## 9. Availability and cited research

### Source policy

1. Sleeper status changes trigger evaluation.
2. Cited news research estimates return timing and confidence.
3. A structured policy maps evidence to weekly play probability and ROS games.
4. Projection models redistribute team opportunity.
5. Validation gates decide whether the result is publishable.

The model may extract claims from reporting, but it cannot directly edit
fantasy points or depth shares.

### Evidence contract

```json
{
  "player_id": "...",
  "status": "questionable|doubtful|out|ir|pup|suspended|healthy",
  "reported_injury": "...",
  "expected_return_min": "2026-09-20",
  "expected_return_max": "2026-10-04",
  "claim_confidence": 0.78,
  "sources": [
    {"url": "https://...", "title": "...", "published_at": "..."}
  ]
}
```

Reject uncited return-date claims. Preserve contradictory sources and reduce
confidence instead of silently selecting one.

### Automatic recovery

- Rebuild live depth from the curated base and currently active events.
- Close an event when a healthy primary-source snapshot is valid and complete.
- Never clear events from a failed, truncated, or implausibly small payload.
- Reproject the affected team and any downstream shared simulation ranks.

## 10. AI assistant boundary

Use the OpenAI Responses API from the server. Official OpenAI documentation
supports built-in web search, typed custom function tools, structured JSON
output, multi-turn conversation state, and returning web-search sources:
<https://developers.openai.com/api/reference/cli/resources/responses/methods/create>.

### Initial tools

- `get_league_context(league_id)`
- `get_matchup(league_id, week, opponent_mode)`
- `get_player_projection(player_id, league_id, horizon)`
- `compare_players(player_ids, league_id, horizon)`
- `recommend_lineup(league_id, week, opponent_mode)`
- `recommend_waivers(league_id, week, budget)`
- `evaluate_trade(league_id, sides, horizon)`
- `suggest_trade_counters(trade_evaluation_id)`
- `get_injury_evidence(player_id)`
- `research_injury(player_id, as_of)`
- `explain_projection_change(player_id, from_run_id, to_run_id)`

Tool responses return compact structured results plus IDs for deeper details.
The assistant is prohibited from inventing projections when a tool fails.

### Model routing

- Cost-sensitive model: evidence extraction, classification, summaries, and
  routine player explanations.
- Balanced reasoning model: multi-asset dynasty trades, counteroffers, and
  ambiguous injury synthesis.
- Deterministic code: scoring, optimization, probabilities, values, gates, and
  publication.

Persist tool calls, source IDs, model ID, token usage, estimated cost, and
latency. Store a hashed user identifier rather than sending the email address
as an external safety/cache identifier.

## 11. HTTP API surface

All authenticated endpoints are under `/api/v1`. Read responses include
`data_as_of`, relevant snapshot IDs, and `projection_run_id`.

### Identity and sync

```text
POST /auth/magic-link
GET  /me
POST /sleeper/connect
POST /sync
GET  /jobs/{job_id}
```

### Leagues and rosters

```text
GET  /leagues
GET  /leagues/{league_id}
GET  /leagues/{league_id}/rules
PUT  /leagues/{league_id}/draft-order-rule
GET  /leagues/{league_id}/rosters
GET  /leagues/{league_id}/matchups/{week}
```

### Projections and decisions

```text
GET  /projections/players/{player_id}
GET  /leagues/{league_id}/rankings?mode=weekly|ros|dynasty
GET  /leagues/{league_id}/lineup/{week}?opponent_mode=current|optimized
GET  /leagues/{league_id}/waivers/{week}
GET  /leagues/{league_id}/draft/board
```

### Trades

```text
POST /leagues/{league_id}/trades/evaluate
POST /leagues/{league_id}/trades/proposals
PUT  /leagues/{league_id}/trades/proposals/{proposal_id}/status
GET  /leagues/{league_id}/trades/history
GET  /leagues/{league_id}/managers/{roster_id}/tendencies
```

### Assistant and evidence

```text
POST /assistant/responses
GET  /players/{player_id}/injury-evidence
GET  /players/{player_id}/projection-changes
```

Mutation endpoints require idempotency keys. Projection publication is not
available through the public browser API.

## 12. Scheduled jobs

Schedules use `America/Los_Angeles`.

| Schedule | Job | Scope |
|---|---|---|
| Daily 5:00 PM except Sunday | `daily-refresh` | Sleeper sync, status diff, research changed players, affected projection publish |
| Sunday 8:45 AM | `sunday-early` | Full status sync, early-window evidence, lineup refresh |
| Sunday 11:45 AM | `sunday-afternoon` | Targeted afternoon-player research and affected refresh |
| Sunday 4:00 PM | `sunday-night` | Targeted SNF research and affected refresh |
| Monday 4:00 PM | `monday-night` | MNF status/evidence and lineup refresh |
| Tuesday 5:00 AM | `weekly-close-preliminary` | Ingest completed games, update rolling features, ROS, waivers, FAAB |
| Wednesday 5:00 PM | `weekly-correction` | Stat corrections, final prior-week snapshot, next-week publish |
| On demand | `full-release` | Full projection, simulation, validation, sealed promotion |

Before executing weekly close, verify scheduled games are final. Postpone the
close if the NFL schedule contains a delayed Monday/Tuesday game.

### Job orchestration

Each job:

1. Obtains a PostgreSQL advisory lock.
2. Creates `job_run` with a correlation ID.
3. Persists raw source snapshots before transformation.
4. Produces a candidate projection run.
5. Runs gates.
6. Promotes atomically or leaves the previous pointer active.
7. Records cost, duration, changes, and failure details.

Retries must be idempotent and reuse content-addressed source snapshots.

## 13. Publishing and incremental simulation

Running fewer or selective draws does not create leakage. Leakage controls
depend on information cutoffs and stored forecasts.

### Fast affected-team path

- Diff source/status/depth inputs.
- Construct an impact set of changed players, their team, opportunity peers,
  opponents when weekly context changes, and league decisions that consume
  them.
- Recompute affected stat distributions using stable random seeds.
- Reuse unchanged draw partitions only when their input hash matches.
- Recompute cross-player ranks and league matchup results.
- Seal a new manifest referencing both new and verified reused partitions.

### Full path

Run the full calibrated simulation after weekly close, model changes, scoring
contract changes, or any invalidated shared factor. Preserve the existing
10,000-draw publish profile until a new draw-count gate authorizes a change.

### Promotion gates

- Source health and freshness.
- Schema and scoring-contract coverage.
- Player/team identity and opportunity conservation.
- Projection change magnitude and affected-player explanation coverage.
- Simulation partition hash/run compatibility.
- Matchup probabilities in `[0, 1]` and summing to one.
- No unsupported nonzero league scoring keys.
- Browser/API smoke tests.

Failure keeps the previous active run and exposes the failed job only in the
private operations view.

## 14. Leakage and evaluation controls

- Every source row has `fetched_at`, source publication time where available,
  and a body/content hash.
- Forecast features must satisfy `available_at <= projection_as_of`.
- Rolling features shift before rolling; the target week's outcomes cannot
  enter its inputs.
- Model weights for a scored week are frozen before the week begins.
- Evaluation reads the stored pre-game forecast, never a regenerated one.
- Injury evidence published after kickoff cannot revise the scored pre-game
  forecast.
- Manager acceptance evaluation uses only proposals known at prediction time.
- Future-pick and contender models use rolling-origin seasons.
- League scoring tests compare compiled results against known Sleeper examples
  and hand-calculated fixtures.

Maintain separate scorecards for:

- Weekly point/rank accuracy and calibration.
- Start/sit regret and matchup Brier score.
- Waiver marginal value and FAAB calibration.
- Trade objective-value stability and acceptance calibration.
- Availability probability and return-window calibration.
- DST/kicker accuracy versus simple baselines.

## 15. Frontend/PWA screens

### Primary navigation

1. **Home:** league selector, current matchup, urgent decisions, data freshness.
2. **Lineup:** current/optimized toggle, swap recommendations, player cards.
3. **Waivers:** roster-aware adds/drops, FAAB range, evidence.
4. **Trade Lab:** construct or log a trade, both-side grades, counters,
   acceptance model.
5. **Dynasty:** manager states, three-year rosters, pick distributions.
6. **Draft:** ported current draft assistant and live Sleeper draft context.
7. **Assistant:** persistent league-aware conversation.
8. **Operations:** last sync, active release, failed gates, costs, manual retry.

### Mobile requirements

- Installable PWA with responsive touch targets and bottom navigation.
- Read-only cached last successful recommendations when offline.
- Prominent `as of` and stale-data indicators.
- Source citations open in a new browser tab.
- No recommendation view may hide projection uncertainty.

## 16. Authentication and security

- Email magic-link authentication with an explicit one-user allowlist.
- Secure, HTTP-only, same-site session cookies.
- CSRF protection on mutations and strict CORS origin allowlist.
- Rate limits on authentication, assistant, research, and manual job endpoints.
- Sleeper identifiers are not secrets, but league data remains behind app
  authentication.
- OpenAI and infrastructure credentials remain server-side secrets.
- Database backups and object versioning are enabled.
- Raw web content is treated as untrusted input; it cannot issue tool calls or
  bypass the deterministic decision layer.
- Dependency and container scans run in CI.

## 17. Observability and cost controls

Track:

- Job success, duration, retries, source freshness, and promoted run.
- API latency/error rate by route.
- Projection counts and material changes by run.
- Assistant tokens, searches, model, latency, and estimated dollars.
- Object-storage growth and retained release count.

Initial budget policy:

- Monthly target: `$30–$55`.
- Warning: `$30` estimated month-to-date.
- Soft limit: `$40`; route routine work to the lower-cost model.
- Hard limit: configurable `$50–$60`; disable nonessential autonomous research
  while preserving deterministic projections and user-requested analysis.

Search only status-changed, ambiguous, rostered, or decision-relevant players.
Group evidence research by team/report where possible. Retain daily releases
for 30 days, weekly releases for the season, and named/promoted releases
indefinitely.

## 18. Delivery phases

### Phase 0 — contracts and consolidation

- Add this blueprint and architecture decision records.
- Define weekly stat, scoring, projection-run, and source-snapshot schemas.
- Inventory v2 modules and copy only the weekly components with provenance.
- Add parity tests against the current v2 weekly outputs.
- Fix availability-event clearing before autonomous publishing.

**Exit:** One command produces the same weekly projection from the consolidated
repo, with an immutable as-of manifest.

### Phase 1 — app foundation and Sleeper sync

- Add API, migrations, Postgres repositories, jobs, and email authentication.
- Connect a Sleeper username and import all six leagues.
- Persist scoring, rosters, matchups, transactions, and traded picks.
- Confirm the four dynasty draft-order rules.
- Build scoring contracts and unsupported-key audit.

**Exit:** Authenticated mobile page lists all six leagues, their rosters, slots,
and fully mapped scoring contracts.

### Phase 2 — weekly scoring and matchup decisions

- Add weekly distributions, first downs, nonlinear bonuses, DST, and kicker.
- Add legal lineup optimization and matchup simulation.
- Add current versus optimized-opponent toggle.
- Build stored pre-game evaluation harness.

**Exit:** Start/sit recommendations are available for every connected league;
all scoring keys are covered and matchup probabilities are reproducible.

### Phase 3 — ROS and waivers

- Add weekly-close ingestion and rolling-feature refresh.
- Produce ROS distributions and playoff schedule features.
- Add available-player pool, adds/drops, and FAAB ranges.

**Exit:** Tuesday 5:00 AM job publishes roster-aware waiver recommendations for
all six leagues or safely retains the last release.

### Phase 4 — dynasty and trades

- Add contender/rebuilder inference and overrides.
- Add three-year player values and future-pick distributions.
- Add objective/fairness trade engine.
- Import completed trades; log proposal outcomes; add bounded manager tendency
  model and counteroffers.

**Exit:** A frozen trade evaluation explains objective impact for both sides,
pick movement, fairness, and calibrated acceptance probability.

### Phase 5 — cited research and assistant

- Add injury evidence research, citation persistence, and availability policy.
- Add Responses API tool gateway, model routing, audits, and cost controls.
- Add league-aware chat and projection-change explanations.

**Exit:** Assistant answers draft, lineup, waiver, injury, and trade questions
using only typed app tools; all news claims retain citations.

### Phase 6 — production hardening

- Add incremental simulation manifests and affected-team recomputation.
- Add operations dashboard, backups, rollback rehearsal, load tests, and
  security review.
- Complete PWA install/offline behavior and deploy the independent website.

**Exit:** Scheduled jobs operate for two consecutive weeks without manual
artifact repair, and rollback restores the last good projection pointer.

## 19. First implementation slice

The first code slice should be deliberately narrow:

1. Define `LeagueRuleSnapshot`, normalized `ScoringContract`, and compiler.
2. Add a Sleeper fixture with standard, PPFD, bonus, superflex, K, and DEF
   settings.
3. Compile the fixture and fail on unsupported nonzero keys.
4. Score a small synthetic draw matrix, including first-down and yardage-bonus
   boundary tests.
5. Expose `GET /api/v1/leagues/{league_id}/rules` from an in-memory repository.

This creates the contract on which weekly projections, matchup simulation, and
all six league-specific decision engines depend, without prematurely coupling
the app to a hosting provider.

## 20. Deferred decisions

These do not block Phase 0:

- Hosting vendor and custom domain.
- Exact email provider for magic links.
- The four league IDs and final per-league draft-order confirmation.
- Whether proposal capture begins as a form, PWA share target, or screenshot
  extraction.
- Exact model-family routing after representative cost/quality evaluation.

