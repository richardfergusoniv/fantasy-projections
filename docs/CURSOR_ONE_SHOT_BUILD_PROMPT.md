# Cursor one-shot build prompt

Copy everything below the divider into Cursor Agent mode from the repository
root.

---

You are the lead engineer responsible for turning this repository into the
private, mobile-first fantasy football decision application specified in
`docs/APP_IMPLEMENTATION_BLUEPRINT.md`.

Your mission is to implement the complete production MVP in this repository,
not merely write a plan or create disconnected scaffolding. Work autonomously
through the full build, make reasonable implementation decisions within the
established architecture, run the relevant tests, fix failures, and leave the
repository in a coherent, runnable state.

## Authoritative context

Before changing code, read these files completely:

1. `docs/APP_IMPLEMENTATION_BLUEPRINT.md` — product and target architecture.
2. `README.md` — current commands and release workflow.
3. `STATE_OF_BUILD.md` — current model roles and production constraints.
4. `docs/PIPELINE_MAP.md` — current projection and release flow.
5. `draft_assistant/README.md` — existing frontend behavior.
6. `src/depth_chart/README.md` — current injury/depth refresh policy.
7. `pyproject.toml` and `.github/workflows/ci.yml` — runtime and CI contract.

Treat `docs/APP_IMPLEMENTATION_BLUEPRINT.md` as authoritative for the new app.
Treat current release manifests, promotion gates, frozen model decisions, and
existing tests as authoritative for existing production behavior unless the
blueprint explicitly replaces them.

The sibling repository `../fantasy-projections-2` may be inspected as a source
for the existing team-first weekly implementation. Port the required weekly
modules into this repository with provenance and parity tests. The deployed app
must not depend on the sibling repository being present.

## Working-tree safety

- Inspect `git status` before editing.
- Preserve every pre-existing tracked and untracked user change.
- Never run `git reset --hard`, destructive checkout commands, broad deletes,
  or commands that overwrite unrelated work.
- Do not modify or reseal an existing release bundle in place.
- Keep generated artifacts out of Git unless the existing repository contract
  explicitly tracks them.
- Use noninteractive commands and make all migrations repeatable.
- Do not commit or push unless explicitly requested. Deliver a working tree
  that the user can review.

## Execution behavior

- Begin with a concise implementation plan, then immediately execute it.
- Do not stop after architecture, schemas, TODOs, or mocked UI screens.
- Continue through implementation, integration, verification, and docs.
- Do not ask questions already answered by the blueprint.
- If a credential, league ID, hosting vendor, or external account is missing,
  implement the adapter, environment contract, development fallback, and tests;
  document the one remaining configuration step and continue with everything
  else.
- Never fabricate model calibration, source evidence, completed jobs, or
  external API results. Use clearly labeled fixtures in tests and development.
- Prefer a smaller complete vertical slice over broad dead scaffolding, but
  complete every required workstream below to production-MVP depth.

## Product requirements

Build a private application for one authenticated user across two redraft and
four dynasty Sleeper leagues. It must support:

- Mobile-first draft assistant.
- Weekly start/sit recommendations optimized for matchup win probability.
- A UI toggle between the opponent's current lineup and an optimized opponent
  lineup.
- Tuesday morning waiver recommendations with suggested FAAB ranges.
- Rest-of-season projections.
- Redraft and dynasty trade analysis for both managers.
- Objective trade benefit, fairness, and bounded acceptance probability.
- Manager tendency learning from completed league trades and manually logged
  incoming/outgoing proposals and outcomes.
- Contender, fringe, retooling, and rebuilding inference with user override.
- Multi-year dynasty values and future rookie-pick distributions.
- Two Superflex leagues.
- One points-per-first-down league.
- One league with yardage bonuses.
- Five leagues with team defense and four with kickers.
- Four dynasty draft-order policies: two max-PF and two reverse standings,
  configurable per league.
- Sleeper-driven rosters, settings, scoring, matchups, transactions, drafts,
  and future picks.
- Sleeper-triggered availability changes augmented by current, cited injury
  research.
- Automatic validated publishing and rollback to the last good release.
- An AI assistant that explains deterministic app results but never invents or
  overrides them.
- Email magic-link login.
- No push notifications and no Sleeper write actions in this version.

## Required technical shape

Implement the blueprint's lean topology:

- Python 3.14 API and worker.
- FastAPI, Pydantic, SQLAlchemy, and Alembic.
- PostgreSQL in production, with a lightweight isolated database strategy for
  unit tests.
- PostgreSQL-backed job state and advisory locking; do not add Redis.
- S3-compatible object-storage adapter for immutable artifacts, plus a local
  filesystem implementation for development/tests.
- TypeScript + React + Vite installable PWA.
- One OCI image usable as API, worker, or migration command, plus a static web
  build.
- Local production-like startup using Docker Compose.
- Environment-driven configuration with `.env.example`; never commit secrets.
- Structured JSON logging and correlation IDs across API and jobs.

Add dependencies cleanly to `pyproject.toml` and the frontend package manifest.
Update the lockfiles. Keep commands cross-platform where practical and make the
documented PowerShell flow work on Windows.

## Workstream 1 — application foundation

Create the `src/app/` boundary described in the blueprint, including:

- Configuration and environment validation.
- FastAPI application factory.
- `/health/live` and `/health/ready` endpoints.
- Versioned `/api/v1` routers.
- SQLAlchemy models and Alembic migrations.
- Repository/service boundaries so decision code does not depend directly on
  HTTP or ORM objects.
- Job-run records, idempotency, advisory lock, retry metadata, and correlation
  IDs.
- S3-compatible and local artifact stores.
- A safe development seed command using fixtures only.

Implement the durable entities specified in the blueprint. Use normalized
columns for identifiers and query keys while retaining immutable raw JSON for
external payload provenance.

## Workstream 2 — authentication

Implement email magic-link authentication behind a provider interface:

- Production provider configuration through environment variables.
- Development provider that prints or returns a one-time link only when an
  explicit development flag is enabled.
- One-user email allowlist.
- Expiring, one-use tokens stored hashed.
- Secure HTTP-only same-site sessions.
- CSRF protection for mutations and strict configurable CORS.
- Authentication and assistant/research rate limits.

Do not expose OpenAI, email, database, or object-storage credentials to the
browser.

## Workstream 3 — Sleeper integration

Implement a rate-conscious, read-only Sleeper client and sync service for:

- User identity and leagues.
- League details, scoring settings, and roster slots.
- League users and rosters.
- Weekly matchups.
- Completed transactions.
- Traded future picks.
- Drafts and draft picks.
- NFL state.
- Player identity/status snapshots and trending adds.

Requirements:

- Persist every raw response as a content-addressed source snapshot before
  transformation.
- Record endpoint, request parameters, fetched time, hash, artifact URI, and a
  source-health verdict.
- Reuse valid daily player snapshots instead of repeatedly fetching the large
  player payload.
- Normalize Sleeper, GSIS, team, and player identities without deleting raw
  identifiers.
- Make retries idempotent.
- Import completed trades automatically.
- Do not pretend Sleeper exposes pending trade proposals; those are manually
  logged in the app.
- Add contract tests using checked-in, minimized, attributed fixtures rather
  than live-network tests.

## Workstream 4 — league scoring compiler

This is a load-bearing contract. Implement it fully before decision engines.

Store raw Sleeper scoring and a normalized, hashed `ScoringContract` supporting:

- Passing/rushing/receiving yards and touchdowns.
- Interceptions, receptions, fumbles lost, and two-point conversions where
  configured.
- Passing, rushing, and receiving first downs.
- Yardage and other supported threshold bonuses evaluated on each simulation
  draw, never against the mean.
- Kicker attempts/makes by configured distance when present.
- Team-defense scoring keys used by the connected league contracts.
- QB/RB/WR/TE/FLEX/SUPER_FLEX/K/DEF slot eligibility.

Create a registry from Sleeper scoring keys to canonical stats/rules. Any
unknown nonzero key must appear in `unsupported_keys` and block recommendation
publication for that league. Never silently ignore it.

Add fixtures representing standard, PPFD, yardage-bonus, Superflex, K, and DEF
leagues. Add exact boundary tests around every nonlinear bonus.

## Workstream 5 — weekly projection consolidation

Inspect `../fantasy-projections-2` and port the required team-first weekly
pipeline into `src/projection/weekly/` or the closest blueprint-consistent
location.

Requirements:

- Preserve attribution and provenance comments.
- Remove any runtime path dependency on the sibling repo.
- Add parity tests against frozen representative v2 outputs before modifying
  behavior.
- Retain leakage-safe lagging: the target week's outcomes must never enter its
  features.
- Produce immutable weekly `projection_run` manifests with model version,
  input hashes, `as_of`, season, week, and availability evidence IDs.
- Store queryable player summaries and partitioned draw artifacts.
- Keep the existing v1 component-stat/release behavior intact while the weekly
  path is introduced.

Extend the canonical weekly stat draw to include first downs and configured
missing scoring stats. If historical data cannot support a trained first-down
model immediately, implement a clearly labeled, backtestable conditional-rate
baseline with shrinkage. Do not claim it is calibrated until the rolling
evaluation passes.

## Workstream 6 — availability lifecycle and cited research

Replace permanent append-only status behavior in the app path with lifecycle
events:

- `active_from`, `active_until`, and `cleared_at`.
- Source snapshot and evidence references.
- Healthy-source payload validation before clearing an event.
- Rebuild live depth from the curated base plus currently active events.
- Never clear availability because of a failed, truncated, or implausibly small
  source payload.
- Support weekly play probability and ROS return windows; do not treat every IR
  designation as automatically season-ending.

Implement an injury-research service behind an interface:

- OpenAI Responses API web-search implementation when configured.
- Deterministic fixture implementation for tests.
- Structured evidence containing source URL/title, published/fetched times,
  claim, return-date range, status, and confidence.
- Reject uncited return-date claims.
- Preserve contradictory evidence and lower confidence.
- The model may extract evidence but may not directly edit fantasy points,
  depth shares, or release pointers.

## Workstream 7 — projections and simulation

Implement four explicit modes with immutable runs:

- `preseason`
- `weekly`
- `ros`
- `dynasty`

Add deliberately simple but honest special-team models:

- DST: opponent-adjusted EPA, pressure/sack rate, regressed turnover rate,
  implied points when available, QB/OL context, venue, and weather adapter.
- Kicker: expected drives/scoring opportunities, red-zone stall tendency,
  distance-regressed accuracy, venue, and weather adapter.

Use free/public inputs already present in the repos where possible. Do not add
proprietary DVOA data; calculate a documented DVOA-like opponent adjustment.

Implement stable seeded simulation and an affected-team path:

- Diff changed inputs.
- Compute the affected player/team/opponent/decision set.
- Reuse partitions only when their complete input hash matches.
- Seal a new manifest referencing verified reused and new partitions.
- Recompute cross-player ranks and affected league outcomes.
- Keep the current 10,000-draw full-publish contract until a test-backed gate
  authorizes a change.

## Workstream 8 — decision engines

Implement deterministic, testable services for:

### Lineups and matchups

- Score universal stat draws under each league contract.
- Optimize legal lineups, including Superflex, K, and DEF.
- Evaluate the user's current starters.
- Evaluate both current-opponent and optimized-opponent scenarios.
- Return win/tie/loss probability, expected points, quantiles, recommended
  swaps, and swap regret.
- Select recommended starts by win probability rather than mean points alone.

### Waivers and FAAB

- Determine rostered and available players from the league snapshot.
- Rank by incremental roster utility and probability of entering a legal
  starting lineup.
- Include ROS, bye weeks, injuries, playoff schedule, replacement level, and
  dynasty manager state.
- Treat trending adds as a market/urgency signal, not a projection target.
- Return an FAAB range and confidence, not false single-dollar precision.

### Manager state and future picks

- Infer contender, fringe, retooling, and rebuilding probabilities.
- Permit a user override while retaining the inferred result.
- For max-PF leagues, project upcoming rookie order from optimal/potential
  points.
- For reverse-standings leagues, project record and configured final-placement
  rules.
- For later picks, infer early/mid/late probabilities from roster trajectory
  and widen uncertainty with distance.

### Trades

Keep three independent outputs:

1. Objective roster utility for each side.
2. Fairness and uncertainty.
3. Acceptance probability.

Objective benefit must contribute 75–90% of acceptance. Bounded manager
tendencies may contribute 10–25% and may not make a materially harmful trade a
recommended offer. Use hierarchical shrinkage until a manager has enough
history.

Learn from imported completed trades and manually logged offered, accepted,
rejected, countered, and expired proposals. Support suggested counteroffers.
Freeze every evaluation against projection and roster snapshot IDs.

## Workstream 9 — release and scheduling

Preserve the existing immutable release-bundle and atomic-pointer philosophy.
Add active pointers for weekly/ROS/dynasty modes and never mutate a sealed
release.

Implement worker commands for these `America/Los_Angeles` schedules:

- Daily 5:00 PM except Sunday: normal refresh.
- Sunday 8:45 AM: full status sync and early-window refresh.
- Sunday 11:45 AM: targeted afternoon refresh.
- Sunday 4:00 PM: targeted SNF refresh.
- Monday 4:00 PM: MNF refresh.
- Tuesday 5:00 AM: preliminary weekly close, ROS, waivers, FAAB.
- Wednesday 5:00 PM: corrections and finalized next-week refresh.
- On demand: full release.

Do not hardcode Tuesday as complete if a delayed game remains. Verify schedule
completion first.

Every job must:

1. Acquire an advisory lock.
2. Create a `job_run`.
3. Persist source snapshots.
4. Build a candidate run.
5. Execute gates.
6. Promote atomically or retain the prior pointer.
7. Record duration, cost, changes, and errors.

## Workstream 10 — AI assistant

Use the OpenAI Responses API server-side behind a provider interface. The app
must work without an OpenAI key; only chat and autonomous research should be
disabled/degraded.

Expose typed tools for:

- League context.
- Matchup retrieval.
- Player projections and comparisons.
- Lineup recommendations.
- Waivers and FAAB.
- Trade evaluation and counteroffers.
- Injury evidence/research.
- Projection-change explanations.

Requirements:

- The assistant must call app tools for authoritative values.
- It must never invent projections when a tool fails.
- Persist tool calls, model, source IDs, token usage, estimated cost, and
  latency.
- Preserve and display web citations.
- Use a hashed user identifier instead of sending the login email externally.
- Route routine extraction/explanations to the cost-sensitive configured model
  and complex dynasty/injury synthesis to the balanced configured model.
- Enforce warning, soft, and hard monthly cost limits from configuration.

## Workstream 11 — mobile PWA

Create a polished, installable, responsive application that preserves the
existing draft assistant's useful concepts and visual identity.

Required screens:

- Home: league selector, matchup, urgent decisions, freshness.
- Lineup: recommendations and current/optimized opponent toggle.
- Waivers: adds/drops, FAAB range, evidence.
- Trade Lab: construct/log trade, both-side grades, tendencies, counters.
- Dynasty: roster trajectory and pick distributions.
- Draft: port current board, cards, tiers, roster construction, and draft state.
- Assistant: league-aware conversation with citations.
- Operations: source freshness, active releases, failed gates, job retry, and
  cost usage.

Requirements:

- Bottom navigation and touch-friendly mobile layout.
- Installable manifest and service worker.
- Cache only the last successful read-only recommendations for offline display.
- Never cache authentication tokens or mutation responses in service-worker
  storage.
- Prominent `as of`, stale-data, uncertainty, and source-citation UI.
- Accessible keyboard/focus states and adequate color contrast.
- API client generated or typed from shared schemas.
- Browser tests for critical mobile flows.

## Workstream 12 — APIs

Implement the `/api/v1` surface from the blueprint, including authentication,
league sync, rules, rosters, matchups, projections, rankings, lineups, waivers,
draft board, trades, manager tendencies, injury evidence, projection changes,
assistant responses, and job status.

All authenticated read responses include `data_as_of`, source/snapshot IDs as
applicable, and `projection_run_id`. All mutations use idempotency keys.
Projection publication is never a browser-public endpoint.

Generate an OpenAPI document and ensure the frontend contracts agree with it.

## Workstream 13 — leakage, validation, and evaluation

Enforce:

- `available_at <= projection_as_of` for all forecast inputs.
- Shift-before-roll weekly features.
- Frozen pregame model versions and stored pregame forecasts.
- Evaluation only against stored forecasts, never regenerated hindsight runs.
- No post-kickoff injury evidence in a scored pregame forecast.
- No future proposal outcomes in acceptance features.
- Rolling-origin dynasty/pick evaluation.
- Scoring parity fixtures and unsupported-key gates.

Add or extend scorecards for weekly point/rank accuracy, calibration, start/sit
regret, matchup Brier score, waiver value, FAAB calibration, trade acceptance,
availability, first downs, DST, and kicker baselines.

Promotion must fail closed on unhealthy sources, unsupported scoring keys,
schema errors, conservation failures, unexplained material changes, simulation
hash mismatch, invalid probabilities, or browser/API smoke-test failure.

## Workstream 14 — local operation, deployment, and CI

Deliver:

- `.env.example` with every setting documented.
- Dockerfile and Docker Compose for database, API, worker command, and web app.
- Migration, seed, sync, project, worker, and frontend commands.
- Readiness checks and graceful shutdown.
- Database backup/restore documentation.
- Object-retention policy: daily 30 days, weekly for season, named releases
  indefinitely.
- CI jobs for Python tests, frontend tests/build, migrations, API contract,
  browser smoke tests, and existing release invariants.
- A deployment-neutral operations runbook.

Do not require real external credentials for CI.

## Required testing strategy

Run the existing test suite before or early in the build to establish the
baseline. Preserve unrelated existing failures as documented baseline issues;
do not hide new failures.

Add tests at these layers:

- Unit tests for scoring, bonuses, lineup eligibility, manager-state, pick
  rules, trade utility, tendency bounds, lifecycle clearing, and hashing.
- Contract tests for Sleeper payload normalization and OpenAI provider output.
- Integration tests for migrations, repositories, idempotent jobs, pointer
  promotion/rollback, and authenticated APIs.
- Leakage/property tests for temporal cutoffs and draw reuse.
- Frontend component tests and end-to-end mobile smoke tests.
- Golden fixtures for all six league rule shapes without including private
  real-league data in Git.

Before finishing, run:

- Python formatting/lint/type checks that you add.
- Full Python tests.
- Frontend lint/type/test/build.
- Alembic upgrade from an empty database.
- Docker Compose configuration validation.
- Existing browser-surface and release-invariant tests.
- A local fixture-based vertical smoke flow:
  login → league sync → scoring compile → projection run → lineup → waiver →
  trade evaluation → assistant explanation.

## Definition of done

The build is complete only when:

1. The existing projection and draft release tests remain green or every
   pre-existing failure is explicitly distinguished from new behavior.
2. A fixture user can log in and see six representative leagues.
3. Every representative scoring rule compiles with no silent omissions.
4. PPFD and yardage bonuses score correctly at draw level.
5. Weekly lineup recommendations work with current and optimized opponents.
6. Waivers and FAAB are roster-aware.
7. Redraft and dynasty trade evaluations report both-side objective impact,
   fairness, and bounded acceptance.
8. Max-PF and reverse-standings pick logic are configurable and tested.
9. Completed trades sync and manual proposal outcomes update tendency inputs.
10. Availability events activate and clear safely with cited evidence.
11. Automatic jobs fail closed and atomic rollback retains the last good run.
12. The assistant uses typed tools, cites research, and degrades safely without
    an API key.
13. The mobile PWA builds, installs, and passes the critical browser flow.
14. A new developer can run the complete fixture application from the README.

## Documentation deliverables

Update or add:

- Root `README.md` quick start for the complete app.
- `docs/APP_IMPLEMENTATION_BLUEPRINT.md` status and any justified deviations.
- `docs/APP_OPERATIONS_RUNBOOK.md`.
- `docs/APP_DATA_CONTRACTS.md`.
- `docs/APP_SECURITY.md`.
- API/OpenAPI usage notes.
- A migration/provenance note for code ported from v2.
- A concise list of configuration that still requires the user's Sleeper
  username, league IDs, login email, infrastructure, and OpenAI credentials.

## Final response format

When the work is complete, report:

1. What was built, led by user-visible outcomes.
2. The final architecture and any deviations from the blueprint.
3. Exact commands to run locally.
4. Tests and smoke flows run, with results.
5. Remaining external configuration steps.
6. Known limitations or unvalidated model components stated plainly.
7. Files changed, grouped by backend, frontend, modeling, infrastructure, and
   documentation.

Do not end with a plan for future implementation. End with a verified,
runnable production MVP and an honest list of only those steps that genuinely
require external credentials, league-specific confirmation, or deployment
account access.

