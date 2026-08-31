# Production readiness audit — fantasy decision app

**Date:** 2026-08-31
**Auditor:** independent engineering pass over the uncommitted implementation
**Working tree:** branch `master`, HEAD `c33c0a2`, large uncommitted change set
(the whole `src/app/`, `web/`, `migrations/` implementation is untracked). No
commits, pushes, deployments, emails, or live Sleeper calls were made.

**Environment:** Windows 11, Python 3.14.4, uv 0.11.2, Node 24.19.0 / npm 11.17.0
(installed but not on `PATH`), Playwright 1.62.1 with Chromium.
**Docker is not installed on this machine**, and no PostgreSQL server is
available. Everything that depends on either is marked `blocked-external` below
and is not claimed as verified.

---

## 1. Headline

The fixture application is real and now works end to end: a clean database
migrates, seeds six correctly distinct leagues, serves recommendations, survives
a refresh and a process restart, and a browser drives the whole owner journey
against the live local API.

The three claims in the prior status line did not survive checking:

| Claim | Reality on inspection |
|---|---|
| "543 passing Python tests" | 710 collected, **2 failing** at the working-tree state |
| "49/49 blueprint audit" | **48/49**; the audit script correctly reported the failure |
| "passing fixture verification script" | It passed — but it and the vertical smoke both ran their reads *before* the refresh, so they could not see that the refresh broke every league |

The most consequential defect was not visible from any of those checks: a data
refresh destroyed lineup recommendations in all six leagues.

---

## 2. Capability matrix

Status vocabulary: `verified` (exercised through a public interface on this
machine), `fixture-verified` (exercised, but against recorded fixtures rather
than live sources), `static-only` (config/structure checked, never executed),
`blocked-external` (needs a credential, service, or runtime not present here).

| Capability | Status | Evidence |
|---|---|---|
| Clean DB → migrate → seed → serve | `verified` | `alembic upgrade head` on an empty SQLite file, then `cli seed`, then `cli api`; six leagues returned by `GET /leagues` |
| Persistence across process restart | `verified` | API stopped and restarted on a new port; six leagues and all seeded state still present |
| Fixture Sleeper sync, six distinct leagues | `fixture-verified` | `tests/app/test_fixture_sync_contract.py`; six distinct `contract_hash` values; sync is idempotent on a second run |
| Scoring compiler, fail-closed on unmapped keys | `verified` | `tests/scoring/`, plus a hand-checked hash-determinism regression |
| PPFD, yardage bonuses at draw level | `fixture-verified` | `tests/scoring/test_scoring_examples.py` (hand-calculated), threshold/bracket rules evaluated per draw |
| Legal lineup construction (FLEX/SUPER_FLEX/K/DEF) | `verified` | All six leagues fill exactly the seats they declare (`test_fixture_sync_contract.py`, `test_matchup_modes.py`) |
| Locked starters/bench respected | `verified` | `test_matchup_modes.py` |
| Current vs optimized opponent modes distinct | `verified` | `test_matchup_modes.py` — previously untested and indistinguishable in the fixtures |
| Matchup probabilities partition to 1 | `verified` | `test_matchup_modes.py`, plus the release gate |
| Waivers: roster need, replacement, FAAB range | `fixture-verified` | 25 recommendations with rationale and bid ranges from the live local API |
| Trending adds reach the waiver engine | `verified` | `test_fixture_sync_contract.py` — the signal was silently always empty before |
| Redraft rejects future picks, dynasty accepts | `verified` | Live API: `picks_not_tradeable_in_redraft` vs a valued dynasty pick |
| Trade objective / fairness / bounded acceptance | `fixture-verified` | Objective weight 0.88 within the mandated 0.75–0.90; tendency shrunk at sample size 1 |
| Contender/rebuilder state from real data | `verified` | `test_dynasty_inference.py`; two rosters now differ, and the state is persisted so trades carry it |
| Rookie-pick rules separate and configurable | `verified` | `max_pf` and `reverse_standings` produce different orderings from different features |
| Availability lifecycle, activate/clear | `fixture-verified` | Existing suite; clearing refuses an implausibly small payload |
| Cited evidence contract | `fixture-verified` | Evidence carries `fixture://` URLs and `synthetic: true` — unmistakably not news |
| Live injury research | `blocked-external` | Requires an OpenAI key; the live provider raises rather than fabricating |
| Assistant degrades without a key | `verified` | Deterministic tool path returns a real lineup result; no key configured |
| Assistant refuses to invent a trade | `verified` | Typed `trade_not_specified` error replaces the previously fabricated trade |
| Assistant prompt-injection resistance | `verified` | Existing adversarial suite, strengthened for the player-id path |
| Live OpenAI model path | `blocked-external` | No API key; never exercised, never claimed |
| Scheduler timezone/uniqueness/idempotency | `verified` | Existing suite; slots keyed per occurrence, DST handled via `zoneinfo` |
| Candidate → gates → atomic promotion → rollback | `fixture-verified` | Existing publication/rollback suite plus the rollback endpoint |
| Auth fail-closed in production | `verified` | 46 security tests; production now also rejects `TRUSTED_HOSTS=*` and fixture Sleeper data |
| CSRF, cookie flags, magic-link hygiene | `verified` | Tokens are 256-bit random, SHA-256 at rest, single-use, expiring |
| Concurrent requests on a file database | `verified` | `test_concurrent_requests.py`; reproduces the 500 without the fix |
| Mobile PWA owner journey | `verified` | 6 Playwright tests against the real API and a seeded database at Pixel 7 |
| Citations open safely in a new tab | `verified` | `target="_blank"` + `rel="noopener noreferrer"`, asserted in the browser |
| Fixture/trained mode labels in the UI | `verified` | Operations screen now states the Sleeper source, artifact state, and publish block |
| Alembic schema drift check | `verified` | `alembic check` passes; it previously failed with 14 differences |
| PostgreSQL-specific behaviour | `blocked-external` | No server available; SQLite was brought to schema parity so the checks mean something, but this is not PostgreSQL evidence |
| Docker build / compose runtime | `blocked-external` | Docker is not installed; only static validation was run |
| Live read-only Sleeper integration | `blocked-external` | Needs the owner's username and league IDs; opt-in smoke script skips cleanly |
| Email delivery of magic links | `blocked-external` | Development provider only; no SMTP/Resend credentials |
| Trained weekly v2 artifacts | `blocked-external` | Bridge reports `fixture`; automatic publishing is blocked by the readiness gate |
| Deployment to a host | `blocked-external` | No provider account, no DNS, no authorization |

---

## 3. Baseline before any change

```
uv sync --frozen --all-extras --dev      → ok (76 packages)
uv run pytest -q                         → 2 failed, 708 passed          (53s)
uv run python scripts/verify_mvp.py      → passed                         (23s)
uv run python scripts/audit_blueprint_mvp.py → 48/49, exit 1              (32s)
uv run python scripts/validate_compose_config.py → passed                 (<1s)
uv run python scripts/vertical_smoke.py  → passed, but logged a swallowed
                                           assistant tool failure          (4s)
uv run alembic check                     → FAILED, 14 schema differences
cd web && npm ci                         → ok, 0 vulnerabilities
cd web && npm test                       → 1 failed / 1 test
cd web && npm run build                  → ok
cd web && npm run test:e2e               → 1 render check; the login flow
                                           was `test.skip`
```

Baseline failures:

- `tests/app/test_sleeper_client.py::test_retry_logs_do_not_leak_usernames_or_payloads`
- `tests/scoring/test_scoring_examples.py::test_contract_hash_is_deterministic_and_rule_sensitive`
- `web` `HomeScreen` unit test (`useAppState must be used within AppStateProvider`)

Both Python failures were order-dependent or latent defects, not flaky tests.

---

## 4. Defects found and disposition

Severity: **S1** wrong user-visible output or data/security loss; **S2** breaks
an operational guarantee; **S3** correctness of tooling, docs, or build.

### S1-1 — A data refresh broke every recommendation (fixed)

`POST /api/v1/sync` persisted Sleeper's own player ids onto roster snapshots,
while every projection, draw, and decision is keyed by the canonical GSIS id.
After a refresh, `GET /leagues/{id}/lineup/{week}` returned
`no_projected_players_on_roster` for all six leagues and waivers returned an
empty list. Reproduced directly: lineup 200 before sync, 400 after.

This is not only a fixture problem — against the real Sleeper API the same
mismatch applies, so the integration could never have worked.

**Fix:** identity resolution at the ingest boundary
(`SleeperSyncService.resolve_player_ids`), backed by a `player_identity`
registry now built from the `players/nfl` payload
(`upsert_player_identities`). Unresolvable ids are kept and reported as
`unresolved_player_ids` in the job metadata rather than silently dropped. Raw
payloads remain untouched in the content-addressed source snapshots.

### S1-2 — The two fixture universes disagreed (fixed)

`src/app/fixtures/seed/` described the six product leagues with real player
ids; `tests/fixtures/sleeper/` described a single toy league whose four roster
ids appeared nowhere else and were absent even from its own `players/nfl`
fixture. A fixture sync therefore could not produce six leagues, which
contradicts the product contract.

**Fix:** `scripts/generate_sleeper_fixtures.py` derives the Sleeper payloads
from the seed fixtures, so there is now one source of truth. The generated set
covers all six leagues with users, rosters, matchups, transactions, traded
picks, and drafts, using Sleeper-shaped ids that exercise both the gsis and
sleeper-id resolution paths.

### S1-3 — The assistant fabricated trades (fixed)

Asked anything matching `trade|offer|swap`, the deterministic gateway evaluated
a **hard-coded** trade between two fixed player ids and returned it as a
deterministic result. `get_injury_evidence` likewise defaulted to a fixed
player when the message named none.

**Fix:** both refuse with a typed error (`trade_not_specified`,
`player_not_specified`) and point at the Trade Lab. Nothing is invented.

### S1-4 — Dynasty inference was hard-coded constants (fixed)

`GET /leagues/{id}/dynasty/{roster_id}` passed `lineup_strength=0.65`,
`ros_win_prob=0.55`, `age_adjusted_value=0.5`, `pick_capital=0.4`,
`optimal_points=1100`, `potential_points=1200`, `projected_record=7.0` into the
inference. Every roster in every league returned "contender, 56%" and the same
projected pick. The state was also never persisted, so the trade engine's
contender/rebuilder context was always `null`.

**Fix:** features are computed from the league's own draws, rosters, and traded
picks — lineup strength and multi-year value as league-relative shares,
rest-of-season win probability from pairwise simulated matchups, pick capital
from `traded_pick` ownership discounted by year and round. Player birthdates are
not stored, so the feature is named `multi_year_value` and the missing
age adjustment is reported in `unavailable_features` rather than implied. State
is persisted, so trades now carry real context.

### S1-5 — Concurrent requests corrupted each other (fixed)

The engine gave every thread the *same* SQLite connection (`StaticPool`).
FastAPI runs sync endpoints in a thread pool and the PWA issues `/leagues` and
`/operations/status` together on first paint, so the two interleaved on one
cursor and one of them returned a 500 (`IndexError: tuple index out of range`
from SQLAlchemy's result processor). Caught by the browser suite, not by any
unit test.

**Fix:** `StaticPool` only for `:memory:` (where a shared connection is
required); file-backed SQLite gets normal pooling, a 30s busy timeout, and WAL.
`tests/app/test_concurrent_requests.py` reproduces the failure without the fix.

### S1-6 — Injury citations were unreachable on the Lineup screen (fixed)

Evidence was fetched only for players involved in a recommended swap. When the
optimizer agrees with the submitted lineup there are no swaps, so a
*questionable starter's* cited report — the thing most likely to change the
owner's mind — was never shown.

**Fix:** evidence and citations render on starter rows as well as swaps.

### S2-1 — Running migrations silenced all application logging (fixed)

`migrations/env.py` called `fileConfig(...)` with alembic's default
`disable_existing_loggers=True`, disabling every `logging.getLogger(__name__)`
created when `src.app.*` was imported. In-process migration (the `app-migrate`
command, tests) left the application mute. This is what made the Sleeper log
redaction test fail in a full run and pass alone.

**Fix:** `disable_existing_loggers=False`.

### S2-2 — `alembic check` failed; run-id columns were too narrow (fixed)

`b7c41d92f0aa` widened the run-id columns and added the missing
`projection_run` foreign keys **on PostgreSQL only**, deliberately, because
plain `ALTER TABLE` cannot express either on SQLite. The consequences were that
a documented verification step failed with 14 differences (so real drift could
not be distinguished from the known gap), and the SQLite test path never
enforced referential integrity.

Composed incremental run ids such as
`weekly-2026-w01-538cf955e04c-inc-7304e071a8` are 43 characters against a
`VARCHAR(36)` column — SQLite ignores the length, PostgreSQL rejects the insert.

**Fix:** revision `d4a1f6c28b57` performs the SQLite copy-and-rebuild via
`batch_alter_table`. `alembic check` now passes, and
`tests/app/test_migrations.py` asserts the column widths, the foreign key, the
season-long pointer uniqueness, and a downgrade/re-upgrade round trip.

### S2-3 — The trending-adds market signal was dead code (fixed)

`WaiverService._trending_adds` read `row.payload_json`, a column that does not
exist on `SourceSnapshot`; the stored envelope keys players as
`sleeper_player_id`/`add_count`, not `player_id`/`count`; and those are Sleeper
ids in a canonical-id context. Three independent faults, so the signal was
always `{}`. It was also unscoped, so with six leagues it would have served one
league's urgency to another.

**Fix:** read the artifact via the store, scope by league, resolve ids through
the identity registry. The existing test asserted trending ids were *disjoint*
from projected ids — which is exactly what made the feature unusable — and now
asserts the invariant that actually matters: syncing the signal writes no
projection rows.

### S2-4 — Fixture data was not distinguishable from live data (fixed)

`use_fixtures` was derived as `app_env != "production"` with no setting and no
label anywhere. A deployment misconfigured as `development` would serve
recorded payloads as if they were the owner's leagues.

**Fix:** an explicit `SLEEPER_USE_FIXTURES` setting; production refuses to start
with fixtures enabled; `operations/status` reports `modes.sleeper_source`; and
the Operations screen states it in words.

### S2-5 — The container images were not deployable as production (fixed, static-only)

The runtime image ran as **root**, installed `--all-extras --dev` (pytest and
the dev group), and copied the whole test suite into the image. The web image
used `npm install` rather than `npm ci` and had no healthcheck. The stack had
**no scheduler** — the `worker` service ran one job and exited — and published
PostgreSQL on all interfaces with a hard-coded password.

**Fix:** non-root `appuser`, `uv sync --frozen --no-dev`, no `tests/`, pinned
uv version, `npm ci`, a web healthcheck, a single-replica `scheduler` service
running `run-due` on a five-minute loop, loopback-only database port, an
overridable password, a named artifact volume, and an explicit compose project
name so this stack cannot collide with another.

**None of this is runtime-verified** — Docker is not installed here.

### S2-6 — Production allowed a wildcard trusted-host list (fixed)

`TRUSTED_HOSTS=*` passed production validation, contradicting §16 of the
blueprint. Now rejected at startup alongside the other unsafe settings.

### S3-1 — The README quick start could not work (fixed)

`.env.example` set `DATABASE_URL=…@db:5432/…`, a hostname that only resolves
inside Docker Compose, and the app has no SQLite path outside `APP_ENV=test`. A
new developer following the README got a connection failure.

**Fix:** the example now defaults to a local SQLite file (production still
refuses SQLite), with the PostgreSQL URLs alongside. Verified by running the
documented commands from a clean database.

### S3-2 — A blank optional env var crashed startup (fixed)

`SLEEPER_USE_FIXTURES=` (the normal way to leave an optional value unset in a
`.env`) raised a Pydantic parse error before the app could start. A blank value
is now treated as unset.

### S3-3 — `.env` was not ignored by git (fixed)

The README instructs `copy .env.example .env`, and the root `.gitignore` did not
ignore `.env` — so following the documentation stages a file that will hold
`APP_SECRET_KEY`, `OPENAI_API_KEY`, and mail credentials. `web/.gitignore` had
it; the root did not. Also ignored: `output/app_artifacts/` (84 generated
runtime artifacts that were sitting untracked) and the browser suite's
disposable `web/.e2e/`.

### S3-4 — The CI dependency audit could never run (fixed)

`uv run pip-audit` failed to spawn — `pip-audit` is not a project dependency —
and `continue-on-error: true` hid it. Replaced with `uvx pip-audit` against the
exported lock, plus `npm audit` for the web tree.

### S3-5 — Test pollution and false-positive checks (fixed)

- `test_migrations.py` set `os.environ["DATABASE_URL"]` with no cleanup,
  repointing every later test in the session.
- `scripts/vertical_smoke.py` ran its reads *before* the refresh and asserted
  only status codes; it printed `OK assistant` while the log showed the
  assistant's lineup tool raising `LeagueContextError`.
- `scripts/validate_compose_config.py` checked only that certain strings
  appeared in the files, so it passed while the image ran as root with dev
  dependencies and the stack had no scheduler.

All three now check the property rather than the appearance.

### Not fixed — recorded instead

- **Sunday windows are one handler.** `sunday-early`, `sunday-afternoon`, and
  `sunday-night` all map to `run_daily_refresh`. Each does a full status sync,
  research, and affected publish, which is behaviourally adequate, but the
  blueprint's "targeted afternoon/SNF research" is not differentiated. Deferred
  rather than invented: the targeting rule needs real kickoff times.
- **App tests share one in-memory database.** `tests/app/conftest.py` yields
  sessions against `sqlite+pysqlite:///:memory:?cache=shared` with no
  truncation between tests, so ordering can leak state (it caused one real
  false failure during this audit). New tests were written to be robust to it;
  reworking the fixture is a follow-up.
- **135 pre-existing ruff findings** under `src/app`, `tests/app`,
  `tests/scoring`. Added to CI as a non-blocking step so the count is visible;
  making it blocking is a separate cleanup.
- **Stray working files** left untouched because they are the user's:
  `baseline_pytest.txt`, `scripts/_tmp_exercise_decisions.py`.

---

## 5. Tests and runtime exercises added

Behavioural, not structural. Each was confirmed to fail against the defect it
covers before the fix landed.

| File | Tests | What it actually proves |
|---|---|---|
| `tests/app/test_fixture_sync_contract.py` | 6 | A full refresh through the public API leaves all six leagues recommendable, with six distinct contracts, canonical roster ids, a live trending signal, and idempotent re-runs |
| `tests/app/test_matchup_modes.py` | 6 | The two opponent modes are different questions with different answers; Superflex eligibility; locked starters/bench; probability partition; the win-probability objective |
| `tests/app/test_dynasty_inference.py` | 7 | Two rosters get different inferences; features are league-relative and bounded; an unsupported feature is declared; the two pick rules order differently; state persists into trades; pick capital follows traded-pick ownership |
| `tests/app/test_concurrent_requests.py` | 2 | Overlapping reads, and a read during a write job, all succeed on a file database |
| `tests/app/test_migrations.py` | 6 | Run-id columns fit a composed run id; the `projection_run` foreign key is enforced; one season-long pointer per mode; downgrade/re-upgrade round trip |
| `web/e2e/owner-journey.spec.ts` | 5 | The real owner journey in a real browser against the real API |
| `web/src/screens/Home.test.tsx` | 4 | Urgent decisions are derived from API responses, and "nothing urgent" is stated rather than left blank |

Runtime exercises performed by hand and recorded:

- Clean SQLite database → `alembic upgrade head` → `cli seed` → `cli api`, then
  sign-in, six leagues, refresh, lineup, waivers, trade, dynasty, operations.
- Process restart with the same database: all state present.
- Redraft vs dynasty future-pick rules through the live HTTP API.
- Dynasty state and rookie-pick ordering for two rosters in two leagues with
  different rules.
- Refresh job producing cited (synthetic) injury evidence, read back through
  `GET /players/{id}/injury-evidence`.

---

## 6. Security findings and mitigations

The existing security work is genuinely strong: 46 behavioural tests covering
production fail-closed startup, CSRF on mutations, cookie flags by environment,
magic-link hashing/expiry/single-use/non-enumeration, session revocation, rate
limits, trusted hosts, non-wildcard CORS methods and headers, error envelopes
without tracebacks, bounded assistant and trade inputs, artifact path traversal,
and log redaction of secrets and emails.

Findings from this pass:

| Finding | Severity | Mitigation |
|---|---|---|
| `.env` not git-ignored while the README tells you to create it | High | Added to `.gitignore` with `!.env.example` |
| `TRUSTED_HOSTS=*` accepted in production | Medium | Rejected at startup |
| Fixture Sleeper data indistinguishable from live | Medium | Explicit setting, production rejection, and a UI label |
| Runtime container ran as root with dev dependencies and the test suite | Medium | Non-root user, `--no-dev`, tests excluded |
| PostgreSQL published on all interfaces with a fixed password | Medium | Loopback binding, overridable password |
| Dependency audit never actually ran in CI | Medium | `uvx pip-audit` + `npm audit` |
| Assistant could return a fabricated trade as a deterministic result | Medium | Typed refusal |
| Application logging silently disabled after in-process migration | Low | `disable_existing_loggers=False` |
| 84 generated artifacts untracked and stage-able | Low | Ignored |

Dependency audit results on this machine:

- `uvx pip-audit --no-deps -r <exported lock>` → **no known vulnerabilities**
  across 209 locked runtime packages.
- `npm ci` in `web/` → **0 vulnerabilities**.
- Container image scanning: **blocked-external** (no Docker, no scanner).

Remaining risks are recorded in §10.

---

## 7. Model and data limitations

- **The weekly v2 bridge is in fixture mode.** `operations/status` reports
  `weekly_v2_state: fixture` and `auto_publish_allowed: false`, the readiness
  gate blocks automatic production publication, and the Operations screen says
  so in words. Fixture parity proves port fidelity only; it is not evidence of
  forecast quality.
- **Offensive players come from a fantasy-point-only release.** Draw sets are
  reported as `mixed` fidelity with an explicit note; league rules that cannot
  be re-applied to points-only players are listed in `unapplied_scoring_rules`.
- **Kicker and team-defense models are the documented simplified ones**, drawn
  at stat level so tiered points-allowed and field-goal-distance rules score
  exactly, with honestly wide uncertainty.
- **No player birthdates are stored**, so dynasty value is multi-year projected
  value, explicitly *not* age-adjusted. The gap is reported per response.
- **Manager tendency samples are tiny** in fixtures (n=1), and the acceptance
  model shrinks accordingly; objective benefit carried 88% of the signal in the
  evaluation exercised here.
- **No model was retrained or promoted** during this audit.

---

## 8. External configuration still required

| Needed | For |
|---|---|
| Owner's Sleeper username and the six league IDs | `SLEEPER_USERNAME`, `SLEEPER_USER_ID`, live read-only sync |
| Confirmation of each dynasty league's rookie-pick rule | `max_pf` vs `reverse_standings` per league; Sleeper does not expose it |
| SMTP or Resend credentials | Real magic-link delivery (`EMAIL_PROVIDER`, `EMAIL_FROM`) |
| OpenAI API key | Narrative assistant and live cited injury research |
| Trained weekly v2 artifacts | Lifting the automatic-publish block |
| Docker runtime | Building and running the container stack |
| A PostgreSQL instance | PostgreSQL-specific migration and integration evidence |
| Hosting account, domain, TLS termination | Deployment; `TRUSTED_HOSTS`, `APP_CORS_ORIGINS`, `APP_PUBLIC_URL` |
| A long random `APP_SECRET_KEY` and the real `APP_ALLOWED_EMAIL` | Production startup (validation refuses the defaults) |

---

## 9. Go / no-go

Four separate decisions.

**1. Fixture use (local, single developer) — GO.**
A clean database migrates, seeds, serves, restarts, and refreshes without
losing state; six leagues produce six distinct, legal, league-specific
recommendations; the browser journey passes end to end. Fixture and untrained
modes are labelled everywhere the owner can see them.

**2. Live read-only Sleeper beta — CONDITIONAL GO.**
The client is GET-only by construction, bounded, retried with jitter,
rate-limit aware, redacts usernames from logs, and refuses to serve stale data
as current. The identity resolution that live data actually depends on is now
in place and tested. Not yet exercised against the real API: run
`scripts/live_sleeper_smoke.py` with the owner's username first, and check
`unresolved_player_ids` in the first refresh — a non-empty list means real
players are unprojectable and must be resolved before trusting any output.

**3. Automatic projection publishing — NO-GO.**
Required trained weekly artifacts are absent. The application already refuses:
`auto_publish_allowed` is `false` and the readiness gate blocks automatic
promotion. This decision should stay NO-GO until trained artifacts exist and
have passed leakage-safe validation through the existing promotion process.

**4. Public-internet deployment — NO-GO.**
Not because of a known defect, but because the entire runtime path is
unverified here: no Docker build has ever been run, no PostgreSQL has ever been
migrated, no email has ever been delivered, and no TLS/reverse-proxy
configuration exists. Every one of those is `blocked-external`, and a container
stack that has only been read is not a deployment candidate that has been
tested.

**No global "production ready" label is claimed.** Several critical production
paths — PostgreSQL, Docker, email, live Sleeper, the trained model — are
represented only by fixtures or static checks.

---

## 10. Residual risks and launch checklist

Residual risks:

1. **PostgreSQL behaviour is inferred, not observed.** SQLite was brought to
   schema parity so the drift check and the FK tests mean something, but
   transaction semantics, advisory locks, and concurrency under PostgreSQL are
   untested here. The advisory lock path in `JobRunner` is a no-op on SQLite,
   so job mutual exclusion has *never* been exercised.
2. **The container stack has never been built or started.**
3. **Live Sleeper identity coverage is unknown.** Fixtures resolve 100%; real
   payloads will contain players the registry has never seen.
4. **Live injury research is unimplemented by design** — the provider raises
   rather than fabricating, so enabling `live` mode without the integration
   yields no evidence rather than bad evidence.
5. **App tests share one database**, so ordering can mask or manufacture
   failures.
6. **Sunday refresh windows are not differentiated.**

Prioritised launch checklist:

1. Provide `APP_SECRET_KEY`, `APP_ALLOWED_EMAIL`, `TRUSTED_HOSTS`, and a
   PostgreSQL `DATABASE_URL`; confirm the app refuses to start if any is unsafe.
2. Install Docker; run `docker compose build`, `docker compose up -d`,
   `pwsh scripts/compose_smoke.ps1`; verify healthchecks, migration ordering,
   the scheduler service, restart behaviour, and clean shutdown.
3. Migrate a real PostgreSQL database to head and re-run `alembic check`
   against it; run the app test suite pointed at PostgreSQL to exercise the
   advisory lock.
4. Run `scripts/live_sleeper_smoke.py` with the owner's username; then a first
   live refresh in a throwaway database, and review `unresolved_player_ids`.
5. Confirm each dynasty league's rookie-pick rule and set it through
   `PUT /leagues/{id}/draft-order-rule`.
6. Configure the email provider; verify a real magic link arrives and that the
   development link no longer appears in the response.
7. Add the OpenAI key with the spend limits already in configuration; verify the
   assistant still degrades correctly when the key is removed.
8. Only then consider trained weekly artifacts and lifting the publish block.
9. Terminate TLS in front of the API, set the production origins, and re-run the
   browser journey against the deployed host.
10. Rehearse backup and restore against the real database before relying on it.

---

## 11. Verification matrix as run

Commands executed on 2026-08-31 after all fixes, on this machine.

| Command | Result | Duration |
|---|---|---|
| `uv sync --frozen --all-extras --dev` | ok, 76 packages | <1s |
| `uv run pytest -q` | **738 passed**, 6 warnings, 7 subtests | 66s |
| `uv run python scripts/verify_mvp.py` | passed | 23s |
| `uv run python scripts/audit_blueprint_mvp.py` | **49/49** | 32s |
| `uv run python scripts/validate_compose_config.py` | passed (static only) | <1s |
| `uv run python scripts/vertical_smoke.py` | passed — 23 content checks, refresh first | 4s |
| `uv run alembic upgrade head` | ok, 3 revisions on an empty database | 3s |
| `uv run alembic check` | **No new upgrade operations detected** | 3s |
| `uv run python scripts/live_sleeper_smoke.py` | **skipped** (opt-in not set) | 3s |
| `cd web && npm ci` | ok, 0 vulnerabilities | 9s |
| `cd web && npm test` | 4 passed | 15s |
| `cd web && npm run build` | ok, PWA precache 11 entries | 16s |
| `cd web && npm run test:e2e` | **6 passed** against the real API | 13s |
| `uvx pip-audit --no-deps -r <exported lock>` | no known vulnerabilities | 40s |

Not run, and why:

| Command | Status |
|---|---|
| `docker compose build` / `up` / `ps` / `logs` | **blocked-external** — Docker is not installed |
| `pwsh scripts/compose_smoke.ps1` | **blocked-external** — no Docker, and `pwsh` is not installed (Windows PowerShell 5.1 only) |
| PostgreSQL migration/integration tests | **blocked-external** — no PostgreSQL server |
| Live read-only Sleeper smoke | **blocked-external** — needs the owner's username |
| OpenAI tool smoke | **blocked-external** — no API key |
| Container image scan | **blocked-external** — no Docker, no scanner |
| `uv run ruff check src/app tests/app tests/scoring` | ran: **135 pre-existing findings**, wired into CI as non-blocking |
