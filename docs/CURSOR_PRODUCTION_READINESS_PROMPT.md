# Cursor production-readiness audit and hardening prompt

Copy everything below the divider into Cursor Agent mode from the repository
root after the fixture MVP build is present in the working tree.

---

You are the senior engineer responsible for independently auditing, correcting,
and hardening the fantasy football application currently implemented in this
repository.

The repository contains a large uncommitted implementation produced in a prior
agent pass. It reports 543 passing Python tests, a 49/49 blueprint audit, a
passing fixture verification script, a statically validated Docker Compose
file, and a built/tested React PWA. Treat those statements as claims to verify,
not as proof. Your task is to turn the fixture-based MVP into an honestly
assessed, production-ready deployment candidate while preserving all useful
work.

Do not merely review the code or produce a report. Inspect the implementation,
run it, find real defects and gaps, fix everything that can be fixed locally,
add meaningful tests, and leave the repository coherent and runnable. Never
claim that a live integration, Docker runtime, email delivery, deployment, or
model has been validated unless you actually exercised it.

## Product context

This is a private, mobile-first fantasy football application for one owner. It
must support six Sleeper leagues:

- Two redraft leagues and four dynasty leagues.
- Two dynasty leagues are Superflex.
- One league uses points per first down.
- One league uses yardage bonuses.
- Five leagues use team defense and four use kickers.
- Two dynasty leagues assign rookie picks by max potential points and two by
  reverse standings.
- Redraft trades do not include future picks; dynasty trades can.

The product includes weekly and rest-of-season projections, lineup/start-sit
recommendations, matchup win probabilities, waivers and FAAB, redraft and
dynasty trade analysis, contender/rebuilder classification, future-pick
valuation, draft assistance outside a live draft, and an AI assistant grounded
in application data. Sleeper is read-only. Recommendations and projections are
published automatically; no action is submitted to Sleeper.

## Authoritative files

Before changing anything, read these files completely:

1. `docs/APP_IMPLEMENTATION_BLUEPRINT.md`
2. `docs/APP_DATA_CONTRACTS.md`
3. `docs/APP_OPERATIONS_RUNBOOK.md`
4. `docs/APP_SECURITY.md`
5. `docs/WEEKLY_V2_PORT_PROVENANCE.md`
6. `docs/PIPELINE_MAP.md`
7. `STATE_OF_BUILD.md`
8. `README.md`
9. `pyproject.toml`, `uv.lock`, and `.github/workflows/ci.yml`
10. `.env.example`, `docker-compose.yml`, `Dockerfile`, `Dockerfile.web`, and
    all files under `docker/`
11. The complete implementation under `src/app/`, `src/projection/weekly/`,
    `src/projection/special_teams/`, `web/`, `migrations/`, and `tests/app/`
12. `scripts/verify_mvp.py`, `scripts/audit_blueprint_mvp.py`,
    `scripts/validate_compose_config.py`, `scripts/vertical_smoke.py`, and
    `scripts/compose_smoke.ps1`

Use the blueprint as the product contract. Preserve existing frozen projection
decisions, release manifests, promotion gates, provenance, and tests unless a
change is demonstrably required. If the implementation and blueprint disagree,
correct the implementation or document a narrowly justified deviation.

## Working-tree rules

- Start by inspecting `git status`, the full diff, and all untracked files.
- Every pre-existing tracked or untracked change belongs to the user.
- Do not discard, revert, reset, stash, or overwrite unrelated work.
- Do not use `git reset --hard`, destructive checkout commands, or broad
  deletion commands.
- Do not mutate an existing sealed release bundle or historical artifact.
- Do not commit, push, open a pull request, deploy, send email, or write to a
  live Sleeper account.
- Do not make the deployed app depend on `../fantasy-projections-2`.
- Keep secrets and generated runtime artifacts out of Git.
- Make focused corrections with clear provenance. If a pre-existing change
  appears broken, repair it in place instead of erasing it.

## Audit discipline

Use this sequence:

1. Establish and record the baseline before editing.
2. Map every blueprint capability to its actual implementation entry point,
   persistence boundary, API route, UI surface, and test.
3. Identify false positives: tests that only check imports, routes, string
   presence, status codes, or self-authored fixture expectations.
4. Exercise important paths through public interfaces and realistic state
   transitions.
5. Fix defects in priority order: data loss/security, incorrect recommendations,
   scheduling/publishing, integration failures, operability, then UX.
6. Re-run the complete verification matrix after changes.
7. Produce an evidence-based readiness report distinguishing verified,
   statically checked, fixture-validated, and externally blocked items.

The existing blueprint audit and MVP verification scripts are useful inventory
checks, but they do not constitute production validation by themselves. Do not
optimize implementation merely to make those scripts pass.

## Phase 1 — establish a trustworthy baseline

Run the existing documented commands exactly as written where the environment
supports them:

```text
uv sync --frozen --all-extras --dev
uv run pytest -q
uv run python scripts/verify_mvp.py
uv run python scripts/audit_blueprint_mvp.py
uv run python scripts/validate_compose_config.py
cd web
npm ci
npm test
npm run build
npm run test:e2e
```

Record command, duration, result, skipped tests, warnings, and environmental
limitations. Investigate warnings and skips instead of hiding them. A missing
executable or credential is an external limitation; a missing package,
incorrect script, broken configuration, or undocumented prerequisite is an
implementation defect.

Review test collection and coverage by subsystem. Add behavior-oriented tests
where critical paths are represented only by import checks or audit assertions.
Do not weaken assertions, inflate counts with trivial parameterizations, mock
the code under test, or make tests tautological.

## Phase 2 — run the application as a system

Prove the non-Docker local path from a clean database:

- Apply all Alembic migrations to an empty database.
- Seed the single authorized fixture user and six representative leagues.
- Start the API and web app.
- Exercise authentication, league switching, all eight screens, assistant
  fallback behavior, and operations status.
- Run at least one complete fixture refresh and confirm that recommendations
  become visible through the API and PWA.
- Restart processes and verify persistence rather than relying on process
  memory.

If Docker is installed and the daemon is available, run the real stack rather
than only validating YAML:

```text
docker compose build
docker compose up -d
pwsh scripts/compose_smoke.ps1
docker compose ps
docker compose logs --no-color
```

Use a project-specific Compose name and volumes so existing user containers and
data are untouched. Verify health checks, API-to-database connectivity, web
proxying, worker behavior, migrations, artifact storage, restart behavior, and
clean shutdown. If Docker is unavailable, retain static validation and report
runtime validation as blocked; never relabel it as passing.

Fix startup races, incorrect hostnames, missing assets, container permissions,
health-check gaps, inconsistent environment defaults, and dev-only behavior
that leaks into production mode.

## Phase 3 — persistence, migrations, and artifact integrity

Audit all SQLAlchemy models, repositories, transactions, and Alembic revisions.
Verify:

- A clean upgrade to head works.
- Migrations do not silently depend on seeded data or SQLite behavior.
- Constraints enforce important uniqueness, ownership, enum, and foreign-key
  invariants.
- Timestamps are timezone-aware UTC in storage.
- JSON payloads retain source data without becoming the only queryable model.
- Re-running sync and jobs is idempotent.
- Concurrent or retried jobs cannot create duplicate releases, evidence, or
  transactions.
- Failed jobs roll back their database changes and do not advance the active
  artifact pointer.
- Artifact manifests include schema/version identity, inputs, timestamps,
  checksums, and provenance.
- Local and S3-compatible artifact implementations obey the same contract.
- A partially written artifact cannot be selected as active.
- Backup and restore instructions are executable and do not imply unimplemented
  automation.

Add migration smoke tests against PostgreSQL when the environment permits.
SQLite-only success is insufficient evidence for PostgreSQL behavior. Do not
drop or rewrite user databases during tests.

## Phase 4 — Sleeper integration and league-rule fidelity

Audit the fixture and live Sleeper clients against the documented, public,
read-only API contract. The application must:

- Resolve a configured username to a Sleeper user.
- Traverse current and previous league IDs safely across seasons.
- Import league settings, scoring, rosters, users, matchups, transactions,
  traded picks, drafts, players, and trending adds required by the product.
- Handle pagination or endpoint limits where applicable.
- Use bounded timeouts, retries with jitter, rate-limit awareness, and useful
  errors.
- Cache responsibly without serving stale data as current after failed refresh.
- Preserve raw source snapshots and observed-at timestamps.
- Avoid logging email addresses, tokens, full sensitive payloads, or unnecessary
  league data.
- Never write lineup, waiver, roster, draft, or trade actions back to Sleeper.

Create an opt-in `scripts/live_sleeper_smoke.py` if no equivalent exists. It
must be read-only, require explicit live configuration, print a privacy-safe
summary, and exit clearly as skipped when credentials/configuration are absent.
It must never become a network-dependent required CI test.

Use the six league fixture shapes as permanent contract tests. The scoring
compiler must retain the raw scoring payload, compile every supported nonzero
key, and fail closed with a precise error for an unsupported nonzero key. It
must not silently approximate ordinary offensive scoring rules.

Test hand-calculated examples for standard scoring, PPR variants, points per
first down, yardage milestones/bonuses, Superflex eligibility, team defense,
and kicker scoring. Verify negative scoring and overlapping rules. Calculate
bonuses and PPFD at simulation-draw level rather than by applying nonlinear
rules only to mean statistics.

## Phase 5 — recommendation and simulation validity

Trace actual data from projection artifacts through scoring and each decision
engine. Replace placeholder constants, fixture-only shortcuts, random outputs,
and fake confidence values in production paths.

Verify with deterministic seeded tests and independently hand-calculated small
examples:

- Legal lineup construction across flex, Superflex, K, and DEF slots.
- Locked or already-started players cannot be moved improperly.
- Current-opponent and optimized-opponent matchup modes produce distinct,
  correctly labeled win probabilities.
- Win probability uses joint draws or an explicitly justified approximation,
  accounts for players already played, and handles ties consistently.
- Recommendations optimize the stated objective, not merely raw projected
  points when win probability is requested.
- Waiver recommendations account for roster need, replacement value, starter
  probability, schedule, budget, bid opportunity cost, league-mate rosters,
  and player availability.
- FAAB bids are bounded and explainable.
- Redraft trades reject future picks; dynasty trades support them.
- Trade output reports objective impact for both managers, fairness, roster
  construction, contender/rebuilder context, uncertainty, and bounded
  acceptance likelihood.
- Acceptance tendency is subordinate to objective roster benefit and does not
  invent precision when proposal-history samples are small.
- Completed Sleeper trades and manually logged proposal outcomes update the
  correct manager profiles without leakage from future outcomes.
- Max-PF and reverse-standings rookie-pick rules are separate and configurable.
- Future pick value reflects class/year uncertainty, likely slot distributions,
  and contending/rebuilding state without circular self-justification.
- DST and kicker models use the documented simplified inputs and identify
  their higher uncertainty honestly.

Weekly v2 fixture parity proves port fidelity only. If trained weights are
missing, the application must label the bridge as fixture/fallback mode in the
operations API and UI. It must not present fixture-derived values as trained
production projections. Add an artifact-readiness gate that prevents automatic
production publication when required trained artifacts are unavailable or
incompatible.

Do not retrain or promote a projection model without the required historical
inputs, leakage-safe validation, and explicit existing promotion process.

## Phase 6 — availability, injury research, and citations

Verify the full availability lifecycle:

- Sleeper is the primary live status source and drives candidate changes.
- Status changes create bounded research work for affected players/teams.
- Evidence records source URL, publisher, title, published time, retrieved
  time, affected player, normalized claim, and confidence/reliability metadata.
- Recommendations distinguish sourced facts from model inference.
- Citations resolve to the actual supporting page and open safely in a new tab.
- Low-quality, duplicate, stale, contradictory, and post-kickoff evidence is
  handled explicitly.
- A later healthy status can clear a prior event without leaving a permanent
  append-only injury penalty.
- Player identity matching cannot apply evidence to a namesake.
- Evidence published after kickoff cannot alter the frozen pregame evaluation.
- Research failure retains the last good release and exposes stale/error state.

Live web research and external news access may require unavailable services.
Build and test the complete evidence contract with fixtures, but label real
source retrieval as unverified unless it is actually exercised. Never fabricate
URLs, publication times, quotes, injury designations, or source agreement.

## Phase 7 — scheduling, incremental recomputation, and publication

All schedules must use `America/Los_Angeles` for presentation and convert to
UTC for storage/execution. Verify daylight-saving transitions and prevent
double or missed execution. Implement and test the blueprint schedule:

- Daily 5:00 PM except Sunday: `daily-refresh`.
- Sunday 8:45 AM: `sunday-early`.
- Sunday 11:45 AM: `sunday-afternoon`.
- Sunday 4:00 PM: `sunday-night`.
- Tuesday 5:00 AM: `weekly-close-preliminary`.
- Wednesday 5:00 PM: `weekly-correction`.

Weekly close must postpone when a delayed Monday or Tuesday NFL game is not
final. A single Sunday refresh is not sufficient because later windows have
separate inactive reports.

Test scheduler uniqueness, leader/advisory locking, retry policy, crash
recovery, idempotency, stale-run detection, affected-team expansion, and manual
reruns. Incremental computation must include every downstream dependency; if
dependency certainty is unavailable, safely widen to a full refresh.

Publication must be candidate-first and atomic:

1. Fetch and persist inputs.
2. Compute an immutable candidate release.
3. Run schema, conservation, bounds, completeness, scoring, and partition
   validation.
4. Promote one pointer transactionally only after all gates pass.
5. Retain and expose the previous good release.
6. Support and test rollback without modifying either immutable bundle.

Prove failure injection at multiple stages does not expose partial or invalid
results.

## Phase 8 — authentication and security

Threat-model the actual implementation and repair concrete issues. At minimum:

- Production mode allows only the configured owner email.
- Dev authentication cannot be enabled accidentally in production.
- Magic links are cryptographically random, hashed at rest, short-lived,
  single-use, and resistant to replay and user enumeration.
- Session cookies use appropriate `HttpOnly`, `Secure`, and `SameSite` settings,
  with rotation/invalidation behavior documented.
- State-changing routes enforce CSRF protection and do not mutate through GET.
- CORS and trusted hosts are explicit and narrow in production.
- Redirects cannot escape allowed origins.
- API and assistant inputs have length, type, and rate limits.
- SQL, filesystem paths, artifact keys, Markdown, citations, and model output are
  handled without injection or traversal.
- The web app does not render unsanitized HTML or leak secrets into its bundle.
- OpenAI, email, database, and object-store credentials stay server-side.
- Logs redact secrets and minimize personal/league data.
- Errors are useful operationally without exposing tracebacks to clients.
- Dependency and container security checks are documented and run when tools
  are locally available.

Add focused security regression tests. Keep fixture/dev convenience explicit
and isolated from production defaults. Update `docs/APP_SECURITY.md` with the
actual threat boundaries and remaining risks, not generic assurances.

## Phase 9 — AI assistant grounding

The assistant is an explanation and orchestration layer, not an independent
source of projections. Verify that it:

- Uses typed, allowlisted, read-only application tools.
- Obtains league-specific projections, scoring, rosters, recommendations, and
  evidence from application services.
- Includes release identifiers and freshness in answers where relevant.
- Provides citations for current injury/news claims.
- Separates fact, projection, and recommendation.
- Cannot be prompt-injected by player names, league names, source content, or
  stored notes into exposing secrets or invoking unauthorized behavior.
- Does not permit arbitrary SQL, URLs, filesystem access, or tool parameters.
- Degrades into a useful deterministic UI/API experience when no OpenAI key is
  configured.
- Uses bounded timeouts, token/output limits, retry rules, and spend controls.
- Avoids sending the owner's email or unnecessary full league payloads to the
  model provider.

Add tool-contract tests and adversarial prompt fixtures. Do not make external
OpenAI calls in required CI. If no API key is available, report the live model
path as unverified rather than claiming it passed.

## Phase 10 — mobile PWA behavior and accessibility

Audit the PWA at narrow mobile widths and desktop widths. Verify:

- Login and session-expiry recovery.
- League selection persists and every view is league-specific.
- Home, lineup, waivers, trade lab, dynasty, draft, assistant, and operations
  screens use real API state rather than hardcoded sample output.
- Loading, empty, partial, stale, offline, and error states are distinguishable.
- Current-versus-optimized opponent mode is an accessible toggle with clear
  semantics.
- Recommendations show rationale, uncertainty, source freshness, and citations.
- Touch targets, contrast, focus order, keyboard operation, labels, reduced
  motion, safe-area padding, and screen-reader basics are sound.
- PWA manifest, icons, service worker scope, cache invalidation, update flow,
  installability, and deep-link refresh work.
- Authenticated API data is not dangerously cached for offline reuse.
- Source links opened in a new tab use safe rel attributes.

Expand Playwright coverage beyond a shell/render check. Include at least one
meaningful end-to-end owner journey using the real local API and seeded database:
login, select a league, inspect a lineup recommendation, change matchup mode,
inspect a waiver, evaluate a trade, view a citation, and check operations
freshness. Use responsive assertions for a representative phone viewport.

## Phase 11 — API contracts and observability

Audit all API routes for consistent schemas, status codes, pagination,
authorization, idempotency, error envelopes, and versioning. Ensure the OpenAPI
document accurately describes authenticated behavior and useful examples.

Operations visibility must expose, without leaking secrets:

- Current release IDs by horizon and league.
- Last successful and failed job timestamps.
- Data and evidence freshness.
- Fixture/fallback/trained/live mode labels.
- Scheduler and artifact-store health.
- Validation failures and last rollback.
- Degraded dependencies.

Use structured logs with request/job correlation IDs. Add basic metrics hooks or
documented health/readiness probes suitable for a single-user deployment. Avoid
high-cardinality identifiers and sensitive payloads.

## Phase 12 — deployment candidate and operations

Keep deployment provider-neutral unless the repository already contains an
approved target. Do not deploy without the user's credentials and explicit
authorization. Prepare everything needed to deploy safely:

- Production container images with pinned dependencies and non-root execution.
- Separate API/web/worker concerns with one authoritative scheduler.
- Explicit production environment validation that fails fast on unsafe values.
- A sanitized `.env.production.example` or equivalent configuration reference.
- TLS/reverse-proxy expectations and trusted-origin configuration.
- Persistent PostgreSQL and object-store backup/restore procedures.
- Migration strategy for deployment and rollback.
- Resource expectations for a low-cost single-user environment.
- Log retention, health checks, restart policy, and operational alerts without
  adding push notifications.
- Email-provider interface and configuration for magic links; fixture/dev email
  must remain available locally.
- A release checklist and first-deployment smoke test.

Update the runbook with exact commands that have been executed. Mark commands
that require a future provider account or secret as pending. Do not invent
provider-specific success.

## Required new audit artifact

Create `docs/PRODUCTION_READINESS_AUDIT.md` containing:

1. Date, environment, and commit/working-tree context.
2. Capability matrix with `verified`, `fixture-verified`, `static-only`,
   `blocked-external`, or `failed` status.
3. Baseline results before changes.
4. Defects found, severity, evidence, and disposition.
5. Tests and runtime exercises added.
6. Security findings and mitigations.
7. Model/data limitations.
8. Exact external configuration still needed.
9. Go/no-go decision for fixture use, live read-only beta, automatic publishing,
   and public-internet deployment as four separate decisions.
10. Residual risks and a short prioritized launch checklist.

Do not give a global “production ready” label if any critical production path
is only represented by fixture or structural tests.

## Required verification matrix

At the end, run every applicable check below and preserve honest output in the
audit document:

```text
uv sync --frozen --all-extras --dev
uv run pytest -q
uv run python scripts/verify_mvp.py
uv run python scripts/audit_blueprint_mvp.py
uv run python scripts/validate_compose_config.py
uv run python scripts/vertical_smoke.py
uv run alembic upgrade head
uv run alembic check
cd web
npm ci
npm test
npm run build
npm run test:e2e
```

Also run, when available and safely configured:

- The real Docker Compose build and smoke test.
- PostgreSQL-specific migration/integration tests.
- The opt-in read-only Sleeper smoke test.
- The opt-in OpenAI tool smoke test without logging model input or secrets.
- Lint, type checking, dependency audit, and container scan commands configured
  by the repository.

If a command fails, diagnose and fix the cause, then rerun it. Record both the
initial failure and final result when it exposed a meaningful defect. Do not
omit failing or unavailable checks from the report.

## Completion gates

The task is complete only when all locally achievable gates below are true:

1. Existing projection and release tests remain green.
2. Critical behavior has non-tautological tests at service and end-to-end
   boundaries.
3. A clean database migrates, seeds, runs, restarts, and retains state.
4. Fixture sync produces six correctly distinct league configurations.
5. Unsupported scoring rules fail closed; PPFD and bonuses are correct at draw
   level.
6. Lineup, matchup, waiver, trade, dynasty, and draft outputs are league-aware
   and traceable to projection releases.
7. Fixture or missing-model fallbacks are impossible to mistake for trained
   production output.
8. Availability evidence has resolvable citations and safe lifecycle behavior.
9. Scheduler jobs are timezone-correct, unique, retry-safe, and idempotent.
10. Candidate validation, atomic publication, failure retention, and rollback
    have executable tests.
11. Production auth defaults fail closed and security regressions are tested.
12. The assistant is constrained to typed read-only tools and degrades safely.
13. The mobile PWA completes a meaningful real-API browser journey.
14. Docker is runtime-verified if available, otherwise explicitly static-only.
15. CI runs the strongest deterministic checks that do not require secrets or
    external network access.
16. A new developer can reproduce the fixture application from the README.
17. The readiness audit separates actual evidence from external blockers.

## Final response

When finished, report:

1. The go/no-go result for fixture use, live Sleeper beta, automatic projection
   publishing, and internet deployment.
2. The most consequential defects discovered and how they were fixed.
3. Exact verification commands and results, including skips and blockers.
4. Current fixture, fallback, trained, or live status for every external/model
   subsystem.
5. Remaining steps that genuinely require the owner's Sleeper username and
   league IDs, email provider, OpenAI key, trained weekly artifacts, Docker
   runtime, hosting account, DNS, or other external authority.
6. Files changed, grouped by backend, frontend, modeling, infrastructure,
   tests, and documentation.

Lead with verified user-visible outcomes. Do not report test counts without
explaining what important behavior they cover. Do not claim production
readiness based on static validation, fixture parity, self-authored audit
scripts, or the absence of exceptions. Finish with an honest, runnable
deployment candidate and a short list of externally blocked launch steps.
