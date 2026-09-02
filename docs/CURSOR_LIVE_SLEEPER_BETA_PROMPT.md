# Cursor live Sleeper beta prompt

Copy everything below the divider into Cursor Agent mode from the repository
root after the production-readiness audit changes are present.

---

You are the senior engineer responsible for moving this fantasy football app
from fixture validation to a safe live, read-only Sleeper shadow beta.

This is an execution task, not a planning exercise. Inspect the current
uncommitted implementation and `docs/PRODUCTION_READINESS_AUDIT.md`, implement
the missing live-league configuration and validation, run a complete live sync,
fix defects discovered, and leave an evidence-backed shadow-beta result. Do not
deploy and do not enable automatic projection publishing.

## Confirmed owner and league configuration

Sleeper username:

```text
<sleeper-username>
```

The read-only smoke test has already resolved this user and found exactly six
2026 leagues. These are the only current leagues in scope:

| League | Sleeper league ID | Format | Rookie-pick order |
| --- | --- | --- | --- |
| Redraft league 1 | `<redraft-league-1>` | Redraft | Not applicable |
| Redraft league 2 | `<redraft-league-2>` | Redraft | Not applicable |
| Dynasty league 1 | `<dynasty-league-1>` | Dynasty | `reverse_standings` |
| Dynasty league 2 | `<dynasty-league-2>` | Dynasty, Superflex | `max_pf` |
| Dynasty league 3 | `<dynasty-league-3>` | Dynasty | `max_pf` |
| Dynasty league 4 | `<dynasty-league-4>` | Dynasty, Superflex | `reverse_standings` |

The rookie-pick rules above are owner-confirmed. Do not infer or replace them
from `playoff_seed_type`, standings settings, league names, or fixture mappings.
Sleeper does not provide an authoritative rookie-draft-order policy for this
purpose.

The existing opt-in smoke command has passed using GET requests only:

```text
$env:LIVE_SLEEPER_SMOKE='1'
$env:SLEEPER_USERNAME='<sleeper-username>'
$env:SLEEPER_SEASON='2026'
uv run python scripts/live_sleeper_smoke.py
```

Observed result: user resolved, six leagues found, sampled league rules and 12
rosters loaded, roster payload classified complete, and no writes attempted.
This proves connectivity only; it does not prove a complete application sync.

## Safety and working-tree requirements

- Read `git status` and the full diff before editing.
- Preserve all existing tracked and untracked user work.
- Never reset, revert, stash, broadly delete, or overwrite unrelated files.
- Do not commit, push, deploy, send email, call paid AI services, or modify any
  Sleeper state.
- Assert that the Sleeper client exposes GET-only behavior before live access.
- Never enable automatic publishing during this task.
- Never promote fixture or fallback weekly projections as trained output.
- Use a separate shadow/staging database and artifact prefix. Do not replace the
  fixture database, existing release artifacts, or any sealed release bundle.
- League IDs are identifiers rather than authentication secrets, but avoid
  unnecessarily printing roster contents, manager identities, transactions, or
  full API payloads in logs and reports.
- Keep local owner configuration out of Git. Commit only a sanitized example or
  schema unless the repository already has an approved private-config policy.

## Read before changing code

Read these files completely:

1. `docs/PRODUCTION_READINESS_AUDIT.md`
2. `docs/APP_IMPLEMENTATION_BLUEPRINT.md`
3. `docs/APP_DATA_CONTRACTS.md`
4. `docs/APP_OPERATIONS_RUNBOOK.md`
5. `docs/APP_SECURITY.md`
6. `docs/WEEKLY_V2_PORT_PROVENANCE.md`
7. `README.md`, `.env.example`, and `.gitignore`
8. `src/app/config.py`
9. All code under `src/app/league/sleeper/`
10. `src/app/persistence/models.py`, repositories, and migrations
11. `src/app/scoring/`
12. `src/app/availability/identity.py`
13. `src/app/decisions/dynasty.py`
14. `src/app/projections/`, `src/app/releases/`, and job handlers
15. `scripts/live_sleeper_smoke.py`, `scripts/vertical_smoke.py`, and current
    fixture generators
16. Relevant tests under `tests/app/` and `tests/scoring/`

Treat the readiness audit as the baseline. If its claims differ from current
code or runtime behavior, record and correct the discrepancy.

## Objective 1 — durable, safe owner league configuration

Implement an explicit league-selection and dynasty-rule configuration boundary.
It must support the six confirmed league IDs above without hardcoding this
owner's values into Python source, migrations, generic fixtures, or frontend
code.

Use the repository's established configuration style. A good solution may be a
validated local JSON/TOML/YAML configuration file named through an environment
variable, or equivalent typed environment configuration. Requirements:

- The configuration is parsed into a typed model.
- It contains the Sleeper username, season, explicit allowed league IDs,
  expected redraft/dynasty type, and the four confirmed rookie-pick rules.
- League names are display/verification metadata, never stable keys.
- Duplicate IDs, unknown rule names, rules on redraft leagues, missing dynasty
  rules, or a league appearing in multiple categories fail fast.
- Extra leagues returned by Sleeper are reported and ignored until explicitly
  allowed.
- Missing configured leagues fail the shadow sync clearly instead of silently
  producing a partial six-league application.
- The production app cannot accidentally fall back to fixture league IDs.
- Owner-specific configuration is gitignored.
- A sanitized example file and documentation describe the schema.
- Configuration logging includes IDs/names and rule type only; it excludes
  private roster/member/transaction payloads.

Persist each confirmed dynasty rule into `league_draft_rule` with a real
`confirmed_at` timestamp. Re-syncing must be idempotent. A changed owner-confirmed
rule must be an explicit, auditable update rather than a duplicate or silent
guess. Redraft leagues must never receive a rookie-pick rule.

Do not derive dynasty status solely from taxi slots: the confirmed C2C league
has no taxi slots. Use Sleeper's league type plus explicit owner configuration,
and fail on a material conflict.

Add focused tests for the exact six-league shape using synthetic IDs in public
fixtures where privacy is preferable. Include conflict, missing-rule,
extra-league, missing-league, idempotent re-sync, and rule-change cases.

## Objective 2 — complete live sync in isolated shadow mode

Create or improve an opt-in command such as:

```text
uv run python -m src.app.cli sleeper-shadow-sync \
  --season 2026 \
  --config <local-owner-config> \
  --report output/live_shadow/sleeper_sync_report.json
```

Use the project's normal CLI conventions; do not add a conflicting command if
one already exists. The command must:

1. Require explicit opt-in to live Sleeper data.
2. Assert read-only client behavior.
3. Refuse the normal production database unless an additional explicit safe
   acknowledgement is supplied.
4. Display the resolved database target and artifact prefix before work without
   exposing credentials.
5. Resolve the configured Sleeper username and verify exactly the six configured leagues.
6. Fetch, snapshot, normalize, and persist required live data for all six.
7. Traverse bounded prior-season chains without importing unrelated leagues.
8. Compile and validate each distinct scoring and lineup contract.
9. Populate player identities from Sleeper's player dataset before ingesting
   roster, matchup, transaction, or draft player IDs.
10. Persist rosters, members, current matchups where available, completed
    transactions, traded picks, drafts, trending adds, and league history needed
    by the product.
11. Generate recommendations in shadow mode without changing the active
    production release pointer.
12. Produce a privacy-safe machine-readable and Markdown report.
13. Exit nonzero on incomplete configured leagues, unsupported nonzero scoring,
    unresolved starter identities, invalid rules, or failed validation gates.
14. Retain diagnostic staging state on failure without promoting it.

Network calls must use bounded timeouts, retries with jitter, and clear endpoint
errors. Cache or source snapshots must record observed times and content hashes.
Never print raw API payloads merely to aid debugging.

## Objective 3 — identity reconciliation gate

The audit found and fixed a critical Sleeper-ID versus GSIS-ID defect. Prove the
fix against live data rather than assuming fixture parity.

The report must include by league and in aggregate:

- Total distinct rostered Sleeper player IDs.
- Resolved canonical IDs.
- Unresolved IDs.
- Ambiguous IDs.
- Resolved starters versus unresolved starters.
- Missing weekly and season-long projections by canonical ID.
- Players legitimately outside the projection universe, categorized where
  possible: retired, free agent, IDP, duplicate/legacy, team defense, kicker,
  or unknown.

Never guess an identity by display name alone. Use stable Sleeper/GSIS mappings.
Create a reviewable unresolved-ID artifact with safe metadata sufficient to fix
crosswalks. Do not allow an unresolved starter or a material unresolved skill
player to disappear from lineup, waiver, trade, or matchup calculations.

Define and test severity thresholds. The shadow run must fail its recommendation
gate for unresolved starters. Non-actionable historical/retired entries may be
warnings if they cannot affect a current decision.

## Objective 4 — live scoring and roster-contract gate

For each of the six leagues, report and validate:

- Sleeper league ID, display name, season, status, and format.
- Team count and roster slot counts.
- Starting slot structure including flex/Superflex/K/DEF.
- All nonzero Sleeper scoring keys.
- Normalized scoring rules and scoring-contract hash.
- Unsupported or approximated keys.
- PPFD rules, yardage bonuses, kicker rules, and defense rules where present.
- Waiver type and budget where needed for FAAB recommendations.
- Playoff settings needed for ROS and matchup simulations.
- Dynasty traded-pick horizon and completeness.
- Current NFL week and whether matchup data is expected to exist.

Unknown nonzero scoring keys must fail closed. Do not modify the compiler merely
to accept a key without implementing and hand-testing its semantics. Add a
regression fixture derived from the rule shapes, not private roster contents,
for every newly encountered supported rule.

Confirm that the two redraft leagues reject future-pick trades and that the
four dynasty leagues use exactly the confirmed rules:

```text
<dynasty-league-3> -> max_pf
<dynasty-league-2> -> max_pf
<dynasty-league-1> -> reverse_standings
<dynasty-league-4> -> reverse_standings
```

## Objective 5 — live shadow recommendations

After successful sync and validation, exercise the public service/API paths for
each league. Generate but do not publish/promote:

- Legal lineup recommendations.
- Current-opponent and optimized-opponent matchup win probabilities.
- Waiver candidates limited to actually available players.
- FAAB guidance where the league uses FAAB.
- At least one structurally valid trade-analysis request per league, with no
  fabricated proposal.
- Dynasty state and rookie-pick projection for each dynasty league.
- Draft-assistant inputs for the upcoming season without managing a live draft.
- Operations/readiness output identifying live Sleeper plus fixture/fallback
  projection status.

The command/report must identify the user's roster in each league using the
resolved Sleeper user ID. Fail rather than silently selecting roster 1 or a
fixture owner.

Because trained weekly v2 artifacts are currently absent:

- `auto_publish_allowed` must remain false.
- Automatic jobs must refuse promotion.
- UI and API must clearly label weekly output as fixture/fallback or unavailable.
- Shadow recommendations may be exercised for integration testing, but the
  report must not call them production-quality football advice.
- Do not manufacture trained weights or relabel fixture artifacts.

If existing season-long production projections are available and compatible,
report their actual release identity and coverage. Do not assume that weekly
fixture parity validates ROS or dynasty quality.

## Objective 6 — source-state and refresh safety

Verify the application can distinguish:

- Live Sleeper source state.
- Fixture injury research versus live research.
- Fixture/fallback weekly projections versus trained projections.
- Local versus S3 artifact storage.
- Shadow candidate versus active published release.

Run the live sync twice and prove idempotency. Then inject a controlled failure
after source persistence but before candidate validation and prove that:

- No partial release becomes active.
- The previous active pointer is unchanged.
- Failure and freshness state appear in operations output.
- A later successful retry does not duplicate league rules, rosters, draft
  rules, transactions, picks, or source snapshots.

Do not run live injury/news research or call OpenAI in this task unless explicit
credentials and opt-in already exist. Sleeper player status is allowed as part
of the read-only sync; external evidence retrieval remains separately labeled.

## Objective 7 — tests and reproducible commands

Keep live network tests opt-in and out of required CI. Add deterministic tests
for all new configuration, selection, rule persistence, identity gates,
reporting, and shadow publication behavior.

Run the existing full deterministic suite after changes:

```text
uv run pytest -q
uv run python scripts/verify_mvp.py
uv run python scripts/audit_blueprint_mvp.py
uv run python scripts/validate_compose_config.py
uv run python scripts/vertical_smoke.py
uv run alembic upgrade head
uv run alembic check
cd web
npm test
npm run build
npm run test:e2e
```

Then run the opt-in live smoke and isolated full shadow sync using
the configured Sleeper owner. If PostgreSQL or Docker remains unavailable, SQLite may support
the shadow product check, but keep PostgreSQL and Docker labeled unverified.

Do not hide warnings, skips, unresolved identities, unsupported scoring keys,
or partial endpoint failures. Fix locally correctable defects, rerun, and retain
the initial failure in the report when it revealed a meaningful issue.

## Required report

Create `docs/LIVE_SLEEPER_BETA_REPORT.md`. Keep private roster and transaction
details out of it. Include:

1. Date, environment, database/artifact isolation, and commands executed.
2. GET-only safety assertion and Sleeper connectivity result.
3. Six-league discovery table and confirmation that no extra league was
   imported.
4. League type and owner-confirmed rookie-pick-rule validation.
5. Scoring/roster contract summary per league.
6. Identity reconciliation counts and unresolved-ID disposition.
7. Historical league, traded-pick, transaction, and matchup completeness.
8. Shadow recommendation smoke result per league.
9. Idempotent second-run and injected-failure results.
10. Projection artifact mode and why automatic publishing remains blocked.
11. Defects found and fixes made.
12. A go/no-go decision for continued live read-only shadow use.
13. Exact blockers for trained projection publication and internet deployment.

Do not include owner email, roster contents, manager identities, auth secrets,
API keys, database credentials, or raw source payloads.

## Completion criteria

This task is complete only when:

1. Owner configuration selects exactly the six confirmed live leagues.
2. All four dynasty rules are persisted exactly as confirmed and redraft rules
   remain absent.
3. A complete isolated live sync runs using GET requests only.
4. Every live scoring contract either compiles correctly or blocks with a
   precise unsupported-rule diagnosis.
5. No current starter remains silently unresolved.
6. All six user rosters are identified correctly.
7. Each league reaches its decision APIs without fixture league IDs or
   hard-coded owner/roster assumptions.
8. The second sync is idempotent.
9. Failure injection cannot advance an active release.
10. Automatic publication remains blocked while trained weekly artifacts are
    absent.
11. Deterministic CI tests remain network-independent and pass.
12. The live beta report distinguishes verified results from remaining
    PostgreSQL, Docker, model, research, email, and deployment blockers.

## Final response

When finished, report:

1. Whether all six leagues synced successfully and whether continued live
   read-only shadow use is a go or no-go.
2. Exact scoring or identity issues encountered and how they were resolved.
3. The persisted rookie-pick-rule mapping.
4. Shadow recommendation coverage by league.
5. Test commands and results, including all skips and external blockers.
6. Why automatic publication remains disabled.
7. Files changed, grouped by configuration, sync/backend, tests, scripts, and
   documentation.

Do not claim that live Sleeper connectivity validates trained projections,
PostgreSQL, Docker, email, OpenAI, or deployment. End with a safe live shadow
beta and a short, evidence-based list of the next blockers.
