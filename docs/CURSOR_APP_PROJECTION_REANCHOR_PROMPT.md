# Cursor prompt: re-anchor the app on the sealed production projection pipeline

You are working in the `fantasy-projections` repository. Implement this task end to end, using the existing code and artifacts as the source of truth. Do not merely write a plan.

## Why this task exists

The repository had a useful, validated projection pipeline before the application build began. That pipeline produced the sealed 2026 release used by the draft assistant. Subsequent work introduced a separate experimental weekly-v2/event/joint-draw architecture. Its evaluation harness is now honest, but that experimental branch has not passed its publication or start/sit gates.

Do not treat the experimental weekly branch's NO-GO as a global failure of the original projection pipeline or of the whole application.

The required architectural correction is:

1. Make the existing sealed production release the application's default projection source.
2. Preserve and quarantine weekly-v2/event/joint-draw work as R&D.
3. Add a conservative, separately gated availability/depth-chart overlay for daily updates.
4. Replace the single global readiness verdict with capability-specific readiness.

Do not weaken any frozen model thresholds. Do not make failed experimental metrics pass through relabeling, draw variance, fixture substitution, or exceptions. Do not delete or rewrite the prior reports or experiment artifacts.

## Existing production source of truth

Start by inspecting and verifying these exact integration points rather than inventing a replacement:

- `draft_assistant/data/active_release_2026.json`
- the release bundle to which that active pointer resolves
- `src/app/projections/loader.py`
- `src/app/projections/service.py`
- the scoring compiler and league rule snapshots
- `STATE_OF_BUILD.md`

At the time this prompt was written, the active pointer identifies namespace `v2_baseline_20260830`, release ID `e92edd22-40d9-4219-87f6-47a651489d15`, and manifest SHA-256 beginning `5a8e145`. Do not hardcode those values; resolve the active pointer dynamically.

The bundle is expected to contain, at minimum:

- `fantasy_points_2026.csv`: selected production board
- `projections_2026.csv`: component/stat projections suitable for league rescoring
- `players_2026.json`: draft-assistant player payload
- v3 simulation summaries/draw artifacts, where present
- the sealed manifest and validation/provenance metadata

The intended model contract is:

- the accuracy-first v1/v2/ADP ensemble remains authoritative for point means;
- v3 simulation is a distributional overlay only and must not silently replace point means;
- known model caveats remain visible;
- weekly-v2/event/joint draws remain experimental until their own gates pass.

If repository evidence differs from any value in this prompt, document the difference and follow the signed/sealed repository evidence.

## Required implementation

### 1. Define explicit projection-source modes

Introduce a typed/configured projection-source contract with these semantics:

- `sealed_release` — default and production-capable source.
- `status_adjusted_release` — sealed release plus a validated immutable availability/depth overlay.
- `weekly_v2_rnd` — explicit opt-in experimental source; never selected by default.

Use clear environment/config names, for example:

```text
APP_PROJECTION_SOURCE=sealed_release
WEEKLY_RND_ENABLED=false
```

Keep backwards compatibility where reasonable, but fail closed on an unknown source. A missing weekly-v2 output directory must not prevent the app from starting with `sealed_release`.

Do not let a background job, API request, or service silently switch the active source.

### 2. Harden `ReleaseBundleLoader` as the production integration seam

Extend the existing loader rather than creating a parallel ad hoc loader. It must:

- resolve the active pointer without hardcoded namespace/release paths;
- validate pointer schema, requested season, release ID, namespace, manifest hash, manifest validation state, and every consumed artifact hash;
- reject path traversal, partial bundles, mismatched manifests, and tampered artifacts;
- load both the selected production board and the full component projections;
- load v3 distribution summaries/draw metadata only as an optional overlay;
- retain provenance: release ID, namespace, artifact hashes, model identity, generation/as-of time, draw profile/count, source commit when present, validation status, and caveats;
- cache only by a key that changes when the active pointer or manifest changes;
- invalidate/reload safely after an atomic pointer swap;
- expose enough source metadata for API responses, assistant citations, UI labels, operations status, and audit logs.

Never mutate a sealed bundle in place.

### 3. Re-score the sealed component projections for each Sleeper league

The application supports six leagues with different lineup and scoring rules. Use the persisted exact Sleeper `scoring_settings` and roster-position snapshots as the authority. Do not substitute the default half-PPR points in `players_2026.json` for every league.

Compile each league's rules and re-score `projections_2026.csv` component projections into league-specific point means. Reuse the existing scoring compiler and extend it only when justified by a live rule.

For nonlinear or insufficiently modeled rules—especially points per first down, yardage bonuses, unusual kicker rules, and defense scoring—return an explicit fidelity classification such as:

- `exact_component_rescore`
- `modeled_approximation`
- `unsupported_rule`

Do not call a result exact when the source bundle lacks the necessary component or draw information. For PPFD and yardage bonuses, use the best supported expected-value approximation and state it. Do not leak the half-PPR production-board total into these leagues as if it were their native score.

Kicker and DST may remain intentionally simplified, but their formulas and uncertainty labels must be explicit and testable.

Use the actual live LeagueRuleSnapshots from the Sleeper shadow data. Fixture-mapped league IDs are not acceptable evidence of exact league scoring.

Configured owner and leagues:

- Sleeper username: `<sleeper-username>`
- `<redraft-league-1>` — Redraft league 1
- `<redraft-league-2>` — Redraft league 2
- `<dynasty-league-1>` — Dynasty league 1, rookie order `reverse_standings`
- `<dynasty-league-2>` — Dynasty league 2, Superflex, rookie order `max_pf`
- `<dynasty-league-3>` — Dynasty league 3, rookie order `max_pf`
- `<dynasty-league-4>` — Dynasty league 4, Superflex, rookie order `reverse_standings`

Do not commit secrets or personal local configuration.

### 4. Implement a conservative daily status/depth overlay

Create a versioned, immutable overlay derived from:

- the sealed base release and its manifest hash;
- the latest successful read-only Sleeper player/status/depth sync;
- cited primary injury/inactive information already admitted by the application's research policy;
- explicit adjustment events and deterministic rules.

The overlay must record:

- base release ID and manifest hash;
- source observations and timestamps;
- affected player/team IDs;
- before/after availability and projection values;
- reason codes and citations;
- algorithm/config version;
- overlay hash and validation result.

This overlay may conservatively zero an OUT/inactive player, adjust questionable/doubtful availability probability, and redistribute vacated opportunity using documented deterministic rules. It must preserve team-level opportunity within declared tolerances and must apply availability exactly once.

It must not pretend to be a newly trained weekly matchup/form model. Label its output `status-adjusted season baseline` or equivalent.

Create a separate promotion gate and atomic pointer for status overlays. The gate must cover identity resolution, source freshness, reason/citation completeness, finite values, position/team conservation, scoring compatibility, idempotency, and rollback. It must not depend on weekly-v2's point-dispersion or event-model gates.

Default schedule in `America/Los_Angeles`:

- Monday through Saturday: 5:00 PM
- Sunday: approximately 8:45 AM, 11:45 AM, and 4:00 PM to capture the early, afternoon, and evening inactive windows

Make schedules configurable. Jobs must be lock-safe, idempotent, observable, and safe to retry. A failed update leaves the prior active overlay in place.

### 5. Separate app capabilities from experimental model readiness

Replace any single global red/green readiness result with a capability matrix. At minimum expose these independently:

| Capability | Intended source/status |
|---|---|
| Draft rankings and roster values | Sealed production release |
| Dynasty values and trade foundations | Sealed production release plus league/roster context |
| League-specific season projections | Sealed component rescore, with scoring-fidelity label |
| Waiver baseline | Production values plus roster availability; current-week breakout signal may be limited |
| Daily injury/depth adjustments | Status overlay after its independent gate passes |
| Start/sit season-baseline comparison | Allowed as clearly labeled advisory output |
| Matchup-specific weekly start/sit probability | Disabled/R&D until weekly decision gates pass |
| Weekly-v2/event/joint automatic publication | Disabled/R&D |
| Public deployment | Independent infrastructure/security decision |

The application must remain useful when the weekly experimental system is absent or NO-GO.

Update API response models and the UI so every projection/recommendation states:

- projection source;
- base release and optional overlay version;
- as-of time;
- scoring-fidelity classification;
- recommendation capability/mode;
- applicable caveats.

The Operations screen should show two distinct panels:

1. Production sealed release and status-overlay health.
2. Weekly modeling R&D readiness and failed gates.

An R&D NO-GO must not mark the entire production app unhealthy.

### 6. Correct feature behavior

- Draft, player values, roster analysis, dynasty, and trade analysis default to the sealed production source.
- Trade analysis must continue to grade fairness and impact for both managers, include contender/rebuilder context, and handle dynasty draft picks according to each league's rules.
- Waiver recommendations may use Sleeper trending data as urgency/interest context, but not as fabricated projection evidence.
- Start/sit may compare status-adjusted season/per-game baselines. Clearly distinguish this from a validated matchup-specific weekly projection or win probability.
- Hide or disable matchup win probability when its required weekly distribution/decision gates are not satisfied.
- The AI assistant must identify the source and fidelity of substantive projection claims and must not present weekly R&D output as production evidence.

Search for every call to weekly promotion/publication services. The default daily production path must not call `WeeklyProjectionService.promote_week` or advance a weekly-v2 pointer. Experimental services may remain callable only behind explicit opt-in and their existing gates.

### 7. Decouple deployment readiness

Do not deploy publicly as part of this task. Do make readiness reporting honest:

- private/local app readiness depends on sealed bundle integrity, league sync/scoring, authentication, persistence, and the specific enabled features;
- status-overlay auto-publication depends on its own gates;
- weekly-v2 auto-publication and matchup-specific start/sit depend on weekly model gates;
- public-internet deployment depends on PostgreSQL/Docker runtime verification, production email/auth, secrets, TLS/hosting, backups, monitoring, and security checks.

The weekly R&D failure alone must not block infrastructure work or a private beta using production-safe capabilities.

## Required tests

Add focused unit, integration, API, and UI tests. At minimum prove:

1. The app starts and core production features work when all weekly-v2 experiment/output directories are absent.
2. `sealed_release` is the default; unknown or unauthorized sources fail closed.
3. Pointer, manifest, and artifact tampering are rejected.
4. Loader cache invalidates after an atomic active-pointer change.
5. Production point means remain from the accuracy-first source while v3 remains a distribution-only overlay.
6. Component projections are re-scored through distinct live league contracts.
7. PPFD/bonus and other approximate rules cannot receive an `exact_component_rescore` label without required inputs.
8. No default half-PPR score silently leaks into a non-half-PPR league.
9. Status overlays are immutable, deterministic, idempotent, availability is applied once, and a failed gate cannot advance the pointer.
10. Overlay rollback restores the prior active version.
11. Weekly-v2/event/joint auto-publication remains disabled by default and failed experimental gates remain visible.
12. API and UI capability labels match actual behavior.
13. Matchup-specific win probability is unavailable when its gate is false, while labeled season-baseline comparisons remain usable.
14. The six configured league IDs and rookie-order rules are preserved without applying dynasty rules to redraft leagues.

Run the existing verification suite plus the relevant new tests. Do not weaken tests to obtain green results. If Node, Docker, PostgreSQL, live Sleeper snapshots, or another dependency is unavailable, distinguish a code failure from an external blocker and provide exact rerun commands.

## Documentation and final report

Create `docs/APP_PROJECTION_REANCHOR_REPORT.md` containing:

1. The resolved active production release and verified hashes.
2. A diagram or concise explanation of the new source/overlay/R&D boundaries.
3. The six live league scoring-fidelity results and unsupported/approximate rules.
4. Status-overlay gate results and rollback evidence.
5. Capability-by-capability GO/NO-GO decisions.
6. Exact commands and test results.
7. External blockers and the narrowest next step for each.
8. Files changed.

Use separate GO/NO-GO decisions for:

- sealed season projection source;
- exact/approximate league rescoring;
- daily status-overlay generation;
- automatic status-overlay publication;
- private core-app beta;
- season-baseline start/sit advisory;
- matchup-specific weekly start/sit/win probability;
- weekly-v2/event/joint automatic publication;
- public-internet deployment.

A valid result may contain a mix of GO and NO-GO decisions. Do not collapse them into one verdict.

## Guardrails

- Preserve the sealed release and all historical experiment artifacts unchanged.
- Preserve the honest event/joint NO-GO reports.
- Do not weaken the 1.52 calibration cap, point-dispersion threshold, event Brier gate, or any other frozen weekly policy.
- Do not use same-week realized outcomes, oracle prevalence, fixture scoring substitutions, or circular labels.
- Do not auto-publish experimental weekly projections.
- Keep Sleeper integration GET-only.
- Do not commit or push.
- Do not expose secrets, tokens, or private configuration in logs or reports.
- Prefer the smallest coherent changes that establish the source boundary; do not launch another model-research project inside this task.

At completion, report the capability matrix first, then implementation summary, verification evidence, remaining blockers, and the narrowest next production step.
