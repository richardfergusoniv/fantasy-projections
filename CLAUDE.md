# fantasy-projections

NFL fantasy projections, simulation, and release promotion. Python API + jobs
(`src/`), React PWA (`web/`).

## Current direction

We are leaving the frozen League Value board behind. The work now is a
**weekly, game-level projection model** we can update as 2026 results come in.
ADP and season-long Vegas are **sanity checks only**, not targets.

**Vegas weekly props is the product benchmark and what the app serves today.**
The new weekly model stays in **shadow** until it is good enough to promote.

### The freeze is over — do not re-derive it from older docs

`STATE_OF_BUILD.md`, `PIPELINE_MAP`, `docs/research/*` and several PR bodies
describe a season freeze (RB/WR ensemble weights, availability α = 0, the
v3-means cutover, 10 000 draws) as locked through 2026 outcomes. **That framing
is historical.** Those documents were accurate when written and have not all
been updated.

Do not treat a change to those knobs as a defect on that basis alone. Judge it
on evidence and on the promotion discipline below.

## Review rules

Ordered by how much they cost when missed.

### 1. As-of / leakage discipline — the primary rule

A model refit during the season is unforgiving here. Train on a column that
encodes the week being predicted and every backtest looks excellent while live
performance quietly fails. This is the defect class to hunt first.

**Any same-week realized quantity used as a prediction feature is a finding.**

The gate is `src/projection/weekly/draws/feature_outcome_split.py`:
`SAME_WEEK_OUTCOME_DENYLIST` plus `is_allowed_prediction_column`. It is
deny-then-allow, so anything not enumerated passes by default — absence from
the list is not evidence a column is safe.

`team_attempts` and `team_air_yards` are same-week team aggregates built in
`panel.py:_add_team_shares` whose player-level equivalents (`attempts`,
`air_yards`) were already denied. They were missing from the denylist and so
reached the prediction frame. **PR #70 adds both.** Until #70 merges the hole is
open on `master`; once it lands this is closed — do not re-raise it.

Sanctioned pre-kickoff features: lagged rolls (`_l3`, `_l5`, `_roll3`),
`_prior` / prior-season means, and pregame schedule (spread, total, rest,
as-of depth snapshots).

Two rolling recipes exist and are **not interchangeable at week 1**:

| Builder | Grain | Week 1 behaviour |
|---|---|---|
| `src/projection/data/features_weekly.py` (v3) | `groupby(player_id)` | roll3 can pull in prior-season weeks |
| `src/projection/weekly/features/rolling.py` (v2) | `groupby(gsis_id, season)` | `_l3` is null; prior season is a separate column |

`scripts/audit_weekly_integrity_extension.py` (PR #70, **not yet merged**) runs
these checks on synthetic fixtures without a database, and adds live checks
when `projections.db` exists.

### 2. Shadow until promoted

The app serves Vegas weekly props. The weekly model is shadow-only. A PR that
moves the app onto the new model — flipping a default, swapping a pointer,
changing `APP_PROJECTION_SOURCE` — is a promotion and needs to be called out as
one, whatever else the PR is nominally about.

**Vegas props promotion is gate-based, not flag-based** (since PR #67).
`run_weekly_props` auto-promotes the `weekly_props` pointer when the scrape
clears its quality and coverage gates; provider failures are isolated, so the
surviving sources can still pass. The review question is therefore *"are the
gates right, and is the promoted line fresh?"* — not *"is a shadow flag set?"*

- `weekly_props_shadow_only()` reads **`WEEKLY_PROPS_FORCE_SHADOW`** and
  defaults to `False`. It is an advanced escape hatch, not the normal path.
- The historical `WEEKLY_PROPS_SHADOW_ONLY` (and its dashboard typo aliases) is
  now **deliberately ignored**, so a leftover secret cannot silently block
  promotion. Do not treat that variable as live config.
- Scheduled jobs read the GitHub secret `PRODUCTION_JOB_ENV`, **not** Vercel env
- `weekly_props` is a **dedicated pointer mode**, distinct from `weekly`
  (which carries weekly-v2 R&D). Conflating them is the bug PR #63 fixed.
- `publish(..., activate=False)` persists a candidate without swapping pointers

Because promotion is automatic, a weakened gate ships straight to the live
board. Gate changes deserve more scrutiny than the flag ever did.

### 3. Conservation in the weekly allocator

`src/projection/weekly_latent/` (PR #71, **not yet merged**) allocates season
volume across weeks. Matchup multipliers reshape the weekly path and must never
move season mass — weights are renormalized per team so they sum to 1.

`evaluate_conservation` enforces this and `scripts/run_weekly_schedule_m1.py`
exits 1 when it fails. Treat a change that weakens either as a finding.

This is the spine of the new weekly model, so the invariant is worth holding
structurally rather than only detecting after the fact.

### Milestones — judge weekly-model PRs against these, not as loose research

The weekly model has a locked roadmap
(`docs/research/WEEKLY_LATENT_MODEL_DESIGN_2026-09-15.md`). A PR landing inside
one milestone should be reviewed against that milestone's scope; work that
belongs to a later one is scope creep, not an improvement.

| | Scope | Explicitly not yet |
|---|---|---|
| **M1** | Deterministic weekly schedule allocation. Matchup multipliers reshape the week; renormalization conserves season totals. No new ML. | Training, same-week realized volume features, replacing Vegas, League Value promote, PWA wiring, ADP/season-Vegas blending |
| **M2** | Team-week latent that *may* move season totals, once M1 conservation is proven. Opponent priors must be lagged or preseason. | ADP or season Vegas as drivers |
| **M3** | Weekly availability and conversion rates. Compare against Vegas, do not replace it. | — |

Across all three: Vegas weekly props is the **benchmark**, not the target to
copy, and ADP / season-long Vegas are market-sanity guardrails at most.

### 4. Vegas props correctness is user-facing

It is the live board, not an experiment. Promotion paths need a freshness
bound and must record staleness in provenance — a days-old closing line served
as current is a user-visible defect.

### 5. ADP and season-long Vegas are checks, not objectives

Flag anything that optimizes *toward* ADP as a target metric, or that treats
agreement with a season-long market as evidence of weekly accuracy.

## Repo conventions

**Research PRs are a normal category.** A dated note under `docs/research/`,
a read-only dump script, and committed artifacts (often force-added past
`.gitignore` so the evidence is reviewable). Review these for whether the
claims match the artifacts, not as production code. Cited numbers should be
reproducible by running the script.

**CI is `test-windows`** (`.github/workflows/ci.yml`). It runs `uv run pytest`,
the web unit tests, and `npm run test:e2e` — Playwright against a real seeded
API via `scripts/e2e_api.py`, not mocks.

**Python is pinned to 3.14.** Note for sandboxes: pydantic 2.13.5 fails on
CPython 3.14.0rc2 (`typing._eval_type() got an unexpected keyword argument
'prefer_fwd_module'`), which breaks any import of `fastapi`. If the whole
`tests/app` suite errors during collection, that is the environment, not the
change under review.

**`web/`**: `npm run lint` is only `tsc -b --noEmit` — no formatter in CI, so
style drift is not caught automatically.

## Do not

- Skip, disable, or quarantine a test to get CI green
- Promote a shadow run, reseal a release, or flip a production default as a
  side effect of an unrelated change
- Treat `scripts/audit_weekly_features.py` exiting 0 as a live-data seal; it
  walks a static contract registry and does not open the database
