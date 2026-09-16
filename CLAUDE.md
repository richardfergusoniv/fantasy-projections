# fantasy-projections

NFL fantasy projections, simulation, and release promotion. Python API + jobs
(`src/`), React PWA (`web/`).

## Current direction

We are leaving the frozen League Value board behind. The work now is a
**weekly, game-level projection model** we can update as 2026 results come in.
ADP and season-long Vegas are **sanity checks only**, not targets.

**Vegas weekly props is the product benchmark and what the app serves today.**
The new weekly model stays in **shadow** until it is good enough to promote.

The modeling north star is
`docs/research/ACADEMIC_REVIEW_WEEKLY_PROJECTION_STACK_2026-09-15.md`. Several
rules below are distilled from it; read it before reviewing substantive
modeling work.

### Related — promotion and props roles

Promotion discipline and the three Vegas weekly-props roles are locked in
[PR #76](https://github.com/richardfergusoniv/fantasy-projections/pull/76)
(lands at `docs/decisions/WEEKLY_MODEL_PROMOTION_AND_PROPS_ROLES_2026-09-15.md`;
the file is not on `master` yet). That note owns the formal promotion gate,
the three-role table, the ADP allowed/not-allowed list, and the M1–M3 ladder
aligned to [PR #71](https://github.com/richardfergusoniv/fantasy-projections/pull/71).
This file owns what to flag on a PR.

### The freeze is over — do not re-derive it from older docs

`STATE_OF_BUILD.md`, `PIPELINE_MAP`, `docs/research/*` and several PR bodies
describe a season freeze (RB/WR ensemble weights, availability α = 0, the
v3-means cutover, 10 000 draws) as locked through 2026 outcomes. **That framing
is historical.** Those documents were accurate when written and have not all
been updated.

Do not treat a change to those knobs as a defect on that basis alone. Judge it
on evidence and on the promotion discipline in the decision note.

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
reached the prediction frame. **#70 closes `team_attempts` / `team_air_yards`.**
Until it merges, treat the hole as open on `master`; once it lands this is
closed — do not re-raise it.

Sanctioned pre-kickoff features: lagged rolls (`_l3`, `_l5`, `_roll3`),
`_prior` / prior-season means, and pregame schedule (spread, total, rest,
as-of depth snapshots).

**Every feature needs a cutoff, not just a lag.** The target state is one row
per player-game keyed `(season, week, game_id, player_id, position)`, with each
feature carrying an `available_at` / snapshot timestamp, and a training row
containing only what was observable at that cutoff. Forecast *vintage* matters
(Tuesday vs Friday vs 90 minutes pre-kickoff) — a feature that is legitimate at
one vintage can be leakage at an earlier one. A PR adding features without a
cutoff concept is incurring the debt even when nothing leaks today.

**Snapshot-dated market data.** Using final or later-season ADP / season-market
values in earlier historical rows is leakage, and it is easy to do accidentally
because those sources are usually distributed as a single current value.

Two rolling recipes exist and are **not interchangeable at week 1**:

| Builder | Grain | Week 1 behaviour |
|---|---|---|
| `src/projection/data/features_weekly.py` (v3) | `groupby(player_id)` | roll3 can pull in prior-season weeks |
| `src/projection/weekly/features/rolling.py` (v2) | `groupby(gsis_id, season)` | `_l3` is null; prior season is a separate column |

`scripts/audit_weekly_integrity_extension.py` (PR #70, **not yet merged**) runs
these checks on synthetic fixtures without a database, and adds live checks
when `projections.db` exists.

### 2. Model shape — generate box scores, do not regress fantasy points

The intended object is a **multistage generative model**: availability → team
volume → player role shares → conversions → yardage → touchdowns, with fantasy
points obtained by Monte Carlo over the component distributions. A PR that
regresses fantasy points directly on season-level inputs is off-plan, however
good its metrics look.

Three defect classes follow, all reviewable from a diff:

- **Internal consistency.** Independently projected components can violate
  hard constraints — receptions cannot exceed targets, completions cannot
  exceed attempts. If components are drawn separately, something must enforce
  or preserve the ordering.
- **Distributional adequacy.** Fantasy outcomes are skewed, zero-inflated for
  marginal players, and touchdown-driven. A normal interval with a fixed
  coefficient of variation is not a credible generative distribution. Counts
  need an overdispersed family, not Poisson or Gaussian — the review's layer
  table names a fit for each component.
- **Pooling.** Partial pooling across player / team / opponent / play-caller
  matters most exactly where the board is weakest: early season, rookies,
  backups, traded players, new coaching staffs. A short trailing window discards
  priors that shrinkage would keep.

Prefer regularized GLM/GAM or gradient boosting with well-encoded lagged
features over deep architectures. Sample size is modest and the regime shifts
each season; complexity is not the bottleneck.

### 3. Shadow until promoted

The app serves Vegas weekly props. The weekly model is shadow-only. A PR that
moves the app onto the new model — flipping a default, swapping a pointer,
changing `APP_PROJECTION_SOURCE` — is a promotion and needs to be called out as
one, whatever else the PR is nominally about.

**Vegas props auto-promote when scrape gates pass** (PR #67): `run_weekly_props`
swaps the `weekly_props` pointer after quality and coverage gates; provider
failures are isolated so surviving sources can still pass. Review the gates and
freshness, not a leftover shadow flag.

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

### 4. Conservation in the weekly allocator

`src/projection/weekly_latent/` (PR #71, **not yet merged**) allocates season
volume across weeks. Matchup multipliers reshape the weekly path and must never
move season mass — weights are renormalized per team so they sum to 1.

`evaluate_conservation` enforces this and `scripts/run_weekly_schedule_m1.py`
exits 1 when it fails. Treat a change that weakens either as a finding.

This is the spine of the new weekly model, so the invariant is worth holding
structurally rather than only detecting after the fact.

### Milestones

Judge weekly-model PRs against the locked M1–M3 milestone they land in, not as
loose research. Work that belongs to a later milestone is scope creep. The
ladder itself — and the formal bar for leaving shadow — is in
[PR #76](https://github.com/richardfergusoniv/fantasy-projections/pull/76),
aligned to [PR #71](https://github.com/richardfergusoniv/fantasy-projections/pull/71).

M1 being deterministic is deliberate scaffolding. Do not fault it for lacking
the learned components rule 2 describes — those arrive in M2/M3.

### 5. Validation — findings, not a second gate copy

This is what replaces the freeze. The freeze said "do not touch." The gate
says "change what you like, clear this bar before it reaches anyone." Sample
sizes, baseline set, and pinball/CRPS/coverage bars live in
[PR #76](https://github.com/richardfergusoniv/fantasy-projections/pull/76).

**Backtests must be rolling-origin, never random splits.** Train through week
`t-1`, predict week `t`, rebuilding every feature as it would have appeared at
that cutoff. A random train/test split on player-weeks is a finding on its own,
regardless of the numbers it produces.

Beating a naive baseline is the floor, not the achievement. A PR that claims
improvement from fantasy-point RMSE alone has not shown its intervals are
still honest. One or two weeks of 2026 results is an update, never a promotion
argument.

This gate is about promoting the **weekly model**. The Vegas props board has
its own scrape gates (rule 3) and is a separate mechanism.

### 6. Vegas props correctness is user-facing

It is the live board, not an experiment. Promotion paths need a freshness
bound and must record staleness in provenance — a days-old closing line served
as current is a user-visible defect.

### 7. ADP and season-long Vegas are checks, not objectives

Flag anything that optimizes *toward* ADP as a target metric, or that treats
agreement with a season-long market as evidence of weekly accuracy. Allowed
uses and the not-allowed list are in the decision note.

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

**Training belongs in batch, not in request handlers.** Fit in GitHub Actions
or another batch runner, write artifacts and results to the database, and let
the API serve completed runs. A model fit inside a Vercel request handler is a
finding — it is also the same serverless function whose cold start PR #65 was
about.

## Do not

- Skip, disable, or quarantine a test to get CI green
- Promote a shadow run, reseal a release, or flip a production default as a
  side effect of an unrelated change
- Treat `scripts/audit_weekly_features.py` exiting 0 as a live-data seal; it
  walks a static contract registry and does not open the database
