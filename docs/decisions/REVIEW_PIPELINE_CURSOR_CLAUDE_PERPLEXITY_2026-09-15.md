# Review pipeline — Cursor / Claude / Perplexity — 2026-09-15

**Date:** 2026-09-15
**Scope:** pull-request process and model-governance review. No change to production projection code, sealed boards, or release pointers.
**PR template:** [`.github/PULL_REQUEST_TEMPLATE.md`](../../.github/PULL_REQUEST_TEMPLATE.md) (may land via a sibling PR if not yet on the default branch)
**Cursor agent snippet:** [`docs/ops/CURSOR_PR_CONTRACT.md`](../ops/CURSOR_PR_CONTRACT.md) (same)
**Ops note:** [`docs/ops/PERPLEXITY_GOVERNANCE_REVIEW.md`](../ops/PERPLEXITY_GOVERNANCE_REVIEW.md)
**Action:** [`.github/workflows/perplexity-governance-review.yml`](../../.github/workflows/perplexity-governance-review.yml)

## Decision

Every PR uses the contract checklist in the GitHub PR template. Review is four roles, in order: **Cursor implements**, **Claude reviews code quality**, **Perplexity reviews systems / model governance**, **a human merges and promotes**.

Perplexity is an independent governance pass. It is **not** a second Claude review and must not merely echo Claude's comments.

The contract language below is summarized from existing pipeline docs, not a new mathematical design. Authoritative behavior remains [`docs/PIPELINE_MAP.md`](../PIPELINE_MAP.md) and the dated records under `docs/decisions/` and `docs/history/`.

## Roles

| Role | Actor | Job |
|---|---|---|
| Implementation | Cursor (and any human author) | Branch, tests, filled PR template, no silent contract skips |
| Code quality | Claude Code | Diff readability, tests, obvious bugs, workflow/CI issues. Runs on PRs via `.github/workflows/claude-code-review.yml` |
| Systems / model governance | Perplexity | Leakage, provenance, release gates, statistical validity. Answers whether the PR preserves the mathematical, calibration, artifact, and release-control contract. Runs on PRs via `.github/workflows/perplexity-governance-review.yml` |
| Final merge / promotion | Human | Merge the PR. Separate human step for pointer promotion / rollback (`promote_release`) |

Claude may catch a broken test or an unclear rename. Perplexity must still ask whether hashes, holdouts, gates, and fail-closed paths still mean what [`PIPELINE_MAP.md`](../PIPELINE_MAP.md) says they mean.

## Ordered workflow

1. **Cursor** opens a branch and PR, runs the relevant tests, and fills `.github/PULL_REQUEST_TEMPLATE.md` when that template is present (see [`docs/ops/CURSOR_PR_CONTRACT.md`](../ops/CURSOR_PR_CONTRACT.md)).
2. **Claude Code** reviews the diff for code quality (existing PR Action).
3. **Perplexity** posts an informational governance comment via the GitHub Action (Phase 1). Cursor agents do **not** need to call Perplexity manually when the Action runs.
4. **Human** merges. Promotion of a sealed namespace remains a separate fail-closed command, not a GitHub merge.

Do not skip Perplexity because Claude was green. Do not treat a Perplexity “looks good” that restates Claude's bullet list as a completed governance review.

## The question Perplexity must answer

> Does this PR preserve the mathematical, calibration, artifact, and release-control contract?

That includes, when in scope:

- Selected-board hash behavior (`selected_board_hash` / `selected_board_sha256` alignment across simulation manifest, players meta, and release report). See PIPELINE_MAP §4 and §8a.
- Canonical projection run identity (`canonical_projection_run_id` on the simulation manifest, matching the board).
- Recenter transform version (`v1_median_correction` in `src/projection/inference/recenter.py`).
- WR calibration artifact binding (`wr_calibration_version = v1_wr_residual_scale`, hash of `output/model_v3/wr_calibration.json`).
- Finish-probability gate and simulated-VORP gate (fail-closed overlay attach; VORP overlay requires a ready finish gate). PIPELINE_MAP §7.
- Partition / hash validation on draw artifacts.
- Fail-closed behavior: missing or `hold` verdicts do not ship overlays; promotion has no force flag.
- Deterministic board-field parity (displayed stats/overlays match the selected board they claim to describe).
- No mutation of the curated depth-chart source (`src/depth_chart/starters_YYYY.csv` is never auto-edited). PIPELINE_MAP §1.

Out of scope is a valid answer. The PR description must mark those contract lines `N/A` instead of leaving them blank.

## What Perplexity reviews that Claude does not own

| Claude (code quality) | Perplexity (governance) |
|---|---|
| Is the diff readable and tested? | Did evaluation use a leakage-safe cutoff / unused holdout? |
| Are names, types, and CI sane? | Are calibration / donor / WR-scale / gate hashes still bound to the artifacts they name? |
| Would this merge break the app build? | Would this merge let a `hold` overlay, unlisted artifact, or pointer rewrite ship? |
| Style and obvious logic bugs | Statistical validity vs the frozen baseline, not vs Sleeper agreement (PIPELINE_MAP §7b: external comparison is diagnostic, never a gate) |

If Perplexity's findings are a paraphrase of Claude's, the review is incomplete. Re-run against the contract question and the checklists below.

## PR-type checklists

These are summaries of existing gates. They are not new thresholds.

### Modeling / calibration

Use when the PR changes point models, ensembles, uncertainty, joint bootstrap, WR residual scale, or calibration scripts.

- Holdout stays unused for selection; rolling-origin / leakage-safe folds only. See [`ACCURACY_FIRST_ENSEMBLE_2026-08-27.md`](ACCURACY_FIRST_ENSEMBLE_2026-08-27.md) and [`docs/history/FREEZE_2026-08-13.md`](../history/FREEZE_2026-08-13.md).
- v3 remains distribution-only unless a separate, already-documented means gate passes. [`V1_PRODUCTION_ROLE_2026-08-29.md`](V1_PRODUCTION_ROLE_2026-08-29.md), [`V3_PROBABILISTIC_PIPELINE.md`](V3_PROBABILISTIC_PIPELINE.md).
- Calibration artifacts stay hash-pinned: uncertainty `artifact_hash`, joint-donor `sha256`, WR calibration hash, segment report hash. PIPELINE_MAP §5–§6.
- Fail-closed: a `hold` or missing verdict zeroes uncertainty / withholds overlay rather than shipping a partial candidate. PIPELINE_MAP §6; `scripts/v3_promotion_gate.py`.
- RB/WR accuracy-first weights stay frozen; the shadow repair track is closed. [`V1_PRODUCTION_ROLE_2026-08-29.md`](V1_PRODUCTION_ROLE_2026-08-29.md).

### Simulation / draw infrastructure

Use when the PR changes draw generation, profiles, partitions, recenter, or release-bundle sealing.

- Draw count and profile identity come from `config/simulation.json` (`profile_key`, `draw_count`, `configuration_hash`, `policy_hash`). PIPELINE_MAP §5; [`DRAW_COUNT_ROLLOUT_2026-08-28.md`](DRAW_COUNT_ROLLOUT_2026-08-28.md).
- Draws stay partitioned with per-partition hashes on the manifest.
- Recenter stays `selected + (v3_draw - v3_p50)`, floored, median-corrected; transform version must match the gate provenance.
- Sealed bundles are immutable. Going live is a pointer swap after six promotion invariants. [`PROMOTION_PROVENANCE_2026-08-30.md`](PROMOTION_PROVENANCE_2026-08-30.md), [`PHASE1_RELEASE_SIGN_OFF_2026-08-29.md`](PHASE1_RELEASE_SIGN_OFF_2026-08-29.md).
- Do not rewrite sealed bytes in place to “fix” a missing field; next namespace picks it up. PIPELINE_MAP §10.

### Finish probability / VORP

Use when the PR changes overlay attach, rank tie policy, replacement contract, or `prepare` export.

- Finish-probability fields attach only after `finish_probability_ready` / publication pass. Simulated VORP requires that finish gate plus its own gate. PIPELINE_MAP §7.
- Overlays are computed on **recentered** draws anchored on the accuracy-first selected forecast.
- Deterministic `vorp`, ranks, and tiers remain authoritative; `sim_vorp_*` is an uncertainty overlay.
- Rank tie policies stay distinct (`first` for finish probabilities, `min` for expected/median rank). Do not unify them silently.
- Overlay coverage must align with the exported board (promotion invariant `overlay_coverage_alignment`).

### Weekly model / feature (shadow)

Use when the PR changes weekly features, weekly-v2, hierarchical weekly ROS, or opportunity-first mean work.

- Target-week outcomes must not enter that week's features (`shift(1)` before rolling). PIPELINE_MAP §2; `scripts/audit_weekly_features.py`.
- Stay shadow until the candidate beats the accuracy-first incumbent on leakage-safe top-120 (or the documented weekly promotion gate). Do not wire into compose / publish / auto-publish on a green unit test alone. PIPELINE_MAP §9.
- Mark production board/simulation/release contracts `N/A` when they are truly untouched, and still fill Evaluation and Risk.

## Phase 1 GitHub Action (live, informational)

The Action is live as of this document. It posts one upserted PR comment (`## Perplexity governance review (informational)`). It is **not** a required merge blocker.

| Behavior | What happens |
|---|---|
| Secret present | Collects a review packet (`gh` + checkout), calls Perplexity Sonar (`POST https://api.perplexity.ai/v1/sonar`, model `sonar-pro`), posts the structured comment, exits 0 |
| Secret missing | Job succeeds and posts: `Perplexity review skipped: PERPLEXITY_API_KEY not configured` |
| Recommendation `REQUEST CHANGES` or `ESCALATE` | Severity is in the comment body. The workflow job still succeeds |
| `cursor[bot]` PR updates | Allowed (same reason Claude sets `allowed_bots: cursor`) |

Hard blocker (required status check) is **deferred until the review is calibrated**. Do not add this workflow to branch protection yet.

### Secret

- **Name:** `PERPLEXITY_API_KEY` (repository Actions secret, not a production-environment secret).
- **Create the key:** [Perplexity API settings](https://www.perplexity.ai/account/api).
- **Store it:** GitHub → repo → Settings → Secrets and variables → Actions → New repository secret.

Richard must add that secret before reviews actually run. Until then the skip comment is the expected, non-blocking behavior.

When the Action is running, Cursor agents do not call Perplexity by hand. See [`docs/ops/PERPLEXITY_GOVERNANCE_REVIEW.md`](../ops/PERPLEXITY_GOVERNANCE_REVIEW.md).

## Suggested Perplexity merge-gate prompt

The Action encodes this prompt (plus the checklists above). Copy and fill only if you are reviewing manually because the Action did not run.

```text
You are the independent model-governance reviewer for Fantasy Decisions
(fantasy-projections). This is not a code-style review. Claude Code already
did that; do not restate Claude's findings unless you are disagreeing with
them or adding a contract issue Claude missed.

Question you must answer:
Does this PR preserve the mathematical, calibration, artifact, and
release-control contract?

Read, in this order:
1. The PR description (contract checklist). Treat blank contract rows as a
   defect. N/A is allowed only when written on the line.
2. The PR diff.
3. Tests / CI status for this PR.
4. docs/PIPELINE_MAP.md (especially §4 board hash, §5–§6 simulation and
   acceptance, §7 overlays, §8a release layer, §9 shadow tracks).
5. docs/decisions/ (especially ACCURACY_FIRST, V1_PRODUCTION_ROLE,
   PROMOTION_PROVENANCE, PHASE1_RELEASE_SIGN_OFF, SIMULATION_MODE,
   DRAW_COUNT_ROLLOUT, V3_PROBABILISTIC_PIPELINE) and docs/history/FREEZE_2026-08-13.md.
6. Any changed manifests, gate JSON, freeze manifests, simulation profile
   hashes, or release-bundle files.

Classify the PR: modeling/calibration, simulation/draw infrastructure,
finish probability/VORP, weekly/shadow, docs/ops only, or mixed. Apply the
matching checklist in docs/decisions/REVIEW_PIPELINE_CURSOR_CLAUDE_PERPLEXITY_2026-09-15.md.

Output exactly one of:
- APPROVE
- REQUEST CHANGES
- ESCALATE

Then list findings ranked by severity (S1 release/leakage/hash contract,
S2 statistical or provenance gap, S3 documentation/process). Each finding
must cite a file, gate, or PIPELINE_MAP section. If you have no contract
findings, say so and still fill Evaluation gaps (tests, holdout, artifacts).

Do not approve because the code “looks clean.” Approve only if the contract
question is answered with evidence from the PR description, tests, and
pinned artifacts — or if the PR is explicitly docs/ops with production
contracts marked N/A.
```

## Phase plan

| Phase | What happens | Merge blocker? |
|---|---|---|
| **1 — now (live)** | GitHub Action posts the governance prompt / checklist as an informational PR comment. Missing `PERPLEXITY_API_KEY` is a green skip, not a red X | No |
| **2 — later** | Calibrate the Action (false-positive rate, comment quality vs Claude) | Still no |
| **3 — not yet** | Hard required status check | Do not enable until a human records that the review is calibrated |

Phase 1 Actions are the adopted process. The earlier “manual comment, Action later” split is closed: the Action is the Phase 1 implementation. Do not make Perplexity a required merge gate until a human records that decision in a follow-up document.

## Cursor agents, when opening a PR

- Fill `.github/PULL_REQUEST_TEMPLATE.md` when it exists. Do not substitute a shorter description.
- For weekly / research / docs PRs, mark production contracts `N/A` and still fill Evaluation and Risk.
- Do not change production projection code, sealed boards, or release pointers unless the task says so and the in-scope contracts are checked with evidence.
- Do not call Perplexity manually when the governance Action is running on the PR.

## Related

- [`docs/ops/PERPLEXITY_GOVERNANCE_REVIEW.md`](../ops/PERPLEXITY_GOVERNANCE_REVIEW.md) — secret, Claude vs Perplexity, agent ops
- [`docs/PIPELINE_MAP.md`](../PIPELINE_MAP.md) — pipeline, gates, live release state
- [`docs/history/FREEZE_2026-08-13.md`](../history/FREEZE_2026-08-13.md) — freeze gates and artifact hashes
- [`ACCURACY_FIRST_ENSEMBLE_2026-08-27.md`](ACCURACY_FIRST_ENSEMBLE_2026-08-27.md)
- [`V1_PRODUCTION_ROLE_2026-08-29.md`](V1_PRODUCTION_ROLE_2026-08-29.md)
- [`PROMOTION_PROVENANCE_2026-08-30.md`](PROMOTION_PROVENANCE_2026-08-30.md)
- [`PHASE1_RELEASE_SIGN_OFF_2026-08-29.md`](PHASE1_RELEASE_SIGN_OFF_2026-08-29.md)
- [`SIMULATION_MODE_2026-08-26.md`](SIMULATION_MODE_2026-08-26.md)
- [`DRAW_COUNT_ROLLOUT_2026-08-28.md`](DRAW_COUNT_ROLLOUT_2026-08-28.md)
- [`V3_PROBABILISTIC_PIPELINE.md`](V3_PROBABILISTIC_PIPELINE.md)
