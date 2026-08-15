# Root markdown inventory — 2026-08-15

25 markdown files sit in the repo root, accumulated across 54 commits and many
"phases". This file says what each one is, whether it is still true, and where
it should live. Companion to [`STATE_OF_BUILD.md`](STATE_OF_BUILD.md).

Built by reading every file plus `git log` / `git ls-files` / `git status` and,
where a claim was checkable, the working-tree source and the artifacts in
`output/`. No source file was modified and no pipeline or test was run.

## Status vocabulary

| Status | Meaning |
|---|---|
| **live** | Currently true and currently useful. Keep in root. |
| **decision record** | Documents a shipped tuning value. Authoritative for *what ships*; the *gate* may be contested — see the note under the table. |
| **historical** | Accurate for the state it describes, superseded by named later work. Valuable as a record; not a guide to the current build. |
| **stale** | Contains at least one claim that is now false, unqualified. Quoted below. |

`git` column: **T** = tracked, **T·M** = tracked and modified in the working
tree, **U** = untracked (exists only in this working tree).

---

## 1. Inventory

| File | Date | git | Purpose | Status |
|---|---|:--:|---|---|
| `STATE_OF_BUILD.md` | 08-15 | U | Current-state orientation document | **live** |
| `DOC_INVENTORY.md` | 08-15 | U | This file | **live** |
| `PROVENANCE_AUDIT.md` | 08-15 | U | Forensic classification of all 43 allocation constants + 19 fitted layers; Sleeper feedback-loop inventory | **live** (another agent's active deliverable — not edited here; one claim overtaken, see §2.4) |
| `REPO_HYGIENE_AUDIT.md` | 08-15 | U | What must be committed, what is generated, what is scratch; staging commands | **live** (another agent's active deliverable — not edited here) |
| `AGE_EFFECT_SHRINKAGE_2026-08-14.md` | 08-14 | T | Ships RB-only age-effect shrink; grid search + rolling-origin + fantasy holdout; discloses its own RB VORP regression traced to David Montgomery | **decision record** |
| `DEPTH_CHART_ALLOCATION_2026-08-14.md` | 08-14 | T | Five-phase allocation remediation: rushing coverage, replacement rows, rookie vacancy netting, usage-share prior, board tripwires | **decision record** (one section now contradicted — §2.3) |
| `DRAFT_CAPITAL_REMOVAL_2026-08-14.md` | 08-14 | T | Removes `draft_round`/`draft_pick` from player-level models; keeps `career_year` | **decision record** |
| `HIERARCHICAL_PASS_MIX_2026-08-14.md` | 08-14 | U | L2 team WR/TE/RB target-share mix + L3 within-group composition; WR formation roles | **decision record** (no LOSO result recorded — §2.5) |
| `HIERARCHICAL_RUSH_MIX_2026-08-14.md` | 08-14 | U | L2 RB/QB/OTHER carry-share mix; LOSO table, `beats_prior=True` | **decision record** |
| `OC_INHERITANCE_FIT_2026-08-14.md` | 08-14 | U | Ships `INHERITANCE_WEIGHTS` 60/40 for internal and outside hires; LOSO grid | **decision record** |
| `OL_TRAILING_2026-08-14.md` | 08-14 | U | Ships `OL_TRAILING_SEASONS = 3`, live predict path only | **decision record** |
| `RB_CARRY_VACANCY_2026-08-14.md` | 08-14 | U | Ships `INCUMBENT_VACANCY_ALPHA['carry'] = 1.0` | **decision record** (contradicts the record above it — §2.3) |
| `FANTASY_EVALUATION_2025_REPORT.md` | 08-13 | T·M | The leakage-safe 2024→2025 holdout report | **stale** — §2.1 |
| `FREEZE_2026-08-13.md` | 08-13 | T | SHA-256 artifact manifest and release gates for the 2026 board | **stale** — §2.2 |
| `VALIDATION_EVALUATION_REMEDIATION.md` | 08-13 | T | Fixes validation leakage, makes the Sleeper proxy auditable | **historical** (superseded by `PHASE7_REMEDIATION_REPORT.md` → the composition unification; its Sleeper-proxy section superseded by `PROVENANCE_AUDIT.md` §4) |
| `PHASE7_REMEDIATION_REPORT.md` | 08-13 | T | Projection-integrity remediation; team-model grain collision fix; structural follow-ups | **historical** (its "Structural follow-ups" list is still live and is carried into `STATE_OF_BUILD.md` §5) |
| `PHASE6_REPORT.md` | 08-05 | T | 2026 team reassignment, curated depth chart, output gating | **historical** (superseded by `roster_moves.py` + `depth_gating.py` + `DEPTH_CHART_ALLOCATION_2026-08-14.md`) |
| `PHASE5_REPORT.md` | 08-03 | T | 2026 projections: intervals, UDFA baselines, OC framing, first deliverable | **historical** (superseded by Phase 7 + `composition.py`; §4 "OC-inheritance framing — stated limitation, not implemented" superseded by `OC_INHERITANCE_FIT_2026-08-14.md`) |
| `PHASE4_REPORT.md` | 08-05 | T | The player models: features, granularity, training scope, backtest, 4 addenda. 46 KB | **historical** (still the best description of *why* the feature set looks like it does; its backtest table and feature list are both superseded — §2.6) |
| `PHASE3_REPORT.md` | 08-02 | T | Coordinator tendency table, OC assignments, inheritance rule | **historical** (its judgment-call inheritance rule superseded by the fitted weights in `OC_INHERITANCE_FIT_2026-08-14.md`) |
| `PHASE2_REBUILD_REPORT.md` | 08-02 | T | Pooled multi-season OL attribution replacing five per-season fits | **historical** (this *is* the shipped OL path; superseded as a description by Phase 7's exact-season historical fix and `OL_TRAILING_2026-08-14.md`) |
| `PHASE2_STABILITY_INVESTIGATION.md` | 08-14 | T·M | Root-causes OL year-over-year correlation of 0.144; four hypotheses tested | **historical** (its recommendation was implemented by `PHASE2_REBUILD_REPORT.md`) |
| `PHASE2_REPORT.md` | 08-02 | T | Original per-season OL ridge fits | **historical** (explicitly superseded — `PHASE2_REBUILD_REPORT.md` says the old path is "**not** the path Phase 4 should read from") |
| `PHASE1_REPORT.md` | 08-02 | T | Data layer: cache, 19 nflverse sources, ID crosswalk finding | **historical** (structure still accurate; season window and source list have moved) |
| `PHASE0_REPORT.md` | 08-02 | T | Data-validation gate for the OL model; participation coverage matrix | **historical** (a gate decision that was passed; of archival interest only) |

Two files named in the current work — `ABLATION_RESULTS.md` and
`SLEEPER_RETIREMENT.md` — **do not exist yet** and are other agents'
deliverables. Add them to this table as **live** when they land.

### On the decision records

All eight `*_2026-08-14.md` files are **live and authoritative for the values
they ship** — those values are in `src/projection/contracts.py` and
`src/coordinator/inheritance.py` right now. They are simultaneously **under
review**: `PROVENANCE_AUDIT.md` reclassifies several of their gates as
proxy-based (share MAE, tendency MAE, coverage fraction, OL-score persistence)
rather than outcome-based, and identifies Sleeper as the deciding evidence
behind others. Both statements hold at once. This inventory records the tension
and does not relitigate it.

---

## 2. Stale and contradicted claims, quoted

### 2.1 `FANTASY_EVALUATION_2025_REPORT.md` — "Production parity and limits"

> "The evaluation applies the shipped veteran depth-rate ladder, predicted
> availability, exposure-weighted receiving composition, mutually exclusive QB
> volume allocation, team passing/rushing anchors, player stat constraints, and
> canonical season-total exposure."

**Now understates coverage.** *Verified in source:* `fantasy_evaluation.py:23`
imports `compose_board` and `leakage_safe_context` from
`src/projection/composition.py`, and `_compose_and_reconcile` "no longer
contains any allocation logic of its own". The harness runs the same 15-stage
sequence `predict.project_season` ships, which additionally includes the
hierarchical pass mix (L2 + L3), the hierarchical rush mix (L2 + L3),
`propagate_team_anchors`, `apply_deep_bench_games_cap` and
`apply_status_overrides`.

> "Historical curated roles, target-year coordinator context, and the production
> elite residual correction are unavailable on a strictly preseason-consistent
> 2025 path and are therefore omitted."

**The specific limits changed.** *Verified in source:*
`run_evaluation.coverage_limits` now enumerates a different set — curated
depth-chart research (2026-only files), dated status overrides, the elite
residual correction, **and prediction intervals** — while target-year
coordinator context is explicitly *not* omitted: `leakage_safe_context`'s
docstring states OC mix inheritance "keys on `(season, team)` in
oc_assignments.csv, whose rows are preseason-known coaching hires, so it is
applied at its real historical value rather than suppressed." Each unavailable
stage now degrades to a recorded pass-through in
`CompositionContext.describe_coverage()` rather than being silently dropped.

> "The full test suite passes: 57 tests."

**Superseded.** `AGE_EFFECT_SHRINKAGE_2026-08-14.md` reports 63; a static count
of `def test_` across the 16 files in `tests/` gives 120. No suite was run for
this inventory.

The report's **result tables are not marked stale** — they were correct when
produced. They are superseded by later runs; see §2.2 and `STATE_OF_BUILD.md`
§3.3. A correction note has been prepended to the report pointing at
`STATE_OF_BUILD.md`; the body is untouched.

### 2.2 `FREEZE_2026-08-13.md` — the manifest matches nothing

> "This snapshot is the frozen analytical baseline for the 2026 fantasy draft
> board."

> "Base repository commit before the working-tree remediation:
> `df37452221a90ff1ebf1e6c00a6e8bcaf610b65b`."

**Void as a manifest.** *Verified:* all six pinned SHA-256 hashes fail against
the current `output/` — zero of six match. `df37452` predates the entire
allocation layer (`team_reconcile.py`, `roster_moves.py`, `replacement.py`,
`depth_gating.py`, `depth_rates.py`, `veterans.py`, `team_pass_mix.py`,
`team_rush_mix.py`, `composition.py`, `contracts.py`, `artifacts.py`), none of
which is committed even now (`git ls-files src/projection/` returns 13 files).
Eleven commits have landed on `master` since.

> "Complete pytest suite: **57 passed**." · "2026 projection: 3,969
> player-stat rows" · "Fantasy board: 768 player-position rows"

Row counts and test count all predate the allocation layer. `PROVENANCE_AUDIT.md`
reads 4,039 rows off the current board.

The freeze's **caveat paragraph** — that the model wins on Spearman and points
MAE but not on tier hits or TE VORP, and that a future change should be held to
the tier and VORP columns — is the most useful thing in the file and remains
sound guidance even though its numbers moved.

### 2.3 Two live decision records contradict each other on the same constant

`DEPTH_CHART_ALLOCATION_2026-08-14.md` §"Still open":

> "The **RB lead-back level bias** noted in project memory still blocks
> `INCUMBENT_VACANCY_ALPHA['carry']`."

`RB_CARRY_VACANCY_2026-08-14.md` §"Decision":

> "**Shipped** `INCUMBENT_VACANCY_ALPHA[\"carry\"] = 1.0` (measured value)."

*Verified in source:* `src/projection/contracts.py:45` reads
`INCUMBENT_VACANCY_ALPHA = {"target": 0.5, "carry": 1.0}`. The second record
governs; the first's "Still open" bullet is stale within an otherwise live
document. `PROVENANCE_AUDIT.md` separately classifies this constant **class C**
(Sleeper-tuned), noting four of the five rows in the justifying table are
Sleeper comparisons. Recorded, not resolved.

### 2.4 `PROVENANCE_AUDIT.md` §0.2 — overtaken by `composition.py`

> "It does **not** import `team_pass_mix`, `team_rush_mix`, `roster_moves`,
> `replacement`, `depth_gating`, or `corrections`."

*Verified in source:* `fantasy_evaluation.py` imports `compose_board`, which
imports `team_pass_mix`, `team_rush_mix`, `depth_gating` and `team_reconcile`.
The pass- and rush-mix layers are therefore now inside the scored path (fit on
`history_seasons <= source_season`), while `roster_moves`, `replacement` and
`corrections` remain outside it. The audit's structural conclusion holds for
those three. `REPO_HYGIENE_AUDIT.md` §5.4 independently notes that
`composition.py` "appeared *during* this audit". **Not edited** — the audit is
another agent's live deliverable; flagged here for its author.

### 2.5 `HIERARCHICAL_PASS_MIX_2026-08-14.md` — a gate with no result

> "Ship the scheme mix only when LOSO MAE beats prior-season mix
> (`beats_prior`)."

No LOSO number appears anywhere in the repo for the pass mix — the rush-mix
record has a table, this one has only the command and the condition. *Verified
in source per `PROVENANCE_AUDIT.md` §2 and consistent with `predict.py`:*
`build_team_pass_mix_profiles` never calls `validate_mix_model`, and
`compose_board` attaches the mix unconditionally. Not stale (it never asserted a
result), but the condition it sets is unverified and the layer ships anyway.

### 2.6 `PHASE4_REPORT.md` — feature list and backtest table superseded

> "the model beats the naive carry-forward baseline on **18 of 20**
> position/stat combinations"

Correct as of Phase 4 and directionally still the right summary, but the table
predates Phase 7's leakage remediation, the draft-capital removal, and the RB
age-shrink retrain. Its §"Phase 5" feature discussion also describes
`draft_round`/`draft_pick` as model inputs; `DRAFT_CAPITAL_REMOVAL_2026-08-14.md`
removed both. Marked **historical** rather than **stale** because the report
explicitly scopes its numbers to its own build.

---

## 3. Proposed archival structure

Root is reserved for documents that describe the build *as it is*. History moves
to `docs/history/`, decision records to `docs/decisions/`.

**After the move, root holds:** `STATE_OF_BUILD.md`, `DOC_INVENTORY.md`,
`PROVENANCE_AUDIT.md`, `REPO_HYGIENE_AUDIT.md`, and — when they land —
`ABLATION_RESULTS.md`, `SLEEPER_RETIREMENT.md`. Plus a root `README.md`, which
does not currently exist and should.

### Do not run this yet

Two other agents hold the tree. Run this only when `git status` is quiet, and
land it as its own commit so the moves are reviewable as renames.

`git mv` requires a tracked file; seven of these are untracked and need plain
`mv`. Both forms are below, separated.

```bash
# ── 0. Directories ────────────────────────────────────────────────────────────
mkdir -p docs/history docs/decisions

# ── 1. Tracked files → git mv ────────────────────────────────────────────────
# Superseded phase reports and one-off remediation write-ups.
git mv PHASE0_REPORT.md                    docs/history/
git mv PHASE1_REPORT.md                    docs/history/
git mv PHASE2_REPORT.md                    docs/history/
git mv PHASE2_REBUILD_REPORT.md            docs/history/
git mv PHASE2_STABILITY_INVESTIGATION.md   docs/history/
git mv PHASE3_REPORT.md                    docs/history/
git mv PHASE4_REPORT.md                    docs/history/
git mv PHASE5_REPORT.md                    docs/history/
git mv PHASE6_REPORT.md                    docs/history/
git mv PHASE7_REMEDIATION_REPORT.md        docs/history/
git mv VALIDATION_EVALUATION_REMEDIATION.md docs/history/

# Stale but historically valuable: the holdout report and the void freeze.
# Keep both — the freeze records what was gated on, and the report records
# numbers that were correct when produced.
git mv FANTASY_EVALUATION_2025_REPORT.md   docs/history/
git mv FREEZE_2026-08-13.md                docs/history/

# Dated decision records (tracked three).
git mv AGE_EFFECT_SHRINKAGE_2026-08-14.md   docs/decisions/
git mv DEPTH_CHART_ALLOCATION_2026-08-14.md docs/decisions/
git mv DRAFT_CAPITAL_REMOVAL_2026-08-14.md  docs/decisions/

# ── 2. Untracked files → plain mv (git mv fails on these) ────────────────────
mv HIERARCHICAL_PASS_MIX_2026-08-14.md docs/decisions/
mv HIERARCHICAL_RUSH_MIX_2026-08-14.md docs/decisions/
mv OC_INHERITANCE_FIT_2026-08-14.md    docs/decisions/
mv OL_TRAILING_2026-08-14.md           docs/decisions/
mv RB_CARRY_VACANCY_2026-08-14.md      docs/decisions/

# ── 3. Stays in root (live) ─────────────────────────────────────────────────
# STATE_OF_BUILD.md  DOC_INVENTORY.md  PROVENANCE_AUDIT.md  REPO_HYGIENE_AUDIT.md
# …plus ABLATION_RESULTS.md and SLEEPER_RETIREMENT.md when they land.

# ── 4. Verify ───────────────────────────────────────────────────────────────
git status --porcelain
ls *.md docs/history docs/decisions
```

### Follow-ups the move creates

1. **Relative links break.** Several decision records link to source with
   root-relative paths — e.g. `HIERARCHICAL_PASS_MIX_2026-08-14.md` contains
   `[src/projection/team_pass_mix.py](src/projection/team_pass_mix.py)`, and
   `OL_TRAILING_2026-08-14.md`, `OC_INHERITANCE_FIT_2026-08-14.md` do the same.
   From `docs/decisions/` these need a `../../` prefix. `STATE_OF_BUILD.md` and
   `DOC_INVENTORY.md` link to each other and to the two audits; those survive
   because all four stay in root. `src/projection/README.md` already uses
   correct relative links and does not move.
2. **Two phase reports are *generated files*, and moving them will silently
   recreate them in root.** *Verified in source:*

   - `src/ol_model/pipeline.py:16` — `REPORT_PATH = os.path.join(REPO_ROOT,
     "PHASE2_REPORT.md")`, written by its `main()`.
   - `src/ol_model/pooled_pipeline.py:20` — same pattern for
     `PHASE2_REBUILD_REPORT.md`.

   Either update both `REPORT_PATH` constants to `docs/history/` in the same
   commit as the move, or leave these two files in root. Note also that
   `pooled_pipeline.py` embeds prose citing `PHASE2_REPORT.md` and
   `PHASE2_STABILITY_INVESTIGATION.md` by root-relative name into the report it
   emits (lines 100–156), so those strings need updating too. This is the one
   item in the move that can fail silently.

3. **Prose cross-references from source comments.** ~25 sites across `src/` name
   a root markdown file — `predict.py` cites `PHASE5_REPORT.md`;
   `contracts.py:44` cites `RB_CARRY_VACANCY_2026-08-14.md`;
   `depth_gating.py:239` cites `PHASE6_REPORT.md`; `coordinator/inheritance.py`
   and `oc_profiles.py` cite `OC_INHERITANCE_FIT_2026-08-14.md`; `backtest.py`
   cites `PHASE1_REPORT.md` and `PHASE5_REPORT.md`; several `ol_model/` modules
   cite the Phase 2 pair. These are comments, not links, so nothing breaks — but
   grep for `PHASE`, `_2026-08-14`, `FREEZE_2026` and `FANTASY_EVALUATION_2025`
   under `src/` and update the paths in the same commit so the references stay
   followable.
4. **`REPO_HYGIENE_AUDIT.md` §7 staging commands name these files at their root
   paths.** Whoever runs the moves must reconcile with that block, or run the
   staging first and the moves second. Staging first is the safer order: it
   makes every subsequent move a clean tracked rename.
5. **Write a root `README.md`** pointing at `STATE_OF_BUILD.md`. Its absence is
   the reason this documentation debt was invisible.
