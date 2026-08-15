# Repository hygiene audit — 2026-08-15

Branch `master`, 54 commits, `.git` is 49 MB. Working tree at audit time: 26 modified,
5 deleted-unstaged, 32 untracked (plus 2 untracked files hidden by an over-broad ignore
rule — see §5.4).

**Nothing was deleted, committed, staged, or reverted.** The only file modified by this
audit is `.gitignore` (two additive lines, §6) and this report.

---

## 1. Production code that must be committed

### 1.1 New `src/` modules (untracked)

`predict.py` is the pipeline entrypoint (`python -m src.projection.predict --season 2026`).
Every module below is on its import graph. Verified by grep across the repo excluding
`.venv`, `.git`, `__pycache__`.

| File | LOC | What imports it (verified) | Action |
|---|---:|---|---|
| `src/projection/contracts.py` | 146 | 13 importers: `predict.py:72`, `depth_chart/live.py:9`, `artifacts.py:12`, `depth_gating.py:13`, `depth_rates.py:9`, `replacement.py:10`, `rookies.py:35`, `roster_moves.py:11`, `team_pass_mix.py:20`, `team_reconcile.py:10`, `ol_quality.py:36`, plus `scripts/diag_rb_carry_vacancy.py:18` | **Commit** |
| `src/projection/artifacts.py` | 64 | `predict.py:114`, `veterans.py:12` (`load_availability_models`) | **Commit** |
| `src/projection/depth_rates.py` | 26 | `predict.py:113`, `depth_gating.py:24` (`depth_rate_factor`) | **Commit** |
| `src/projection/depth_gating.py` | 314 | `predict.py:120`, `veterans.py:13`; **patched by two tracked tests** — `tests/test_depth_refresh.py:191`, `tests/test_runtime_output_correctness.py:544,561` | **Commit** |
| `src/projection/roster_moves.py` | 696 | `predict.py:129`, `veterans.py:27`, `scripts/diag_rb_carry_vacancy.py:24` | **Commit** |
| `src/projection/replacement.py` | 148 | `predict.py:139` | **Commit** |
| `src/projection/veterans.py` | 366 | `predict.py:145` | **Commit** |
| `src/projection/team_reconcile.py` | 1224 | `predict.py:159`, `veterans.py:31` | **Commit** |
| `src/projection/team_pass_mix.py` | 559 | `predict.py:61`, `team_rush_mix.py:17`, `coordinator/inheritance.py:78-79`, `tests/test_team_pass_mix.py:7` | **Commit** |
| `src/projection/team_rush_mix.py` | 416 | `predict.py:66`, `tests/test_team_rush_mix.py:7` | **Commit** |
| `src/coordinator/inheritance.py` | 250 | `coordinator/oc_profiles.py:40` (`INHERITANCE_WEIGHTS`), `team_pass_mix.py:19`, `team_rush_mix.py:15`, `scripts/fit_oc_inheritance_weights.py:11` | **Commit** |

> **Note — four modules beyond the six flagged in the brief.** The brief named six
> production modules. `veterans.py`, `team_reconcile.py`, `team_pass_mix.py`, and
> `team_rush_mix.py` are equally live (all four imported directly by `predict.py`) and
> total 2,565 LOC. `team_reconcile.py` alone is the single largest untracked file in the
> repo. Any staging command that misses these leaves `predict.py` unimportable.

**Breakage check:** `src/coordinator/oc_profiles.py` and `src/projection/ol_quality.py`
are *tracked and modified* files that import *untracked* modules. Committing the tracked
modifications without the untracked modules produces a repo that cannot import
`predict.py` at all. These must go in the same commit.

### 1.2 New tests (untracked)

| File | LOC | Evidence | Action |
|---|---:|---|---|
| `tests/test_team_pass_mix.py` | 149 | Imports `src.projection.team_pass_mix`; matches `tests/test_*.py` collection convention; the 13 sibling tests are all tracked | **Commit** |
| `tests/test_team_rush_mix.py` | 86 | Imports `src.projection.team_rush_mix`; same convention | **Commit** |

These are the only two files in `tests/` not tracked. `FREEZE_2026-08-13.md` cites a
release gate of "57 passed" — the suite is a gate, so its members belong in git.

### 1.3 Documentation (untracked)

| File | Evidence | Action |
|---|---|---|
| `src/projection/README.md` | Architecture doc for the pipeline; explicitly names the stage modules extracted from the former monolithic predict path (`depth_rates`, `artifacts`, `depth_gating`, `roster_moves`, `replacement`, `veterans`, `team_reconcile`) and states the curated-depth-chart rule. Documents exactly the refactor in §1.1. | **Commit** |

### 1.4 Fitting / evaluation harnesses (untracked, `scripts/`)

These are cited *by name* in the decision records as the method behind shipped constants.
Discarding them makes those records unreproducible.

| File | LOC | Evidence | Action |
|---|---:|---|---|
| `scripts/fit_oc_inheritance_weights.py` | 35 | Named in `OC_INHERITANCE_FIT_2026-08-14.md` as the fit that produced the shipped `INHERITANCE_WEIGHTS`; imports `loso_fit_inheritance_weights` from `inheritance.py` | **Commit** |
| `scripts/diag_rb_carry_vacancy.py` | 133 | Named in `RB_CARRY_VACANCY_2026-08-14.md`; imports `contracts` and `roster_moves` | **Commit** |
| `scripts/eval_hierarchical_pass_mix.py` | 76 | Docstring: "Phase-4 spot check for hierarchical pass mix on the 2026 board"; calls `validate_mix_model` — the `beats_prior` gate that `HIERARCHICAL_PASS_MIX_2026-08-14.md` says must pass before shipping. *No file references it by name* — inferred keeper from its docstring and gate role. | **Commit** (low confidence — see §4) |

### 1.5 Decision records (untracked, repo root)

Confirmed by reading each: these document shipped constants and the evidence for them,
matching the existing `PHASE*_REPORT.md` / `FREEZE_*.md` convention already tracked.

| File | Documents | Action |
|---|---|---|
| `HIERARCHICAL_PASS_MIX_2026-08-14.md` | L2/L3 hierarchical pass distribution; records `USAGE_SHARE_BLEND_W` stays 0 and `FORMATION_ROLE_BLEND_W` behaviour — both live in `contracts.py` (`USAGE_SHARE_BLEND_W = 0.0` verified at `contracts.py:70`) | **Commit** |
| `HIERARCHICAL_RUSH_MIX_2026-08-14.md` | Rush mix; cites `python -m src.projection.team_rush_mix` | **Commit** |
| `OC_INHERITANCE_FIT_2026-08-14.md` | LOSO fit table; **shipped** `INHERITANCE_WEIGHTS` = 60/40, beating judgment 0.430 and team-only 0.437 at 0.419 MAE | **Commit** |
| `OL_TRAILING_2026-08-14.md` | `OL_TRAILING_SEASONS` (constant confirmed live at `ol_quality.py:36`, re-exported `predict.py:197`) | **Commit** |
| `RB_CARRY_VACANCY_2026-08-14.md` | RB carry vacancy; cites `scripts/diag_rb_carry_vacancy.py` | **Commit** |

### 1.6 Tracked-and-modified files

All 26 modified files are already-tracked source, tests, curated CSVs, and the draft
assistant UI. Three are regenerable outputs and are treated separately in §2.1. The
curated inputs `src/depth_chart/starters_2026.csv` and `status_overrides_2026.csv` are
hand-maintained source of truth (`src/depth_chart/README.md:42` — "curated base (never
auto-edited)"), not generated: **commit**.

---

## 2. Generated output

### 2.1 `output/` CSVs — currently tracked, and deliberately so

26 files under `output/` are tracked. Each is regenerable; the documented commands are in
`src/depth_chart/README.md:20-23`:

| Artifact | Regenerated by | Verified writer |
|---|---|---|
| `output/projections_2026.csv` | `python -m src.projection.predict --season 2026` | `predict.py:540,554` |
| `output/fantasy_points_2026.csv` | `python -m src.projection.fantasy_points --season 2026` | `fantasy_points.py:229-234` |
| `output/sleeper_comparison_2026.csv` | `python -m src.comparison.sleeper_compare` | `sleeper_compare.py:392-396` |
| `output/fantasy_evaluation_2025.*` | `python -m src.projection.fantasy_evaluation` | `fantasy_evaluation.py:716,728-729` |
| `output/depth_refresh_proposals_2026.csv` | `python -m src.depth_chart.refresh --season 2026` | `refresh.py:30,144-145` |
| `output/sleeper_snapshots/*.json` | side effect of `sleeper_compare` (content-addressed by SHA-256) | `sleeper_compare.py:63-94` |

**Measured churn (last 15 commits):**

| File | Changed lines |
|---|---:|
| `output/projections_2026.csv` | **94,268** |
| `output/sleeper_comparison_2026.csv` | 17,308 |
| `output/fantasy_points_2026.csv` | 15,978 |
| **Total** | **~127,554** |

These are *higher* than the ~64k/12k/11k in the brief — that estimate undercounted by
roughly a third. `.git` is 49 MB across 54 commits, dominated by these CSVs
(`projections_2026.csv` alone is 3.3 MB per revision).

**This is a decision, not a defect, and the evidence points to keeping them tracked.**

`FREEZE_2026-08-13.md` pins SHA-256 hashes for all six `output/` artifacts as the frozen
analytical baseline. `src/projection/fantasy_evaluation.py:78-82` states the intent in a
source comment: these belong in `output/` because `models/` is gitignored "(so a freeze
manifest could hash artifacts that were never committed)".

The reproducibility argument is stronger than it first looks:

- `models/` (36 joblib files, 2.3 MB) is **gitignored and untracked**.
- `data/` (1.9 GB, incl. a 1.79 GB `projections.db`) is **gitignored and untracked**.
- Therefore a fresh clone **cannot** regenerate `projections_2026.csv`. It has neither the
  trained models nor the source database. `FREEZE_2026-08-13.md` says so directly: the
  model binaries "must be retained with this local snapshot or regenerated through the
  documented pipeline."

So "regenerable" is true on *this machine* and false anywhere else. The tracked CSVs are
the only durable record of shipped boards. **Recommendation: keep tracking them.** The
churn is the price of that record, and 49 MB is not yet a problem. Revisit if `.git`
approaches a few hundred MB — the mitigation then is a squash/archive of historical board
revisions, not an ignore rule (which would strand the freeze manifest).

### 2.2 New sleeper snapshots (untracked)

| File | Classification | Evidence | Action |
|---|---|---|---|
| `output/sleeper_snapshots/players_nfl_367e9053bb043da7.{json,metadata.json}` | Generated audit record | Content-addressed cache written by `sleeper_compare.py:63-94`; SHA-256-keyed so it is append-only and never rewritten. 16 sibling snapshot files are already tracked. | **Commit** — for consistency with the tracked siblings and the freeze audit trail |
| `output/sleeper_snapshots/players_nfl_de312a7fc0a80242.{json,metadata.json}` | same | same | **Commit** |

These accumulate one immutable pair per distinct upstream Sleeper payload. Append-only,
so they add size but no churn. If size later matters, prune old snapshots deliberately
rather than ignoring the directory — the metadata is what makes `sleeper_comparison` auditable.

### 2.3 `output/depth_refresh_proposals_2026.csv` (untracked)

Referenced in `src/depth_chart/README.md:45` as "review queue" and written by
`refresh.py:145`. It is a transient human review queue, not a deliverable — regenerated
on every `refresh` run. **Recommend gitignore** (proposed, not applied — see §6).

### 2.4 Caches — all correctly ignored

Verified with `git check-ignore -v`:

| Path | Status |
|---|---|
| `.venv/` | ignored, `.gitignore:1` |
| `__pycache__/` | ignored, `.gitignore:2` (covers `scripts/__pycache__`, `tests/__pycache__`, root) |
| `models/` | ignored, `.gitignore:9` — **0 joblib files tracked anywhere** (`git ls-files \| grep joblib` empty) |
| `data/` | ignored, `.gitignore:4` |
| `.pytest_cache/` | was ignored **only** by its own self-generated `.pytest_cache/.gitignore`, not by the repo's. Now explicit — see §6. |

**`models/*.joblib` implication.** Not tracking 2.3 MB of binaries is the right call —
they are opaque to diff, would balloon history, and are pinned by manifest hash in the
freeze doc. The cost is that model provenance rests entirely on one un-backed-up local
directory. See §5.1.

---

## 3. Scratch / diagnostic — recommend discarding (NOT deleted)

### 3.1 Untracked scratch

| File | Evidence | Recommendation |
|---|---|---|
| `output/_diag_pierce.py` | 4.3 KB ad-hoc script. Reads `output/projections_2026.csv`, prints Alec Pierce driver columns and the IND WR room. Underscore-prefixed, sitting in the output directory rather than `scripts/`. **Nothing references it.** | Discard (user's call) |
| `output/_diag_pierce2.py` | Same pattern; queries `data/projections.db` directly for Pierce/IND target history. Only reference to `_before_formation_role_snapshot` is inside this file. | Discard (user's call) |
| `output/_before_formation_role_snapshot.csv` | 1.3 KB. Referenced solely by `_diag_pierce2.py`. A manual before/after capture for the `FORMATION_ROLE_BLEND_W` work — and that finding is already written up in `HIERARCHICAL_PASS_MIX_2026-08-14.md`, so the durable record survives. | Discard (user's call) |

All three are one-off investigation aids for a WR-room question whose conclusion is
already committed to a decision record. Nothing imports them.

### 3.2 Deleted-but-unstaged — intentional cleanup, safe to stage

For each, I grepped the full repo for the filename and module name. Results:

| File | Referenced by anything? | Verdict | Action |
|---|---|---|---|
| `phase0_validation.py` | **No references.** Added in `7ec6161` "Phase 0: validate nflverse data coverage for OL attribution model". Phase 0 concluded; `PHASE0_REPORT.md` (tracked) retains the findings. | Intentional cleanup | Stage the deletion |
| `scratch_stability_diag.py` | Referenced once — `PHASE2_STABILITY_INVESTIGATION.md:7` reads: "`scratch_stability_diag.py` (deleted; findings below are retained)." **The tracked doc explicitly records the deletion as deliberate.** Named `scratch_*` by the author. | Intentional cleanup, self-documented | Stage the deletion |
| `scratch_stability_output.txt` | **No references.** Captured stdout from the above. Its content is preserved in `PHASE2_STABILITY_INVESTIGATION.md`. | Intentional cleanup | Stage the deletion |
| `scripts/apply_wr_usage_priors.py` | **No references.** Its own docstring: *"One-shot: set researched WR usage_share_prior + usage_share_reviewed on starters_2026.csv."* The effect is already baked into `src/depth_chart/starters_2026.csv` (modified in this tree), and `contracts.py:70` confirms `USAGE_SHARE_BLEND_W = 0.0`, matching the "Usage priors" section of `HIERARCHICAL_PASS_MIX_2026-08-14.md`. | Intentional — a spent one-shot migration | Stage the deletion |
| `smoke_test_ingest.py` | **No references.** Added in `59bdff7` "Phase 1: data layer". Superseded by the 15-file `tests/` suite (`FREEZE` cites 57 passing tests). | Intentional cleanup | Stage the deletion |

No accidental losses found. Every deletion is either self-documented, spent, or superseded,
and none is imported by surviving code.

> One caveat, stated plainly: `PHASE2_STABILITY_INVESTIGATION.md` is itself *modified* in
> this tree, and the line documenting the `scratch_stability_diag.py` deletion is part of
> that uncommitted change. The doc and the deletion belong in the same commit.

---

## 4. Ambiguous — needs your decision

| File | Why it is ambiguous | Question for you |
|---|---|---|
| `pull_alignment.py` (403 LOC, root) | **Not orphaned.** `test_offline.py:22` does `import pull_alignment as pa`. But nothing in `src/`, `scripts/`, or `tests/` imports it, and no `output/` artifact traces to it. It is a self-contained RotoWire receiver-alignment scraper (`discover` / `pull` / `build` CLI) hitting an undocumented public JSON endpoint with a spoofed desktop UA. It looks like an in-progress data-source spike that has not been wired into the pipeline. | Is the RotoWire alignment feed a live data source you intend to integrate, or an abandoned spike? If live → commit to `scripts/`. If abandoned → discard both it and `test_offline.py`. Separately: are you comfortable committing a scraper that sends a browser UA specifically to sidestep a robots.txt UA block? That is documented in its own docstring (lines 14-15) and is a policy call, not a hygiene one. |
| `test_offline.py` (281 LOC, root) | **Does not duplicate anything under `tests/`** — no `tests/` file imports `pull_alignment`, and there is no `tests/test_offline.py`. It is a real test, but for `pull_alignment` only, and it is misplaced: it sits at repo root, so `pytest tests/` never collects it. Its own docstring flags a provenance problem: the fixtures "were not present in the repo", the DJ Moore fixture is "reconstructed", others "synthetic" — so it proves the parser behaves, not that fixtures match real responses. | Same decision as `pull_alignment.py` — they live or die together. If kept, move it to `tests/test_pull_alignment_offline.py` so the suite actually runs it. Keeping it at root means it silently never runs. |
| `scripts/eval_hierarchical_pass_mix.py` | Nothing references it by name, unlike its two `scripts/` siblings which are each cited in a decision record. But it runs `validate_mix_model` — the `beats_prior` gate `HIERARCHICAL_PASS_MIX_2026-08-14.md` requires before shipping the scheme mix. It also writes an untracked side artifact, `output/projections_2026_hierarchical.csv`. | Is this the harness you re-run to re-check the `beats_prior` gate? If yes → commit (my recommendation). If it was a one-time Phase-4 check → discard. |
| `phase0_results.json` (root) | **Tracked, but also listed in `.gitignore:7`.** The ignore rule has no effect on an already-tracked file, so the two are in contradiction — the only such file in the repo (`git ls-files -i -c` returns exactly this one). | Did you intend to untrack this? Untracking is a deletion from the index, so I did not do it. Either `git rm --cached phase0_results.json`, or drop line 7 from `.gitignore` to stop implying it is ignored. |
| `output/depth_refresh_proposals_2026.csv` | A review queue (§2.3), regenerated each `refresh` run — but its 16 sibling `output/` artifacts are all tracked, and you may want the queue in the freeze record. | Ignore it as transient, or track it alongside the other `output/` artifacts? |

---

## 5. Additional findings not in the original scope

### 5.1 Single point of failure on `models/`

`models/` is gitignored (correctly) and `data/projections.db` is 1.79 GB and gitignored
(correctly). Neither is in version control. `FREEZE_2026-08-13.md` pins the models by
manifest hash `aa76d9fc…` and notes they "must be retained with this local snapshot."

If this machine's `models/` directory is lost, the frozen 2026 board cannot be reproduced
bit-for-bit — retraining would produce different binaries and break the freeze hash. This
is not a git problem, but it is the largest reproducibility risk in the repo. Consider an
out-of-band backup of `models/` (2.3 MB — trivially small) keyed to the freeze commit.

### 5.2 The `data/` ignore pattern is unanchored

`.gitignore:4` is `data/`, with no leading slash, so git matches a directory named `data`
at **any depth**. Verified:

```
$ git check-ignore -v draft_assistant/data/players_2026.json
.gitignore:4:data/      draft_assistant/data/players_2026.json
```

`draft_assistant/data/players_2026.json` and `draft_assistant/data/team_stats_2026.json`
exist on disk and are silently ignored — they never appeared in your `git status`, which
is why they were not in the audit brief. They are genuinely regenerable
(`src/draft_assistant/prepare.py:194-195`, `src/team_stats/prepare.py:264-265`), and
`draft_assistant/README.md:30-31` documents the commands, so the *outcome* is fine. But it
happened by accident, not by intent, and a future `foo/data/` will vanish the same way.

Anchoring it to `/data/` would express the actual intent. I did **not** apply this — it
alters an existing entry, which is outside my remit, and it would newly surface those two
JSON files as untracked. Your call (§7).

### 5.3 Untracked modules under tracked, modified parents

`src/coordinator/oc_profiles.py` and `src/projection/ol_quality.py` are tracked, modified,
and import untracked modules (`inheritance.py`, `contracts.py`). A partial commit that
takes the tracked modifications but not the untracked modules yields a repo where
`import src.projection.predict` fails outright. The staging block in §6 keeps them together.

### 5.4 `src/projection/composition.py` appeared *during* this audit

Not present in the working tree when the audit started; it exists as of 11:41 today,
12.8 KB, untracked. Its docstring describes an in-progress refactor unifying the two
composition paths — `predict.project_season` (16 stages, ships) and
`fantasy_evaluation._compose_and_reconcile` (7 stages, the only one scored against real
outcomes) — which it says had drifted, with new allocation layers landing on the side that
is never measured.

**Nothing imports it yet** (grepped: no `from src.projection.composition` / `import
composition` anywhere). So it is live, uncommitted, mid-refactor work, not scratch and not
yet wired in.

It is **deliberately excluded from the staging block in §7** — you are apparently still
writing it, and staging a half-finished module alongside a freeze-relevant commit is the
wrong call. Add it yourself once it is wired up and the suite passes. Flagging it because
it did not exist when this audit was scoped, and it is exactly the kind of file that gets
lost in a 34-file untracked list.

### 5.5 `src/depth_chart/live_depth_2026.csv` is absent

`src/depth_chart/README.md:43` describes it as the derived chart produced by
`refresh --apply`. It is neither tracked nor present on disk. Expected if `--apply` has not
been run since a clean-up; noted only so it is not mistaken for a loss.

---

## 6. `.gitignore`

### Applied (unambiguous, additive only)

Two entries appended. Existing lines 1-9 untouched.

```gitignore
# Test/tooling caches. .pytest_cache was previously ignored only by the
# self-generated .pytest_cache/.gitignore; make it explicit here.
.pytest_cache/
*.pyo
```

`.pytest_cache/` was already effectively ignored via the cache directory's own
self-generated `.gitignore`, so this changes no file's status today — it just stops the
repo depending on pytest to regenerate that file. Zero risk of masking anything.

### Proposed — NOT applied, needs your approval

Each of these could mask something you want tracked, so they are quoted rather than written:

```gitignore
# Ad-hoc diagnostics parked in output/ (underscore-prefixed by convention)
output/_*

# Transient depth-chart review queue, regenerated by
#   python -m src.depth_chart.refresh --season 2026
output/depth_refresh_proposals_2026.csv

# Side artifact of scripts/eval_hierarchical_pass_mix.py
output/projections_2026_hierarchical.csv
```

Why each is held back:

- `output/_*` would ignore any future underscore-prefixed output. It happens to catch
  exactly the three scratch files in §3.1 today, but it is a wildcard over your deliverable
  directory — I would rather you agree to the convention than have me impose it.
- `depth_refresh_proposals_*.csv` is ambiguous per §4 (its 16 siblings are all tracked).
- The `_hierarchical.csv` rule is moot if you discard `eval_hierarchical_pass_mix.py`.

And one change I deliberately did **not** make, because it edits an existing line:

```gitignore
# line 4: `data/` → `/data/`  (anchor to repo root; see §5.2)
```

---

## 7. Staging commands

Copy-pasteable. **Not run.** This stages exactly the production code, tests, docs,
harnesses, and decision records — and the five intentional deletions — while leaving every
scratch file and every ambiguous file untouched in your working tree.

```bash
cd /c/Users/rdfer/Projects/fantasy-projections

# --- 1. New production modules (untracked) ---
git add src/projection/contracts.py \
        src/projection/artifacts.py \
        src/projection/depth_rates.py \
        src/projection/depth_gating.py \
        src/projection/roster_moves.py \
        src/projection/replacement.py \
        src/projection/veterans.py \
        src/projection/team_reconcile.py \
        src/projection/team_pass_mix.py \
        src/projection/team_rush_mix.py \
        src/coordinator/inheritance.py

# --- 2. New tests + pipeline doc ---
git add tests/test_team_pass_mix.py \
        tests/test_team_rush_mix.py \
        src/projection/README.md

# --- 3. Fit / eval harnesses cited by the decision records ---
#     (drop eval_hierarchical_pass_mix.py if you decide it was one-time — §4)
git add scripts/fit_oc_inheritance_weights.py \
        scripts/diag_rb_carry_vacancy.py \
        scripts/eval_hierarchical_pass_mix.py

# --- 4. Decision records (shipped tuning values) ---
git add HIERARCHICAL_PASS_MIX_2026-08-14.md \
        HIERARCHICAL_RUSH_MIX_2026-08-14.md \
        OC_INHERITANCE_FIT_2026-08-14.md \
        OL_TRAILING_2026-08-14.md \
        RB_CARRY_VACANCY_2026-08-14.md

# --- 5. Tracked source / test / curated-CSV modifications ---
#     Required in the SAME commit: oc_profiles.py and ol_quality.py import
#     the untracked modules staged in step 1 (§5.3).
git add PHASE2_STABILITY_INVESTIGATION.md \
        draft_assistant/README.md \
        draft_assistant/css/styles.css \
        draft_assistant/index.html \
        draft_assistant/js/app.js \
        src/coordinator/oc_profiles.py \
        src/coordinator/tendencies.py \
        src/depth_chart/live.py \
        src/depth_chart/starters_2026.csv \
        src/depth_chart/status_overrides_2026.csv \
        src/draft_assistant/prepare.py \
        src/draft_assistant/tiers.py \
        src/draft_assistant/vorp.py \
        src/projection/data_prep.py \
        src/projection/features.py \
        src/projection/ol_quality.py \
        src/projection/predict.py \
        src/projection/rookies.py \
        tests/test_data_prep_appearances.py \
        tests/test_depth_refresh.py \
        tests/test_draft_tiers.py \
        tests/test_draft_vorp.py \
        tests/test_runtime_output_correctness.py

# --- 6. Regenerated board outputs (keep tracked — §2.1) ---
git add output/projections_2026.csv \
        output/fantasy_points_2026.csv \
        output/sleeper_comparison_2026.csv \
        output/sleeper_snapshots/players_nfl_367e9053bb043da7.json \
        output/sleeper_snapshots/players_nfl_367e9053bb043da7.metadata.json \
        output/sleeper_snapshots/players_nfl_de312a7fc0a80242.json \
        output/sleeper_snapshots/players_nfl_de312a7fc0a80242.metadata.json

# --- 7. Intentional deletions (§3.2) ---
git rm --cached phase0_validation.py \
                scratch_stability_diag.py \
                scratch_stability_output.txt \
                scripts/apply_wr_usage_priors.py \
                smoke_test_ingest.py

# --- 8. The .gitignore addition from §6 ---
git add .gitignore

# --- Review before committing ---
git status
git diff --cached --stat
```

Deliberately **not** staged: `output/_diag_pierce.py`, `output/_diag_pierce2.py`,
`output/_before_formation_role_snapshot.csv` (scratch, §3.1); `pull_alignment.py`,
`test_offline.py`, `output/depth_refresh_proposals_2026.csv` (ambiguous, §4); and
`src/projection/composition.py` (in-progress refactor written during this audit, nothing
imports it yet — §5.4).

> Step 7 uses `git rm --cached`, which stages the deletion **without touching your disk**.
> The files are already gone from the working tree; this only records that in the index.
> Nothing is destroyed.

> Consider splitting this into two commits — steps 1-5 (code + docs) and step 6 (regenerated
> board) — so the 127k-line CSV diff does not bury the source review.

---

## 8. Needs your decision

1. **`pull_alignment.py` + `test_offline.py`** — live data source or abandoned spike? They
   live or die together (`test_offline.py:22` imports the scraper). If kept, move the test
   under `tests/` or it will never be collected. Separate question: are you comfortable
   committing a scraper that spoofs a browser UA to get around a robots.txt UA block?
2. **`scripts/eval_hierarchical_pass_mix.py`** — recurring `beats_prior` gate harness
   (commit) or one-time Phase-4 check (discard)?
3. **`phase0_results.json`** — tracked *and* gitignored, a contradiction. Untrack it with
   `git rm --cached`, or drop `.gitignore:7`?
4. **`output/depth_refresh_proposals_2026.csv`** — ignore as a transient review queue, or
   track it with its 16 `output/` siblings?
5. **`output/_*` ignore rule** — adopt underscore-prefix as the scratch convention for
   `output/`, or keep reviewing those files by hand?
6. **Anchor `data/` → `/data/`** (§5.2)? This would newly surface
   `draft_assistant/data/*.json` as untracked. Yes/no.
7. **Delete the three scratch files in §3.1?** I have not touched them. Confirm and they
   can go.
8. **Back up `models/` out of band** (§5.1)? 2.3 MB, currently the single point of failure
   for reproducing the frozen board.

---

## Verified vs inferred

**Verified by command:**

- Import graph for all 11 untracked `src/` modules — grepped repo-wide for
  `import X` / `from …X` (excluding `.venv`, `.git`, `__pycache__`); file:line cited in §1.
- Zero references to the 5 deleted files, except the one self-documenting mention at
  `PHASE2_STABILITY_INVESTIGATION.md:7`.
- Zero references to the three `output/_*` scratch files outside `_diag_pierce2.py` itself.
- `test_offline.py:22` imports `pull_alignment`; no `tests/` file imports it; no
  `tests/test_offline.py` exists (so: not a duplicate).
- Tracked-file inventory via `git ls-files`: 26 files under `output/`, **0** under
  `models/`, **0** joblib anywhere, **0** `__pycache__`/`.pytest_cache`/`.venv`.
- Ignore status of 9 paths via `git check-ignore -v`, including the `draft_assistant/data/`
  over-match and the `.pytest_cache` self-ignore.
- Churn via `git log -15 --numstat` per file; `.git` size 49 MB; 54 commits.
- `git ls-files -i -c` → exactly one tracked-but-ignored file (`phase0_results.json`).
- Regeneration commands traced to their writing `to_csv` call sites (§2.1 table).
- Read in full: all 5 decision records, `FREEZE_2026-08-13.md`, `src/projection/README.md`,
  both `_diag_pierce` scripts, headers of `pull_alignment.py` and `test_offline.py`.

**Inferred, not verified:**

- That `scripts/eval_hierarchical_pass_mix.py` is a keeper — based on its docstring and its
  use of the `beats_prior` gate, with no naming reference to confirm it. Flagged in §4.
- That the two new sleeper snapshots should be committed — from consistency with 16 tracked
  siblings, not from an explicit policy statement.
- That the 26 modified tracked files are all wanted changes. **I did not read their diffs**
  — that is a code review, not a hygiene audit. The staging block assumes you want them.
- **I did not run the test suite.** The claim that the untracked modules import cleanly
  rests on static grep, not execution. Run `python -m pytest tests/` before committing —
  the `FREEZE` gate is 57 passing tests, and the two new test files should push that higher.
