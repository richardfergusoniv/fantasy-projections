# State of the build — 2026-08-15

> **Update (2026-08-27 — accuracy-first 2026 board):** A leakage-safe
> top-120 ADP bake-off promotes a separate accuracy-first point board: 2025
> holdout MAE 58.72 → 53.14 and Spearman .504 → .602. The gain comes from
> ADP-assisted RB/WR weights; v3 receives zero selected point weight and stays
> the calibrated distribution overlay. Native v1 output remains unchanged.
> See `docs/decisions/ACCURACY_FIRST_ENSEMBLE_2026-08-27.md`.

> **Update (2026-08-25 — v3 point-engine decision):** v3 does **not** replace
> the LightGBM/`compose_board` point engine yet. Hardened
> `scripts/v3_promotion_gate.py` splits `simulation_ready` (percentile UI)
> from `promote_v3_means` (requires `output/model_v3/means_backtest.json`
> generative win vs v1 and blend). Draft `--v3-means` cutover exists but
> defaults off. See `docs/decisions/V3_PROBABILISTIC_PIPELINE.md`.

> **Update (2026-08-25 — v3 probabilistic pipeline):** A parallel v3 path adds
> rolling backtest persistence, learned reconcile weights, conditional interval
> models, Monte Carlo simulation, compositional share models, and generative
> conversion layers under `src/projection/{evaluation,models,inference,data}/`.
> Outputs land in `output/model_v3/` and `output/backtest/`. v1 ship path and
> v1/v2 draft ensemble remain production defaults until
> `scripts/v3_promotion_gate.py` passes. See
> `docs/decisions/V3_PROBABILISTIC_PIPELINE.md`.

> **Update (2026-08-24 — diagnostic player sentiment):** The 32 local
> `perplexity research/` team summaries and the frozen ECR/ADP snapshot now
> produce an audited, position-relative player sentiment score for every
> projected QB/RB/WR/TE. The fields ship in projection/fantasy CSVs and both
> dashboards; missing evidence stays null. `models/sentiment_manifest.json`
> keeps every position inactive because only one point-in-time season exists,
> so sentiment currently changes no projection, rank, tier, or VORP. See
> `src/sentiment/README.md`.

> **Update (2026-08-24 — two models detangled):** This repo is the **v1
> rate-forecast** pipeline. The sibling folder `../fantasy-projections-2` is a
> separate **v2 team-first** model. Canonical `output/fantasy_points_*.csv` and
> `output/projections_*.csv` here are v1 only. Optional archived v2 syncs land
> under `output/model_v2/` via `src.draft_assistant.from_v2` and must not
> overwrite the native board. Head-to-head 2025 holdout numbers:
> `output/model_accuracy_compare_2025.json` (`scripts/compare_model_accuracy.py`).
> Draft UI: this repo port **8766** (v1); v2 repo port **8765**.
> Season pass/catch identities (recv yds = pass yds, etc.) are restored on
> shipped season totals by `reconcile_team_season_identities` in
> `compose_board` — rates untouched; no v2 re-merge.
>
> **Update (2026-08-24 — draft ensemble shipped):** Test-before-rewrite go/no-go
> was `do_not_rewrite`. Draft `prepare` defaults to a v1/v2 season-points blend
> (`src/draft_assistant/ensemble_weights.json` + `output/model_v2/`) when both
> exist; `--no-ensemble` for pure v1. Does not change `compose_board` or
> LightGBM. Decision: `docs/decisions/TEST_BEFORE_REWRITE_2026-08-24.md`.

> **Update (volume composition retired):** `compose_board` no longer runs
> hierarchical pass/rush, usage-share priors, QB volume-game reconcile, or
> team volume normalizers. Shipped `pred_pg` is forecast rates after Gate A/B
> plus availability hygiene. Sections below that list deleted stages are
> historical until rewritten.

The single current-state document. If you are new, or returning after three
weeks, read this first and treat every other root markdown file as either a
decision record or history — see [`DOC_INVENTORY.md`](DOC_INVENTORY.md) for
which is which.


**Scope of this document.** It describes what the pipeline *is* and what is
*known about it*. It does not relitigate tuning decisions (those live in the
dated decision records) and it does not duplicate the provenance forensics
(those live in [`PROVENANCE_AUDIT.md`](PROVENANCE_AUDIT.md)).

**Verification legend.** Every claim below is tagged:

- **[code]** — read directly out of the working-tree source or a committed
  artifact during this pass.
- **[git]** — read out of `git log` / `git ls-files` / `git status`.
- **[doc]** — taken from an existing repo document, not independently re-derived.
  Numbers tagged `[doc]` were correct when produced; several are superseded, and
  §3.3 says which.

No pipeline stage and no test was run to produce this document.

---

## 1. What the repo produces

| Artifact | What it is |
|---|---|
| `output/projections_2026.csv` | One row per (player, position, stat): per-game rate, 80% interval, season total, team anchors, every allocation scale factor, provenance flags. **[code]** `OUTPUT_COLUMNS` in `src/projection/contracts.py` |
| `output/fantasy_points_2026.csv` | Half-PPR, 4-point passing TD, per-game and season. **[code]** `SCORING` in `src/projection/fantasy_points.py` |
| `output/sleeper_comparison_2026.csv` | Read-only agreement diagnostic against Sleeper's consensus. **Not an optimization target** — see §5. |
| `output/fantasy_evaluation_2025.{csv,json}` + `_summary_` | The leakage-safe 2024→2025 holdout scoreboard. The only harness scored against real fantasy outcomes. |
| `draft_assistant/` | Static local app: VORP draft board + team projections, fed by the two CSVs above. **[code]** `draft_assistant/README.md` |

There is **no root `README.md`** **[code]**. This document is the closest thing
until one exists.

---

## 2. Architecture as it actually is

```
nflverse / Sleeper / FTN / PFR
        │  src/ingest/sources.py  →  src/cache.py  (per-season parquet)
        ▼
data/projections.db (or configured external data dir)  src/db/load.py
        │
        ├── src/ol_model/pooled_pipeline.py   → pooled OL attribution coefficients
        ├── src/coordinator/tendencies.py     → OC tendency profiles
        │   src/coordinator/inheritance.py    → first-year OC inheritance weights
        ▼
FEATURES        src/projection/data_prep.py, features.py, ol_quality.py
        ▼
MODEL LAYER     src/projection/train.py  →  models/*.joblib
                LightGBM per-game rate/share models, Ridge team anchors,
                availability model; src/projection/backtest.py writes
                models/interval_residuals.csv
        ▼
FORECAST STAGE  veterans.py (+ depth_rates, depth_history, corrections)
                rookies.py  (rule path, always low_confidence)
                replacement.py (curated players neither path reaches)
                roster_moves.py (team reassignment, vacancy boosts)
        ▼
COMPOSITION / ALLOCATION LAYER      src/projection/composition.py
                compose_board(): 15 stages, one implementation, two callers
        ▼
OUTPUTS         predict.py → fantasy_points.py → sleeper_compare.py
                draft_assistant/prepare.py + team_stats/prepare.py → JSON
                draft_assistant/serve.py → local board
```

### 2.1 The composition layer, and why it matters

**[code]** `src/projection/composition.py` is the newest structural change in
the repo and the thing most likely to surprise someone working from older docs.
There used to be two implementations of "turn per-player model output into a
reconciled board": `predict.project_season` (which ships) and
`fantasy_evaluation._compose_and_reconcile` (which was the only thing scored).
They drifted, and every new allocation layer was added on the side that is never
measured. `composition.py` collapses them into one.

`compose_board(rows, ctx)` runs these **15 stages in this order** **[code]**:

| # | Stage | Module |
|---|---|---|
| 1 | `apply_deep_bench_games_cap` | `depth_gating.py` |
| 2 | `apply_status_overrides` | `depth_gating.py` |
| 3 | `propagate_team_anchors` | `team_reconcile.py` |
| 4 | `reconcile_qb_projected_volume_games` | `team_reconcile.py` |
| 5 | `apply_usage_share_prior` | `team_reconcile.py` |
| 6 | `attach_team_pass_mix` (L2) | `team_pass_mix.py` |
| 7 | `apply_hierarchical_pass_distribution` (L3) | `team_pass_mix.py` |
| 8 | `attach_team_rush_mix` (L2) | `team_rush_mix.py` |
| 9 | `apply_hierarchical_rush_distribution` (L3) | `team_rush_mix.py` |
| 10 | `normalize_team_passing_volume` | `team_reconcile.py` |
| 11 | `normalize_team_rushing_volume` | `team_reconcile.py` |
| 12 | `reconcile_stat_constraints` | `team_reconcile.py` |
| 13 | `reconcile_team_pass_receive_counts` | `team_reconcile.py` |
| 14 | `add_team_pass_catch_coherence_flag` | `team_reconcile.py` |
| 15 | `add_projected_season_totals` | `team_reconcile.py` |

The order is load-bearing and `compose_board`'s docstring is its single source
of truth. `tests/test_composition_unification.py` pins the order and asserts
that neither caller grows a private copy **[code]**.

The two callers differ only in **artifact provenance**, carried by a
`CompositionContext` **[code]**:

| | `shipped_context()` | `leakage_safe_context()` |
|---|---|---|
| Called by | `predict.project_season` | `fantasy_evaluation._forecast_from_history` |
| Rate/anchor models | loaded from `models/`, trained on every season | refit on `season <= source_season` |
| L2 pass/rush mix | fit on all history | `history_seasons = 2016..source_season` |
| Usage-share priors | fit on all history | fit on `<= source_season` |
| Curated depth chart / status overrides | 2026 research files | same loader → **empty** for any other season |
| Elite correction, interval residuals | applied | **omitted** (would leak) |

**Honest-coverage rule** **[code]**: a stage whose input does not exist must
no-op *visibly*. `CompositionContext.describe_coverage()` emits a per-stage
`active` / `active, degraded` / `no-op` map computed from the artifacts actually
present, and the evaluation harness copies it into
`metadata.composition_stage_coverage` so a coverage number can never be misread
as a performance number.

---

## 3. What is validated, and what is not

### 3.1 The model layer is validated and beats baselines — this result is real

State it plainly, because the rest of this section is caveats and the caveats do
not touch this:

- **Rate models beat naive carry-forward on 18 of 20 position/stat cells** on a
  2024→2025 holdout, the exceptions being RB targets and RB receptions **[doc]**
  (`PHASE4_REPORT.md` §"Backtest"; superseded in level by later retrains, not in
  direction).
- **On the leakage-safe 2024→2025 fantasy holdout the model beats both baselines
  (carry-forward and availability-adjusted) on Spearman rank correlation and
  season-point MAE at all four positions** **[doc]**
  (`FANTASY_EVALUATION_2025_REPORT.md`, `FREEZE_2026-08-13.md`). Spearman margins
  of roughly .08–.15.
- The Gate A availability model is genuinely outcome-validated (held-out
  games-played MAE, 8/8 folds at every position) **[doc]** (`PROVENANCE_AUDIT.md`
  §2).

Equally plainly, the same holdout says the model **does not** win everywhere,
and the losses are on the draft-relevant columns: **QB top-12 tier hits 6/12
against carry-forward's 7/12; WR tied with both baselines; TE VORP MAE worse
than both** **[doc]** (`FREEZE_2026-08-13.md`). The defensible claim is that the
model *orders a whole position board* better than the baselines, not that it
identifies the top tier better.

### 3.2 The harness that produces that result was just rewritten

**[code]** The 2024→2025 harness now runs the shipped `compose_board` stage
sequence over leakage-safe artifacts. Relative to the state
`FANTASY_EVALUATION_2025_REPORT.md` describes, this **widens** what is measured:
the hierarchical pass mix (L2+L3), the hierarchical rush mix (L2+L3),
`propagate_team_anchors`, the deep-bench games cap and status-override stages
all now execute inside the scored path, where previously the harness ran a
private 7-stage reconciler.

What still cannot be measured on a historical fold, each degrading to a recorded
pass-through rather than being faked **[code]**
(`fantasy_evaluation.run_evaluation.coverage_limits`):

- **Curated depth-chart research** — `src/depth_chart/starters_<season>.csv` is
  hand-built and exists for 2026 only. So curated membership, roles, the
  LWR/RWR/SWR formation split, reviewed usage priors, and replacement-level rows
  for curated players all no-op on the 2025 fold.
- **Dated status overrides** (IR/PUP) — likewise 2026-only.
- **Elite residual correction** — `models/corrections.joblib` is fit on residuals
  spanning the target season; passed as `corrections=None`.
- **Prediction intervals** — `models/interval_residuals.csv` is fit by
  `backtest.py` across the target season; an empty residual table is passed, so
  `pred_pg_low/high` collapse onto the point estimate. Nothing scored reads them.
- **Roster reassignment** (`reassign_team_changers`) and the **incumbent /
  team-changer vacancy boosts** — `fantasy_evaluation.py` does not import
  `roster_moves.py` or `replacement.py` at all **[code]**. The target team comes
  from the frozen Week-1 roster, a stricter preseason source.

### 3.3 The published evaluation tables are stale — including the freeze

**[code]** Three separate checks, all failing:

1. `output/fantasy_evaluation_summary_2025.json` on disk is dated **2026-08-14
   12:39**, older than `composition.py` (2026-08-15 11:41) and
   `fantasy_evaluation.py` (11:53). **The harness has not been re-run since the
   rewrite.**
2. That JSON contains no `composition_pipeline`, `composition_artifact_provenance`,
   or `composition_stage_coverage` key — keys the current code emits
   unconditionally. Confirms (1) independently.
3. The model / `all_eligible` metrics on disk match **no published table**:

   | Pos | On disk **[code]** | `FREEZE` / `FANTASY_EVALUATION_2025_REPORT` **[doc]** | Last published delta **[doc]** |
   |---|---|---|---|
   | QB | ρ 0.7824, MAE 41.78, tier 6/12, VORP 46.29 | ρ 0.792, 41.72, 6/12, 47.98 | `AGE_…`: 0.7822 / 42.39 / 46.47 |
   | RB | ρ 0.7294, MAE 32.88, tier **16/24**, VORP 33.20 | ρ 0.729, 33.27, **15/24**, 35.74 | `AGE_…`: 0.7324 / 33.55 / **36.59** |
   | WR | ρ 0.7726, MAE 22.72, tier 22/36, VORP 24.59 | ρ 0.769, 23.11, 22/36, 25.72 | `AGE_…`: 0.7726 / 22.721 / 24.588 |
   | TE | ρ 0.8324, MAE 16.57, tier 5/12, VORP 35.17 | ρ 0.837, 16.55, 5/12, 39.28 | `AGE_…`: 0.8324 / 16.568 / 35.166 |

   RB gained a tier hit (15→16) and its VORP MAE moved 36.59→33.20 after
   `AGE_EFFECT_SHRINKAGE`'s table was written — that is the depth-chart
   allocation work (`fc0d88f`, which regenerated the evaluation CSVs **[git]**),
   never published in any document.

**`FREEZE_2026-08-13.md` is void as a manifest.** **[code]** All six SHA-256
hashes it pins fail against the current tree — zero of six match. Its stated base
commit `df37452` predates the entire allocation layer **[git]**.

**Action item:** re-run `python -m src.projection.fantasy_evaluation` and publish
a fresh table with the stage-coverage map, then re-freeze. Until then, quote the
on-disk JSON, not the reports.

### 3.4 The allocation layer's provenance problem — summary only

**[doc]** Full forensics in [`PROVENANCE_AUDIT.md`](PROVENANCE_AUDIT.md); do not
duplicate it here. The short version:

- Of 43 constants in `src/projection/contracts.py`, 30 are real tuning knobs.
  **4 were set against real held-out outcomes; 8 against a proxy metric (share
  MAE, coverage fraction, OL-score persistence); 1 against Sleeper; and 17 have
  no measurement of any kind.**
- Counting *decisions* rather than constants, Sleeper agreement is the deciding
  evidence in roughly half the allocation layer.
- Several unmeasured knobs bind on the shipped board.

One important tension to be aware of, surfaced rather than resolved:
`PROVENANCE_AUDIT.md` §0.2 states that `fantasy_evaluation.py` "does **not**
import `team_pass_mix`, `team_rush_mix`, `roster_moves`, `replacement`,
`depth_gating`, or `corrections`", and concludes the highest-leverage layers
cannot be scored by any harness. **[code]** As of the working tree, the harness
imports `compose_board`, which imports `team_pass_mix`, `team_rush_mix`,
`depth_gating` and `team_reconcile` — so the pass- and rush-mix layers *are* now
inside the scored path, while `roster_moves`, `replacement` and `corrections`
still are not. The audit's structural conclusion holds for the latter three; its
specific import claim was overtaken by `composition.py`, which
`REPO_HYGIENE_AUDIT.md` §5.4 notes "appeared *during* this audit" **[doc]**. The
audit is another agent's live deliverable and is not edited here.

### 3.5 Nothing in the allocation layer is committed

**[git]** `git ls-files src/projection/` returns 13 files. `composition.py`,
`contracts.py`, `team_reconcile.py`, `team_pass_mix.py`, `team_rush_mix.py`,
`roster_moves.py`, `replacement.py`, `depth_gating.py`, `depth_rates.py`,
`veterans.py`, `artifacts.py`, `src/coordinator/inheritance.py`, and four test
files are all untracked. Five of the eight decision records and both audits are
untracked too. The build currently exists only in one working tree.
Staging commands are in `REPO_HYGIENE_AUDIT.md` §7.

---

## 4. How to run everything

All commands from the repo root, with the project venv active.

The database and raw parquet cache default to `data/`. Set
`FANTASY_PROJECTIONS_DATA_DIR` to keep both on another drive; this workstation
uses `D:\fantasy-projections-data`. `FANTASY_PROJECTIONS_DB_PATH` and
`FANTASY_PROJECTIONS_RAW_DIR` can override the two locations independently.

```bash
# 0. Data layer (slow; only when refreshing sources)
python -m src.db.load                       # nflverse → data/projections.db
python -m src.ol_model.pooled_pipeline      # pooled OL attribution coefficients
python -m src.coordinator.tendencies        # OC tendency profiles
python -m src.coordinator.oc_profiles       # + inheritance for first-year seats

# 1. Train the model layer  → models/*.joblib
python -m src.projection.train

# 2. Backtest               → models/interval_residuals.csv  (REQUIRED by predict)
python -m src.projection.backtest

# 3. Project                → output/projections_2026.csv
python -m src.projection.predict --season 2026
#    optional: --as-of YYYY-MM-DD   (dated status overrides + nflverse depth snapshot)
#    optional: --out PATH

# 4. Score                  → output/fantasy_points_2026.csv
python -m src.projection.fantasy_points --season 2026

# 5. Consensus diagnostic   → output/sleeper_comparison_2026.csv   (READ-ONLY, §5)
python -m src.comparison.sleeper_compare --season 2026
python -m src.comparison.spot_check --season 2026

# 6. The scoreboard         → output/fantasy_evaluation_*_2025.*
python -m src.projection.fantasy_evaluation
#    optional: --source-season / --target-season / --tier-ranks / --replacement-ranks

# 7. Draft assistant
python -m src.draft_assistant.prepare --season 2026
python -m src.team_stats.prepare --season 2026
python -m src.draft_assistant.serve --open      # http://127.0.0.1:8766/

# Tests
python -m pytest
```

Steps 1→2→3 are strictly ordered: `predict` refuses to run without
`models/interval_residuals.csv`, which only `backtest` writes **[code]**.
Step 6 is independent of 1–5 — it refits everything it needs, leakage-safely.

**In-season depth refresh** **[code]** (`src/depth_chart/README.md`):

```bash
python -m src.depth_chart.refresh --season 2026            # dry run → proposals
python -m src.depth_chart.refresh --season 2026 --apply    # auto-safe IR/PUP events
# then re-run steps 3, 4, 7
```

`src/depth_chart/starters_2026.csv` is the curated base and is **never**
auto-edited.

**Standalone gates** for the mix layers **[code]**:

```bash
python -m src.projection.team_pass_mix    # LOSO validation of the L2 pass mix
python -m src.projection.team_rush_mix    # LOSO validation of the L2 rush mix
```

Both are **advisory only** — `predict` attaches the scheme mix unconditionally
whether or not the gate passes **[doc]** (`PROVENANCE_AUDIT.md` §2).

---

## 5. Known open defects and decisions

| # | Item | Status | Where |
|---|---|---|---|
| 1 | Evaluation artifacts and freeze predate the harness rewrite | **Open, blocking any new claim** | §3.3 |
| 2 | Whole allocation layer uncommitted | **Open** | `REPO_HYGIENE_AUDIT.md` §1, §7 |
| 3 | 17 of 30 tuning knobs unmeasured; several bind on the board | **Open** | `PROVENANCE_AUDIT.md` §1, §3 |
| 4 | Sleeper is the deciding evidence in ~half the allocation decisions | **Open — retirement plan pending** | `PROVENANCE_AUDIT.md` §4 |
| 5 | L2 mix gates are advisory; `mix_source == 'scheme_model'` on 100% of rows | **Open** | `PROVENANCE_AUDIT.md` §2 |
| 6 | No pass-mix LOSO result recorded anywhere; the doc gives a command and a condition, no number | **Open** | `HIERARCHICAL_PASS_MIX_2026-08-14.md` |
| 7 | `WR_FORMATION_ROLE_PRIORS` re-enters the live path at weight 0.5 after the same fitted priors were disabled at `USAGE_SHARE_BLEND_W = 0.0` for losing the fantasy evaluation | **Open — highest-priority finding after §3.4** | `PROVENANCE_AUDIT.md` §2 |
| 8 | Elite shrinkage sits at season-consistency 2.1 against its own gate of 2.0 | **Watch** | `PHASE7_REMEDIATION_REPORT.md` |
| 9 | TE replacement-level calibration: raised in the freeze at 92.99 vs actual 133.00 **[doc]**; currently 98.71 vs 133.00 on disk **[code]** — still a ~34-point undershoot | **Watch** | `FREEZE_2026-08-13.md` |
| 10 | Three curated QBs (Watson, Bennett, DeVito) have no projection row by design; `MISSING` tripwire fires every run | **Accepted** | `DEPTH_CHART_ALLOCATION_2026-08-14.md` |
| 11 | Team-total model fit on ~32 rows/season; its error lands on every receiver of a team at once | **Open, named as highest-leverage** | `PHASE7_REMEDIATION_REPORT.md` |
| 12 | Fantasy intervals are componentwise, not a joint fantasy-score interval | **Open** | `PHASE7_REMEDIATION_REPORT.md` |
| 13 | `DEPTH_CHART_ALLOCATION_2026-08-14.md` §"Still open" says `INCUMBENT_VACANCY_ALPHA['carry']` remains blocked; `RB_CARRY_VACANCY_2026-08-14.md` ships it at 1.0, and `contracts.py:45` confirms 1.0 **[code]**. Both records are dated 2026-08-14 and both are live | **Contradiction — resolve in the docs** | both records |
| 14 | `DEPTH_RANK_TO_WR_FORMATION_ROLE` is dead code with zero consumers | **Delete candidate** | `PROVENANCE_AUDIT.md` §1 |
| 15 | Test count: `FREEZE` claims 57, `AGE_EFFECT_SHRINKAGE` claims 63; a static count of `def test_` across 16 files gives 120 **[code]**. No suite was run for this document | **Stale doc numbers** | §3.3 |

**Live board tripwires** **[code]** (`predict._warn_board_level_allocation`) —
stderr only, never change a number: `CAPPED` (a rate pinned on a support
ceiling), `MISSING` (a curated contributor with no row), `RB SHARE` (one back
over 70% of team carries), `NEWCOMER` (a rookie or arrival projected past the
curated starter ahead of him). Read them on every run.

---

## 6. What is deliberately not measured, and why

1. **Sleeper agreement is not an accuracy metric and is not a target.**
   Sleeper projects a full slate; this pipeline projects expected value with
   availability priced in, so a systematic negative bias against Sleeper is the
   *expected* signature of a working availability model, not a defect. The
   comparison ships as a read-only diagnostic. That said, §5 item 4 records that
   past decisions *did* use it as the deciding evidence; the honest statement is
   that Sleeper *should not be* a target, not that it never was.
2. **Rookies are never mixed in as equally-confident numbers.** Every rookie row
   carries `low_confidence = True` and `source = 'rookie_rule'` **[code]**, and
   the rookie interval uses a within-bucket multiplicative ratio rather than the
   veteran residual table, because rookie-year variance is a different regime.
   Rookie-path MAE is reported at n = 7–25 per position **[doc]** and is
   directional only.
3. **The curated depth chart is not scored on a historical fold** and cannot be —
   it is hand research that exists for 2026 alone. §3.2. This is the single
   largest measurement gap and it is structural, not an oversight.
4. **Fumbles and two-point conversions are excluded from scoring** because they
   are not modeled upstream **[code]** (`fantasy_evaluation` module docstring).
5. **August camp cuts are outside the evaluation universe.** The population is
   the earliest regular-season Week-1 roster snapshot; a player cut before Week 1
   cannot be evaluated **[code]**.
6. **Veterans with no source-season feature row are not dropped.** They stay in
   `all_eligible` scoring zero model points rather than disappearing through an
   inner join, so coverage cannot be laundered into performance. Both
   `all_eligible` and `forecast_covered` scopes are always emitted **[code]**.
7. **Interval calibration is marginal, not joint.** Strictly-forward marginal
   coverage 0.820 against a nominal 0.800 **[doc]**; there is no joint
   fantasy-score interval. §5 item 12.
8. **Defense, kicking, and special teams are not modeled at all.** The universe
   is QB / RB / WR / TE **[code]** (`TARGET_STATS`).

---

## 7. Document map

[`DOC_INVENTORY.md`](DOC_INVENTORY.md) classifies all 25 root markdown files as
live / historical / stale / decision record, quotes the specific stale claims,
and proposes an archival layout (`docs/history/`, `docs/decisions/`, root
reserved for live documents).

Short version — the documents that are **live** right now:

- This file.
- `DOC_INVENTORY.md`.
- `PROVENANCE_AUDIT.md`, `REPO_HYGIENE_AUDIT.md` (other agents' active
  deliverables).
- The eight dated decision records (`*_2026-08-14.md`) — authoritative for the
  tuning values they ship, and simultaneously under review, since
  `PROVENANCE_AUDIT.md` has since reclassified several of their gates as
  proxy-based rather than outcome-based. Both statements are true at once; the
  values are what ships, the gates are contested.
