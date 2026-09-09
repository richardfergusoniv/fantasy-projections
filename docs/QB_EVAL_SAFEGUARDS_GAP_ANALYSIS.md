# QB evaluation safeguards — gap analysis

**Date:** 2026-09-09  
**Base:** current production (`master`)  
**Compared to:** PR #8 tip `4d61c891b24e817995b82efa2f9ecaa09b067a92`  
**Negative research only:** PR #18 tip `9ad7ec4d6854902835b543dc0107071820d7483e`  
**Decision:** do **not** merge H3 or H4; extract production-safe evaluation/allocation infrastructure only.

No applicable `AGENTS.md` was present in the repository at the time of this work.

---

## Why neither research PR is promoted

### PR #8 (H3 infra repair + frozen rerun)

H3 repaired evaluation infrastructure and then still **failed** its predeclared promotion gate: the 2025 primary-cohort paired bootstrap CI included zero. Material residual cohort concerns remain (2025 insufficient-history, depth-1, 2024 top-12). Merging the full research branch would also activate H3 model IDs, experiment packages, and frozen candidate pipelines that are explicitly out of scope for production.

### PR #18 (H4 insufficient-history prior)

H4 is research-only, stacked on PR #8 (not on production). Latest chronological OOS (2025) failed materially (full-universe ΔMAE +9.28; H3-comparable +2.68 outside +2% tolerance; primary bootstrap CI crossed zero). **Do not** copy the H4 prior, weaken gates, retune against 2025, or start H5.

---

## Inspection summary

| Area | Production (`master`) | PR #8 infra | Port? |
|---|---|---|---|
| Fail-fast on missing/zero-byte `projections.db` | Mentioned as blocked in repair script notes; **no reusable guard** that refuses silent skip | `qb_h3/projections_db.py` | **Yes** |
| Portable reconcile fixture + schema/leakage contract | Absent | `qb_h3/portable_contract.py` + fixture under `output/qb_h3/infra/` | **Yes** (renamed away from H3) |
| Role-aware expected-start allocation (QB1 vs residual backups; room = 17) | `qb_repair/allocation.py` protects starter **pass-share** under team reconcile; does **not** allocate expected starts by preseason role | `qb_h3/role_allocation.py` | **Yes** (self-contained; no H3 active-rate package) |
| Availability applied exactly once | Compose/exposure helpers exist; **no** explicit once-availability identity contract for QB eval | `qb_h3/composition_contract.py` | **Yes** |
| Team pass / QB-rush conservation reporting | `qb_repair` reports pass-volume conservation on experimental arm; no shared eval invariant helper | H3 pipeline asserts exact conservation | **Yes** (shared helper) |
| Missing designed/scramble ≠ pocket | `classify_qb_archetype` uses carries only → low carries ⇒ `"pocket"` even when designed/scramble are null | H3 classifier requires **observed** low designed **and** scramble for pocket | **Yes** (safeguard classifier + harden repair classifier) |
| H3 forecast / hierarchical rush priors / end-to-end candidate | Absent (correct) | `qb_h3/forecast.py`, `pipeline.py`, eval scripts | **No** |
| H3/H4 model IDs, gates, ensemble weights | Sealed production unchanged | Frozen H3 / H4 research | **No** |
| H4 rookie/insufficient-history priors + designed coverage expand | Absent | PR #18 only | **No** |
| Player-name hardcodes / ADP-ECR targets / release pointer moves | Must stay absent | Not promoted | **No** |

---

## Already present on production (keep; do not regress)

1. **Leakage-safe history boundary** — `qb_repair.history.history_before` and tests.
2. **Experimental starter/backup pass-share allocation** — `qb_repair.allocation.reconcile_qb_volume_with_allocation` with conservation reporting (arm-only; not a production default switch).
3. **Non-QB invariance helper** — `qb_repair.apply_board.non_qb_invariance_check`.
4. **Sealed release + active pointer** — `v2_baseline_20260830` / `active_release_2026.json` untouched by this work.
5. **Promotion principle** — rolling-origin, leakage-safe evidence required before any model promotion (unchanged).

---

## Missing from production (ported here)

Implemented under `src/projection/qb_eval_safeguards/`:

1. **Fail-fast reconciliation contract** — refuse missing/empty DB and refuse proceeding without a validated portable fixture when reconciliation is required.
2. **Portable fixture schema/version + leakage audit** — prediction vs label column split; deterministic content hash; schema version `qb_eval_reconcile_contract_v1`.
3. **Role-aware expected starts** — preseason depth/role at cutoff; backups get residual starts only; room conserves `SEASON_GAMES` (17); productivity-while-active does not imply starter status.
4. **Availability-once composition contract** — identity checks + double-availability detector.
5. **Team volume conservation helper** — explicit tolerance; fail loudly on incomplete rooms.
6. **Missingness-safe archetype classification** — observed pocket / mobile / insufficient_history / missing_identity; null designed never maps to pocket.

Also: harden `qb_repair.rate_prior.classify_qb_archetype` so null designed/scramble with only low carries cannot silently become `"pocket"` when designed/scramble columns are present-but-null.

---

## Deliberately excluded (H3/H4 research)

- Entire `src/projection/qb_h3/` model pipeline, forecast, hierarchical priors, and end-to-end eval scripts
- Entire `src/projection/qb_active_archetype/` experiment package and H3/H4 thresholds/gates
- `scripts/qb_h3_*.py`, `scripts/qb_h4_*.py`, `scripts/qb_active_archetype_eval.py`
- H4 experience taxonomy / insufficient-history empirical-Bayes priors
- Any activation of H3/H4 model IDs or new global QB production default
- Sealed artifact regeneration, `active_release_2026.json` moves, ADP/ECR fitting, player hardcodes, H5

---

## Validation stance preserved

- Fit/derive prediction inputs only from seasons before the target season.
- Keep season-level and cohort visibility (do not pool away regressions).
- Treat 2026 as prospective diagnostics only.
- ADP/ECR remain diagnostic-only.
- Infrastructure correctness ≠ model promotion.
