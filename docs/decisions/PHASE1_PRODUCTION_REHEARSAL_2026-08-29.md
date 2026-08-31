# Phase 1 production rehearsal — 2026-08-29

Empirical acceptance test for manifest-first accuracy promotion: full 10,000-draw 2026 candidate build, validate, promote, browser surfaces, rollback.

## Namespaces

| Role | Namespace | Manifest SHA-256 |
|------|-----------|------------------|
| Candidate (10k publish) | `phase1_rehearsal_20260829` | `a89865441775f67b46ceb6ce50898b859e4a4fc4295585a0b3d2b62bd121ab0c` |
| Prior (rollback target) | `phase1_rehearsal_prior` | `4bf619dc1b9d09aa18ab482d5025824cf0af4d893721566fe0c61f5242ed70a0` |

## Publish run

- **Command:** `python -m src.projection.publish --season 2026 --simulation-profile publish --artifact-namespace phase1_rehearsal_20260829`
- **Wall time:** ~9,191 s (~2 h 33 min) — simulation completed; sealing failed initially (see defects)
- **Release ID:** `c17b8f4c-d8bc-4a3f-8077-974eb533a03b`
- **Model:** `accuracy_first_ensemble`
- **Draw count:** 10,000 (`publish` profile)
- **Simulated players:** 778
- **Artifact count (sealed):** 56

### Defect found and fixed

Sealing failed with `ReleaseBundleError: namespace contains unlisted files: ['simulations_2026.parquet']`. The publish pipeline wrote the consolidated draws parquet but did not enumerate it in `artifact_specs`. Fixed in `release_bundle_publish.py`; completed seal via `scripts/complete_release_bundle.py` against staged artifacts (no re-simulation).

## Validation (pre-promote)

All 17 checks **pass** on candidate:

- Sealed manifest canonical hash
- Full artifact hash + enumeration (no unlisted files)
- `accuracy_first_ensemble` model id
- Application contract hash alignment
- Publish draw count (10k) and profile
- Overlay population hash (778 players)
- Selected board / points vector hashes

Public browser copies verified byte-identical to sealed manifest entries.

## Promotion chain

1. Copied candidate → `phase1_rehearsal_prior` (distinct namespace + release id for rollback chain)
2. Promoted prior → active (first pointer write)
3. Promoted candidate → active (`previous` = prior)
4. `validate --require-active` on candidate: **pass**
5. Browser verification (all 5 surfaces): **pass** — namespaced URLs under `data/releases/phase1_rehearsal_20260829/`

## Rollback

- **Command:** `python -m src.projection.promote_release --season 2026 --rollback`
- Repointed active pointer to `phase1_rehearsal_prior` (manifest `4bf619dc…`)
- `validate --require-active` on prior: **pass**
- Browser verification (all 5 surfaces): **pass** — all assets resolve to `data/releases/phase1_rehearsal_prior/`

## Browser surfaces exercised

Draft, Teams, Totals, Compare, Sleepers — each verified for:

- Active pointer fetch + manifest hash match
- Browser-consumed artifacts (players, team_stats, comparison, deep_band_accuracy)
- Per-surface asset URLs via `release_loader.js` resolution

HTTP server: `python -m http.server 8765` in `draft_assistant/`.

## Post-rehearsal state

- **Active pointer:** `phase1_rehearsal_prior` (rollback end state)
- Legacy flat files under `draft_assistant/data/` unchanged from pre-rehearsal hashes

## Verdict

**Production rehearsal passed.** Real 2026 data, 10k-draw simulation, immutable bundle enumeration, public copy routing, active-pointer promotion, and rollback all work together within observed runtime and storage constraints.

Minor CLI fix: `--artifact-namespace` is no longer required when `--rollback` is set.
