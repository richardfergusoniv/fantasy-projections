# App Projection Re-Anchor Report

Generated: 2026-08-31 (America/Los_Angeles)

## 1. Resolved Active Production Release

Dynamically resolved from `draft_assistant/data/active_release_2026.json`:

| Field | Value |
|---|---|
| Namespace | `v2_baseline_20260830` |
| Release ID | `e92edd22-40d9-4219-87f6-47a651489d15` |
| Manifest SHA-256 | `5a8e14536aa7b062b1e5ff6e64aa78356847fb063579d0a42cdbdc5cc159fbb1` |
| Player count | 778 |
| Model ID | `accuracy_first_ensemble` |

**Repository evidence note:** The public browser copy under `draft_assistant/data/releases/v2_baseline_20260830/` contains `players_2026.json`, `team_stats_2026.json`, `comparison_2026.json`, `deep_band_accuracy.json`, and `release_bundle_manifest.json` only. Full bundle artifacts (`fantasy_points_2026.csv`, `projections_2026.csv`, simulation partitions) are referenced in the manifest but not copied to the public tree. The hardened loader validates consumed artifacts only and records caveats:

- `missing_optional_artifact:selected_board`
- `missing_optional_artifact:projections`
- `missing_optional_artifact:simulation_summary`
- `component_projections_from_output_fallback` (uses `output/projections_2026.csv` when bundle copy absent)

This matches the prompt's instruction to follow signed/sealed repository evidence rather than assuming a complete local copy.

Verified artifact hashes (from manifest, players artifact present locally):

- `players` SHA-256: `3baad7401f84fba4b3561ba4b6429020b415624a9d848dc037528b7ee22195b8`

## 2. Source / Overlay / R&D Boundaries

```
┌─────────────────────────────────────────────────────────────────┐
│                    APP_PROJECTION_SOURCE                        │
│              default: sealed_release (production)               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         ▼                  ▼                  ▼
  sealed_release    status_adjusted_release   weekly_v2_rnd
  (default)         (overlay pointer)       (WEEKLY_RND_ENABLED)
         │                  │                  │
         ▼                  ▼                  ▼
  active_release_    active_status_       weekly_v2 artifacts
  2026.json pointer  overlay pointer      + promote_week (opt-in)
         │                  │
         ▼                  ▼
  ReleaseBundleLoader   Immutable overlay
  manifest+hash verify  availability/depth
  players + components  conservative adjust
         │
         ▼
  ProjectionService ──► decisions (draft, waivers, trades, start/sit)
         │
         ▼
  league_rescore.py ──► per-league component rescoring + fidelity label
```

**Contract preserved:**
- v1/v2/ADP ensemble point means remain authoritative (`mean_points` from `fantasy_pts_season / projected_games`)
- v3 simulation quantiles are distributional overlay only (never replace means)
- Weekly-v2/event/joint draws remain R&D until their own gates pass

## 3. Six Live League Scoring-Fidelity Results

League rescoring uses `LeagueRuleSnapshot` scoring_settings from persisted Sleeper sync (not fixture-mapped IDs). With `output/projections_2026.csv` as component source:

| Capability | Expected fidelity | Notes |
|---|---|---|
| Standard PPR/redraft | `exact_component_rescore` or `modeled_approximation` | Depends on PPFD/bonus rules in live snapshot |
| Superflex dynasty | `exact_component_rescore` or `modeled_approximation` | Slot mapping via `compile_sleeper_scoring` |
| Leagues with unsupported keys | `unsupported_rule` | Blocks publication per existing gate |

**Configured leagues** (from owner config, not committed):

| League ID | Name | Type | Rookie order |
|---|---|---|---|
| `<redraft-league-1>` | Redraft league 1 | redraft | — |
| `<redraft-league-2>` | Redraft league 2 | redraft | — |
| `<dynasty-league-1>` | Dynasty league 1 | dynasty | reverse_standings |
| `<dynasty-league-2>` | Dynasty league 2 | dynasty Superflex | max_pf |
| `<dynasty-league-3>` | Dynasty league 3 | dynasty | max_pf |
| `<dynasty-league-4>` | Dynasty league 4 | dynasty Superflex | reverse_standings |

**External blocker:** Live `LeagueRuleSnapshot` rows require PostgreSQL + Sleeper sync. Run:

```bash
APP_ENV=development .venv/bin/python -m src.app.cli sleeper-shadow-sync
# or daily-refresh with SLEEPER_USE_FIXTURES=false
```

Then:

```bash
APP_ENV=test .venv/bin/python -c "
from src.app.persistence.database import init_db, get_session
from src.app.projections.league_rescore import rescore_configured_leagues
from pathlib import Path
init_db()
with get_session() as s:
    for r in rescore_configured_leagues(s, components_path=Path('output/projections_2026.csv')):
        print(r.to_dict())
"
```

Unit tests prove PPFD cannot receive `exact_component_rescore` without components and that half-PPR does not leak into full-PPR leagues.

## 4. Status-Overlay Gate and Rollback

**Gate checks** (`validate_overlay_gate`):
- Base release ID and manifest hash present
- Finite values, valid availability ∈ [0,1]
- OUT/inactive players zeroed
- Reason codes required per adjustment

**Rollback evidence:** `test_overlay_rollback_restores_prior_pointer` promotes two overlay versions and verifies `rollback_overlay_pointer` restores the prior `overlay_hash`.

**Schedule** (existing `scheduler.py`, unchanged):
- Mon–Sat 5:00 PM PT (`daily-refresh`)
- Sun 8:45 AM, 11:45 AM, 4:00 PM PT
- Overlay refresh runs inside `daily-refresh` when `STATUS_OVERLAY_AUTO_PUBLISH=true`

## 5. Capability GO/NO-GO Matrix

| Capability | Verdict | Rationale |
|---|---|---|
| Sealed season projection source | **GO** | Active pointer resolves; 778 players load; manifest hash verified |
| Exact/approximate league rescoring | **GO** (code) / **BLOCKED** (live data) | Compiler + rescore implemented; needs live league snapshots |
| Daily status-overlay generation | **GO** | Build + gate implemented |
| Automatic status-overlay publication | **DEGRADED** | No active overlay pointer until first successful daily refresh |
| Private core-app beta | **GO** | App starts without weekly-v2; sealed source default |
| Season-baseline start/sit advisory | **GO** | Labeled advisory; expected points from sealed/overlay baseline |
| Matchup-specific weekly start/sit/win probability | **NO-GO** | Weekly-v2 R&D gates not passed; `matchup_win_probability_available=false` |
| Weekly-v2/event/joint auto-publication | **NO-GO** | `auto_publish_allowed=false` per existing evaluation |
| Public-internet deployment | **NO-GO** | PostgreSQL, TLS, production email/auth not verified in this task |

**Important:** Weekly R&D NO-GO does **not** mark production unhealthy (`production_healthy` is independent).

## 6. Commands and Test Results

```bash
# New re-anchor tests (19 passed)
APP_ENV=test .venv/Scripts/python.exe -m pytest tests/app/test_projection_reanchor.py tests/app/test_projections_loader.py -q

# Result: 19 passed
```

```bash
# Full app suite (may require PostgreSQL for some tests)
APP_ENV=test .venv/Scripts/python.exe -m pytest tests/app/ -q
```

## 7. External Blockers

| Blocker | Narrowest next step |
|---|---|
| PostgreSQL not running locally | `docker compose up -d db` then re-run league rescoring script |
| Full bundle artifacts not in public tree | Copy sealed bundle from `output/model_v3/release_bundles/...` or run publish pipeline; removes loader caveats |
| Live Sleeper league snapshots | `sleeper-shadow-sync` with `config/sleeper_owner.json` (gitignored) |
| Weekly-v2 R&D gates | Continue event/joint eval repair per existing NO-GO reports; do not weaken frozen thresholds |
| Public deployment | Verify production config, TLS, backups per `Settings.production_config_problems()` |

## 8. Files Changed

| File | Change |
|---|---|
| `src/app/projections/source.py` | **New** — projection source contract |
| `src/app/projections/service.py` | **New** — unified projection facade |
| `src/app/projections/loader.py` | Hardened validation, caching, provenance |
| `src/app/projections/league_rescore.py` | **New** — component rescoring + fidelity |
| `src/app/projections/status_overlay.py` | **New** — immutable overlay + gate |
| `src/app/readiness/capabilities.py` | **New** — capability matrix |
| `src/app/readiness/__init__.py` | **New** |
| `src/app/config.py` | `APP_PROJECTION_SOURCE`, `WEEKLY_RND_ENABLED`, `STATUS_OVERLAY_AUTO_PUBLISH` |
| `src/app/jobs/handlers.py` | Daily path: sealed + overlay; weekly opt-in only |
| `src/app/api/v1/operations.py` | Production vs R&D panels + capabilities |
| `src/app/decisions/services.py` | Source metadata; matchup win-prob gated |
| `.env.example` | New projection source env vars |
| `web/src/screens/Operations.tsx` | Split production / weekly R&D panels |
| `web/src/api/types.ts` | Operations capability types |
| `tests/app/test_projection_reanchor.py` | **New** — 16 focused tests |

**Preserved unchanged:** sealed release bundles, experiment artifacts, weekly NO-GO reports, frozen model thresholds.
