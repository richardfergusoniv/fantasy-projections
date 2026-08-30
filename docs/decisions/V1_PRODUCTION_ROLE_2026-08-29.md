# v1 Production Role (2026-08-29)

## Canonical role

v1 is the **canonical full component-statline engine** for this repository.

| Responsibility | Owner |
|---|---|
| Roster, depth, availability, team identities, reconciliation | v1 |
| Structural input to simulation before selected-board recentering | v1 |
| Fallback mean for players absent from v2/ADP and for the non-market tail | v1 |
| Current ensemble signal for QB and TE | v1 |
| Approved primary top-120 RB/WR ranking signal | **Not v1** |

v3 remains **distribution-only**. Production RB/WR ensemble weights stay frozen through the 2026 validation season.

## What v1 is not

- v1 is not presumed to explain WR elite correction. The current elite correction is **TE-only** (`models/corrections.joblib`).
- v1 is **not** an approved primary top-120 RB/WR ranking signal. Shadow repair did not produce a freezable cutoff-safe candidate.

## Retired artifacts

`models/depth_rate_calibration.csv` and the depth-ladder multiplier path are **retired**. Availability owns games-played; the ladder is not applied in production (`role_discount_factor ≡ 1.0` audit constant). Depth reaches rates via `ROLE_FEATURES` / `depth_tier`, not a post-hoc multiplier. Documentation that still describes the obsolete multiplier should be treated as historical only.

## Shadow repair track — CLOSED (2026-08-30)

The `shadow_v1_rb_wr` track is **closed**. Further in-season RB/WR repair candidates are not authorized under this track.

### What was measured

1. Leakage-safe rolling-origin attribution (`scripts/shadow_v1_rb_wr_attribution.py`) with traced compose stages.
2. Error decomposition: raw-rate and availability are co-dominant and largely cancel; composition rate effect is small; finalization remainder is material (~6.5–7.0 abs on top-120) but fully explained by `reconcile_team_season_identities` (season columns only; rates untouched).
3. Step-6 oracle counterfactuals: `availability_only` and `raw_rate_only` clear diagnostic gates but are **not implementable** without outcome leakage or rate retunes.
4. Implementable Gate-A exposure blend (`shadow_availability_gate_a_blend_v1`) failed freeze gates (nested fit collapses to flat-17 after cold-start).

### Finalization / ladder / corrections audit

Bounded audit (`scripts/shadow_v1_rb_wr_close_repair_track.py` → `output/shadow_v1_rb_wr/finalization_audit/audit.json`):

| Check | Result |
|---|---|
| Depth-ladder live application | None on composition / depth_gating / eval / veterans / predict / backtest |
| `corrections.joblib` | TE-only; omitted on leakage-safe folds; zero `elite_correction_pg` on traced boards |
| Finalization remainder | ≡ team-identity season scaling (`corr=1` vs identity delta); not Gate-A mismatch |

**Finding:** `no_cutoff_available_defect`.

### Formal role after closeout

Canonical policy seal: `output/shadow_v1_rb_wr/repair_track_closed.json`

- Freeze and repair code read **only** this path for authorization.
- The finalization audit directory stores a pointer (`repair_track_closed_pointer.json`), not a second policy source.
- **v1 role:** structural / diagnostic only for top-120 RB/WR (plus continuing QB/TE ensemble and full component-statline / simulation structure).
- **Promotion:** not authorized from this track.
- **Further repair:** not authorized while the seal is absent, invalid, or `further_repair_authorized=false`. Reopening requires an explicit hash-valid authorization record with cutoff-available defect evidence — not deletion of the seal.
- **After 2026 outcomes:** RB/WR weight may change only via the unchanged accuracy-first selector on untouched outcomes — not via reopening this shadow repair loop.

### Availability-only candidate (historical)

`shadow_availability_gate_a_blend_v1` remains on disk under `availability_repair/` as a sealed hold, not a production candidate.
