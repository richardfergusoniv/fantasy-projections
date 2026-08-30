# v1 Production Role (2026-08-29)

## Canonical role

v1 is the **canonical full component-statline engine** for this repository.

| Responsibility | Owner |
|---|---|
| Roster, depth, availability, team identities, reconciliation | v1 |
| Structural input to simulation before selected-board recentering | v1 |
| Fallback mean for players absent from v2/ADP and for the non-market tail | v1 |
| Current ensemble signal for QB and TE | v1 |
| Approved primary top-120 RB/WR ranking signal | **Not v1** unless later earned |

v3 remains **distribution-only**. Production RB/WR ensemble weights stay frozen through the 2026 validation season.

## What v1 is not

- v1 is not presumed to explain WR elite correction. The current elite correction is **TE-only**.
- v1 is not an approved primary top-120 RB/WR ranking signal unless a frozen shadow candidate later earns nonzero weight under the unchanged accuracy-first selector.

## Retired artifacts

`models/depth_rate_calibration.csv` and the depth-ladder multiplier path are **retired**. Availability owns games-played; the ladder is not applied in production. Documentation that still describes the obsolete multiplier should be treated as historical only.

## Shadow repair track (`shadow_v1_rb_wr`)

Until 2026 outcomes exist:

1. Run the non-mutating rolling-origin attribution study (`scripts/shadow_v1_rb_wr_attribution.py`).
2. Compare raw v1, composed v1, v2, ADP, and selected mean for `all_eligible` and frozen top-120 populations.
3. Separate projected-games error from per-game role/stat error.
4. Implement at most one candidate fix at a time; evaluate with nested rolling-origin fits.
5. Freeze any candidate as `shadow_v1_rb_wr` — **never** change production weights in-season.

After 2026 outcomes:

- A frozen candidate may regain RB/WR weight only when the unchanged accuracy-first selector gives it nonzero weight, every affected position improves MAE without reducing Spearman, and the overall ensemble passes the same rule.
- Otherwise close the repair track and retain v1's structural/fallback role.
