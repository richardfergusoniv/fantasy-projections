# Retire the interim simulation arm; ship generative percentiles

**Date:** 2026-08-26
**Verdict:** `ship_generative_percentiles`
**Scope:** `src/projection/inference/simulate.py` — `SIMULATION_MODE` is now `"full"`

## Why interim was retired

The interim arm drew `clip(pred + residual, 0)` per stat, independently.

Two defects followed from that, and they invert between grains:

- **The zero floor is asymmetric.** `E[max(0, X)] > E[X]`. 73.5% of stat rows
  have residual pools that can go negative and 28.9% of draws are clipped,
  worth **+8.33** fantasy points per player against an observed sim-mean-vs-v1
  gap of **+8.65**.
- **Residual right-skew** pushes each stat's *median* below its point estimate.

Summing eight stats removes most of the skew, so the total's median migrates
off the second bias onto the first: per-stat medians sum to **−3.93** against
v1 while the median of the sum is **+4.26**, a gap of +8.19 for 90.5% of
players. Scoring per-stat rates gives the opposite answer to scoring summed
fantasy points, and the shipped artifact is the latter.

Underneath both: per-stat residuals were drawn **independently**, while within
a player they correlate **+0.876** (passing yards/TDs), **+0.871** (receiving
yards/receptions), **+0.672** (rushing yards/TDs). Independence understates
the summed spread by **31%**.

## Measured, held out on 2025

Board built from history through 2024 only, scored against realised points
(`scripts/compare_simulation_modes.py`):

| mode | coverage | width | p50 MAE | p50 bias | rho |
|---|---|---|---|---|---|
| interim | 0.5050 | 62.23 | 34.563 | +1.036 | .6956 |
| **full (generative)** | **0.5367** | **60.60** | **33.863** | −2.065 | **.7414** |

Generative wins on every metric. Switched.

## Open defect: the band is not calibrated

**Coverage is 0.537 against a 0.80 target.** Shipping continues with that
known, on the reasoning that generative is strictly better than what it
replaces and the band is a secondary display column — not that it is correct.

The switch was recommended on the argument that a shared volume draw makes a
player's stats move together, so correlation would come for free. Measurement
says that argument is wrong twice over:

1. **No projection uncertainty.** Generative's spread is sampling noise around
   a **fixed** team volume and **fixed** conversion rates. The dominant real
   uncertainty — that the projection itself is wrong — is not represented at
   all. The residual bootstrap does represent it and merely scrambles the
   correlation, which is why a **joint bootstrap** (resampling a whole donor
   player-season's residual vector) reaches **0.757** on the same grain.
2. **Season-scale efficiency noise decorrelates the stats.** The conversion
   draws multiply volume by a fresh lognormal whose sigma (0.35 receiving,
   0.25 rushing, 0.20 passing) was chosen when the path emitted *per-game*
   lines. Over a season, volume is nearly deterministic (CV ~5%) while that
   factor carries CV ~36%, so it dominates: simulated yards correlate with
   receptions at **0.13**, against +0.871 in real residuals. Shrinking sigma
   restores it (0.15 → 0.31, 0.05 → 0.71).

So the two flaws partly offset — too narrow from (1), too noisy on efficiency
from (2) — and the net is a band that is both too tight overall and wrongly
shaped within a player.

Pinned by `test_season_scale_efficiency_noise_decorrelates_volume_stats`,
which asserts the defect and fails once it is fixed.

## Next, in order

1. Recalibrate the conversion sigmas for season aggregates.
2. Add projection uncertainty — `team_environment` already stores `resid_std`
   per team stat and it is currently unused.
3. Re-measure; if coverage still falls short, take the band from a joint
   bootstrap and keep generative for p50 and rank.
4. Re-point the calibration gate at fantasy-points coverage. It currently
   reports 0.8013, which is genuine but per-stat-rate; the percentiles it
   authorises cover 0.537.

## Note on interim

Retired as a shipping mode, kept callable so
`scripts/compare_simulation_modes.py` can still score it. `--mode interim` on
`scripts/run_v3_simulation.py` reaches it for comparison only.
