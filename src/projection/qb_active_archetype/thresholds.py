"""Predeclared evaluation thresholds for the active-start / archetype experiment.

These constants are fixed before candidate results are inspected. Do not edit
after seeing fold metrics. Existing release gates are not weakened.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


# --- Active-start identification (historical, leakage-safe) -----------------
# A REG week counts as an active start when the QB had meaningful opportunity.
# Thresholds are fixed a priori from common starter-usage floors.
ACTIVE_MIN_ATTEMPTS = 10
ACTIVE_MIN_TOUCHES = 12  # attempts + carries
# Partial / early-exit proxy among weeks with some usage but below active floor.
PARTIAL_MAX_ATTEMPTS = 9
PARTIAL_MIN_ATTEMPTS = 1

# Availability lookback for expected games / partial-exit rates.
AVAIL_LOOKBACK_SEASONS = 4
AVAIL_FULL_SEASON_GAMES = 17.0
# Shrink expected games toward league starter mean with this prior strength.
AVAIL_PRIOR_STRENGTH_GAMES = 8.0
LEAGUE_STARTER_EXPECTED_GAMES = 15.0
LEAGUE_PARTIAL_EXIT_RATE = 0.08

# --- Archetype classification (prior seasons only) --------------------------
ARCHETYPE_LOOKBACK = 4
ARCHETYPE_MIN_ACTIVE_STARTS = 8
DESIGNED_RUNNER_DESIGNED_PER_START = 4.0
MOBILE_SCRAMBLER_SCRAMBLE_PER_DB = 0.08
POCKET_MAX_DESIGNED_PER_START = 2.0
POCKET_MAX_SCRAMBLE_PER_DB = 0.05

# Hierarchical prior shrink toward archetype mean.
ARCHETYPE_PRIOR_STRENGTH_STARTS = 12.0

# --- Rolling-origin evaluation seasons --------------------------------------
# Trustworthy seasons = those with fantasy_evaluation CSVs in this environment.
# Weekly active-start labels cover 2018–2024; 2025 rates use evaluation approx.
EVAL_SEASONS = (2023, 2024, 2025)
FIT_SEASONS = (2023, 2024)
HOLDOUT_SEASON = 2025
MIN_EVAL_GAMES = 6
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 42

# Dual-threat / pocket labels for cohort metrics use the SAME pre-season
# archetype classifier (no future labels).


@dataclass(frozen=True)
class GateThresholds:
    """Non-inferiority + cohort improvement rules (predeclared)."""

    # Candidate overall QB season-points MAE may not exceed baseline by more
    # than this relative tolerance.
    overall_mae_non_inferiority_tol: float = 0.02
    # Primary cohort (dual-threat ∪ returning-injury) must improve MAE on at
    # least this many chronological fit folds.
    # With only two fit seasons available, require improvement on both.
    cohort_improve_min_fit_folds: int = 2
    # Holdout cohort MAE must improve (strict).
    holdout_cohort_must_improve: bool = True
    # Paired bootstrap CI on holdout primary-cohort MAE delta must exclude 0
    # on the helpful side (upper bound < 0).
    holdout_bootstrap_ci_must_exclude_zero: bool = True
    # Top-12 MAE must not worsen beyond this relative tolerance on holdout.
    top12_mae_non_inferiority_tol: float = 0.03
    # Spearman on all QB may not fall by more than this absolute amount.
    spearman_max_drop: float = 0.02
    # 2026 diagnostics never decide promotion.
    use_2026_for_selection: bool = False


GATES = GateThresholds()


def thresholds_dict() -> dict:
    return {
        "active_min_attempts": ACTIVE_MIN_ATTEMPTS,
        "active_min_touches": ACTIVE_MIN_TOUCHES,
        "partial_max_attempts": PARTIAL_MAX_ATTEMPTS,
        "avail_lookback_seasons": AVAIL_LOOKBACK_SEASONS,
        "archetype_lookback": ARCHETYPE_LOOKBACK,
        "designed_runner_designed_per_start": DESIGNED_RUNNER_DESIGNED_PER_START,
        "mobile_scrambler_scramble_per_db": MOBILE_SCRAMBLER_SCRAMBLE_PER_DB,
        "pocket_max_designed_per_start": POCKET_MAX_DESIGNED_PER_START,
        "pocket_max_scramble_per_db": POCKET_MAX_SCRAMBLE_PER_DB,
        "eval_seasons": list(EVAL_SEASONS),
        "fit_seasons": list(FIT_SEASONS),
        "holdout_season": HOLDOUT_SEASON,
        "gates": asdict(GATES),
        "note": "Frozen before candidate result inspection; do not weaken after seeing results.",
    }
