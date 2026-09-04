"""H4 frozen decision policy — predeclared before final evaluation.

Immutable relative to H3: same folds, primary cohort, paired bootstrap,
promotion thresholds, availability semantics, and conservation invariants.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from src.projection.qb_active_archetype.thresholds import (
    EVAL_SEASONS,
    FIT_SEASONS,
    GATES,
    HOLDOUT_SEASON,
)

MODEL_ID = "h4_insufficient_history_prior"
H3_MODEL_ID = "h3_repaired_frozen"
SEALED_COMPARATOR = "model_points_end_to_end"
H3_BASE_COMMIT = "4d61c891b24e817995b82efa2f9ecaa09b067a92"

# Experience thresholds (preseason-available only).
ESTABLISHED_MIN_PRIOR_ACTIVE_STARTS = 24  # ~1.5 seasons of starter-level activity
LIMITED_MAX_PRIOR_ACTIVE_STARTS = 23
LIMITED_MIN_PRIOR_ACTIVE_STARTS = 1
# Rookie = is_rookie_at_cutoff from preseason population (fantasy_evaluation).
# insufficient_history = non-rookie with 0 prior active starts OR missing identity.
# missing_identity = empty player_id / unmatched history key.


@dataclass(frozen=True)
class H4DecisionGates:
    """GO only if ALL of these hold. Do not weaken after seeing results."""

    # Inherit H3 frozen GateThresholds semantics vs sealed e2e.
    overall_mae_non_inferiority_tol: float = GATES.overall_mae_non_inferiority_tol
    cohort_improve_min_fit_folds: int = GATES.cohort_improve_min_fit_folds
    holdout_cohort_must_improve: bool = GATES.holdout_cohort_must_improve
    # Exact CI policy (unchanged from H3): upper bound of 95% paired bootstrap
    # CI on (candidate_MAE − sealed_MAE) for the primary cohort on the latest
    # chronological OOS fold must be strictly < 0.
    holdout_bootstrap_ci_must_exclude_zero: bool = GATES.holdout_bootstrap_ci_must_exclude_zero
    top12_mae_non_inferiority_tol: float = GATES.top12_mae_non_inferiority_tol
    spearman_max_drop: float = GATES.spearman_max_drop
    # H4-specific protections (predeclared).
    established_veteran_mae_non_inferiority_tol: float = 0.02
    # Rookie ∪ insufficient_history must not improve by pushing top-12 beyond tol.
    protected_top12_tol: float = GATES.top12_mae_non_inferiority_tol
    require_zero_conservation_violations: bool = True
    require_zero_double_availability: bool = True
    require_zero_non_qb_changes: bool = True
    use_2026_for_selection: bool = False


H4_GATES = H4DecisionGates()


def decision_policy_dict() -> dict:
    return {
        "model_id": MODEL_ID,
        "h3_model_id": H3_MODEL_ID,
        "sealed_comparator": SEALED_COMPARATOR,
        "h3_base_commit": H3_BASE_COMMIT,
        "eval_seasons": list(EVAL_SEASONS),
        "fit_seasons": list(FIT_SEASONS),
        "latest_chronological_oos": HOLDOUT_SEASON,
        "primary_cohort": "dual_threat ∪ returning_injury (same as H3)",
        "paired_bootstrap_ci_policy": (
            "On the latest chronological OOS fold, for the primary cohort, "
            "compute paired bootstrap of ΔMAE = H4_MAE − sealed_MAE. "
            "Gate passes only if the 95% CI upper bound is strictly < 0."
        ),
        "gates": asdict(H4_GATES),
        "go_requires": [
            "All frozen H3 gates vs sealed model_points_end_to_end",
            "Latest OOS overall ΔMAE within +2% of sealed MAE",
            "Primary-cohort paired bootstrap CI upper < 0 on latest OOS",
            "No material established_veteran regression beyond +2% tol on latest OOS",
            "Top-12 non-inferiority on latest OOS (≤ +3%)",
            "Zero team conservation violations",
            "Zero double-availability applications",
            "Zero non-QB projection changes",
            "H3 remains reproducible and unchanged",
        ],
        "do_not": [
            "Weaken any threshold after seeing results",
            "Tune against the 2025 final evaluation fold",
            "Use same-season outcomes as features",
            "Substitute a favorable point estimate for the CI gate",
            "Classify null designed-run history as pocket",
            "Publish or promote any model",
            "Change production projection source",
        ],
        "predeclared_before_final_eval": True,
    }
