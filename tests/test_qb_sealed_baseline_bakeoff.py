"""Phase 1 sealed-baseline verification tests (frozen config)."""
from __future__ import annotations

import json
from pathlib import Path

from src.projection.qb_active_archetype.thresholds import GATES, thresholds_dict

ROOT = Path(__file__).resolve().parents[1]


def test_thresholds_still_frozen_values():
    """Guard against accidental retune after the prior holdout report."""
    assert GATES.overall_mae_non_inferiority_tol == 0.02
    assert GATES.cohort_improve_min_fit_folds == 2
    assert GATES.holdout_bootstrap_ci_must_exclude_zero is True
    assert GATES.use_2026_for_selection is False
    assert thresholds_dict()["holdout_season"] == 2025


def test_prior_comparator_was_conflated_not_sealed():
    """Documented: prior GO used comparator A, not sealed final B."""
    # Source of truth in evaluate.predict_player modes
    from src.projection.qb_active_archetype import evaluate as ev

    src = Path(ev.__file__).read_text(encoding="utf-8")
    assert "baseline_conflated" in src
    assert "candidate_active_archetype" in src


def test_sealed_bakeoff_artifacts_declare_comparator_and_nogo():
    decl = ROOT / "output" / "qb_sealed_baseline_bakeoff" / "comparator_declaration.json"
    decision = ROOT / "output" / "qb_sealed_baseline_bakeoff" / "selection_decision.json"
    assert decl.exists(), "run scripts/qb_sealed_baseline_bakeoff.py first"
    assert decision.exists()
    d = json.loads(decl.read_text(encoding="utf-8"))
    assert d["prior_reported_comparator"] == "A"
    assert d["this_bakeoff_comparator"] == "B"
    assert d["thresholds_frozen"] is True
    assert d["no_2025_retune"] is True
    sel = json.loads(decision.read_text(encoding="utf-8"))
    assert sel["decision"]["verdict"] == "NO-GO"
    assert sel["decision"]["production_promotion"] == "NO"
    assert sel["decision"]["comparator"]["prior_experiment"] == "A_injury_diluted_conflated_carry_forward"
    assert "B_sealed_final" in sel["decision"]["comparator"]["this_bakeoff"]
    # Burrow units must distinguish per-active vs availability-adjusted
    units = sel["burrow_units"]["2026_candidate_decomposition"]
    assert units["attempts_per_active_start"] > 35
    assert units["availability_adjusted_attempts_per_scheduled_team_game"] < units["attempts_per_active_start"]
    assert units["expected_season_attempts"] == units["attempts_per_active_start"] * units["expected_starts"]
