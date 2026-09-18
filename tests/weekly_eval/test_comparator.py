"""Role 2 comparator: score shadow means against snapshots and actuals."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.projection.weekly_eval.comparator import compare_shadow_to_vegas
from src.projection.weekly_eval.errors import (
    MissingAsOfError,
    OutcomeFeatureLeakageError,
    Role3BlendForbiddenError,
)
from src.projection.weekly_eval.schema import (
    load_outcomes,
    load_prop_snapshots,
    load_shadow_board,
)

FIXTURE_DIR = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "projection"
    / "weekly_eval"
    / "fixtures"
)


def test_compare_fixture_board_reports_role2_metrics_not_promotion():
    summary = compare_shadow_to_vegas(
        board=load_shadow_board(FIXTURE_DIR / "shadow_board.csv"),
        snapshots=load_prop_snapshots(FIXTURE_DIR / "prop_snapshots.csv"),
        outcomes=load_outcomes(FIXTURE_DIR / "outcomes.csv"),
    )
    assert summary["role"] == "evaluation_comparator"
    assert summary["role3_blend"] is False
    assert summary["promoting"] is False
    assert summary["gate_verdict"] == "not_promoting"
    assert "not a promotion sample" in summary["caveat"].lower()
    assert summary["n_matched"] >= 3
    assert summary["n_with_actuals"] >= 3
    metrics = summary["metrics"]
    for key in (
        "mae_model",
        "mae_market",
        "rmse_model",
        "brier_model",
        "brier_market",
        "crps_gaussian_model",
        "pinball_q50_model",
        "mae_model_vs_market",
    ):
        assert key in metrics
        assert metrics[key] == metrics[key]  # not NaN
    assert metrics["mae_model"] >= 0.0
    assert metrics["brier_model"] >= 0.0
    assert metrics["crps_gaussian_model"] >= 0.0
    assert "by_market" in summary


def test_compare_rejects_smuggled_outcome_columns_on_board():
    board = load_shadow_board(FIXTURE_DIR / "shadow_board.csv")
    board = board.copy()
    board["fantasy_points"] = 20.0
    with pytest.raises(OutcomeFeatureLeakageError, match="fantasy_points"):
        compare_shadow_to_vegas(
            board=board,
            snapshots=load_prop_snapshots(FIXTURE_DIR / "prop_snapshots.csv"),
        )


def test_compare_rejects_smuggled_actual_column_on_board():
    board = load_shadow_board(FIXTURE_DIR / "shadow_board.csv")
    board = board.copy()
    board["actual"] = 20.0
    with pytest.raises(OutcomeFeatureLeakageError, match="actual"):
        compare_shadow_to_vegas(
            board=board,
            snapshots=load_prop_snapshots(FIXTURE_DIR / "prop_snapshots.csv"),
        )


def test_compare_rejects_missing_as_of_in_snapshot_frame():
    snaps = pd.DataFrame(
        [
            {
                "player_id": "p1",
                "season": 2026,
                "week": 1,
                "market": "rec_yards",
                "line": 70.5,
                "kickoff_at": "2026-09-11T17:00:00+00:00",
            }
        ]
    )
    board = pd.DataFrame(
        [
            {
                "player_id": "p1",
                "season": 2026,
                "week": 1,
                "market": "rec_yards",
                "model_mean": 72.0,
            }
        ]
    )
    with pytest.raises(MissingAsOfError):
        compare_shadow_to_vegas(board=board, snapshots=snaps)


def test_compare_forbids_role3_blend_weights():
    with pytest.raises(Role3BlendForbiddenError):
        compare_shadow_to_vegas(
            board=load_shadow_board(FIXTURE_DIR / "shadow_board.csv"),
            snapshots=load_prop_snapshots(FIXTURE_DIR / "prop_snapshots.csv"),
            blend_weights={"model": 0.5, "vegas": 0.5},
        )


def test_compare_without_outcomes_still_reports_model_vs_market():
    summary = compare_shadow_to_vegas(
        board=load_shadow_board(FIXTURE_DIR / "shadow_board.csv"),
        snapshots=load_prop_snapshots(FIXTURE_DIR / "prop_snapshots.csv"),
        outcomes=None,
    )
    assert summary["n_with_actuals"] == 0
    assert summary["metrics"]["mae_model_vs_market"] >= 0.0
    assert summary["metrics"]["mae_model"] is None
    assert summary["promoting"] is False


def test_compare_medians_multi_book_duplicates_not_pick_latest():
    """Per-book rows stay in the CSV; join grain uses robust_median (no book-shop)."""
    board = pd.DataFrame(
        [
            {
                "player_id": "00-0034857",
                "season": 2026,
                "week": 2,
                "market": "pass_yards",
                "model_mean": 266.0,
            }
        ]
    )
    # Later as_of is the worse book (270.5). Pick-latest would book-shop; median = 266.5.
    snaps = pd.DataFrame(
        [
            {
                "player_id": "00-0034857",
                "season": 2026,
                "week": 2,
                "market": "pass_yards",
                "line": 262.5,
                "implied_mean": 262.5,
                "as_of": "2026-09-16T12:00:00+00:00",
                "kickoff_at": "2026-09-17T20:15:00+00:00",
                "source": "draftkings",
            },
            {
                "player_id": "00-0034857",
                "season": 2026,
                "week": 2,
                "market": "pass_yards",
                "line": 270.5,
                "implied_mean": 270.5,
                "as_of": "2026-09-16T18:00:00+00:00",
                "kickoff_at": "2026-09-17T20:15:00+00:00",
                "source": "fanduel",
            },
        ]
    )
    summary = compare_shadow_to_vegas(board=board, snapshots=snaps, outcomes=None)
    assert summary["n_matched"] == 1
    assert summary["n_snapshots"] == 2
    assert summary["metrics"]["mae_model_vs_market"] == pytest.approx(0.5)
