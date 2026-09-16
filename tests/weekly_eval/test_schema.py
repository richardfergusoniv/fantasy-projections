"""Fail-closed snapshot schema for Role 2 Vegas weekly props evaluation."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from src.projection.weekly_eval.errors import (
    MissingAsOfError,
    MissingKickoffError,
    PostKickoffSnapshotError,
)
from src.projection.weekly_eval.schema import (
    load_outcomes,
    load_prop_snapshots,
    load_shadow_board,
    prop_snapshot_from_row,
)

FIXTURE_DIR = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "projection"
    / "weekly_eval"
    / "fixtures"
)


def test_prop_snapshot_requires_as_of():
    with pytest.raises(MissingAsOfError, match="as_of"):
        prop_snapshot_from_row(
            {
                "player_id": "p1",
                "season": 2026,
                "week": 1,
                "market": "rec_yards",
                "line": 75.5,
                "kickoff_at": "2026-09-11T17:00:00+00:00",
            }
        )


def test_prop_snapshot_rejects_blank_as_of():
    with pytest.raises(MissingAsOfError, match="as_of"):
        prop_snapshot_from_row(
            {
                "player_id": "p1",
                "season": 2026,
                "week": 1,
                "market": "rec_yards",
                "line": 75.5,
                "as_of": "   ",
                "kickoff_at": "2026-09-11T17:00:00+00:00",
            }
        )


def test_prop_snapshot_requires_kickoff_cutoff():
    with pytest.raises(MissingKickoffError, match="kickoff"):
        prop_snapshot_from_row(
            {
                "player_id": "p1",
                "season": 2026,
                "week": 1,
                "market": "rec_yards",
                "line": 75.5,
                "as_of": "2026-09-09T12:00:00+00:00",
            }
        )


def test_prop_snapshot_rejects_as_of_after_kickoff():
    with pytest.raises(PostKickoffSnapshotError, match="kickoff"):
        prop_snapshot_from_row(
            {
                "player_id": "p1",
                "season": 2026,
                "week": 1,
                "market": "rec_yards",
                "line": 75.5,
                "as_of": "2026-11-15T12:00:00+00:00",
                "kickoff_at": "2026-09-11T17:00:00+00:00",
            }
        )


def test_prop_snapshot_accepts_pre_kickoff_as_of():
    snap = prop_snapshot_from_row(
        {
            "player_id": "p_chase",
            "season": 2026,
            "week": 1,
            "market": "rec_yards",
            "line": 75.5,
            "implied_p_over": 0.52,
            "implied_mean": 76.2,
            "as_of": "2026-09-09T12:00:00+00:00",
            "kickoff_at": "2026-09-11T17:00:00+00:00",
        }
    )
    assert snap.player_id == "p_chase"
    assert snap.as_of == datetime(2026, 9, 9, 12, 0, tzinfo=UTC)
    assert snap.kickoff_at == datetime(2026, 9, 11, 17, 0, tzinfo=UTC)
    assert snap.implied_location() == pytest.approx(76.2)


def test_committed_fixture_snapshots_load():
    snaps = load_prop_snapshots(FIXTURE_DIR / "prop_snapshots.csv")
    assert len(snaps) >= 3
    assert all(s.as_of is not None for s in snaps)
    assert all(s.kickoff_at is not None for s in snaps)
    assert {s.market for s in snaps} <= {
        "rec_yards",
        "rush_yards",
        "pass_yards",
        "receptions",
    }


def test_load_prop_snapshots_fail_closed_on_missing_as_of(tmp_path: Path):
    path = tmp_path / "bad_snaps.csv"
    pd.DataFrame(
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
    ).to_csv(path, index=False)
    with pytest.raises(MissingAsOfError):
        load_prop_snapshots(path)


def test_committed_shadow_board_and_outcomes_load():
    board = load_shadow_board(FIXTURE_DIR / "shadow_board.csv")
    outcomes = load_outcomes(FIXTURE_DIR / "outcomes.csv")
    assert not board.empty
    assert {"player_id", "season", "week", "market", "model_mean"} <= set(board.columns)
    assert not outcomes.empty
    assert {"player_id", "season", "week", "market", "actual"} <= set(outcomes.columns)
