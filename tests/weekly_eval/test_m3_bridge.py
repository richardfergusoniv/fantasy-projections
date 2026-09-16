"""M3 shadow board → Role 2 compare: matching, leakage, live-data blocker."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.projection.weekly_eval.cli import main
from src.projection.weekly_eval.errors import (
    OutcomeFeatureLeakageError,
    PostKickoffSnapshotError,
    Role3BlendForbiddenError,
)
from src.projection.weekly_latent.constants import DEFAULT_VEGAS_PROPS_M3_REL
from src.projection.weekly_latent.run import run_milestone3
from src.projection.weekly_latent.vegas_hook import (
    compare_m3_to_vegas_props,
    export_role2_board,
)

ROOT = Path(__file__).resolve().parents[2]
M3_FIXTURE = ROOT / DEFAULT_VEGAS_PROPS_M3_REL


def test_m3_dry_run_role2_compare_matches_timestamped_fixture(tmp_path: Path):
    result = run_milestone3(dry_run=True, output_dir=tmp_path, run_backtest=False)
    compare = result.vegas_compare
    assert compare["role"] == "evaluation_comparator"
    assert compare["promoting"] is False
    assert compare["role3_blend"] is False
    assert compare["gate_verdict"] in {"not_promoting", "not promoting"}
    assert compare["harness"] == "weekly_eval"
    assert int(compare["n_matched"]) > 0
    assert compare["metrics"]["mae_model_vs_market"] is not None
    assert compare["leakage"]["post_kickoff_rejected"] is True
    assert compare["leakage"]["as_of_required"] is True
    match = compare["match"]
    assert match["join_keys"] == ["player_id", "season", "week", "market"]
    assert int(match["n_snapshot_unmatched"]) >= 0
    board_path = tmp_path / "shadow_board_role2.csv"
    assert board_path.is_file()
    board = pd.read_csv(board_path)
    assert {"player_id", "season", "week", "market", "model_mean"} <= set(board.columns)
    assert "actual" not in board.columns
    assert "team_attempts" not in board.columns


def test_m3_dry_run_compare_rejects_post_kickoff_snapshot(tmp_path: Path):
    result = run_milestone3(dry_run=True, output_dir=tmp_path, run_backtest=False)
    board = export_role2_board(result.tables.player_weeks)
    leaked = M3_FIXTURE.read_text(encoding="utf-8").replace(
        "2026-09-09T18:00:00+00:00",
        "2026-09-11T18:00:00+00:00",
    )
    bad = tmp_path / "leaked_props.csv"
    bad.write_text(leaked, encoding="utf-8")
    with pytest.raises(PostKickoffSnapshotError):
        compare_m3_to_vegas_props(
            board=board,
            snapshots_path=bad,
            dry_run=False,
            repo_root=ROOT,
        )


def test_m3_compare_rejects_role3_blend_and_outcome_columns(tmp_path: Path):
    result = run_milestone3(dry_run=True, output_dir=tmp_path, run_backtest=False)
    board = export_role2_board(result.tables.player_weeks)
    dirty = board.copy()
    dirty["fantasy_points"] = 12.0
    with pytest.raises((OutcomeFeatureLeakageError, ValueError), match="fantasy_points"):
        compare_m3_to_vegas_props(
            board=dirty,
            snapshots_path=M3_FIXTURE,
            dry_run=False,
            repo_root=ROOT,
        )
    with pytest.raises((ValueError, Role3BlendForbiddenError), match="Role 3|blend"):
        compare_m3_to_vegas_props(
            board=board,
            snapshots_path=M3_FIXTURE,
            dry_run=False,
            blend_weights={"vegas": 0.4},
            repo_root=ROOT,
        )


def test_full_board_without_live_props_is_explicit_blocker_not_silent_zero():
    board = pd.DataFrame(
        [
            {
                "player_id": "00-0034857",
                "season": 2026,
                "week": 1,
                "market": "pass_yards",
                "model_mean": 268.0,
            }
        ]
    )
    compare = compare_m3_to_vegas_props(
        board=board,
        snapshots_path=None,
        dry_run=False,
        repo_root=ROOT,
    )
    assert compare["promoting"] is False
    assert compare["gate_verdict"] in {"not_promoting", "not promoting"}
    assert compare["n_matched"] == 0
    assert compare["harness"] == "missing_live_snapshots"
    blocker = compare["live_data_blocker"]
    assert blocker["status"] == "blocked"
    assert "player_id" in str(blocker["must_supply"]).lower()
    assert "as_of" in str(blocker["must_supply"]).lower()
    assert "kickoff_at" in str(blocker["must_supply"]).lower()
    assert compare.get("snapshot_source") != "m3_synthetic_fixture"


def test_mismatched_ids_report_unmatched_sample():
    board = pd.DataFrame(
        [
            {
                "player_id": "00-0034857",
                "season": 2026,
                "week": 1,
                "market": "pass_yards",
                "model_mean": 268.0,
            }
        ]
    )
    compare = compare_m3_to_vegas_props(
        board=board,
        snapshots_path=M3_FIXTURE,
        dry_run=False,
        repo_root=ROOT,
    )
    assert compare["n_matched"] == 0
    match = compare["match"]
    assert int(match["n_board_unmatched"]) == 1
    assert int(match["n_snapshot_unmatched"]) >= 1
    sample = match["unmatched_snapshot_ids_sample"]
    assert any(pid in {"p-qb", "p-rb", "q-qb", "p-wr"} for pid in sample)


def test_cli_m3_dry_run_writes_matched_role2_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    out_dir = tmp_path / "shadow_vegas_props_compare"
    monkeypatch.delenv("APP_PROJECTION_SOURCE", raising=False)
    code = main(["--m3-dry-run", "--output", str(out_dir)])
    assert code == 0
    payload = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["role"] == "evaluation_comparator"
    assert payload["promoting"] is False
    assert payload["gate_verdict"] == "not_promoting"
    assert payload["source"] == "m3_dry_run"
    assert int(payload["n_matched"]) > 0
    assert payload["harness"] == "weekly_eval"
    assert "APP_PROJECTION_SOURCE" not in payload
