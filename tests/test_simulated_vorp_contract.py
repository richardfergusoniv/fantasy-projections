"""Tests for simulated VORP replacement contract."""
from __future__ import annotations

import json

import pandas as pd

from src.draft_assistant.prepare import attach_draft_value_overlay
from src.draft_assistant.replacement_contract import (
    build_replacement_contract,
    replacement_points_map,
)
from src.draft_assistant.simulated_vorp import process_draw_partition
from src.projection.evaluation.finish_probability_gate import VERDICT_READY as FINISH_READY
from src.projection.evaluation.simulated_vorp_contract_tests import run_contract_tests
from src.draft_assistant.positional_ranks import rank_positional_draws, simulation_rank_metadata
from src.projection.evaluation.simulated_vorp_gate import (
    build_simulated_vorp_gate,
    validate_simulated_vorp_publication,
)


def test_replacement_contract_hash_stable_across_rebuilds():
    board = pd.DataFrame(
        [
            {"player_id": "wr1", "position": "WR", "fantasy_pts_season": 200.0, "projected_games": 17},
            {"player_id": "wr2", "position": "WR", "fantasy_pts_season": 50.0, "projected_games": 17},
        ]
    )
    board["selected_fantasy_points"] = board["fantasy_pts_season"]
    kwargs = dict(
        season=2026,
        selected_board_hash="abc",
        selected_board_model_id="accuracy_first_ensemble",
        canonical_projection_run_id="run-1",
    )
    first = build_replacement_contract(board, **kwargs)
    second = build_replacement_contract(board, **kwargs)
    assert first["contract_hash"] == second["contract_hash"]
    assert "generated_at" in first
    assert "generated_at" in second


def test_simulation_rank_metadata_documents_tie_split():
    meta = simulation_rank_metadata()
    finish = meta["finish_probability_fields"]
    sim_rank = meta["simulated_positional_rank_fields"]
    assert finish["tie_policy"] == "first_occurrence"
    assert sim_rank["tie_policy"] == "minimum_competition_rank"
    assert finish["tie_policy"] != sim_rank["tie_policy"]

    report = run_contract_tests()
    assert report["passes"] is True


def test_minimum_competition_rank_example():
    frame = pd.DataFrame(
        [
            {"player_id": "a", "position": "WR", "draw": 0, "fantasy_pts_season": 300.0},
            {"player_id": "b", "position": "WR", "draw": 0, "fantasy_pts_season": 290.0},
            {"player_id": "c", "position": "WR", "draw": 0, "fantasy_pts_season": 290.0},
            {"player_id": "d", "position": "WR", "draw": 0, "fantasy_pts_season": 275.0},
        ]
    )
    assert rank_positional_draws(frame).tolist() == [1.0, 2.0, 2.0, 4.0]


def test_sim_vorp_preserves_negative_values():
    board = pd.DataFrame(
        [
            {"player_id": "wr1", "position": "WR", "fantasy_pts_season": 200.0, "projected_games": 17},
            {"player_id": "wr2", "position": "WR", "fantasy_pts_season": 50.0, "projected_games": 17},
        ]
    )
    board["selected_fantasy_points"] = board["fantasy_pts_season"]
    contract = build_replacement_contract(
        board,
        season=2026,
        selected_board_hash="abc",
        selected_board_model_id="accuracy_first_ensemble",
        canonical_projection_run_id="run-1",
    )
    draws = pd.DataFrame(
        [{"player_id": "wr2", "position": "WR", "draw": 0, "fantasy_pts_season": 10.0}]
    )
    enriched = process_draw_partition(draws, replacement_points=replacement_points_map(contract))
    assert float(enriched["sim_vorp_draw"].iloc[0]) < 0.0


def test_gate_holds_without_finish_gate():
    board = pd.DataFrame(
        [{"player_id": "qb1", "position": "QB", "fantasy_pts_season": 300.0, "projected_games": 17}]
    )
    board["selected_fantasy_points"] = board["fantasy_pts_season"]
    contract = build_replacement_contract(
        board,
        season=2026,
        selected_board_hash="abc",
        selected_board_model_id="accuracy_first_ensemble",
        canonical_projection_run_id="run-1",
    )
    gate = build_simulated_vorp_gate(
        season=2026,
        manifest={"selected_board_hash": "abc", "selected_board_model_id": "accuracy_first_ensemble"},
        replacement_contract=contract,
        contract_tests=run_contract_tests(),
        finish_gate=None,
    )
    assert gate["publication_verdict"] == "hold"


def test_validate_sim_vorp_publication_holds_on_gate_hold():
    manifest = {
        "selected_board_hash": "abc",
        "selected_board_model_id": "accuracy_first_ensemble",
        "canonical_projection_run_id": "run-1",
        "transform_version": "v1_median_correction",
        "partition_hashes": [],
    }
    contract = {
        "selected_board_hash": "abc",
        "selected_board_model_id": "accuracy_first_ensemble",
        "canonical_projection_run_id": "run-1",
        "contract_hash": "deadbeef",
    }
    finish_gate = {"state": FINISH_READY, "publication_verdict": "pass"}
    vorp_gate = {
        "state": "hold",
        "publication_verdict": "hold",
        "manifest": {"transform_version": "v1_median_correction"},
        "replacement_contract": {"contract_hash": "deadbeef"},
    }
    ok, validation = validate_simulated_vorp_publication(
        manifest=manifest,
        finish_gate=finish_gate,
        replacement_contract=contract,
        vorp_gate=vorp_gate,
    )
    assert ok is False
    assert "simulated_vorp_gate_not_ready" in validation["failures"]
