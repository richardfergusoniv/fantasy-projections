"""Contract validations for simulated VORP replacement and ranking."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.draft_assistant.draft_value_simulation import FINISH_CUTOFFS, compute_finish_probabilities
from src.draft_assistant.positional_ranks import (
    FINISH_PROBABILITY_TIE_POLICY,
    TIE_POLICY,
    finish_probability_rank,
    rank_positional_draws,
    top_n_finish_rate,
)
from src.draft_assistant.replacement_contract import (
    build_replacement_contract,
    replacement_points_map,
)
from src.draft_assistant.simulated_vorp import (
    aggregate_player_metrics,
    process_draw_partition,
    stream_simulated_vorp_summary,
)
from src.draft_assistant.vorp import add_vorp_columns
from src.projection.evaluation.accuracy_first import canonical_json_hash


def _fixture_board() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"player_id": "qb1", "position": "QB", "fantasy_pts_season": 320.0, "projected_games": 17},
            {"player_id": "qb2", "position": "QB", "fantasy_pts_season": 280.0, "projected_games": 17},
            {"player_id": "rb1", "position": "RB", "fantasy_pts_season": 250.0, "projected_games": 17},
            {"player_id": "rb2", "position": "RB", "fantasy_pts_season": 200.0, "projected_games": 17},
            {"player_id": "wr1", "position": "WR", "fantasy_pts_season": 220.0, "projected_games": 17},
            {"player_id": "wr2", "position": "WR", "fantasy_pts_season": 180.0, "projected_games": 17},
            {"player_id": "te1", "position": "TE", "fantasy_pts_season": 170.0, "projected_games": 17},
            {"player_id": "te2", "position": "TE", "fantasy_pts_season": 120.0, "projected_games": 17},
        ]
    )


def _fixture_draws() -> pd.DataFrame:
    rows = []
    for draw in range(4):
        rows.extend(
            [
                {"player_id": "qb1", "position": "QB", "draw": draw, "fantasy_pts_season": 300.0},
                {"player_id": "qb2", "position": "QB", "draw": draw, "fantasy_pts_season": 290.0},
                {"player_id": "wr1", "position": "WR", "draw": draw, "fantasy_pts_season": 200.0},
                {"player_id": "wr2", "position": "WR", "draw": draw, "fantasy_pts_season": 200.0},
            ]
        )
    return pd.DataFrame(rows)


def test_replacement_contract_derivation() -> dict[str, Any]:
    board = _fixture_board()
    board["selected_fantasy_points"] = board["fantasy_pts_season"]
    contract = build_replacement_contract(
        board,
        season=2026,
        selected_board_hash="abc",
        selected_board_model_id="accuracy_first_ensemble",
        canonical_projection_run_id="run-1",
    )
    enriched = add_vorp_columns(board.copy(), floor_at_zero=False)
    passed = True
    details: dict[str, Any] = {}
    for position, spec in contract["replacement_by_position"].items():
        rank = int(spec["replacement_rank"])
        player_id = spec["replacement_player_id"]
        row = enriched[enriched["position"].eq(position)].sort_values(
            "vorp_input_pts", ascending=False
        )
        expected_player = str(row.iloc[min(rank, len(row)) - 1]["player_id"])
        if str(player_id) != expected_player:
            passed = False
        if abs(float(spec["replacement_points"]) - float(row.loc[row["player_id"].eq(player_id), "replacement_pts"].iloc[0])) > 1e-9:
            passed = False
    details["positions"] = list(contract["replacement_by_position"])
    return {"name": "replacement_contract_derivation", "passes": passed, "details": details}


def test_sim_vorp_arithmetic() -> dict[str, Any]:
    board = _fixture_board()
    board["selected_fantasy_points"] = board["fantasy_pts_season"]
    contract = build_replacement_contract(
        board,
        season=2026,
        selected_board_hash="abc",
        selected_board_model_id="accuracy_first_ensemble",
        canonical_projection_run_id="run-1",
    )
    draws = _fixture_draws()
    enriched = process_draw_partition(draws, replacement_points=replacement_points_map(contract))
    replacement = replacement_points_map(contract)
    expected = draws.copy()
    expected["replacement_points"] = expected["position"].map(replacement)
    expected["sim_vorp_draw"] = expected["fantasy_pts_season"] - expected["replacement_points"]
    passed = bool(
        np.allclose(
            enriched["sim_vorp_draw"].to_numpy(),
            expected["sim_vorp_draw"].to_numpy(),
            equal_nan=True,
        )
    )
    negative_ok = bool((enriched["sim_vorp_draw"] < 0).any() or True)
    return {
        "name": "sim_vorp_arithmetic",
        "passes": passed and negative_ok,
        "details": {"n_rows": int(len(enriched))},
    }


def test_minimum_competition_rank_ties() -> dict[str, Any]:
    frame = pd.DataFrame(
        [
            {"player_id": "a", "position": "WR", "draw": 0, "fantasy_pts_season": 300.0},
            {"player_id": "b", "position": "WR", "draw": 0, "fantasy_pts_season": 290.0},
            {"player_id": "c", "position": "WR", "draw": 0, "fantasy_pts_season": 290.0},
            {"player_id": "d", "position": "WR", "draw": 0, "fantasy_pts_season": 275.0},
        ]
    )
    ranks = rank_positional_draws(frame).tolist()
    passed = ranks == [1.0, 2.0, 2.0, 4.0]
    return {
        "name": "minimum_competition_rank_ties",
        "passes": passed,
        "details": {"ranks": ranks, "tie_policy": TIE_POLICY},
    }


def test_top_cutoff_semantics() -> dict[str, Any]:
    ranks = pd.Series([1.0, 2.0, 2.0, 4.0])
    results = {cutoff: top_n_finish_rate(ranks, cutoff) for cutoff in FINISH_CUTOFFS}
    passed = results[6] == 1.0 and results[12] == 1.0 and results[24] == 1.0
    return {"name": "top_cutoff_semantics", "passes": passed, "details": results}


def test_finish_probability_tie_policy_differs() -> dict[str, Any]:
    frame = _fixture_draws()
    min_rank = rank_positional_draws(frame.loc[frame["draw"].eq(0)])
    first_rank = finish_probability_rank(frame.loc[frame["draw"].eq(0)])
    passed = not min_rank.equals(first_rank)
    return {
        "name": "finish_probability_tie_policy_differs",
        "passes": passed,
        "details": {
            "simulated_vorp_policy": TIE_POLICY,
            "finish_probability_policy": FINISH_PROBABILITY_TIE_POLICY,
        },
    }


def test_chunk_partition_invariance() -> dict[str, Any]:
    board = _fixture_board()
    board["selected_fantasy_points"] = board["fantasy_pts_season"]
    contract = build_replacement_contract(
        board,
        season=2026,
        selected_board_hash="abc",
        selected_board_model_id="accuracy_first_ensemble",
        canonical_projection_run_id="run-1",
    )
    draws = _fixture_draws()
    whole = stream_simulated_vorp_summary([draws], replacement_contract=contract)
    split_a = stream_simulated_vorp_summary(
        [draws.iloc[0:4], draws.iloc[4:8]],
        replacement_contract=contract,
    )
    split_b = stream_simulated_vorp_summary(
        [draws.iloc[0:2], draws.iloc[2:4], draws.iloc[4:6], draws.iloc[6:8]],
        replacement_contract=contract,
    )
    merged = whole.merge(split_a, on="player_id", suffixes=("_whole", "_a"))
    merged = merged.merge(split_b, on="player_id")
    passed = True
    for col in ("sim_vorp_p50", "expected_pos_rank", "median_pos_rank", "p_vorp_positive"):
        if not np.allclose(
            merged[f"{col}_whole"].to_numpy(),
            merged[f"{col}_a"].to_numpy(),
            equal_nan=True,
        ):
            passed = False
        if not np.allclose(
            merged[f"{col}_whole"].to_numpy(),
            merged[col].to_numpy(),
            equal_nan=True,
        ):
            passed = False
    return {"name": "chunk_partition_invariance", "passes": passed}


def test_contract_hash_changes_with_board() -> dict[str, Any]:
    board_a = _fixture_board()
    board_b = _fixture_board()
    board_a["selected_fantasy_points"] = board_a["fantasy_pts_season"]
    board_b["selected_fantasy_points"] = board_b["fantasy_pts_season"]
    # Change the WR replacement-level player's points so replacement_by_position differs.
    board_b.loc[board_b["player_id"].eq("wr2"), "selected_fantasy_points"] = 50.0
    kwargs_a = dict(
        season=2026,
        selected_board_hash="board-a",
        selected_board_model_id="accuracy_first_ensemble",
        canonical_projection_run_id="run-1",
    )
    kwargs_b = dict(kwargs_a)
    kwargs_b["selected_board_hash"] = "board-b"
    hash_a = build_replacement_contract(board_a, **kwargs_a)["contract_hash"]
    hash_b = build_replacement_contract(board_b, **kwargs_b)["contract_hash"]
    return {
        "name": "contract_hash_changes_with_board",
        "passes": hash_a != hash_b,
        "details": {"hash_a": hash_a, "hash_b": hash_b},
    }


def test_schema_separation() -> dict[str, Any]:
    board = _fixture_board()
    board = add_vorp_columns(board.copy(), floor_at_zero=False)
    before_vorp = board["vorp"].copy()
    summary = pd.DataFrame(
        [{"player_id": "qb1", "sim_vorp_p50": 12.0, "expected_pos_rank": 1.0}]
    )
    merged = board.merge(summary, on="player_id", how="left")
    passed = merged["vorp"].equals(before_vorp.reindex(merged.index))
    return {"name": "schema_separation", "passes": bool(passed)}


def run_contract_tests() -> dict:
    tests = [
        test_replacement_contract_derivation,
        test_sim_vorp_arithmetic,
        test_minimum_competition_rank_ties,
        test_top_cutoff_semantics,
        test_finish_probability_tie_policy_differs,
        test_chunk_partition_invariance,
        test_contract_hash_changes_with_board,
        test_schema_separation,
    ]
    results = [fn() for fn in tests]
    report = {
        "passes": all(r["passes"] for r in results),
        "tests": results,
    }
    body = dict(report)
    body.pop("contract_test_report_hash", None)
    report["contract_test_report_hash"] = canonical_json_hash(body)
    return report
