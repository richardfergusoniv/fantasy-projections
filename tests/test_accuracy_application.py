"""Frozen accuracy application contract: no fit, no eligible-input fallback."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.projection.accuracy_application import (
    ApplicationContractError,
    apply_application_contract,
    build_application_contract,
    serialize_isotonic,
)
from sklearn.isotonic import IsotonicRegression


def _linear_curves() -> dict:
    curves = {}
    for position in ("QB", "RB", "WR", "TE"):
        model = IsotonicRegression(increasing=False, out_of_bounds="clip", y_min=0.0)
        model.fit(np.array([1.0, 120.0]), np.array([200.0, 50.0]))
        curves[position] = serialize_isotonic(model)
    return curves


def _contract(**overrides):
    fixture = {
        "player_id": "rb-star",
        "position": "RB",
        "inputs": {"v1_pred": 100.0, "v2_pred": 120.0, "adp": 1.0},
        "expected_treatment": "selected",
        "expected_points": None,
    }
    curves = _linear_curves()
    adp_points = float(np.interp(1.0, curves["RB"]["x_thresholds"], curves["RB"]["y_thresholds"]))
    fixture["expected_points"] = 0.1 * 100.0 + 0.3 * 120.0 + 0.6 * adp_points
    payload = dict(
        positions={
            "QB": {"arm": "incumbent", "weights": {"v1_pred": 0.4, "v2_pred": 0.6}},
            "RB": {"arm": "market_no_v3", "weights": {"v1_pred": 0.1, "v2_pred": 0.3, "adp_points": 0.6}},
            "WR": {"arm": "market_no_v3", "weights": {"v1_pred": 0.0, "v2_pred": 0.55, "adp_points": 0.45}},
            "TE": {"arm": "incumbent", "weights": {"v1_pred": 0.9, "v2_pred": 0.1}},
        },
        market_curves=curves,
        eligibility_ids=["rb-star", "qb-vet"],
        universe_ids=["rb-star", "qb-vet", "wr-deep"],
        reference_fixture=fixture,
        source_hashes={"ensemble_weights": "a" * 64},
        incumbent_weights={
            "QB": {"v1_pred": 0.4, "v2_pred": 0.6},
            "RB": {"v1_pred": 1.0, "v2_pred": 0.0},
            "WR": {"v1_pred": 0.0, "v2_pred": 1.0},
            "TE": {"v1_pred": 0.9, "v2_pred": 0.1}},
    )
    payload.update(overrides)
    return build_application_contract(**payload)


def test_reference_fixture_is_replayed_on_build():
    contract = _contract()
    assert contract["incumbent_fallback"] is False
    assert contract["contract_hash"]


def test_eligible_player_missing_inputs_is_fatal():
    contract = _contract()
    board = pd.DataFrame(
        [{"player_id": "rb-star", "position": "RB", "v1_pred": 100.0, "fantasy_pts_season": 100.0}]
    )
    with pytest.raises(ApplicationContractError, match="missing contract inputs"):
        apply_application_contract(board, contract, v2_by_id={}, adp_by_id={})


def test_v1_only_cannot_bypass_missing_eligible_inputs():
    contract = _contract()
    board = pd.DataFrame(
        [
            {"player_id": "rb-star", "position": "RB", "v1_pred": 100.0, "fantasy_pts_season": 100.0},
            {"player_id": "brand-new", "position": "WR", "v1_pred": 40.0, "fantasy_pts_season": 40.0},
        ]
    )
    with pytest.raises(ApplicationContractError, match="rb-star"):
        apply_application_contract(
            board,
            contract,
            v2_by_id={},
            adp_by_id={},
        )


def test_genuine_new_player_is_recorded_v1_only():
    contract = _contract()
    board = pd.DataFrame(
        [
            {
                "player_id": "rb-star",
                "position": "RB",
                "v1_pred": 100.0,
                "fantasy_pts_season": 100.0,
                "projected_games": 17.0,
            },
            {
                "player_id": "qb-vet",
                "position": "QB",
                "v1_pred": 200.0,
                "fantasy_pts_season": 200.0,
                "projected_games": 17.0,
            },
            {
                "player_id": "wr-deep",
                "position": "WR",
                "v1_pred": 80.0,
                "fantasy_pts_season": 80.0,
                "projected_games": 17.0,
            },
            {
                "player_id": "brand-new",
                "position": "RB",
                "v1_pred": 33.0,
                "fantasy_pts_season": 33.0,
                "projected_games": 17.0,
            },
        ]
    )
    applied, treatments = apply_application_contract(
        board,
        contract,
        v2_by_id={"rb-star": 120.0, "qb-vet": 180.0, "wr-deep": 90.0},
        adp_by_id={"rb-star": 1.0, "qb-vet": 10.0},
    )
    by_id = applied.set_index("player_id")
    assert by_id.loc["brand-new", "contract_treatment"] == "new_player_v1_only"
    assert by_id.loc["brand-new", "fantasy_pts_season"] == 33.0
    assert by_id.loc["rb-star", "contract_treatment"] == "selected"
    assert by_id.loc["qb-vet", "contract_treatment"] == "incumbent"
    assert by_id.loc["wr-deep", "contract_treatment"] == "incumbent"
    assert treatments["new_player_v1_only"]["count"] == 1
    assert "brand-new" in treatments["player_ids"]["new_player_v1_only"]
    assert by_id.loc["rb-star", "accuracy_ensemble_applied"]
    assert not by_id.loc["brand-new", "accuracy_ensemble_applied"]


def test_contract_rejects_incumbent_fallback_flag():
    contract = _contract()
    contract["incumbent_fallback"] = True
    with pytest.raises(ApplicationContractError, match="incumbent fallback"):
        apply_application_contract(
            pd.DataFrame([{"player_id": "rb-star", "position": "RB", "v1_pred": 100.0}]),
            contract,
            v2_by_id={"rb-star": 120.0},
            adp_by_id={"rb-star": 1.0},
        )
