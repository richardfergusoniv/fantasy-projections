"""Focused tests for production-safe QB evaluation safeguards.

Infrastructure only — does not activate H3/H4 model candidates or touch sealed
release artifacts / active pointers.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.projection.qb_eval_safeguards.archetype import classify_qb_archetype_safe
from src.projection.qb_eval_safeguards.composition_contract import (
    assert_availability_applied_once,
    compose_season_opportunity,
    detect_double_availability,
)
from src.projection.qb_eval_safeguards.conservation import (
    assert_team_qb_room_conserved,
    assert_team_volume_conserved,
)
from src.projection.qb_eval_safeguards.portable_contract import (
    PREDICTION_COLUMNS,
    SCHEMA_VERSION,
    ReconciliationSkipped,
    assert_no_label_leakage,
    leakage_audit,
    resolve_reconciliation_source,
    validate_portable_fixture,
)
from src.projection.qb_eval_safeguards.projections_db import (
    ProjectionsDbUnusable,
    projections_db_status,
    require_usable_projections_db,
)
from src.projection.qb_eval_safeguards.role_allocation import (
    IncompleteQbRoom,
    allocate_team_expected_starts,
    assert_backups_do_not_inherit_starter_volume,
    expected_availability,
    role_from_preseason,
)
from src.projection.qb_repair.rate_prior import classify_qb_archetype
from src.projection.transitions import SEASON_GAMES

REPO = Path(__file__).resolve().parents[1]
ACTIVE_POINTER = REPO / "draft_assistant" / "data" / "active_release_2026.json"
SEALED_DIR = REPO / "draft_assistant" / "data" / "releases" / "v2_baseline_20260830"


def _synthetic_fixture_frame() -> pd.DataFrame:
    rows = []
    for team, qb1, bu in (("AAA", "qb1a", "bu1a"), ("BBB", "qb1b", "bu1b")):
        rows.append(
            {
                "prediction_season": 2024,
                "prediction_cutoff": "2024-08-01",
                "team": team,
                "player_id": qb1,
                "display_name": qb1,
                "preseason_depth_tier": 1.0,
                "preseason_role": "starter",
                "is_rookie_at_cutoff": False,
                "prior_active_starts_sum": 32.0,
                "prior_active_starts_mean": 16.0,
                "prior_partial_exits_sum": 1.0,
                "prior_player_attempts_per_active": 34.0,
                "prior_player_carries_per_active": 3.0,
                "prior_team_pass_attempts": 580.0,
                "prior_team_qb_carries": 80.0,
                "pred_team_pass_attempts_pg": 580.0 / 17.0,
                "pred_team_qb_carries_pg": 80.0 / 17.0,
                "destination_team_at_cutoff": team,
                "actual_starts": 15.0,
                "actual_attempts": 500.0,
                "actual_carries": 40.0,
                "actual_passing_yards": 3500.0,
                "actual_rushing_yards": 180.0,
                "actual_passing_tds": 25.0,
                "actual_rushing_tds": 2.0,
                "actual_points": 280.0,
                "sealed_model_points_end_to_end": 270.0,
                "sealed_projected_games": 16.0,
                "source_weekly": "synthetic",
                "source_eval": "synthetic",
                "source_active_rates": "synthetic",
                "schema_version": SCHEMA_VERSION,
            }
        )
        rows.append(
            {
                "prediction_season": 2024,
                "prediction_cutoff": "2024-08-01",
                "team": team,
                "player_id": bu,
                "display_name": bu,
                "preseason_depth_tier": 2.0,
                "preseason_role": "backup",
                "is_rookie_at_cutoff": False,
                "prior_active_starts_sum": 4.0,
                "prior_active_starts_mean": 2.0,
                "prior_partial_exits_sum": 0.0,
                "prior_player_attempts_per_active": 38.0,
                "prior_player_carries_per_active": 1.0,
                "prior_team_pass_attempts": 580.0,
                "prior_team_qb_carries": 80.0,
                "pred_team_pass_attempts_pg": 580.0 / 17.0,
                "pred_team_qb_carries_pg": 80.0 / 17.0,
                "destination_team_at_cutoff": team,
                "actual_starts": 2.0,
                "actual_attempts": 40.0,
                "actual_carries": 2.0,
                "actual_passing_yards": 250.0,
                "actual_rushing_yards": 5.0,
                "actual_passing_tds": 1.0,
                "actual_rushing_tds": 0.0,
                "actual_points": 20.0,
                "sealed_model_points_end_to_end": 18.0,
                "sealed_projected_games": 2.0,
                "source_weekly": "synthetic",
                "source_eval": "synthetic",
                "source_active_rates": "synthetic",
                "schema_version": SCHEMA_VERSION,
            }
        )
    return pd.DataFrame(rows)


def test_projections_db_placeholder_fails_fast(tmp_path):
    empty = tmp_path / "projections.db"
    empty.write_bytes(b"")
    status = projections_db_status(empty)
    assert status["placeholder"] is True
    assert status["usable"] is False
    with pytest.raises(ProjectionsDbUnusable, match="zero bytes"):
        require_usable_projections_db(empty)
    missing = tmp_path / "nope.db"
    with pytest.raises(ProjectionsDbUnusable):
        require_usable_projections_db(missing)


def test_resolve_reconciliation_refuses_silent_skip(tmp_path):
    db = tmp_path / "projections.db"
    db.write_bytes(b"")
    missing_fixture = tmp_path / "absent.parquet"
    with pytest.raises(ReconciliationSkipped, match="Refusing to skip"):
        resolve_reconciliation_source(
            require_reconciliation=True,
            fixture_path=missing_fixture,
            db_path=db,
        )


def test_resolve_reconciliation_accepts_validated_fixture(tmp_path):
    db = tmp_path / "projections.db"
    db.write_bytes(b"")
    fixture = tmp_path / "portable.parquet"
    frame = _synthetic_fixture_frame()
    frame.to_parquet(fixture, index=False)
    meta = resolve_reconciliation_source(
        require_reconciliation=True,
        fixture_path=fixture,
        db_path=db,
    )
    assert meta["source"] == "portable_fixture"
    assert meta["reconciliation_will_run"] is True
    validate_portable_fixture(frame)


def test_portable_fixture_rejects_schema_drift_and_leakage():
    frame = _synthetic_fixture_frame()
    bad = frame.copy()
    bad["schema_version"] = "qb_h3_reconcile_contract_v1"
    with pytest.raises(ReconciliationSkipped, match="schema_version"):
        validate_portable_fixture(bad)

    audit = leakage_audit(frame, feature_columns=["actual_starts"])
    assert audit["ok"] is False
    with pytest.raises(AssertionError, match="label leakage"):
        assert_no_label_leakage(frame, ["actual_starts"])


def test_role_allocation_backup_cannot_inherit_starter_volume():
    history = pd.DataFrame(
        [
            {"player_id": "qb1", "season": 2022, "active_starts": 16, "partial_exit_rate": 0.05},
            {"player_id": "backup", "season": 2022, "active_starts": 4, "partial_exit_rate": 0.1},
        ]
    )
    room = pd.DataFrame(
        [
            {"player_id": "qb1", "preseason_depth_tier": 1.0, "is_rookie_at_cutoff": False},
            {
                "player_id": "backup",
                "preseason_depth_tier": 2.0,
                "is_rookie_at_cutoff": False,
                "attempts_per_active": 38.0,
            },
        ]
    )
    out = allocate_team_expected_starts(history=history, target_season=2023, room=room)
    qb1 = out[out.player_id == "qb1"].iloc[0]
    bu = out[out.player_id == "backup"].iloc[0]
    assert bool(qb1["is_qb1"])
    assert not bool(bu["is_qb1"])
    assert float(bu["allocated_expected_starts"]) < float(qb1["allocated_expected_starts"])
    assert float(bu["allocated_expected_starts"]) < 7.0
    assert float(out["allocated_expected_starts"].sum()) == pytest.approx(SEASON_GAMES, abs=1e-6)
    assert_team_qb_room_conserved(out.assign(team="ZZ"))
    assert not assert_backups_do_not_inherit_starter_volume(out)


def test_incomplete_room_fails_explicitly():
    with pytest.raises(IncompleteQbRoom):
        allocate_team_expected_starts(
            history=pd.DataFrame(columns=["player_id", "season", "active_starts"]),
            target_season=2023,
            room=pd.DataFrame(),
        )


def test_future_season_cannot_enter_availability_features():
    history = pd.DataFrame(
        [
            {"player_id": "qb1", "season": 2022, "active_starts": 16, "partial_exit_rate": 0.05},
            {"player_id": "qb1", "season": 2024, "active_starts": 17, "partial_exit_rate": 0.0},
        ]
    )
    avail = expected_availability(history, player_id="qb1", target_season=2023)
    assert max(avail["input_seasons"] or [0]) < 2023
    assert 2024 not in avail["input_seasons"]


def test_role_from_preseason_uses_cutoff_roles():
    assert role_from_preseason(depth_tier=1, is_rookie=True) == "rookie_starter"
    assert role_from_preseason(depth_tier=2, is_rookie=True) == "rookie_backup"
    assert role_from_preseason(depth_tier=2, is_rookie=False) == "backup"
    assert role_from_preseason(depth_tier=5, is_rookie=False) == "package"


def test_availability_applied_once_and_detects_double():
    opp = compose_season_opportunity(
        attempts_per_active=35.0,
        carries_per_active=4.0,
        expected_active_starts=15.0,
        partial_exit_rate=0.0,
    )
    assert_availability_applied_once(35.0, 15.0, opp.season_attempts)
    once = detect_double_availability(35.0, 15.0, 35.0 * 15.0)
    assert once["matches_once"] is True
    doubled = detect_double_availability(
        35.0, 15.0, 35.0 * 15.0 * (15.0 / SEASON_GAMES)
    )
    assert doubled["matches_double_sched"] is True


def test_team_pass_and_rush_conservation():
    report = assert_team_volume_conserved(
        team="ZZ",
        realized_pass_attempts=600.0,
        target_pass_attempts=600.0,
        realized_qb_carries=85.0,
        target_qb_carries=85.0,
    )
    assert report["ok"] is True
    with pytest.raises(AssertionError, match="conservation failed"):
        assert_team_volume_conserved(
            team="ZZ",
            realized_pass_attempts=500.0,
            target_pass_attempts=600.0,
            realized_qb_carries=85.0,
            target_qb_carries=85.0,
        )


def test_null_designed_is_not_pocket():
    hist = pd.DataFrame(
        [
            {
                "player_id": "x",
                "season": yr,
                "active_starts": 15,
                "carries_per_active": 2.0,
                "designed_carries_per_active": None,
                "scramble_per_dropback": None,
            }
            for yr in (2019, 2020, 2021, 2022)
        ]
    )
    meta = classify_qb_archetype_safe(hist, player_id="x", target_season=2023)
    assert meta["archetype"] == "insufficient_history"
    assert meta["reason"] == "missing_designed_or_scramble_not_pocket"


def test_observed_pocket_requires_designed_and_scramble():
    hist = pd.DataFrame(
        [
            {
                "player_id": "p",
                "season": yr,
                "active_starts": 16,
                "carries_per_active": 2.0,
                "designed_carries_per_active": 1.0,
                "scramble_per_dropback": 0.02,
            }
            for yr in (2020, 2021, 2022)
        ]
    )
    assert (
        classify_qb_archetype_safe(hist, player_id="p", target_season=2023)["archetype"]
        == "pocket_passer"
    )


def test_mobile_from_carries_when_designed_missing():
    hist = pd.DataFrame(
        [
            {
                "player_id": "m",
                "season": yr,
                "active_starts": 16,
                "carries_per_active": 8.0,
                "designed_carries_per_active": None,
                "scramble_per_dropback": None,
            }
            for yr in (2020, 2021, 2022)
        ]
    )
    meta = classify_qb_archetype_safe(hist, player_id="m", target_season=2023)
    assert meta["archetype"] == "mobile_scrambler"


def test_qb_repair_classifier_null_designed_not_pocket():
    rows = []
    for season in (2022, 2023, 2024):
        rows.append(
            {
                "player_id": "sparse",
                "season": season,
                "games": 16,
                "attempts": 560,
                "completions": 350,
                "passing_yards": 3800,
                "passing_tds": 25,
                "interceptions": 10,
                "carries": 32,
                "rushing_yards": 100,
                "rushing_tds": 1,
                "designed_carries": None,
                "scramble_carries": None,
            }
        )
    hist = pd.DataFrame(rows)
    assert classify_qb_archetype(hist, "sparse", target_season=2025) == "insufficient_history"


def test_missing_identity_archetype():
    meta = classify_qb_archetype_safe(pd.DataFrame(), player_id="", target_season=2023)
    assert meta["archetype"] == "missing_identity"


def test_non_qb_and_release_artifacts_unchanged_by_safeguards_package():
    # Safeguards must not rewrite sealed release files.
    assert ACTIVE_POINTER.exists()
    before = ACTIVE_POINTER.read_bytes()
    # Touching the package imports must not mutate pointer/sealed artifacts.
    from src.projection import qb_eval_safeguards as _pkg  # noqa: F401

    assert ACTIVE_POINTER.read_bytes() == before
    assert SEALED_DIR.exists()
    # Prediction columns must never include actual_* labels.
    assert not any(c.startswith("actual_") for c in PREDICTION_COLUMNS)
