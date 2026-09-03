"""H3 composition-contract and archetype fix tests."""
from __future__ import annotations

import pytest

from src.projection.qb_h3.composition_contract import (
    allocate_starter_backup_season,
    assert_availability_applied_once,
    compose_season_opportunity,
    detect_double_availability,
)
from src.projection.qb_h3.archetype import classify_archetype_h3
import pandas as pd


def test_availability_identity_once():
    opp = compose_season_opportunity(
        attempts_per_active=37.0, carries_per_active=3.0, expected_active_starts=13.0, partial_exit_rate=0.0
    )
    assert_availability_applied_once(37.0, 13.0, opp.season_attempts)
    assert opp.avail_adj_attempts_per_sched_game == pytest.approx(37.0 * 13.0 / 17.0)


def test_double_availability_detected():
    once = detect_double_availability(37.0, 13.0, 37.0 * 13.0)
    assert once["matches_once"] is True
    assert once["matches_double_17"] is False
    double = detect_double_availability(37.0, 13.0, 37.0 * 13.0 * 13.0 / 17.0)
    assert double["matches_double_17"] is True


def test_assert_raises_on_double():
    with pytest.raises(AssertionError):
        assert_availability_applied_once(37.0, 13.0, 37.0 * 13.0 * 13.0 / 17.0)


def test_backup_cannot_reduce_starter_active_rate():
    out = allocate_starter_backup_season(
        team_season_attempts=600.0,
        starter_attempts_per_active=37.0,
        starter_expected_starts=15.0,
        backup_attempts_per_active=30.0,
    )
    assert out["starter_attempts_per_active_preserved"] == 37.0
    assert out["starter_season_attempts"] == pytest.approx(37.0 * 15.0)
    assert out["conserved_total"] == pytest.approx(600.0)
    assert out["backup_season_attempts"] == pytest.approx(600.0 - 37.0 * 15.0)


def test_partial_exit_reduces_season_but_identity_holds_on_effective():
    opp = compose_season_opportunity(
        attempts_per_active=37.0,
        carries_per_active=3.0,
        expected_active_starts=14.0,
        partial_exit_rate=0.2,
    )
    effective = opp.season_attempts / 37.0
    assert_availability_applied_once(37.0, effective, opp.season_attempts)
    assert opp.season_attempts < 37.0 * 14.0
    da = detect_double_availability(37.0, effective, opp.season_attempts)
    assert da["matches_once"] is True
    assert da["matches_double_17"] is False


def test_h3_archetype_does_not_pocket_on_null_designed():
    hist = pd.DataFrame(
        [
            {"player_id": "m", "season": 2020, "active_starts": 16, "carries_per_active": 10.0,
             "designed_carries_per_active": None, "scramble_per_dropback": None,
             "rushing_yards_per_active": 50.0, "rushing_tds_per_active": 0.3,
             "designed_ypc": None, "scramble_ypa": None},
            {"player_id": "m", "season": 2021, "active_starts": 16, "carries_per_active": 9.5,
             "designed_carries_per_active": None, "scramble_per_dropback": None,
             "rushing_yards_per_active": 48.0, "rushing_tds_per_active": 0.3,
             "designed_ypc": None, "scramble_ypa": None},
            {"player_id": "m", "season": 2022, "active_starts": 12, "carries_per_active": 9.3,
             "designed_carries_per_active": None, "scramble_per_dropback": None,
             "rushing_yards_per_active": 55.0, "rushing_tds_per_active": 0.3,
             "designed_ypc": None, "scramble_ypa": None},
        ]
    )
    meta = classify_archetype_h3(hist, player_id="m", target_season=2023)
    assert meta["archetype"] == "mobile_scrambler"
