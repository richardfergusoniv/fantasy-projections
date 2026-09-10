"""H3 evaluation-infrastructure repair tests (frozen model spec)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.projection.qb_h3.archetype import classify_archetype_h3
from src.projection.qb_h3.portable_contract import (
    PREDICTION_COLUMNS,
    assert_no_label_leakage,
    leakage_audit,
    resolve_reconciliation_source,
)
from src.projection.qb_h3.projections_db import (
    ProjectionsDbUnusable,
    projections_db_status,
    require_usable_projections_db,
)
from src.projection.qb_h3.role_allocation import (
    allocate_team_expected_starts,
    assert_backups_do_not_inherit_starter_volume,
    role_from_preseason,
)
from src.projection.transitions import SEASON_GAMES

REPO = Path(__file__).resolve().parents[1]

# The six 2023 rows the frozen classifier mapped to pocket on null designed.
MISLABELED_2023 = {
    "Josh Allen": "00-0034857",
    "Jalen Hurts": "00-0036389",
    "Lamar Jackson": "00-0034796",
    "Justin Fields": "00-0036945",
    "Kyler Murray": "00-0035228",
    "Deshaun Watson": "00-0033537",
}

# 2023 targeted allocation players.
ALLOC_2023 = {
    "Jameis Winston": "00-0031503",
    "C.J. Beathard": "00-0033936",
    "Mike White": "00-0034401",
    "Cooper Rush": "00-0033662",
    "Kyle Allen": "00-0034577",
    "Sam Darnold": "00-0034869",
    "Kyler Murray": "00-0035228",
    "Ryan Tannehill": "00-0029701",
    "Justin Herbert": "00-0036355",
    "Jalen Hurts": "00-0036389",
}


def _dual_hist(pid: str, carries: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_id": pid,
                "season": yr,
                "active_starts": 15,
                "carries_per_active": carries,
                "designed_carries_per_active": None,
                "scramble_per_dropback": None,
                "rushing_yards_per_active": carries * 5.0,
                "rushing_tds_per_active": 0.3,
            }
            for yr in (2019, 2020, 2021, 2022)
        ]
    )


def test_null_designed_low_carries_is_insufficient_not_pocket():
    hist = _dual_hist("pocketish", 2.0)
    meta = classify_archetype_h3(hist, player_id="pocketish", target_season=2023)
    assert meta["archetype"] == "insufficient_history"


def test_pocket_requires_observed_designed_and_scramble():
    hist = pd.DataFrame(
        [
            {
                "player_id": "p",
                "season": yr,
                "active_starts": 16,
                "carries_per_active": 2.0,
                "designed_carries_per_active": 1.0,
                "scramble_per_dropback": 0.02,
                "rushing_yards_per_active": 8.0,
                "rushing_tds_per_active": 0.05,
            }
            for yr in (2020, 2021, 2022)
        ]
    )
    assert classify_archetype_h3(hist, player_id="p", target_season=2023)["archetype"] == "pocket_passer"


def _committed_history() -> pd.DataFrame:
    """Use the tracked active-rate table, not the gitignored weekly cache."""
    path = REPO / "output" / "qb_active_archetype" / "active_season_rates.parquet"
    hist = pd.read_parquet(path)
    hist["player_id"] = hist["player_id"].astype(str)
    return hist


@pytest.mark.parametrize("name,pid", list(MISLABELED_2023.items()))
def test_six_mislabeled_2023_not_pocket(name, pid):
    hist = _committed_history()
    meta = classify_archetype_h3(hist, player_id=pid, target_season=2023)
    assert meta["archetype"] != "pocket_passer", (name, meta)
    # Classified from available prior seasons (carries), not a future label.
    assert meta["input_seasons"]
    assert max(meta["input_seasons"]) < 2023
    if name in ("Lamar Jackson", "Jalen Hurts"):
        assert meta["archetype"] == "mobile_scrambler"
        assert (meta["features"].get("carries_per_active") or 0) >= 5.5


def test_merge_rush_splits_empty_left_does_not_raise():
    from src.projection.qb_active_archetype.active_rates import merge_rush_splits

    out = merge_rush_splits(pd.DataFrame())
    assert out.empty
    assert "designed_carries_per_active" in out.columns


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


def test_workspace_db_is_unusable_placeholder():
    status = projections_db_status()
    assert status["usable"] is False


def test_leakage_audit_rejects_actual_in_features():
    frame = pd.DataFrame([{"prediction_season": 2023, "actual_starts": 10}])
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
                # Strong historical rate must not grant starter starts.
                "attempts_per_active": 38.0,
            },
        ]
    )
    out = allocate_team_expected_starts(history=history, target_season=2023, room=room)
    qb1 = out[out.player_id == "qb1"].iloc[0]
    bu = out[out.player_id == "backup"].iloc[0]
    assert qb1["is_qb1"]
    assert not bu["is_qb1"]
    assert bu["allocated_expected_starts"] < qb1["allocated_expected_starts"]
    assert bu["allocated_expected_starts"] < 7.0
    assert out["allocated_expected_starts"].sum() == pytest.approx(SEASON_GAMES, abs=1e-6)
    assert not assert_backups_do_not_inherit_starter_volume(out)


def test_role_from_preseason_rookies():
    assert role_from_preseason(depth_tier=1, is_rookie=True) == "rookie_starter"
    assert role_from_preseason(depth_tier=2, is_rookie=True) == "rookie_backup"
    assert role_from_preseason(depth_tier=2, is_rookie=False) == "backup"


def test_2023_named_allocation_from_fixture():
    from src.projection.qb_h3.portable_contract import load_portable_fixture
    from src.projection.qb_h3.role_allocation import allocate_league_expected_starts

    fixture = load_portable_fixture()
    hist = _committed_history()
    room = fixture[fixture.prediction_season == 2023]
    allocated = allocate_league_expected_starts(
        history=hist, target_season=2023, rooms=room, team_col="team"
    )
    by_id = allocated.set_index("player_id")
    backups = [
        "00-0031503",  # Winston
        "00-0033936",  # Beathard
        "00-0034401",  # White
        "00-0033662",  # Rush
        "00-0034577",  # Kyle Allen
        "00-0034869",  # Darnold
    ]
    for pid in backups:
        row = by_id.loc[pid]
        assert not bool(row["is_qb1"]), pid
        assert float(row["allocated_expected_starts"]) < 7.0, (pid, row["allocated_expected_starts"])
    starters = ["00-0029701", "00-0036355", "00-0036389"]  # Tannehill, Herbert, Hurts
    for pid in starters:
        row = by_id.loc[pid]
        assert bool(row["is_qb1"]), pid
        assert float(row["allocated_expected_starts"]) >= 10.0, pid
    # Murray is depth 5 on ARI behind Dobbs — package residual, not starter volume.
    murray = by_id.loc["00-0035228"]
    assert not bool(murray["is_qb1"])
    assert float(murray["allocated_expected_starts"]) < 7.0
    # Hurts' active rate is not reduced by Mariota.
    hurts = by_id.loc["00-0036389"]
    assert hurts["preseason_role"] in ("starter", "rookie_starter")
