"""Conservation and leakage tests for weekly latent Milestone 1.

Does not train, promote, or touch League Value.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.projection.contracts import REPO_ROOT
from src.projection.shadow.forbidden import local_import_graph
from src.projection.weekly_latent.allocate import allocate_team_weeks, shrunk_opponent_mult
from src.projection.weekly_latent.constants import (
    AWAY_MULT,
    FORBIDDEN_SAME_WEEK_TRAINING_FEATURES,
    HOME_MULT,
    NEUTRAL_MULT,
    opponent_shrinkage_lambda,
)
from src.projection.weekly_latent.run import production_fingerprint, run_milestone1
from src.projection.weekly_latent.schedule import explode_team_weeks

ROOT = Path(REPO_ROOT)
SEALED = (
    ROOT
    / "draft_assistant"
    / "data"
    / "releases"
    / "v2_baseline_20260830"
    / "projections_2026.csv"
)
SCHEDULE = ROOT / "src" / "projection" / "weekly_latent" / "fixtures" / "nfl_schedules_2026_reg.csv"


def _manual_team_weeks(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["season"] = 2026
    frame["A_team_w"] = (1 - frame["is_bye"]).astype(float)
    if "is_neutral" not in frame.columns:
        frame["is_neutral"] = 0
    frame["is_home_for_mult"] = (
        (frame["is_home"] == 1) & (frame["is_neutral"] == 0) & (frame["is_bye"] == 0)
    ).astype(int)
    frame["shrinkage_lambda"] = frame["week"].map(opponent_shrinkage_lambda)
    for col in ("roof", "surface", "stadium", "gameday", "rest_days"):
        if col not in frame.columns:
            frame[col] = None
    return frame


def test_horizon_shrinkage_week1_stronger_than_week16():
    assert opponent_shrinkage_lambda(1) == 1.0
    assert opponent_shrinkage_lambda(4) == 1.0
    assert opponent_shrinkage_lambda(15) == 0.10
    assert opponent_shrinkage_lambda(18) == 0.10
    tough = 0.80
    m1 = shrunk_opponent_mult(tough, opponent_shrinkage_lambda(1))
    m16 = shrunk_opponent_mult(tough, opponent_shrinkage_lambda(16))
    assert abs(m1 - 0.80) < 1e-12
    assert abs(m16 - (1.0 + 0.10 * (0.80 - 1.0))) < 1e-12
    assert abs(m1 - 1.0) > abs(m16 - 1.0)


def test_matchup_weights_renormalize_and_home_gets_more_than_away():
    team_weeks = _manual_team_weeks(
        [
            {
                "team": "AAA",
                "week": 1,
                "opponent": "BBB",
                "is_home": 1,
                "is_bye": 0,
                "is_neutral": 0,
            },
            {
                "team": "AAA",
                "week": 2,
                "opponent": "CCC",
                "is_home": 0,
                "is_bye": 0,
                "is_neutral": 0,
            },
        ]
    )
    volume = pd.DataFrame(
        {
            "team": ["AAA"],
            "team_pass_attempts": [170.0],
            "team_rush_attempts": [170.0],
            "team_passing_yards": [3400.0],
            "team_rushing_yards": [1700.0],
        }
    )
    out = allocate_team_weeks(team_weeks, volume)
    assert abs(out["team_pass_attempts"].sum() - 170.0) < 1e-9
    home = float(out.loc[out["week"].eq(1), "team_pass_attempts"].iloc[0])
    away = float(out.loc[out["week"].eq(2), "team_pass_attempts"].iloc[0])
    assert home > away
    expected_home = 170.0 * HOME_MULT / (HOME_MULT + AWAY_MULT)
    assert abs(home - expected_home) < 1e-9


def test_bye_week_is_zero_and_neutral_skips_home_boost():
    team_weeks = _manual_team_weeks(
        [
            {
                "team": "AAA",
                "week": 1,
                "opponent": "BBB",
                "is_home": 1,
                "is_bye": 0,
                "is_neutral": 1,
            },
            {
                "team": "AAA",
                "week": 2,
                "opponent": None,
                "is_home": 0,
                "is_bye": 1,
                "is_neutral": 0,
            },
            {
                "team": "AAA",
                "week": 3,
                "opponent": "CCC",
                "is_home": 0,
                "is_bye": 0,
                "is_neutral": 0,
            },
        ]
    )
    volume = pd.DataFrame(
        {
            "team": ["AAA"],
            "team_pass_attempts": [100.0],
            "team_rush_attempts": [100.0],
            "team_passing_yards": [1000.0],
            "team_rushing_yards": [500.0],
        }
    )
    out = allocate_team_weeks(team_weeks, volume)
    bye = out.loc[out["is_bye"].eq(1)]
    assert float(bye["team_pass_attempts"].sum()) == 0.0
    assert float(out["team_pass_attempts"].sum()) == pytest.approx(100.0)
    w1 = float(out.loc[out["week"].eq(1), "home_away_mult"].iloc[0])
    w3 = float(out.loc[out["week"].eq(3), "home_away_mult"].iloc[0])
    assert w1 == NEUTRAL_MULT
    assert w3 == AWAY_MULT


def test_tough_opponent_hurts_week1_more_than_week16_after_shrinkage():
    team_weeks = _manual_team_weeks(
        [
            {
                "team": "AAA",
                "week": 1,
                "opponent": "BBB",
                "is_home": 1,
                "is_bye": 0,
                "is_neutral": 0,
            },
            {
                "team": "AAA",
                "week": 16,
                "opponent": "BBB",
                "is_home": 1,
                "is_bye": 0,
                "is_neutral": 0,
            },
        ]
    )
    volume = pd.DataFrame(
        {
            "team": ["AAA"],
            "team_pass_attempts": [200.0],
            "team_rush_attempts": [200.0],
            "team_passing_yards": [2000.0],
            "team_rushing_yards": [1000.0],
        }
    )
    priors = pd.DataFrame(
        {"opponent": ["BBB"], "pass_factor": [0.80], "rush_factor": [0.80]}
    )
    out = allocate_team_weeks(team_weeks, volume, opponent_priors=priors)
    w1 = float(out.loc[out["week"].eq(1), "team_pass_attempts"].iloc[0])
    w16 = float(out.loc[out["week"].eq(16), "team_pass_attempts"].iloc[0])
    assert w1 + w16 == pytest.approx(200.0)
    assert w16 > w1


def test_dry_run_conservation_and_share_simplex(tmp_path):
    result = run_milestone1(dry_run=True, output_dir=tmp_path)
    assert result.conservation["passes"] is True
    shares = result.tables.shares
    named = shares[shares["stat"].eq("attempts")].groupby("team")["share"].sum()
    other = shares[shares["stat"].eq("attempts")].groupby("team")["other_share"].first()
    assert ((named + other) - 1.0).abs().max() < 1e-9
    bye_fp = result.tables.player_weeks.loc[
        result.tables.player_weeks["is_bye"].eq(1), "fantasy_points"
    ]
    assert float(bye_fp.abs().sum()) == 0.0
    assert result.summary["production_hash_drift"] == {}
    forbidden = FORBIDDEN_SAME_WEEK_TRAINING_FEATURES.intersection(
        result.tables.team_weeks.columns
    )
    assert forbidden == set()


def test_m1_does_not_import_promote_or_team_pass_rate_join():
    graph = local_import_graph(["src.projection.weekly_latent.run"])
    assert "src.projection.promote_release" not in graph
    assert "src.projection.release_bundle_publish" not in graph
    assert "src.projection.weekly.features.team_context" not in graph


def test_2026_fixture_has_32_teams_and_byes():
    sched = pd.read_csv(SCHEDULE)
    weeks = explode_team_weeks(sched, require_full_season=True)
    assert weeks["team"].nunique() == 32
    assert int(weeks.groupby("team")["is_bye"].sum().max()) == 1
    assert int(weeks.groupby("team")["active"].sum().min()) == 17
    intl = weeks[weeks["is_international"].eq(1)]
    assert not intl.empty


@pytest.mark.skipif(not SEALED.is_file(), reason="sealed projections CSV missing")
def test_sealed_board_m1_conservation(tmp_path):
    before = production_fingerprint(ROOT)
    result = run_milestone1(
        season=2026,
        projections_path=SEALED,
        schedule_path=SCHEDULE,
        output_dir=tmp_path,
    )
    assert result.conservation["passes"] is True
    assert result.summary["n_teams"] == 32
    assert result.tables.team_weeks.groupby("team")["team_pass_attempts"].sum().max() > 0
    after = production_fingerprint(ROOT)
    assert before == after
    assert result.summary["product_split"]["vegas"] == "external_weekly_benchmark"
    assert "replace Vegas weekly props" in result.summary["does_not"]
