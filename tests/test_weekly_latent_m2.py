"""Milestone 2 identities, leakage, and available_at tests.

Research/shadow only. Does not train M3, promote, or touch League Value.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.projection.contracts import REPO_ROOT
from src.projection.shadow.forbidden import local_import_graph
from src.projection.weekly_latent.allocate import allocate_team_weeks, allocate_team_weeks_m2
from src.projection.weekly_latent.backtest import (
    HISTORICAL_BOARD_AVAILABLE_AT,
    historical_features_for_week,
    historical_schedule_history,
    poison_same_week_volume,
    run_historical_schedule_backtest,
    run_m2_backtests,
    run_synthetic_rolling_origin,
)
from src.projection.weekly_latent.constants import (
    FORBIDDEN_MARKET_DRIVERS,
    FORBIDDEN_SAME_WEEK_TRAINING_FEATURES,
    M1_AVAILABLE_AT,
    PRIOR_SEASON_EPA_AVAILABLE_AT,
    opponent_shrinkage_lambda,
)
from src.projection.weekly_latent.environment import refuse_forbidden_m2_columns
from src.projection.weekly_latent.priors import assert_priors_are_as_of, build_as_of_priors
from src.projection.weekly_latent.run import production_fingerprint, run_milestone2

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
EPA_FIXTURE = (
    ROOT / "src" / "projection" / "weekly_latent" / "fixtures" / "opp_def_epa_prior_2025.csv"
)


def _manual_team_weeks(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["season"] = 2026
    frame["A_team_w"] = (1 - frame["is_bye"]).astype(float)
    if "is_neutral" not in frame.columns:
        frame["is_neutral"] = 0
    if "is_international" not in frame.columns:
        frame["is_international"] = 0
    if "rest_days" not in frame.columns:
        frame["rest_days"] = 7
    frame["is_home_for_mult"] = (
        (frame["is_home"] == 1) & (frame["is_neutral"] == 0) & (frame["is_bye"] == 0)
    ).astype(int)
    frame["shrinkage_lambda"] = frame["week"].map(opponent_shrinkage_lambda)
    for col in ("roof", "surface", "stadium", "gameday"):
        if col not in frame.columns:
            frame[col] = None
    return frame


def _volume() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "team": ["AAA"],
            "team_pass_attempts": [170.0],
            "team_rush_attempts": [170.0],
            "team_passing_yards": [3400.0],
            "team_rushing_yards": [1700.0],
        }
    )


def test_m2_season_mass_may_differ_from_sealed_while_m1_conserves():
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
    volume = _volume()
    priors = pd.DataFrame(
        {
            "opponent": ["BBB", "CCC"],
            "pass_factor": [0.80, 1.20],
            "rush_factor": [0.90, 1.10],
            "available_at": [PRIOR_SEASON_EPA_AVAILABLE_AT, PRIOR_SEASON_EPA_AVAILABLE_AT],
        }
    )
    m1 = allocate_team_weeks(team_weeks, volume, opponent_priors=priors)
    m2 = allocate_team_weeks_m2(team_weeks, volume, opponent_priors=priors)
    assert m1["team_pass_attempts"].sum() == pytest.approx(170.0)
    assert m2["team_pass_attempts"].sum() != pytest.approx(170.0)
    assert float(m2.loc[m2["week"].eq(1), "team_pass_attempts"].iloc[0]) < float(
        m2.loc[m2["week"].eq(2), "team_pass_attempts"].iloc[0]
    )
    # Identity: weekly = A * (sealed/n_active) * matchup.
    n_active = 2.0
    w1 = m2.loc[m2["week"].eq(1)].iloc[0]
    intended = 1.0 * (170.0 / n_active) * float(w1["pass_matchup_mult"])
    assert float(w1["team_pass_attempts"]) == pytest.approx(intended)


def test_m2_overwrites_available_at_when_prior_is_later_than_m1_stamp():
    team_weeks = _manual_team_weeks(
        [
            {
                "team": "AAA",
                "week": 3,
                "opponent": "BBB",
                "is_home": 1,
                "is_bye": 0,
                "is_neutral": 0,
            }
        ]
    )
    later = "2026-10-14T00:00:00+00:00"
    priors = pd.DataFrame(
        {
            "opponent": ["BBB"],
            "pass_factor": [1.05],
            "rush_factor": [0.95],
            "available_at": [later],
        }
    )
    m2 = allocate_team_weeks_m2(team_weeks, _volume(), opponent_priors=priors)
    assert (m2["available_at"] == later).all()
    assert later != M1_AVAILABLE_AT
    m1 = allocate_team_weeks(team_weeks, _volume(), opponent_priors=priors)
    assert (m1["available_at"] == M1_AVAILABLE_AT).all()


def test_m2_available_at_is_max_not_blind_m1_when_prior_is_earlier():
    team_weeks = _manual_team_weeks(
        [
            {
                "team": "AAA",
                "week": 1,
                "opponent": "BBB",
                "is_home": 1,
                "is_bye": 0,
                "is_neutral": 0,
            }
        ]
    )
    priors = pd.DataFrame(
        {
            "opponent": ["BBB"],
            "pass_factor": [1.0],
            "rush_factor": [1.0],
            "available_at": [PRIOR_SEASON_EPA_AVAILABLE_AT],
        }
    )
    m2 = allocate_team_weeks_m2(team_weeks, _volume(), opponent_priors=priors)
    assert (m2["available_at"] == M1_AVAILABLE_AT).all()
    # An earlier board cutoff still loses to the 2026 schedule-env vintage.
    early_board = "2025-12-01T00:00:00+00:00"
    m2_early = allocate_team_weeks_m2(
        team_weeks,
        _volume(),
        opponent_priors=priors,
        board_available_at=early_board,
    )
    assert (m2_early["available_at"] == M1_AVAILABLE_AT).all()
    assert (m2_early["available_at_board"] == early_board).all()


def test_m2_refuses_same_week_team_attempts():
    team_weeks = _manual_team_weeks(
        [
            {
                "team": "AAA",
                "week": 1,
                "opponent": "BBB",
                "is_home": 1,
                "is_bye": 0,
                "is_neutral": 0,
                "team_attempts": 40.0,
            }
        ]
    )
    with pytest.raises(ValueError, match="same-week"):
        allocate_team_weeks_m2(team_weeks, _volume())


def test_m2_refuses_adp_and_vegas_drivers():
    team_weeks = _manual_team_weeks(
        [
            {
                "team": "AAA",
                "week": 1,
                "opponent": "BBB",
                "is_home": 1,
                "is_bye": 0,
                "is_neutral": 0,
                "adp": 12.0,
            }
        ]
    )
    with pytest.raises(ValueError, match="ADP"):
        allocate_team_weeks_m2(team_weeks, _volume())


def test_as_of_priors_drop_current_week_and_fail_if_labeled_current():
    weekly = pd.DataFrame(
        {
            "team": ["BBB", "BBB"],
            "week": [1, 2],
            "pass_epa": [-4.0, 99.0],
            "rush_epa": [-1.0, 99.0],
            "gameday": ["2026-09-10", "2026-09-17"],
        }
    )
    priors = build_as_of_priors(weekly, as_of_week=2)
    assert priors["prior_source"].iloc[0] == "lagged_weeks_before_as_of"
    assert float(priors["def_pass_epa_allowed"].iloc[0]) == pytest.approx(-4.0)
    only_week2 = build_as_of_priors(weekly[weekly["week"].eq(2)], as_of_week=3)
    assert float(only_week2["def_pass_epa_allowed"].iloc[0]) == pytest.approx(99.0)
    labeled = priors.copy()
    labeled["week"] = 2
    with pytest.raises(ValueError, match="as_of_week"):
        assert_priors_are_as_of(labeled, as_of_week=2)


def test_as_of_priors_fill_unobserved_opponents_from_prior_season():
    weekly = pd.DataFrame(
        {
            "team": ["BBB"],
            "week": [1],
            "pass_epa": [-4.0],
            "rush_epa": [-1.0],
            "gameday": ["2025-09-07"],
        }
    )
    prior = pd.DataFrame(
        {
            "team": ["BBB", "CCC"],
            "def_pass_epa_allowed": [-4.0, 5.0],
            "def_rush_epa_allowed": [-1.0, 1.0],
            "available_at": [PRIOR_SEASON_EPA_AVAILABLE_AT, PRIOR_SEASON_EPA_AVAILABLE_AT],
        }
    )
    priors = build_as_of_priors(weekly, as_of_week=2, prior_season=prior)
    by_opp = priors.set_index("opponent")
    assert by_opp.loc["BBB", "prior_source"] == "lagged_weeks_before_as_of"
    assert by_opp.loc["CCC", "prior_source"] == "prior_season_fallback"
    assert float(by_opp.loc["BBB", "def_pass_epa_allowed"]) == pytest.approx(-4.0)
    assert float(by_opp.loc["CCC", "def_pass_epa_allowed"]) == pytest.approx(5.0)


def test_synthetic_rolling_origin_fails_closed_and_beats_naive():
    result = run_synthetic_rolling_origin()
    assert result["poison_same_week_raised"] is True
    assert result["forbidden_columns_on_schedule"] == []
    assert result["passes"] is True
    assert result["later_weeks_advance_available_at"] is True
    assert result["not_a_promotion"] is True


def test_historical_schedule_backtest_fails_closed_on_denylist_siblings():
    history = historical_schedule_history()
    team_week, _priors = historical_features_for_week(history, as_of_week=2)
    leaked = FORBIDDEN_SAME_WEEK_TRAINING_FEATURES.intersection(team_week.columns)
    assert leaked == set()
    leaked_outcomes = FORBIDDEN_SAME_WEEK_TRAINING_FEATURES.intersection(
        history["schedule"].columns
    )
    assert leaked_outcomes == set()
    for col in ("team_attempts", "team_carries", "team_targets", "team_air_yards"):
        poisoned = team_week.copy()
        poisoned[col] = 999.0
        with pytest.raises(ValueError, match="same-week"):
            refuse_forbidden_m2_columns(poisoned, where=f"historical {col}")
        with pytest.raises(ValueError, match="same-week"):
            refuse_forbidden_m2_columns(
                poison_same_week_volume(team_week, column=col),
                where=f"poison helper {col}",
            )
    result = run_historical_schedule_backtest()
    assert result["poison_same_week_raised"] is True
    assert set(result["poisoned_columns"]) == set(FORBIDDEN_SAME_WEEK_TRAINING_FEATURES)
    assert result["forbidden_columns_on_features"] == []


def test_historical_schedule_backtest_is_rolling_origin_as_of():
    result = run_historical_schedule_backtest()
    assert result["schema_version"] == "weekly_latent_m2_backtest_historical_v1"
    assert result["object_evaluated"] == "allocate_team_weeks_m2"
    assert result["passes"] is True
    assert result["not_a_promotion"] is True
    assert result["week1_prior_source"] == "prior_season_fallback"
    assert result["later_prior_source"] == "lagged_weeks_before_as_of"
    assert result["week1_available_at"] == HISTORICAL_BOARD_AVAILABLE_AT
    assert result["week1_available_at"] != M1_AVAILABLE_AT
    assert result["later_weeks_advance_available_at"] is True
    assert result["season_mass_moved_vs_sealed"] is True
    assert "6-8" in result["holdout"] or "not 6" in result["holdout"].lower()
    assert "promotion" in result["holdout"].lower()
    weeks = {row["week"] for row in result["weeks"]}
    assert weeks == {1, 2, 3, 4, 5}
    # Same-week EPA of 99 must not enter week-3 features.
    history = historical_schedule_history()
    weekly = history["weekly_epa"].copy()
    weekly.loc[weekly["week"].eq(3), "pass_epa"] = 99.0
    _team_week, priors = historical_features_for_week(
        {**history, "weekly_epa": weekly}, as_of_week=3
    )
    assert "week" not in priors.columns or (priors["week"] < 3).all()
    assert float(priors["def_pass_epa_allowed"].max()) < 50.0


def test_historical_and_synthetic_backtests_run_together():
    combined = run_m2_backtests()
    assert combined["passes"] is True
    assert combined["not_a_promotion"] is True
    assert combined["synthetic"]["passes"] is True
    assert combined["historical"]["passes"] is True
    assert combined["synthetic"]["schema_version"] == "weekly_latent_m2_backtest_synthetic_v1"
    assert combined["historical"]["schema_version"] == "weekly_latent_m2_backtest_historical_v1"


def test_dry_run_m2_shares_and_production_untouched(tmp_path):
    result = run_milestone2(dry_run=True, output_dir=tmp_path)
    assert result.conservation["passes"] is True
    assert result.backtest["passes"] is True
    assert result.backtest["historical"]["passes"] is True
    assert result.backtest["synthetic"]["passes"] is True
    shares = result.tables.shares
    named = shares[shares["stat"].eq("attempts")].groupby("team")["share"].sum()
    other = shares[shares["stat"].eq("attempts")].groupby("team")["other_share"].first()
    assert ((named + other) - 1.0).abs().max() < 1e-9
    assert result.summary["production_hash_drift"] == {}
    assert result.summary["still_shadow"] is True
    assert result.summary["gate_verdict"] == "not promoting"
    for col in ("prior_available_at", "available_at_board", "env_available_at"):
        assert col in result.tables.player_weeks.columns
    forbidden = FORBIDDEN_SAME_WEEK_TRAINING_FEATURES.intersection(
        result.tables.team_weeks.columns
    )
    market = FORBIDDEN_MARKET_DRIVERS.intersection(result.tables.team_weeks.columns)
    assert forbidden == set()
    assert market == set()
    m1_pass = result.m1_tables.team_weeks.groupby("team")["team_pass_attempts"].sum()
    m2_pass = result.tables.team_weeks.groupby("team")["team_pass_attempts"].sum()
    sealed = result.tables.team_volume.set_index("team")["team_pass_attempts"]
    assert (m1_pass - sealed).abs().max() < 1e-6
    assert (m2_pass - sealed).abs().max() > 1e-6


def test_forbidden_same_week_features_match_canonical_team_denylist():
    from src.projection.weekly.draws.feature_outcome_split import SAME_WEEK_OUTCOME_DENYLIST

    canonical_team = frozenset(
        name for name in SAME_WEEK_OUTCOME_DENYLIST if name.startswith("team_")
    )
    pr70_team_aggregates = frozenset({"team_attempts", "team_air_yards"})
    expected = canonical_team | pr70_team_aggregates
    assert FORBIDDEN_SAME_WEEK_TRAINING_FEATURES == expected


def test_m2_does_not_import_promote_or_team_pass_rate_join():
    graph = local_import_graph(["src.projection.weekly_latent.run"])
    assert "src.projection.promote_release" not in graph
    assert "src.projection.release_bundle_publish" not in graph
    assert "src.projection.weekly.features.team_context" not in graph
    assert "src.projection.weekly.draws.feature_outcome_split" not in graph


def test_refuse_helper_lists_all_denylist_siblings():
    for col in ("team_attempts", "team_carries", "team_targets", "team_air_yards"):
        with pytest.raises(ValueError):
            refuse_forbidden_m2_columns(pd.DataFrame({col: [1]}), where="unit")


@pytest.mark.skipif(not SEALED.is_file() or not EPA_FIXTURE.is_file(), reason="fixtures missing")
def test_sealed_board_m2_identities(tmp_path):
    before = production_fingerprint(ROOT)
    result = run_milestone2(
        season=2026,
        projections_path=SEALED,
        schedule_path=SCHEDULE,
        output_dir=tmp_path,
    )
    assert result.conservation["passes"] is True
    assert result.summary["n_teams"] == 32
    assert result.backtest["passes"] is True
    after = production_fingerprint(ROOT)
    assert before == after
    m1_sum = result.m1_tables.team_weeks.groupby("team")["team_pass_attempts"].sum()
    sealed = result.tables.team_volume.set_index("team")["team_pass_attempts"]
    assert (m1_sum - sealed).abs().max() < 1e-6
    m2_sum = result.tables.team_weeks.groupby("team")["team_pass_attempts"].sum()
    assert (m2_sum - sealed).abs().max() > 1e-3
    assert "replace Vegas weekly props" in result.summary["does_not"]
    assert (result.tables.player_weeks["is_bye"].eq(1)).any()
    bye_fp = result.tables.player_weeks.loc[
        result.tables.player_weeks["is_bye"].eq(1), "fantasy_points"
    ]
    assert float(bye_fp.abs().sum()) == 0.0
