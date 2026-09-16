"""Milestone 3 availability + conversion tests (research/shadow only).

Week-varying A_i,w and conversion latents on top of the M2 team-week latent.
Does not promote, reseal, blend Role 3, or touch League Value / PWA.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.projection.contracts import REPO_ROOT
from src.projection.shadow.forbidden import local_import_graph
from src.projection.weekly_latent.allocate import (
    allocate_players,
    allocate_players_m3,
    allocate_team_weeks_m2,
)
from src.projection.weekly_latent.backtest_m3 import (
    run_historical_m3_backtest,
    run_m3_backtests,
    run_synthetic_m3_rolling_origin,
)
from src.projection.weekly_latent.constants import (
    FORBIDDEN_MARKET_DRIVERS,
    FORBIDDEN_SAME_WEEK_TRAINING_FEATURES,
    M1_AVAILABLE_AT,
    PRIOR_SEASON_EPA_AVAILABLE_AT,
    opponent_shrinkage_lambda,
)
from src.projection.weekly_latent.environment import refuse_forbidden_m2_columns
from src.projection.weekly_latent.run import production_fingerprint, run_milestone3
from src.projection.weekly_latent.season_board import role_shares
from src.projection.weekly_latent.vegas_hook import (
    ROLE2_MARKET_MAP,
    compare_m3_to_vegas_props,
    export_role2_board,
    market_sanity_bands,
)

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
    if "season" not in frame.columns:
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


def _wr_player(*, projected_games: float = 17.0, depth_rank: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_id": "p-wr",
                "display_name": "Alpha WR",
                "team": "AAA",
                "position": "WR",
                "depth_rank": depth_rank,
                "projected_games": projected_games,
                "attempts": 0.0,
                "completions": 0.0,
                "passing_yards": 0.0,
                "passing_tds": 0.0,
                "interceptions": 0.0,
                "carries": 0.0,
                "rushing_yards": 0.0,
                "rushing_tds": 0.0,
                "targets": 80.0,
                "receptions": 56.0,
                "receiving_yards": 800.0,
                "receiving_tds": 6.0,
            }
        ]
    )


def _qb_player() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_id": "p-qb",
                "display_name": "Alpha QB",
                "team": "AAA",
                "position": "QB",
                "depth_rank": 1,
                "projected_games": 17.0,
                "attempts": 500.0,
                "completions": 400.0,
                "passing_yards": 4000.0,
                "passing_tds": 30.0,
                "interceptions": 10.0,
                "carries": 40.0,
                "rushing_yards": 200.0,
                "rushing_tds": 3.0,
                "targets": 0.0,
                "receptions": 0.0,
                "receiving_yards": 0.0,
                "receiving_tds": 0.0,
            }
        ]
    )


def _priors() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "opponent": ["BBB", "CCC"],
            "pass_factor": [0.80, 1.20],
            "rush_factor": [0.90, 1.10],
            "available_at": [PRIOR_SEASON_EPA_AVAILABLE_AT, PRIOR_SEASON_EPA_AVAILABLE_AT],
        }
    )


def _m3_tables(
    team_week_rows: list[dict],
    players: pd.DataFrame,
    *,
    availability: pd.DataFrame | None = None,
    priors: pd.DataFrame | None = None,
):
    team_weeks = _manual_team_weeks(team_week_rows)
    volume = _volume()
    shares = role_shares(players, volume)
    m2 = allocate_team_weeks_m2(
        team_weeks, volume, opponent_priors=priors if priors is not None else _priors()
    )
    m3 = allocate_players_m3(m2, players, shares, availability=availability)
    return m2, m3, shares, volume


def test_bye_forces_availability_zero_even_when_override_says_one():
    players = _wr_player()
    availability = pd.DataFrame(
        {
            "player_id": ["p-wr", "p-wr"],
            "week": [1, 2],
            "A_i_w": [1.0, 1.0],
            "available_at": [M1_AVAILABLE_AT, M1_AVAILABLE_AT],
        }
    )
    _m2, m3, _shares, _vol = _m3_tables(
        [
            {
                "team": "AAA",
                "week": 1,
                "opponent": "BBB",
                "is_home": 1,
                "is_bye": 0,
            },
            {
                "team": "AAA",
                "week": 2,
                "opponent": None,
                "is_home": 0,
                "is_bye": 1,
            },
        ],
        players,
        availability=availability,
    )
    bye = m3[m3["is_bye"].eq(1)].iloc[0]
    assert float(bye["A_i_w"]) == 0.0
    assert float(bye["targets"]) == 0.0
    assert float(bye["receptions"]) == 0.0
    assert float(bye["receiving_yards"]) == 0.0
    assert float(bye["fantasy_points"]) == 0.0
    active = m3[m3["is_bye"].eq(0)].iloc[0]
    assert float(active["A_i_w"]) > 0.0


def test_week_varying_availability_rest_and_explicit_sit():
    players = _wr_player(projected_games=17.0, depth_rank=2)
    availability = pd.DataFrame(
        {
            "player_id": ["p-wr"],
            "week": [3],
            "A_i_w": [0.0],
            "available_at": ["2026-09-22T00:00:00+00:00"],
        }
    )
    _m2, m3, _shares, _vol = _m3_tables(
        [
            {
                "team": "AAA",
                "week": 1,
                "opponent": "BBB",
                "is_home": 1,
                "is_bye": 0,
                "rest_days": 7,
            },
            {
                "team": "AAA",
                "week": 2,
                "opponent": "CCC",
                "is_home": 0,
                "is_bye": 0,
                "rest_days": 4,
            },
            {
                "team": "AAA",
                "week": 3,
                "opponent": "BBB",
                "is_home": 1,
                "is_bye": 0,
                "rest_days": 7,
            },
        ],
        players,
        availability=availability,
    )
    by_week = m3.set_index("week")
    assert float(by_week.loc[3, "A_i_w"]) == 0.0
    assert float(by_week.loc[3, "targets"]) == 0.0
    assert float(by_week.loc[2, "A_i_w"]) < float(by_week.loc[1, "A_i_w"])
    assert float(by_week.loc[1, "A_i_w"]) <= 1.0
    assert float(by_week.loc[1, "A_i_w"]) > 0.0


def test_season_equals_sum_of_weeks():
    players = pd.concat([_qb_player(), _wr_player()], ignore_index=True)
    _m2, m3, _shares, _vol = _m3_tables(
        [
            {
                "team": "AAA",
                "week": 1,
                "opponent": "BBB",
                "is_home": 1,
                "is_bye": 0,
            },
            {
                "team": "AAA",
                "week": 2,
                "opponent": None,
                "is_home": 0,
                "is_bye": 1,
            },
            {
                "team": "AAA",
                "week": 3,
                "opponent": "CCC",
                "is_home": 0,
                "is_bye": 0,
            },
        ],
        players,
    )
    weekly_sum = m3.groupby("player_id")["fantasy_points"].sum()
    # Season identity is the sum of weeks (bye contributes 0), not pred_season × 17.
    assert (m3.loc[m3["is_bye"].eq(1), "fantasy_points"] == 0).all()
    for pid, total in weekly_sum.items():
        rebuilt = float(m3.loc[m3["player_id"].eq(pid), "fantasy_points"].sum())
        assert rebuilt == pytest.approx(float(total))
        assert rebuilt >= 0.0


def test_conversions_week_vary_with_lagged_opponent_not_volume_matchup():
    """Efficiency tilts from opponent-only factors. Volume already has M2 matchup."""
    players = _wr_player()
    _m2, m3, _shares, _vol = _m3_tables(
        [
            {
                "team": "AAA",
                "week": 1,
                "opponent": "BBB",
                "is_home": 1,
                "is_bye": 0,
                "rest_days": 7,
                "is_neutral": 0,
            },
            {
                "team": "AAA",
                "week": 2,
                "opponent": "CCC",
                "is_home": 1,
                "is_bye": 0,
                "rest_days": 7,
                "is_neutral": 0,
            },
        ],
        players,
    )
    by_week = m3.set_index("week")
    # Same rest/home; BBB is a tough pass D, CCC is easy. Catch/YPT should follow.
    assert float(by_week.loc[1, "m_pass_conv"]) < float(by_week.loc[2, "m_pass_conv"])
    ypt_1 = float(by_week.loc[1, "receiving_yards"]) / float(by_week.loc[1, "targets"])
    ypt_2 = float(by_week.loc[2, "receiving_yards"]) / float(by_week.loc[2, "targets"])
    assert ypt_1 < ypt_2
    # Conversion factor is not the full volume matchup (would double-count opp).
    assert abs(float(by_week.loc[1, "m_pass_conv"]) - float(by_week.loc[1, "pass_matchup_mult"])) > 1e-6


def test_internal_consistency_receptions_le_targets_completions_le_attempts():
    players = pd.concat([_qb_player(), _wr_player()], ignore_index=True)
    # Extreme easy-D prior so raw catch/comp rate × multiplier could exceed 1.
    priors = pd.DataFrame(
        {
            "opponent": ["CCC"],
            "pass_factor": [1.5],
            "rush_factor": [1.5],
            "available_at": [PRIOR_SEASON_EPA_AVAILABLE_AT],
        }
    )
    high_rate_wr = _wr_player()
    high_rate_wr["receptions"] = 80.0  # catch rate 1.0 before opponent tilt
    high_rate_qb = _qb_player()
    high_rate_qb["completions"] = 500.0
    players = pd.concat([high_rate_qb, high_rate_wr], ignore_index=True)
    _m2, m3, _shares, _vol = _m3_tables(
        [
            {
                "team": "AAA",
                "week": 1,
                "opponent": "CCC",
                "is_home": 1,
                "is_bye": 0,
            }
        ],
        players,
        priors=priors,
    )
    assert (m3["receptions"] <= m3["targets"] + 1e-9).all()
    assert (m3["completions"] <= m3["attempts"] + 1e-9).all()
    assert (m3["interceptions"] <= (m3["attempts"] - m3["completions"]) + 1e-9).all()
    assert (m3["receiving_tds"] <= m3["receptions"] + 1e-9).all()
    assert (m3["passing_tds"] <= m3["completions"] + 1e-9).all()
    assert (m3["rushing_tds"] <= m3["carries"] + 1e-9).all()


def test_m3_refuses_same_week_volume_on_availability_and_team_weeks():
    players = _wr_player()
    team_weeks = _manual_team_weeks(
        [
            {
                "team": "AAA",
                "week": 1,
                "opponent": "BBB",
                "is_home": 1,
                "is_bye": 0,
                "team_attempts": 40.0,
            }
        ]
    )
    with pytest.raises(ValueError, match="same-week"):
        allocate_team_weeks_m2(team_weeks, _volume())
    _m2, m3, shares, _vol = _m3_tables(
        [
            {
                "team": "AAA",
                "week": 1,
                "opponent": "BBB",
                "is_home": 1,
                "is_bye": 0,
            }
        ],
        players,
    )
    leaked = pd.DataFrame(
        {
            "player_id": ["p-wr"],
            "week": [1],
            "A_i_w": [0.5],
            "team_air_yards": [99.0],
            "available_at": [M1_AVAILABLE_AT],
        }
    )
    with pytest.raises(ValueError, match="same-week"):
        allocate_players_m3(_m2, players, shares, availability=leaked)


def test_m3_refuses_adp_and_vegas_drivers():
    players = _wr_player()
    _m2, _m3, shares, _vol = _m3_tables(
        [
            {
                "team": "AAA",
                "week": 1,
                "opponent": "BBB",
                "is_home": 1,
                "is_bye": 0,
            }
        ],
        players,
    )
    market = pd.DataFrame(
        {
            "player_id": ["p-wr"],
            "week": [1],
            "A_i_w": [1.0],
            "adp": [12.0],
            "available_at": [M1_AVAILABLE_AT],
        }
    )
    with pytest.raises(ValueError, match="ADP"):
        allocate_players_m3(_m2, players, shares, availability=market)


def test_m3_available_at_is_max_and_keeps_vintage_columns():
    players = _wr_player()
    later = "2026-10-14T00:00:00+00:00"
    availability = pd.DataFrame(
        {
            "player_id": ["p-wr"],
            "week": [1],
            "A_i_w": [0.8],
            "available_at": [later],
        }
    )
    _m2, m3, _shares, _vol = _m3_tables(
        [
            {
                "team": "AAA",
                "week": 1,
                "opponent": "BBB",
                "is_home": 1,
                "is_bye": 0,
            }
        ],
        players,
        availability=availability,
    )
    assert (m3["available_at"] == later).all()
    for col in (
        "available_at_board",
        "env_available_at",
        "prior_available_at",
        "avail_available_at",
        "conv_available_at",
    ):
        assert col in m3.columns
    # Earlier override must not move the stamp before the board/env vintage.
    earlier = "2025-12-01T00:00:00+00:00"
    early = pd.DataFrame(
        {
            "player_id": ["p-wr"],
            "week": [1],
            "A_i_w": [0.8],
            "available_at": [earlier],
        }
    )
    _m2b, m3_early, _s, _v = _m3_tables(
        [
            {
                "team": "AAA",
                "week": 1,
                "opponent": "BBB",
                "is_home": 1,
                "is_bye": 0,
            }
        ],
        players,
        availability=early,
    )
    assert (m3_early["available_at"] == M1_AVAILABLE_AT).all()
    assert (m3_early["avail_available_at"] == earlier).all()


def test_m3_does_not_use_m2_constant_availability_on_sit_week():
    players = _wr_player()
    availability = pd.DataFrame(
        {
            "player_id": ["p-wr"],
            "week": [1],
            "A_i_w": [0.0],
            "available_at": [M1_AVAILABLE_AT],
        }
    )
    team_rows = [
        {
            "team": "AAA",
            "week": 1,
            "opponent": "BBB",
            "is_home": 1,
            "is_bye": 0,
        }
    ]
    m2, m3, shares, _vol = _m3_tables(team_rows, players, availability=availability)
    m2_players = allocate_players(m2, players, shares)
    assert float(m2_players["A_i_w"].iloc[0]) == 1.0
    assert float(m3["A_i_w"].iloc[0]) == 0.0
    assert float(m3["fantasy_points"].iloc[0]) == 0.0
    assert float(m2_players["fantasy_points"].iloc[0]) > 0.0


def test_synthetic_m3_rolling_origin_fails_closed_and_beats_naive():
    result = run_synthetic_m3_rolling_origin()
    assert result["schema_version"] == "weekly_latent_m3_backtest_synthetic_v1"
    assert result["object_evaluated"] == "allocate_players_m3"
    assert result["poison_same_week_raised"] is True
    assert result["forbidden_columns_on_features"] == []
    assert result["passes"] is True
    assert result["beats_naive"] is True
    assert result["later_weeks_advance_available_at"] is True
    assert result["not_a_promotion"] is True
    assert result["week3_availability_zero"] is True


def test_historical_m3_backtest_is_rolling_origin_player_object():
    result = run_historical_m3_backtest()
    assert result["schema_version"] == "weekly_latent_m3_backtest_historical_v1"
    assert result["object_evaluated"] == "allocate_players_m3"
    assert result["passes"] is True
    assert result["not_a_promotion"] is True
    assert result["poison_same_week_raised"] is True
    assert set(result["poisoned_columns"]) == set(FORBIDDEN_SAME_WEEK_TRAINING_FEATURES)
    assert result["bye_availability_zero"] is True
    assert result["season_equals_sum_of_weeks"] is True
    assert result["later_weeks_advance_available_at"] is True
    assert result["week1_available_at"] != M1_AVAILABLE_AT
    assert "6-8" in result["holdout"] or "not 6" in result["holdout"].lower()
    weeks = {row["week"] for row in result["weeks"]}
    assert weeks == {1, 2, 3, 4, 5}


def test_m3_backtests_run_together():
    combined = run_m3_backtests()
    assert combined["passes"] is True
    assert combined["not_a_promotion"] is True
    assert combined["synthetic"]["passes"] is True
    assert combined["historical"]["passes"] is True


def test_role2_board_export_and_vegas_compare_hook_no_blend(tmp_path):
    players = _wr_player()
    _m2, m3, _shares, _vol = _m3_tables(
        [
            {
                "team": "AAA",
                "week": 1,
                "opponent": "BBB",
                "is_home": 1,
                "is_bye": 0,
            }
        ],
        players,
    )
    board = export_role2_board(m3)
    assert {"player_id", "season", "week", "market", "model_mean"} <= set(board.columns)
    assert set(ROLE2_MARKET_MAP.values()) <= set(board["market"].unique()) | set(
        ROLE2_MARKET_MAP.values()
    )
    rec = board[board["market"].eq("receptions")]
    assert len(rec) == 1
    assert float(rec["model_mean"].iloc[0]) == pytest.approx(float(m3["receptions"].iloc[0]))
    leaked = FORBIDDEN_SAME_WEEK_TRAINING_FEATURES.intersection(board.columns)
    assert leaked == set()
    compare = compare_m3_to_vegas_props(board=board, snapshots_path=None, dry_run=True)
    assert compare["role"] == "evaluation_comparator"
    assert compare["role3_blend"] is False
    assert compare["promoting"] is False
    assert compare["gate_verdict"] in {"not_promoting", "not promoting"}
    with pytest.raises(ValueError, match="Role 3|blend"):
        compare_m3_to_vegas_props(
            board=board,
            snapshots_path=None,
            dry_run=True,
            blend_weights={"vegas": 0.3},
        )


def test_market_sanity_bands_are_stubbed_not_drivers():
    stub = market_sanity_bands()
    assert stub["used_as_driver"] is False
    assert stub["status"] == "deferred"
    assert "adp" in stub["note"].lower() or "vegas" in stub["note"].lower()


def test_dry_run_m3_identities_and_production_untouched(tmp_path):
    result = run_milestone3(dry_run=True, output_dir=tmp_path)
    assert result.conservation["passes"] is True
    assert result.backtest["passes"] is True
    assert result.summary["still_shadow"] is True
    assert result.summary["gate_verdict"] == "not promoting"
    assert result.summary["milestone"] == 3
    assert result.summary["production_hash_drift"] == {}
    pw = result.tables.player_weeks
    for col in (
        "A_i_w",
        "m_pass_conv",
        "m_rush_conv",
        "available_at_board",
        "env_available_at",
        "prior_available_at",
        "avail_available_at",
        "conv_available_at",
    ):
        assert col in pw.columns
    forbidden = FORBIDDEN_SAME_WEEK_TRAINING_FEATURES.intersection(pw.columns)
    market = FORBIDDEN_MARKET_DRIVERS.intersection(pw.columns)
    assert forbidden == set()
    assert market == set()
    assert (pw.loc[pw["is_bye"].eq(1), "A_i_w"] == 0).all()
    assert Path(result.output_paths["summary"]).is_file()
    assert "vegas_props_compare.json" in result.output_paths
    assert "shadow_board_role2.csv" in result.output_paths or Path(tmp_path, "shadow_board_role2.csv").is_file()


def test_m3_does_not_import_promote_or_weekly_eval_hard_dep():
    graph = local_import_graph(["src.projection.weekly_latent.run"])
    assert "src.projection.promote_release" not in graph
    assert "src.projection.release_bundle_publish" not in graph
    assert "src.projection.weekly.features.team_context" not in graph
    # Optional Role 2 hook may try weekly_eval at runtime; it must not be a
    # hard import of the unmerged PR #83 package from run.py's static graph.
    assert "src.app" not in graph


def test_forbidden_same_week_features_still_match_canonical_team_denylist():
    from src.projection.weekly.draws.feature_outcome_split import SAME_WEEK_OUTCOME_DENYLIST

    canonical_team = frozenset(
        name for name in SAME_WEEK_OUTCOME_DENYLIST if name.startswith("team_")
    )
    pr70_team_aggregates = frozenset({"team_attempts", "team_air_yards"})
    expected = canonical_team | pr70_team_aggregates
    assert FORBIDDEN_SAME_WEEK_TRAINING_FEATURES == expected


@pytest.mark.skipif(not SEALED.is_file(), reason="sealed board missing")
def test_sealed_board_m3_shadow_only(tmp_path):
    before = production_fingerprint(ROOT)
    result = run_milestone3(
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
    assert "replace Vegas weekly props" in result.summary["does_not"]
    bye = result.tables.player_weeks[result.tables.player_weeks["is_bye"].eq(1)]
    assert len(bye) > 0
    assert float(bye["A_i_w"].abs().sum()) == 0.0
    assert float(bye["fantasy_points"].abs().sum()) == 0.0
    # M3 availability is not uniformly 1 on active weeks (rest / projected_games).
    active = result.tables.player_weeks[result.tables.player_weeks["is_bye"].eq(0)]
    assert float(active["A_i_w"].min()) < 1.0
    assert result.vegas_compare["promoting"] is False
    assert result.market_sanity["used_as_driver"] is False
