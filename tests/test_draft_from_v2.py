"""Tests for fantasy-projections-2 → draft CSV mapper."""

from __future__ import annotations

import pandas as pd
import pytest

from src.draft_assistant.from_v2 import (
    FANTASY_POINTS_COLS,
    STAT_COLS,
    map_season_df_to_fantasy_points,
    map_season_df_to_long_projections,
)


def _sample_v2_season() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "gsis_id": "00-001",
                "player_name": "Alpha QB",
                "position": "QB",
                "team": "KC",
                "fantasy_pts": 18.5,
                "fantasy_pts_low": 12.0,
                "fantasy_pts_high": 25.0,
                "fantasy_pts_season": 18.5 * 17,
                "projected_games": 17,
                "source": "v2_team_first",
                "low_confidence": False,
                "role": "starter",
                "depth_chart_status": "starter",
                "depth_rank": 1.0,
                "attempts": 34.0,
                "completions": 22.0,
                "passing_yards": 260.0,
                "passing_tds": 1.8,
                "interceptions": 0.7,
                "carries": 3.0,
                "rushing_yards": 12.0,
                "rushing_tds": 0.2,
                "targets": 0.0,
                "receptions": 0.0,
                "receiving_yards": 0.0,
                "receiving_tds": 0.0,
            },
            {
                "gsis_id": "00-002",
                "player_name": "Beta RB",
                "position": "RB",
                "team": "SF",
                "fantasy_pts": 14.2,
                "fantasy_pts_low": 8.0,
                "fantasy_pts_high": 20.0,
                "fantasy_pts_season": 14.2 * 17,
                "projected_games": 17,
                "source": "v2_team_first",
                "low_confidence": False,
                "role": "starter",
                "depth_chart_status": "starter",
                "depth_rank": 1.0,
                "attempts": 0.0,
                "completions": 0.0,
                "passing_yards": 0.0,
                "passing_tds": 0.0,
                "interceptions": 0.0,
                "carries": 16.0,
                "rushing_yards": 70.0,
                "rushing_tds": 0.6,
                "targets": 4.0,
                "receptions": 3.0,
                "receiving_yards": 25.0,
                "receiving_tds": 0.1,
            },
            {
                "gsis_id": "00-003",
                "player_name": "Gamma WR",
                "position": "WR",
                "team": "MIA",
                "fantasy_pts": 13.1,
                "floor": 7.0,
                "ceiling": 19.0,
                "projected_games": 17,
                "attempts": 0.0,
                "completions": 0.0,
                "passing_yards": 0.0,
                "passing_tds": 0.0,
                "interceptions": 0.0,
                "carries": 0.5,
                "rushing_yards": 2.0,
                "rushing_tds": 0.0,
                "targets": 8.0,
                "receptions": 5.0,
                "receiving_yards": 70.0,
                "receiving_tds": 0.5,
            },
            {
                "gsis_id": "00-004",
                "player_name": "Delta TE",
                "position": "TE",
                "team": "BAL",
                "fantasy_pts": 9.5,
                "fantasy_pts_low": 5.0,
                "fantasy_pts_high": 14.0,
                "projected_games": 17,
                "attempts": 0.0,
                "completions": 0.0,
                "passing_yards": 0.0,
                "passing_tds": 0.0,
                "interceptions": 0.0,
                "carries": 0.0,
                "rushing_yards": 0.0,
                "rushing_tds": 0.0,
                "targets": 6.0,
                "receptions": 4.0,
                "receiving_yards": 45.0,
                "receiving_tds": 0.4,
            },
        ]
    )


def test_map_fantasy_points_required_columns_and_positions():
    raw = _sample_v2_season()
    out = map_season_df_to_fantasy_points(raw, season=2026)
    for col in FANTASY_POINTS_COLS:
        assert col in out.columns, f"missing {col}"
    assert set(out["position"]) == {"QB", "RB", "WR", "TE"}
    assert out["fantasy_pts"].is_monotonic_decreasing
    assert out.loc[out["position"] == "WR", "fantasy_pts_low"].notna().all()
    assert (out["source"] == "v2_team_first").all() or out["source"].notna().all()


def test_map_long_projections_has_stat_rows():
    raw = _sample_v2_season()
    long = map_season_df_to_long_projections(raw, season=2026)
    assert not long.empty
    assert set(STAT_COLS).issubset(set(long["stat"].unique()))
    qb = long[long["player_id"] == "00-001"]
    attempts = qb[qb["stat"] == "attempts"].iloc[0]
    assert attempts["pred_pg"] == 34.0
    assert attempts["pred_season"] == 34.0 * 17


def _long_row(pid, team, pos, stat, pg, games):
    return {
        "player_id": pid, "display_name": pid, "team": team, "position": pos,
        "stat": stat, "pred_pg": pg, "pred_season": pg * games,
        "projected_games": games,
    }


def test_reconcile_team_season_identities_enforces_the_identity():
    """A team's season receiving yards must equal its season passing yards.

    The failure this guards: a quarterback room that does not cover the season.
    QB1 plays 9 games, the backup 15 at a token rate, and the receivers pile up
    ~15 games of production behind quarterbacks who threw for nine.
    """
    from src.draft_assistant.from_v2 import reconcile_team_season_identities

    rows = [
        _long_row("qb1", "CLE", "QB", "passing_yards", 210.0, 9.0),
        _long_row("qb2", "CLE", "QB", "passing_yards", 2.0, 15.0),
        _long_row("wr1", "CLE", "WR", "receiving_yards", 120.0, 15.0),
        _long_row("te1", "CLE", "TE", "receiving_yards", 90.0, 15.0),
    ]
    long = pd.DataFrame(rows)
    before_recv = long[long.position != "QB"]["pred_season"].sum()
    before_pass = long[long.position == "QB"]["pred_season"].sum()
    assert before_recv / before_pass > 1.5  # the defect

    out = reconcile_team_season_identities(long)
    recv = out[out.position != "QB"]["pred_season"].sum()
    pas = out[out.position == "QB"]["pred_season"].sum()
    assert recv == pytest.approx(pas, rel=1e-9)
    # Symmetric: receivers come down and passers go up, neither forced onto the other.
    assert recv < before_recv
    assert pas > before_pass
    # Per-game rates untouched.
    assert out["pred_pg"].tolist() == long["pred_pg"].tolist()


def test_reconcile_targets_cannot_exceed_attempts():
    """targets/attempts is bounded below 1 -- every target is a pass attempt."""
    from src.draft_assistant.from_v2 import (
        TARGETS_PER_ATTEMPT,
        reconcile_team_season_identities,
    )

    # Engine hands us an impossible ratio (more targets than attempts).
    long = pd.DataFrame([
        _long_row("qb1", "BUF", "QB", "attempts", 30.0, 15.0),
        _long_row("wr1", "BUF", "WR", "targets", 20.0, 15.0),
        _long_row("te1", "BUF", "TE", "targets", 12.0, 15.0),
    ])
    assert long[long.position != "QB"]["pred_season"].sum() > long[
        long.position == "QB"
    ]["pred_season"].sum()

    out = reconcile_team_season_identities(long)
    ratio = (
        out[out.position != "QB"]["pred_season"].sum()
        / out[out.position == "QB"]["pred_season"].sum()
    )
    assert ratio == pytest.approx(TARGETS_PER_ATTEMPT, rel=1e-9)
    assert ratio < 1.0


def test_reconcile_is_idempotent():
    from src.draft_assistant.from_v2 import reconcile_team_season_identities

    long = pd.DataFrame([
        _long_row("qb1", "KC", "QB", "passing_yards", 250.0, 16.0),
        _long_row("wr1", "KC", "WR", "receiving_yards", 150.0, 14.0),
        _long_row("te1", "KC", "TE", "receiving_yards", 100.0, 12.0),
    ])
    once = reconcile_team_season_identities(long)
    twice = reconcile_team_season_identities(once)
    pd.testing.assert_series_equal(once["pred_season"], twice["pred_season"])
