"""Regression tests that document defects in the legacy independent scaled-draw path.

These tests assert the *current* trained-path behavior that the joint mixture
architecture must replace. They are not aspirational for the new engine.
"""

from __future__ import annotations

import inspect

import numpy as np
import polars as pl
import pytest

from src.app.decisions import draws as decision_draws
from src.app.projections.weekly_draws import (
    PARTITION_SCHEMA_VERSION,
    generate_player_stat_draws,
)
from src.app.projections.weekly_stat_draw import WEEKLY_TO_DRAW_STAT, weekly_row_to_stat_draw
from src.projection.special_teams.models import (
    KickerContext,
    TeamContext,
    simulate_dst_draw,
    simulate_kicker_draw,
)


def _skill_row(**overrides) -> dict:
    base = {
        "gsis_id": "00-legacy-qb",
        "position": "QB",
        "team": "KC",
        "fantasy_points": 20.0,
        "floor": 12.0,
        "ceiling": 28.0,
        "attempts": 35.0,
        "completions": 23.0,
        "passing_yards": 275.0,
        "passing_tds": 2.0,
        "interceptions": 0.5,
        "carries": 3.0,
        "rushing_yards": 15.0,
        "rushing_tds": 0.2,
        "targets": 0.0,
        "receptions": 0.0,
        "receiving_yards": 0.0,
        "receiving_tds": 0.0,
    }
    base.update(overrides)
    return base


def test_legacy_partition_schema_is_v1():
    assert PARTITION_SCHEMA_VERSION == 1


def test_legacy_scaled_components_move_in_perfect_lockstep():
    """All component stats share one fantasy-point scale factor (defect #1)."""
    player = {
        "player_id": "00-legacy-qb",
        "position": "QB",
        "fantasy_points": 20.0,
        "floor": 12.0,
        "ceiling": 28.0,
        "components": weekly_row_to_stat_draw(_skill_row()),
    }
    draws = generate_player_stat_draws(player, draw_count=40, seed_salt="defect-lockstep")
    assert len(draws) == 40

    base = player["components"]
    base_sum = sum(base.values())
    ratios = []
    for draw in draws:
        scale = draw["pass_attempts"] / base["pass_attempts"]
        for key, base_val in base.items():
            if base_val == 0:
                assert draw[key] == 0.0
                continue
            assert draw[key] == pytest.approx(base_val * scale, rel=1e-9)
        # Relative composition is identical to the mean vector.
        ratios.append(draw["pass_yards"] / draw["pass_attempts"])
    assert max(ratios) - min(ratios) < 1e-12
    # Fractional discrete events are produced (implausible for counts).
    fractional = any(abs(d["pass_tds"] - round(d["pass_tds"])) > 1e-9 for d in draws)
    assert fractional
    assert base_sum > 0


def test_legacy_draws_have_no_discrete_dnp_event():
    """No participation / DNP Bernoulli; near-zero means still produce positive mass (defect #2)."""
    player = {
        "player_id": "00-bench-wr",
        "position": "WR",
        "fantasy_points": 0.4,
        "floor": 0.1,
        "ceiling": 0.8,
        "components": {
            "targets": 0.3,
            "receptions": 0.2,
            "rec_yards": 2.5,
            "rec_tds": 0.02,
        },
    }
    draws = generate_player_stat_draws(player, draw_count=200, seed_salt="defect-dnp")
    exact_zero = sum(1 for d in draws if all(v == 0.0 for v in d.values()))
    # Independent split-normal around a positive mean does not emit a discrete
    # DNP atom near the historical ~50% roster-week zero rate.
    assert exact_zero / len(draws) < 0.20
    assert all("active" not in d and "participated" not in d for d in draws)


def test_legacy_teammate_draws_are_independent():
    """Shared seed salt still yields independent player RNGs (defect #3)."""
    qb = {
        "player_id": "00-qb",
        "position": "QB",
        "fantasy_points": 18.0,
        "floor": 10.0,
        "ceiling": 26.0,
        "components": {"pass_attempts": 34.0, "pass_yards": 250.0, "pass_tds": 1.8},
    }
    wr = {
        "player_id": "00-wr",
        "position": "WR",
        "fantasy_points": 14.0,
        "floor": 8.0,
        "ceiling": 22.0,
        "components": {"targets": 8.0, "receptions": 5.0, "rec_yards": 70.0, "rec_tds": 0.5},
    }
    qb_draws = generate_player_stat_draws(qb, draw_count=500, seed_salt="same-game")
    wr_draws = generate_player_stat_draws(wr, draw_count=500, seed_salt="same-game")
    qb_yds = np.array([d["pass_yards"] for d in qb_draws])
    wr_yds = np.array([d["rec_yards"] for d in wr_draws])
    corr = float(np.corrcoef(qb_yds, wr_yds)[0, 1])
    assert abs(corr) < 0.15


def test_legacy_team_totals_need_not_reconcile():
    """Independent scaled draws do not conserve team pass/receive identities (defect #4)."""
    frame = pl.DataFrame(
        [
            _skill_row(gsis_id="qb1", position="QB", attempts=35.0, passing_yards=280.0, passing_tds=2.0),
            _skill_row(
                gsis_id="wr1",
                position="WR",
                attempts=0.0,
                completions=0.0,
                passing_yards=0.0,
                passing_tds=0.0,
                targets=9.0,
                receptions=6.0,
                receiving_yards=80.0,
                receiving_tds=0.7,
                fantasy_points=14.0,
                floor=8.0,
                ceiling=20.0,
            ),
            _skill_row(
                gsis_id="wr2",
                position="WR",
                attempts=0.0,
                completions=0.0,
                passing_yards=0.0,
                passing_tds=0.0,
                targets=7.0,
                receptions=4.0,
                receiving_yards=55.0,
                receiving_tds=0.4,
                fantasy_points=10.0,
                floor=5.0,
                ceiling=16.0,
            ),
        ]
    )
    players = [
        {
            "player_id": "qb1",
            "fantasy_points": 20.0,
            "floor": 12.0,
            "ceiling": 28.0,
            "components": weekly_row_to_stat_draw(frame.row(0, named=True)),
        },
        {
            "player_id": "wr1",
            "fantasy_points": 14.0,
            "floor": 8.0,
            "ceiling": 20.0,
            "components": weekly_row_to_stat_draw(frame.row(1, named=True)),
        },
        {
            "player_id": "wr2",
            "fantasy_points": 10.0,
            "floor": 5.0,
            "ceiling": 16.0,
            "components": weekly_row_to_stat_draw(frame.row(2, named=True)),
        },
    ]
    qb = generate_player_stat_draws(players[0], draw_count=30, seed_salt="team-a")
    wr1 = generate_player_stat_draws(players[1], draw_count=30, seed_salt="team-a")
    wr2 = generate_player_stat_draws(players[2], draw_count=30, seed_salt="team-a")
    mismatches = 0
    for i in range(30):
        pass_yds = qb[i]["pass_yards"]
        rec_yds = wr1[i]["rec_yards"] + wr2[i]["rec_yards"]
        if abs(pass_yds - rec_yds) > 1.0:
            mismatches += 1
    assert mismatches > 20


def test_legacy_scaled_draws_omit_first_downs_when_means_lack_them():
    """PPFD understated: trained mean rows historically omit first downs (defect #5)."""
    row = _skill_row()  # no passing/rushing/receiving_first_downs keys
    components = weekly_row_to_stat_draw(row)
    assert "pass_first_downs" not in components
    assert "rush_first_downs" not in components
    assert "rec_first_downs" not in components
    player = {
        "player_id": "00-legacy-qb",
        "fantasy_points": 20.0,
        "floor": 12.0,
        "ceiling": 28.0,
        "components": components,
    }
    draws = generate_player_stat_draws(player, draw_count=5, seed_salt="no-fd")
    assert all("pass_first_downs" not in d for d in draws)
    # Mapping may exist for joint path, but legacy scaler cannot invent FD events.
    assert "pass_first_downs" in set(WEEKLY_TO_DRAW_STAT.values())


def test_decision_draws_module_documents_independence():
    """decisions/draws.py explicitly documents independent player sampling (defect #6)."""
    doc = inspect.getdoc(decision_draws) or ""
    compact = " ".join(doc.lower().split())
    assert "independently" in compact
    assert "team-level correlation" in compact


def test_legacy_k_dst_are_independent_of_game_script():
    """K/DST use fixed priors and independent RNG calls (defect #7)."""
    dst_a = simulate_dst_draw(TeamContext(opponent_implied_points=22.0), seed=1)
    dst_b = simulate_dst_draw(TeamContext(opponent_implied_points=35.0), seed=1)
    # Same seed + different context still shares the same random stream shape;
    # points_allowed mean shifts but sacks/ints are not linked to an offense draw.
    assert "sacks" in dst_a and "points_allowed" in dst_a
    assert dst_a["sacks"] == dst_b["sacks"]  # first gauss draw identical before points shift usage
    kick = simulate_kicker_draw(KickerContext(), seed=7)
    assert "xpm" in kick
    # No game_id / team_tds / opponent offense inputs exist on the context objects.
    assert not hasattr(TeamContext, "game_pass_attempts")
    assert not hasattr(KickerContext, "team_touchdowns")
