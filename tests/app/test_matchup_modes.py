"""Opponent-mode semantics and lineup legality, on hand-built draw sets.

The seeded fixtures happen to submit an already-optimal opponent lineup, so they
cannot tell the two matchup modes apart. These tests construct the case that
distinguishes them — an opponent leaving points on the bench — and assert the
labelling and the probability difference directly, rather than inferring either
from a status code.
"""

from __future__ import annotations

from src.app.decisions.draws import build_draw_set
from src.app.decisions.lineup import matchup_probabilities, optimize_lineup
from src.app.projections.loader import PlayerSummary
from src.app.scoring.compiler import compile_sleeper_scoring

DRAWS = 4000

#: Half-PPR with a Superflex seat, so QB-vs-flex eligibility is exercised.
SUPERFLEX_SLOTS = ["QB", "RB", "WR", "TE", "FLEX", "SUPER_FLEX", "BN", "BN"]
SCORING = {"pass_yd": 0.04, "pass_td": 4, "rush_yd": 0.1, "rush_td": 6, "rec": 0.5, "rec_yd": 0.1}


def _player(player_id: str, position: str, mean: float) -> PlayerSummary:
    spread = mean * 0.35
    return PlayerSummary(
        player_id=player_id,
        name=player_id,
        position=position,
        team="BUF",
        mean_points=mean,
        quantiles={"p10": mean - spread, "p50": mean, "p90": mean + spread},
        availability_probability=1.0,
    )


def _draw_set(players, contract, *, seed_salt: str = "matchup-modes"):
    return build_draw_set(
        contract=contract,
        run_id="test-run",
        week=1,
        draw_count=DRAWS,
        point_summaries=players,
        seed_salt=seed_salt,
    )


def _roster(prefix: str, quality: float) -> list[PlayerSummary]:
    """Eight players: six clearly better than the two spares."""
    return [
        _player(f"{prefix}-qb1", "QB", 22.0 * quality),
        _player(f"{prefix}-qb2", "QB", 18.0 * quality),
        _player(f"{prefix}-rb1", "RB", 16.0 * quality),
        _player(f"{prefix}-wr1", "WR", 15.0 * quality),
        _player(f"{prefix}-te1", "TE", 12.0 * quality),
        _player(f"{prefix}-wr2", "WR", 11.0 * quality),
        _player(f"{prefix}-rb2", "RB", 4.0 * quality),
        _player(f"{prefix}-wr3", "WR", 3.0 * quality),
    ]


def _context():
    contract = compile_sleeper_scoring(SCORING, SUPERFLEX_SLOTS)
    players = _roster("me", 1.0) + _roster("opp", 1.0)
    return contract, _draw_set(players, contract)


OPTIMAL_OPPONENT = ["opp-qb1", "opp-rb1", "opp-wr1", "opp-te1", "opp-wr2", "opp-qb2"]
#: The same manager, having benched their second quarterback and best receiver.
SLOPPY_OPPONENT = ["opp-qb1", "opp-rb1", "opp-wr3", "opp-te1", "opp-wr2", "opp-rb2"]
MY_CANDIDATES = [p.player_id for p in _roster("me", 1.0)]
OPP_CANDIDATES = [p.player_id for p in _roster("opp", 1.0)]


def test_the_two_opponent_modes_are_different_questions():
    """Against a sloppy lineup you are favoured; against their best, less so."""
    contract, draws = _context()

    current = matchup_probabilities(
        draws,
        contract,
        user_candidate_ids=MY_CANDIDATES,
        opponent_candidate_ids=OPP_CANDIDATES,
        user_starters=MY_CANDIDATES[:6],
        opponent_mode="current",
        opponent_submitted_starters=SLOPPY_OPPONENT,
    )
    optimized = matchup_probabilities(
        draws,
        contract,
        user_candidate_ids=MY_CANDIDATES,
        opponent_candidate_ids=OPP_CANDIDATES,
        user_starters=MY_CANDIDATES[:6],
        opponent_mode="optimized",
        opponent_submitted_starters=SLOPPY_OPPONENT,
    )

    assert current["opponent_lineup_source"] == "current"
    assert optimized["opponent_lineup_source"] == "optimized"
    assert current["opponent_starters"] == SLOPPY_OPPONENT
    assert set(optimized["opponent_starters"]) == set(OPTIMAL_OPPONENT)

    # The optimized opponent scores more, so our win probability must fall.
    assert optimized["opponent_expected_points"] > current["opponent_expected_points"]
    assert (
        optimized["recommended_probabilities"]["win"]
        < current["recommended_probabilities"]["win"]
    )


def test_current_mode_says_so_when_the_opponent_has_not_submitted_a_lineup():
    """An empty submitted lineup is reported, never silently treated as optimal."""
    contract, draws = _context()

    result = matchup_probabilities(
        draws,
        contract,
        user_candidate_ids=MY_CANDIDATES,
        opponent_candidate_ids=OPP_CANDIDATES,
        opponent_mode="current",
        opponent_submitted_starters=[],
    )

    assert result["opponent_lineup_source"] == "optimized_fallback_no_submitted_lineup"


def test_probabilities_are_a_partition_over_win_tie_loss():
    contract, draws = _context()

    result = matchup_probabilities(
        draws,
        contract,
        user_candidate_ids=MY_CANDIDATES,
        opponent_candidate_ids=OPP_CANDIDATES,
        opponent_mode="optimized",
    )
    probs = result["recommended_probabilities"]

    assert set(probs) == {"win", "tie", "loss"}
    assert all(0.0 <= value <= 1.0 for value in probs.values())
    assert abs(sum(probs.values()) - 1.0) < 1e-9


def test_superflex_seat_is_filled_by_a_second_quarterback():
    """A Superflex league must be able to start two QBs; a flex-only must not."""
    contract, draws = _context()

    superflex = optimize_lineup(
        draws, contract, candidate_ids=MY_CANDIDATES, objective="points"
    )
    assert superflex.assignments["me-qb2"] == "SUPER_FLEX"

    flex_only_contract = compile_sleeper_scoring(
        SCORING, ["QB", "RB", "WR", "TE", "FLEX", "BN", "BN", "BN"]
    )
    flex_draws = _draw_set(_roster("me", 1.0), flex_only_contract)
    flex_only = optimize_lineup(
        flex_draws, flex_only_contract, candidate_ids=MY_CANDIDATES, objective="points"
    )
    assert "me-qb2" not in flex_only.starters
    assert set(flex_only.assignments.values()) <= {"QB", "RB", "WR", "TE", "FLEX"}


def test_a_locked_starter_cannot_be_benched_by_the_optimizer():
    """A player whose game already kicked off must stay where the manager left them."""
    contract, draws = _context()

    unlocked = optimize_lineup(
        draws, contract, candidate_ids=MY_CANDIDATES, objective="points"
    )
    assert "me-wr3" not in unlocked.starters

    locked = optimize_lineup(
        draws,
        contract,
        candidate_ids=MY_CANDIDATES,
        objective="points",
        locked_starter_ids=["me-wr3"],
    )
    assert "me-wr3" in locked.starters
    # And a locked bench player cannot be promoted into the lineup.
    benched = optimize_lineup(
        draws,
        contract,
        candidate_ids=MY_CANDIDATES,
        objective="points",
        locked_bench_ids=["me-qb1"],
    )
    assert "me-qb1" not in benched.starters


def test_win_probability_objective_can_differ_from_maximum_points():
    """The stated objective is win probability, not raw projected points."""
    contract, draws = _context()

    result = matchup_probabilities(
        draws,
        contract,
        user_candidate_ids=MY_CANDIDATES,
        opponent_candidate_ids=OPP_CANDIDATES,
        opponent_mode="optimized",
    )
    win_prob_lineup = result["recommended"]
    points_lineup = optimize_lineup(
        draws, contract, candidate_ids=MY_CANDIDATES, objective="points"
    )

    # Both must be legal lineups of the same size, filling the same seats.
    assert len(win_prob_lineup.starters) == len(points_lineup.starters)
    assert sorted(win_prob_lineup.assignments.values()) == sorted(
        points_lineup.assignments.values()
    )
    # The win-probability lineup is never worse at the objective it optimizes.
    opponent_totals = draws.totals_for(result["opponent_starters"])
    from src.app.decisions.lineup import win_probability

    assert (
        win_probability(win_prob_lineup.totals, opponent_totals)["win"]
        >= win_probability(points_lineup.totals, opponent_totals)["win"] - 1e-9
    )
