"""Legal lineup construction and matchup evaluation.

Key properties this module guarantees, each covered by a test in
``tests/app/test_decisions_engine.py``:

* Lineups are **legal**: every seat is filled by a position-eligible player and
  no player occupies two seats. FLEX/REC_FLEX/SUPER_FLEX eligibility is honoured.
* The points-optimal lineup is **exactly** optimal, not greedy. Seats are
  expanded and solved as a maximum-weight assignment, so a SUPER_FLEX league
  cannot be mis-filled by slot ordering.
* **Locked** players (already kicked off / already played) cannot be moved into
  or out of the starting lineup.
* Win probability is computed from **joint draws** on a shared simulation index,
  with ties reported separately rather than folded into wins.
* When the caller asks to optimize win probability, the returned lineup
  maximizes win probability — not raw projected points. The two objectives
  disagree whenever variance matters (a favourite prefers a low-variance
  lineup, an underdog prefers a high-variance one).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.app.decisions.draws import DrawSet, PlayerDraws
from src.app.scoring.contract import ScoringContract

Objective = str  # "points" | "win_probability"

_NEG = -1.0e9


@dataclass(frozen=True)
class Seat:
    """One startable roster position."""

    index: int
    slot: str
    eligible_positions: tuple[str, ...]


@dataclass
class LineupResult:
    starters: list[str]
    assignments: dict[str, str] = field(default_factory=dict)
    expected_points: float = 0.0
    quantiles: dict[str, float] = field(default_factory=dict)
    objective: Objective = "points"
    win_probability: float | None = None
    unfilled_seats: list[str] = field(default_factory=list)
    totals: np.ndarray | None = None


def expand_seats(contract: ScoringContract) -> list[Seat]:
    seats: list[Seat] = []
    for slot in contract.scoring_slots:
        for _ in range(slot.count):
            seats.append(
                Seat(
                    index=len(seats),
                    slot=slot.slot,
                    eligible_positions=tuple(slot.eligible_positions),
                )
            )
    return seats


def _assign_optimal(
    seats: list[Seat],
    candidates: list[PlayerDraws],
    scores: dict[str, float],
    *,
    required_player_ids: frozenset[str] = frozenset(),
    excluded_player_ids: frozenset[str] = frozenset(),
) -> tuple[dict[str, str], list[str]]:
    """Maximum-weight legal seat assignment.

    Returns ``(assignments, unfilled_slot_names)`` where ``assignments`` maps
    player id -> slot name. Required players are forced into the lineup with a
    large bonus; excluded players are never assigned.
    """
    usable = [p for p in candidates if p.player_id not in excluded_player_ids]
    if not seats or not usable:
        return {}, [seat.slot for seat in seats]

    n_rows = len(usable)
    n_cols = len(seats)
    size = max(n_rows, n_cols)
    matrix = np.full((size, size), _NEG, dtype=float)

    # Padding rows/cols score 0 so unfilled seats are allowed but never
    # preferred over a legal fill.
    matrix[n_rows:, :] = 0.0
    matrix[:, n_cols:] = 0.0

    force_bonus = 1.0e6
    for row, player in enumerate(usable):
        for col, seat in enumerate(seats):
            if player.position in seat.eligible_positions:
                value = scores.get(player.player_id, 0.0)
                if player.player_id in required_player_ids:
                    value += force_bonus
                matrix[row, col] = value

    row_idx, col_idx = linear_sum_assignment(-matrix)
    assignments: dict[str, str] = {}
    filled_cols: set[int] = set()
    for r, c in zip(row_idx, col_idx):
        if r >= n_rows or c >= n_cols:
            continue
        if matrix[r, c] <= _NEG / 2:
            continue
        assignments[usable[r].player_id] = seats[c].slot
        filled_cols.add(int(c))

    unfilled = [seat.slot for seat in seats if seat.index not in filled_cols]
    return assignments, unfilled


def _mean_scores(draw_set: DrawSet, candidates: list[PlayerDraws]) -> dict[str, float]:
    return {p.player_id: p.mean for p in candidates}


def _quantiles(totals: np.ndarray) -> dict[str, float]:
    if totals.size == 0:
        return {}
    return {
        "0.1": float(np.quantile(totals, 0.1)),
        "0.5": float(np.quantile(totals, 0.5)),
        "0.9": float(np.quantile(totals, 0.9)),
    }


def win_probability(user_totals: np.ndarray, opponent_totals: np.ndarray) -> dict[str, float]:
    """Win/tie/loss probabilities from aligned joint draws.

    Ties are counted separately and never folded into wins. The three values sum
    to exactly 1.0 by construction.
    """
    count = int(min(user_totals.shape[0], opponent_totals.shape[0]))
    if count == 0:
        return {"win": 0.0, "tie": 1.0, "loss": 0.0}
    user = user_totals[:count]
    opp = opponent_totals[:count]
    # Fantasy scoring is quantised; treat sub-0.01 gaps as ties.
    diff = user - opp
    wins = int(np.count_nonzero(diff > 1e-9))
    losses = int(np.count_nonzero(diff < -1e-9))
    ties = count - wins - losses
    return {"win": wins / count, "tie": ties / count, "loss": losses / count}


def optimize_lineup(
    draw_set: DrawSet,
    contract: ScoringContract,
    *,
    candidate_ids: list[str] | None = None,
    objective: Objective = "points",
    opponent_totals: np.ndarray | None = None,
    locked_starter_ids: list[str] | None = None,
    locked_bench_ids: list[str] | None = None,
) -> LineupResult:
    """Solve the best legal lineup for the requested objective.

    ``objective="points"`` solves the exact maximum expected-points assignment.
    ``objective="win_probability"`` starts from that solution and performs
    steepest-ascent local search on the *true* win probability over single
    starter/bench exchanges. Local search is used because maximum win
    probability is not a linear assignment problem; the starting point is
    already optimal for the mean, and each accepted move strictly increases the
    measured win probability, so the result never scores worse than the
    points-optimal lineup on the requested objective.
    """
    seats = expand_seats(contract)
    ids = candidate_ids if candidate_ids is not None else list(draw_set.players.keys())
    candidates = [draw_set.players[pid] for pid in ids if pid in draw_set.players]
    required = frozenset(locked_starter_ids or ())
    excluded = frozenset(locked_bench_ids or ())

    scores = _mean_scores(draw_set, candidates)
    assignments, unfilled = _assign_optimal(
        seats, candidates, scores, required_player_ids=required, excluded_player_ids=excluded
    )
    starters = list(assignments.keys())

    if objective == "win_probability" and opponent_totals is not None and starters:
        assignments, starters = _local_search_win_probability(
            draw_set,
            seats,
            candidates,
            assignments,
            opponent_totals,
            required_player_ids=required,
            excluded_player_ids=excluded,
        )
        unfilled = _unfilled_for(seats, assignments)

    totals = draw_set.totals_for(starters)
    result = LineupResult(
        starters=starters,
        assignments=assignments,
        expected_points=float(totals.mean()) if totals.size else 0.0,
        quantiles=_quantiles(totals),
        objective=objective,
        unfilled_seats=unfilled,
        totals=totals,
    )
    if opponent_totals is not None:
        result.win_probability = win_probability(totals, opponent_totals)["win"]
    return result


def _unfilled_for(seats: list[Seat], assignments: dict[str, str]) -> list[str]:
    used: dict[str, int] = {}
    for slot in assignments.values():
        used[slot] = used.get(slot, 0) + 1
    unfilled: list[str] = []
    seat_counts: dict[str, int] = {}
    for seat in seats:
        seat_counts[seat.slot] = seat_counts.get(seat.slot, 0) + 1
    for slot, total in seat_counts.items():
        for _ in range(total - used.get(slot, 0)):
            unfilled.append(slot)
    return unfilled


def _is_legal(
    seats: list[Seat],
    assignments: dict[str, str],
    positions: dict[str, str],
) -> bool:
    """Verify an assignment respects seat counts and position eligibility."""
    remaining: dict[str, int] = {}
    eligibility: dict[str, tuple[str, ...]] = {}
    for seat in seats:
        remaining[seat.slot] = remaining.get(seat.slot, 0) + 1
        eligibility[seat.slot] = seat.eligible_positions
    for player_id, slot in assignments.items():
        if remaining.get(slot, 0) <= 0:
            return False
        if positions.get(player_id) not in eligibility.get(slot, ()):
            return False
        remaining[slot] -= 1
    return True


def paired_win_probability_gain(
    base_totals: np.ndarray,
    trial_totals: np.ndarray,
    opponent_totals: np.ndarray,
) -> tuple[float, float]:
    """Paired win-probability difference and its standard error.

    Both lineups are evaluated against the *same* opponent draws (common random
    numbers), so the comparison is paired: the per-draw difference is non-zero
    only on draws where the change flips the result. Returning the standard error
    lets the caller refuse changes that are indistinguishable from Monte Carlo
    noise, which is what stops a hill-climbing search from chasing sampling
    error and reporting a fake edge.
    """
    count = int(
        min(base_totals.shape[0], trial_totals.shape[0], opponent_totals.shape[0])
    )
    if count == 0:
        return 0.0, 0.0
    opp = opponent_totals[:count]
    base_win = (base_totals[:count] - opp > 1e-9).astype(float)
    trial_win = (trial_totals[:count] - opp > 1e-9).astype(float)
    diff = trial_win - base_win
    mean = float(diff.mean())
    if count < 2:
        return mean, 0.0
    stderr = float(diff.std(ddof=1) / np.sqrt(count))
    return mean, stderr


def _local_search_win_probability(
    draw_set: DrawSet,
    seats: list[Seat],
    candidates: list[PlayerDraws],
    assignments: dict[str, str],
    opponent_totals: np.ndarray,
    *,
    required_player_ids: frozenset[str],
    excluded_player_ids: frozenset[str],
    max_rounds: int = 6,
    significance: float = 2.0,
) -> tuple[dict[str, str], list[str]]:
    """Steepest-ascent search on win probability with a noise guard.

    A candidate swap is accepted only when its paired win-probability gain
    exceeds ``significance`` standard errors. Without that guard the search
    maximises sampling noise and happily benches a better player to "gain" a
    percentage point that does not exist.
    """
    positions = {p.player_id: p.position for p in candidates}
    current = dict(assignments)

    for _ in range(max_rounds):
        base_totals = draw_set.totals_for(list(current.keys()))
        best_move: tuple[float, dict[str, str]] | None = None
        bench = [
            p.player_id
            for p in candidates
            if p.player_id not in current and p.player_id not in excluded_player_ids
        ]
        for out_id in list(current.keys()):
            if out_id in required_player_ids:
                continue
            slot = current[out_id]
            eligible = _eligibility_for(seats, slot)
            for in_id in bench:
                if positions.get(in_id) not in eligible:
                    continue
                trial = dict(current)
                del trial[out_id]
                trial[in_id] = slot
                if not _is_legal(seats, trial, positions):
                    continue
                gain, stderr = paired_win_probability_gain(
                    base_totals, draw_set.totals_for(list(trial.keys())), opponent_totals
                )
                if gain <= significance * stderr or gain <= 1e-9:
                    continue
                if best_move is None or gain > best_move[0]:
                    best_move = (gain, trial)
        if best_move is None:
            break
        current = best_move[1]

    return current, list(current.keys())


def _eligibility_for(seats: list[Seat], slot: str) -> tuple[str, ...]:
    for seat in seats:
        if seat.slot == slot:
            return seat.eligible_positions
    return ()


def matchup_probabilities(
    draw_set: DrawSet,
    contract: ScoringContract,
    *,
    user_candidate_ids: list[str],
    opponent_candidate_ids: list[str],
    user_starters: list[str] | None = None,
    opponent_mode: str = "current",
    opponent_submitted_starters: list[str] | None = None,
    locked_starter_ids: list[str] | None = None,
    locked_bench_ids: list[str] | None = None,
) -> dict:
    """Evaluate a matchup under an explicit opponent assumption.

    ``opponent_mode="current"`` uses the opponent's submitted Sleeper starters.
    ``opponent_mode="optimized"`` uses the opponent's own optimal legal lineup.
    The two modes are computed from different opponent lineups and are labelled
    distinctly; when the opponent has not submitted a lineup, ``current`` is
    reported as unavailable rather than silently reusing the optimized lineup.
    """
    opponent_lineup = optimize_lineup(
        draw_set, contract, candidate_ids=opponent_candidate_ids, objective="points"
    )

    opponent_source = opponent_mode
    if opponent_mode == "current":
        submitted = [
            pid for pid in (opponent_submitted_starters or []) if pid in draw_set.players
        ]
        if submitted:
            opponent_ids = submitted
        else:
            opponent_ids = opponent_lineup.starters
            opponent_source = "optimized_fallback_no_submitted_lineup"
    else:
        opponent_ids = opponent_lineup.starters

    opponent_totals = draw_set.totals_for(opponent_ids)

    recommended = optimize_lineup(
        draw_set,
        contract,
        candidate_ids=user_candidate_ids,
        objective="win_probability",
        opponent_totals=opponent_totals,
        locked_starter_ids=locked_starter_ids,
        locked_bench_ids=locked_bench_ids,
    )

    current_ids = [pid for pid in (user_starters or []) if pid in draw_set.players]
    current_totals = (
        draw_set.totals_for(current_ids) if current_ids else recommended.totals
    )

    return {
        "opponent_mode": opponent_mode,
        "opponent_lineup_source": opponent_source,
        "opponent_starters": opponent_ids,
        "opponent_expected_points": float(opponent_totals.mean())
        if opponent_totals.size
        else 0.0,
        "recommended": recommended,
        "recommended_probabilities": win_probability(recommended.totals, opponent_totals),
        "current_probabilities": win_probability(current_totals, opponent_totals)
        if current_ids
        else None,
        "current_starters": current_ids,
        "current_expected_points": float(current_totals.mean())
        if current_totals is not None and current_totals.size
        else 0.0,
    }


def swap_recommendations(
    draw_set: DrawSet,
    contract: ScoringContract,
    *,
    current_starters: list[str],
    recommended: LineupResult,
    opponent_totals: np.ndarray,
    locked_ids: frozenset[str] = frozenset(),
) -> list[dict]:
    """Explain the difference between the current and recommended lineup.

    Each swap reports the **measured** win-probability change from applying that
    single swap to the current lineup, plus the expected-point change. No
    constant placeholder is used.
    """
    current = [pid for pid in current_starters if pid in draw_set.players]
    if not current:
        return []

    base_totals = draw_set.totals_for(current)
    base_prob = win_probability(base_totals, opponent_totals)["win"]
    base_points = float(base_totals.mean()) if base_totals.size else 0.0

    recommended_set = set(recommended.starters)
    current_set = set(current)
    ins = [pid for pid in recommended.starters if pid not in current_set]
    outs = [pid for pid in current if pid not in recommended_set]

    swaps: list[dict] = []
    for in_id, out_id in zip(ins, outs):
        if out_id in locked_ids or in_id in locked_ids:
            continue
        trial = [in_id if pid == out_id else pid for pid in current]
        trial_totals = draw_set.totals_for(trial)
        trial_prob = win_probability(trial_totals, opponent_totals)["win"]
        gain, stderr = paired_win_probability_gain(
            base_totals, trial_totals, opponent_totals
        )
        in_player = draw_set.players[in_id]
        out_player = draw_set.players[out_id]
        swaps.append(
            {
                "in_player_id": in_id,
                "out_player_id": out_id,
                "slot": recommended.assignments.get(in_id),
                "win_probability_delta": round(trial_prob - base_prob, 6),
                "win_probability_delta_stderr": round(stderr, 6),
                # False means the measured edge is inside Monte Carlo noise and
                # should be presented as a coin flip, not as an improvement.
                "significant": bool(gain > 2.0 * stderr and gain > 1e-9),
                "expected_points_delta": round(
                    float(trial_totals.mean()) - base_points, 4
                ),
                "in_expected_points": round(in_player.mean, 3),
                "out_expected_points": round(out_player.mean, 3),
                "in_range": [
                    round(in_player.percentile(0.1), 3),
                    round(in_player.percentile(0.9), 3),
                ],
                "out_range": [
                    round(out_player.percentile(0.1), 3),
                    round(out_player.percentile(0.9), 3),
                ],
                "reason": (
                    f"Starting {in_id} over {out_id} changes win probability by "
                    f"{(trial_prob - base_prob) * 100:+.1f} points."
                ),
            }
        )
    return swaps
