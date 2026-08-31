"""Waiver and FAAB recommendation engine.

Candidates are ranked by *incremental roster utility*, not by universal
rest-of-season points. The inputs the blueprint requires are all explicit
arguments so that none of them can quietly default to a hardcoded constant:

* replacement level, derived from the best freely available player at each
  position under this league's slots and scoring — not a fixed table;
* the probability the player actually enters the user's starting lineup, derived
  from the user's current starters at eligible slots;
* positional depth and bye-week exposure on the user's roster;
* playoff-window schedule weight;
* league-mate rosters, as positional scarcity;
* Sleeper trending adds, used only as a market/urgency signal;
* remaining FAAB and the opportunity cost of spending it now.

Suggested bids are ranges with a confidence, never a single-dollar figure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.app.scoring.contract import ScoringContract


@dataclass(frozen=True)
class WaiverPlayer:
    player_id: str
    name: str
    position: str
    mean_points: float
    p10: float
    p90: float
    team: str | None = None
    bye_week: int | None = None
    availability_probability: float = 1.0

    @property
    def spread(self) -> float:
        return max(self.p90 - self.p10, 1e-6)


@dataclass
class WaiverCandidate:
    player_id: str
    name: str
    position: str
    incremental_utility: float
    start_probability: float
    replacement_level: float
    faab_low: float
    faab_high: float
    confidence: float
    rationale: list[str] = field(default_factory=list)
    inputs: dict = field(default_factory=dict)


def compute_replacement_levels(
    pool: list[WaiverPlayer], contract: ScoringContract
) -> dict[str, float]:
    """Replacement level = the best player you could add for free at that slot.

    Only positions that can actually start in this league are considered, so a
    league without kickers does not generate a kicker replacement level.
    """
    startable = contract.eligible_positions()
    levels: dict[str, float] = {}
    for position in startable:
        at_position = [p.mean_points for p in pool if p.position == position]
        if not at_position:
            continue
        at_position.sort(reverse=True)
        # Use the best freely available option rather than the pool mean: that is
        # the true alternative to making this claim.
        levels[position] = at_position[0]
    return levels


def _worst_eligible_starter(
    roster: list[WaiverPlayer], contract: ScoringContract, position: str
) -> float | None:
    """Mean points of the weakest current starter this player could displace."""
    eligible_slots = [
        slot for slot in contract.scoring_slots if position in slot.eligible_positions
    ]
    if not eligible_slots:
        return None
    displaceable_positions: set[str] = set()
    for slot in eligible_slots:
        displaceable_positions.update(slot.eligible_positions)

    seats = contract.starting_slot_count
    ranked = sorted(roster, key=lambda p: p.mean_points, reverse=True)
    starters = [p for p in ranked[:seats] if p.position in displaceable_positions]
    if not starters:
        return None
    return min(p.mean_points for p in starters)


def _start_probability(
    candidate: WaiverPlayer, bar: float | None
) -> float:
    """Probability the candidate outproduces the player they would replace.

    Uses the candidate's own projection spread as the scale, so a volatile
    streamer with a high ceiling is not treated as a certainty.
    """
    if bar is None:
        return 0.0
    edge = candidate.mean_points - bar
    # Logistic on the candidate's own spread; +/- one spread ~ 88%/12%.
    z = edge / (candidate.spread / 2.0)
    probability = 1.0 / (1.0 + pow(2.718281828459045, -z))
    return max(0.0, min(0.98, probability * candidate.availability_probability))


def recommend_waivers(
    pool: list[WaiverPlayer],
    *,
    contract: ScoringContract,
    roster: list[WaiverPlayer],
    remaining_faab: float,
    week: int,
    weeks_remaining: int,
    playoff_start_week: int | None = None,
    league_position_counts: dict[str, int] | None = None,
    trending_adds: dict[str, int] | None = None,
    limit: int = 25,
) -> list[WaiverCandidate]:
    trending_adds = trending_adds or {}
    league_position_counts = league_position_counts or {}
    replacement_levels = compute_replacement_levels(pool, contract)
    startable = contract.eligible_positions()
    max_trend = max(trending_adds.values()) if trending_adds else 0

    roster_by_position: dict[str, int] = {}
    for player in roster:
        roster_by_position[player.position] = roster_by_position.get(player.position, 0) + 1

    candidates: list[WaiverCandidate] = []
    for player in pool:
        if player.position not in startable:
            continue
        bar = _worst_eligible_starter(roster, contract, player.position)
        replacement = replacement_levels.get(player.position, 0.0)
        start_prob = _start_probability(player, bar)

        # Weekly starting value above the player currently in that seat.
        weekly_edge = max(0.0, player.mean_points - (bar if bar is not None else replacement))
        # Depth insurance: value even when not starting, scaled by how thin the
        # user is at the position and by bye-week exposure.
        depth_count = roster_by_position.get(player.position, 0)
        depth_need = 1.0 / (1.0 + depth_count)
        bye_exposure = 0.0
        if player.bye_week is not None and player.bye_week > week:
            bye_exposure = 0.15
        insurance = max(0.0, player.mean_points - replacement) * depth_need * 0.35

        # Playoff-window schedule weight.
        schedule_weight = 1.0
        if playoff_start_week is not None and playoff_start_week > week:
            schedule_weight = 1.0 + 0.10 * min(
                1.0, max(0, weeks_remaining) / max(1, playoff_start_week - week)
            )

        utility = (
            (weekly_edge * start_prob + insurance + bye_exposure * player.mean_points)
            * schedule_weight
        )
        if utility <= 0:
            continue

        # League-mate scarcity: fewer rostered players at this position across
        # the league means the add is more contested and more valuable.
        rostered_at_position = league_position_counts.get(player.position, 0)
        scarcity = 1.0 + 1.0 / (1.0 + rostered_at_position / 6.0)

        # Market urgency from Sleeper trending adds (never a projection input).
        trend = trending_adds.get(player.player_id, 0)
        urgency = 1.0 + (0.25 * trend / max_trend if max_trend else 0.0)

        # Opportunity cost: money kept is worth more when many weeks remain.
        opportunity_cost = 1.0 / (1.0 + 0.06 * max(0, weeks_remaining))
        bid_share = min(
            0.5,
            (utility * scarcity * urgency * opportunity_cost)
            / max(1.0, sum(replacement_levels.values()) or 1.0),
        )
        centre = max(0.0, remaining_faab * bid_share)

        # Range width tracks the projection's own uncertainty.
        relative_uncertainty = min(0.8, player.spread / max(player.mean_points, 1.0) / 2.0)
        low = max(0.0, centre * (1.0 - relative_uncertainty))
        high = min(remaining_faab, max(low + 1.0, centre * (1.0 + relative_uncertainty)))

        confidence = round(
            max(
                0.05,
                min(
                    0.9,
                    0.25
                    + 0.45 * start_prob
                    + (0.1 if bar is not None else 0.0)
                    - 0.2 * relative_uncertainty,
                ),
            ),
            4,
        )

        rationale = [
            f"Replacement level at {player.position}: {replacement:.2f} pts/wk "
            "(best freely available player).",
            (
                f"Would displace a {bar:.2f} pts/wk starter; start probability "
                f"{start_prob:.0%}."
                if bar is not None
                else "No displaceable starter at an eligible slot; valued as depth only."
            ),
            f"Roster depth at {player.position}: {depth_count}; depth weight {depth_need:.2f}.",
            f"Projection range {player.p10:.1f}-{player.p90:.1f} pts drives the bid width.",
            f"Opportunity cost over {weeks_remaining} remaining weeks: "
            f"{opportunity_cost:.2f}x.",
            f"League scarcity at {player.position}: {scarcity:.2f}x "
            f"({rostered_at_position} rostered league-wide).",
        ]
        if player.bye_week is not None:
            rationale.append(f"Bye week {player.bye_week}.")
        if trend:
            rationale.append(
                f"Sleeper trending adds: {trend} (market urgency only, not a projection)."
            )

        candidates.append(
            WaiverCandidate(
                player_id=player.player_id,
                name=player.name,
                position=player.position,
                incremental_utility=round(utility, 4),
                start_probability=round(start_prob, 4),
                replacement_level=round(replacement, 4),
                faab_low=round(low, 1),
                faab_high=round(high, 1),
                confidence=confidence,
                rationale=rationale,
                inputs={
                    "weekly_edge": round(weekly_edge, 4),
                    "displaced_starter_points": round(bar, 4) if bar is not None else None,
                    "depth_count": depth_count,
                    "schedule_weight": round(schedule_weight, 4),
                    "scarcity": round(scarcity, 4),
                    "urgency": round(urgency, 4),
                    "opportunity_cost": round(opportunity_cost, 4),
                    "trending_adds": trend,
                },
            )
        )

    candidates.sort(key=lambda c: c.incremental_utility, reverse=True)
    return candidates[:limit]
