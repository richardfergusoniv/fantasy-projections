"""Deterministic joint draw construction for decision engines.

Why this module exists
----------------------
Decision engines need *fantasy-point draws per player per simulation index* so
that matchup win probability, milestone bonuses, and points-per-first-down are
evaluated on realized outcomes rather than on means.

Two draw sources are supported, and the distinction is deliberately visible to
callers because it changes what the league scoring contract can express:

``stat_level``
    Draws carry component statistics (yards, receptions, first downs, tiered
    defensive inputs). The league contract is applied exactly, including
    threshold and bracket rules, on every draw.

``baseline_points_only``
    Draws carry only fantasy points produced under the projection release's own
    baseline scoring. League-specific *linear* differences cannot be recovered
    and nonlinear rules cannot be evaluated at all. Recommendations built from
    this source are labelled, and the specific rules that could not be applied
    are reported, so the output can never be mistaken for league-exact output.

Randomness
----------
Each player is sampled from its own generator seeded by a stable digest of
(player id, run id, week, salt). Consequences:

* Results are reproducible for a given projection run.
* Players are drawn **independently**, so a draw index is a joint sample across
  the roster rather than a shared percentile. Perfectly correlated percentile
  draws would make every matchup probability collapse to 0 or 1.
* Independence is an explicit modelling assumption. There is no team-level
  correlation (game environment, QB/WR stacking) in this layer; see
  ``docs/PRODUCTION_READINESS_AUDIT.md`` for that limitation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

from src.app.scoring.compiler import score_stat_draw
from src.app.scoring.contract import ScoringContract

DrawMode = Literal["stat_level", "baseline_points_only", "actual", "mixed"]

#: Draw count for interactive decisions. The 10,000-draw profile in
#: ``Settings.simulation_draw_count`` remains the publish-time profile; decision
#: endpoints use a smaller count for latency and report Monte Carlo error
#: alongside every probability so the difference is visible rather than hidden.
DEFAULT_DRAW_COUNT = 2000


def stable_seed(*parts: object) -> int:
    """A process-stable seed. Python's ``hash()`` is randomized per process."""
    joined = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _quantile(quantiles: dict[str, float], key: str, fallback: float) -> float:
    for candidate in (key, f"{float(key):g}", f"{float(key):.1f}"):
        if candidate in quantiles:
            try:
                return float(quantiles[candidate])
            except (TypeError, ValueError):
                continue
    return fallback


@dataclass
class PlayerDraws:
    """Per-player fantasy-point draws plus provenance."""

    player_id: str
    position: str
    mode: DrawMode
    points: np.ndarray
    #: Present only in ``stat_level`` mode.
    stat_draws: list[dict[str, float]] | None = None
    availability_probability: float = 1.0
    #: True when the player's week is already decided (already played / locked).
    locked: bool = False
    projected_mean: float = 0.0

    def __len__(self) -> int:
        return int(self.points.shape[0])

    @property
    def mean(self) -> float:
        return float(self.points.mean()) if len(self) else 0.0

    def percentile(self, q: float) -> float:
        if not len(self):
            return 0.0
        return float(np.quantile(self.points, q))


@dataclass
class DrawSet:
    """A joint set of player draws sharing one simulation index space."""

    players: dict[str, PlayerDraws]
    draw_count: int
    mode: DrawMode
    contract_hash: str
    #: League rules that the draw source could not express.
    unapplied_rules: list[str] = field(default_factory=list)
    seed_salt: str = ""
    #: Players whose published uncertainty band had to be recentred on the
    #: promoted mean because the release publishes them from different models.
    recentred_players: list[str] = field(default_factory=list)

    def get(self, player_id: str) -> PlayerDraws | None:
        return self.players.get(player_id)

    def positions(self) -> dict[str, str]:
        return {pid: p.position for pid, p in self.players.items()}

    def totals_for(self, player_ids: Sequence[str]) -> np.ndarray:
        """Summed point draws for a lineup, aligned on the simulation index."""
        total = np.zeros(self.draw_count, dtype=float)
        for pid in player_ids:
            player = self.players.get(pid)
            if player is None:
                continue
            total += player.points
        return total

    @property
    def scoring_fidelity(self) -> str:
        return self.mode

    def fidelity_note(self) -> str:
        if self.mode == "stat_level":
            return "League scoring applied exactly to component-stat draws."
        if self.mode == "mixed":
            return (
                "Team defense and kicker draws carry component statistics and are "
                "scored exactly. Offensive players come from a fantasy-point-only "
                "projection release, so league-specific rules listed in "
                "unapplied_rules could not be re-applied to them."
            )
        return (
            "Projection artifacts expose fantasy-point summaries only, so "
            "league-specific scoring rules could not be re-applied. "
            f"Unapplied rules: {self.unapplied_rules or ['none']}."
        )

    def players_by_mode(self) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        for pid, player in self.players.items():
            grouped.setdefault(player.mode, []).append(pid)
        return {mode: sorted(ids) for mode, ids in grouped.items()}


#: Relative tolerance before a published quantile envelope is treated as being
#: on a different scale from the published mean.
LOCATION_TOLERANCE = 0.05


def reconcile_location(
    *, mean_points: float, p10: float, p50: float, p90: float
) -> tuple[float, float, float, bool]:
    """Recentre a quantile envelope on the promoted mean, preserving its shape.

    The active release publishes means from one model (the accuracy-first
    ensemble) and its uncertainty band from a separate distributional overlay.
    The release notes state the overlay is "distributional only; means unchanged",
    so the two series are deliberately on different locations: for some players
    the published p10 sits *above* the published mean.

    Consuming both as-is would let the overlay silently override the promoted
    point forecast, which is what drove start/sit recommendations before this
    fix. Instead the envelope is rescaled multiplicatively so its median equals
    the promoted mean while its relative spread is preserved. Multiplicative
    rescaling keeps the band non-negative and keeps p10/p90 ratios intact.

    Returns ``(p10, p50, p90, was_recentred)``.
    """
    if mean_points <= 0:
        # No usable location; fall back to a proportional band around zero.
        return 0.0, 0.0, 0.0, False
    if p50 <= 0:
        return (
            mean_points * 0.6,
            mean_points,
            mean_points * 1.4,
            True,
        )
    if abs(p50 - mean_points) <= LOCATION_TOLERANCE * mean_points:
        return p10, p50, p90, False
    scale = mean_points / p50
    return p10 * scale, mean_points, p90 * scale, True


def _sample_from_quantiles(
    rng: np.random.Generator,
    *,
    count: int,
    p10: float,
    p50: float,
    p90: float,
    floor: float | None = 0.0,
) -> np.ndarray:
    """Sample from a distribution pinned to p10/p50/p90.

    Uses the standard-normal quantile mapping: a uniform draw is converted to a
    z-score and then mapped through two half-normal scales, so the resulting
    sample reproduces the supplied 10th, 50th, and 90th percentiles and keeps a
    plausible asymmetric shape between them. This is interpolation of a supplied
    marginal, not an independent forecast.
    """
    z90 = 1.2815515655446004  # Phi^-1(0.9)
    lower_scale = max((p50 - p10) / z90, 1e-9)
    upper_scale = max((p90 - p50) / z90, 1e-9)
    z = rng.standard_normal(count)
    values = np.where(z < 0, p50 + z * lower_scale, p50 + z * upper_scale)
    if floor is not None:
        values = np.maximum(values, floor)
    return values


def build_points_only_draws(
    summaries: Sequence[object],
    *,
    contract: ScoringContract,
    run_id: str,
    week: int | None,
    draw_count: int = DEFAULT_DRAW_COUNT,
    locked_player_ids: Sequence[str] = (),
    actual_points: dict[str, float] | None = None,
    seed_salt: str = "",
) -> DrawSet:
    """Build a joint draw set from fantasy-point summaries.

    ``summaries`` items must expose ``player_id``, ``position``, ``mean_points``,
    ``quantiles`` and optionally ``availability_probability``.
    """
    actual_points = actual_points or {}
    locked = set(locked_player_ids)
    players: dict[str, PlayerDraws] = {}
    recentred_players: list[str] = []

    for summary in summaries:
        player_id = str(getattr(summary, "player_id"))
        position = str(getattr(summary, "position", "RB") or "RB")
        mean_points = float(getattr(summary, "mean_points", 0.0) or 0.0)
        quantiles = dict(getattr(summary, "quantiles", {}) or {})
        availability = float(getattr(summary, "availability_probability", 1.0) or 0.0)
        availability = min(max(availability, 0.0), 1.0)

        if player_id in actual_points:
            # Already played: the week is decided, so there is no uncertainty.
            value = float(actual_points[player_id])
            players[player_id] = PlayerDraws(
                player_id=player_id,
                position=position,
                mode="actual",
                points=np.full(draw_count, value, dtype=float),
                availability_probability=1.0,
                locked=True,
                projected_mean=value,
            )
            continue

        p50 = _quantile(quantiles, "0.5", mean_points)
        p10 = _quantile(quantiles, "0.1", mean_points * 0.6)
        p90 = _quantile(quantiles, "0.9", mean_points * 1.4)
        p10, p50, p90 = sorted((p10, p50, p90))
        p10, p50, p90, recentred = reconcile_location(
            mean_points=mean_points, p10=p10, p50=p50, p90=p90
        )
        if recentred:
            recentred_players.append(player_id)

        rng = np.random.default_rng(
            stable_seed(player_id, run_id, week, seed_salt) % (2**63)
        )
        points = _sample_from_quantiles(
            rng, count=draw_count, p10=p10, p50=p50, p90=p90
        )
        if availability < 1.0:
            inactive = rng.random(draw_count) >= availability
            points = np.where(inactive, 0.0, points)

        players[player_id] = PlayerDraws(
            player_id=player_id,
            position=position,
            mode="baseline_points_only",
            points=points,
            availability_probability=availability,
            locked=player_id in locked,
            projected_mean=mean_points,
        )

    return DrawSet(
        players=players,
        draw_count=draw_count,
        mode="baseline_points_only",
        contract_hash=contract.contract_hash,
        unapplied_rules=_unapplied_rules(
            contract, {p.position for p in players.values()}
        ),
        seed_salt=seed_salt,
        recentred_players=sorted(recentred_players),
    )


def build_stat_level_draws(
    stat_draws_by_player: dict[str, list[dict[str, float]]],
    positions: dict[str, str],
    *,
    contract: ScoringContract,
    locked_player_ids: Sequence[str] = (),
) -> DrawSet:
    """Build a joint draw set by scoring component-stat draws exactly.

    This is the correct path for points-per-first-down and yardage-bonus
    leagues: every threshold and bracket rule is evaluated per draw.
    """
    if not stat_draws_by_player:
        return DrawSet(
            players={},
            draw_count=0,
            mode="stat_level",
            contract_hash=contract.contract_hash,
        )

    draw_count = min(len(draws) for draws in stat_draws_by_player.values())
    locked = set(locked_player_ids)
    players: dict[str, PlayerDraws] = {}

    for player_id, draws in stat_draws_by_player.items():
        position = positions.get(player_id, "RB")
        scored = np.array(
            [
                score_stat_draw(draws[idx], contract, position=position)
                for idx in range(draw_count)
            ],
            dtype=float,
        )
        players[player_id] = PlayerDraws(
            player_id=player_id,
            position=position,
            mode="stat_level",
            points=scored,
            stat_draws=list(draws[:draw_count]),
            locked=player_id in locked,
            projected_mean=float(scored.mean()) if draw_count else 0.0,
        )

    return DrawSet(
        players=players,
        draw_count=draw_count,
        mode="stat_level",
        contract_hash=contract.contract_hash,
        unapplied_rules=[],
    )


def build_draw_set(
    *,
    contract: ScoringContract,
    run_id: str,
    week: int | None,
    draw_count: int = DEFAULT_DRAW_COUNT,
    point_summaries: Sequence[object] = (),
    stat_draws_by_player: dict[str, list[dict[str, float]]] | None = None,
    positions: dict[str, str] | None = None,
    locked_player_ids: Sequence[str] = (),
    actual_points: dict[str, float] | None = None,
    seed_salt: str = "",
) -> DrawSet:
    """Build one joint draw set that may mix stat-level and points-only players.

    This mixed case is the real production shape today: team defenses and kickers
    are simulated at stat level (so tiered points-allowed and field-goal-distance
    rules score exactly), while offensive players come from a release bundle that
    only publishes fantasy-point summaries. The resulting mode is reported as
    ``mixed`` and the unapplied-rule list describes what the points-only portion
    could not express.
    """
    points_set = build_points_only_draws(
        point_summaries,
        contract=contract,
        run_id=run_id,
        week=week,
        draw_count=draw_count,
        locked_player_ids=locked_player_ids,
        actual_points=actual_points,
        seed_salt=seed_salt,
    )
    merged: dict[str, PlayerDraws] = dict(points_set.players)

    if stat_draws_by_player:
        stat_set = build_stat_level_draws(
            {
                pid: draws[:draw_count]
                for pid, draws in stat_draws_by_player.items()
                if len(draws) >= 1
            },
            positions or {},
            contract=contract,
            locked_player_ids=locked_player_ids,
        )
        for pid, player in stat_set.players.items():
            if player.points.shape[0] < draw_count:
                # Tile the available draws up to the shared index length so every
                # player spans the same simulation space.
                reps = int(np.ceil(draw_count / max(player.points.shape[0], 1)))
                player.points = np.tile(player.points, reps)[:draw_count]
            merged[pid] = player

    modes = {player.mode for player in merged.values() if player.mode != "actual"}
    if not modes:
        mode: DrawMode = "baseline_points_only"
    elif modes == {"stat_level"}:
        mode = "stat_level"
    elif modes == {"baseline_points_only"}:
        mode = "baseline_points_only"
    else:
        mode = "mixed"  # type: ignore[assignment]

    if mode == "stat_level":
        unapplied: list[str] = []
    else:
        # Only report rules the *points-only* players could not express. A tiered
        # points-allowed rule is applied exactly when every defense in the set is
        # stat-level, so listing it as unapplied would be misleading.
        points_only_positions = {
            player.position
            for player in merged.values()
            if player.mode == "baseline_points_only"
        }
        unapplied = _unapplied_rules(contract, points_only_positions)

    recentred = list(points_set.recentred_players)
    return DrawSet(
        players=merged,
        draw_count=draw_count,
        mode=mode,
        contract_hash=contract.contract_hash,
        recentred_players=recentred,
        unapplied_rules=unapplied,
        seed_salt=seed_salt,
    )


#: Stats that only a team defense produces, and only kickers produce.
DEFENSE_ONLY_STATS = frozenset(
    {
        "points_allowed",
        "yards_allowed",
        "sacks",
        "interceptions",
        "fumble_recoveries",
        "forced_fumbles",
        "def_tds",
        "safeties",
        "blocked_kicks",
        "def_two_pt",
        "fourth_down_stops",
        "three_and_outs",
        "punt_return_tds",
        "kick_return_tds",
        "blocked_kick_return_yards",
    }
)
KICKER_ONLY_STATS = frozenset(
    {
        "fgm",
        "fgm_0_19",
        "fgm_20_29",
        "fgm_30_39",
        "fgm_40_49",
        "fgm_50p",
        "fgmiss",
        "fgmiss_0_19",
        "fgmiss_20_29",
        "fgmiss_30_39",
        "fgmiss_40_49",
        "fgmiss_50p",
        "xpm",
        "xpmiss",
    }
)


def _rule_is_reachable(stat: str, points_only_positions: set[str] | None) -> bool:
    """True when a points-only player could have been affected by this rule."""
    if points_only_positions is None:
        return True
    if stat in DEFENSE_ONLY_STATS:
        return bool(points_only_positions & {"DEF", "DST"})
    if stat in KICKER_ONLY_STATS:
        return "K" in points_only_positions
    return bool(points_only_positions - {"K", "DEF", "DST"})


def _unapplied_rules(
    contract: ScoringContract, points_only_positions: set[str] | None = None
) -> list[str]:
    """League rules a points-only draw source cannot reproduce.

    ``points_only_positions`` restricts the report to rules that could actually
    have applied to a player drawn from point summaries.
    """
    unapplied: list[str] = []
    for rule in contract.threshold_rules:
        if _rule_is_reachable(rule.stat, points_only_positions):
            unapplied.append(f"threshold:{rule.stat}{rule.comparison}{rule.threshold:g}")
    for rule in contract.bracket_rules:
        if _rule_is_reachable(rule.stat, points_only_positions):
            unapplied.append(f"bracket:{rule.stat}:{rule.group}")
    for rule in contract.first_down_rules():
        if _rule_is_reachable(rule.stat, points_only_positions):
            unapplied.append(f"linear:{rule.stat}")
    for rule in contract.linear_rules:
        if rule.positions and _rule_is_reachable(rule.stat, points_only_positions):
            unapplied.append(f"position_premium:{rule.stat}:{'/'.join(rule.positions)}")
    return sorted(set(unapplied))
