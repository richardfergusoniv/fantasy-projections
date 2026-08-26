"""Value Over Replacement Player (VORP) for draft rankings.

Ranks overall boards by surplus season points over a position-specific
replacement baseline for a 1QB / 2RB / 3WR / 1TE / 1FLEX roster.

Two construction choices are load-bearing and were both wrong before:

* **Season points, not points per game.** A draft pick buys a season, so a
  player projected for 9 games is worth roughly half of an identical player
  projected for 17. Ranking on a per-game rate makes availability invisible at
  exactly the moment it is being paid for.
* **VORP is not floored at zero.** Flooring collapsed every sub-replacement
  player onto a single tied value -- 73% of the board on the 2026 run -- so
  everything from roughly round 9 down was ordered by whatever tiebreak the
  sort happened to land on rather than by value. Sub-replacement players still
  have a real ordering among themselves, and that ordering is most of the late
  board.

Tried and rejected: dividing sub-replacement surplus by the position's starting
slots, on the theory that a bench player is worth the chance he is ever started
and a 3.5-slot position exercises that option more often than a 1.1-slot one.
It does fix deep tight ends floating up the board (TE bias vs ECR -92.5 -> +5.8)
but it is the wrong instrument: QB and TE both start one player, so the scaling
treats them alike, while the market drafts QBs roughly twice as deep past
replacement as it drafts TEs. Net effect was worse on both consensus sources
(ADP rank correlation 0.752 -> 0.743), so the late board keeps plain signed
surplus.
"""

from __future__ import annotations

import json
import math
import os

import pandas as pd

from src.projection.contracts import ROOKIE_RANK_SCALE

DEFAULT_TEAM_COUNT = 12

# Games in a regular season. A starting slot consumes more than one player over
# a full season, because starters miss games.
SEASON_GAMES = 17.0

# Starter slots per team for the supported roster.
STARTERS: dict[str, int] = {
    "QB": 1,
    "RB": 2,
    "WR": 3,
    "TE": 1,
}

# Expected share of the single FLEX starter filled by each position (half-PPR).
FLEX_SHARE: dict[str, float] = {
    "QB": 0.0,
    "RB": 0.40,
    "WR": 0.50,
    "TE": 0.10,
}

# Fitted median realized season points by positional rank, from
# scripts/fit_position_curves.py. Used to correct the SHAPE of a position's
# surplus curve, anchored at that position's own replacement level.
CURVES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "position_curves.json")

# How much of a position's surplus-over-replacement curve to take from the
# fitted historical shape instead of our own projections. 0.0 leaves a position
# untouched.
#
# TE only, and it is not a small correction, because the distortion is not
# small. Measured against the 2022-2025 median curves with the board's own
# replacement ranks, TE's replacement level is right (106.4 against 107.2) and
# so is its tail, but TE2 through TE8 carry roughly TWICE the surplus they
# should: ratios of 1.95, 1.85, 2.12, 2.05, 2.33, 2.14 at ranks 2-8. The board
# had eight tight ends bunched between 174 and 245 points where history spreads
# them 143 to 182, which is what pushed the TE1 to 5th overall against a
# historical-implied 14th.
#
# WR and RB need no correction -- their surplus ratios are 0.85 to 1.11 across
# the same probe ranks. QB looks badly off (its curve is far flatter than the
# realized one) but is deliberately left alone: a realized QB curve is the
# steepest thing in football precisely because which of 32 near-equal starters
# finishes QB1 is close to random, so no honest expected-value projection can
# match it, and ADP agrees with where the board already puts quarterbacks.
#
# The magnitudes being replaced are not information we have reason to trust:
# they come from an elite-TE target-share floor firing on roughly seven teams a
# season. The ORDERING -- which tight ends are best -- is ours and is preserved.
POSITION_CURVE_WEIGHT: dict[str, float] = {"QB": 0.0, "RB": 0.0, "WR": 0.0, "TE": 1.0}

# Absolute season-point VORP drop that helps start a new overall tier
# (with 4% relative). Scaled from the old per-game gap of 0.75 by a 17-game
# season so tier granularity is unchanged by the switch to a season basis.
OVERALL_VORP_TIER_GAP = 12.75


def replacement_rank(
    position: str, team_count: int, availability_factor: float = 1.0
) -> int:
    """1-based positional rank of the replacement player.

    ``availability_factor`` deepens the baseline to account for starters missing
    games. A slot that must be filled for 17 weeks by players who average 15
    consumes roughly 17/15 players over the season, so the player a team
    actually ends up starting in that slot is deeper than the nominal last
    starter. The factor is measured from the board's own projected games, not
    assumed.
    """
    starters = STARTERS.get(position)
    if starters is None:
        raise ValueError(f"Unsupported position for VORP: {position}")
    share = FLEX_SHARE.get(position, 0.0)
    n = int(team_count)
    demand = (n * starters + n * share) * float(availability_factor)
    return int(math.floor(demand)) + 1


def availability_factors(
    df: pd.DataFrame,
    *,
    team_count: int = DEFAULT_TEAM_COUNT,
    position_col: str = "position",
    games_col: str = "projected_games",
) -> dict[str, float]:
    """Per-position 17/mean_projected_games over that position's starter pool.

    Returns 1.0 for any position without usable projected games, so a board that
    does not carry availability simply keeps the nominal baseline.
    """
    factors: dict[str, float] = {}
    for pos in STARTERS:
        factors[pos] = 1.0
        if games_col not in df.columns or position_col not in df.columns:
            continue
        grp = df[df[position_col] == pos]
        if grp.empty:
            continue
        depth = replacement_rank(pos, team_count) - 1
        top = grp.nlargest(max(depth, 1), "vorp_input_pts") if "vorp_input_pts" in grp.columns else grp
        games = pd.to_numeric(top[games_col], errors="coerce").dropna()
        games = games[games > 0]
        if games.empty:
            continue
        factors[pos] = float(SEASON_GAMES / games.mean())
    return factors


def replacement_ranks(
    team_count: int = DEFAULT_TEAM_COUNT,
    availability: dict[str, float] | None = None,
) -> dict[str, int]:
    availability = availability or {}
    return {
        pos: replacement_rank(pos, team_count, availability.get(pos, 1.0))
        for pos in STARTERS
    }


def load_position_curves(path: str = CURVES_PATH) -> dict[str, list[float]]:
    """Fitted historical points-by-rank curves; empty dict if the file is absent."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh).get("curves") or {}


def _kth_score(values: pd.Series, rank: int) -> float:
    ordered = values.dropna().astype(float).sort_values(ascending=False)
    if ordered.empty:
        return 0.0
    idx = min(max(int(rank), 1), len(ordered)) - 1
    return float(ordered.iloc[idx])


def add_vorp_columns(
    df: pd.DataFrame,
    *,
    team_count: int = DEFAULT_TEAM_COUNT,
    points_col: str = "fantasy_pts_season",
    position_col: str = "position",
    rookie_rank_scale: float = ROOKIE_RANK_SCALE,
    floor_at_zero: bool = False,
    adjust_replacement_for_availability: bool = True,
    curve_weight: dict[str, float] | None = None,
    curves: dict[str, list[float]] | None = None,
) -> pd.DataFrame:
    """Add replacement_pts and vorp using season points.

    VORP is signed by default: sub-replacement players keep their ordering
    instead of collapsing onto a tie at zero. Pass ``floor_at_zero=True`` for
    the old clipped behaviour.

    Rookie rows (``low_confidence`` or ``source == 'rookie_rule'``) rank on
    ``points_col × rookie_rank_scale`` without changing the stored projection.
    """
    out = df.copy()
    if curve_weight is None:
        curve_weight = POSITION_CURVE_WEIGHT
    if curves is None:
        curves = load_position_curves()
    replacement_pts = pd.Series(index=out.index, dtype=float)
    vorp = pd.Series(index=out.index, dtype=float)
    curve_applied = pd.Series(0.0, index=out.index, dtype=float)
    vorp_input = out[points_col].astype(float).copy()
    scale = float(rookie_rank_scale)
    if scale != 1.0:
        rookie_mask = pd.Series(False, index=out.index)
        if "low_confidence" in out.columns:
            rookie_mask |= out["low_confidence"].fillna(False).astype(bool)
        if "source" in out.columns:
            rookie_mask |= out["source"].astype(str).eq("rookie_rule")
        vorp_input.loc[rookie_mask] = vorp_input.loc[rookie_mask] * scale
    out["vorp_input_pts"] = vorp_input
    out["rookie_rank_scale"] = scale

    availability = (
        availability_factors(out, team_count=team_count, position_col=position_col)
        if adjust_replacement_for_availability
        else None
    )
    ranks = replacement_ranks(team_count, availability)
    out["vorp_replacement_rank"] = out[position_col].map(ranks)

    for pos, group in out.groupby(position_col, sort=False):
        pos_key = str(pos)
        if pos_key not in ranks:
            replacement_pts.loc[group.index] = 0.0
            vorp.loc[group.index] = 0.0
            continue
        baseline = _kth_score(group["vorp_input_pts"], ranks[pos_key])
        replacement_pts.loc[group.index] = baseline
        surplus = group["vorp_input_pts"].astype(float) - baseline

        weight = float(curve_weight.get(pos_key, 0.0)) if curve_weight else 0.0
        curve = curves.get(pos_key) if curves else None
        if weight > 0.0 and curve:
            # Blend our surplus curve toward the fitted historical one at the
            # same positional rank. Both are anchored at this position's own
            # replacement level, so this corrects shape, not level, and cannot
            # reorder the position -- both curves are monotone in rank.
            pos_rank = surplus.rank(ascending=False, method="first")
            curve_repl = (
                curve[min(ranks[pos_key], len(curve)) - 1] if curve else baseline
            )

            def _curve_surplus(r: float) -> float:
                idx = min(max(int(r), 1), len(curve)) - 1
                return float(curve[idx]) - float(curve_repl)

            hist_surplus = pos_rank.map(_curve_surplus)
            # Ranks past the fitted depth keep our own surplus: the tail is
            # where our curves already agree with history.
            beyond = pos_rank > len(curve)
            blended = (1.0 - weight) * surplus + weight * hist_surplus
            surplus = surplus.where(beyond, blended)
            curve_applied.loc[group.index] = weight

        if floor_at_zero:
            vorp.loc[group.index] = surplus.clip(lower=0.0)
            continue
        vorp.loc[group.index] = surplus

    out["replacement_pts"] = replacement_pts
    out["vorp"] = vorp
    out["vorp_curve_weight"] = curve_applied
    out["vorp_team_count"] = int(team_count)
    out.attrs["vorp_replacement_ranks"] = ranks
    out.attrs["vorp_curve_weight"] = dict(curve_weight)
    out.attrs["vorp_availability_factors"] = availability or {}
    return out
