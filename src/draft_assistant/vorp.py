"""Value Over Replacement Player (VORP) for draft rankings.

Ranks overall boards by surplus points-per-game over a position-specific
replacement baseline for a 1QB / 2RB / 3WR / 1TE / 1FLEX roster.
"""

from __future__ import annotations

import math

import pandas as pd

DEFAULT_TEAM_COUNT = 12

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

# Absolute PPG-VORP drop that helps start a new overall tier (with 4% relative).
OVERALL_VORP_TIER_GAP = 0.75


def replacement_rank(position: str, team_count: int) -> int:
    """1-based positional rank of the replacement player."""
    starters = STARTERS.get(position)
    if starters is None:
        raise ValueError(f"Unsupported position for VORP: {position}")
    share = FLEX_SHARE.get(position, 0.0)
    n = int(team_count)
    return int(math.floor(n * starters + n * share)) + 1


def replacement_ranks(team_count: int = DEFAULT_TEAM_COUNT) -> dict[str, int]:
    return {pos: replacement_rank(pos, team_count) for pos in STARTERS}


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
    points_col: str = "fantasy_pts",
    position_col: str = "position",
) -> pd.DataFrame:
    """Add replacement_pts and vorp (floored at 0) using points per game."""
    out = df.copy()
    ranks = replacement_ranks(team_count)
    replacement_pts = pd.Series(index=out.index, dtype=float)
    vorp = pd.Series(index=out.index, dtype=float)

    for pos, group in out.groupby(position_col, sort=False):
        pos_key = str(pos)
        if pos_key not in ranks:
            replacement_pts.loc[group.index] = 0.0
            vorp.loc[group.index] = 0.0
            continue
        baseline = _kth_score(group[points_col], ranks[pos_key])
        replacement_pts.loc[group.index] = baseline
        vorp.loc[group.index] = (group[points_col].astype(float) - baseline).clip(lower=0.0)

    out["replacement_pts"] = replacement_pts
    out["vorp"] = vorp
    out["vorp_team_count"] = int(team_count)
    return out
