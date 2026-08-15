"""Assign draft tiers from projected fantasy points or VORP.

Players whose values are within a configurable gap of the tier anchor stay
grouped together; a larger drop starts a new tier. Position-specific
thresholds reflect how tightly clustered each position board is.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Absolute point drop that starts a new tier, by position.
# RB/WR boards are dense; QB has clearer cliffs; TE is sparse.
DEFAULT_TIER_GAPS: dict[str, float] = {
    "QB": 0.85,
    "RB": 0.75,
    "WR": 0.55,
    "TE": 0.50,
}

# When sorting overall (all positions), use a blended threshold.
OVERALL_TIER_GAP = 1.0

# RB/WR/TE combined board for FLEX roster slots.
FLEX_POSITIONS = frozenset({"RB", "WR", "TE"})
FLEX_TIER_GAP = 0.65


@dataclass(frozen=True)
class TierConfig:
    position_gaps: dict[str, float]
    overall_gap: float = OVERALL_TIER_GAP

    def gap_for(self, position: str | None) -> float:
        if position is None:
            return self.overall_gap
        return self.position_gaps.get(position, self.overall_gap)


def assign_tiers(
    points: pd.Series,
    *,
    gap: float,
    pct_gap: float | None = 0.035,
) -> pd.Series:
    """Return 1-based tier labels for a descending-sorted point series.

    A new tier starts when the drop to the next player exceeds `gap` points
    or `pct_gap` relative to the higher player's projection (whichever applies).
    """
    if points.empty:
        return pd.Series(dtype=int)

    values = points.astype(float).tolist()
    tiers: list[int] = [1]
    current_tier = 1
    for idx in range(1, len(values)):
        prev = values[idx - 1]
        drop = prev - values[idx]
        rel_drop = drop / prev if prev > 0 else 0.0
        cliff = drop > gap
        if pct_gap is not None:
            cliff = cliff or rel_drop > pct_gap
        if cliff:
            current_tier += 1
        tiers.append(current_tier)
    return pd.Series(tiers, index=points.index, dtype=int)


def add_tier_columns(
    df: pd.DataFrame,
    *,
    points_col: str = "fantasy_pts",
    overall_points_col: str | None = None,
    overall_gap: float | None = None,
    position_col: str = "position",
    config: TierConfig | None = None,
) -> pd.DataFrame:
    """Add overall_tier and pos_tier columns to a player dataframe.

    Positional / FLEX ranks use `points_col` (typically PPG). Overall ranks use
    `overall_points_col` when provided (typically PPG VORP).
    """
    cfg = config or TierConfig(position_gaps=DEFAULT_TIER_GAPS)
    out = df.copy()
    overall_col = overall_points_col or points_col
    overall_tier_gap = cfg.overall_gap if overall_gap is None else overall_gap

    overall_order = out.sort_values(overall_col, ascending=False)
    out.loc[overall_order.index, "overall_rank"] = range(
        1, len(overall_order) + 1
    )
    out.loc[overall_order.index, "overall_tier"] = assign_tiers(
        overall_order[overall_col],
        gap=overall_tier_gap,
        pct_gap=0.04,
    ).values

    pos_tier = pd.Series(index=out.index, dtype="Int64")
    pos_rank = pd.Series(index=out.index, dtype="Int64")
    for pos, group in out.groupby(position_col, sort=False):
        ordered = group.sort_values(points_col, ascending=False)
        pos_rank.loc[ordered.index] = range(1, len(ordered) + 1)
        pos_tier.loc[ordered.index] = assign_tiers(
            ordered[points_col],
            gap=cfg.gap_for(str(pos)),
            pct_gap=0.03,
        ).values

    out["pos_rank"] = pos_rank.astype(int)
    out["pos_tier"] = pos_tier.astype(int)

    flex_rank = pd.Series(index=out.index, dtype="Int64")
    flex_tier = pd.Series(index=out.index, dtype="Int64")
    flex_pool = out[out[position_col].isin(FLEX_POSITIONS)]
    flex_order = flex_pool.sort_values(points_col, ascending=False)
    flex_rank.loc[flex_order.index] = range(1, len(flex_order) + 1)
    flex_tier.loc[flex_order.index] = assign_tiers(
        flex_order[points_col],
        gap=FLEX_TIER_GAP,
        pct_gap=0.03,
    ).values
    out["flex_rank"] = flex_rank
    out["flex_tier"] = flex_tier

    return out
