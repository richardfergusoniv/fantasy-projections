"""Convert output/projections_<season>.csv (one row per player-stat) into
fantasy points, half-PPR / 4-pt-passing-TD scoring (the user's league
settings). Fumbles lost and 2pt conversions are NOT modeled anywhere
upstream (no such target stat exists in TARGET_STATS), so they're absent
from the point total by construction, not silently zeroed - this is a real
gap for a player who fumbles a lot, stated here rather than hidden.

Usage: `python -m src.projection.fantasy_points --season 2026`
"""
import argparse
import os

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(REPO_ROOT, "output")

# points per unit of each raw stat. Missing stats (e.g. a WR has no
# "attempts") simply don't contribute - handled by the pivot naturally
# leaving those columns NaN, treated as 0 in the weighted sum.
SCORING = {
    "passing_yards": 1 / 25,
    "passing_tds": 4,
    "interceptions": -2,
    "rushing_yards": 1 / 10,
    "rushing_tds": 6,
    "receiving_yards": 1 / 10,
    "receiving_tds": 6,
    "receptions": 0.5,
    # attempts, completions, carries, targets: no direct point value in
    # this scoring format, excluded from the weighted sum on purpose.
}

# Stable key to pivot on - every other descriptive column (display_name,
# roster_status, depth_rank, etc.) is constant per (player_id, position,
# season) but some can be NaN (e.g. a rookie's roster_status), and
# pivot_table silently drops rows with any NaN in its index - so those
# columns are re-attached afterward via a separate lookup, not pivoted on.
KEY_COLS = ["player_id", "position", "season"]
DESCRIPTIVE_COLS = [
    "display_name", "team", "source", "low_confidence", "rookie_tier",
    "team_changed", "roster_status", "depth_rank", "role", "depth_chart_status",
    "projected_games",  # independent offensive-appearance forecast
    # The volume discount, carried through so it is visible in the two
    # deliverables a reader actually opens. Both were computed upstream and
    # then dropped here, which meant a depth-scaled number arrived in
    # fantasy_points_<season>.csv and sleeper_comparison_<season>.csv with
    # nothing on the row saying it had been scaled at all - the project's
    # rule is that a discount must be visible in the output table, and this
    # was a live violation of it. `role_discount_factor` is the honest
    # single column (the multiplier applied, 1.0 = none);
    # `role_discount_applied` is kept beside it as a direct audit boolean.
    "role_discount_factor", "role_discount_applied",
    # The ladder's input (Gate B). Without it role_discount_factor is an
    # unexplainable number on the row - rank is what determines it, and the
    # curated `depth_rank` beside it is a different, shallower chart.
    "nfl_depth_rank",
    "projected_volume_games",  # canonical season-total multiplier (QB rooms sum to 17)
    "athletic_tier", "target_depth_rank", "rookie_depth_band",
    "rookie_vacancy_scale", "rookie_id_unresolved",
    "rookie_availability_cell_n", "rookie_availability_fallback_used",
    "team_pass_catch_ratio_pre_normalization",
    "team_pass_catch_pre_normalization_flag",
    "team_pass_catch_ratio", "team_pass_catch_coherence_flag",
]


def _score(df, value_col):
    """Sum SCORING-weighted stat columns from a wide (player, stat->value_col) frame."""
    pivot = df.pivot_table(index=KEY_COLS, columns="stat", values=value_col, aggfunc="first")
    weights = pd.Series({k: v for k, v in SCORING.items() if k in pivot.columns})
    return (pivot[weights.index] * weights).sum(axis=1)


def _score_interval(df, bound):
    """Componentwise fantasy-point envelope with correct score direction.

    For positive-scoring stats, a low fantasy bound uses the stat's low
    endpoint. For negative-scoring stats (currently interceptions), it must
    use the stat's *high* endpoint; more interceptions means fewer fantasy
    points. The high bound reverses those choices.

    These remain componentwise envelopes, not a claim that independently
    estimated marginal quantiles form a calibrated joint 80% interval.
    """
    if bound not in {"low", "high"}:
        raise ValueError("bound must be 'low' or 'high'")
    pieces = []
    for stat, weight in SCORING.items():
        if stat not in set(df["stat"]):
            continue
        if bound == "low":
            endpoint = "pred_pg_low" if weight >= 0 else "pred_pg_high"
        else:
            endpoint = "pred_pg_high" if weight >= 0 else "pred_pg_low"
        values = (
            df[df["stat"] == stat]
            .set_index(KEY_COLS)[endpoint]
            .mul(weight)
            .rename(stat)
        )
        pieces.append(values)
    if not pieces:
        return pd.Series(dtype=float)
    # min_count=1 avoids silently converting an entirely unavailable
    # interval into a plausible-looking zero.
    return pd.concat(pieces, axis=1).sum(axis=1, min_count=1)


def compute_fantasy_points(long_df, floor_low_at_zero=True):
    """long_df: the tidy projections_<season>.csv frame. Returns one row per
    player-position with fantasy_pts/fantasy_pts_low/fantasy_pts_high plus
    every raw per-game stat as its own column, for reference."""
    pts = _score(long_df, "pred_pg").rename("fantasy_pts")
    pts_low_raw = _score_interval(long_df, "low").rename("fantasy_pts_low_raw")
    floor_applied = (pts_low_raw < 0).rename("fantasy_low_floor_applied")
    pts_low = (
        pts_low_raw.clip(lower=0) if floor_low_at_zero else pts_low_raw
    ).rename("fantasy_pts_low")
    pts_high = _score_interval(long_df, "high").rename("fantasy_pts_high")

    raw = long_df.pivot_table(index=KEY_COLS, columns="stat", values="pred_pg", aggfunc="first")
    raw = raw.add_prefix("pg_")

    # interval_low_n_flag: True if ANY component stat was flagged (a player's
    # total shouldn't read as fully reliable if even one input stat wasn't).
    any_low_n = long_df.groupby(KEY_COLS)["interval_low_n_flag"].any().rename("any_stat_low_n_flag")

    any_constraint = None
    if "stat_constraint_applied" in long_df.columns:
        any_constraint = (
            long_df.groupby(KEY_COLS)["stat_constraint_applied"]
            .any()
            .rename("any_stat_constraint_applied")
        )

    pieces = [pts, pts_low_raw, pts_low, pts_high, floor_applied, raw, any_low_n]
    if any_constraint is not None:
        pieces.append(any_constraint)
    out = pd.concat(pieces, axis=1).reset_index()

    available_descriptive = [c for c in DESCRIPTIVE_COLS if c in long_df.columns]
    descriptive = long_df.drop_duplicates(subset=KEY_COLS)[KEY_COLS + available_descriptive]
    out = out.merge(descriptive, on=KEY_COLS, how="left")

    # Season-long value (Phase 11) = per-game points x projected games.
    # Kept as a SEPARATE column rather than replacing fantasy_pts, because
    # the two answer different questions: fantasy_pts is the start/sit
    # number ("how good is he in a game he plays"), fantasy_pts_season is
    # the draft number ("what is he worth over a season"). Shipping only
    # the per-game figure made an 8-game player look identical to a
    # 16-game player at the same rate. NaN where no availability estimate
    # exists (an older models/ directory) rather than silently assuming a
    # full season - see predict.load_availability_models.
    if "projected_games" in out.columns:
        exposure = (
            out["projected_volume_games"].fillna(out["projected_games"])
            if "projected_volume_games" in out.columns else out["projected_games"]
        )
        out["fantasy_pts_season"] = out["fantasy_pts"] * exposure
        out["fantasy_pts_season_low"] = out["fantasy_pts_low"] * exposure
        out["fantasy_pts_season_high"] = out["fantasy_pts_high"] * exposure
    return out.sort_values("fantasy_pts", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--in-path", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    in_path = args.in_path or os.path.join(OUTPUT_DIR, f"projections_{args.season}.csv")
    out_path = args.out or os.path.join(OUTPUT_DIR, f"fantasy_points_{args.season}.csv")

    long_df = pd.read_csv(in_path)
    out = compute_fantasy_points(long_df)
    out.to_csv(out_path, index=False)

    pd.set_option("display.width", 200)
    sort_col = "fantasy_pts_season" if "fantasy_pts_season" in out.columns else "fantasy_pts"
    cols = [c for c in ["display_name", "team", "position", "fantasy_pts",
                        "projected_games", "projected_volume_games",
                        "fantasy_pts_season", "low_confidence"]
            if c in out.columns]
    print(f"--- Top 30 by {sort_col} ---")
    print(out.sort_values(sort_col, ascending=False)[cols].head(30).to_string(index=False))
    print(f"\n{len(out)} player-position rows scored (half-PPR, 4pt passing TD) -> {out_path}")


if __name__ == "__main__":
    main()
