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
    "projected_games",  # Phase 11 - the multiplier behind fantasy_pts_season
    # The volume discount, carried through so it is visible in the two
    # deliverables a reader actually opens. Both were computed upstream and
    # then dropped here, which meant a 0.15x-ed number arrived in
    # fantasy_points_<season>.csv and sleeper_comparison_<season>.csv with
    # nothing on the row saying it had been scaled at all - the project's
    # rule is that a discount must be visible in the output table, and this
    # was a live violation of it. `role_discount_factor` is the honest
    # single column (the multiplier applied, 1.0 = none);
    # `role_discount_applied` is kept beside it because it distinguishes
    # the curated committee/backup path from the deep-bench one, which the
    # factor alone cannot (both DEEP_BENCH_DISCOUNT and
    # ROLE_VOLUME_DISCOUNT['backup'] are 0.15).
    "role_discount_factor", "role_discount_applied",
    # The ladder's input (Gate B). Without it role_discount_factor is an
    # unexplainable number on the row - rank is what determines it, and the
    # curated `depth_rank` beside it is a different, shallower chart.
    "nfl_depth_rank",
]


def _score(df, value_col):
    """Sum SCORING-weighted stat columns from a wide (player, stat->value_col) frame."""
    pivot = df.pivot_table(index=KEY_COLS, columns="stat", values=value_col, aggfunc="first")
    weights = pd.Series({k: v for k, v in SCORING.items() if k in pivot.columns})
    return (pivot[weights.index] * weights).sum(axis=1)


def compute_fantasy_points(long_df):
    """long_df: the tidy projections_<season>.csv frame. Returns one row per
    player-position with fantasy_pts/fantasy_pts_low/fantasy_pts_high plus
    every raw per-game stat as its own column, for reference."""
    pts = _score(long_df, "pred_pg").rename("fantasy_pts")
    pts_low = _score(long_df, "pred_pg_low").rename("fantasy_pts_low")
    pts_high = _score(long_df, "pred_pg_high").rename("fantasy_pts_high")

    raw = long_df.pivot_table(index=KEY_COLS, columns="stat", values="pred_pg", aggfunc="first")
    raw = raw.add_prefix("pg_")

    # interval_low_n_flag: True if ANY component stat was flagged (a player's
    # total shouldn't read as fully reliable if even one input stat wasn't).
    any_low_n = long_df.groupby(KEY_COLS)["interval_low_n_flag"].any().rename("any_stat_low_n_flag")

    out = pd.concat([pts, pts_low, pts_high, raw, any_low_n], axis=1).reset_index()

    descriptive = long_df.drop_duplicates(subset=KEY_COLS)[KEY_COLS + DESCRIPTIVE_COLS]
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
        out["fantasy_pts_season"] = out["fantasy_pts"] * out["projected_games"]
        out["fantasy_pts_season_low"] = out["fantasy_pts_low"] * out["projected_games"]
        out["fantasy_pts_season_high"] = out["fantasy_pts_high"] * out["projected_games"]
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
                        "projected_games", "fantasy_pts_season", "low_confidence"]
            if c in out.columns]
    print(f"--- Top 30 by {sort_col} ---")
    print(out.sort_values(sort_col, ascending=False)[cols].head(30).to_string(index=False))
    print(f"\n{len(out)} player-position rows scored (half-PPR, 4pt passing TD) -> {out_path}")


if __name__ == "__main__":
    main()
