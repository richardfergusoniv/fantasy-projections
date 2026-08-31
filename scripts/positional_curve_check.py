"""Compare our projected positional point curves to realized history.

A consensus-free calibration check. If our TE8 is projected well above what
TE8 has actually finished with in recent seasons, that is a cross-position
scale defect in the projections, not a disagreement with the market.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts._history import realized_weekly  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORING = {
    "passing_yards": 1 / 25,
    "passing_tds": 4,
    "interceptions": -2,
    "rushing_yards": 1 / 10,
    "rushing_tds": 6,
    "receiving_yards": 1 / 10,
    "receiving_tds": 6,
    "receptions": 0.5,
}

POSITIONS = ("QB", "RB", "WR", "TE")
PROBE_RANKS = {
    "QB": [1, 3, 6, 9, 12, 15, 18, 24],
    "RB": [1, 3, 6, 12, 18, 24, 30, 36, 48],
    "WR": [1, 3, 6, 12, 18, 24, 36, 48, 60],
    "TE": [1, 3, 6, 9, 12, 15, 18, 24],
}


def realized_curves(seasons: list[int]) -> pd.DataFrame:
    df = realized_weekly(seasons, list(SCORING), positions=POSITIONS)
    df["pts"] = sum(df[c] * w for c, w in SCORING.items())
    df["pos_rank"] = df.groupby(["season", "position"])["pts"].rank(
        ascending=False, method="first"
    )
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--history", default="2022,2023,2024,2025")
    ap.add_argument("--board", default=None)
    args = ap.parse_args()

    seasons = [int(s) for s in args.history.split(",")]
    hist = realized_curves(seasons)

    board_path = args.board or os.path.join(REPO_ROOT, "output", f"fantasy_points_{args.season}.csv")
    proj = pd.read_csv(board_path)
    proj = proj[proj["position"].isin(POSITIONS)].copy()
    proj["pos_rank"] = proj.groupby("position")["fantasy_pts_season"].rank(
        ascending=False, method="first"
    )

    print(f"Projected {args.season} season points vs realized {seasons} (half-PPR, 4pt pass TD)")
    print(f"{'pos':<4} {'rank':>5} {'ours':>8} {'hist_med':>9} {'hist_min':>9} {'hist_max':>9} {'delta':>8} {'ratio':>7}")
    rows = []
    for pos in POSITIONS:
        for rank in PROBE_RANKS[pos]:
            ours = proj[(proj["position"] == pos) & (proj["pos_rank"] == rank)]
            if ours.empty:
                continue
            ours_pts = float(ours["fantasy_pts_season"].iloc[0])
            h = hist[(hist["position"] == pos) & (hist["pos_rank"] == rank)]["pts"]
            if h.empty:
                continue
            med = float(h.median())
            delta = ours_pts - med
            ratio = ours_pts / med if med else float("nan")
            rows.append((pos, rank, ours_pts, med, ratio))
            print(
                f"{pos:<4} {rank:>5} {ours_pts:>8.1f} {med:>9.1f} {float(h.min()):>9.1f} "
                f"{float(h.max()):>9.1f} {delta:>+8.1f} {ratio:>7.3f}"
            )
    print("\nMean ratio (ours / historical median) by position — 1.00 is calibrated:")
    df = pd.DataFrame(rows, columns=["pos", "rank", "ours", "hist", "ratio"])
    for pos in POSITIONS:
        sub = df[df["pos"] == pos]
        if len(sub):
            print(f"  {pos}: {sub['ratio'].mean():.3f}   (n={len(sub)} probe ranks)")


if __name__ == "__main__":
    main()
