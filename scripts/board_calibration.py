"""One-shot cross-position calibration summary for the draft board.

Reports, per position, the ratio of projected top-N season points to the
realized top-N total, and that ratio relative to WR. WR is the reference
because an expected-value projection is legitimately flatter than a realized
order statistic (the realized "WR12" is the ex-post winner of a survivorship
race), so the absolute ratio is expected to sit below 1.0 for every position.
What is NOT legitimate is one position sitting far above the others: that is a
cross-position scale error, and it lands directly on overall draft rank.

Choose the history window deliberately. Position shares are not stationary:
TE's share of the receiving pie ran 21.1% in 2021, 21.7% in 2023, 22.5% in 2024
and 24.1% in 2025. Benchmarking a 2026 projection against a 2019-2024 mean
therefore understates the current TE level by more than a point and invites an
over-correction. Prefer the most recent two or three seasons, and check the
trend before trusting any multi-year average.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts._history import realized_weekly  # noqa: E402

SCORING = {
    "passing_yards": 1 / 25, "passing_tds": 4, "interceptions": -2,
    "rushing_yards": 1 / 10, "rushing_tds": 6,
    "receiving_yards": 1 / 10, "receiving_tds": 6, "receptions": 0.5,
}
DEPTH = {"QB": 24, "RB": 48, "WR": 60, "TE": 18}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--history", default="2024,2025")
    ap.add_argument("--curve", action="store_true", help="also print the top-12 curves")
    ap.add_argument("--fantasy-path", default=None)
    args = ap.parse_args()
    seasons = [int(s) for s in args.history.split(",")]

    hist = realized_weekly(seasons, list(SCORING))
    hist["pts"] = sum(hist[c] * w for c, w in SCORING.items())
    hist["r"] = hist.groupby(["season", "position"])["pts"].rank(ascending=False, method="first")

    fantasy_path = args.fantasy_path or os.path.join(
        "output", f"fantasy_points_{args.season}.csv"
    )
    proj = pd.read_csv(fantasy_path)
    proj["r"] = proj.groupby("position")["fantasy_pts_season"].rank(ascending=False, method="first")

    ratios = {}
    for pos, n in DEPTH.items():
        a = proj[(proj.position == pos) & (proj.r <= n)]["fantasy_pts_season"].sum()
        b = hist[(hist.position == pos) & (hist.r <= n)].groupby("season")["pts"].sum().mean()
        ratios[pos] = (a, b, a / b)

    print(f"{'pos':<5}{'N':>4}{'projected':>11}{'realized':>11}{'ratio':>8}{'vs WR':>8}")
    wr = ratios["WR"][2]
    for pos, n in DEPTH.items():
        a, b, r = ratios[pos]
        print(f"{pos:<5}{n:>4}{a:>11.0f}{b:>11.0f}{r:>8.3f}{r / wr:>8.3f}")
    spread = max(r / wr for _, _, r in ratios.values()) - min(r / wr for _, _, r in ratios.values())
    print(f"\ncross-position spread (max-min of vs-WR) = {spread:.3f}   [0.000 is perfect]")

    if args.curve:
        for pos in DEPTH:
            pr = [round(x) for x in proj[(proj.position == pos) & (proj.r <= 12)]
                  .sort_values("r")["fantasy_pts_season"].tolist()]
            hr = [round(hist[(hist.position == pos) & (hist.r == i)]["pts"].median())
                  for i in range(1, 13)]
            print(f"\n {pos} proj: {pr}")
            print(f" {pos} hist: {hr}")


if __name__ == "__main__":
    main()
