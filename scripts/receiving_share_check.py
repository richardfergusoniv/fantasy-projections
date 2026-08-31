"""Position share of the team receiving pie: projected vs realized.

Shares are scale-free, so the league-wide ~0.84 expected-value deflation
cancels out. Also reports shares restricted to the top-N players per position,
which controls for our projection universe covering a different depth of tail
than the realized population does.

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

STATS = ["targets", "receptions", "receiving_yards", "receiving_tds"]
RECV_POS = ("RB", "WR", "TE")

def realized(seasons: list[int]) -> pd.DataFrame:
    return realized_weekly(seasons, STATS, positions=RECV_POS)


def projected(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["position"].isin(RECV_POS)]
    wide = df.pivot_table(
        index=["player_id", "position"], columns="stat", values="pred_season", aggfunc="first"
    ).reset_index()
    for s in STATS:
        if s not in wide.columns:
            wide[s] = 0.0
    return wide


def shares(df: pd.DataFrame, top_n: dict[str, int] | None = None) -> pd.Series:
    d = df
    if top_n:
        keep = []
        for pos, n in top_n.items():
            sub = d[d["position"] == pos].nlargest(n, "targets")
            keep.append(sub)
        d = pd.concat(keep)
    tot = d.groupby("position")[STATS].sum()
    return tot / tot.sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--history", default="2024,2025")
    ap.add_argument("--projections", default=None)
    args = ap.parse_args()

    seasons = [int(s) for s in args.history.split(",")]
    hist = realized(seasons)
    proj = projected(args.projections or os.path.join("output", f"projections_{args.season}.csv"))

    print("Player counts with any projected/realized targets:")
    for pos in RECV_POS:
        p = int((proj[(proj.position == pos) & (proj.targets > 0)]).shape[0])
        h = hist[(hist.position == pos) & (hist.targets > 0)].groupby("season").size().mean()
        print(f"  {pos}: projected {p:>4}   realized/season {h:>6.0f}")

    for label, top_n in (
        ("FULL universe", None),
        ("top 36 RB / 72 WR / 24 TE", {"RB": 36, "WR": 72, "TE": 24}),
        ("top 48 RB / 96 WR / 32 TE", {"RB": 48, "WR": 96, "TE": 32}),
    ):
        ps = shares(proj, top_n)
        hs_all = []
        for s in seasons:
            hs_all.append(shares(hist[hist.season == s], top_n))
        hs = sum(hs_all) / len(hs_all)
        print(f"\n=== Position share of receiving pie — {label} ===")
        print(f"{'stat':<17} {'pos':<4} {'projected':>10} {'realized':>10} {'delta':>8} {'rel':>7}")
        for stat in STATS:
            for pos in RECV_POS:
                p, h = float(ps.loc[pos, stat]), float(hs.loc[pos, stat])
                print(
                    f"{stat:<17} {pos:<4} {p:>9.1%} {h:>10.1%} {p - h:>+8.1%} {p / h:>7.3f}"
                )
            print()


if __name__ == "__main__":
    main()
