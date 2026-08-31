"""Team target share by position and within-position rank: projected vs realized.

The aggregate position share can look calibrated while the top of every depth
chart is starved, because this pipeline projects a much deeper universe than
the league uses and the tail absorbs the difference. The draft board only ever
reads the top of each room, so this is the calibration target that matters.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts._history import realized_weekly  # noqa: E402

RECV_POS = ("WR", "RB", "TE")
RANKS = (1.0, 2.0, 3.0, 4.0, 5.0)


def _shares(df: pd.DataFrame, team_col: str, group: list[str], value: str) -> pd.DataFrame:
    df = df.copy()
    df["share"] = df[value] / df.groupby(group)[value].transform("sum")
    df["r"] = df.groupby(group + ["position"])[value].rank(ascending=False, method="first")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--history", default="2019,2020,2021,2022,2023,2024")
    ap.add_argument("--stat", default="targets")
    ap.add_argument("--projection-path", default=None)
    args = ap.parse_args()

    projection_path = args.projection_path or os.path.join(
        "output", f"projections_{args.season}.csv"
    )
    proj = pd.read_csv(projection_path)
    proj = proj[(proj["stat"] == args.stat) & proj["position"].isin(RECV_POS)].copy()
    proj["v"] = proj["pred_season"].fillna(0.0)
    proj = _shares(proj, "team", ["team"], "v")

    hist = realized_weekly([int(s) for s in args.history.split(",")], [args.stat], RECV_POS)
    # realized_weekly has no team column; rank within season+position league-wide
    # is not equivalent, so pull team from the DB directly.
    from scripts._history import connect

    conn = connect()
    teams = pd.read_sql(
        "SELECT DISTINCT season, player_id, recent_team AS team FROM weekly "
        "WHERE recent_team IS NOT NULL",
        conn,
    )
    conn.close()
    hist = hist.merge(teams, on=["season", "player_id"], how="inner")
    hist = _shares(hist, "team", ["season", "team"], args.stat)

    print(f"Share of team {args.stat} by position and within-position rank")
    print(f"{'pos':<4}{'rank':>5}{'proj':>9}{'hist':>9}{'delta':>9}{'rel':>7}")
    worst = 0.0
    for pos in RECV_POS:
        for r in RANKS:
            a = proj[(proj.position == pos) & (proj.r == r)]["share"].mean()
            b = hist[(hist.position == pos) & (hist.r == r)]["share"].mean()
            rel = a / b if b else float("nan")
            worst = max(worst, abs(rel - 1.0))
            print(f"{pos:<4}{int(r):>5}{a:>9.4f}{b:>9.4f}{a - b:>+9.4f}{rel:>7.3f}")
        print()
    for pos in RECV_POS:
        a = proj[(proj.position == pos) & (proj.r == 1)]["share"]
        b = hist[(hist.position == pos) & (hist.r == 1)]["share"]
        print(
            f"{pos}1 dispersion: projected sd={a.std():.4f} p90={a.quantile(.9):.4f} "
            f"max={a.max():.4f} | realized sd={b.std():.4f} p90={b.quantile(.9):.4f} "
            f"max={b.max():.4f}"
        )
    print(f"\nworst |rel-1| over ranks 1-5 = {worst:.3f}")


if __name__ == "__main__":
    main()
