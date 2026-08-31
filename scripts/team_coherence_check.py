"""Per-team internal consistency: do the receivers add up to the quarterbacks?

League-wide totals can balance almost exactly while individual teams are badly
off in both directions, because the errors cancel in aggregate. Football has a
hard identity here -- a team's receiving yards ARE its passing yards -- so any
per-team gap is a defect, not a modelling choice.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts._history import realized_weekly  # noqa: E402

PAIRS = [
    ("receiving_yards", "passing_yards"),
    ("receptions", "completions"),
    ("targets", "attempts"),
    ("receiving_tds", "passing_tds"),
]
RECV_POS = ("RB", "WR", "TE")


def projected(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    stats = sorted({s for pair in PAIRS for s in pair})
    wide = df.pivot_table(
        index=["player_id", "team", "position"],
        columns="stat",
        values="pred_season",
        aggfunc="first",
    ).reset_index()
    for s in stats:
        if s not in wide.columns:
            wide[s] = 0.0
    return wide


def team_pairs(wide: pd.DataFrame) -> pd.DataFrame:
    recv = wide[wide.position.isin(RECV_POS)].groupby("team").sum(numeric_only=True)
    pas = wide[wide.position == "QB"].groupby("team").sum(numeric_only=True)
    out = pd.DataFrame(index=sorted(set(recv.index) | set(pas.index)))
    for r_stat, p_stat in PAIRS:
        out[r_stat] = recv[r_stat].reindex(out.index).fillna(0.0)
        out[p_stat] = pas[p_stat].reindex(out.index).fillna(0.0)
        out[f"ratio_{r_stat}"] = out[r_stat] / out[p_stat].replace(0, float("nan"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--history", default="2023,2024,2025")
    ap.add_argument("--projections", default=None)
    ap.add_argument("--tol", type=float, default=0.05)
    args = ap.parse_args()

    path = args.projections or os.path.join("output", f"projections_{args.season}.csv")
    proj = team_pairs(projected(path))

    stats = sorted({s for pair in PAIRS for s in pair})
    hist = realized_weekly([int(s) for s in args.history.split(",")], stats)
    conn_needed = True
    if conn_needed:
        from scripts._history import connect

        conn = connect()
        teams = pd.read_sql(
            "SELECT DISTINCT season, player_id, recent_team AS team FROM weekly "
            "WHERE recent_team IS NOT NULL",
            conn,
        )
        conn.close()
    hist = hist.merge(teams, on=["season", "player_id"], how="inner")

    print(f"Per-team identity checks, projected {args.season}")
    print(f"{'pair':<34}{'mean':>8}{'sd':>8}{'min':>8}{'max':>8}{'>tol':>7}")
    for r_stat, p_stat in PAIRS:
        col = proj[f"ratio_{r_stat}"].dropna()
        off = ((col - 1.0).abs() > args.tol).sum()
        print(
            f"{r_stat+' / '+p_stat:<34}{col.mean():>8.3f}{col.std():>8.3f}"
            f"{col.min():>8.3f}{col.max():>8.3f}{off:>7}"
        )

    # Realized reference, computed per team-season the same way.
    print(f"\nRealized {args.history} reference for the same ratios:")
    for r_stat, p_stat in PAIRS:
        recv = hist[hist.position.isin(RECV_POS)].groupby(["season", "team"])[r_stat].sum()
        pas = hist[hist.position == "QB"].groupby(["season", "team"])[p_stat].sum()
        ratio = (recv / pas.replace(0, float("nan"))).dropna()
        off = ((ratio - 1.0).abs() > args.tol).sum()
        print(
            f"{r_stat+' / '+p_stat:<34}{ratio.mean():>8.3f}{ratio.std():>8.3f}"
            f"{ratio.min():>8.3f}{ratio.max():>8.3f}{off:>7} of {len(ratio)}"
        )

    worst = proj.reindex(
        (proj["ratio_receiving_yards"] - 1.0).abs().sort_values(ascending=False).index
    )
    print("\nWorst teams by receiving/passing yards:")
    print(f"{'team':<6}{'recv_yds':>10}{'pass_yds':>10}{'ratio':>8}{'gap_yds':>9}")
    for team, row in worst.head(12).iterrows():
        print(
            f"{team:<6}{row['receiving_yards']:>10.0f}{row['passing_yards']:>10.0f}"
            f"{row['ratio_receiving_yards']:>8.3f}{row['receiving_yards']-row['passing_yards']:>9.0f}"
        )


if __name__ == "__main__":
    main()
