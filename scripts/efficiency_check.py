"""Per-position receiving/rushing efficiency: projected vs realized.

Volume share can be calibrated while points are still wrong, because points are
volume x efficiency. This isolates the second factor.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts._history import realized_weekly  # noqa: E402

POS = ("QB", "RB", "WR", "TE")
STATS = [
    "targets", "receptions", "receiving_yards", "receiving_tds",
    "carries", "rushing_yards", "rushing_tds",
    "attempts", "completions", "passing_yards", "passing_tds",
]

RATES = {
    "catch_rate": ("receptions", "targets"),
    "yards_per_rec": ("receiving_yards", "receptions"),
    "yards_per_target": ("receiving_yards", "targets"),
    "rec_td_per_target": ("receiving_tds", "targets"),
    "yards_per_carry": ("rushing_yards", "carries"),
    "rush_td_per_carry": ("rushing_tds", "carries"),
    "yards_per_attempt": ("passing_yards", "attempts"),
    "pass_td_per_attempt": ("passing_tds", "attempts"),
}


def _rates(tot: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for name, (num, den) in RATES.items():
        if num in tot.columns and den in tot.columns:
            out[name] = tot[num] / tot[den].replace(0, float("nan"))
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--history", default="2022,2023,2024,2025")
    ap.add_argument("--projection-path", default=None)
    args = ap.parse_args()
    seasons = [int(s) for s in args.history.split(",")]

    projection_path = args.projection_path or os.path.join(
        "output", f"projections_{args.season}.csv"
    )
    proj = pd.read_csv(projection_path)
    proj = proj[proj["position"].isin(POS)]
    wide = proj.pivot_table(
        index=["player_id", "position"], columns="stat", values="pred_season", aggfunc="first"
    ).reset_index()
    p_tot = wide.groupby("position")[[c for c in STATS if c in wide.columns]].sum()

    hist = realized_weekly(seasons, STATS, POS)
    h_tot = hist.groupby("position")[STATS].sum() / len(seasons)

    pr, hr = _rates(p_tot), _rates(h_tot)
    print(f"League-aggregate efficiency: projected {args.season} vs realized {seasons}\n")
    print(f"{'rate':<22}{'pos':<5}{'projected':>11}{'realized':>11}{'ratio':>8}")
    for name in RATES:
        if name not in pr.columns:
            continue
        for pos in POS:
            a = pr.loc[pos, name] if pos in pr.index else float("nan")
            b = hr.loc[pos, name] if pos in hr.index else float("nan")
            if pd.isna(a) or pd.isna(b):
                continue
            print(f"{name:<22}{pos:<5}{a:>11.4f}{b:>11.4f}{a / b:>8.3f}")
        print()


if __name__ == "__main__":
    main()
