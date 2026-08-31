"""League-wide conservation and volume check — survivorship-free.

Order-statistic curves (the RB12 of a season) are inflated by survivorship, so
comparing a projected curve to a realized one overstates any gap. League-wide
TOTALS have no such bias and obey hard identities: total receiving yards must
equal total passing yards, receptions must equal completions, and league volume
per team is stable season to season. Any position that fails these is
miscalibrated in a way survivorship cannot explain.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts._history import realized_weekly  # noqa: E402

STAT_COLS = [
    "attempts", "completions", "passing_yards", "passing_tds", "interceptions",
    "carries", "rushing_yards", "rushing_tds",
    "targets", "receptions", "receiving_yards", "receiving_tds",
]
POSITIONS = ("QB", "RB", "WR", "TE")

def realized_totals(seasons: list[int]) -> pd.DataFrame:
    df = realized_weekly(seasons, STAT_COLS, positions=POSITIONS)
    return df.groupby(["season", "position"], as_index=False)[STAT_COLS].sum()


def projected_totals(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["position"].isin(POSITIONS)]
    wide = df.pivot_table(
        index=["player_id", "position"], columns="stat", values="pred_season", aggfunc="first"
    ).reset_index()
    return wide.groupby("position")[[c for c in STAT_COLS if c in wide.columns]].sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--history", default="2023,2024,2025")
    ap.add_argument("--projections", default=None)
    args = ap.parse_args()

    seasons = [int(s) for s in args.history.split(",")]
    hist = realized_totals(seasons)
    hist_mean = hist.groupby("position")[STAT_COLS].mean()

    path = args.projections or os.path.join("output", f"projections_{args.season}.csv")
    proj = projected_totals(path)

    print(f"League-wide season totals: projected {args.season} vs mean realized {seasons}\n")
    print(f"{'stat':<16} {'pos':<4} {'projected':>12} {'realized':>12} {'ratio':>7}")
    for stat in STAT_COLS:
        if stat not in proj.columns:
            continue
        for pos in POSITIONS:
            p = float(proj.loc[pos, stat]) if pos in proj.index else 0.0
            h = float(hist_mean.loc[pos, stat]) if pos in hist_mean.index else 0.0
            ratio = p / h if h else float("nan")
            print(f"{stat:<16} {pos:<4} {p:>12,.0f} {h:>12,.0f} {ratio:>7.3f}")
        pt, ht = float(proj[stat].sum()), float(hist_mean[stat].sum())
        print(f"{stat:<16} {'ALL':<4} {pt:>12,.0f} {ht:>12,.0f} {pt/ht if ht else float('nan'):>7.3f}")
        print()

    print("=== Hard identities (projected) ===")
    def tot(stat):
        return float(proj[stat].sum()) if stat in proj.columns else float("nan")
    checks = [
        ("receiving_yards vs passing_yards", tot("receiving_yards"), tot("passing_yards")),
        ("receptions vs completions", tot("receptions"), tot("completions")),
        ("targets vs attempts", tot("targets"), tot("attempts")),
        ("receiving_tds vs passing_tds", tot("receiving_tds"), tot("passing_tds")),
    ]
    for label, a, b in checks:
        print(f"  {label:<36} {a:>12,.0f} vs {b:>12,.0f}   ratio {a/b if b else float('nan'):.3f}")

    print("\n=== Same identities, realized (sanity: should be ~1.000) ===")
    for label, a, b in [
        ("receiving_yards vs passing_yards", float(hist_mean["receiving_yards"].sum()), float(hist_mean["passing_yards"].sum())),
        ("receptions vs completions", float(hist_mean["receptions"].sum()), float(hist_mean["completions"].sum())),
        ("targets vs attempts", float(hist_mean["targets"].sum()), float(hist_mean["attempts"].sum())),
    ]:
        print(f"  {label:<36} {a:>12,.0f} vs {b:>12,.0f}   ratio {a/b:.3f}")


if __name__ == "__main__":
    main()
