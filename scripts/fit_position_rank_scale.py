"""Fit a per-position ranking scale for the draft board, consensus-free.

The board ranks positions against each other through VORP, so a position whose
projected point curve sits on a different scale than the others lands in the
wrong place overall even when its own internal order is right.

The reference here is a board built from *realized* historical curves under the
identical VORP rules. That controls for the thing a raw projected-vs-realized
ratio does not: survivorship. A realized "TE5" is the ex-post fifth-best tight
end, so realized curves are steeper than any honest expected-value projection --
but running both through the same replacement-level arithmetic compares boards
to boards rather than a projection to an order statistic.

Reports, per position, the multiplier on ranking points that best aligns our
overall board positions with the historical-implied ones.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts._history import realized_weekly  # noqa: E402
from src.draft_assistant.vorp import replacement_ranks  # noqa: E402

SCORING = {
    "passing_yards": 1 / 25, "passing_tds": 4, "interceptions": -2,
    "rushing_yards": 1 / 10, "rushing_tds": 6,
    "receiving_yards": 1 / 10, "receiving_tds": 6, "receptions": 0.5,
}
POSITIONS = ("QB", "RB", "WR", "TE")
PROBES = {"QB": [1, 2, 3, 5, 8, 12], "RB": [1, 2, 3, 5, 8, 12, 18, 24],
          "WR": [1, 2, 3, 5, 8, 12, 18, 24], "TE": [1, 2, 3, 5, 8, 12]}


def historical_curves(seasons: list[int], depth: int = 80) -> pd.DataFrame:
    h = realized_weekly(seasons, list(SCORING))
    h["pts"] = sum(h[c] * w for c, w in SCORING.items())
    h["r"] = h.groupby(["season", "position"])["pts"].rank(ascending=False, method="first")
    med = h[h.r <= depth].groupby(["position", "r"])["pts"].median().reset_index()
    return med


def board_from_curves(curves: pd.DataFrame, ranks: dict[str, int],
                      scale: dict[str, float] | None = None) -> pd.DataFrame:
    scale = scale or {}
    rows = []
    for pos in POSITIONS:
        sub = curves[curves.position == pos].sort_values("r").copy()
        sub["pts"] = sub["pts"] * scale.get(pos, 1.0)
        rk = ranks[pos]
        match = sub[sub.r == rk]
        base = float(match["pts"].iloc[0]) if len(match) else float(sub["pts"].iloc[-1])
        sub["vorp"] = sub["pts"] - base
        rows.append(sub)
    d = pd.concat(rows).sort_values("vorp", ascending=False).reset_index(drop=True)
    d["overall"] = d.index + 1
    return d


def our_board(path: str) -> pd.DataFrame:
    b = json.load(open(path, encoding="utf-8"))["players"]
    d = pd.DataFrame(b)
    d["r"] = d.groupby("position")["vorp"].rank(ascending=False, method="first")
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--history", default="2022,2023,2024,2025")
    ap.add_argument("--board", default=None)
    args = ap.parse_args()

    curves = historical_curves([int(s) for s in args.history.split(",")])
    ranks = replacement_ranks()
    ref = board_from_curves(curves, ranks)
    ours = our_board(
        args.board or os.path.join("draft_assistant", "data", f"players_{args.season}.json")
    )

    print(f"Reference replacement ranks: {ranks}")
    print("\nOverall board position, ours vs the historical-implied reference:")
    print(f"{'pos':<4}{'posrank':>8}{'ref_ovr':>9}{'our_ovr':>9}{'delta':>7}")
    gaps: dict[str, list[float]] = {p: [] for p in POSITIONS}
    for pos in POSITIONS:
        for rk in PROBES[pos]:
            r = ref[(ref.position == pos) & (ref.r == rk)]
            o = ours[(ours.position == pos) & (ours.r == rk)]
            if r.empty or o.empty:
                continue
            ro, oo = int(r["overall"].iloc[0]), int(o["overall_rank"].iloc[0])
            gaps[pos].append(oo - ro)
            print(f"{pos:<4}{rk:>8}{ro:>9}{oo:>9}{oo - ro:>+7}")
        print()

    print("Mean overall-rank gap (negative = we rank the position too high):")
    for pos in POSITIONS:
        if gaps[pos]:
            print(f"  {pos}: {sum(gaps[pos]) / len(gaps[pos]):+.1f}  (n={len(gaps[pos])} probes)")

    print("\nScale search — multiplier on ranking points, scored by mean |rank gap|:")
    for pos in POSITIONS:
        best = None
        for i in range(50, 131):
            sc = i / 100.0
            b = board_from_curves(curves, ranks)  # reference unchanged
            # rescale OUR points for this position and re-rank our board
            o = ours.copy()
            o["adj"] = o["vorp_input_pts"] * o["position"].map(
                lambda p: sc if p == pos else 1.0
            )
            adj_rows = []
            for p in POSITIONS:
                sub = o[o.position == p].sort_values("adj", ascending=False).copy()
                rk = ranks[p]
                base = float(sub["adj"].iloc[min(rk, len(sub)) - 1])
                sub["v2"] = sub["adj"] - base
                sub["r2"] = range(1, len(sub) + 1)
                adj_rows.append(sub)
            oo = pd.concat(adj_rows).sort_values("v2", ascending=False).reset_index(drop=True)
            oo["ovr2"] = oo.index + 1
            err = []
            for rk in PROBES[pos]:
                rr = b[(b.position == pos) & (b.r == rk)]
                mm = oo[(oo.position == pos) & (oo.r2 == rk)]
                if not rr.empty and not mm.empty:
                    err.append(abs(int(mm["ovr2"].iloc[0]) - int(rr["overall"].iloc[0])))
            if err:
                score = sum(err) / len(err)
                if best is None or score < best[1]:
                    best = (sc, score)
        if best:
            print(f"  {pos}: best scale {best[0]:.2f}  (mean |rank gap| {best[1]:.1f})")


if __name__ == "__main__":
    main()
