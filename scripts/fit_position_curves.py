"""Fit per-position season-point curves from realized history for board ranking.

Writes src/draft_assistant/position_curves.json: the median realized season
fantasy points at each positional rank. `src/draft_assistant/vorp.py` uses these
to correct a position whose projected curve has the wrong *shape* relative to
the others, which lands it in the wrong place on a cross-position board even
when its own internal ordering is right.

Regenerate whenever the history window should move:
    python scripts/fit_position_curves.py --history 2022,2023,2024,2025
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts._history import realized_weekly  # noqa: E402

SCORING = {
    "passing_yards": 1 / 25, "passing_tds": 4, "interceptions": -2,
    "rushing_yards": 1 / 10, "rushing_tds": 6,
    "receiving_yards": 1 / 10, "receiving_tds": 6, "receptions": 0.5,
}
POSITIONS = ("QB", "RB", "WR", "TE")
OUT_PATH = os.path.join("src", "draft_assistant", "position_curves.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default="2022,2023,2024,2025")
    ap.add_argument("--depth", type=int, default=60)
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()
    seasons = [int(s) for s in args.history.split(",")]

    h = realized_weekly(seasons, list(SCORING))
    h["pts"] = sum(h[c] * w for c, w in SCORING.items())
    h["r"] = h.groupby(["season", "position"])["pts"].rank(ascending=False, method="first")
    h = h[h.r <= args.depth]

    curves: dict[str, list[float]] = {}
    for pos in POSITIONS:
        med = (
            h[h.position == pos]
            .groupby("r")["pts"]
            .median()
            .sort_index()
        )
        # Enforce monotone decreasing so a blend can never invert an ordering.
        vals: list[float] = []
        for v in med.tolist():
            vals.append(float(v) if not vals else float(min(v, vals[-1])))
        curves[pos] = [round(v, 3) for v in vals]

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "history_seasons": seasons,
        "scoring": "half-PPR, 4pt passing TD",
        "note": (
            "Median realized season fantasy points by positional rank. These are "
            "order statistics and therefore survivorship-inflated relative to any "
            "honest expected-value projection; they are used for SHAPE within a "
            "position, anchored at that position's own replacement level, never as "
            "an absolute level target."
        ),
        "curves": curves,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    print(f"Wrote {args.out}")
    for pos in POSITIONS:
        print(f"  {pos}: {len(curves[pos])} ranks, top {curves[pos][:5]}")


if __name__ == "__main__":
    main()
