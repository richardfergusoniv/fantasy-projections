"""Measure the spread between our draft board and market consensus (ECR/ADP).

Reports rank agreement inside the matched set only. Both sides are re-ranked
within the intersection first, because our board and the market rank different
size universes -- comparing raw ranks across universes manufactures a constant
positional bias that is an artifact of universe size, not of the model.

Usage:
    python scripts/consensus_spread.py [--season 2026] [--board PATH] [--label NAME]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONSENSUS_DIR = os.path.join(REPO_ROOT, "data", "consensus")
DRAFT_DATA_DIR = os.path.join(REPO_ROOT, "draft_assistant", "data")

POSITIONS = ("QB", "RB", "WR", "TE")


def _norm_name(name: str | None) -> str:
    if not name:
        return ""
    s = str(name).lower().strip()
    s = s.replace(".", "").replace("'", "").replace("’", "")
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_consensus(season: int) -> list[dict]:
    path = os.path.join(CONSENSUS_DIR, f"consensus_{season}.json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["rows"]


def load_board(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["players"]


def _pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 2:
        return float("nan")
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((x - mb) ** 2 for x in b))
    return num / den if den else float("nan")


def join(
    board: list[dict], consensus: list[dict], key: str, max_market_rank: int | None = None
) -> list[dict]:
    """Join board to consensus on player_id, falling back to normalized name+pos."""
    by_id = {str(p.get("player_id")): p for p in board}
    by_name = {(_norm_name(p.get("display_name")), p.get("position")): p for p in board}
    matched: list[dict] = []
    for c in consensus:
        if c.get(key) is None:
            continue
        p = by_id.get(str(c["player_id"])) or by_name.get(
            (_norm_name(c["display_name"]), c["position"])
        )
        if p is None or p.get("overall_rank") is None:
            continue
        matched.append(
            {
                "name": c["display_name"],
                "position": c["position"],
                "team": c.get("team"),
                "our_raw": float(p["overall_rank"]),
                "mkt_raw": float(c[key]),
                "pts": p.get("fantasy_pts_season"),
                "vorp": p.get("vorp"),
            }
        )
    # Optionally restrict to the draftable range before re-ranking, so the
    # metric is not dominated by players nobody drafts.
    if max_market_rank is not None:
        matched.sort(key=lambda r: r["mkt_raw"])
        matched = matched[:max_market_rank]
    # Re-rank both sides inside the matched set.
    matched.sort(key=lambda r: r["our_raw"])
    for i, r in enumerate(matched):
        r["our"] = i + 1
    matched.sort(key=lambda r: r["mkt_raw"])
    for i, r in enumerate(matched):
        r["mkt"] = i + 1
        r["d"] = r["our"] - r["mkt"]
    return matched


def summarize(matched: list[dict], key: str) -> dict[str, Any]:
    if not matched:
        return {}
    n = len(matched)
    deltas = [r["d"] for r in matched]
    absd = sorted(abs(x) for x in deltas)
    out: dict[str, Any] = {
        "n": n,
        "spearman": _pearson([r["our"] for r in matched], [r["mkt"] for r in matched]),
        "mean_abs": statistics.mean(absd),
        "median_abs": statistics.median(absd),
        "p90_abs": absd[min(int(0.9 * n), n - 1)],
        "max_abs": absd[-1],
        "bands": {},
        "positions": {},
    }
    for lo, hi in ((1, 24), (25, 60), (61, 120), (121, 10_000)):
        band = [r for r in matched if lo <= r["mkt"] <= hi]
        if band:
            label = f"{lo}-{min(hi, n)}"
            out["bands"][label] = {
                "n": len(band),
                "mean_abs": statistics.mean(abs(r["d"]) for r in band),
                "median_abs": statistics.median(abs(r["d"]) for r in band),
            }
    for pos in POSITIONS:
        grp = [r for r in matched if r["position"] == pos]
        if grp:
            out["positions"][pos] = {
                "n": len(grp),
                "mean_abs": statistics.mean(abs(r["d"]) for r in grp),
                "bias": statistics.mean(r["d"] for r in grp),
            }
    out["worst"] = sorted(matched, key=lambda r: -abs(r["d"]))[:15]
    return out


def report(
    board_path: str,
    season: int,
    label: str,
    show_worst: bool = True,
    max_market_rank: int | None = None,
) -> dict:
    board = load_board(board_path)
    consensus = load_consensus(season)
    result = {"label": label, "board": board_path, "board_size": len(board)}
    for key in ("ecr", "adp"):
        matched = join(board, consensus, key, max_market_rank)
        summary = summarize(matched, key)
        result[key] = summary
        if not summary:
            continue
        print(f"\n=== {label} vs {key.upper()}  (n={summary['n']}) ===")
        print(
            f"  spearman {summary['spearman']:.4f} | "
            f"mean|d| {summary['mean_abs']:.1f} | median|d| {summary['median_abs']:.1f} | "
            f"p90 {summary['p90_abs']:.0f} | max {summary['max_abs']:.0f}"
        )
        for band, stats in summary["bands"].items():
            print(
                f"    market {band:<10} n={stats['n']:<4} mean|d|={stats['mean_abs']:6.1f} "
                f"median={stats['median_abs']:6.1f}"
            )
        for pos, stats in summary["positions"].items():
            print(
                f"    {pos} n={stats['n']:<4} mean|d|={stats['mean_abs']:6.1f} "
                f"bias={stats['bias']:+7.1f}"
            )
        if show_worst:
            print("    worst:")
            for r in summary["worst"][:10]:
                print(
                    f"      {r['name']:<24} {r['position']} {str(r['team']):<4} "
                    f"ours={r['our']:<4} mkt={r['mkt']:<4} d={r['d']:+5}  pts={r['pts']}"
                )
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--board", default=None)
    ap.add_argument("--label", default="board")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--no-worst", action="store_true")
    ap.add_argument(
        "--restrict-to-board",
        default=None,
        help="score only players that also appear on this other board JSON, so "
             "two boards of different size can be compared on a common set",
    )
    ap.add_argument(
        "--max-market-rank",
        type=int,
        default=None,
        help="restrict to the top N by market rank (the draftable range)",
    )
    args = ap.parse_args()
    board = args.board or os.path.join(DRAFT_DATA_DIR, f"players_{args.season}.json")
    if args.restrict_to_board:
        keep = {str(p.get("player_id")) for p in load_board(args.restrict_to_board)}
        _orig = load_board

        def load_board_filtered(path: str) -> list[dict]:  # noqa: D401
            return [p for p in _orig(path) if str(p.get("player_id")) in keep]

        globals()["load_board"] = load_board_filtered
    result = report(
        board,
        args.season,
        args.label,
        show_worst=not args.no_worst,
        max_market_rank=args.max_market_rank,
    )
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
