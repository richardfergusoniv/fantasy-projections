"""Reorder the draft checklist to match a league-scoring VORP board.

The Fantasy Decisions checklist keeps market ADP/ECR + context checks, but the
All-tab queue should read like the Sep 2026 Fantasy Projections draft rankings
(League scoring, ordered by VORP) — not raw ADP (where Chase sits too high).

O-line team ranks in the checklist artifact are left untouched; the O-line pane
is separate and is not rewritten here.

Usage:
  python -m src.draft_assistant.checklist_apply_board_order \\
    --projections /tmp/legacy_projections.json --season 2026
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.paths import REPO_ROOT

DRAFT_DATA_DIR = Path(REPO_ROOT) / "draft_assistant" / "data"

# Matches the legacy /draft "League" tab copy: 12-team starters + flex baselines.
DEFAULT_REPLACEMENT = {"QB": 12, "RB": 30, "WR": 42, "TE": 12}

POS_ORDER = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}


def league_points(row: dict[str, Any]) -> float:
    """6-pt pass TD, -3 INT, no PPR — same as the legacy League tab."""
    return (
        float(row.get("proj_passing_yards") or 0) / 25.0
        + float(row.get("proj_passing_tds") or 0) * 6.0
        + float(row.get("proj_interceptions") or 0) * (-3.0)
        + float(row.get("proj_rushing_yards") or 0) / 10.0
        + float(row.get("proj_rushing_tds") or 0) * 6.0
        + float(row.get("proj_receiving_yards") or 0) / 10.0
        + float(row.get("proj_receiving_tds") or 0) * 6.0
    )


def vorp_ranks(
    projections: list[dict[str, Any]],
    *,
    replacement: dict[str, int] = DEFAULT_REPLACEMENT,
) -> tuple[dict[str, int], dict[str, int], dict[str, float], dict[str, float]]:
    """Return overall_rank, pos_rank, league_pts, vorp by player_gsis_id."""
    by_pos: dict[str, list[tuple[float, dict[str, Any]]]] = {
        "QB": [],
        "RB": [],
        "WR": [],
        "TE": [],
    }
    for row in projections:
        pos = str(row.get("position") or "")
        if pos not in by_pos:
            continue
        pid = str(row.get("player_gsis_id") or "")
        if not pid:
            continue
        by_pos[pos].append((league_points(row), row))

    for pos in by_pos:
        by_pos[pos].sort(
            key=lambda item: (-item[0], str(item[1].get("full_name") or ""))
        )

    repl_pts: dict[str, float] = {}
    for pos, rank in replacement.items():
        cohort = by_pos[pos]
        repl_pts[pos] = cohort[rank - 1][0] if 0 <= rank - 1 < len(cohort) else 0.0

    scored: list[tuple[float, float, str, str, dict[str, Any]]] = []
    pos_rank: dict[str, int] = {}
    league_pts: dict[str, float] = {}
    for pos, cohort in by_pos.items():
        for index, (pts, row) in enumerate(cohort, start=1):
            pid = str(row["player_gsis_id"])
            pos_rank[pid] = index
            league_pts[pid] = pts
            scored.append((pts - repl_pts[pos], pts, str(row.get("full_name") or ""), pos, row))

    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    overall: dict[str, int] = {}
    vorp: dict[str, float] = {}
    for index, (value, _pts, _name, _pos, row) in enumerate(scored, start=1):
        pid = str(row["player_gsis_id"])
        overall[pid] = index
        vorp[pid] = value
    return overall, pos_rank, league_pts, vorp


def apply_board_order(
    checklist: dict[str, Any],
    projections: list[dict[str, Any]],
    *,
    replacement: dict[str, int] = DEFAULT_REPLACEMENT,
    board_as_of: str | None = None,
    board_source: str = "fantasy-projections.vercel.app/draft",
) -> dict[str, Any]:
    """Mutate a checklist payload so All/pos ranks follow league VORP order."""
    overall, pos_rank, league_pts, vorp = vorp_ranks(projections, replacement=replacement)
    players = list(checklist.get("players") or [])
    matched = 0
    for row in players:
        pid = str(row.get("player_id") or "")
        if pid in overall:
            matched += 1
            row["overall_rank"] = overall[pid]
            row["pos_market_rank"] = pos_rank[pid]
            row["league_pts"] = round(league_pts[pid], 1)
            # Keep sealed-board ``vorp`` off checklist rows (API contract).
            row.pop("vorp", None)
        else:
            # Keep checks/market metadata; push unmatched behind the board.
            row["overall_rank"] = 10_000 + int(row.get("pos_market_rank") or 9999)
            row.pop("league_pts", None)
            row.pop("vorp", None)

    players.sort(
        key=lambda row: (
            POS_ORDER.get(str(row.get("position") or ""), 9),
            int(row.get("pos_market_rank") or 9999),
            str(row.get("name") or ""),
        )
    )
    checklist["players"] = players

    meta = dict(checklist.get("meta") or {})
    meta["rank_source"] = "league_vorp"
    meta["generated_at"] = datetime.now(timezone.utc).isoformat()
    meta["board_order"] = {
        "scoring": "league",
        "pass_td": 6,
        "int": -3,
        "ppr": 0,
        "replacement": dict(replacement),
        "source": board_source,
        "as_of": board_as_of,
        "matched_players": matched,
        "projection_count": len(projections),
    }
    # Market ADP/ECR remain available on each row; keep market_as_of provenance.
    checklist["meta"] = meta
    return checklist


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument(
        "--projections",
        type=Path,
        required=True,
        help="JSON list of legacy /draft projection rows (player_gsis_id + proj_*)",
    )
    parser.add_argument(
        "--checklist",
        type=Path,
        default=None,
        help="Checklist JSON to rewrite (default: draft_assistant/data/draft_checklist_{season}.json)",
    )
    parser.add_argument("--board-as-of", default="2026-09-04")
    parser.add_argument(
        "--board-source",
        default="fantasy-projections.vercel.app/draft",
    )
    args = parser.parse_args(argv)

    checklist_path = args.checklist or (
        DRAFT_DATA_DIR / f"draft_checklist_{args.season}.json"
    )
    projections = json.loads(args.projections.read_text(encoding="utf-8"))
    if isinstance(projections, dict):
        projections = list(projections.get("projections") or projections.get("players") or [])
    checklist = json.loads(checklist_path.read_text(encoding="utf-8"))
    apply_board_order(
        checklist,
        projections,
        board_as_of=args.board_as_of,
        board_source=args.board_source,
    )
    checklist_path.write_text(
        json.dumps(checklist, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    top = sorted(
        (p for p in checklist["players"] if p.get("overall_rank", 10_000) < 10_000),
        key=lambda p: p["overall_rank"],
    )[:8]
    print(f"Wrote {checklist_path} ({len(checklist['players'])} players)")
    print("Top overall:", [(p["overall_rank"], p["name"], p["position"]) for p in top])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
