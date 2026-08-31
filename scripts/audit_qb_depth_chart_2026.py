"""Audit 2026 curated QB depth chart vs 2025 usage.

2025 attempt-share is a review flag only. It does not imply the 2026 starter
or team: Kyler Murray led ARI in 2025 and is MIN QB1 in 2026.

Usage:
    python scripts/audit_qb_depth_chart_2026.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.qb_tracks_util import board_qb_snapshot, write_json
from src.projection.data_prep import get_conn

STARTERS_PATH = ROOT / "src" / "depth_chart" / "starters_2026.csv"


def _qb_attempt_leaders_2025(conn) -> pd.DataFrame:
    from src.projection.features import build_player_season_features

    feat = build_player_season_features(conn)
    df = feat[(feat["season"].eq(2025)) & (feat["position"].eq("QB"))][
        ["player_id", "team", "attempts"]
    ].copy()
    df["display_name"] = df["player_id"]
    df["team_rank"] = df.groupby("team")["attempts"].rank(ascending=False, method="first")
    return df


def audit_chart(chart: pd.DataFrame, usage: pd.DataFrame) -> list[dict]:
    flags = []
    qb_chart = chart[chart["position"].eq("QB")].copy()
    for team, group in qb_chart.groupby("team"):
        qb1 = group[group["depth_rank"].eq(1)]
        if qb1.empty:
            continue
        qb1_row = qb1.iloc[0]
        team_usage = usage[usage["team"].eq(team)].sort_values("attempts", ascending=False)
        if team_usage.empty:
            continue
        leader = team_usage.iloc[0]
        if qb1_row["gsis_id"] != leader["player_id"]:
            flags.append({
                "team": team,
                "curated_qb1": qb1_row["player_name"],
                "curated_qb1_gsis": qb1_row["gsis_id"],
                "usage_leader_2025": leader.get("display_name", leader["player_id"]),
                "usage_leader_gsis": leader["player_id"],
                "leader_attempts": int(leader["attempts"]),
                "severity": "review",
            })
    return flags


def main() -> None:
    chart = pd.read_csv(STARTERS_PATH)
    conn = get_conn()
    try:
        usage = _qb_attempt_leaders_2025(conn)
    finally:
        conn.close()

    flags = audit_chart(chart, usage)
    board = board_qb_snapshot(ROOT / "output" / "fantasy_points_2026.csv")

    payload = {
        "season": 2026,
        "flags": flags,
        "board": board,
        "notes": (
            "Flagged teams have a curated 2026 QB1 who is not the 2025 attempt "
            "leader on that team. Do not auto-promote 2025 usage leaders: "
            "Kyler Murray is MIN QB1 in 2026 after leaving ARI."
        ),
    }
    out_path = ROOT / "output" / "qb_depth_chart_audit_2026.json"
    write_json(out_path, payload)
    print(f"Wrote {out_path}")
    print(f"Flagged {len(flags)} team(s) with curated QB1 != 2025 attempt leader")
    for f in flags:
        print(
            f"  {f['team']}: chart={f['curated_qb1']} vs leader={f['usage_leader_2025']} "
            f"({f['leader_attempts']} att)"
        )


if __name__ == "__main__":
    main()
