"""Export projections into team-stats JSON for the ESPN-style dashboard."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone

import pandas as pd

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
OUTPUT_DIR = os.path.join(REPO_ROOT, "output")
TEAM_STATS_DATA_DIR = os.path.join(REPO_ROOT, "draft_assistant", "data")

TEAM_META = [
    {"abbr": "ARI", "name": "Arizona Cardinals", "conference": "NFC", "division": "West"},
    {"abbr": "ATL", "name": "Atlanta Falcons", "conference": "NFC", "division": "South"},
    {"abbr": "BAL", "name": "Baltimore Ravens", "conference": "AFC", "division": "North"},
    {"abbr": "BUF", "name": "Buffalo Bills", "conference": "AFC", "division": "East"},
    {"abbr": "CAR", "name": "Carolina Panthers", "conference": "NFC", "division": "South"},
    {"abbr": "CHI", "name": "Chicago Bears", "conference": "NFC", "division": "North"},
    {"abbr": "CIN", "name": "Cincinnati Bengals", "conference": "AFC", "division": "North"},
    {"abbr": "CLE", "name": "Cleveland Browns", "conference": "AFC", "division": "North"},
    {"abbr": "DAL", "name": "Dallas Cowboys", "conference": "NFC", "division": "East"},
    {"abbr": "DEN", "name": "Denver Broncos", "conference": "AFC", "division": "West"},
    {"abbr": "DET", "name": "Detroit Lions", "conference": "NFC", "division": "North"},
    {"abbr": "GB", "name": "Green Bay Packers", "conference": "NFC", "division": "North"},
    {"abbr": "HOU", "name": "Houston Texans", "conference": "AFC", "division": "South"},
    {"abbr": "IND", "name": "Indianapolis Colts", "conference": "AFC", "division": "South"},
    {"abbr": "JAX", "name": "Jacksonville Jaguars", "conference": "AFC", "division": "South"},
    {"abbr": "KC", "name": "Kansas City Chiefs", "conference": "AFC", "division": "West"},
    {"abbr": "LA", "name": "Los Angeles Rams", "conference": "NFC", "division": "West"},
    {"abbr": "LAC", "name": "Los Angeles Chargers", "conference": "AFC", "division": "West"},
    {"abbr": "LV", "name": "Las Vegas Raiders", "conference": "AFC", "division": "West"},
    {"abbr": "MIA", "name": "Miami Dolphins", "conference": "AFC", "division": "East"},
    {"abbr": "MIN", "name": "Minnesota Vikings", "conference": "NFC", "division": "North"},
    {"abbr": "NE", "name": "New England Patriots", "conference": "AFC", "division": "East"},
    {"abbr": "NO", "name": "New Orleans Saints", "conference": "NFC", "division": "South"},
    {"abbr": "NYG", "name": "New York Giants", "conference": "NFC", "division": "East"},
    {"abbr": "NYJ", "name": "New York Jets", "conference": "AFC", "division": "East"},
    {"abbr": "PHI", "name": "Philadelphia Eagles", "conference": "NFC", "division": "East"},
    {"abbr": "PIT", "name": "Pittsburgh Steelers", "conference": "AFC", "division": "North"},
    {"abbr": "SEA", "name": "Seattle Seahawks", "conference": "NFC", "division": "West"},
    {"abbr": "SF", "name": "San Francisco 49ers", "conference": "NFC", "division": "West"},
    {"abbr": "TB", "name": "Tampa Bay Buccaneers", "conference": "NFC", "division": "South"},
    {"abbr": "TEN", "name": "Tennessee Titans", "conference": "AFC", "division": "South"},
    {"abbr": "WAS", "name": "Washington Commanders", "conference": "NFC", "division": "East"},
]

STAT_COLS = [
    "attempts",
    "completions",
    "passing_yards",
    "passing_tds",
    "interceptions",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
]

DRIVER_COLS = [
    "fantasy_pts",
    "fantasy_pts_season",
    "fantasy_pts_low",
    "fantasy_pts_high",
    "fantasy_pts_season_low",
    "fantasy_pts_season_high",
    "role_discount_factor",
    "role_discount_applied",
    "nfl_depth_rank",
    "projected_volume_games",
    "projected_games_raw",
    "team_changed",
    "roster_status",
    "rookie_tier",
    "athletic_tier",
    "rookie_depth_band",
    "rookie_vacancy_scale",
    "target_depth_rank",
    "qb_volume_games_scale",
    "qb_volume_allocation_adjusted",
    "team_qb_volume_allocation_direction",
    "normalization_scale_attempts",
    "normalization_scale_passing_yards",
    "normalization_scale_receiving_yards",
    "normalization_scale_carries",
    "normalization_scale_rushing_yards",
    "normalization_scale_receptions",
    "normalization_scale_receiving_tds",
    "any_receiving_share_capped",
    "any_receiving_share_normalized",
    "any_stat_low_n_flag",
    "team_pass_attempts_pg_pred",
    "team_passing_yards_pg_pred",
    "team_carries_pg_pred",
    "team_rushing_yards_pg_pred",
    "team_anchor_source_season",
    "team_pass_catch_ratio_pre_normalization",
]


def to_json_value(val):
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except TypeError:
        pass
    if isinstance(val, bool):
        return val
    if isinstance(val, int) and not isinstance(val, bool):
        return int(val)
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return None
        return round(val, 2)
    return str(val)


def _pivot_stats(df: pd.DataFrame, value_col: str, prefix: str) -> pd.DataFrame:
    wide = (
        df.pivot_table(
            index="player_id",
            columns="stat",
            values=value_col,
            aggfunc="first",
        )
        .reindex(columns=STAT_COLS)
        .reset_index()
    )
    wide.columns = [
        f"{prefix}_{c}" if c != "player_id" else c for c in wide.columns
    ]
    return wide


def load_and_build(season: int) -> list[dict]:
    proj_path = os.path.join(OUTPUT_DIR, f"projections_{season}.csv")
    fantasy_path = os.path.join(OUTPUT_DIR, f"fantasy_points_{season}.csv")
    if not os.path.exists(proj_path):
        raise FileNotFoundError(f"Missing projection file: {proj_path}")

    long = pd.read_csv(proj_path)
    long = long[long["position"].isin(["QB", "RB", "WR", "TE"])].copy()

    meta_cols = [
        "player_id",
        "display_name",
        "team",
        "position",
        "projected_games",
        "role",
        "depth_rank",
        "depth_chart_status",
        "source",
        "low_confidence",
    ]
    player_meta = (
        long[meta_cols]
        .drop_duplicates(subset=["player_id"])
        .copy()
    )

    pg = _pivot_stats(long, "pred_pg", "pg")
    season_stats = _pivot_stats(long, "pred_season", "season")

    players = player_meta.merge(pg, on="player_id", how="left").merge(
        season_stats, on="player_id", how="left"
    )

    if os.path.exists(fantasy_path):
        fantasy = pd.read_csv(fantasy_path)
        keep = ["player_id"] + [c for c in DRIVER_COLS if c in fantasy.columns]
        players = players.merge(fantasy[keep], on="player_id", how="left")
    else:
        for col in DRIVER_COLS:
            if col not in players.columns:
                players[col] = None

    records: list[dict] = []
    for row in players.itertuples(index=False):
        pg_stats = {s: to_json_value(getattr(row, f"pg_{s}", None)) for s in STAT_COLS}
        season_vals = {
            s: to_json_value(getattr(row, f"season_{s}", None)) for s in STAT_COLS
        }
        drivers = {}
        for col in DRIVER_COLS:
            if col in ("fantasy_pts", "fantasy_pts_season"):
                continue
            raw = getattr(row, col, None)
            if col in (
                "role_discount_applied",
                "team_changed",
                "qb_volume_allocation_adjusted",
                "any_receiving_share_capped",
                "any_receiving_share_normalized",
                "any_stat_low_n_flag",
            ):
                drivers[col] = bool(raw) if pd.notna(raw) else False
            else:
                drivers[col] = to_json_value(raw)

        records.append(
            {
                "player_id": str(row.player_id),
                "display_name": str(row.display_name),
                "team": str(row.team) if pd.notna(row.team) else None,
                "position": str(row.position),
                "projected_games": to_json_value(row.projected_games),
                "role": to_json_value(row.role),
                "depth_rank": to_json_value(row.depth_rank),
                "depth_chart_status": to_json_value(row.depth_chart_status),
                "source": to_json_value(row.source),
                "low_confidence": bool(row.low_confidence)
                if pd.notna(row.low_confidence)
                else False,
                "fantasy_pts": to_json_value(getattr(row, "fantasy_pts", None)),
                "fantasy_pts_season": to_json_value(
                    getattr(row, "fantasy_pts_season", None)
                ),
                "drivers": drivers,
                "pg": pg_stats,
                "season": season_vals,
            }
        )

    records.sort(
        key=lambda r: (
            r["team"] or "",
            {"QB": 0, "RB": 1, "WR": 2, "TE": 3}.get(r["position"], 9),
            r["depth_rank"] if r["depth_rank"] is not None else 99,
            -(r["fantasy_pts"] or 0),
        )
    )
    return records


def export_team_stats(season: int) -> str:
    players = load_and_build(season)
    teams_present = {p["team"] for p in players if p["team"]}
    teams = [t for t in TEAM_META if t["abbr"] in teams_present]

    payload = {
        "meta": {
            "season": season,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "player_count": len(players),
            "team_count": len(teams),
            "source_file": f"output/projections_{season}.csv",
        },
        "teams": teams,
        "players": players,
    }

    os.makedirs(TEAM_STATS_DATA_DIR, exist_ok=True)
    out_path = os.path.join(TEAM_STATS_DATA_DIR, f"team_stats_{season}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, allow_nan=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export team stats dashboard data")
    parser.add_argument("--season", type=int, default=2026)
    args = parser.parse_args()
    path = export_team_stats(args.season)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
