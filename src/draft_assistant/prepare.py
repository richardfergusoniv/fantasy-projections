"""Export projection CSV to draft-assistant JSON with tiers and metadata."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone

import pandas as pd

from src.draft_assistant.tiers import (
    DEFAULT_TIER_GAPS,
    FLEX_TIER_GAP,
    OVERALL_TIER_GAP,
    TierConfig,
    add_tier_columns,
)

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
OUTPUT_DIR = os.path.join(REPO_ROOT, "output")
DRAFT_DATA_DIR = os.path.join(REPO_ROOT, "draft_assistant", "data")

EXPORT_COLS = [
    "player_id",
    "display_name",
    "position",
    "team",
    "fantasy_pts",
    "fantasy_pts_low",
    "fantasy_pts_high",
    "fantasy_pts_season",
    "projected_games",
    "source",
    "low_confidence",
    "role",
    "depth_chart_status",
    "overall_rank",
    "overall_tier",
    "pos_rank",
    "pos_tier",
    "flex_rank",
    "flex_tier",
]


def load_projections(season: int) -> pd.DataFrame:
    path = os.path.join(OUTPUT_DIR, f"fantasy_points_{season}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing projection file: {path}")
    df = pd.read_csv(path)
    df = df[df["position"].isin(["QB", "RB", "WR", "TE"])].copy()
    df = df.sort_values("fantasy_pts", ascending=False).reset_index(drop=True)
    return df


def to_json_value(val, *, as_bool: bool = False):
    """Convert a pandas/scalar value to strict JSON-compatible Python types."""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except TypeError:
        pass
    if as_bool:
        return bool(val)
    if isinstance(val, bool):
        return val
    if isinstance(val, int) and not isinstance(val, bool):
        return int(val)
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return None
        return round(val, 2)
    return str(val)


def build_player_records(df: pd.DataFrame) -> list[dict]:
    records: list[dict] = []
    for row in df.itertuples(index=False):
        rec = {}
        for col in EXPORT_COLS:
            raw = getattr(row, col, None)
            if col == "low_confidence":
                rec[col] = to_json_value(raw, as_bool=True) or False
            elif col in (
                "overall_rank",
                "overall_tier",
                "pos_rank",
                "pos_tier",
                "flex_rank",
                "flex_tier",
            ):
                rec[col] = int(raw) if pd.notna(raw) else None
            else:
                rec[col] = to_json_value(raw)
        records.append(rec)
    return records


def tier_summary(df: pd.DataFrame) -> dict:
    summary: dict = {"overall": {}, "by_position": {}}
    for tier, group in df.groupby("overall_tier"):
        summary["overall"][str(int(tier))] = {
            "count": int(len(group)),
            "top": group.iloc[0]["display_name"],
            "pts_range": [
                round(float(group["fantasy_pts"].max()), 2),
                round(float(group["fantasy_pts"].min()), 2),
            ],
        }
    for pos in ["QB", "RB", "WR", "TE"]:
        pos_df = df[df["position"] == pos]
        summary["by_position"][pos] = {}
        for tier, group in pos_df.groupby("pos_tier"):
            summary["by_position"][pos][str(int(tier))] = {
                "count": int(len(group)),
                "top": group.iloc[0]["display_name"],
                "pts_range": [
                    round(float(group["fantasy_pts"].max()), 2),
                    round(float(group["fantasy_pts"].min()), 2),
                ],
            }
    flex_df = df[df["position"].isin(["RB", "WR", "TE"])]
    summary["flex"] = {}
    for tier, group in flex_df.groupby("flex_tier"):
        summary["flex"][str(int(tier))] = {
            "count": int(len(group)),
            "top": group.sort_values("fantasy_pts", ascending=False).iloc[0][
                "display_name"
            ],
            "pts_range": [
                round(float(group["fantasy_pts"].max()), 2),
                round(float(group["fantasy_pts"].min()), 2),
            ],
        }
    return summary


def export_draft_data(
    season: int,
    *,
    tier_config: TierConfig | None = None,
) -> str:
    df = load_projections(season)
    df = add_tier_columns(df, config=tier_config)
    players = build_player_records(df)

    payload = {
        "meta": {
            "season": season,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "player_count": len(players),
            "scoring": "half-PPR, 4pt passing TD",
            "source_file": f"output/fantasy_points_{season}.csv",
        },
        "tier_gaps": {
            "overall": OVERALL_TIER_GAP,
            "flex": FLEX_TIER_GAP,
            **DEFAULT_TIER_GAPS,
        },
        "tier_summary": tier_summary(df),
        "players": players,
    }

    os.makedirs(DRAFT_DATA_DIR, exist_ok=True)
    out_path = os.path.join(DRAFT_DATA_DIR, f"players_{season}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, allow_nan=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export draft assistant data")
    parser.add_argument("--season", type=int, default=2026)
    args = parser.parse_args()
    path = export_draft_data(args.season)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
