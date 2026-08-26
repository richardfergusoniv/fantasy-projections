"""Audit whether sentiment history is sufficient to attempt model activation.

This command deliberately does not activate a position. It verifies the
minimum data prerequisites before an end-to-end rolling fantasy ablation is
meaningful; activation still requires the performance thresholds documented
in the sentiment manifest/report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.sentiment.snapshot import DEFAULT_SENTIMENT_DIR


MIN_SEASONS = 3
MIN_NON_NULL_PLAYER_SEASONS = 200
MIN_POSITION_COVERAGE = 0.40


def audit_snapshots(paths: list[str | Path]) -> dict:
    frames = []
    for path_value in paths:
        path = Path(path_value)
        frame = pd.read_csv(path)
        required = {"player_id", "position", "sentiment_feature", "sentiment_as_of"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} missing sentiment columns: {sorted(missing)}")
        match = path.stem.split("_")
        season = next((int(part) for part in match if part.isdigit() and len(part) == 4), None)
        if season is None:
            raise ValueError(f"Could not infer season from {path.name}")
        frame = frame.copy()
        frame["snapshot_season"] = season
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    seasons = sorted(combined.get("snapshot_season", pd.Series(dtype=int)).unique().tolist())
    by_position: dict[str, dict] = {}
    for position in ("QB", "RB", "WR", "TE"):
        subset = combined[combined["position"].eq(position)] if not combined.empty else combined
        non_null = int(subset["sentiment_feature"].notna().sum()) if not subset.empty else 0
        coverage = float(subset["sentiment_feature"].notna().mean()) if len(subset) else 0.0
        prerequisites = (
            len(seasons) >= MIN_SEASONS
            and non_null >= MIN_NON_NULL_PLAYER_SEASONS
            and coverage >= MIN_POSITION_COVERAGE
        )
        by_position[position] = {
            "rows": int(len(subset)),
            "non_null_player_seasons": non_null,
            "coverage": coverage,
            "prerequisites_met": prerequisites,
            "model_active": False,
        }
    return {
        "snapshot_count": len(paths),
        "seasons": seasons,
        "minimums": {
            "seasons": MIN_SEASONS,
            "non_null_player_seasons": MIN_NON_NULL_PLAYER_SEASONS,
            "position_coverage": MIN_POSITION_COVERAGE,
        },
        "by_position": by_position,
        "ready_for_ablation": any(v["prerequisites_met"] for v in by_position.values()),
        "note": "Passing prerequisites permits an ablation; it does not activate sentiment.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Sentiment snapshot CSVs")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    paths = args.paths or sorted(DEFAULT_SENTIMENT_DIR.glob("sentiment_*.csv"))
    report = audit_snapshots(paths)
    rendered = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
