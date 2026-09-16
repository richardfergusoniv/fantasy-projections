"""Run Milestone 1 weekly schedule allocation (research/shadow only).

Does not promote, reseal, train, or change League Value / freeze knobs / PWA.

Examples:

  uv run python scripts/run_weekly_schedule_m1.py --dry-run
  uv run python scripts/run_weekly_schedule_m1.py --season 2026

Windows, against the local data dir:

  set FANTASY_PROJECTIONS_DATA_DIR=D:\\fantasy-projections-data
  set FANTASY_PROJECTIONS_DB_PATH=D:\\fantasy-projections-data\\projections.db
  uv run python scripts/run_weekly_schedule_m1.py --season 2026
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.projection.weekly_latent.run import run_milestone1  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Milestone 1: deterministic weekly schedule allocation (shadow only)."
    )
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument(
        "--projections",
        default=None,
        help="Sealed long projections CSV (default: v2_baseline_20260830 public copy).",
    )
    parser.add_argument(
        "--schedule",
        default=None,
        help="REG schedule CSV (default: committed 2026 fixture).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Shadow output directory (default: output/shadow_weekly_schedule_m1).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Synthetic 2-team slate; does not need the sealed board or DB.",
    )
    args = parser.parse_args()
    result = run_milestone1(
        season=args.season,
        projections_path=args.projections,
        schedule_path=args.schedule,
        output_dir=args.output,
        dry_run=args.dry_run,
    )
    print(
        json.dumps(
            {
                "conservation_passes": result.conservation["passes"],
                "failing_checks": result.conservation["failing_checks"],
                "output": result.output_paths,
                "dry_run": result.summary["dry_run"],
                "n_player_weeks": result.summary["n_player_weeks"],
            },
            indent=2,
        )
    )
    return 0 if result.conservation["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
