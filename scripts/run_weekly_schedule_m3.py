"""Run Milestone 3 weekly availability + conversions (research/shadow only).

Does not promote, reseal, replace Vegas weekly props, blend Role 3, or
change League Value / freeze knobs / APP_PROJECTION_SOURCE / PWA.

Week-varying A_i,w and conversion latents on top of the M2 team-week latent.
Bye forces A=0. Season = sum of weeks. Same-week team_attempts / team_carries /
team_targets / team_air_yards are refused. ADP and season-long Vegas are not
drivers.

Examples:

  uv run python scripts/run_weekly_schedule_m3.py --dry-run
  uv run python scripts/run_weekly_schedule_m3.py --season 2026

Windows, against the local data dir (optional; fixture priors work without it):

  set FANTASY_PROJECTIONS_DATA_DIR=D:\\fantasy-projections-data
  set FANTASY_PROJECTIONS_DB_PATH=D:\\fantasy-projections-data\\projections.db
  uv run python scripts/run_weekly_schedule_m3.py --season 2026
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.projection.weekly_latent.run import run_milestone3  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Milestone 3: week-varying availability + conversions (shadow only)."
        )
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
        "--opp-epa-priors",
        default=None,
        help="Prior-season defensive EPA CSV (default: committed 2025 fixture).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Shadow output directory (default: output/shadow_weekly_schedule_m3).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Synthetic 2-team slate; does not need the sealed board or DB.",
    )
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Skip the synthetic + historical rolling-origin leakage harness.",
    )
    args = parser.parse_args()
    result = run_milestone3(
        season=args.season,
        projections_path=args.projections,
        schedule_path=args.schedule,
        output_dir=args.output,
        opp_epa_priors_path=args.opp_epa_priors,
        dry_run=args.dry_run,
        run_backtest=not args.skip_backtest,
    )
    print(
        json.dumps(
            {
                "conservation_passes": result.conservation["passes"],
                "failing_checks": result.conservation["failing_checks"],
                "backtest_passes": result.backtest.get("passes"),
                "output": result.output_paths,
                "dry_run": result.summary["dry_run"],
                "n_player_weeks": result.summary["n_player_weeks"],
                "still_shadow": True,
                "gate_verdict": "not promoting",
                "vegas_compare_role": result.vegas_compare.get("role"),
                "role3_blend": result.vegas_compare.get("role3_blend"),
            },
            indent=2,
        )
    )
    ok = result.conservation["passes"] and result.backtest.get("passes", True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
