"""Ablation: QB TD architecture variants vs baseline holdout.

Variants:
  baseline     - ship defaults (rush TD clip hi=0.04, no T1-lite)
  t3b_008      - widen rush-TD/carry upper bound to 0.08
  t3b_010      - widen rush-TD/carry upper bound to 0.10
  t1_lite      - post-compose pass TD re-derive from attempts
  t3b010_t1    - stack T3b 0.10 + T1-lite

Usage:
    python scripts/ablate_qb_td_architecture.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.qb_tracks_util import board_qb_snapshot, holdout_qb_metrics, write_json
from src.projection.fantasy_evaluation import run_evaluation

ARMS = (
    {"name": "baseline", "qb_rush_td_clip_hi": None, "qb_pass_td_t1_lite": False},
    {"name": "t3b_008", "qb_rush_td_clip_hi": 0.08, "qb_pass_td_t1_lite": False},
    {"name": "t3b_010", "qb_rush_td_clip_hi": 0.10, "qb_pass_td_t1_lite": False},
    {"name": "t1_lite", "qb_rush_td_clip_hi": None, "qb_pass_td_t1_lite": True},
    {"name": "t3b010_t1", "qb_rush_td_clip_hi": 0.10, "qb_pass_td_t1_lite": True},
)


def main() -> None:
    board_base = board_qb_snapshot(ROOT / "output" / "fantasy_points_2026.csv")
    results = []
    for arm in ARMS:
        print(f"\n=== {arm['name']} ===")
        _, summary, metadata = run_evaluation(
            qb_rush_td_clip_hi=arm["qb_rush_td_clip_hi"],
            qb_pass_td_t1_lite=arm["qb_pass_td_t1_lite"],
        )
        metrics = holdout_qb_metrics(summary)
        row = {"variant": arm["name"], **arm, "holdout": metrics}
        results.append(row)
        starter = metrics.get("starter_depth_tier_1", {})
        print(
            f"  starter rate_spearman={starter.get('rate_spearman')} "
            f"rate_mae={starter.get('rate_mae')} mean_bias={starter.get('mean_bias')} "
            f"tier={starter.get('tier_hits')}"
        )

    payload = {
        "description": "2024->2025 holdout QB TD architecture ablation",
        "board_baseline_2026": board_base,
        "variants": results,
    }
    out_path = ROOT / "output" / "ablation_qb_td_architecture_2025.json"
    write_json(out_path, payload)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
