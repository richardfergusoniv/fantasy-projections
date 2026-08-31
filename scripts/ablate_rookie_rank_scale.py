"""Ablation: rookie ranking-only VORP haircut (R1).

Reports how ``ROOKIE_RANK_SCALE`` moves drafted rookies on the overall VORP
board relative to veterans. Scaling is uniform among rookies, so rookie-only
Spearman is unchanged; the effect is cross-position rank vs veterans.

Usage:
    python scripts/ablate_rookie_rank_scale.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.draft_assistant.tiers import add_tier_columns
from src.draft_assistant.vorp import add_vorp_columns, OVERALL_VORP_TIER_GAP
from src.projection.fantasy_evaluation import run_evaluation

SCALES = (1.0, 0.95, 0.90, 0.85, 0.80, 0.75)
WATCH_ROOKIES = (
    "Jeremiyah Love",
    "Carnell Tate",
)


def _rank_board(df: pd.DataFrame, scale: float) -> pd.DataFrame:
    out = add_vorp_columns(df.copy(), rookie_rank_scale=scale)
    out = add_tier_columns(
        out,
        points_col="vorp",
        overall_points_col="vorp",
        overall_gap=OVERALL_VORP_TIER_GAP,
    )
    return out.sort_values("overall_rank")


def main() -> None:
    fp = pd.read_csv(ROOT / "output" / "fantasy_points_2026.csv")
    fp = fp[fp["position"].isin(["QB", "RB", "WR", "TE"])].copy()

    holdout_rows, _, _ = run_evaluation()
    holdout = holdout_rows[holdout_rows["is_rookie"]].copy()

    results = []
    for scale in SCALES:
        board = _rank_board(fp, scale)
        rookies = board[board["low_confidence"].fillna(False) | board["source"].eq("rookie_rule")]
        watch = {
            r["display_name"]: int(r["overall_rank"])
            for _, r in board[board["display_name"].isin(WATCH_ROOKIES)].iterrows()
        }
        over_ranked = int(
            (
                (holdout["model_rate_points"] > holdout["actual_points"])
                & holdout["actual_points"].gt(0)
            ).sum()
        )
        arm = {
            "rookie_rank_scale": scale,
            "2026_rookie_top_overall_rank": int(rookies["overall_rank"].min()) if not rookies.empty else None,
            "2026_rookie_median_overall_rank": float(rookies["overall_rank"].median()) if not rookies.empty else None,
            "2026_watch_overall_ranks": watch,
            "2025_holdout_rookie_overprojected_n": over_ranked,
        }
        results.append(arm)
        print(
            f"scale={scale:.2f}  rookie_top=#{arm['2026_rookie_top_overall_rank']}  "
            f"median=#{arm['2026_rookie_median_overall_rank']:.0f}  "
            f"Love=#{watch.get('Jeremiyah Love', '-')}  "
            f"Tate=#{watch.get('Carnell Tate', '-')}"
        )

    out_path = ROOT / "output" / "ablation_rookie_rank_scale_2025.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
