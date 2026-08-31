"""Ablation: QB partial-season prior shrink (Track 5).

Compares baseline vs shrinking prior_role_rate when source games < 12.

Usage:
    python scripts/ablate_qb_injury_prior.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.qb_tracks_util import holdout_qb_metrics, write_json
from src.projection.data_prep import get_conn
from src.projection.features import build_player_season_features
from src.projection.fantasy_evaluation import run_evaluation

PARTIAL_CASES = (
    ("Lamar Jackson", "00-0034796"),
    ("Joe Burrow", "00-0036442"),
)


def _partial_season_cases(feat: pd.DataFrame) -> list[dict]:
    rows = []
    for name, pid in PARTIAL_CASES:
        s2024 = feat[(feat["player_id"].eq(pid)) & (feat["season"].eq(2024))]
        if s2024.empty:
            continue
        gp = float(s2024.iloc[0]["games_played"])
        rows.append({"display_name": name, "player_id": pid, "source_2024_games": gp})
    return rows


def main() -> None:
    conn = get_conn()
    try:
        feat = build_player_season_features(conn)
    finally:
        conn.close()

    arms = (
        {"name": "baseline", "qb_partial_prior_shrink": False},
        {"name": "partial_prior_shrink", "qb_partial_prior_shrink": True},
    )
    results = []
    for arm in arms:
        print(f"\n=== {arm['name']} ===")
        ranked, summary, _ = run_evaluation(qb_partial_prior_shrink=arm["qb_partial_prior_shrink"])
        metrics = holdout_qb_metrics(summary)
        partial_eval = []
        for case in _partial_season_cases(feat):
            row = ranked[ranked["player_id"].eq(case["player_id"])]
            if row.empty:
                continue
            r = row.iloc[0]
            partial_eval.append({
                **case,
                "actual_rate_ppg": round(float(r["actual_rate_points"]), 3),
                "model_rate_ppg": round(float(r["model_rate_points"]), 3),
                "bias_ppg": round(float(r["model_rate_points"] - r["actual_rate_points"]), 3),
            })
        arm_out = {"variant": arm["name"], "holdout": metrics, "partial_2024_cases": partial_eval}
        results.append(arm_out)
        starter = metrics.get("starter_depth_tier_1", {})
        print(
            f"  starter rate_spearman={starter.get('rate_spearman')} "
            f"mean_bias={starter.get('mean_bias')} tier={starter.get('tier_hits')}"
        )
        for pe in partial_eval:
            print(
                f"  {pe['display_name']} (2024 gp={pe['source_2024_games']:.0f}): "
                f"proj={pe['model_rate_ppg']} act={pe['actual_rate_ppg']} bias={pe['bias_ppg']:+.2f}"
            )

    payload = {
        "description": "2024->2025 holdout QB partial-season prior shrink",
        "threshold_games": 12,
        "variants": results,
    }
    out_path = ROOT / "output" / "ablation_qb_injury_prior_2025.json"
    write_json(out_path, payload)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
