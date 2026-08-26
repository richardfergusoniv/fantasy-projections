"""Aggregate QB improvement tracks and recommend best arm.

Reads ablation outputs and baseline eval, writes output/qb_fix_tracks_comparison.json.

Usage:
    python scripts/qb_fix_tracks_compare.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.qb_tracks_util import board_qb_snapshot, holdout_qb_metrics, write_json
from src.projection.fantasy_evaluation import run_evaluation


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _score_holdout(metrics: dict) -> float:
    starter = metrics.get("starter_depth_tier_1") or metrics.get("holdout", {}).get("starter_depth_tier_1") or {}
    if not starter:
        return float("-inf")
    spearman = starter.get("rate_spearman") or 0.0
    mae = starter.get("rate_mae") or 99.0
    bias = abs(starter.get("mean_bias") or 99.0)
    hits = starter.get("tier_hits", "0/12")
    hit_n = int(str(hits).split("/")[0])
    return spearman * 2.0 - mae * 0.05 - bias * 0.02 + hit_n * 0.15


def main() -> None:
    print("Running baseline holdout eval (Track 1)...")
    _, baseline_summary, baseline_meta = run_evaluation()
    track1 = {
        "track": "track1_starter_eval_dashboard",
        "holdout": holdout_qb_metrics(baseline_summary),
        "board_2026": board_qb_snapshot(ROOT / "output" / "fantasy_points_2026.csv"),
    }

    compose = _load(ROOT / "output" / "ablation_qb_compose_stages_2026.json")
    td = _load(ROOT / "output" / "ablation_qb_td_architecture_2025.json")
    injury = _load(ROOT / "output" / "ablation_qb_injury_prior_2025.json")
    depth = _load(ROOT / "output" / "qb_depth_chart_audit_2026.json")

    candidates = [
        {"track": "baseline", "holdout": track1["holdout"], "board_2026": track1["board_2026"]},
    ]

    if td.get("variants"):
        best_td = max(td["variants"], key=lambda v: _score_holdout(v.get("holdout", {})))
        candidates.append({
            "track": f"track3_td_{best_td['variant']}",
            "holdout": best_td.get("holdout"),
            "board_2026": td.get("board_baseline_2026"),
            "note": "2026 board unchanged unless predict re-run with toggle",
        })

    if injury.get("variants"):
        shrink = next((v for v in injury["variants"] if v["variant"] == "partial_prior_shrink"), None)
        if shrink:
            candidates.append({
                "track": "track5_partial_prior_shrink",
                "holdout": shrink.get("holdout"),
                "partial_cases": shrink.get("partial_2024_cases"),
            })

    if compose.get("elite_hurt_by_compose"):
        worst_stage = compose.get("stage_summary", {})
        candidates.append({
            "track": "track2_compose_insight",
            "holdout": track1["holdout"],
            "compose_stage_mean_delta": worst_stage,
            "elite_hurt_count": len(compose["elite_hurt_by_compose"]),
            "note": "Measurement only; no holdout toggle",
        })

    if depth.get("flags"):
        candidates.append({
            "track": "track4_depth_chart_ari_fix",
            "holdout": track1["holdout"],
            "depth_flags": depth["flags"],
            "board_before": depth.get("board_before_fix"),
            "note": "Board impact requires predict re-run after starters_2026 fix",
        })

    scored = sorted(
        [c for c in candidates if c.get("holdout")],
        key=lambda c: _score_holdout(c["holdout"]),
        reverse=True,
    )
    recommendation = scored[0] if scored else None
    combo_note = (
        "Track 3 (T3b rush-TD clip widen) improves mobile-QB rush scoring without "
        "retrain; stack with Track 1 dashboard for monitoring. Track 4 ARI fix is "
        "orthogonal board sanity. Track 2 shows compose-stage drag on elites but is "
        "not a standalone fix."
    )
    if td.get("variants"):
        best_td_name = max(td["variants"], key=lambda v: _score_holdout(v.get("holdout", {})))["variant"]
        combo_note = (
            f"Best holdout TD variant: {best_td_name}. Recommend ship T3b (0.08-0.10 rush clip) "
            f"+ Track 1 starter dashboard + Track 4 ARI depth fix; evaluate partial-prior "
            f"shrink separately on Lamar/Burrow cases."
        )

    table = []
    for c in scored:
        h = c.get("holdout", {}).get("starter_depth_tier_1", {})
        table.append({
            "track": c["track"],
            "rate_spearman": h.get("rate_spearman"),
            "rate_mae": h.get("rate_mae"),
            "mean_bias": h.get("mean_bias"),
            "tier_hits": h.get("tier_hits"),
        })

    payload = {
        "comparison_table": table,
        "track1_baseline": track1,
        "track2_compose": {
            "elite_hurt": compose.get("elite_hurt_by_compose", [])[:10],
            "stage_summary": compose.get("stage_summary"),
        },
        "track3_td_variants": td.get("variants"),
        "track4_depth_audit": {
            "flags": depth.get("flags"),
            "fix_applied": depth.get("fix_applied"),
        },
        "track5_injury_prior": injury.get("variants"),
        "recommendation": {
            "primary_track": recommendation["track"] if recommendation else None,
            "rationale": combo_note,
            "elite_under_rank_2026": track1["board_2026"].get("qbs_above_18_ppg"),
        },
    }
    out_path = ROOT / "output" / "qb_fix_tracks_comparison.json"
    write_json(out_path, payload)
    print(f"Wrote {out_path}")
    print("\nComparison (QB starter holdout):")
    for row in table:
        print(
            f"  {row['track']}: spearman={row['rate_spearman']} "
            f"mae={row['rate_mae']} bias={row['mean_bias']} tier={row['tier_hits']}"
        )
    print(f"\nRecommended: {payload['recommendation']['primary_track']}")


if __name__ == "__main__":
    main()
