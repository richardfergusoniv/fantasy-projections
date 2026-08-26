"""Ablation: exposure blend alpha and rate Spearman (E1 / E3).

Runs the leakage-safe 2024→2025 harness at α ∈ {0, 0.25, 0.5, 0.75, 1.0} and
prints season-total Spearman/MAE plus model rate Spearman (E3).

Usage:
    python scripts/ablate_exposure.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.projection.fantasy_evaluation import run_evaluation

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
POSITIONS = ("QB", "RB", "WR", "TE")


def main() -> None:
    results = []
    for alpha in ALPHAS:
        _, summary, metadata = run_evaluation(exposure_blend_alpha=alpha)
        model = summary[
            (summary["method"] == "model") & (summary["scope"] == "all_eligible")
        ]
        arm = {"exposure_blend_alpha": alpha}
        for pos in POSITIONS:
            row = model[model["position"] == pos]
            if row.empty:
                continue
            r = row.iloc[0]
            arm[pos] = {
                "spearman": round(float(r["spearman"]), 4),
                "rate_spearman": round(float(r["rate_spearman"]), 4),
                "points_mae": round(float(r["points_mae"]), 2),
                "tier_hits": f"{int(r['tier_hits'])}/{int(r['tier_rank'])}",
            }
        results.append(arm)
        print(f"\n=== alpha={alpha} ===")
        for pos in POSITIONS:
            if pos in arm:
                print(
                    f"  {pos}: spearman_season={arm[pos]['spearman']:.4f} "
                    f"spearman_rate={arm[pos]['rate_spearman']:.4f} "
                    f"MAE={arm[pos]['points_mae']:.1f} "
                    f"tier={arm[pos]['tier_hits']}"
                )

    out_path = ROOT / "output" / "ablation_exposure_blend_2025.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
