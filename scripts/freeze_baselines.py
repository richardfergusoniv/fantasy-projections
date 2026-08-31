"""Freeze baseline artifacts and publish the draft-edge scoreboard contract.

Does not modify training or composition. Writes a manifest under
output/test_before_rewrite/ for the Phase-1 tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "output" / "test_before_rewrite"

# Frozen baselines referenced by the test-before-rewrite plan.
BASELINE_PATHS = [
    "output/fantasy_points_2026.csv",
    "output/projections_2026.csv",
    "output/fantasy_evaluation_2025.csv",
    "output/fantasy_evaluation_summary_2025.json",
    "output/model_accuracy_compare_2025.json",
    "data/consensus/consensus_2026.json",
    "output/backtest/manifest.json",
    "output/backtest/residuals_rolling.parquet",
    "output/backtest/calibration_report.json",
    "output/model_v2/fantasy_points_2026.csv",
]

SCOREBOARD = {
    "primary_decision": "preseason_draft_value_vs_adp",
    "primary_metrics": [
        {
            "id": "spearman_vs_adp",
            "description": "Rank Spearman of model vs contemporaneous ADP inside matched set",
        },
        {
            "id": "draft_edge_proxy",
            "description": (
                "After residualizing on ADP rank, correlate (model_rank - adp_rank) "
                "with actual season points; negative correlation => actionable edge"
            ),
        },
        {
            "id": "points_surplus_model_higher",
            "description": (
                "Mean actual points when model ranks player above ADP minus "
                "when model ranks below"
            ),
        },
    ],
    "secondary_metrics": [
        "holdout_spearman",
        "holdout_points_mae",
        "tier_hit_rate",
    ],
    "non_goals": [
        "sleeper_agreement",
        "prop_closing_line_value",
        "dfs_lineup_roi",
    ],
    "join_rules": [
        "Match player_id then normalized name+position",
        "Re-rank both sides inside the matched set",
        "Optionally truncate to max_market_rank (draftable window)",
    ],
}


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest() -> dict:
    artifacts = []
    for rel in BASELINE_PATHS:
        path = REPO_ROOT / rel
        artifacts.append(
            {
                "path": rel.replace("\\", "/"),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else None,
                "sha256": _sha256(path),
            }
        )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Freeze v1/v2 boards and define draft-edge scoreboard (no model rewrite)",
        "scoreboard": SCOREBOARD,
        "artifacts": artifacts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_DIR / "freeze_manifest.json",
    )
    args = parser.parse_args()
    manifest = build_manifest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    missing = [a["path"] for a in manifest["artifacts"] if not a["exists"]]
    print(f"Wrote {args.out}")
    print(f"Artifacts: {len(manifest['artifacts'])} tracked, {len(missing)} missing")
    for m in missing:
        print(f"  missing: {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
