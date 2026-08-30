#!/usr/bin/env python3
"""Compare overlay metrics across draw profiles when contracts match."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.projection.evaluation.draw_profile_comparison import compare_draw_profile_overlays
from src.projection.evaluation.evidence_freeze import load_freeze_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--rc-namespace", required=True)
    parser.add_argument("--freeze-id", default="draw_stability_intermediate_v20k_2026")
    parser.add_argument(
        "--production-players",
        type=Path,
        default=Path("draft_assistant/data/players_2026.json"),
    )
    parser.add_argument(
        "--production-manifest",
        type=Path,
        default=Path("output/model_v3/simulation_manifest_2026.json"),
    )
    args = parser.parse_args()

    from src.projection.release_candidate import rc_namespace_dir

    rc_dir = rc_namespace_dir(args.season, args.rc_namespace)
    profiles = {
        "production_1k": {
            "manifest": args.production_manifest,
            "players": args.production_players,
        },
        f"rc_{args.rc_namespace}": {
            "manifest": rc_dir / f"simulation_manifest_{args.season}.json",
            "players": rc_dir / f"players_{args.season}_rc.json",
        },
    }
    freeze = load_freeze_manifest(args.freeze_id)
    report = compare_draw_profile_overlays(
        season=args.season,
        profiles=profiles,
    )
    report["frozen_evidence"] = {
        "freeze_id": args.freeze_id,
        "selected_board_hash": freeze.get("selected_board_hash"),
        "canonical_projection_run_id": freeze.get("canonical_projection_run_id"),
    }
    print(json.dumps(report, indent=2))
    return 0 if report.get("comparison_verdict") == "compare" else 1


if __name__ == "__main__":
    raise SystemExit(main())
