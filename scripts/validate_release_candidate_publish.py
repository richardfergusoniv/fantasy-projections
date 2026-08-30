#!/usr/bin/env python3
"""Validate a namespaced release-candidate publish."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.projection.evaluation.release_candidate_validation import validate_release_candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--artifact-namespace", required=True)
    parser.add_argument("--freeze-id", default="draw_stability_intermediate_v20k_2026")
    parser.add_argument("--expected-overlay-players", type=int, default=778)
    args = parser.parse_args()

    report = validate_release_candidate(
        season=args.season,
        namespace=args.artifact_namespace,
        freeze_id=args.freeze_id,
        expected_overlay_players=args.expected_overlay_players,
    )
    print(json.dumps(report, indent=2))
    return 0 if report.get("verdict") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
