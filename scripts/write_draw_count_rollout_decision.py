#!/usr/bin/env python3
"""Write draw_count_rollout_decision.json from frozen evidence."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.projection.evaluation.draw_count_rollout import write_draw_count_rollout_decision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument(
        "--freeze-id",
        default="draw_stability_intermediate_v20k_2026",
    )
    parser.add_argument(
        "--artifact-namespace",
        default="rc_10k_20260828",
    )
    parser.add_argument(
        "--rollout-label",
        default="decision-stable_numerically-not-validated_rc",
    )
    parser.add_argument(
        "--runtime-estimate-minutes",
        type=float,
        default=84.0,
        help="Planning estimate only; RC run replaces with measured runtime.",
    )
    parser.add_argument(
        "--runtime-estimate-basis",
        default="prior full-manifest simulation timing (~806s at 1k); 10k not linear from 1k alone",
    )
    args = parser.parse_args()

    path = write_draw_count_rollout_decision(
        season=args.season,
        freeze_id=args.freeze_id,
        chosen_rollout_namespace=args.artifact_namespace,
        rollout_label=args.rollout_label,
        runtime_estimate_minutes=args.runtime_estimate_minutes,
        runtime_estimate_basis=args.runtime_estimate_basis,
    )
    print(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
