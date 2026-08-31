#!/usr/bin/env python3
"""Write Phase 2 rollout closure artifact and apply production risk flag."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.projection.evaluation.draw_count_rollout import (
    OPERATIONAL_POLICIES,
    apply_draw_count_risk_flag,
    update_release_pointer_profile,
    write_phase2_rollout_closure,
    profile_for_operational_policy,
)

DEFAULT_RATIONALE = (
    "Phase 2 RC at 10k passed operational validation (778 overlays, public artifacts unchanged, "
    "frozen board hash aligned) but overlay comparison held on replacement_contract_hash, and "
    "all sub-20k nested-prefix candidates fail the strict numerical gate vs 20k. Measured RC "
    "runtime was 9774s (~163 min), making an immediate 20k production move costly without a "
    "prior 20k RC. Retain 1,000 draws as provisional_current_configuration with an explicit "
    "release-report risk flag until runtime capacity improves or sampling design changes."
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--freeze-id", default="draw_stability_intermediate_v20k_2026")
    parser.add_argument("--rc-namespace", default="rc_10k_20260828")
    parser.add_argument(
        "--operational-policy",
        choices=OPERATIONAL_POLICIES,
        default="maintain_1000_temporarily",
    )
    parser.add_argument("--decided-by", default="Richard")
    parser.add_argument("--rationale", default=DEFAULT_RATIONALE)
    parser.add_argument(
        "--human-decision-record",
        default="docs/decisions/DRAW_COUNT_ROLLOUT_HUMAN_DECISION_2026-08-28.md",
    )
    parser.add_argument(
        "--overlay-comparison-json",
        type=Path,
        default=None,
        help="Optional path to compare_draw_profile_overlays JSON output",
    )
    parser.add_argument(
        "--promotion-gate-json",
        type=Path,
        default=None,
        help="Optional path to evaluate_decision_stable_10k_promotion JSON output",
    )
    parser.add_argument(
        "--skip-release-report-patch",
        action="store_true",
        help="Do not append risk flag to public release_report_<season>.json",
    )
    args = parser.parse_args()

    overlay_comparison = None
    if args.overlay_comparison_json and args.overlay_comparison_json.exists():
        overlay_comparison = json.loads(args.overlay_comparison_json.read_text(encoding="utf-8"))
    promotion_gate = None
    if args.promotion_gate_json and args.promotion_gate_json.exists():
        promotion_gate = json.loads(args.promotion_gate_json.read_text(encoding="utf-8"))

    path = write_phase2_rollout_closure(
        season=args.season,
        freeze_id=args.freeze_id,
        rc_namespace=args.rc_namespace,
        operational_policy=args.operational_policy,
        decided_by=args.decided_by,
        decision_rationale=args.rationale,
        human_decision_record_path=args.human_decision_record,
        overlay_comparison=overlay_comparison,
        promotion_gate=promotion_gate,
    )
    profile, _ = profile_for_operational_policy(args.operational_policy)
    update_release_pointer_profile(season=args.season, profile=profile)
    if not args.skip_release_report_patch:
        apply_draw_count_risk_flag(
            season=args.season,
            operational_policy=args.operational_policy,
        )

    print(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
