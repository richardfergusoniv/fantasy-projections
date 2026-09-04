#!/usr/bin/env python3
"""Build H4 designed/scramble coverage fixture from weekly PBP cache."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.projection.qb_h4.designed_coverage import write_coverage_fixture
from src.projection.qb_h4.decision_policy import decision_policy_dict

OUT = ROOT / "output" / "qb_h4"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    # Predeclare decision policy before any final fold metrics are written.
    policy = decision_policy_dict()
    (OUT / "predeclared_decision_policy.json").write_text(
        json.dumps(policy, indent=2), encoding="utf-8"
    )
    manifest = write_coverage_fixture()
    print("coverage seasons", manifest["seasons_with_coverage"])
    print("uncovered", manifest["seasons_uncovered_no_pbp_in_repo"])
    print("n_player_seasons", manifest["n_player_seasons"])
    print("content_hash", manifest["content_hash"][:12])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
