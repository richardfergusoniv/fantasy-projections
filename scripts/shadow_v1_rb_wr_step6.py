"""Step-6 fold×position×top-120 decision table and freeze gate."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.projection.shadow.forbidden import ForbiddenImportGuard
from src.projection.shadow.step6_decision import run_step6_decision


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    with ForbiddenImportGuard():
        decision = run_step6_decision(out_dir=args.out_dir)
    print(json.dumps({
        "verdict": decision.get("verdict"),
        "recommended_direction": decision.get("recommended_direction"),
        "selected_for_freeze": decision.get("selected_for_freeze"),
        "pipeline_location": (decision.get("labeling") or {}).get("pipeline_location"),
        "codominant_error_components": (decision.get("labeling") or {}).get(
            "codominant_error_components"
        ),
        "producing_commit": decision.get("producing_commit"),
        "production_weights_unchanged": decision.get("production_weights_unchanged"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
