"""Run sealed availability-only Gate-A blend shadow repair."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.projection.shadow.availability_repair import run_availability_repair
from src.projection.shadow.forbidden import ForbiddenImportGuard


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    with ForbiddenImportGuard():
        seal = run_availability_repair(out_dir=args.out_dir)
    print(json.dumps({
        "candidate_id": seal.get("candidate_id"),
        "verdict": (seal.get("freeze") or {}).get("verdict"),
        "gate_passed": (seal.get("freeze") or {}).get("gate_passed"),
        "promotion_authorized": (seal.get("freeze") or {}).get("promotion_authorized"),
        "producing_commit": seal.get("producing_commit"),
        "config_sha256": seal.get("config_sha256"),
        "code_bundle_sha256": (seal.get("code_identity") or {}).get(
            "entrypoint_bundle_sha256"
        ),
        "production_weights_unchanged": seal.get("production_weights_unchanged"),
        "nested_alpha_fits": [
            {
                "fold": f.get("fold"),
                "alpha_by_position": f.get("alpha_by_position"),
            }
            for f in (seal.get("nested_alpha_fits") or [])
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
