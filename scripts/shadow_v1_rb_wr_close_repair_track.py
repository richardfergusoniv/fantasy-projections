"""Audit finalization/ladder/corrections; close RB/WR repair track if clean."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.projection.shadow.finalization_audit import run_finalization_audit
from src.projection.shadow.forbidden import ForbiddenImportGuard


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--no-close",
        action="store_true",
        help="Run audit only; do not write repair_track_closed.json",
    )
    args = parser.parse_args(argv)
    with ForbiddenImportGuard():
        result = run_finalization_audit(
            out_dir=args.out_dir,
            close_track=not args.no_close,
        )
    closeout = result.get("closeout") or {}
    print(json.dumps({
        "finding": result.get("finding"),
        "defects": result.get("defects"),
        "verdict": closeout.get("verdict"),
        "v1_role": closeout.get("v1_role"),
        "repair_track_status": closeout.get("repair_track_status"),
        "further_repair_authorized": closeout.get("further_repair_authorized"),
        "production_weights_unchanged": result.get("production_weights_unchanged"),
        "audit_path": result.get("audit_path"),
    }, indent=2))
    return 0 if result.get("finding") == "no_cutoff_available_defect" else 1


if __name__ == "__main__":
    raise SystemExit(main())
