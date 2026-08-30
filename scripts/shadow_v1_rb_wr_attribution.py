"""Non-mutating rolling-origin attribution for v1 RB/WR repair investigation.

Produces read-only diagnostics under output/shadow_v1_rb_wr/. Does not change
production ensemble weights or publish paths. Consensus membership is
hash-pinned and fail-closed; Sleeper-derived evidence is excluded.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.projection.shadow.consensus_pin import ConsensusPinError
from src.projection.shadow.forbidden import ForbiddenImportGuard
from src.projection.shadow.rb_wr_attribution import FOLDS, run_shadow_attribution


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override output directory (default: output/shadow_v1_rb_wr)",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=500,
        help="Bootstrap draws for repair-candidate intervals",
    )
    args = parser.parse_args(argv)
    try:
        with ForbiddenImportGuard():
            manifest = run_shadow_attribution(out_dir=args.out_dir, folds=FOLDS, n_boot=args.n_boot)
    except ConsensusPinError as exc:
        print(json.dumps({"status": "fail_closed", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps({
        "status": manifest.get("status"),
        "diagnosis": manifest.get("diagnosis"),
        "producing_commit": manifest.get("producing_commit"),
        "production_weights_unchanged": manifest.get("production_weights_unchanged"),
        "written": str(Path(args.out_dir) if args.out_dir else ROOT / "output" / "shadow_v1_rb_wr"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
