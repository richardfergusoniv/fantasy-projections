"""CLI for weekly feature as-of contract audit."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.projection.weekly_audit.feature_contract import audit_feature_contracts, write_audit_artifacts


def main() -> int:
    report = audit_feature_contracts()
    paths = write_audit_artifacts(report)
    print(json.dumps({"passes": report["passes"], "paths": paths}, indent=2))
    return 0 if report["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
