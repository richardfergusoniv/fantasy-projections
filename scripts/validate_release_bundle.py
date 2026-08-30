#!/usr/bin/env python3
"""Validate a sealed namespaced release bundle."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.projection.evaluation.release_bundle_validation import validate_release_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--artifact-namespace", required=True)
    parser.add_argument("--require-active", action="store_true")
    args = parser.parse_args()
    report = validate_release_bundle(
        season=args.season,
        namespace=args.artifact_namespace,
        require_active=args.require_active,
    )
    print(json.dumps(report, indent=2))
    return 0 if report.get("verdict") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
