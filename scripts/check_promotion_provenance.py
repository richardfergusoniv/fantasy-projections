#!/usr/bin/env python3
"""Check derived promotion provenance and whether a bundle is promotable now.

A naive ``validate_sealed_promotion_invariants`` call defaults to ``initial`` git
provenance and can report a false failure for a previously activated namespace
when ``HEAD`` has moved forward. This script derives ``initial`` vs ``restore``
from the pointer/receipt control plane before running the sealed invariant gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.projection.evaluation.promotion_invariants import validate_sealed_promotion_invariants
from src.projection.evaluation.release_bundle_validation import validate_release_bundle
from src.projection.promotion_receipt import derive_provenance_mode
from src.projection.release_bundle import bundle_root, load_sealed_manifest


def check_promotion_provenance(*, season: int, namespace: str) -> dict:
    manifest, manifest_sha256 = load_sealed_manifest(bundle_root(season, namespace))
    release_id = str(manifest["bundle"]["release_id"])
    source_commit = str((manifest.get("git") or {}).get("source_commit") or "")

    bundle_validation = validate_release_bundle(
        season=season,
        namespace=namespace,
        require_active=False,
    )
    provenance_mode = derive_provenance_mode(
        season=season,
        namespace=namespace,
        release_id=release_id,
        manifest_sha256=manifest_sha256,
    )
    promotion_invariants = validate_sealed_promotion_invariants(
        season=season,
        namespace=namespace,
        provenance_mode=provenance_mode,
    )

    naive_initial = None
    if provenance_mode == "restore":
        naive_initial = validate_sealed_promotion_invariants(
            season=season,
            namespace=namespace,
            provenance_mode="initial",
        )
        if (
            promotion_invariants.get("verdict") == "pass"
            and naive_initial.get("verdict") == "fail"
        ):
            git_check = next(
                (c for c in naive_initial.get("checks", []) if c.get("check") == "git_provenance"),
                None,
            )
            naive_warning = (
                "derived restore mode passes, but a naive initial-mode git check would fail"
            )
            if git_check and git_check.get("error"):
                naive_warning = (
                    f"{naive_warning}: {git_check['error']}"
                )
        else:
            naive_warning = None
    else:
        naive_warning = None

    promotable = (
        bundle_validation.get("verdict") == "pass"
        and promotion_invariants.get("verdict") == "pass"
    )
    return {
        "season": season,
        "namespace": namespace,
        "release_id": release_id,
        "manifest_sha256": manifest_sha256,
        "source_commit": source_commit,
        "provenance_mode": provenance_mode,
        "bundle_validation_verdict": bundle_validation.get("verdict"),
        "promotion_invariants_verdict": promotion_invariants.get("verdict"),
        "promotable": promotable,
        "verdict": "pass" if promotable else "fail",
        "naive_initial_warning": naive_warning,
        "bundle_validation": bundle_validation,
        "promotion_invariants": promotion_invariants,
        "naive_initial_invariants": naive_initial,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--artifact-namespace", required=True)
    args = parser.parse_args()
    report = check_promotion_provenance(
        season=args.season,
        namespace=args.artifact_namespace,
    )
    print(json.dumps(report, indent=2))
    return 0 if report.get("verdict") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
