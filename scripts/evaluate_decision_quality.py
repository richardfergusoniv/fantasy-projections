"""Orchestrate E1 decision-quality evaluation and gate artifacts."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.projection.evaluation.decision_quality import (
    DECISION_QUALITY_DIR,
    DEFAULT_FOLDS,
    FROZEN_BASELINE_ID,
    evaluate_rolling_folds,
    write_evidence_bundle,
)
from src.projection.evaluation.decision_quality_gate import (
    build_decision_quality_gate,
    load_frozen_baseline_manifest,
    write_decision_quality_gate,
)


def _freeze_baseline(manifest_path: Path) -> Path:
    frozen_root = DECISION_QUALITY_DIR / "frozen" / FROZEN_BASELINE_ID
    bundle_root = manifest_path.parent
    if frozen_root.exists():
        shutil.rmtree(frozen_root)
    shutil.copytree(bundle_root, frozen_root)
    frozen_manifest = json.loads((frozen_root / "manifest.json").read_text(encoding="utf-8"))
    frozen_manifest["frozen_baseline_id"] = FROZEN_BASELINE_ID
    frozen_manifest["bundle_id"] = FROZEN_BASELINE_ID
    (frozen_root / "manifest.json").write_text(json.dumps(frozen_manifest, indent=2), encoding="utf-8")
    return frozen_root / "manifest.json"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", default=",".join(str(f) for f in DEFAULT_FOLDS))
    parser.add_argument("--bundle-id", default="decision_quality_current")
    parser.add_argument("--freeze-baseline", action="store_true")
    parser.add_argument("--market-blend", type=float, default=0.35)
    args = parser.parse_args(argv)

    folds = tuple(int(x.strip()) for x in args.folds.split(",") if x.strip())
    payload = evaluate_rolling_folds(folds=folds, market_blend=args.market_blend)
    manifest_path = write_evidence_bundle(payload, bundle_id=args.bundle_id)
    print(f"Wrote evidence bundle manifest: {manifest_path}")

    frozen_manifest = load_frozen_baseline_manifest()
    if args.freeze_baseline or frozen_manifest is None:
        frozen_path = _freeze_baseline(manifest_path)
        frozen_manifest = load_frozen_baseline_manifest()
        print(f"Froze baseline: {frozen_path}")

    evidence_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gate = build_decision_quality_gate(
        evidence_manifest=evidence_manifest,
        evaluation_payload=payload,
        frozen_baseline_manifest=frozen_manifest,
        required_folds=folds,
    )
    gate_path = write_decision_quality_gate(gate)
    print(f"Gate verdict: {gate['verdict']} ({', '.join(gate['reasons']) or 'no failures'})")
    print(f"Wrote gate artifact: {gate_path}")


if __name__ == "__main__":
    main()
