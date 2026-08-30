"""Orchestrate E2 QB-context paired evaluation, evidence freeze, and gate.

Mandatory stop after this script completes: a passing gate authorizes review
only, not production integration or O1/A1/P1/W1 work.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.projection.data_prep import get_conn
from src.projection.evaluation.decision_quality import DEFAULT_FOLDS
from src.projection.evaluation.qb_context_evaluation import (
    FROZEN_BASELINE_ID,
    QB_CONTEXT_EVAL_DIR,
    evaluate_qb_context_rolling,
    run_temporal_mutation_check,
    write_qb_context_evidence,
)
from src.projection.evaluation.qb_context_gate import (
    build_qb_context_gate,
    load_frozen_qb_baseline_manifest,
    write_qb_context_gate,
)
from src.projection.features import build_player_season_features


def _freeze_no_context_baseline(manifest_path: Path) -> Path:
    frozen_root = QB_CONTEXT_EVAL_DIR / "frozen" / FROZEN_BASELINE_ID
    bundle_root = manifest_path.parent
    if frozen_root.exists():
        shutil.rmtree(frozen_root)
    shutil.copytree(bundle_root, frozen_root)
    frozen_manifest = json.loads((frozen_root / "manifest.json").read_text(encoding="utf-8"))
    frozen_manifest["bundle_id"] = FROZEN_BASELINE_ID
    frozen_manifest["frozen_baseline_id"] = FROZEN_BASELINE_ID
    (frozen_root / "manifest.json").write_text(json.dumps(frozen_manifest, indent=2), encoding="utf-8")
    return frozen_root / "manifest.json"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", default=",".join(str(f) for f in DEFAULT_FOLDS))
    parser.add_argument("--bundle-id", default="qb_context_candidate")
    parser.add_argument("--freeze-no-context-baseline", action="store_true")
    parser.add_argument("--run-mutation-check", action="store_true")
    args = parser.parse_args(argv)

    folds = tuple(int(x.strip()) for x in args.folds.split(",") if x.strip())

    if args.run_mutation_check:
        conn = get_conn()
        try:
            feat = build_player_season_features(conn)
            ok = run_temporal_mutation_check(conn, feat, source_season=2024, target_season=2025)
            print(f"Temporal mutation invariance: {'pass' if ok else 'FAIL'}")
            if not ok:
                raise SystemExit(1)
        finally:
            conn.close()

    payload = evaluate_qb_context_rolling(folds=folds)
    manifest_path = write_qb_context_evidence(payload, bundle_id=args.bundle_id)
    print(f"Wrote E2 evidence manifest: {manifest_path}")

    frozen_manifest = load_frozen_qb_baseline_manifest()
    if args.freeze_no_context_baseline or frozen_manifest is None:
        frozen_path = _freeze_no_context_baseline(manifest_path)
        frozen_manifest = load_frozen_qb_baseline_manifest()
        print(f"Froze no-context baseline: {frozen_path}")

    evidence_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gate = build_qb_context_gate(
        evidence_manifest=evidence_manifest,
        evaluation_payload=payload,
        frozen_baseline_manifest=frozen_manifest,
        required_folds=folds,
    )
    gate_path = write_qb_context_gate(gate)
    print(f"E2 gate verdict: {gate['verdict']} ({', '.join(gate['reasons']) or 'no failures'})")
    print(f"Wrote E2 gate artifact: {gate_path}")
    print("Mandatory stop: review E2 evidence before any O1/A1/P1/W1 work.")


if __name__ == "__main__":
    main()
