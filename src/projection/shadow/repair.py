"""Bounded shadow-only repair candidates for v1 RB/WR (never mutates production)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.projection.contracts import OUTPUT_DIR
from src.projection.evaluation.accuracy_first import canonical_json_hash, sha256_file
from src.projection.shadow.decision_rules import repair_gate
from src.projection.shadow.forbidden import assert_no_forbidden_imports
from src.projection.shadow.production_guard import (
    assert_production_unchanged,
    snapshot_production_artifacts,
)

SHADOW_OUTPUT_DIR = Path(OUTPUT_DIR) / "shadow_v1_rb_wr"
REPAIR_ENTRYPOINTS = (
    "src.projection.shadow.repair",
    "src.projection.shadow.rb_wr_attribution",
    "src.projection.shadow.decision_rules",
)


def freeze_shadow_candidate(
    *,
    candidate_id: str,
    code_identity: dict[str, Any],
    evidence: dict[str, Any],
    source_hashes: dict[str, str],
    board_2026_path: str | Path | None = None,
    fold_mae_relative_deltas: list[float],
    pooled_top120_spearman_baseline: float,
    pooled_top120_spearman_candidate: float,
    all_eligible_ok: bool = True,
    coverage_unchanged: bool = True,
    team_identity_unchanged: bool = True,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Freeze one shadow candidate if it clears the repair gate; else hold."""
    assert_no_forbidden_imports(REPAIR_ENTRYPOINTS)
    before = snapshot_production_artifacts()
    gate = repair_gate(
        fold_mae_deltas=fold_mae_relative_deltas,
        pooled_top120_spearman_baseline=pooled_top120_spearman_baseline,
        pooled_top120_spearman_candidate=pooled_top120_spearman_candidate,
        all_eligible_ok=all_eligible_ok,
        coverage_unchanged=coverage_unchanged,
        team_identity_unchanged=team_identity_unchanged,
    )
    dest = Path(out_dir or SHADOW_OUTPUT_DIR)
    dest.mkdir(parents=True, exist_ok=True)
    board_hash = None
    if board_2026_path is not None and Path(board_2026_path).is_file():
        board_hash = sha256_file(board_2026_path)

    payload = {
        "schema_version": "shadow_v1_rb_wr_candidate_freeze_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_id": candidate_id,
        "code_identity": code_identity,
        "evidence": evidence,
        "source_hashes": source_hashes,
        "board_2026_sha256": board_hash,
        "gate": gate,
        "production_weights_unchanged": True,
        "promotion_authorized": False,
        "note": (
            "Production RB/WR weights remain frozen until untouched 2026 outcomes "
            "run through the unchanged accuracy-first selector."
        ),
    }
    if gate["passed"]:
        path = dest / f"freeze_{candidate_id}.json"
    else:
        path = dest / "hold_v1_structural_role.json"
        payload["verdict"] = "hold_v1_structural_role"
    payload["artifact_hash"] = canonical_json_hash(payload)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    assert_production_unchanged(before)
    return payload
