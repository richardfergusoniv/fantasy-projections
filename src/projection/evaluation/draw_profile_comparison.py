"""Compare overlay metrics across draw profiles when contract identity matches."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.projection.contracts import MODEL_V3_DIR
from src.projection.evaluation.draw_count_rollout import (
    OVERLAY_COMPARISON_IDENTITY_KEYS,
    build_replacement_identity,
    compare_overlay_identities,
    full_overlay_identity,
    manifest_identity_fields,
)
from src.projection.evaluation.evidence_freeze import load_freeze_manifest
from src.projection.release_candidate import rc_namespace_dir

OVERLAY_METRICS = (
    "fantasy_pts_p50",
    "fantasy_pts_p10",
    "fantasy_pts_p90",
    "p_finish_top12",
    "p_finish_top24",
)


def _load_players_overlay(path: Path) -> pd.DataFrame:
    doc = json.loads(path.read_text(encoding="utf-8"))
    players = doc.get("players") or []
    return pd.DataFrame(players)


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_draw_profile_overlays(
    *,
    season: int,
    profiles: dict[str, dict[str, Path]],
    reference_identity: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    """Compare overlay columns when all profiles share contract identity."""
    identities: dict[str, dict[str, str | None]] = {}
    overlays: dict[str, pd.DataFrame] = {}

    manifests: dict[str, dict[str, Any]] = {}
    for label, paths in profiles.items():
        manifests[label] = _load_manifest(paths["manifest"])
        overlays[label] = _load_players_overlay(paths["players"])

    # Shared board/run → one replacement identity (avoids redundant rebuilds).
    shared_board: str | None = None
    shared_run: str | None = None
    shared_model = "accuracy_first_ensemble"
    board_run_aligned = True
    for label, manifest in manifests.items():
        fields = manifest_identity_fields(manifest)
        board = fields.get("selected_board_hash")
        run = fields.get("canonical_projection_run_id")
        model = fields.get("selected_board_model_id") or "accuracy_first_ensemble"
        if shared_board is None:
            shared_board = str(board) if board else None
            shared_run = str(run) if run else None
            shared_model = str(model)
        elif board != shared_board or run != shared_run:
            board_run_aligned = False
            break

    shared_replacement: dict[str, str] | None = None
    if board_run_aligned and shared_board and shared_run:
        shared_replacement = build_replacement_identity(
            season=season,
            selected_board_hash=shared_board,
            canonical_projection_run_id=shared_run,
            selected_board_model_id=shared_model,
        )

    for label, manifest in manifests.items():
        identity = full_overlay_identity(manifest, season=season)
        if shared_replacement is not None:
            identity.update(shared_replacement)
        identities[label] = identity

    ref_label = next(iter(identities))
    ref_identity = reference_identity or identities[ref_label]
    mismatches_by_profile: dict[str, list[str]] = {}
    for label, identity in identities.items():
        mismatches_by_profile[label] = compare_overlay_identities(identity, ref_identity)

    identity_ok = all(not mismatches for mismatches in mismatches_by_profile.values())
    if not identity_ok:
        return {
            "comparison_verdict": "hold",
            "reason": "board_or_contract_identity_mismatch",
            "identity_keys_checked": list(OVERLAY_COMPARISON_IDENTITY_KEYS),
            "mismatches_by_profile": mismatches_by_profile,
        }

    merged = None
    metric_deltas: dict[str, Any] = {}
    for label, frame in overlays.items():
        cols = ["player_id", *[c for c in OVERLAY_METRICS if c in frame.columns]]
        slim = frame[cols].copy()
        slim["player_id"] = slim["player_id"].astype(str)
        renamed = slim.rename(columns={c: f"{label}:{c}" for c in cols if c != "player_id"})
        merged = renamed if merged is None else merged.merge(renamed, on="player_id", how="outer")

    if merged is not None:
        for metric in OVERLAY_METRICS:
            profile_cols = [c for c in merged.columns if c.endswith(f":{metric}")]
            if len(profile_cols) < 2:
                continue
            deltas = merged[profile_cols].max(axis=1) - merged[profile_cols].min(axis=1)
            metric_deltas[metric] = {
                "median_abs_delta": float(pd.to_numeric(deltas, errors="coerce").median()),
                "p95_abs_delta": float(pd.to_numeric(deltas, errors="coerce").quantile(0.95)),
            }

    return {
        "comparison_verdict": "compare",
        "reason": None,
        "identity_keys_checked": list(OVERLAY_COMPARISON_IDENTITY_KEYS),
        "player_count": int(merged["player_id"].nunique()) if merged is not None else 0,
        "metric_deltas": metric_deltas,
        "mismatches_by_profile": mismatches_by_profile,
    }


def default_profile_paths(
    *,
    season: int,
    rc_namespace: str,
    freeze_id: str,
) -> dict[str, dict[str, Path]]:
    rc_dir = rc_namespace_dir(season, rc_namespace)
    production_players = Path("draft_assistant/data") / f"players_{season}.json"
    return {
        "production_1k": {
            "manifest": Path(MODEL_V3_DIR) / f"simulation_manifest_{season}.json",
            "players": production_players,
        },
        f"rc_{rc_namespace}": {
            "manifest": rc_dir / f"simulation_manifest_{season}.json",
            "players": rc_dir / f"players_{season}_rc.json",
        },
    }
