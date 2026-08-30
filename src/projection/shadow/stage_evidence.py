"""Stage-evidence generation and completeness contract for shadow RB/WR."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.projection.composition import COMPOSE_CHECKPOINT_NAMES
from src.projection.evaluation.accuracy_first import sha256_file
from src.projection.shadow.forbidden import assert_input_path_allowed


class AttributionIncompleteError(RuntimeError):
    """Required compose-stage evidence is missing or fails parity."""


def stage_fold_dir(out_dir: Path, source_season: int, target_season: int) -> Path:
    return Path(out_dir) / "stages" / f"fold_{source_season}_{target_season}"


def stage_evidence_complete(stage_scores: dict[str, Any] | None) -> bool:
    if not stage_scores:
        return False
    return all(name in stage_scores for name in COMPOSE_CHECKPOINT_NAMES)


def assert_traced_points_match_eval(
    player_rates: pd.DataFrame,
    eval_frame: pd.DataFrame,
    *,
    atol: float = 1e-4,
) -> dict[str, Any]:
    """Traced final season points must equal model_points_end_to_end."""
    left = player_rates[["player_id", "traced_v1_pred"]].copy()
    left["player_id"] = left["player_id"].astype(str)
    right = eval_frame[["player_id", "model_points_end_to_end"]].copy()
    right["player_id"] = right["player_id"].astype(str)
    right["model_points_end_to_end"] = pd.to_numeric(
        right["model_points_end_to_end"], errors="coerce"
    )
    merged = left.merge(right, on="player_id", how="inner")
    if merged.empty:
        raise AttributionIncompleteError(
            "No overlapping players between traced stages and fantasy_evaluation"
        )
    delta = (
        pd.to_numeric(merged["traced_v1_pred"], errors="coerce").fillna(0.0)
        - merged["model_points_end_to_end"].fillna(0.0)
    )
    worst = float(np.nanmax(np.abs(delta.to_numpy(dtype=float))))
    if worst > atol:
        raise AttributionIncompleteError(
            f"traced final points != model_points_end_to_end; max |delta|={worst}"
        )
    return {
        "ok": True,
        "n": int(len(merged)),
        "max_abs_delta": worst,
        "atol": atol,
    }


def analyze_finalization_remainder(
    players: pd.DataFrame,
    *,
    material_threshold: float = 5.0,
) -> dict[str, Any]:
    """Explain whether finalization remainder is material after traced rates."""
    top = players[
        players.get("draft_relevant_top120", True)
        & players["position"].isin(("RB", "WR"))
    ].copy()
    if top.empty:
        top = players[players["position"].isin(("RB", "WR"))].copy()
    rem = pd.to_numeric(top.get("finalization_remainder"), errors="coerce")
    mean_rem = float(rem.mean()) if len(rem) else float("nan")
    mean_abs = float(rem.abs().mean()) if len(rem) else float("nan")
    # Identity that should hold after using compose projected_games:
    # traced_v1 ≈ composed_rate_ppg * projected_games + finalization_remainder
    identity_gap = None
    if {
        "traced_v1_pred",
        "composed_rate_ppg",
        "projected_games",
        "finalization_remainder",
    }.issubset(top.columns):
        recon = (
            pd.to_numeric(top["composed_rate_ppg"], errors="coerce").fillna(0.0)
            * pd.to_numeric(top["projected_games"], errors="coerce").fillna(0.0)
            + pd.to_numeric(top["finalization_remainder"], errors="coerce").fillna(0.0)
        )
        identity_gap = float(
            (
                recon - pd.to_numeric(top["traced_v1_pred"], errors="coerce").fillna(0.0)
            ).abs().max()
        )
    material = bool(mean_abs >= material_threshold) if np.isfinite(mean_abs) else False
    return {
        "mean_finalization_remainder": mean_rem,
        "mean_abs_finalization_remainder": mean_abs,
        "material": material,
        "material_threshold": material_threshold,
        "identity_recon_max_abs_gap": identity_gap,
        "explanation": (
            "Finalization remainder is season-total / team-identity drift after "
            "rate stages: pred_season (post reconcile_team_season_identities) "
            "minus composed_rate_ppg × compose projected_games. It is not "
            "Gate-A games mismatch when stages are traced."
            if material
            else "Finalization remainder is below the material threshold."
        ),
    }


def persist_fold_stage_artifacts(
    *,
    out_dir: Path,
    source_season: int,
    target_season: int,
    checkpoint: dict[str, Any],
) -> dict[str, str]:
    """Write precompose/final boards, player rates, and stage score JSON."""
    dest = stage_fold_dir(out_dir, source_season, target_season)
    dest.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}

    pre_path = dest / "precompose.parquet"
    final_path = dest / "final.parquet"
    rates_path = dest / "player_rates.parquet"
    scores_path = dest / "stage_scores.json"
    meta_path = dest / "meta.json"

    checkpoint["precompose"].to_parquet(pre_path, index=False)
    checkpoint["final"].to_parquet(final_path, index=False)
    checkpoint["player_rates"].to_parquet(rates_path, index=False)

    # Serialize stage scores without nested boards.
    serializable = {
        name: checkpoint["stage_scores"][name]
        for name in COMPOSE_CHECKPOINT_NAMES
        if name in checkpoint["stage_scores"]
    }
    scores_path.write_text(json.dumps(serializable), encoding="utf-8")
    meta = {
        "source_season": source_season,
        "target_season": target_season,
        "complete": True,
        "checkpoint_names": list(COMPOSE_CHECKPOINT_NAMES),
        "parity": checkpoint.get("parity"),
        "stage_coverage": checkpoint.get("stage_coverage"),
        "artifact_provenance": checkpoint.get("artifact_provenance"),
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    for path in (pre_path, final_path, rates_path, scores_path, meta_path):
        assert_input_path_allowed(path)
        hashes[path.name] = sha256_file(path)
    return hashes


def load_fold_stage_scores(out_dir: Path, source_season: int, target_season: int) -> dict:
    path = stage_fold_dir(out_dir, source_season, target_season) / "stage_scores.json"
    if not path.is_file():
        raise AttributionIncompleteError(f"Missing stage scores: {path}")
    assert_input_path_allowed(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not stage_evidence_complete(payload):
        raise AttributionIncompleteError(
            f"Incomplete stage scores for {source_season}->{target_season}"
        )
    return payload


def load_fold_player_rates(out_dir: Path, source_season: int, target_season: int) -> pd.DataFrame:
    path = stage_fold_dir(out_dir, source_season, target_season) / "player_rates.parquet"
    if not path.is_file():
        raise AttributionIncompleteError(f"Missing player rates: {path}")
    assert_input_path_allowed(path)
    frame = pd.read_parquet(path)
    frame["player_id"] = frame["player_id"].astype(str)
    return frame
