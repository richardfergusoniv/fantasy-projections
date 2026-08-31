"""Bounded audit of traced finalization / depth-ladder / corrections paths.

Returns a sealed finding for the shadow v1 RB/WR repair track: either a
specific cutoff-available defect to repair, or ``no_cutoff_available_defect``
so the track can close with v1 retained as structural/diagnostic only.
"""
from __future__ import annotations

import ast
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.projection.contracts import CORRECTIONS_PATH, REPO_ROOT
from src.projection.evaluation.accuracy_first import canonical_json_hash, sha256_file
from src.projection.fantasy_evaluation import _score_long_board_points
from src.projection.shadow.contracts import (
    CLOSEOUT_POINTER_NAME,
    CLOSEOUT_POINTER_SCHEMA,
    CLOSEOUT_SCHEMA,
    SHADOW_V1_RB_WR_DIR,
    SHADOW_V1_REPAIR_TRACK_CLOSED,
)
from src.projection.shadow.forbidden import (
    ForbiddenImportGuard,
    assert_input_path_allowed,
    assert_no_forbidden_imports,
)
from src.projection.shadow.production_guard import (
    assert_production_unchanged,
    snapshot_production_artifacts,
)

SHADOW_DIR = SHADOW_V1_RB_WR_DIR
AUDIT_SCHEMA = "shadow_v1_rb_wr_finalization_audit_v1"

ENTRYPOINTS = (
    "src.projection.shadow.finalization_audit",
    "src.projection.shadow.forbidden",
    "src.projection.shadow.production_guard",
)

# Live paths that must not reintroduce the retired depth-rate ladder.
LADDER_GUARD_MODULES = (
    "src/projection/composition.py",
    "src/projection/depth_gating.py",
    "src/projection/fantasy_evaluation.py",
    "src/projection/veterans.py",
    "src/projection/predict.py",
    "src/projection/backtest.py",
)


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(REPO_ROOT),
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def _module_calls_name(path: Path, name: str) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == name:
            return True
        if isinstance(node, ast.Attribute) and node.attr == name:
            return True
    return False


def audit_depth_ladder() -> dict[str, Any]:
    """Confirm the post-hoc depth-rate ladder is retired on live paths."""
    calls = {}
    for rel in LADDER_GUARD_MODULES:
        path = Path(REPO_ROOT) / rel
        calls[rel] = {
            "apply_depth_rate_ladder": _module_calls_name(path, "apply_depth_rate_ladder"),
            "depth_rate_factors": _module_calls_name(path, "depth_rate_factors"),
        }
    any_live = any(
        v["apply_depth_rate_ladder"] or v["depth_rate_factors"] for v in calls.values()
    )
    # depth_rates.py may still define the helper; that is not a live call site.
    return {
        "retired_per_decision_doc": True,
        "live_call_sites": calls,
        "live_ladder_application_detected": bool(any_live),
        "defect": (
            "depth_ladder_still_applied_on_live_path"
            if any_live
            else None
        ),
        "note": (
            "role_discount_factor is an audit constant (1.0); depth reaches "
            "rates via ROLE_FEATURES / depth_tier, not a post-hoc multiplier."
        ),
    }


def audit_corrections_joblib(shadow_dir: Path) -> dict[str, Any]:
    """TE-only elite correction; must be absent from leakage-safe traced boards."""
    corr_path = Path(CORRECTIONS_PATH)
    payload: dict[str, Any] = {
        "path": str(corr_path).replace("\\", "/"),
        "exists": corr_path.is_file(),
        "sha256": sha256_file(corr_path) if corr_path.is_file() else None,
        "positions": [],
        "rb_wr_params_present": False,
        "traced_boards_nonzero_elite_correction": False,
        "eval_path_omits_corrections": True,
        "defect": None,
    }
    if corr_path.is_file():
        assert_input_path_allowed(corr_path)
        params = joblib.load(corr_path)
        if isinstance(params, dict):
            payload["positions"] = sorted(params.keys())
            payload["rb_wr_params_present"] = any(
                pos in params for pos in ("RB", "WR")
            )

    # Eval harness must pass corrections=None (leakage-safe).
    fe_src = (Path(REPO_ROOT) / "src/projection/fantasy_evaluation.py").read_text(
        encoding="utf-8"
    )
    payload["eval_path_omits_corrections"] = (
        "corrections=None" in fe_src
        and "spanning the target season" in fe_src
    )

    nonzero = 0
    rows = 0
    for fold_dir in sorted((shadow_dir / "stages").glob("fold_*")):
        final_path = fold_dir / "final.parquet"
        if not final_path.is_file():
            continue
        assert_input_path_allowed(final_path)
        final = pd.read_parquet(final_path)
        if "elite_correction_pg" not in final.columns:
            continue
        vals = pd.to_numeric(final["elite_correction_pg"], errors="coerce").fillna(0.0)
        nonzero += int((vals.abs() > 1e-12).sum())
        rows += int(len(vals))
    payload["traced_elite_correction_nonzero_rows"] = nonzero
    payload["traced_elite_correction_rows_scanned"] = rows
    payload["traced_boards_nonzero_elite_correction"] = nonzero > 0

    defects = []
    if payload["rb_wr_params_present"]:
        defects.append("corrections_joblib_has_rb_wr_params")
    if payload["traced_boards_nonzero_elite_correction"]:
        defects.append("leakage_safe_boards_carry_elite_correction")
    if not payload["eval_path_omits_corrections"]:
        defects.append("eval_path_does_not_omit_corrections")
    payload["defect"] = defects[0] if defects else None
    payload["defects"] = defects
    payload["note"] = (
        "corrections.joblib is TE-only elite shrinkage on the ship path; "
        "leakage-safe folds omit it by design (fit spans the target season)."
    )
    return payload


def audit_finalization_mechanics(shadow_dir: Path) -> dict[str, Any]:
    """Show finalization_remainder equals team-identity season scaling only."""
    players_path = shadow_dir / "attribution_players.parquet"
    assert_input_path_allowed(players_path)
    players = pd.read_parquet(players_path)
    top = players[
        players["draft_relevant_top120"] & players["position"].isin(("RB", "WR"))
    ].copy()

    fold_reports = []
    max_recon_gap = 0.0
    for fold, grp in top.groupby("fold", sort=True):
        source, target = str(fold).split("->")
        final_path = (
            shadow_dir
            / "stages"
            / f"fold_{source.strip()}_{target.strip()}"
            / "final.parquet"
        )
        assert_input_path_allowed(final_path)
        final = pd.read_parquet(final_path)
        if "role_discount_factor" in final.columns:
            factors = pd.to_numeric(final["role_discount_factor"], errors="coerce")
            ladder_live = bool(((factors - 1.0).abs() > 1e-12).any())
        else:
            ladder_live = False

        rxg = (
            pd.to_numeric(final["pred_pg"], errors="coerce")
            * pd.to_numeric(final["projected_games"], errors="coerce")
        )
        pre = _score_long_board_points(
            final.assign(pred_season=rxg), value_col="pred_season"
        )
        post = _score_long_board_points(final, value_col="pred_season")
        identity_delta = (post - pre).rename("identity_delta")

        ids = grp["player_id"].astype(str)
        fin = (
            grp.set_index(grp["player_id"].astype(str))["finalization_remainder"]
            .reindex(ids)
            .astype(float)
        )
        delta = identity_delta.reindex(ids).astype(float)
        paired = pd.concat([fin, delta], axis=1).dropna()
        gap = float((paired.iloc[:, 0] - paired.iloc[:, 1]).abs().max()) if len(paired) else 0.0
        max_recon_gap = max(max_recon_gap, gap)

        # Stage fantasy_ppg deltas at finalization should be ~0 (rates untouched).
        stage_path = shadow_dir / "stage_attribution.csv"
        stage_zero = None
        if stage_path.is_file():
            stage = pd.read_csv(stage_path)
            cell = stage[
                stage["stage"].eq("season_total_finalization")
                & stage["draft_relevant_top120"]
                & stage["target_season"].eq(int(target.strip()))
                & stage["position"].isin(("RB", "WR"))
            ]
            if not cell.empty:
                stage_zero = bool(
                    np.allclose(
                        pd.to_numeric(cell["delta_from_prior"], errors="coerce")
                        .fillna(0.0)
                        .to_numpy(),
                        0.0,
                        atol=1e-9,
                    )
                )

        fold_reports.append({
            "fold": str(fold),
            "n_top120": int(len(paired)),
            "mean_abs_finalization": float(paired.iloc[:, 0].abs().mean())
            if len(paired)
            else float("nan"),
            "identity_recon_max_abs_gap": gap,
            "corr_finalization_vs_identity_delta": float(
                paired.iloc[:, 0].corr(paired.iloc[:, 1])
            )
            if len(paired) > 2
            else float("nan"),
            "role_discount_non_identity": ladder_live,
            "stage_ppg_delta_at_finalization_is_zero": stage_zero,
        })

    rates_untouched = all(
        r.get("stage_ppg_delta_at_finalization_is_zero") in (True, None)
        for r in fold_reports
    )
    identity_explained = max_recon_gap < 1e-6
    ladder_on_boards = any(r["role_discount_non_identity"] for r in fold_reports)

    defects = []
    if not identity_explained:
        defects.append("finalization_not_explained_by_team_identity_scaling")
    if ladder_on_boards:
        defects.append("traced_boards_apply_nontrivial_role_discount")
    if any(r.get("stage_ppg_delta_at_finalization_is_zero") is False for r in fold_reports):
        defects.append("finalization_moved_pred_pg_rates")

    return {
        "folds": fold_reports,
        "identity_recon_max_abs_gap": max_recon_gap,
        "explained_by_reconcile_team_season_identities": identity_explained,
        "rates_untouched_at_finalization": rates_untouched,
        "defect": defects[0] if defects else None,
        "defects": defects,
        "note": (
            "Material finalization_remainder is expected: "
            "reconcile_team_season_identities scales pred_season only to "
            "restore pass/catch identities. It is not a Gate-A mismatch and "
            "is not a cutoff-available ranking defect for RB/WR."
        ),
    }


def run_finalization_audit(
    *,
    out_dir: str | Path | None = None,
    close_track: bool = True,
) -> dict[str, Any]:
    """Audit ladder/corrections/finalization; optionally seal repair-track closeout."""
    assert_no_forbidden_imports(ENTRYPOINTS)
    before = snapshot_production_artifacts()
    shadow_dir = Path(out_dir or SHADOW_DIR)
    dest = shadow_dir / "finalization_audit"
    dest.mkdir(parents=True, exist_ok=True)

    with ForbiddenImportGuard():
        ladder = audit_depth_ladder()
        corrections = audit_corrections_joblib(shadow_dir)
        finalization = audit_finalization_mechanics(shadow_dir)

        defects = [
            d
            for d in (
                ladder.get("defect"),
                corrections.get("defect"),
                finalization.get("defect"),
            )
            if d
        ]
        finding = (
            "no_cutoff_available_defect"
            if not defects
            else "cutoff_available_defect_found"
        )

        input_hashes = {}
        for rel in (
            "manifest.json",
            "attribution_players.parquet",
            "attribution_summary.json",
            "stage_attribution.csv",
            "step6_decision.json",
        ):
            path = shadow_dir / rel
            if path.is_file():
                assert_input_path_allowed(path)
                input_hashes[rel] = sha256_file(path)
        if Path(CORRECTIONS_PATH).is_file():
            input_hashes["models/corrections.joblib"] = sha256_file(CORRECTIONS_PATH)

        audit = {
            "schema_version": AUDIT_SCHEMA,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "producing_commit": _git_commit(),
            "finding": finding,
            "defects": defects,
            "depth_ladder": ladder,
            "corrections_joblib": corrections,
            "finalization": finalization,
            "input_hashes": input_hashes,
            "production_weights_unchanged": True,
            "promotion_authorized": False,
        }
        audit["artifact_hash"] = canonical_json_hash(audit)
        audit_path = dest / "audit.json"
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

        closeout = None
        if close_track and finding == "no_cutoff_available_defect":
            closeout = {
                "schema_version": CLOSEOUT_SCHEMA,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "producing_commit": _git_commit(),
                "verdict": "close_shadow_repair_track",
                "v1_role": "structural_diagnostic_only",
                "approved_primary_top120_rb_wr": False,
                "repair_track_status": "closed",
                "reason": (
                    "Bounded audit of traced stage/finalization mechanics, "
                    "retired depth-ladder path, and corrections.joblib found "
                    "no specific cutoff-available defect to repair. Oracle "
                    "directions (availability_only / raw_rate_only) are not "
                    "implementable without outcome leakage or rate retunes "
                    "outside this track's scope. v1 retains roster/depth/"
                    "availability/team-identity/reconciliation and QB/TE "
                    "ensemble signal; it is not an approved primary top-120 "
                    "RB/WR ranking signal."
                ),
                "audit_sha256": sha256_file(audit_path),
                "prior_holds": {
                    "availability_gate_a_blend": "hold_v1_structural_role",
                    "step6_oracle_directions": "not_eligible_for_freeze",
                },
                "production_weights_unchanged": True,
                "promotion_authorized": False,
                "further_repair_authorized": False,
            }
            closeout["artifact_hash"] = canonical_json_hash(closeout)
            # Canonical policy artifact — freeze/repair reads only this path.
            close_path = Path(SHADOW_V1_REPAIR_TRACK_CLOSED)
            if shadow_dir.resolve() != SHADOW_V1_RB_WR_DIR.resolve():
                close_path = shadow_dir / "repair_track_closed.json"
            close_path.parent.mkdir(parents=True, exist_ok=True)
            close_path.write_text(json.dumps(closeout, indent=2), encoding="utf-8")
            # Audit dir stores a pointer only (not a second policy source).
            pointer = {
                "schema_version": CLOSEOUT_POINTER_SCHEMA,
                "closeout_path": str(close_path).replace("\\", "/"),
                "closeout_sha256": sha256_file(close_path),
                "closeout_artifact_hash": closeout["artifact_hash"],
                "further_repair_authorized": False,
            }
            pointer["artifact_hash"] = canonical_json_hash(pointer)
            (dest / CLOSEOUT_POINTER_NAME).write_text(
                json.dumps(pointer, indent=2), encoding="utf-8"
            )

        assert_production_unchanged(before)
        return {
            "finding": finding,
            "defects": defects,
            "audit_path": str(audit_path).replace("\\", "/"),
            "closeout": closeout,
            "production_weights_unchanged": True,
        }
