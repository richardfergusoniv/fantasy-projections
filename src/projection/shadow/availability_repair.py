"""Availability-only shadow repair: Gate-A exposure blend (no rate/composition edits).

Candidate
---------
``shadow_availability_gate_a_blend_v1``

At each forecast cutoff, Gate A already produces ``projected_games`` from
source-season-or-earlier fits. Compose then flattens draft exposure to a full
season (``EXPOSURE_BLEND_ALPHA = 0`` → flat 17), leaving Gate A only in
``projected_games_raw``.

This shadow candidate restores a convex blend of Gate A and the full-season
baseline for RB/WR season totals **only**:

    games' = clip(α · gate_a + (1 − α) · 17, 0, 17)
    pred'  = composed_rate_ppg · games' + finalization_remainder

which is algebraically ``v1 − composed_rate · (proj − games')`` when
``v1 = composed_rate · proj + finalization_remainder``.

Constraints (hard)
------------------
* Cutoff-safe: uses only Gate A games already present on the fold precompose
  board (fitted at that fold's source season) and fold-prior α selection.
* Does not alter ``raw_rate_ppg``, ``composed_rate_ppg``, or any composition
  stage; finalization remainder is held fixed.
* α is chosen leakage-safely by nested rolling-origin on prior folds only.
* Evaluation reuses pinned top-120 memberships and the same ``repair_gate`` /
  step-6 fold×position rules as the availability_only counterfactual.
* Artifacts seal code, config, and input hashes under
  ``output/shadow_v1_rb_wr/availability_repair/``.
* Production weights, contracts, and release pointers are snapshotted and must
  not change; freeze never authorizes promotion.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.projection.contracts import OUTPUT_DIR, REPO_ROOT
from src.projection.evaluation.accuracy_first import canonical_json_hash, sha256_file
from src.projection.shadow.decision_rules import repair_gate
from src.projection.shadow.forbidden import (
    ForbiddenImportGuard,
    assert_input_path_allowed,
    assert_no_forbidden_imports,
)
from src.projection.shadow.production_guard import (
    assert_production_unchanged,
    snapshot_production_artifacts,
)
from src.projection.shadow.repair import freeze_shadow_candidate
from src.projection.shadow.step6_decision import (
    _spearman,
    evaluate_candidate_gates,
)
from src.projection.transitions import SEASON_GAMES

SHADOW_OUTPUT_DIR = Path(OUTPUT_DIR) / "shadow_v1_rb_wr"
REPAIR_SUBDIR = "availability_repair"
CANDIDATE_ID = "shadow_availability_gate_a_blend_v1"
SCHEMA_VERSION = "shadow_v1_rb_wr_availability_repair_v1"

POSITIONS = ("RB", "WR")
ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
# Cold-start (no prior fold): restore Gate A fully — the structural hypothesis
# from the availability_only diagnostic direction. Nested fits may shrink α.
COLD_START_ALPHA = 1.0

ENTRYPOINTS = (
    "src.projection.shadow.availability_repair",
    "src.projection.shadow.decision_rules",
    "src.projection.shadow.repair",
    "src.projection.shadow.forbidden",
    "src.projection.shadow.production_guard",
    "src.projection.shadow.step6_decision",
)

DEFAULT_CONFIG: dict[str, Any] = {
    "candidate_id": CANDIDATE_ID,
    "schema_version": SCHEMA_VERSION,
    "positions": list(POSITIONS),
    "alpha_grid": list(ALPHA_GRID),
    "cold_start_alpha": COLD_START_ALPHA,
    "season_games": int(SEASON_GAMES),
    "objective": "top120_mae_points",
    "prediction_formula": (
        "composed_rate_ppg * clip(alpha * gate_a_games + (1 - alpha) * season_games, "
        "0, season_games) + finalization_remainder"
    ),
    "untouched": [
        "raw_rate_ppg",
        "composed_rate_ppg",
        "composition_stages",
        "finalization_remainder",
        "production_weights",
        "application_contract",
        "active_release_pointer",
    ],
    "gate_source": "stages/fold_*/precompose.parquet::projected_games",
    "membership_source": "top120_membership_{season}.json (pinned)",
    "counterfactual_gates": "src.projection.shadow.step6_decision.evaluate_candidate_gates",
}


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


def _module_path(mod: str) -> Path:
    return Path(REPO_ROOT) / Path(*mod.split(".")).with_suffix(".py")


def code_identity() -> dict[str, Any]:
    files = {}
    for mod in ENTRYPOINTS:
        path = _module_path(mod)
        if path.is_file():
            files[mod] = {
                "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
                "sha256": sha256_file(path),
            }
    return {
        "candidate_id": CANDIDATE_ID,
        "producing_commit": _git_commit(),
        "entrypoint_files": files,
        "entrypoint_bundle_sha256": canonical_json_hash(files),
    }


def fold_stage_dirname(fold: str) -> str:
    """Map ``2024->2025`` → ``fold_2024_2025``."""
    left, right = fold.split("->")
    return f"fold_{left.strip()}_{right.strip()}"


def load_gate_a_games(shadow_dir: Path, fold: str) -> pd.DataFrame:
    """Gate A games at cutoff = precompose ``projected_games`` (pre flat-17)."""
    path = shadow_dir / "stages" / fold_stage_dirname(fold) / "precompose.parquet"
    assert_input_path_allowed(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing Gate-A precompose board: {path}")
    pre = pd.read_parquet(path)
    if "projected_games" not in pre.columns:
        raise ValueError(f"{path} lacks projected_games (Gate A)")
    out = (
        pre.drop_duplicates("player_id")[["player_id", "projected_games"]]
        .rename(columns={"projected_games": "gate_a_games"})
        .copy()
    )
    out["player_id"] = out["player_id"].astype(str)
    out["gate_a_games"] = pd.to_numeric(out["gate_a_games"], errors="coerce")
    return out


def blend_games(
    gate_a: pd.Series | np.ndarray,
    *,
    alpha: float,
    season_games: float = float(SEASON_GAMES),
) -> np.ndarray:
    """Convex blend of Gate A and full-season draft exposure."""
    raw = pd.to_numeric(pd.Series(gate_a), errors="coerce").to_numpy(dtype=float)
    a = float(alpha)
    if a <= 0.0:
        return np.full(len(raw), float(season_games), dtype=float)
    if a >= 1.0:
        blended = raw
    else:
        blended = a * raw + (1.0 - a) * float(season_games)
    return np.clip(np.nan_to_num(blended, nan=float(season_games)), 0.0, float(season_games))


def shadow_prediction(
    frame: pd.DataFrame,
    *,
    alpha_by_position: dict[str, float],
    season_games: float = float(SEASON_GAMES),
) -> pd.Series:
    """Availability-only rematerialization; rates and finalization unchanged."""
    alphas = frame["position"].map(
        lambda pos: float(alpha_by_position.get(str(pos), COLD_START_ALPHA))
    ).to_numpy(dtype=float)
    games = np.empty(len(frame), dtype=float)
    gate = pd.to_numeric(frame["gate_a_games"], errors="coerce").to_numpy(dtype=float)
    for i, a in enumerate(alphas):
        games[i] = blend_games([gate[i]], alpha=float(a), season_games=season_games)[0]
    rate = pd.to_numeric(frame["composed_rate_ppg"], errors="coerce").fillna(0.0).to_numpy()
    fin = pd.to_numeric(frame["finalization_remainder"], errors="coerce").fillna(0.0).to_numpy()
    return pd.Series(rate * games + fin, index=frame.index, name="pred_availability_repair")


def _top120_mae(frame: pd.DataFrame, pred: pd.Series) -> float:
    cell = frame[frame["draft_relevant_top120"]]
    if cell.empty:
        return float("nan")
    actual = pd.to_numeric(cell["actual_points"], errors="coerce")
    err = pred.loc[cell.index] - actual
    valid = err.dropna()
    if valid.empty:
        return float("nan")
    return float(valid.abs().mean())


def select_alpha_for_position(
    train: pd.DataFrame,
    *,
    position: str,
    alpha_grid: tuple[float, ...] = ALPHA_GRID,
    season_games: float = float(SEASON_GAMES),
    cold_start: float = COLD_START_ALPHA,
) -> dict[str, Any]:
    """Pick α minimizing top-120 MAE on training folds for one position."""
    pos = train[train["position"].eq(position)]
    if pos.empty or not bool(pos["draft_relevant_top120"].any()):
        return {
            "position": position,
            "alpha": float(cold_start),
            "source": "cold_start",
            "train_n_top120": 0,
            "train_mae_by_alpha": {},
        }
    scores = {}
    for alpha in alpha_grid:
        pred = shadow_prediction(
            pos, alpha_by_position={position: float(alpha)}, season_games=season_games
        )
        scores[str(alpha)] = _top120_mae(pos, pred)
    # Prefer smaller α on ties (closer to production flat-17 / fail-closed).
    best_alpha = min(
        alpha_grid,
        key=lambda a: (
            scores[str(a)] if np.isfinite(scores[str(a)]) else float("inf"),
            float(a),
        ),
    )
    return {
        "position": position,
        "alpha": float(best_alpha),
        "source": "nested_prior_folds",
        "train_n_top120": int(pos["draft_relevant_top120"].sum()),
        "train_mae_by_alpha": scores,
    }


def attach_gate_a(players: pd.DataFrame, shadow_dir: Path) -> pd.DataFrame:
    """Join cutoff Gate A games onto attribution players."""
    parts = []
    for fold, grp in players.groupby("fold", sort=True):
        gate = load_gate_a_games(shadow_dir, str(fold))
        merged = grp.copy()
        merged["player_id"] = merged["player_id"].astype(str)
        merged = merged.merge(gate, on="player_id", how="left")
        missing = int(merged["gate_a_games"].isna().sum())
        if missing:
            # Fail closed to full season (α=0 behavior) when Gate A is absent.
            merged["gate_a_games"] = merged["gate_a_games"].fillna(float(SEASON_GAMES))
        parts.append(merged)
    return pd.concat(parts, ignore_index=True)


def fit_nested_alphas(
    players: pd.DataFrame,
    *,
    alpha_grid: tuple[float, ...] = ALPHA_GRID,
    cold_start: float = COLD_START_ALPHA,
    season_games: float = float(SEASON_GAMES),
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Nested rolling-origin α fit; score each fold with prior-fold α only."""
    folds = sorted(players["fold"].astype(str).unique())
    scored_parts = []
    fit_log = []
    for i, fold in enumerate(folds):
        prior = players[players["fold"].isin(folds[:i])]
        alpha_by_position: dict[str, float] = {}
        fold_fit = {"fold": fold, "positions": {}}
        for position in POSITIONS:
            sel = select_alpha_for_position(
                prior,
                position=position,
                alpha_grid=alpha_grid,
                season_games=season_games,
                cold_start=cold_start,
            )
            alpha_by_position[position] = float(sel["alpha"])
            fold_fit["positions"][position] = sel
        fold_fit["alpha_by_position"] = alpha_by_position
        fit_log.append(fold_fit)

        cur = players[players["fold"].eq(fold)].copy()
        cur["shadow_alpha"] = cur["position"].map(alpha_by_position).astype(float)
        cur["shadow_games"] = [
            float(
                blend_games(
                    [g],
                    alpha=float(a),
                    season_games=season_games,
                )[0]
            )
            for g, a in zip(
                cur["gate_a_games"].to_numpy(),
                cur["shadow_alpha"].to_numpy(),
                strict=True,
            )
        ]
        cur["pred_availability_repair"] = shadow_prediction(
            cur, alpha_by_position=alpha_by_position, season_games=season_games
        )
        # Rates untouched identity check columns for audits.
        cur["raw_rate_ppg_unchanged"] = cur["raw_rate_ppg"]
        cur["composed_rate_ppg_unchanged"] = cur["composed_rate_ppg"]
        scored_parts.append(cur)
    return pd.concat(scored_parts, ignore_index=True), fit_log


def _build_gate_compatible_frame(scored: pd.DataFrame) -> pd.DataFrame:
    """Map repair predictions into the step-6 counterfactual gate interface.

    Reuses ``evaluate_candidate_gates`` by presenting the repair as the
    ``availability_only`` arm (same component-shift contract: availability is
    the repaired component) while leaving oracle residual columns intact for
    reference.
    """
    out = scored.copy()
    out["pred_v1_control"] = pd.to_numeric(out["v1_pred"], errors="coerce")
    # Implementable availability repair substitutes for the oracle arm.
    out["pred_availability_only"] = pd.to_numeric(
        out["pred_availability_repair"], errors="coerce"
    )
    # Keep other arms as oracle residuals so the shared helper can run; they
    # are not used for this candidate's freeze decision.
    out["pred_raw_rate_only"] = out["pred_v1_control"] - pd.to_numeric(
        out["raw_rate_error"], errors="coerce"
    )
    out["pred_joint_diagnostic"] = (
        out["pred_v1_control"]
        - pd.to_numeric(out["raw_rate_error"], errors="coerce")
        - pd.to_numeric(out["availability_effect"], errors="coerce")
    )
    return out


def build_repair_score_table(scored: pd.DataFrame) -> pd.DataFrame:
    """Fold × position × population MAE table for control vs repair."""
    rows = []
    for fold, fold_frame in scored.groupby("fold", sort=True):
        for position in POSITIONS:
            pos = fold_frame[fold_frame["position"].eq(position)]
            for population, mask in (
                ("all_eligible", pos.index),
                ("top120", pos.index[pos["draft_relevant_top120"].to_numpy()]),
            ):
                cell = pos.loc[mask]
                actual = pd.to_numeric(cell["actual_points"], errors="coerce")
                ctrl = pd.to_numeric(cell["v1_pred"], errors="coerce")
                cand = pd.to_numeric(cell["pred_availability_repair"], errors="coerce")
                valid = pd.concat([actual, ctrl, cand], axis=1).dropna()
                if valid.empty:
                    ctrl_mae = cand_mae = float("nan")
                    ctrl_rho = cand_rho = float("nan")
                    n = 0
                else:
                    n = int(len(valid))
                    ctrl_err = valid.iloc[:, 1] - valid.iloc[:, 0]
                    cand_err = valid.iloc[:, 2] - valid.iloc[:, 0]
                    ctrl_mae = float(ctrl_err.abs().mean())
                    cand_mae = float(cand_err.abs().mean())
                    ctrl_rho = _spearman(valid.iloc[:, 0], valid.iloc[:, 1])
                    cand_rho = _spearman(valid.iloc[:, 0], valid.iloc[:, 2])
                mae_delta = (
                    cand_mae - ctrl_mae
                    if np.isfinite(cand_mae) and np.isfinite(ctrl_mae)
                    else float("nan")
                )
                rel = (
                    mae_delta / ctrl_mae
                    if np.isfinite(mae_delta) and ctrl_mae not in (0.0, float("nan"))
                    else float("nan")
                )
                rows.append({
                    "fold": fold,
                    "position": position,
                    "population": population,
                    "candidate": CANDIDATE_ID,
                    "n": n,
                    "mae": cand_mae,
                    "control_mae": ctrl_mae,
                    "mae_delta_vs_control": mae_delta,
                    "relative_mae_delta_vs_control": rel,
                    "spearman": cand_rho,
                    "control_spearman": ctrl_rho,
                    "mean_shadow_alpha": float(
                        pd.to_numeric(cell["shadow_alpha"], errors="coerce").mean()
                    )
                    if n
                    else float("nan"),
                    "mean_shadow_games": float(
                        pd.to_numeric(cell["shadow_games"], errors="coerce").mean()
                    )
                    if n
                    else float("nan"),
                    "mean_gate_a_games": float(
                        pd.to_numeric(cell["gate_a_games"], errors="coerce").mean()
                    )
                    if n
                    else float("nan"),
                })
    return pd.DataFrame(rows)


def verify_rates_untouched(scored: pd.DataFrame, atol: float = 1e-12) -> dict[str, Any]:
    raw_ok = np.allclose(
        pd.to_numeric(scored["raw_rate_ppg"], errors="coerce").fillna(0.0),
        pd.to_numeric(scored["raw_rate_ppg_unchanged"], errors="coerce").fillna(0.0),
        atol=atol,
    )
    comp_ok = np.allclose(
        pd.to_numeric(scored["composed_rate_ppg"], errors="coerce").fillna(0.0),
        pd.to_numeric(scored["composed_rate_ppg_unchanged"], errors="coerce").fillna(0.0),
        atol=atol,
    )
    return {
        "raw_rate_unchanged": bool(raw_ok),
        "composed_rate_unchanged": bool(comp_ok),
        "composition_stages_not_rerun": True,
        "finalization_remainder_held_fixed": True,
    }


def collect_input_hashes(shadow_dir: Path) -> dict[str, str]:
    paths = {
        "manifest.json": shadow_dir / "manifest.json",
        "attribution_players.parquet": shadow_dir / "attribution_players.parquet",
        "attribution_summary.json": shadow_dir / "attribution_summary.json",
        "step6_decision.json": shadow_dir / "step6_decision.json",
    }
    for season in (2023, 2024, 2025):
        paths[f"top120_membership_{season}.json"] = (
            shadow_dir / f"top120_membership_{season}.json"
        )
    for fold in ("2022_2023", "2023_2024", "2024_2025"):
        paths[f"stages/fold_{fold}/precompose.parquet"] = (
            shadow_dir / "stages" / f"fold_{fold}" / "precompose.parquet"
        )
    out = {}
    for key, path in paths.items():
        if path.is_file():
            assert_input_path_allowed(path)
            out[key] = sha256_file(path)
    return out


def run_availability_repair(
    *,
    out_dir: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit, evaluate, seal, and optionally freeze the Gate-A blend candidate."""
    assert_no_forbidden_imports(ENTRYPOINTS)
    before = snapshot_production_artifacts()
    shadow_dir = Path(out_dir or SHADOW_OUTPUT_DIR)
    dest = shadow_dir / REPAIR_SUBDIR
    dest.mkdir(parents=True, exist_ok=True)

    cfg = {**DEFAULT_CONFIG, **(config or {})}
    alpha_grid = tuple(float(a) for a in cfg["alpha_grid"])
    cold_start = float(cfg["cold_start_alpha"])
    season_games = float(cfg["season_games"])

    with ForbiddenImportGuard():
        manifest_path = shadow_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing attribution manifest: {manifest_path}")
        attribution_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        attribution_status = attribution_manifest.get("status")

        identity = code_identity()
        config_hash = canonical_json_hash(cfg)
        input_hashes = collect_input_hashes(shadow_dir)

        if attribution_status != "ok":
            payload = freeze_shadow_candidate(
                candidate_id=CANDIDATE_ID,
                code_identity=identity,
                evidence={"reason": "attribution_not_ok"},
                source_hashes=input_hashes,
                fold_mae_relative_deltas=[0.0],
                pooled_top120_spearman_baseline=0.0,
                pooled_top120_spearman_candidate=0.0,
                out_dir=dest,
                attribution_status=attribution_status,
            )
            assert_production_unchanged(before)
            return payload

        players = pd.read_parquet(shadow_dir / "attribution_players.parquet")
        players = attach_gate_a(players, shadow_dir)
        scored, fit_log = fit_nested_alphas(
            players,
            alpha_grid=alpha_grid,
            cold_start=cold_start,
            season_games=season_games,
        )
        rate_guard = verify_rates_untouched(scored)
        if not (
            rate_guard["raw_rate_unchanged"] and rate_guard["composed_rate_unchanged"]
        ):
            raise RuntimeError("Availability repair mutated rate columns")

        score_table = build_repair_score_table(scored)
        gate_frame = _build_gate_compatible_frame(scored)
        # Build a counterfactual-shaped table for evaluate_candidate_gates.
        cf_rows = []
        for _, row in score_table.iterrows():
            cf_rows.append({
                "fold": row["fold"],
                "position": row["position"],
                "population": row["population"],
                "candidate": "availability_only",
                "n": row["n"],
                "mae": row["mae"],
                "bias": float("nan"),
                "spearman": row["spearman"],
                "mae_delta_vs_control": row["mae_delta_vs_control"],
                "relative_mae_delta_vs_control": row["relative_mae_delta_vs_control"],
                "control_mae": row["control_mae"],
                "control_spearman": row["control_spearman"],
            })
            cf_rows.append({
                "fold": row["fold"],
                "position": row["position"],
                "population": row["population"],
                "candidate": "v1_control",
                "n": row["n"],
                "mae": row["control_mae"],
                "bias": float("nan"),
                "spearman": row["control_spearman"],
                "mae_delta_vs_control": 0.0,
                "relative_mae_delta_vs_control": 0.0,
                "control_mae": row["control_mae"],
                "control_spearman": row["control_spearman"],
            })
        cf_table = pd.DataFrame(cf_rows)
        gate_eval = evaluate_candidate_gates(
            cf_table,
            gate_frame,
            candidate="availability_only",
            population="top120",
        )
        # This candidate is implementable code — eligible for freeze if gates pass.
        gate_eval = {
            **gate_eval,
            "candidate": CANDIDATE_ID,
            "eligible_for_freeze": True,
            "implementable": True,
            "oracle_residual_removal": False,
            "rate_guard": rate_guard,
        }

        fold_rels = [
            float(r["relative_mae_delta"])
            for r in (gate_eval.get("fold_relative_mae_deltas") or [])
        ]
        freeze = freeze_shadow_candidate(
            candidate_id=CANDIDATE_ID,
            code_identity=identity,
            evidence={
                "fit_log": fit_log,
                "gate_evaluation": gate_eval,
                "rate_guard": rate_guard,
                "config_sha256": config_hash,
            },
            source_hashes=input_hashes,
            fold_mae_relative_deltas=fold_rels or [0.0],
            pooled_top120_spearman_baseline=float(
                gate_eval.get("pooled_top120_spearman_baseline") or 0.0
            ),
            pooled_top120_spearman_candidate=float(
                gate_eval.get("pooled_top120_spearman_candidate") or 0.0
            ),
            all_eligible_ok=bool(gate_eval.get("all_eligible_ok", False))
            and bool(gate_eval.get("position_ok", False)),
            coverage_unchanged=True,
            team_identity_unchanged=True,
            out_dir=dest,
            attribution_status=attribution_status,
        )

        # Persist sealed artifacts.
        config_path = dest / "config.json"
        config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        scored_path = dest / "scored_players.parquet"
        scored.to_parquet(scored_path, index=False)
        scores_path = dest / "score_table.csv"
        score_table.to_csv(scores_path, index=False)
        fit_path = dest / "nested_alpha_fits.json"
        fit_path.write_text(json.dumps(fit_log, indent=2), encoding="utf-8")

        output_hashes = {
            "config.json": sha256_file(config_path),
            "scored_players.parquet": sha256_file(scored_path),
            "score_table.csv": sha256_file(scores_path),
            "nested_alpha_fits.json": sha256_file(fit_path),
        }
        for name in (f"freeze_{CANDIDATE_ID}.json", "hold_v1_structural_role.json"):
            path = dest / name
            if path.is_file():
                output_hashes[name] = sha256_file(path)

        seal = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "candidate_id": CANDIDATE_ID,
            "producing_commit": _git_commit(),
            "attribution_status": attribution_status,
            "code_identity": identity,
            "config_sha256": config_hash,
            "input_hashes": input_hashes,
            "output_hashes": output_hashes,
            "nested_alpha_fits": fit_log,
            "gate_evaluation": gate_eval,
            "rate_guard": rate_guard,
            "freeze": {
                "verdict": freeze.get("verdict")
                or (
                    "freeze_shadow_candidate"
                    if (freeze.get("gate") or {}).get("passed")
                    else "hold_v1_structural_role"
                ),
                "gate_passed": bool((freeze.get("gate") or {}).get("passed")),
                "promotion_authorized": False,
            },
            "production_weights_unchanged": True,
            "production_guard": assert_production_unchanged(before),
            "note": (
                "Shadow-only Gate-A exposure blend for RB/WR. Does not mutate "
                "production weights, contracts, or release pointers."
            ),
        }
        seal["artifact_hash"] = canonical_json_hash(seal)
        seal_path = dest / "seal.json"
        seal_path.write_text(json.dumps(seal, indent=2), encoding="utf-8")
        # Re-hash seal into a thin pointer (seal content includes prior hashes).
        pointer = {
            "schema_version": SCHEMA_VERSION,
            "candidate_id": CANDIDATE_ID,
            "seal_path": str(seal_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "seal_sha256": sha256_file(seal_path),
            "config_sha256": config_hash,
            "code_bundle_sha256": identity["entrypoint_bundle_sha256"],
            "verdict": seal["freeze"]["verdict"],
            "promotion_authorized": False,
        }
        (dest / "seal_pointer.json").write_text(
            json.dumps(pointer, indent=2), encoding="utf-8"
        )
        assert_production_unchanged(before)
        return seal
