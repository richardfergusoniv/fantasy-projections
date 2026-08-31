"""Shadow v1 RB/WR attribution pipeline (read-only; production immutable)."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.projection.contracts import OUTPUT_DIR, REPO_ROOT
from src.projection.evaluation.accuracy_first import (
    TOP_ADP,
    apply_market_curves,
    fit_market_curves,
    sha256_file,
)
from src.projection.shadow.consensus_pin import (
    ConsensusPinError,
    load_pinned_consensus,
    persist_top120_membership,
    require_all_pinned_consensus,
)
from src.projection.shadow.decision_rules import (
    classify_diagnosis,
    flag_repair_candidate,
)
from src.projection.shadow.step6_decision import identify_codominant_components
from src.projection.shadow.error_decomposition import (
    decompose_prediction_error,
    stage_point_deltas,
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
from src.projection.shadow.stage_evidence import (
    AttributionIncompleteError,
    analyze_finalization_remainder,
    assert_traced_points_match_eval,
    load_fold_player_rates,
    load_fold_stage_scores,
    persist_fold_stage_artifacts,
    stage_evidence_complete,
)

SHADOW_OUTPUT_DIR = Path(OUTPUT_DIR) / "shadow_v1_rb_wr"
SCHEMA_VERSION = "shadow_v1_rb_wr_attribution_v1"
RB_WR = ("RB", "WR")
# Rolling-origin folds: train through source, score target.
FOLDS = ((2022, 2023), (2023, 2024), (2024, 2025))
SHADOW_ENTRYPOINTS = (
    "src.projection.shadow.rb_wr_attribution",
    "src.projection.shadow.consensus_pin",
    "src.projection.shadow.error_decomposition",
    "src.projection.shadow.decision_rules",
    "src.projection.shadow.production_guard",
    "src.projection.shadow.forbidden",
    "src.projection.shadow.repair",
    "src.projection.shadow.stage_evidence",
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


def _load_eval_frame(season: int) -> pd.DataFrame:
    path = Path(OUTPUT_DIR) / f"fantasy_evaluation_{season}.csv"
    assert_input_path_allowed(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    frame["player_id"] = frame["player_id"].astype(str)
    out = pd.DataFrame({
        "player_id": frame["player_id"],
        "display_name": frame.get("display_name"),
        "position": frame["preseason_position"].astype(str),
        "season": int(season),
        "is_rookie": frame.get("is_rookie", False),
        "forecast_covered": frame.get("forecast_covered", True),
        "v1_pred": pd.to_numeric(frame["model_points_end_to_end"], errors="coerce"),
        "composed_rate_ppg": pd.to_numeric(frame["model_rate_points"], errors="coerce"),
        "projected_games": pd.to_numeric(frame["projected_games"], errors="coerce"),
        "projected_volume_games": pd.to_numeric(
            frame.get("projected_volume_games", frame["projected_games"]),
            errors="coerce",
        ),
        "actual_points": pd.to_numeric(frame["actual_points"], errors="coerce").fillna(0.0),
        "actual_games_played": pd.to_numeric(
            frame["actual_games_played"], errors="coerce"
        ).fillna(0.0),
        "population_source": frame.get("population_source"),
        "preseason_team": frame.get("preseason_team"),
    })
    # Without a pre-compose board, raw rate equals composed rate and the
    # composition-rate effect is zero; stage CSV still reports when stages run.
    out["raw_rate_ppg"] = out["composed_rate_ppg"]
    out["composition_stages_applied"] = False
    return out


def _try_load_v2(season: int) -> pd.DataFrame:
    """Join historical v2 points with explicit coverage (null when uncovered)."""
    try:
        from scripts.ensemble_v1_v2 import DEFAULT_V2_ROOT, load_v2
    except ImportError:
        return pd.DataFrame(columns=["player_id", "v2_pred", "v2_covered"])
    root = Path(DEFAULT_V2_ROOT)
    assert_input_path_allowed(root)
    frame = load_v2(season, root)
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["player_id", "v2_pred", "v2_covered"])
    out = frame[["player_id", "v2_pred"]].copy()
    out["player_id"] = out["player_id"].astype(str)
    out["v2_pred"] = pd.to_numeric(out["v2_pred"], errors="coerce")
    out["v2_covered"] = out["v2_pred"].notna()
    return out


def _attach_consensus(
    frame: pd.DataFrame,
    consensus: pd.DataFrame,
    *,
    pin_record: dict[str, Any],
) -> pd.DataFrame:
    cols = ["player_id", "adp"]
    if "ecr" in consensus.columns:
        cols.append("ecr")
    merged = frame.merge(consensus[cols], on="player_id", how="left")
    merged["adp"] = pd.to_numeric(merged["adp"], errors="coerce")
    merged["draft_relevant_top120"] = merged["adp"].notna() & merged["adp"].le(TOP_ADP)
    merged["adp_covered"] = merged["adp"].notna()
    merged["consensus_as_of"] = pin_record.get("as_of")
    merged["consensus_sha256"] = pin_record.get("actual_hash")
    return merged


def _attach_adp_points(
    frame: pd.DataFrame,
    *,
    target_season: int,
    consensus_by_season: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """Fit ADP curves on earlier seasons only; 2023 reports rank diagnostics only."""
    out = frame.copy()
    out["adp_points"] = np.nan
    out["adp_points_calibrated"] = False
    history_seasons = [s for s in sorted(consensus_by_season) if s < target_season]
    if not history_seasons:
        # 2023: no earlier calibration season — MAE against ADP points omitted.
        out["adp_calibration_season"] = None
        return out
    hist_rows = []
    for season in history_seasons:
        eval_path = Path(OUTPUT_DIR) / f"fantasy_evaluation_{season}.csv"
        if not eval_path.is_file():
            continue
        assert_input_path_allowed(eval_path)
        ev = pd.read_csv(eval_path)
        ev["player_id"] = ev["player_id"].astype(str)
        cons = consensus_by_season[season]
        merged = ev.merge(cons[["player_id", "adp"]], on="player_id", how="inner")
        merged["position"] = merged["preseason_position"].astype(str)
        merged["actual_points"] = pd.to_numeric(merged["actual_points"], errors="coerce")
        merged["adp"] = pd.to_numeric(merged["adp"], errors="coerce")
        hist_rows.append(
            merged[["player_id", "position", "actual_points", "adp"]].dropna()
        )
    if not hist_rows:
        return out
    history = pd.concat(hist_rows, ignore_index=True)
    try:
        curves = fit_market_curves(history)
    except ValueError:
        return out
    out["adp_points"] = apply_market_curves(out, curves)
    out["adp_points_calibrated"] = out["adp_points"].notna()
    out["adp_calibration_season"] = max(history_seasons)
    return out


def _population_slice(frame: pd.DataFrame, *, population: str) -> pd.DataFrame:
    if population == "all_eligible":
        return frame.copy()
    if population == "top120":
        return frame[frame["draft_relevant_top120"]].copy()
    raise ValueError(f"Unknown population: {population}")


def _metric_row(
    frame: pd.DataFrame,
    *,
    pred_col: str,
    season: int,
    source_season: int,
    population: str,
    position: str,
) -> dict[str, Any]:
    sub = frame[frame["position"].eq(position)].dropna(subset=["actual_points", pred_col])
    actual = sub["actual_points"].to_numpy(dtype=float)
    pred = sub[pred_col].to_numpy(dtype=float)
    mae = float(np.mean(np.abs(pred - actual))) if len(sub) else float("nan")
    if len(sub) >= 3:
        rho = float(pd.Series(actual).corr(pd.Series(pred), method="spearman"))
    else:
        rho = float("nan")
    # ADP rank correlation (always safe); ADP-point MAE only when calibrated.
    adp_rank_rho = float("nan")
    if "adp" in sub.columns and sub["adp"].notna().sum() >= 3:
        adp_rank_rho = float(
            pd.Series(actual).corr(
                -pd.to_numeric(sub["adp"], errors="coerce"), method="spearman"
            )
        )
    row = {
        "source_season": int(source_season),
        "target_season": int(season),
        "population": population,
        "position": position,
        "signal": pred_col,
        "n": int(len(sub)),
        "mae": mae,
        "spearman": rho,
        "adp_rank_spearman": adp_rank_rho,
        "v2_coverage": float(sub["v2_covered"].mean()) if "v2_covered" in sub else None,
        "adp_coverage": float(sub["adp_covered"].mean()) if "adp_covered" in sub else None,
        "evidence_role": (
            "holdout"
            if season == 2025
            else ("training_diagnostic" if season == 2024 else "membership_diagnostic")
        ),
    }
    if season == 2023 and pred_col == "adp_points":
        row["mae"] = None
        row["note"] = "adp_point_mae_omitted_no_earlier_calibration_season"
    return row


def apply_traced_rates(
    frame: pd.DataFrame,
    player_rates: pd.DataFrame,
) -> pd.DataFrame:
    """Overwrite rates / v1 points from leakage-safe compose player_rates."""
    if player_rates is None or player_rates.empty:
        raise AttributionIncompleteError("player_rates missing or empty")
    required = (
        "player_id",
        "raw_rate_ppg",
        "composed_rate_ppg",
        "traced_v1_pred",
        "projected_games",
    )
    missing_cols = [c for c in required if c not in player_rates.columns]
    if missing_cols:
        raise AttributionIncompleteError(
            f"player_rates missing columns: {missing_cols}"
        )
    out = frame.copy()
    out["player_id"] = out["player_id"].astype(str)
    rates = player_rates.loc[:, list(required)].copy()
    rates["player_id"] = rates["player_id"].astype(str)
    for col in ("raw_rate_ppg", "composed_rate_ppg", "traced_v1_pred", "projected_games"):
        rates[col] = pd.to_numeric(rates[col], errors="coerce")
    drop = [
        c
        for c in (
            "raw_rate_ppg",
            "composed_rate_ppg",
            "traced_v1_pred",
            "projected_games",
        )
        if c in out.columns
    ]
    if drop:
        out = out.drop(columns=drop)
    out = out.merge(rates, on="player_id", how="left")
    uncovered = (
        out["raw_rate_ppg"].isna()
        | out["composed_rate_ppg"].isna()
        | out["traced_v1_pred"].isna()
        | out["projected_games"].isna()
    )
    # Players absent from the compose long board (uncovered / zero-component)
    # keep explicit zeros rather than silently inventing rates.
    if bool(uncovered.any()):
        out.loc[uncovered, "raw_rate_ppg"] = 0.0
        out.loc[uncovered, "composed_rate_ppg"] = 0.0
        out.loc[uncovered, "traced_v1_pred"] = 0.0
        if "projected_volume_games" in out.columns:
            out.loc[uncovered, "projected_games"] = pd.to_numeric(
                out.loc[uncovered, "projected_volume_games"], errors="coerce"
            ).fillna(0.0)
        else:
            out.loc[uncovered, "projected_games"] = 0.0
    out["stage_rate_covered"] = ~uncovered
    out["v1_pred"] = out["traced_v1_pred"]
    out["composition_stages_applied"] = True
    return out


def build_fold_attribution(
    *,
    source_season: int,
    target_season: int,
    consensus: pd.DataFrame,
    pin_record: dict[str, Any],
    consensus_by_season: dict[int, pd.DataFrame],
    stage_scores: dict[str, dict[str, dict]] | None = None,
    player_rates: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], pd.DataFrame]:
    """Build player attribution + metric rows for one rolling-origin fold."""
    if not stage_evidence_complete(stage_scores):
        raise AttributionIncompleteError(
            f"Incomplete stage scores for {source_season}->{target_season}"
        )
    if player_rates is None or player_rates.empty:
        raise AttributionIncompleteError(
            f"Missing player rates for {source_season}->{target_season}"
        )

    players = _load_eval_frame(target_season)
    players = players[players["position"].isin(RB_WR)].copy()
    players = _attach_consensus(players, consensus, pin_record=pin_record)
    v2 = _try_load_v2(target_season)
    players = players.merge(v2, on="player_id", how="left")
    players["v2_covered"] = players.get("v2_covered", False)
    if "v2_pred" not in players.columns:
        players["v2_pred"] = np.nan
    players["v2_covered"] = players["v2_pred"].notna()
    players = _attach_adp_points(
        players, target_season=target_season, consensus_by_season=consensus_by_season
    )
    players = apply_traced_rates(players, player_rates)
    players["source_season"] = int(source_season)
    players["fold"] = f"{source_season}->{target_season}"
    players = decompose_prediction_error(players)

    metric_rows: list[dict[str, Any]] = []
    for population in ("all_eligible", "top120"):
        sliced = _population_slice(players, population=population)
        for position in RB_WR:
            for signal in ("v1_pred", "v2_pred", "adp_points", "composed_rate_ppg"):
                if signal not in sliced.columns:
                    continue
                metric_rows.append(
                    _metric_row(
                        sliced,
                        pred_col=signal,
                        season=target_season,
                        source_season=source_season,
                        population=population,
                        position=position,
                    )
                )

    stage_rows = stage_point_deltas(
        stage_scores, player_ids=players["player_id"].astype(str).tolist()
    )
    if not stage_rows.empty:
        stage_rows["source_season"] = int(source_season)
        stage_rows["target_season"] = int(target_season)
        stage_rows = stage_rows.merge(
            players[["player_id", "draft_relevant_top120", "actual_points"]],
            on="player_id",
            how="left",
        )
    return players, metric_rows, stage_rows


def generate_fold_stage_evidence(
    *,
    conn: Any,
    feature_table: pd.DataFrame,
    out_dir: Path,
    source_season: int,
    target_season: int,
) -> dict[str, Any]:
    """Build, parity-check, and persist compose stage evidence for one fold."""
    from src.projection.fantasy_evaluation import build_leakage_safe_compose_checkpoints

    checkpoint = build_leakage_safe_compose_checkpoints(
        conn, feature_table, source_season, target_season
    )
    eval_path = Path(OUTPUT_DIR) / f"fantasy_evaluation_{target_season}.csv"
    assert_input_path_allowed(eval_path)
    if not eval_path.is_file():
        raise AttributionIncompleteError(f"Missing evaluation frame: {eval_path}")
    eval_frame = pd.read_csv(eval_path)
    eval_frame["player_id"] = eval_frame["player_id"].astype(str)
    parity = assert_traced_points_match_eval(checkpoint["player_rates"], eval_frame)
    artifact_hashes = persist_fold_stage_artifacts(
        out_dir=Path(out_dir),
        source_season=source_season,
        target_season=target_season,
        checkpoint=checkpoint,
    )
    return {
        "source_season": int(source_season),
        "target_season": int(target_season),
        "stage_scores": checkpoint["stage_scores"],
        "player_rates": checkpoint["player_rates"],
        "parity": parity,
        "artifact_hashes": artifact_hashes,
        "checkpoint": checkpoint,
    }


def _component_dominance(players: pd.DataFrame) -> dict[str, float]:
    top = players[players["draft_relevant_top120"] & players["position"].isin(RB_WR)]
    if top.empty:
        return {}
    return {
        col: float(top[col].mean())
        for col in (
            "raw_rate_error",
            "composition_rate_effect",
            "availability_effect",
            "finalization_remainder",
        )
    }


def _write_status_manifest(
    out_dir: Path,
    *,
    status: str,
    error: Exception | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "producing_commit": _git_commit(),
        "production_weights_unchanged": True,
        "candidate_freeze_allowed": status == "ok",
    }
    if error is not None:
        payload["error_type"] = type(error).__name__
        payload["error"] = str(error)
    if extra:
        payload.update(extra)
        # Freeze gate is status-driven even if extra tries to override.
        payload["candidate_freeze_allowed"] = status == "ok"
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    if status in {"fail_closed", "attribution_incomplete"}:
        for name in (
            "attribution_players.parquet",
            "attribution_metrics.csv",
            "stage_attribution.csv",
            "attribution_summary.json",
        ):
            victim = out_dir / name
            if victim.exists():
                victim.unlink()
    return path


def run_shadow_attribution(
    *,
    out_dir: str | Path | None = None,
    folds: tuple[tuple[int, int], ...] = FOLDS,
    stage_scores_by_season: dict[int, dict] | None = None,
    player_rates_by_season: dict[int, pd.DataFrame] | None = None,
    generate_stages: bool = True,
    n_boot: int = 500,
) -> dict[str, Any]:
    """Run the full leakage-safe attribution study under output/shadow_v1_rb_wr/."""
    # Static graph must be clean; runtime guard blocks dynamic imports mid-run.
    # Do not require forbidden modules to be absent from sys.modules — other
    # tests may import Sleeper comparison modules in the same pytest process.
    assert_no_forbidden_imports(SHADOW_ENTRYPOINTS)
    dest = Path(out_dir or SHADOW_OUTPUT_DIR)
    dest.mkdir(parents=True, exist_ok=True)

    scores_by_season: dict[int, dict] = dict(stage_scores_by_season or {})
    rates_by_season: dict[int, pd.DataFrame] = dict(player_rates_by_season or {})

    before = snapshot_production_artifacts()
    try:
        with ForbiddenImportGuard():
            pins = require_all_pinned_consensus(
                seasons=tuple(sorted({target for _, target in folds}))
            )
            consensus_by_season = {}
            for season, pin in pins.items():
                rows, _ = load_pinned_consensus(
                    season, expected_hash=pin["expected_hash"]
                )
                consensus_by_season[season] = rows
                persist_top120_membership(
                    season,
                    rows,
                    out_path=dest / f"top120_membership_{season}.json",
                    pin_record=pin,
                )

            need_generate: list[tuple[int, int]] = []
            eval_parity_reports: dict[str, Any] = {}
            for source_season, target_season in folds:
                scores = scores_by_season.get(target_season)
                rates = rates_by_season.get(target_season)
                if stage_evidence_complete(scores) and rates is not None and not rates.empty:
                    continue
                try:
                    scores = load_fold_stage_scores(dest, source_season, target_season)
                    rates = load_fold_player_rates(dest, source_season, target_season)
                    scores_by_season[target_season] = scores
                    rates_by_season[target_season] = rates
                except AttributionIncompleteError:
                    if generate_stages:
                        need_generate.append((source_season, target_season))
                    else:
                        raise

            if need_generate:
                from src.projection.data_prep import get_conn
                from src.projection.features import build_player_season_features

                conn = get_conn()
                try:
                    feature_table = build_player_season_features(conn)
                    for source_season, target_season in need_generate:
                        generated = generate_fold_stage_evidence(
                            conn=conn,
                            feature_table=feature_table,
                            out_dir=dest,
                            source_season=source_season,
                            target_season=target_season,
                        )
                        scores_by_season[target_season] = generated["stage_scores"]
                        rates_by_season[target_season] = generated["player_rates"]
                        eval_parity_reports[f"{source_season}->{target_season}"] = generated[
                            "parity"
                        ]
                        eval_parity_reports[f"{source_season}->{target_season}"][
                            "compose_board_parity"
                        ] = (generated.get("checkpoint") or {}).get("parity")
                finally:
                    conn.close()

            for source_season, target_season in folds:
                if not stage_evidence_complete(scores_by_season.get(target_season)):
                    raise AttributionIncompleteError(
                        f"Incomplete stage evidence for {source_season}->{target_season}"
                    )
                rates = rates_by_season.get(target_season)
                if rates is None or rates.empty:
                    raise AttributionIncompleteError(
                        f"Missing player rates for {source_season}->{target_season}"
                    )

            player_frames = []
            metric_rows: list[dict[str, Any]] = []
            stage_frames = []
            for source_season, target_season in folds:
                players, metrics, stage_df = build_fold_attribution(
                    source_season=source_season,
                    target_season=target_season,
                    consensus=consensus_by_season[target_season],
                    pin_record=pins[target_season],
                    consensus_by_season=consensus_by_season,
                    stage_scores=scores_by_season[target_season],
                    player_rates=rates_by_season[target_season],
                )
                player_frames.append(players)
                metric_rows.extend(metrics)
                if stage_df is not None and not stage_df.empty:
                    stage_frames.append(stage_df)

            players_all = pd.concat(player_frames, ignore_index=True)
            metrics_df = pd.DataFrame(metric_rows)
            stage_df = (
                pd.concat(stage_frames, ignore_index=True)
                if stage_frames
                else pd.DataFrame()
            )
            if stage_df.empty:
                raise AttributionIncompleteError(
                    "stage_attribution.csv would be empty; complete stage evidence required"
                )

            dominance = _component_dominance(players_all)
            finalization_analysis = analyze_finalization_remainder(players_all)
            parity_defects = [
                fold
                for fold, report in eval_parity_reports.items()
                if not (report or {}).get("ok", True)
            ]
            flagged = []
            for position in RB_WR:
                for stage in (
                    "team_volume_reconcile",
                    "concentration",
                    "td_constraints",
                    "counting_stat_constraints",
                    "season_total_finalization",
                ):
                    pos_players = players_all[
                        players_all["position"].eq(position)
                        & players_all["draft_relevant_top120"]
                    ]
                    flagged.append(
                        flag_repair_candidate(
                            stage=stage,
                            position=position,
                            fold_metrics=[],
                            pooled_actual=pos_players["actual_points"].to_numpy(dtype=float),
                            pooled_baseline=pos_players["v1_pred"].to_numpy(dtype=float),
                            pooled_stage=pos_players["v1_pred"].to_numpy(dtype=float),
                            all_eligible_spearman_baseline=0.0,
                            all_eligible_spearman_without_stage=0.0,
                            n_boot=n_boot,
                        )
                    )

            diagnosis = classify_diagnosis(
                parity_defects=parity_defects,
                component_dominance=dominance,
                flagged_stages=flagged,
                stages_complete=True,
                finalization_analysis=finalization_analysis,
            )
            codominant = identify_codominant_components(dominance)
            labeling = {
                "pipeline_location": diagnosis,
                "codominant_error_components": codominant,
                "note": (
                    "pipeline_location is the diagnosed locus; "
                    "codominant_error_components are magnitude leaders "
                    "and may include raw_rate_error / availability_effect "
                    "even when pipeline_location is composition_defect."
                ),
            }

            players_path = dest / "attribution_players.parquet"
            metrics_path = dest / "attribution_metrics.csv"
            stage_path = dest / "stage_attribution.csv"
            summary_path = dest / "attribution_summary.json"
            players_all.to_parquet(players_path, index=False)
            metrics_df.to_csv(metrics_path, index=False)
            stage_df.to_csv(stage_path, index=False)

            output_hashes = {
                "attribution_players.parquet": sha256_file(players_path),
                "attribution_metrics.csv": sha256_file(metrics_path),
                "stage_attribution.csv": sha256_file(stage_path),
            }
            for season, pin in pins.items():
                member_path = dest / f"top120_membership_{season}.json"
                output_hashes[member_path.name] = sha256_file(member_path)

            summary = {
                "schema_version": SCHEMA_VERSION,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "producing_commit": _git_commit(),
                "folds": [f"{a}->{b}" for a, b in folds],
                "positions": list(RB_WR),
                "populations": ["all_eligible", "top120"],
                "status": "ok",
                "stages_complete": True,
                "diagnosis": diagnosis,
                "labeling": labeling,
                "component_dominance": dominance,
                "finalization_analysis": finalization_analysis,
                "eval_csv_parity": eval_parity_reports,
                "parity_defects": parity_defects,
                "repair_candidates": flagged,
                "candidate_freeze_allowed": True,
                "hold_verdict": "hold_v1_structural_role",
                "notes": {
                    "2023": "ADP rank/membership diagnostics only; no ADP-point MAE",
                    "2024": "training/diagnostic evidence",
                    "2025": "existing holdout",
                    "selected_ensemble_replay": "descriptive_not_untouched_evidence",
                    "stages": "leakage_safe_compose_checkpoints_required",
                    "v1_source": "traced_compose_season_totals",
                    "eval_csv": (
                        "soft_parity_only; stale CSVs yield parity_or_data_defect "
                        "without discarding traced attribution"
                    ),
                },
                "player_rows": int(len(players_all)),
                "metric_rows": int(len(metrics_df)),
                "stage_rows": int(len(stage_df)),
            }
            summary_path.write_text(
                json.dumps(summary, indent=2, default=str), encoding="utf-8"
            )
            output_hashes["attribution_summary.json"] = sha256_file(summary_path)

            after = snapshot_production_artifacts()
            prod = assert_production_unchanged(before, after)

            manifest = _write_status_manifest(
                dest,
                status="ok",
                extra={
                    "generated_at": summary["generated_at"],
                    "producing_commit": _git_commit(),
                    "production_guard": prod,
                    "consensus_pins": pins,
                    "membership_hashes": {
                        str(season): pin["top120_membership_hash"]
                        for season, pin in pins.items()
                    },
                    "input_hashes": {
                        f"fantasy_evaluation_{target}.csv": sha256_file(
                            Path(OUTPUT_DIR) / f"fantasy_evaluation_{target}.csv"
                        )
                        for _, target in folds
                        if (Path(OUTPUT_DIR) / f"fantasy_evaluation_{target}.csv").is_file()
                    },
                    "output_hashes": output_hashes,
                    "diagnosis": diagnosis,
                    "finalization_analysis": finalization_analysis,
                    "forbidden_modules_checked": True,
                    "stages_complete": True,
                },
            )
            return json.loads(manifest.read_text(encoding="utf-8"))
    except AttributionIncompleteError as exc:
        _write_status_manifest(dest, status="attribution_incomplete", error=exc)
        assert_production_unchanged(before)
        raise
    except ConsensusPinError as exc:
        _write_status_manifest(dest, status="fail_closed", error=exc)
        assert_production_unchanged(before)
        raise
    except Exception:
        assert_production_unchanged(before)
        raise


apply_stage_rates = apply_traced_rates
