"""Decision-quality evaluation for rolling-origin fantasy folds.

Reuses leakage-safe populations from ``fantasy_evaluation``, market matching
from ``market_metrics``, and the authoritative VORP/tier contracts from
``draft_assistant``.  Computes top-N precision/recall grids, tier calibration,
one-dimensional segments, and roster-independent ADP-choice regret.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.draft_assistant import tiers as tier_contract
from src.draft_assistant import vorp as vorp_contract
from src.draft_assistant.tiers import TierConfig
from src.draft_assistant.vorp import add_vorp_columns
from src.projection.contracts import OUTPUT_DIR, REPO_ROOT
from src.projection.evaluation.accuracy_first import (
    TOP_ADP,
    apply_market_curves,
    fit_market_curves,
    load_consensus_snapshot,
    sha256_file,
)
from src.projection.evaluation.calibration_segments import MINIMUM_N_FOR_GATE
from src.projection.fantasy_evaluation import POSITIONS, run_evaluation
from src.projection.market_metrics import matched_market_frame

TOP_N_GRID = (6, 12, 24, 36, 48)
FORECAST_FAMILIES = ("pure_model", "market_informed", "adp")
EVAL_SCOPES = (
    "all_eligible",
    "forecast_covered",
    "starter_depth_tier_1",
    "starter_8plus_games",
)
DEFAULT_FOLDS = (2023, 2024, 2025)

ADP_BANDS = (
    ("adp_1_24", 1, 24),
    ("adp_25_60", 25, 60),
    ("adp_61_120", 61, 120),
    ("adp_above_120", 121, None),
    ("adp_unranked", None, None),
)

AGE_BANDS: dict[str, list[tuple[str, float | None, float | None]]] = {
    "QB": [("young", None, 26.0), ("prime", 26.0, 32.0), ("veteran", 32.0, None)],
    "RB": [("young", None, 24.0), ("prime", 24.0, 28.0), ("veteran", 28.0, None)],
    "WR": [("young", None, 25.0), ("prime", 25.0, 29.0), ("veteran", 29.0, None)],
    "TE": [("young", None, 26.0), ("prime", 26.0, 30.0), ("veteran", 30.0, None)],
}

REGRET_WINDOWS = (6, 12, 24)
DEFAULT_REGRET_WINDOW = 12

DECISION_QUALITY_DIR = Path(OUTPUT_DIR) / "evaluation" / "decision_quality"
FROZEN_BASELINE_ID = "decision_quality_baseline_v1"


def vorp_tier_contract_hashes() -> dict[str, str]:
    """Stable hashes for the tier/VORP contract modules."""
    vorp_path = Path(vorp_contract.__file__)
    tiers_path = Path(tier_contract.__file__)
    return {
        "vorp_module_sha256": sha256_file(vorp_path),
        "tiers_module_sha256": sha256_file(tiers_path),
        "tier_config": {
            "position_gaps": tier_contract.DEFAULT_TIER_GAPS,
            "overall_gap": tier_contract.OVERALL_TIER_GAP,
            "flex_gap": tier_contract.FLEX_TIER_GAP,
        },
        "vorp_replacement_contract": {
            "starters": vorp_contract.STARTERS,
            "flex_share": vorp_contract.FLEX_SHARE,
            "season_games": vorp_contract.SEASON_GAMES,
        },
    }


def _kth_score(values: pd.Series, rank: int) -> float:
    ordered = pd.to_numeric(values, errors="coerce").dropna().sort_values(ascending=False)
    if ordered.empty:
        return float("nan")
    idx = min(max(int(rank), 1), len(ordered)) - 1
    return float(ordered.iloc[idx])


def _scope_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    masks: dict[str, pd.Series] = {
        "all_eligible": pd.Series(True, index=frame.index),
        "forecast_covered": frame.get("forecast_covered", pd.Series(False, index=frame.index)).astype(bool),
    }
    if "depth_tier" in frame.columns:
        starter = frame["depth_tier"].eq(1.0)
        masks["starter_depth_tier_1"] = starter
        if "actual_games_played" in frame.columns:
            masks["starter_8plus_games"] = starter & frame["actual_games_played"].ge(8)
        else:
            masks["starter_8plus_games"] = pd.Series(False, index=frame.index)
    return masks


def attach_forecast_family_points(
    frame: pd.DataFrame,
    *,
    calibration_frame: pd.DataFrame | None = None,
    market_curves: dict | None = None,
    market_blend: float = 0.35,
) -> pd.DataFrame:
    """Add pure, market-informed, and ADP-only point columns."""
    out = frame.copy()
    out["pure_model_points"] = pd.to_numeric(
        out.get("model_points_end_to_end", out.get("model_points", 0.0)),
        errors="coerce",
    ).fillna(0.0)
    out["market_informed_points"] = out["pure_model_points"]
    out["adp_points"] = np.nan
    if calibration_frame is not None and market_curves is not None and "adp" in calibration_frame.columns:
        cal = calibration_frame.copy()
        cal["adp"] = pd.to_numeric(cal["adp"], errors="coerce")
        cal["position"] = cal.get("position", cal.get("preseason_position"))
        cal["adp_points"] = apply_market_curves(cal, market_curves)
        adp_map = cal.set_index("player_id")["adp_points"]
        raw_adp = cal.set_index("player_id")["adp"]
        out["adp_points"] = out["player_id"].map(adp_map)
        out["adp"] = out["player_id"].map(raw_adp)
        blend = float(market_blend)
        has_market = out["adp_points"].notna()
        out.loc[has_market, "market_informed_points"] = (
            (1.0 - blend) * out.loc[has_market, "pure_model_points"]
            + blend * out.loc[has_market, "adp_points"]
        )
    family_cols = {
        "pure_model": "pure_model_points",
        "market_informed": "market_informed_points",
        "adp": "adp_points",
    }
    for family, col in family_cols.items():
        out[f"{family}_points"] = pd.to_numeric(out.get(col), errors="coerce")
    return out


def top_n_precision_recall_rows(
    frame: pd.DataFrame,
    *,
    source_season: int,
    target_season: int,
    forecast_family: str,
    points_col: str,
) -> list[dict[str, Any]]:
    """Tie-aware top-N precision/recall at each grid N and evaluation scope."""
    rows: list[dict[str, Any]] = []
    masks = _scope_masks(frame)
    for position in POSITIONS:
        pos = frame[frame["preseason_position"].eq(position)]
        if pos.empty:
            continue
        actual_cut_cache: dict[int, pd.Series] = {}
        for n in TOP_N_GRID:
            actual_cut_cache[n] = pos["actual_points"].ge(_kth_score(pos["actual_points"], n))
        for scope, mask in masks.items():
            scoped = pos[mask.loc[pos.index].fillna(False)]
            if scoped.empty:
                continue
            pred_values = pd.to_numeric(scoped[points_col], errors="coerce").fillna(0.0)
            for n in TOP_N_GRID:
                pred_cut = _kth_score(pred_values, n)
                pred_top = pred_values.ge(pred_cut)
                actual_top = scoped["actual_points"].ge(_kth_score(scoped["actual_points"], n))
                hits = int((pred_top & actual_top).sum())
                pred_n = int(pred_top.sum())
                actual_n = int(actual_top.sum())
                rows.append({
                    "source_season": int(source_season),
                    "target_season": int(target_season),
                    "position": position,
                    "scope": scope,
                    "forecast_family": forecast_family,
                    "top_n": int(n),
                    "n": int(len(scoped)),
                    "predicted_top_n": pred_n,
                    "actual_top_n": actual_n,
                    "hits": hits,
                    "precision": hits / pred_n if pred_n else float("nan"),
                    "recall": hits / actual_n if actual_n else float("nan"),
                    "hit_rate": hits / min(pred_n, actual_n) if pred_n and actual_n else float("nan"),
                })
    return rows


def _assign_tiers(points: pd.Series, position: str, *, overall: bool = False) -> pd.Series:
    cfg = TierConfig(position_gaps=tier_contract.DEFAULT_TIER_GAPS)
    gap = cfg.overall_gap if overall else cfg.gap_for(position)
    ordered = points.sort_values(ascending=False)
    tier_vals = tier_contract.assign_tiers(ordered, gap=gap, pct_gap=0.03 if not overall else 0.04)
    out = pd.Series(index=points.index, dtype=int)
    out.loc[ordered.index] = tier_vals.values
    return out


def tier_calibration_rows(
    frame: pd.DataFrame,
    *,
    source_season: int,
    target_season: int,
    forecast_family: str,
    points_col: str,
) -> list[dict[str, Any]]:
    """Predicted-vs-realized tier matrix and summary rates."""
    rows: list[dict[str, Any]] = []
    work = frame.copy()
    work["actual_vorp_frame"] = pd.to_numeric(work["actual_points"], errors="coerce").fillna(0.0)
    realized = add_vorp_columns(
        work.rename(columns={"preseason_position": "position"}),
        points_col="actual_vorp_frame",
        position_col="position",
    )
    work["realized_vorp"] = realized["vorp"].values
    for position in POSITIONS:
        pos = work[work["preseason_position"].eq(position)].copy()
        if pos.empty:
            continue
        pos["realized_pos_tier"] = _assign_tiers(pos["actual_points"], position)
        pred_pts = pd.to_numeric(pos[points_col], errors="coerce").fillna(0.0)
        pos["predicted_pos_tier"] = _assign_tiers(pred_pts, position)
        matrix = (
            pos.groupby(["predicted_pos_tier", "realized_pos_tier"], observed=True)
            .size()
            .reset_index(name="count")
        )
        exact = float((pos["predicted_pos_tier"] == pos["realized_pos_tier"]).mean())
        within_one = float((pos["predicted_pos_tier"] - pos["realized_pos_tier"]).abs().le(1).mean())
        bias = float((pos["predicted_pos_tier"] - pos["realized_pos_tier"]).mean())
        for pred_tier, grp in pos.groupby("predicted_pos_tier", observed=True):
            rows.append({
                "source_season": int(source_season),
                "target_season": int(target_season),
                "position": position,
                "forecast_family": forecast_family,
                "predicted_tier": int(pred_tier),
                "exact_tier_rate": exact,
                "within_one_tier_rate": within_one,
                "directional_tier_bias": bias,
                "realized_mean_points": float(grp["actual_points"].mean()),
                "realized_mean_vorp": float(grp["realized_vorp"].mean()),
                "n_in_predicted_tier": int(len(grp)),
                "tier_matrix": matrix.to_dict(orient="records"),
            })
    return rows


def _assign_adp_band(adp: pd.Series) -> pd.Series:
    values = pd.to_numeric(adp, errors="coerce")
    labels = pd.Series("adp_unranked", index=adp.index, dtype=str)
    ranked = values.notna()
    labels.loc[ranked & values.le(24)] = "adp_1_24"
    labels.loc[ranked & values.between(25, 60)] = "adp_25_60"
    labels.loc[ranked & values.between(61, 120)] = "adp_61_120"
    labels.loc[ranked & values.gt(120)] = "adp_above_120"
    return labels


def _assign_age_band(frame: pd.DataFrame) -> pd.Series:
    age = pd.to_numeric(frame.get("age"), errors="coerce")
    pos = frame["preseason_position"].astype(str)
    out = pd.Series("unknown_age", index=frame.index, dtype=str)
    for position, bands in AGE_BANDS.items():
        mask = pos.eq(position) & age.notna()
        for label, lo, hi in bands:
            band = mask.copy()
            if lo is not None:
                band &= age.gt(lo)
            if hi is not None:
                band &= age.le(hi)
            out.loc[band] = label
    return out


def _assign_team_change(frame: pd.DataFrame) -> pd.Series:
    prior_team = frame.get("prior_team")
    if prior_team is None:
        return pd.Series("unknown_cohort", index=frame.index, dtype=str)
    same = frame["preseason_team"].astype(str).eq(prior_team.astype(str))
    rookie = frame.get("is_rookie", False).astype(bool)
    out = pd.Series("changed_team", index=frame.index, dtype=str)
    out.loc[same] = "same_team"
    out.loc[rookie | prior_team.isna()] = "rookie_or_no_prior_team"
    return out


def segment_metric_rows(
    frame: pd.DataFrame,
    *,
    source_season: int,
    target_season: int,
    forecast_family: str,
    points_col: str,
    minimum_n: int = MINIMUM_N_FOR_GATE,
) -> list[dict[str, Any]]:
    """One-dimensional segment diagnostics; sub-minimum segments are not gate-eligible."""
    work = frame.copy()
    segments: dict[str, pd.Series] = {
        "adp_band": _assign_adp_band(work.get("adp", pd.Series(index=work.index))),
        "age_band": _assign_age_band(work),
        "preseason_team": work["preseason_team"].astype(str),
        "team_change_cohort": _assign_team_change(work),
    }
    rows: list[dict[str, Any]] = []
    actual = pd.to_numeric(work["actual_points"], errors="coerce")
    pred = pd.to_numeric(work[points_col], errors="coerce").fillna(0.0)
    for segment_type, labels in segments.items():
        for label, grp in work.groupby(labels, observed=True):
            if grp.empty:
                continue
            sub_actual = actual.loc[grp.index]
            sub_pred = pred.loc[grp.index]
            n = int(len(grp))
            rows.append({
                "source_season": int(source_season),
                "target_season": int(target_season),
                "forecast_family": forecast_family,
                "segment_type": segment_type,
                "segment_value": str(label),
                "n": n,
                "eligible_for_gate": n >= minimum_n,
                "points_mae": float((sub_pred - sub_actual).abs().mean()),
                "spearman": float(sub_pred.corr(sub_actual, method="spearman"))
                if n >= 3 else float("nan"),
                "mean_bias": float((sub_pred - sub_actual).mean()),
            })
    return rows


def adp_choice_regret_rows(
    matched: pd.DataFrame,
    *,
    source_season: int,
    target_season: int,
    window: int = DEFAULT_REGRET_WINDOW,
) -> list[dict[str, Any]]:
    """Roster-independent ADP-window regret on the matched top-120 population."""
    if matched.empty:
        return []
    work = matched.sort_values("mkt_raw").reset_index(drop=True).copy()
    work["actual_vorp"] = add_vorp_columns(
        work,
        points_col="actual_points",
        position_col="position",
    )["vorp"].values
    strategies = ("pure_model", "market_informed", "adp")
    rows: list[dict[str, Any]] = []
    n_picks = len(work)
    for pick_idx in range(n_picks):
        end = min(pick_idx + int(window), n_picks)
        window_frame = work.iloc[pick_idx:end]
        if window_frame.empty:
            continue
        adp_band = _assign_adp_band(window_frame["mkt_raw"]).iloc[0]
        available_vorp = window_frame.set_index("player_id")["actual_vorp"]
        best_vorp = float(available_vorp.max())
        for strategy in strategies:
            if strategy == "adp":
                chosen_idx = window_frame["mkt_raw"].idxmin()
            elif strategy == "pure_model":
                chosen_idx = window_frame["pure_model_points"].idxmax()
            else:
                chosen_idx = window_frame["market_informed_points"].idxmax()
            chosen_id = window_frame.loc[chosen_idx, "player_id"]
            chosen_vorp = float(available_vorp.loc[chosen_id])
            regret = best_vorp - chosen_vorp
            adp_choice = window_frame.loc[window_frame["mkt_raw"].idxmin(), "player_id"]
            adp_vorp = float(available_vorp.loc[adp_choice])
            rows.append({
                "source_season": int(source_season),
                "target_season": int(target_season),
                "pick_index": int(pick_idx + 1),
                "window": int(window),
                "strategy": strategy,
                "selected_position": str(window_frame.loc[chosen_idx, "position"]),
                "adp_band": str(adp_band),
                "chosen_vorp": chosen_vorp,
                "best_available_vorp": best_vorp,
                "regret": regret,
                "zero_regret": bool(np.isclose(regret, 0.0)),
                "gain_vs_adp": chosen_vorp - adp_vorp,
            })
    return rows


def summarize_regret(regret_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not regret_rows:
        return []
    df = pd.DataFrame(regret_rows)
    summaries: list[dict[str, Any]] = []
    group_cols = ["source_season", "target_season", "window", "strategy"]
    for keys, grp in df.groupby(group_cols, observed=True):
        regret = grp["regret"].astype(float)
        summaries.append({
            "source_season": keys[0],
            "target_season": keys[1],
            "window": keys[2],
            "strategy": keys[3],
            "n_picks": int(len(grp)),
            "mean_regret": float(regret.mean()),
            "median_regret": float(regret.median()),
            "p90_regret": float(regret.quantile(0.90)),
            "zero_regret_rate": float(grp["zero_regret"].mean()),
            "mean_gain_vs_adp": float(grp["gain_vs_adp"].mean()),
        })
    for keys, grp in df.groupby(
        ["source_season", "target_season", "window", "strategy", "adp_band"], observed=True
    ):
        regret = grp["regret"].astype(float)
        summaries.append({
            "source_season": keys[0],
            "target_season": keys[1],
            "window": keys[2],
            "strategy": keys[3],
            "adp_band": keys[4],
            "breakdown": "adp_band",
            "n_picks": int(len(grp)),
            "mean_regret": float(regret.mean()),
            "median_regret": float(regret.median()),
            "p90_regret": float(regret.quantile(0.90)),
            "zero_regret_rate": float(grp["zero_regret"].mean()),
            "mean_gain_vs_adp": float(grp["gain_vs_adp"].mean()),
        })
    return summaries


def evaluate_fold(
    source_season: int,
    target_season: int,
    *,
    consensus_dir: Path | None = None,
    market_blend: float = 0.35,
) -> dict[str, Any]:
    """Run one rolling-origin fold and return metric tables plus evaluation rows."""
    rows, summary, metadata = run_evaluation(source_season, target_season)
    consensus_dir = consensus_dir or (Path(REPO_ROOT) / "data" / "consensus")
    consensus_path = consensus_dir / f"consensus_{target_season}.json"
    consensus_meta: dict[str, Any] = {}
    market_curves = None
    calibration_frame = None
    if consensus_path.exists():
        consensus, consensus_meta = load_consensus_snapshot(
            consensus_path, expected_season=target_season
        )
        calib_source_path = consensus_dir / f"consensus_{source_season}.json"
        if calib_source_path.exists():
            calib_consensus, _ = load_consensus_snapshot(
                calib_source_path, expected_season=source_season
            )
            calib_rows = rows.merge(
                calib_consensus[["player_id", "adp"]], on="player_id", how="left"
            )
            calib_rows["position"] = calib_rows["preseason_position"]
            calib_rows = calib_rows.dropna(subset=["adp", "actual_points"])
            if not calib_rows.empty:
                market_curves = fit_market_curves(calib_rows.rename(columns={"actual_points": "actual_points"}))
                calibration_frame = rows.merge(
                    consensus[["player_id", "adp"]], on="player_id", how="left"
                )
    enriched = attach_forecast_family_points(
        rows,
        calibration_frame=calibration_frame,
        market_curves=market_curves,
        market_blend=market_blend,
    )
    if "prior_team" not in enriched.columns and "team" in rows.columns:
        enriched["prior_team"] = rows.get("team")
    prior_source = rows if "age" not in rows.columns else rows
    if "age" not in enriched.columns and "age" in prior_source.columns:
        enriched["age"] = prior_source["age"]

    metric_rows: list[dict[str, Any]] = []
    tier_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    family_points = {
        "pure_model": "pure_model_points",
        "market_informed": "market_informed_points",
        "adp": "adp_points",
    }
    for family, col in family_points.items():
        if family == "adp" and enriched["adp_points"].isna().all():
            continue
        metric_rows.extend(
            top_n_precision_recall_rows(
                enriched,
                source_season=source_season,
                target_season=target_season,
                forecast_family=family,
                points_col=col,
            )
        )
        tier_rows.extend(
            tier_calibration_rows(
                enriched,
                source_season=source_season,
                target_season=target_season,
                forecast_family=family,
                points_col=col,
            )
        )
        segment_rows.extend(
            segment_metric_rows(
                enriched,
                source_season=source_season,
                target_season=target_season,
                forecast_family=family,
                points_col=col,
            )
        )

    matched = pd.DataFrame()
    regret_detail: list[dict[str, Any]] = []
    regret_summary: list[dict[str, Any]] = []
    if consensus_path.exists() and not consensus.empty:
        board = enriched.rename(columns={"preseason_position": "position"})
        matched = matched_market_frame(
            board,
            consensus,
            model_points_col="pure_model_points",
            actual_points_col="actual_points",
            max_market_rank=int(TOP_ADP),
        )
        if not matched.empty:
            matched = matched.merge(
                enriched[
                    ["player_id", "pure_model_points", "market_informed_points", "adp"]
                ],
                on="player_id",
                how="left",
            )
            for window in REGRET_WINDOWS:
                regret_detail.extend(
                    adp_choice_regret_rows(
                        matched,
                        source_season=source_season,
                        target_season=target_season,
                        window=window,
                    )
                )
            regret_summary = summarize_regret(regret_detail)

    return {
        "source_season": int(source_season),
        "target_season": int(target_season),
        "metadata": metadata,
        "fantasy_eval_summary": summary,
        "evaluation_rows": enriched,
        "top_n_metrics": pd.DataFrame(metric_rows),
        "tier_metrics": pd.DataFrame(tier_rows),
        "segment_metrics": pd.DataFrame(segment_rows),
        "regret_detail": pd.DataFrame(regret_detail),
        "regret_summary": pd.DataFrame(regret_summary),
        "matched_market": matched,
        "consensus_meta": consensus_meta,
        "market_snapshot_timestamp": consensus_meta.get("as_of"),
    }


def evaluate_rolling_folds(
    folds: tuple[int, ...] = DEFAULT_FOLDS,
    **kwargs: Any,
) -> dict[str, Any]:
    """Evaluate all folds and concatenate metric tables."""
    fold_results = []
    for target in folds:
        source = target - 1
        fold_results.append(evaluate_fold(source, target, **kwargs))
    return aggregate_fold_results(fold_results)


def aggregate_fold_results(fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    def _concat(key: str) -> pd.DataFrame:
        parts = [r[key] for r in fold_results if isinstance(r.get(key), pd.DataFrame) and not r[key].empty]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    rows = pd.concat(
        [r["evaluation_rows"] for r in fold_results if isinstance(r.get("evaluation_rows"), pd.DataFrame)],
        ignore_index=True,
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "folds": [
            {
                "source_season": r["source_season"],
                "target_season": r["target_season"],
                "market_snapshot_timestamp": r.get("market_snapshot_timestamp"),
            }
            for r in fold_results
        ],
        "evaluation_rows": rows,
        "top_n_metrics": _concat("top_n_metrics"),
        "tier_metrics": _concat("tier_metrics"),
        "segment_metrics": _concat("segment_metrics"),
        "regret_detail": _concat("regret_detail"),
        "regret_summary": _concat("regret_summary"),
        "contract_hashes": vorp_tier_contract_hashes(),
        "fold_results": fold_results,
    }


def write_evidence_bundle(
    payload: dict[str, Any],
    *,
    bundle_id: str,
    output_dir: Path | None = None,
) -> Path:
    """Write versioned E1 evidence artifacts and manifest."""
    root = Path(output_dir or DECISION_QUALITY_DIR) / bundle_id
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    sha256: dict[str, str] = {}

    rows = payload.get("evaluation_rows")
    if isinstance(rows, pd.DataFrame) and not rows.empty:
        path = root / "evaluation_rows.parquet"
        rows.to_parquet(path, index=False)
        paths["evaluation_rows.parquet"] = str(path.relative_to(REPO_ROOT)).replace("\\", "/")
        sha256["evaluation_rows.parquet"] = sha256_file(path)

    for name in ("top_n_metrics", "tier_metrics", "segment_metrics", "regret_detail", "regret_summary"):
        frame = payload.get(name)
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            path = root / f"{name}.parquet"
            frame.to_parquet(path, index=False)
            rel = str(path.relative_to(REPO_ROOT)).replace("\\", "/")
            paths[name + ".parquet"] = rel
            sha256[name + ".parquet"] = sha256_file(path)

    manifest = {
        "bundle_id": bundle_id,
        "bundle_type": "decision_quality_evidence",
        "generated_at": payload.get("generated_at") or datetime.now(timezone.utc).isoformat(),
        "folds": payload.get("folds") or [],
        "contract_hashes": payload.get("contract_hashes") or vorp_tier_contract_hashes(),
        "source_paths": paths,
        "sha256": sha256,
        "frozen_baseline_id": payload.get("frozen_baseline_id"),
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["sha256"]["manifest.json"] = sha256_file(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path
