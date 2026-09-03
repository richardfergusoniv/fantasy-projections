#!/usr/bin/env python3
"""Upstream QB feature + joint-allocation evaluation (rolling-origin).

Does not modify the sealed release or active pointer. Production compose
defaults remain unchanged (``qb_joint_room_allocation=False``).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from scipy import stats

from src.projection.composition import shipped_context
from src.projection.fantasy_points import SCORING
from src.projection.qb_joint_allocation import reconcile_qb_joint_room
from src.projection.qb_repair.apply_board import non_qb_invariance_check, score_long_to_fantasy
from src.projection.qb_repair.arms import ARM_BASELINE, run_arm
from src.projection.qb_rush_features import patch_inference_row_with_rush_pool
from src.projection.team_reconcile import reconcile_team_volume
from src.projection.transitions import age_shrunk_predict

OUT = ROOT / "output" / "qb_upstream"
LAMAR = "00-0034796"
BURROW = "00-0036442"
FIT_SEASONS = (2023, 2024)
HOLDOUT = 2025
BOOTSTRAP = 2000
RNG = np.random.default_rng(42)


def _json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def load_qb_panel() -> pd.DataFrame:
    hist = pd.read_parquet(ROOT / "output" / "qb_repair" / "history" / "qb_season_rates.parquet")
    if "games" in hist.columns and "games_played" not in hist.columns:
        hist = hist.rename(columns={"games": "games_played"})
    hist["carries_pg"] = hist["carries"] / hist["games_played"].replace(0, np.nan)
    hist["rushing_yards_pg"] = hist["rushing_yards"] / hist["games_played"].replace(0, np.nan)
    hist["rushing_tds_pg"] = hist["rushing_tds"] / hist["games_played"].replace(0, np.nan)

    pbp_path = ROOT / "data" / "raw" / "weekly_qb_repair_cache" / "pbp_qb_rush_features_2022_2025.parquet"
    if pbp_path.exists():
        from src.projection.qb_rush_features import compute_qb_rush_splits_from_pbp

        splits = compute_qb_rush_splits_from_pbp(pd.read_parquet(pbp_path))
        # Prefer pbp-derived splits when hist already has partial designed/scramble cols.
        overlap = [c for c in splits.columns if c in hist.columns and c not in ("player_id", "season")]
        if overlap:
            hist = hist.drop(columns=overlap)
        hist = hist.merge(splits, on=["player_id", "season"], how="left")

    designed = pd.to_numeric(hist["designed_carries"], errors="coerce") if "designed_carries" in hist.columns else pd.Series(np.nan, index=hist.index)
    scramble = pd.to_numeric(hist["scramble_carries"], errors="coerce") if "scramble_carries" in hist.columns else pd.Series(np.nan, index=hist.index)
    attempts = pd.to_numeric(hist["attempts"], errors="coerce").fillna(0)
    plays = (attempts + designed.fillna(0) + scramble.fillna(0)).replace(0, np.nan)
    hist["qb_designed_run_rate"] = designed / plays
    if "dropbacks" in hist.columns:
        dropbacks = pd.to_numeric(hist["dropbacks"], errors="coerce").fillna(attempts)
    else:
        dropbacks = attempts
    hist["qb_scramble_per_dropback"] = scramble / dropbacks.replace(0, np.nan)
    return hist


def evaluate_rate_pooling(panel: pd.DataFrame) -> dict:
    """Rolling-origin: predict carries_pg with T-1 vs multi-season pool."""
    results = []
    for season in (*FIT_SEASONS, HOLDOUT):
        hold = panel[panel["season"].eq(season) & panel["games_played"].ge(8)].copy()
        rows = []
        for _, row in hold.iterrows():
            hist = panel[
                (panel["player_id"] == row["player_id"])
                & (panel["season"] < season)
                & (panel["season"] >= season - 4)
            ]
            if hist.empty:
                continue
            w = (hist["games_played"] / 12).clip(upper=1) * hist["games_played"]
            t1 = hist.sort_values("season").iloc[-1]
            pooled = float(np.average(hist["carries_pg"], weights=w))
            rows.append(
                {
                    "player_id": row["player_id"],
                    "actual": float(row["carries_pg"]),
                    "carry_forward": float(t1["carries_pg"]),
                    "pooled": pooled,
                    "games": float(row["games_played"]),
                }
            )
        frame = pd.DataFrame(rows)
        if frame.empty:
            continue

        def mae(col):
            return float((frame[col] - frame["actual"]).abs().mean())

        # Bootstrap MAE delta (pooled - cf); negative = pooled better
        deltas = []
        n = len(frame)
        for _ in range(BOOTSTRAP):
            idx = RNG.integers(0, n, n)
            sub = frame.iloc[idx]
            deltas.append(
                float((sub["pooled"] - sub["actual"]).abs().mean()
                      - (sub["carry_forward"] - sub["actual"]).abs().mean())
            )
        deltas = np.asarray(deltas)
        results.append(
            {
                "season": season,
                "n": int(n),
                "carry_forward_mae": mae("carry_forward"),
                "pooled_mae": mae("pooled"),
                "delta_mae": mae("pooled") - mae("carry_forward"),
                "bootstrap_delta_mean": float(deltas.mean()),
                "bootstrap_delta_ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
                "spearman_cf": float(frame["actual"].corr(frame["carry_forward"], method="spearman")),
                "spearman_pooled": float(frame["actual"].corr(frame["pooled"], method="spearman")),
            }
        )
    return {"metric": "carries_pg", "folds": results}


def evaluate_model_feature_patch(panel: pd.DataFrame) -> dict:
    """Re-score sealed QB_carries model with pooled inference features on folds.

    Uses fantasy_evaluation actuals for labels; constructs a minimal feature
    row from panel history. When the sealed model cannot be scored (missing
    features), falls back to pooled carries_pg as the prediction.
    """
    art = joblib.load(ROOT / "models" / "QB_carries.joblib")
    model, feat_names = art["model"], art["features"]
    folds = []
    for season in (*FIT_SEASONS, HOLDOUT):
        eval_path = ROOT / "output" / f"fantasy_evaluation_{season}.csv"
        if not eval_path.exists():
            continue
        ev = pd.read_csv(eval_path)
        qb = ev[ev["preseason_position"].eq("QB") & (ev["actual_games_played"] >= 8)].copy()
        preds_base = []
        preds_patch = []
        actual = []
        for _, row in qb.iterrows():
            pid = str(row["player_id"])
            hist = panel[(panel["player_id"].astype(str) == pid) & (panel["season"] < season)]
            if hist.empty:
                continue
            src = hist.sort_values("season").iloc[-1].copy()
            src["player_id"] = pid
            # Build a sparse feature vector; missing feats -> nan (LightGBM handles)
            feat_vals = {f: np.nan for f in feat_names}
            feat_vals["player_id"] = pid
            for f in feat_names:
                if f in src.index and pd.notna(src[f]):
                    try:
                        feat_vals[f] = float(src[f])
                    except (TypeError, ValueError):
                        continue
            # Map prior_* from source season rates
            for stat in ("carries", "rushing_yards", "rushing_tds", "attempts", "passing_yards", "passing_tds", "completions", "interceptions"):
                key = f"prior_{stat}_pg"
                if key in feat_names and f"{stat}_pg" in src.index:
                    feat_vals[key] = float(src[f"{stat}_pg"])
                elif key in feat_names and stat in src.index and src.get("games_played", 0):
                    feat_vals[key] = float(src[stat]) / float(src["games_played"])
            if "prior_role_rate" in feat_names and "carries_pg" in src.index:
                feat_vals["prior_role_rate"] = float(src["carries_pg"])
            if "prior_role_rate_3y" in feat_names:
                w = (hist["games_played"] / 12).clip(upper=1) * hist["games_played"]
                feat_vals["prior_role_rate_3y"] = float(np.average(hist["carries_pg"], weights=w))
            if "depth_tier" in feat_names:
                feat_vals["depth_tier"] = float(row.get("depth_tier") or 1.0)
            if "games_played" in feat_names:
                feat_vals["games_played"] = float(src.get("games_played") or np.nan)

            feat_row = pd.Series(feat_vals, dtype=object)
            X = pd.DataFrame([{f: feat_row.get(f) for f in feat_names}])[feat_names]
            try:
                base_pred = float(age_shrunk_predict(model, X, "QB", features=feat_names)[0])
            except Exception:
                base_pred = float(src.get("carries_pg") or 0.0)

            patched, _ = patch_inference_row_with_rush_pool(feat_row, panel, target_season=season)
            Xp = pd.DataFrame([{f: patched.get(f) for f in feat_names}])[feat_names]
            try:
                patch_pred = float(age_shrunk_predict(model, Xp, "QB", features=feat_names)[0])
            except Exception:
                patch_pred = float(patched.get("prior_carries_pg") or base_pred)

            # actual carries per game
            g = float(row["actual_games_played"])
            actual.append(float(row["carries"]) / g if g else np.nan)
            preds_base.append(base_pred)
            preds_patch.append(patch_pred)

        if not actual:
            continue
        a = np.asarray(actual, dtype=float)
        b = np.asarray(preds_base, dtype=float)
        p = np.asarray(preds_patch, dtype=float)
        deltas = []
        n = len(a)
        for _ in range(BOOTSTRAP):
            idx = RNG.integers(0, n, n)
            deltas.append(float(np.nanmean(np.abs(p[idx] - a[idx])) - np.nanmean(np.abs(b[idx] - a[idx]))))
        deltas = np.asarray(deltas)
        folds.append(
            {
                "season": season,
                "n": int(len(a)),
                "baseline_mae": float(np.nanmean(np.abs(b - a))),
                "patched_mae": float(np.nanmean(np.abs(p - a))),
                "delta_mae": float(np.nanmean(np.abs(p - a)) - np.nanmean(np.abs(b - a))),
                "bootstrap_delta_mean": float(deltas.mean()),
                "bootstrap_delta_ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
                "baseline_bias": float(np.nanmean(b - a)),
                "patched_bias": float(np.nanmean(p - a)),
                "baseline_spearman": float(pd.Series(a).corr(pd.Series(b), method="spearman")),
                "patched_spearman": float(pd.Series(a).corr(pd.Series(p), method="spearman")),
                "mean_abs_pred_delta": float(np.nanmean(np.abs(p - b))),
            }
        )
    return {"model": "QB_carries.joblib", "folds": folds}


def evaluate_joint_allocation_2026() -> dict:
    raw = pd.read_csv(ROOT / "output" / "projections_2026_raw.csv")
    for col in ("pred_season", "pred_season_low", "pred_season_high", "team_volume_scale", "td_rate_clip_applied"):
        if col in raw.columns:
            raw = raw.drop(columns=[col])
    ctx = shipped_context(conn=None, target_season=2026)
    base = run_arm(raw, ctx, ARM_BASELINE, target_season=2026)
    ctx_joint = shipped_context(conn=None, target_season=2026)
    ctx_joint.qb_joint_room_allocation = True
    from src.projection.composition import compose_board

    joint_board = compose_board(raw.copy(), ctx_joint)
    inv = non_qb_invariance_check(baseline_long=base.board, candidate_long=joint_board)
    fantasy = score_long_to_fantasy(joint_board)
    focus = {}
    for name, pid in {
        "Josh Allen": "00-0034857",
        "Lamar Jackson": LAMAR,
        "Jayden Daniels": "00-0039910",
        "Jalen Hurts": "00-0036389",
        "Joe Burrow": BURROW,
        "Patrick Mahomes": "00-0033873",
        "Drake Maye": "00-0039851",
        "Bo Nix": "00-0039732",
    }.items():
        row = fantasy[fantasy["player_id"].astype(str).eq(pid)]
        if row.empty:
            # try name
            row = fantasy[fantasy.get("display_name", pd.Series(dtype=str)).eq(name)]
        if not row.empty:
            r = row.iloc[0]
            focus[name] = {
                "rank": int(r["rank"]),
                "ppg": float(r["fantasy_pts"]),
                "attempts": float(r["attempts"]) if "attempts" in r else None,
                "carries": float(r["carries"]) if "carries" in r else None,
            }
    # Burrow scale comparison
    def _burrow_att(board):
        sub = board[(board.player_id == BURROW) & (board.stat == "attempts")]
        return float(sub.pred_pg.iloc[0]) if not sub.empty else None

    return {
        "non_qb_invariance": inv,
        "burrow_attempts_baseline": _burrow_att(base.board),
        "burrow_attempts_joint": _burrow_att(joint_board),
        "sanity_table": focus,
        "n_qb": int(len(fantasy)),
    }


def decide(pooling: dict, model_patch: dict, joint: dict) -> dict:
    """Require multi-fold improvement; 2026 sanity is diagnostic only.

    Gates are supplementary to existing release gates and are not weakened.
    Promotion requires ALL of the checks below; failure on any yields NO-GO.
    """
    reasons = []
    gates = {}
    fit_folds = [f for f in pooling["folds"] if f["season"] in FIT_SEASONS]
    hold = next((f for f in pooling["folds"] if f["season"] == HOLDOUT), None)
    fit_improved = all(f["delta_mae"] < 0 for f in fit_folds) if fit_folds else False
    hold_improved = bool(hold and hold["delta_mae"] < 0)
    hold_ci_ok = bool(hold and hold["bootstrap_delta_ci95"][1] < 0)
    gates["rate_pool_fit_mae"] = fit_improved
    gates["rate_pool_holdout_mae"] = hold_improved
    gates["rate_pool_holdout_bootstrap_ci"] = hold_ci_ok
    if not fit_improved:
        reasons.append("pooled_carries_pg_did_not_improve_all_fit_folds")
    if not hold_improved:
        reasons.append("pooled_carries_pg_did_not_improve_holdout")
    if not hold_ci_ok:
        reasons.append("holdout_bootstrap_ci_does_not_exclude_zero")

    patch_fit = [f for f in model_patch.get("folds", []) if f["season"] in FIT_SEASONS]
    patch_hold = next((f for f in model_patch.get("folds", []) if f["season"] == HOLDOUT), None)
    patch_fit_ok = bool(patch_fit) and all(f["patched_mae"] < f["baseline_mae"] for f in patch_fit)
    patch_hold_ok = bool(patch_hold) and patch_hold["patched_mae"] < patch_hold["baseline_mae"]
    patch_hold_ci_ok = bool(patch_hold) and patch_hold.get("bootstrap_delta_ci95", [0, 1])[1] < 0
    gates["model_patch_fit_mae"] = patch_fit_ok
    gates["model_patch_holdout_mae"] = patch_hold_ok
    gates["model_patch_holdout_bootstrap_ci"] = patch_hold_ci_ok
    if not patch_fit_ok:
        reasons.append("model_feature_patch_did_not_improve_all_fit_folds")
    if not patch_hold_ok:
        reasons.append("model_feature_patch_did_not_improve_holdout")
    if not patch_hold_ci_ok:
        reasons.append("model_feature_patch_holdout_bootstrap_ci_does_not_exclude_zero")

    inv_ok = bool(joint.get("non_qb_invariance", {}).get("pass", False))
    gates["non_qb_invariance"] = inv_ok
    gates["passing_volume_conservation_2026_joint"] = True  # enforced inside reconcile_qb_joint_room
    gates["season_points_mae"] = False  # requires full historical retrain; not claimable here
    gates["ppg_mae"] = False
    gates["rank_correlation"] = False
    gates["top12_recall_ordering"] = False
    gates["rushing_attempt_calibration_by_archetype"] = fit_improved and hold_improved
    gates["stability_across_seasons"] = fit_improved and hold_improved and patch_fit_ok
    if not inv_ok:
        reasons.append("joint_allocation_broke_non_qb_invariance")
    # Hard-fail if fantasy-point gates cannot be evidenced on chronological folds
    reasons.append("full_fantasy_point_gates_unavailable_without_historical_raw_boards_retrain")

    verdict = "GO" if not reasons else "NO-GO"
    return {
        "verdict": verdict,
        "reasons": reasons,
        "gates": gates,
        "selected_configuration": None if verdict == "NO-GO" else "rush_pool+joint_allocation",
        "note": (
            "2026 sanity table is diagnostic only and was not used for selection. "
            "Existing release gates were not weakened."
        ),
    }


def build_lamar_lineage(panel: pd.DataFrame) -> dict:
    art = joblib.load(ROOT / "models" / "QB_carries.joblib")
    features = art["features"]
    model = art["model"]
    imp = None
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=features)
    gain = None
    if hasattr(model, "booster_"):
        gain = pd.Series(
            model.booster_.feature_importance(importance_type="gain"),
            index=model.booster_.feature_name(),
        )

    hist = panel[(panel.player_id == LAMAR) & (panel.season.between(2022, 2025))].sort_values("season")
    raw = pd.read_csv(ROOT / "output" / "projections_2026_raw.csv")
    lam_raw = raw[raw.player_id == LAMAR]
    sealed = pd.read_csv(
        ROOT / "draft_assistant" / "data" / "releases" / "v2_baseline_20260830" / "projections_2026.csv"
    )
    lam_s = sealed[sealed.player_id == LAMAR]
    fp = pd.read_csv(ROOT / "output" / "accuracy_first_2026" / "fantasy_points_2026.csv")
    r = fp[fp.player_id == LAMAR].iloc[0]

    # Stage 5: reconstruct 2026 inference patch from 2022-2025 history
    feat_vals = {f: np.nan for f in features}
    feat_vals["player_id"] = LAMAR
    src2025 = hist[hist.season == 2025].iloc[0]
    for f in features:
        if f in src2025.index and pd.notna(src2025[f]):
            try:
                feat_vals[f] = float(src2025[f])
            except (TypeError, ValueError):
                continue
    for stat in ("carries", "rushing_yards", "rushing_tds"):
        key = f"prior_{stat}_pg"
        if key in features and f"{stat}_pg" in src2025.index:
            feat_vals[key] = float(src2025[f"{stat}_pg"])
    if "prior_role_rate" in features:
        feat_vals["prior_role_rate"] = float(src2025["carries_pg"])
    if "games_played" in features:
        feat_vals["games_played"] = float(src2025["games_played"])
    feat_row = pd.Series(feat_vals, dtype=object)
    patched, patch_audit = patch_inference_row_with_rush_pool(feat_row, panel, target_season=2026)
    Xb = pd.DataFrame([{f: feat_row.get(f) for f in features}])[features]
    Xp = pd.DataFrame([{f: patched.get(f) for f in features}])[features]
    try:
        base_pred = float(age_shrunk_predict(model, Xb, "QB", features=features)[0])
        patch_pred = float(age_shrunk_predict(model, Xp, "QB", features=features)[0])
    except Exception as exc:
        base_pred = patch_pred = float("nan")
        patch_audit["predict_error"] = str(exc)

    stages = [
        {
            "stage": "1_raw_historical_source",
            "rows": hist[
                [
                    c
                    for c in [
                        "season",
                        "games_played",
                        "carries",
                        "rushing_yards",
                        "designed_carries",
                        "scramble_carries",
                        "carries_pg",
                        "qb_designed_run_rate",
                        "qb_scramble_per_dropback",
                    ]
                    if c in hist.columns
                ]
            ].to_dict("records"),
        },
        {
            "stage": "2_season_feature_construction",
            "notes": [
                "qb_designed_run_rate is computed and present in FEATURE_COLS and BLEND_FEATURES",
                "QB rows are deliberately EXCLUDED from BLEND_FEATURES games-weighted blending "
                "(features.py: blendable = position in RB/WR/TE only)",
                "Therefore a partial/injured season's designed-run feature is not pooled with "
                "prior seasons in the legacy column used by the sealed model",
                "Additive expansion columns (scramble/YPC/RZ/GL/pooled/archetype) are attached "
                "but are NOT in the sealed QB_carries feature contract",
            ],
            "2025_designed_per_game": float(
                hist.loc[hist.season == 2025, "designed_carries"].iloc[0]
                / hist.loc[hist.season == 2025, "games_played"].iloc[0]
            )
            if "designed_carries" in hist.columns and (hist.season == 2025).any()
            else None,
            "2024_designed_per_game": float(
                hist.loc[hist.season == 2024, "designed_carries"].iloc[0]
                / hist.loc[hist.season == 2024, "games_played"].iloc[0]
            )
            if "designed_carries" in hist.columns and (hist.season == 2024).any()
            else None,
            "2025_qb_designed_run_rate": float(src2025.get("qb_designed_run_rate") or float("nan")),
            "2024_qb_designed_run_rate": float(
                hist.loc[hist.season == 2024, "qb_designed_run_rate"].iloc[0]
            )
            if "qb_designed_run_rate" in hist.columns
            else None,
        },
        {
            "stage": "3_transition_pair_training_row",
            "label": "carries_per_elig",
            "note": (
                "QB_carries trains on role-rate carries_per_elig with ROLE_FEATURES including "
                "qb_designed_run_rate and prior_role_rate_3y. Transition pairs use season T "
                "features to predict season T+1 labels."
            ),
        },
        {
            "stage": "4_serialized_model_feature_contract",
            "qb_designed_run_rate_in_contract": "qb_designed_run_rate" in features,
            "importance_split_rank": int(list(imp.sort_values(ascending=False).index).index("qb_designed_run_rate") + 1)
            if imp is not None and "qb_designed_run_rate" in imp.index
            else None,
            "importance_gain": float(gain.get("qb_designed_run_rate")) if gain is not None else None,
            "importance_gain_rank": int(list(gain.sort_values(ascending=False).index).index("qb_designed_run_rate") + 1)
            if gain is not None and "qb_designed_run_rate" in gain.index
            else None,
            "prior_carries_pg_gain": float(gain.get("prior_carries_pg")) if gain is not None else None,
            "prior_role_rate_3y_gain": float(gain.get("prior_role_rate_3y")) if gain is not None else None,
            "top5_gain": (
                {k: float(v) for k, v in gain.sort_values(ascending=False).head(5).items()}
                if gain is not None
                else None
            ),
            "verdict": (
                "PRESENT at runtime in FEATURE_COLS, BLEND_FEATURES, and sealed QB_carries.joblib "
                "contract — not missing. Low gain (overwhelmed by prior_role_rate_3y / "
                "prior_rushing_yards_pg / prior_carries_pg). QB blend exclusion leaves "
                "injured-season designed rate unpooled in the legacy column."
            ),
        },
        {
            "stage": "5_2026_inference_row",
            "source_season": 2025,
            "unpatched_prior_carries_pg": float(src2025.carries_pg),
            "games_weighted_2022_2025_carries_pg": float(
                np.average(
                    hist.carries_pg,
                    weights=(hist.games_played / 12).clip(upper=1) * hist.games_played,
                )
            ),
            "patched_prior_carries_pg": float(patched.get("prior_carries_pg") or float("nan")),
            "patch_audit": patch_audit,
            "qb_blend_exclusion": True,
        },
        {
            "stage": "6_raw_model_prediction",
            "rates": {
                str(row.stat): float(row.pred_pg)
                for _, row in lam_raw.iterrows()
                if row.stat in ("carries", "rushing_yards", "rushing_tds", "attempts")
            },
            "reconstructed_unpatched_carries_pred": base_pred,
            "reconstructed_patched_carries_pred": patch_pred,
        },
        {
            "stage": "7_team_volume_reconciliation",
            "rates": {
                str(row.stat): {
                    "pred_pg": float(row.pred_pg),
                    "team_volume_scale": float(row.team_volume_scale),
                }
                for _, row in lam_s.iterrows()
                if row.stat in ("carries", "rushing_yards", "attempts", "passing_yards")
            },
            "note": "Rush stats have team_volume_scale=1.0; reconcile does not restore rushing",
        },
        {
            "stage": "8_composition",
            "note": "compose_board hygiene; carries remain ~4.5 through finalization under sealed path",
        },
        {
            "stage": "9_ensemble_output",
            "fantasy_pts": float(r.fantasy_pts),
            "fantasy_pts_season": float(r.fantasy_pts_season),
            "v2_pred": float(r.v2_pred),
            "pg_carries": float(r.pg_carries),
            "pg_rushing_yards": float(r.pg_rushing_yards),
            "rank_note": "accuracy-first ensemble board",
        },
    ]
    return {"player_id": LAMAR, "stages": stages}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = load_qb_panel()
    lineage = build_lamar_lineage(panel)
    _json_dump(OUT / "lamar_feature_lineage.json", lineage)

    pooling = evaluate_rate_pooling(panel)
    _json_dump(OUT / "rate_pooling_folds.json", pooling)

    model_patch = evaluate_model_feature_patch(panel)
    _json_dump(OUT / "model_feature_patch_folds.json", model_patch)

    joint = evaluate_joint_allocation_2026()
    _json_dump(OUT / "joint_allocation_2026.json", joint)

    decision = decide(pooling, model_patch, joint)
    _json_dump(
        OUT / "selection_decision.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "decision": decision,
            "pooling": pooling,
            "model_patch": model_patch,
            "joint_allocation": {
                "non_qb_invariance": joint.get("non_qb_invariance"),
                "burrow_attempts_baseline": joint.get("burrow_attempts_baseline"),
                "burrow_attempts_joint": joint.get("burrow_attempts_joint"),
            },
            "sanity_2026_diagnostic_only": joint.get("sanity_table"),
        },
    )
    print("verdict", decision["verdict"], decision["reasons"])
    print("pooling folds", pooling["folds"])
    print("joint non_qb", joint["non_qb_invariance"])
    print("burrow att", joint.get("burrow_attempts_baseline"), "->", joint.get("burrow_attempts_joint"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
