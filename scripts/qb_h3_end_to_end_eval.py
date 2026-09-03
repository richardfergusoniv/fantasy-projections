#!/usr/bin/env python3
"""H3 end-to-end QB experiment evaluation vs sealed model_points_end_to_end.

Terminology: 2025 is the latest chronological OOS fold (not a pristine holdout).
2026 is the prospective holdout (diagnostics only).

Does not retune frozen H1/H2 thresholds. Does not create a release bundle.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.qb_sealed_baseline_bakeoff import build_history
from src.projection.qb_active_archetype.thresholds import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    EVAL_SEASONS,
    FIT_SEASONS,
    GATES,
    HOLDOUT_SEASON,
    MIN_EVAL_GAMES,
    thresholds_dict,
)
from src.projection.qb_h3.forecast import predict_h3
from src.projection.qb_h3.composition_contract import detect_double_availability

OUT = ROOT / "output" / "qb_h3"
RNG = np.random.default_rng(BOOTSTRAP_SEED)

# Same frozen gates — applied vs sealed e2e. 2025 labeled latest OOS fold.
LATEST_OOS_FOLD = HOLDOUT_SEASON  # 2025


def _dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _mae(y, p):
    return float(np.mean(np.abs(np.asarray(p, float) - np.asarray(y, float))))


def _spearman(y, p):
    if len(y) < 3:
        return float("nan")
    return float(pd.Series(y).corr(pd.Series(p), method="spearman"))


def evaluate_season(history: pd.DataFrame, season: int) -> dict:
    ev = pd.read_csv(ROOT / "output" / f"fantasy_evaluation_{season}.csv")
    qb = ev[ev.preseason_position.astype(str).eq("QB")].copy()
    qb = qb[pd.to_numeric(qb.actual_games_played, errors="coerce").fillna(0) >= MIN_EVAL_GAMES]
    rows = []
    for _, r in qb.iterrows():
        pid = str(r["player_id"])
        hist = history[(history.player_id.astype(str) == pid) & (history.season < season)]
        if hist.empty:
            continue
        assert int(hist.season.max()) < season
        pred = predict_h3(history, player_id=pid, target_season=season)
        if not pred["ok"]:
            continue
        # Double-availability check
        da = detect_double_availability(
            pred["rates_per_active"]["attempts"],
            pred["effective_starts"],
            pred["season_stats"]["attempts"],
        )
        actual_pts = float(r["actual_points"])
        actual_g = float(r["actual_games_played"])
        sealed = float(r["model_points_end_to_end"]) if pd.notna(r["model_points_end_to_end"]) else float(
            r["model_forecast_points"]
        )
        sealed_ppg = float(r["model_rate_points"]) if pd.notna(r["model_rate_points"]) else np.nan
        prior = history[
            (history.player_id.astype(str) == pid)
            & (history.season < season)
            & (history.season >= season - 4)
        ]
        last = prior.sort_values("season").iloc[-1] if not prior.empty else None
        earlier = prior[prior.season < last.season] if last is not None else prior
        returning = bool(
            last is not None
            and float(last.get("active_starts") or 0) < 12
            and (earlier.empty or float(earlier["active_starts"].max()) >= 14)
        )
        rows.append(
            {
                "player_id": pid,
                "display_name": r.get("display_name"),
                "actual_points": actual_pts,
                "actual_ppg": actual_pts / actual_g,
                "actual_games": actual_g,
                "actual_attempts_pa": float(r["attempts"]) / actual_g,
                "actual_carries_pa": float(r["carries"]) / actual_g,
                "sealed_points": sealed,
                "sealed_ppg": sealed_ppg,
                "h3_points": pred["expected_season_points"],
                "h3_pp_active": pred["points_per_active_start"],
                "h3_avail_adj_ppg": pred["availability_adjusted_ppg"],
                "h3_expected_starts": pred["availability"]["expected_active_starts"],
                "h3_attempts_pa": pred["rates_per_active"]["attempts"],
                "h3_carries_pa": pred["rates_per_active"]["carries"],
                "archetype": pred["archetype"],
                "returning_injury": returning,
                "double_avail_once": da["matches_once"],
                "depth_tier": float(r["depth_tier"]) if pd.notna(r.get("depth_tier")) else np.nan,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {"season": season, "n": 0}
    # Depth filter for starter-relevant overall: still report all, but note backups
    frame = frame.sort_values("actual_points", ascending=False).reset_index(drop=True)
    frame["actual_rank"] = frame.index + 1

    def cohort(sub):
        if sub.empty:
            return {}
        return {
            "n": int(len(sub)),
            "sealed_points_mae": _mae(sub.actual_points, sub.sealed_points),
            "h3_points_mae": _mae(sub.actual_points, sub.h3_points),
            "delta_points_mae": _mae(sub.actual_points, sub.h3_points)
            - _mae(sub.actual_points, sub.sealed_points),
            "sealed_ppg_mae": _mae(sub.actual_ppg, sub.sealed_ppg.fillna(sub.actual_ppg)),
            "h3_pp_active_mae": _mae(sub.actual_ppg, sub.h3_pp_active),
            "sealed_spearman": _spearman(sub.actual_points, sub.sealed_points),
            "h3_spearman": _spearman(sub.actual_points, sub.h3_points),
            "h3_attempts_mae": _mae(sub.actual_attempts_pa, sub.h3_attempts_pa.fillna(sub.actual_attempts_pa)),
            "h3_carries_mae": _mae(sub.actual_carries_pa, sub.h3_carries_pa.fillna(sub.actual_carries_pa)),
            "h3_starts_mae": _mae(sub.actual_games, sub.h3_expected_starts),
            "double_avail_violations": int((~sub.double_avail_once).sum()),
        }

    masks = {
        "all": pd.Series(True, index=frame.index),
        "depth1": frame.depth_tier.fillna(99).eq(1),
        "dual_threat": frame.archetype.isin(["designed_runner", "mobile_scrambler"]),
        "pocket_passer": frame.archetype.eq("pocket_passer"),
        "insufficient_history": frame.archetype.eq("insufficient_history"),
        "returning_injury": frame.returning_injury.astype(bool),
        "primary": frame.archetype.isin(["designed_runner", "mobile_scrambler"])
        | frame.returning_injury.astype(bool),
        "top12_actual": frame.actual_rank <= 12,
    }
    metrics = {
        "season": season,
        "n": int(len(frame)),
        "fold_label": "latest_chronological_oos" if season == LATEST_OOS_FOLD else "fit",
        "cohorts": {k: cohort(frame[m]) for k, m in masks.items()},
    }
    primary = frame[masks["primary"]]
    if len(primary) >= 5:
        y, s, c = primary.actual_points.to_numpy(), primary.sealed_points.to_numpy(), primary.h3_points.to_numpy()
        deltas = []
        n = len(primary)
        for _ in range(BOOTSTRAP_DRAWS):
            idx = RNG.integers(0, n, n)
            deltas.append(_mae(y[idx], c[idx]) - _mae(y[idx], s[idx]))
        deltas = np.asarray(deltas)
        metrics["primary_bootstrap_delta_mae"] = {
            "mean": float(deltas.mean()),
            "ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
        }
    # Per-player error contribution
    frame["ae_delta"] = (frame.h3_points - frame.actual_points).abs() - (
        frame.sealed_points - frame.actual_points
    ).abs()
    metrics["top10_negative"] = frame.nlargest(10, "ae_delta")[
        ["display_name", "actual_points", "sealed_points", "h3_points", "ae_delta", "archetype"]
    ].to_dict("records")
    metrics["top10_positive"] = frame.nsmallest(10, "ae_delta")[
        ["display_name", "actual_points", "sealed_points", "h3_points", "ae_delta", "archetype"]
    ].to_dict("records")
    _dump(OUT / f"fold_{season}_rows.json", {"season": season, "rows": frame.to_dict("records")})
    return metrics


def decide(folds: list[dict]) -> dict:
    reasons: list[str] = []
    gates: dict = {}
    fit = [f for f in folds if f.get("season") in FIT_SEASONS and f.get("n")]
    latest = next((f for f in folds if f.get("season") == LATEST_OOS_FOLD), None)

    def c(f, name="all"):
        return (f.get("cohorts") or {}).get(name) or {}

    # Hard fail: sealed leakage-safe feature→train→reconcile→compose path unavailable.
    db_present = (ROOT / "data" / "projections.db").exists()
    gates["sealed_pipeline_injection"] = bool(db_present)
    if not db_present:
        reasons.append(
            "sealed_leakage_safe_refit_blocked_without_projections_db"
        )

    # No material single-season regression like 2023 +5.11 vs sealed e2e.
    for f in fit + ([latest] if latest else []):
        ac = c(f, "all")
        if not ac:
            continue
        sealed_mae = float(ac.get("sealed_points_mae") or 0.0)
        delta = float(ac.get("delta_points_mae") or 0.0)
        tol = sealed_mae * GATES.overall_mae_non_inferiority_tol
        if delta > tol + 1e-9:
            reasons.append(
                f"overall_mae_regression_season_{f['season']}_delta_{delta:.2f}"
            )
        if delta >= 5.0:
            reasons.append(
                f"material_single_season_regression_like_2023_season_{f['season']}_delta_{delta:.2f}"
            )
        if int(ac.get("double_avail_violations") or 0) > 0:
            reasons.append(f"double_availability_violations_season_{f['season']}")

    gates["no_material_single_season_regression"] = not any(
        "material_single_season_regression" in r or "overall_mae_regression" in r
        for r in reasons
    )
    gates["overall_non_inferiority"] = all(
        (not c(f, "all"))
        or c(f, "all")["h3_points_mae"]
        <= c(f, "all")["sealed_points_mae"] * (1.0 + GATES.overall_mae_non_inferiority_tol)
        for f in fit + ([latest] if latest else [])
    )
    if not gates["overall_non_inferiority"] and not any(
        "overall_mae_regression" in r for r in reasons
    ):
        reasons.append("overall_non_inferiority_failed")

    improved = sum(
        1
        for f in fit
        if c(f, "primary") and c(f, "primary").get("delta_points_mae", 1) < 0
    )
    gates["primary_cohort_fit_folds"] = improved >= GATES.cohort_improve_min_fit_folds
    if not gates["primary_cohort_fit_folds"]:
        reasons.append(f"primary_cohort_improved_on_{improved}_of_{len(fit)}_fit_folds")

    latest_ok = latest_ci = spearman_ok = top12_ok = False
    if latest and latest.get("n"):
        pc, ac, t12 = c(latest, "primary"), c(latest, "all"), c(latest, "top12_actual")
        latest_ok = bool(pc) and pc.get("delta_points_mae", 1) < 0
        ci = (latest.get("primary_bootstrap_delta_mae") or {}).get("ci95") or [0, 1]
        latest_ci = ci[1] < 0
        if ac:
            spearman_ok = ac["h3_spearman"] >= ac["sealed_spearman"] - GATES.spearman_max_drop
        if t12:
            top12_ok = t12["h3_points_mae"] <= t12["sealed_points_mae"] * (
                1 + GATES.top12_mae_non_inferiority_tol
            )
        if GATES.holdout_cohort_must_improve and not latest_ok:
            reasons.append("latest_oos_primary_cohort_did_not_improve")
        if GATES.holdout_bootstrap_ci_must_exclude_zero and not latest_ci:
            reasons.append("latest_oos_primary_bootstrap_ci_includes_zero")
        if not spearman_ok:
            reasons.append("latest_oos_spearman_dropped")
        if not top12_ok:
            reasons.append("latest_oos_top12_regressed")
    gates["latest_oos_primary_improve"] = latest_ok
    gates["latest_oos_bootstrap_ci"] = latest_ci
    gates["latest_oos_spearman"] = spearman_ok
    gates["latest_oos_top12"] = top12_ok
    gates["ensemble_weights"] = "unchanged_sealed_weights_only_no_nested_reweight_attempted"
    gates["pipeline_note"] = (
        "H3 forecasts are experimental avail×opportunity×efficiency→FP compared to "
        "sealed model_points_end_to_end. True sealed feature→train→reconcile→compose "
        "injection requires projections.db (absent). Ensemble weights unchanged; "
        "no nested weight selection run."
    )
    gates["non_qb_projection_changes"] = 0
    gates["team_volume_conservation"] = (
        "unit_contract_proven; full_board_conservation_requires_sealed_team_reconcile"
    )

    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    reasons = uniq

    verdict = "GO FOR CANDIDATE PACKAGING" if not reasons else "NO-GO"
    failing = {
        "stages": [
            "sealed_leakage_safe_feature_retrain_blocked_without_projections_db",
            "experimental_h3_e2e_vs_sealed_e2e_points_only",
        ],
        "cohorts": list(reasons),
    }
    return {
        "verdict": verdict,
        "reasons": reasons,
        "gates": gates,
        "failing_stage_and_cohorts": failing if verdict == "NO-GO" else None,
        "terminology": {
            "2025": "latest_chronological_oos_fold",
            "2026": "prospective_holdout_diagnostics_only",
            "not_pristine_holdout": True,
        },
        "ensemble": "sealed_weights_unchanged",
        "next_hypothesis": None
        if verdict.startswith("GO")
        else (
            "NO-GO: do not start H4 automatically. Failing stage is sealed-path "
            "injection without projections.db and/or material MAE regressions vs "
            "model_points_end_to_end. Identify cohorts in fold_*_rows.json."
        ),
    }


def sanity_2026(history: pd.DataFrame) -> dict:
    focus_ids = {
        "Josh Allen": "00-0034857",
        "Lamar Jackson": "00-0034796",
        "Jayden Daniels": "00-0039910",
        "Jalen Hurts": "00-0036389",
        "Joe Burrow": "00-0036442",
        "Patrick Mahomes": "00-0033873",
    }
    sealed = pd.read_csv(
        ROOT / "draft_assistant/data/releases/v2_baseline_20260830/projections_2026.csv"
    )
    qb_att = sealed[(sealed.position == "QB") & (sealed.stat == "attempts")].copy()
    if "depth_rank" in qb_att.columns:
        starters = qb_att[pd.to_numeric(qb_att.depth_rank, errors="coerce").eq(1.0)]
    else:
        starters = qb_att
    starter_ids = set(starters.player_id.astype(str).unique()) | set(focus_ids.values())
    name_map = {
        str(r.player_id): r.get("display_name")
        for _, r in qb_att.drop_duplicates("player_id").iterrows()
    }

    fp = pd.read_csv(ROOT / "output/accuracy_first_2026/fantasy_points_2026.csv")
    fp_qb = (
        fp[fp.position.astype(str).eq("QB")]
        .sort_values("fantasy_pts_season", ascending=False)
        .reset_index(drop=True)
    )
    fp_qb["rank"] = fp_qb.index + 1
    sealed_pts = {str(r.player_id): float(r.fantasy_pts_season) for _, r in fp_qb.iterrows()}
    sealed_rank = {str(r.player_id): int(r["rank"]) for _, r in fp_qb.iterrows()}

    rows = []
    for pid in sorted(starter_ids):
        pred = predict_h3(history, player_id=pid, target_season=2026)
        if not pred["ok"]:
            continue
        rows.append(
            {
                "player_id": pid,
                "player": name_map.get(pid),
                "expected_starts": pred["availability"]["expected_active_starts"],
                "points_per_active_start": pred["points_per_active_start"],
                "expected_season_points": pred["expected_season_points"],
                "availability_adjusted_ppg": pred["availability_adjusted_ppg"],
                "attempts_per_active_start": pred["rates_per_active"]["attempts"],
                "expected_season_attempts": pred["season_stats"]["attempts"],
                "designed_carries_per_active": pred["efficiency"].get(
                    "designed_carries_per_active"
                ),
                "scrambles_per_active_start": pred["efficiency"].get(
                    "scrambles_per_active_start"
                ),
                "carries_per_active_start": pred["rates_per_active"]["carries"],
                "expected_season_carries": pred["season_stats"]["carries"],
                "passing_tds_season": pred["season_stats"].get("passing_tds"),
                "rushing_tds_season": pred["season_stats"].get("rushing_tds"),
                "sealed_final_points": sealed_pts.get(pid),
                "experimental_final_points": pred["expected_season_points"],
                "difference": (
                    pred["expected_season_points"] - sealed_pts[pid]
                    if pid in sealed_pts
                    else None
                ),
                "sealed_rank": sealed_rank.get(pid),
                "archetype": pred["archetype"],
            }
        )
    table = (
        pd.DataFrame(rows)
        .sort_values("expected_season_points", ascending=False)
        .reset_index(drop=True)
    )
    table["final_qb_rank_experimental"] = table.index + 1
    focus = {}
    for name, pid in focus_ids.items():
        sub = table[table.player_id == pid]
        focus[name] = sub.iloc[0].to_dict() if len(sub) else {"missing": True, "player_id": pid}
    qb12 = table.iloc[11].to_dict() if len(table) >= 12 else None
    return {
        "n_starters": int(len(table)),
        "focus": focus,
        "qb12_boundary": qb12,
        "starters": table.to_dict("records"),
        "note": (
            "Prospective holdout diagnostics only; not used for GO/NO-GO. "
            "Trained through permitted 2025 cutoff via seasons < 2026 history."
        ),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    _dump(
        OUT / "run_meta.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "thresholds_frozen": thresholds_dict(),
            "h1_h2_retune": False,
            "projections_db_present": (ROOT / "data" / "projections.db").exists(),
            "sealed_refit_available": False,
        },
    )
    history = build_history()
    folds = [evaluate_season(history, s) for s in EVAL_SEASONS]
    # strip bulky nested from summary dump already in fold files
    summary_folds = []
    for f in folds:
        summary_folds.append(
            {k: v for k, v in f.items() if k not in ("top10_negative", "top10_positive")}
            | {
                "top10_negative": f.get("top10_negative"),
                "top10_positive": f.get("top10_positive"),
            }
        )
    decision = decide(folds)
    sanity = sanity_2026(history)
    _dump(
        OUT / "selection_decision.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "decision": decision,
            "folds": summary_folds,
        },
    )
    _dump(OUT / "sanity_2026.json", sanity)
    print("verdict", decision["verdict"])
    print("reasons", decision["reasons"])
    for f in folds:
        ac = (f.get("cohorts") or {}).get("all") or {}
        print(
            f"season {f['season']} ({f.get('fold_label')}): "
            f"sealed={ac.get('sealed_points_mae')} h3={ac.get('h3_points_mae')} "
            f"delta={ac.get('delta_points_mae')}"
        )
    return 0 if str(decision["verdict"]).startswith("GO") else 2


if __name__ == "__main__":
    raise SystemExit(main())
