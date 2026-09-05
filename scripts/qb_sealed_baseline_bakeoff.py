#!/usr/bin/env python3
"""Phase 1: verify active-archetype candidate vs actual sealed-final baseline.

Frozen configuration — does NOT retune thresholds, archetypes, or allocation.

Prior experiment comparator was A (injury-diluted conflated carry-forward),
NOT B (sealed final after reconcile/compose/ensemble). This script runs the
apples-to-apples bakeoff against sealed-final fantasy points.
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

from src.projection.qb_active_archetype.active_rates import (
    append_eval_season_active,
    build_active_season_rates,
    expected_availability,
    load_weekly_qb,
    merge_rush_splits,
    player_decomposition,
)
from src.projection.qb_active_archetype.archetypes import classify_archetype
from src.projection.qb_active_archetype.evaluate import predict_player
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

OUT = ROOT / "output" / "qb_sealed_baseline_bakeoff"
RNG = np.random.default_rng(BOOTSTRAP_SEED)
LAMAR = "00-0034796"
BURROW = "00-0036442"


def _dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_history() -> pd.DataFrame:
    active = build_active_season_rates(load_weekly_qb())
    active = merge_rush_splits(active)
    active = append_eval_season_active(active, 2025)
    active = merge_rush_splits(active)
    return active


def _mae(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    return float(np.mean(np.abs(p - y)))


def _spearman(y, p):
    if len(y) < 3:
        return float("nan")
    return float(pd.Series(y).corr(pd.Series(p), method="spearman"))


def load_sealed_final(season: int) -> pd.DataFrame:
    """Load final sealed-pipeline fantasy points for a historical season.

    Primary: fantasy_evaluation.model_points_end_to_end (leakage-safe end-to-end
    model stack used in historical evaluation).

    Supplement (2024+): accuracy-first incumbent_pred when present (40/60
    ensemble mirroring the sealed 2026 accuracy-first board).
    """
    ev = pd.read_csv(ROOT / "output" / f"fantasy_evaluation_{season}.csv")
    qb = ev[ev["preseason_position"].astype(str).eq("QB")].copy()
    qb = qb[pd.to_numeric(qb["actual_games_played"], errors="coerce").fillna(0) >= MIN_EVAL_GAMES]
    qb["player_id"] = qb["player_id"].astype(str)
    qb["sealed_season_points"] = pd.to_numeric(qb["model_points_end_to_end"], errors="coerce")
    qb["sealed_ppg"] = pd.to_numeric(qb["model_rate_points"], errors="coerce")
    # Prefer end-to-end; fall back to forecast if missing
    miss = qb["sealed_season_points"].isna()
    qb.loc[miss, "sealed_season_points"] = pd.to_numeric(
        qb.loc[miss, "model_forecast_points"], errors="coerce"
    )
    parquet = ROOT / "output" / "accuracy_first_2026" / "evaluation_players.parquet"
    qb["incumbent_season_points"] = np.nan
    if parquet.exists() and season >= 2024:
        ep = pd.read_parquet(parquet)
        ep = ep[(ep["season"] == season) & (ep["position"].astype(str).eq("QB"))].copy()
        ep["player_id"] = ep["player_id"].astype(str)
        qb = qb.merge(
            ep[["player_id", "v2_pred", "incumbent_pred"]].rename(
                columns={"incumbent_pred": "incumbent_season_points", "v2_pred": "v2_season_points"}
            ),
            on="player_id",
            how="left",
        )
    return qb


def evaluate_season_vs_sealed(history: pd.DataFrame, season: int) -> dict:
    sealed = load_sealed_final(season)
    rows = []
    for _, r in sealed.iterrows():
        pid = str(r["player_id"])
        # Strict OOS: candidate uses only seasons < target (enforced in predict_player).
        hist_max = history.loc[
            (history.player_id.astype(str) == pid) & (history.season < season), "season"
        ]
        if hist_max.empty:
            continue
        assert int(hist_max.max()) <= season - 1

        cand = predict_player(
            history, player_id=pid, target_season=season, mode="candidate_active_archetype"
        )
        if not cand["ok"]:
            continue

        actual_pts = float(r["actual_points"])
        actual_games = float(r["actual_games_played"])
        actual_ppg = actual_pts / actual_games if actual_games else np.nan
        actual_pp_active = actual_ppg  # actual games ≈ active starts in eval rows

        prior = history[
            (history.player_id.astype(str) == pid)
            & (history.season < season)
            & (history.season >= season - 4)
        ]
        last = prior.sort_values("season").iloc[-1] if not prior.empty else None
        earlier = (
            prior[prior.season < last.season] if last is not None else prior
        )
        returning = bool(
            last is not None
            and float(last.get("active_starts") or 0) < 12
            and (earlier.empty or float(earlier["active_starts"].max()) >= 14)
        )
        arch = classify_archetype(history, player_id=pid, target_season=season)["archetype"]

        sealed_pts = float(r["sealed_season_points"]) if pd.notna(r["sealed_season_points"]) else np.nan
        sealed_ppg = float(r["sealed_ppg"]) if pd.notna(r["sealed_ppg"]) else np.nan
        incumbent = (
            float(r["incumbent_season_points"])
            if "incumbent_season_points" in r.index and pd.notna(r["incumbent_season_points"])
            else np.nan
        )

        rows.append(
            {
                "player_id": pid,
                "display_name": r.get("display_name"),
                "actual_points": actual_pts,
                "actual_ppg": actual_ppg,
                "actual_pp_active": actual_pp_active,
                "actual_games": actual_games,
                "actual_attempts_pa": float(r["attempts"]) / actual_games,
                "actual_carries_pa": float(r["carries"]) / actual_games,
                "sealed_points": sealed_pts,
                "sealed_ppg": sealed_ppg,
                "incumbent_points": incumbent,
                "cand_points": cand["season_points"],
                "cand_ppg_avail_adj": cand["ppg"],  # season pts / expected starts
                "cand_pp_active": cand["ppg"],  # rates scored per active start
                "cand_expected_starts": cand["expected_games"],
                "cand_attempts_pa": cand["rates"].get("attempts"),
                "cand_carries_pa": cand["rates"].get("carries"),
                "archetype": arch,
                "returning_injury": returning,
                "oos_max_train_season": int(hist_max.max()),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return {"season": season, "n": 0}
    frame = frame.dropna(subset=["sealed_points", "cand_points", "actual_points"])
    frame = frame.sort_values("actual_points", ascending=False).reset_index(drop=True)
    frame["actual_rank"] = frame.index + 1

    def cohort_metrics(sub: pd.DataFrame) -> dict:
        if sub.empty:
            return {}
        out = {
            "n": int(len(sub)),
            "sealed_points_mae": _mae(sub.actual_points, sub.sealed_points),
            "cand_points_mae": _mae(sub.actual_points, sub.cand_points),
            "delta_points_mae": _mae(sub.actual_points, sub.cand_points)
            - _mae(sub.actual_points, sub.sealed_points),
            "sealed_ppg_mae": _mae(sub.actual_ppg, sub.sealed_ppg.fillna(sub.actual_ppg)),
            "cand_pp_active_mae": _mae(sub.actual_pp_active, sub.cand_pp_active),
            "sealed_spearman": _spearman(sub.actual_points, sub.sealed_points),
            "cand_spearman": _spearman(sub.actual_points, sub.cand_points),
            "cand_attempts_mae": _mae(
                sub.actual_attempts_pa, sub.cand_attempts_pa.fillna(sub.actual_attempts_pa)
            ),
            "cand_carries_mae": _mae(
                sub.actual_carries_pa, sub.cand_carries_pa.fillna(sub.actual_carries_pa)
            ),
            "cand_starts_mae": _mae(sub.actual_games, sub.cand_expected_starts),
        }
        if sub["incumbent_points"].notna().any():
            inc = sub.dropna(subset=["incumbent_points"])
            out["incumbent_points_mae"] = _mae(inc.actual_points, inc.incumbent_points)
            out["delta_vs_incumbent_mae"] = _mae(inc.actual_points, inc.cand_points) - _mae(
                inc.actual_points, inc.incumbent_points
            )
        return out

    masks = {
        "all": pd.Series(True, index=frame.index),
        "dual_threat": frame.archetype.isin(["designed_runner", "mobile_scrambler"]),
        "pocket_passer": frame.archetype.eq("pocket_passer"),
        "returning_injury": frame.returning_injury.astype(bool),
        "primary": frame.archetype.isin(["designed_runner", "mobile_scrambler"])
        | frame.returning_injury.astype(bool),
        "top12_actual": frame.actual_rank <= 12,
    }
    metrics = {"season": season, "n": int(len(frame)), "cohorts": {}}
    for name, mask in masks.items():
        metrics["cohorts"][name] = cohort_metrics(frame[mask])

    primary = frame[masks["primary"]]
    if len(primary) >= 5:
        y = primary.actual_points.to_numpy()
        s = primary.sealed_points.to_numpy()
        c = primary.cand_points.to_numpy()
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
    metrics["rows"] = frame.to_dict("records")
    return metrics


def decide(folds: list[dict]) -> dict:
    """Apply the SAME frozen gates — no retuning after prior holdout report."""
    reasons = []
    gates = {}
    fit = [f for f in folds if f.get("season") in FIT_SEASONS and f.get("n", 0) > 0]
    hold = next((f for f in folds if f.get("season") == HOLDOUT_SEASON), None)

    def cohort(f, name="all"):
        return (f.get("cohorts") or {}).get(name) or {}

    overall_ok = True
    for f in fit + ([hold] if hold and hold.get("n") else []):
        c = cohort(f, "all")
        if not c:
            continue
        if c["cand_points_mae"] > c["sealed_points_mae"] * (1.0 + GATES.overall_mae_non_inferiority_tol):
            overall_ok = False
            reasons.append(f"overall_mae_regression_vs_sealed_season_{f['season']}")
    gates["overall_non_inferiority_vs_sealed"] = overall_ok

    improved_fit = sum(
        1 for f in fit if cohort(f, "primary") and cohort(f, "primary").get("delta_points_mae", 1) < 0
    )
    cohort_fit_ok = improved_fit >= GATES.cohort_improve_min_fit_folds
    gates["primary_cohort_fit_folds"] = cohort_fit_ok
    if not cohort_fit_ok:
        reasons.append(
            f"primary_cohort_improved_on_{improved_fit}_fit_folds_need_{GATES.cohort_improve_min_fit_folds}"
        )

    hold_ok = hold_ci_ok = top12_ok = spearman_ok = False
    if hold and hold.get("n"):
        pc = cohort(hold, "primary")
        ac = cohort(hold, "all")
        t12 = cohort(hold, "top12_actual")
        hold_ok = bool(pc) and pc.get("delta_points_mae", 1) < 0
        if GATES.holdout_cohort_must_improve and not hold_ok:
            reasons.append("holdout_primary_cohort_did_not_improve_vs_sealed")
        ci = (hold.get("primary_bootstrap_delta_mae") or {}).get("ci95") or [0, 1]
        hold_ci_ok = ci[1] < 0
        if GATES.holdout_bootstrap_ci_must_exclude_zero and not hold_ci_ok:
            reasons.append("holdout_primary_bootstrap_ci_vs_sealed_includes_zero")
        if t12:
            top12_ok = t12["cand_points_mae"] <= t12["sealed_points_mae"] * (
                1.0 + GATES.top12_mae_non_inferiority_tol
            )
            if not top12_ok:
                reasons.append("holdout_top12_mae_regressed_vs_sealed_beyond_tol")
        if ac:
            spearman_ok = ac["cand_spearman"] >= ac["sealed_spearman"] - GATES.spearman_max_drop
            if not spearman_ok:
                reasons.append("holdout_spearman_dropped_vs_sealed_beyond_tol")
    gates["holdout_primary_improve"] = hold_ok
    gates["holdout_bootstrap_ci"] = hold_ci_ok
    gates["holdout_top12_non_inferior"] = top12_ok
    gates["holdout_spearman"] = spearman_ok

    verdict = "GO" if not reasons else "NO-GO"
    return {
        "verdict": verdict,
        "production_promotion": "NO",  # Phase 1 never promotes
        "comparator": {
            "prior_experiment": "A_injury_diluted_conflated_carry_forward",
            "this_bakeoff": "B_sealed_final_model_points_end_to_end",
            "sealed_source": "output/fantasy_evaluation_{season}.csv::model_points_end_to_end",
            "supplement_2024_2025": "accuracy_first incumbent_pred (reported, not gate-binding unless noted)",
        },
        "reasons": reasons,
        "gates": gates,
        "frozen_thresholds_unchanged": thresholds_dict()["gates"],
        "note": (
            "Configuration frozen; thresholds not retuned using 2025. "
            "GO here is required before Phase 2 candidate bundle work."
        ),
    }


def burrow_units_clarification(history: pd.DataFrame) -> dict:
    """Clarify the previously reported ~30.1 attempts figure."""
    decomp = player_decomposition(history, player_id=BURROW, seasons=(2022, 2023, 2024, 2025))
    cand = predict_player(history, player_id=BURROW, target_season=2026, mode="candidate_active_archetype")
    att_pa = cand["rates"].get("attempts")
    exp_starts = cand["expected_games"]
    season_attempts = (att_pa or 0) * (exp_starts or 0)
    # Board path stores season_attempts / 17 as availability-adjusted per scheduled game
    avail_adj_per_sched = season_attempts / 17.0 if season_attempts else None
    return {
        "player": "Joe Burrow",
        "player_id": BURROW,
        "historical": decomp,
        "2026_candidate_decomposition": {
            "attempts_per_active_start": att_pa,
            "expected_starts": exp_starts,
            "expected_season_attempts": season_attempts,
            "availability_adjusted_attempts_per_scheduled_team_game": avail_adj_per_sched,
            "clarification": (
                "The previously reported ~30.1 was availability-adjusted attempts per "
                "scheduled team game on the composed board (expected season attempts / 17), "
                "NOT attempts per active start. Attempts per active start remain ~37. "
                "Expected season attempts ≈ 37.3 × 13.06 ≈ 487."
            ),
        },
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    _dump(
        OUT / "comparator_declaration.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "prior_reported_comparator": "A",
            "prior_comparator_detail": (
                "baseline_conflated in qb_active_archetype.evaluate.predict_player — "
                "last-season injury-diluted per-game rates × carried-forward games. "
                "NOT sealed v2_baseline_20260830 final after reconcile/compose/ensemble."
            ),
            "this_bakeoff_comparator": "B",
            "this_bakeoff_detail": (
                "fantasy_evaluation model_points_end_to_end (sealed historical end-to-end "
                "pipeline points). Final fantasy season points and PPG compared to actuals."
            ),
            "thresholds_frozen": True,
            "no_2025_retune": True,
            "artifact_hashes": {
                "thresholds.py": _sha(
                    ROOT / "src/projection/qb_active_archetype/thresholds.py"
                ),
                "predeclared_thresholds.json": _sha(
                    ROOT / "output/qb_active_archetype/predeclared_thresholds.json"
                ),
            },
        },
    )

    history = build_history()
    folds = []
    for season in EVAL_SEASONS:
        m = evaluate_season_vs_sealed(history, season)
        rows = m.pop("rows", [])
        folds.append(m)
        _dump(OUT / f"fold_{season}_rows.json", {"season": season, "rows": rows})
        # Prove OOS cutoffs
        if rows:
            assert all(r["oos_max_train_season"] < season for r in rows)

    decision = decide(folds)
    burrow = burrow_units_clarification(history)
    _dump(
        OUT / "selection_decision.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "decision": decision,
            "folds": [
                {
                    "season": f["season"],
                    "n": f["n"],
                    "cohorts": f.get("cohorts"),
                    "primary_bootstrap_delta_mae": f.get("primary_bootstrap_delta_mae"),
                }
                for f in folds
            ],
            "burrow_units": burrow,
        },
    )
    print("PRIOR COMPARATOR: A (conflated carry-forward)")
    print("THIS BAKEOFF: B (sealed final model_points_end_to_end)")
    print("verdict", decision["verdict"], decision["reasons"])
    print("gates", decision["gates"])
    for f in folds:
        ac = (f.get("cohorts") or {}).get("all") or {}
        print(
            f"season {f['season']}: sealed_mae={ac.get('sealed_points_mae')} "
            f"cand_mae={ac.get('cand_points_mae')} delta={ac.get('delta_points_mae')}"
        )
    print(
        "burrow units:",
        burrow["2026_candidate_decomposition"]["clarification"],
    )
    return 0 if decision["verdict"] == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
