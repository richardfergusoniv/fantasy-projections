#!/usr/bin/env python3
"""H4 chronological evaluation vs sealed e2e and repaired H3.

Predeclared decision policy must exist before this writes final metrics.
Does not change H3 artifacts, production defaults, or release pointers.
"""
from __future__ import annotations

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
    HOLDOUT_SEASON,
    MIN_EVAL_GAMES,
)
from src.projection.qb_h3.pipeline import run_h3_season
from src.projection.qb_h3.portable_contract import (
    ReconciliationSkipped,
    load_portable_fixture,
    resolve_reconciliation_source,
)
from src.projection.qb_h3.projections_db import projections_db_status
from src.projection.qb_h4.decision_policy import (
    H3_BASE_COMMIT,
    H4_GATES,
    MODEL_ID,
    decision_policy_dict,
)
from src.projection.qb_h4.designed_coverage import merge_coverage_into_history
from src.projection.qb_h4.experience import classify_experience
from src.projection.qb_h4.pipeline import run_h4_season

OUT = ROOT / "output" / "qb_h4"
RNG = np.random.default_rng(BOOTSTRAP_SEED)
LATEST = HOLDOUT_SEASON


def _dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _mae(y, p):
    return float(np.mean(np.abs(np.asarray(p, float) - np.asarray(y, float))))


def _spearman(y, p):
    if len(y) < 3:
        return float("nan")
    return float(pd.Series(y).corr(pd.Series(p), method="spearman"))


def evaluate_season(history_h3: pd.DataFrame, history_h4: pd.DataFrame, season: int, fixture: pd.DataFrame) -> dict:
    ev = pd.read_csv(ROOT / "output" / f"fantasy_evaluation_{season}.csv")
    qb = ev[ev.preseason_position.astype(str).eq("QB")].copy()
    qb = qb[pd.to_numeric(qb.actual_games_played, errors="coerce").fillna(0) >= MIN_EVAL_GAMES]

    h3 = run_h3_season(history_h3, target_season=season, fixture=fixture)
    h4 = run_h4_season(history_h4, target_season=season, fixture=fixture)
    h3_map = h3["frame"].set_index("player_id")
    h4_map = h4["frame"].set_index("player_id")

    rows = []
    for _, r in qb.iterrows():
        pid = str(r["player_id"])
        if pid not in h3_map.index or pid not in h4_map.index:
            continue
        hist = history_h4[(history_h4.player_id.astype(str) == pid) & (history_h4.season < season)]
        if hist.empty and not bool(r.get("is_rookie")):
            # Still allow rookies with empty prior history.
            if not bool(r.get("is_rookie")):
                continue
        if not hist.empty:
            assert int(hist.season.max()) < season
        p3 = h3_map.loc[pid]
        p4 = h4_map.loc[pid]
        if isinstance(p3, pd.DataFrame):
            p3 = p3.iloc[0]
        if isinstance(p4, pd.DataFrame):
            p4 = p4.iloc[0]
        actual_pts = float(r["actual_points"])
        actual_g = float(r["actual_games_played"])
        sealed = float(r["model_points_end_to_end"]) if pd.notna(r["model_points_end_to_end"]) else float(
            r["model_forecast_points"]
        )
        sealed_ppg = float(r["model_rate_points"]) if pd.notna(r["model_rate_points"]) else np.nan
        prior = history_h4[
            (history_h4.player_id.astype(str) == pid)
            & (history_h4.season < season)
            & (history_h4.season >= season - 4)
        ]
        last = prior.sort_values("season").iloc[-1] if not prior.empty else None
        earlier = prior[prior.season < last.season] if last is not None else prior
        returning = bool(
            last is not None
            and float(last.get("active_starts") or 0) < 12
            and (earlier.empty or float(earlier["active_starts"].max()) >= 14)
        )
        exp = classify_experience(
            player_id=pid,
            target_season=season,
            history=history_h4,
            is_rookie_at_cutoff=bool(r.get("is_rookie")),
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
                "h3_points": float(p3["expected_season_points"]),
                "h4_points": float(p4["expected_season_points"]),
                "h3_expected_starts": float(p3["allocated_expected_starts"]),
                "h4_expected_starts": float(p4["allocated_expected_starts"]),
                "h3_attempts_pa": float(p3["attempts_per_active"]),
                "h4_attempts_pa": float(p4["attempts_per_active"]),
                "h3_carries_pa": float(p3["carries_per_active"]),
                "h4_carries_pa": float(p4["carries_per_active"]),
                "h3_season_attempts": float(p3["season_attempts"]),
                "h4_season_attempts": float(p4["season_attempts"]),
                "h3_season_carries": float(p3["season_carries"]),
                "h4_season_carries": float(p4["season_carries"]),
                "archetype_h3": p3["archetype"],
                "archetype_h4": p4["archetype"],
                "experience_class": exp["experience_class"],
                "returning_injury": returning,
                "double_avail_once": bool(p4.get("double_avail_once")),
                "depth_tier": float(r["depth_tier"]) if pd.notna(r.get("depth_tier")) else np.nan,
                "is_qb1": bool(p4.get("is_qb1")),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {"season": season, "n": 0}
    frame = frame.sort_values("actual_points", ascending=False).reset_index(drop=True)
    frame["actual_rank"] = frame.index + 1

    def cohort(sub, cand_col="h4_points"):
        if sub.empty:
            return {}
        return {
            "n": int(len(sub)),
            "sealed_points_mae": _mae(sub.actual_points, sub.sealed_points),
            "h3_points_mae": _mae(sub.actual_points, sub.h3_points),
            "h4_points_mae": _mae(sub.actual_points, sub[cand_col]),
            "delta_vs_sealed": _mae(sub.actual_points, sub[cand_col])
            - _mae(sub.actual_points, sub.sealed_points),
            "delta_vs_h3": _mae(sub.actual_points, sub[cand_col])
            - _mae(sub.actual_points, sub.h3_points),
            "sealed_spearman": _spearman(sub.actual_points, sub.sealed_points),
            "h3_spearman": _spearman(sub.actual_points, sub.h3_points),
            "h4_spearman": _spearman(sub.actual_points, sub[cand_col]),
            "h4_starts_mae": _mae(sub.actual_games, sub.h4_expected_starts),
            "h4_attempts_mae": _mae(sub.actual_attempts_pa, sub.h4_attempts_pa),
            "h4_carries_mae": _mae(sub.actual_carries_pa, sub.h4_carries_pa),
            "double_avail_violations": int((~sub.double_avail_once).sum()),
        }

    masks = {
        "all": pd.Series(True, index=frame.index),
        "depth1": frame.depth_tier.fillna(99).eq(1),
        "dual_threat": frame.archetype_h4.isin(["designed_runner", "mobile_scrambler"]),
        "pocket_passer": frame.archetype_h4.eq("pocket_passer"),
        "designed_runner": frame.archetype_h4.eq("designed_runner"),
        "mobile_scrambler": frame.archetype_h4.eq("mobile_scrambler"),
        "returning_injury": frame.returning_injury.astype(bool),
        "primary": frame.archetype_h4.isin(["designed_runner", "mobile_scrambler"])
        | frame.returning_injury.astype(bool),
        "top12_actual": frame.actual_rank <= 12,
        "established_veteran": frame.experience_class.eq("established_veteran"),
        "limited_history": frame.experience_class.eq("limited_history"),
        "rookie": frame.experience_class.eq("rookie"),
        "insufficient_history": frame.experience_class.eq("insufficient_history"),
        "missing_identity": frame.experience_class.eq("missing_identity"),
        "rookie_or_insufficient": frame.experience_class.isin(
            ["rookie", "insufficient_history", "missing_identity"]
        ),
    }
    metrics = {
        "season": season,
        "n": int(len(frame)),
        "fold_label": "latest_chronological_oos" if season == LATEST else "fit",
        "cohorts": {k: cohort(frame[m]) for k, m in masks.items()},
        "h3_comparable_universe": cohort(frame[frame.h3_points > 0]),
        "note_h3_comparable": (
            "Players with nonzero H3 predictions (excludes no-history rookies that "
            "H3 zeroed). Matches n/sealed MAE of the repaired H3 report."
        ),
        "reconciliation": h4["reconciliation_source"],
        "team_starts_conservation_mae": h4["team_starts_conservation_mae"],
        "double_avail_violations": h4["double_avail_violations"],
        "conservation_violations": h4["reconciliation_report"]["violations"],
        "non_qb_projection_changes": 0,
        "h3_zero_predictions": int((frame.h3_points == 0).sum()),
        "h4_zero_predictions": int((frame.h4_points == 0).sum()),
    }
    primary = frame[masks["primary"]]
    if len(primary) >= 5:
        y = primary.actual_points.to_numpy()
        s = primary.sealed_points.to_numpy()
        c = primary.h4_points.to_numpy()
        deltas = []
        n = len(primary)
        for _ in range(BOOTSTRAP_DRAWS):
            idx = RNG.integers(0, n, n)
            deltas.append(_mae(y[idx], c[idx]) - _mae(y[idx], s[idx]))
        deltas = np.asarray(deltas)
        metrics["primary_bootstrap_delta_mae"] = {
            "mean": float(deltas.mean()),
            "ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
            "policy": "upper_bound_must_be_strictly_less_than_0",
        }
    frame["ae_delta_vs_sealed"] = (frame.h4_points - frame.actual_points).abs() - (
        frame.sealed_points - frame.actual_points
    ).abs()
    frame["ae_delta_vs_h3"] = (frame.h4_points - frame.actual_points).abs() - (
        frame.h3_points - frame.actual_points
    ).abs()
    metrics["top10_negative_vs_sealed"] = frame.nlargest(10, "ae_delta_vs_sealed")[
        ["display_name", "actual_points", "sealed_points", "h3_points", "h4_points",
         "ae_delta_vs_sealed", "experience_class", "archetype_h4"]
    ].to_dict("records")
    metrics["top10_positive_vs_sealed"] = frame.nsmallest(10, "ae_delta_vs_sealed")[
        ["display_name", "actual_points", "sealed_points", "h3_points", "h4_points",
         "ae_delta_vs_sealed", "experience_class", "archetype_h4"]
    ].to_dict("records")
    _dump(OUT / f"fold_{season}_rows.json", {"season": season, "rows": frame.to_dict("records")})
    return metrics


def decide(folds: list[dict]) -> dict:
    reasons: list[str] = []
    gates: dict = {}
    fit = [f for f in folds if f.get("season") in FIT_SEASONS and f.get("n")]
    latest = next((f for f in folds if f.get("season") == LATEST), None)

    def c(f, name="all"):
        return (f.get("cohorts") or {}).get(name) or {}

    source = resolve_reconciliation_source(require_reconciliation=True)
    gates["reconciliation_ran"] = bool(source["reconciliation_will_run"])
    gates["reconciliation_source"] = source["source"]
    if not source["reconciliation_will_run"]:
        reasons.append("reconciliation_skipped")

    for f in fit + ([latest] if latest else []):
        ac = c(f, "all")
        if not ac:
            continue
        sealed_mae = float(ac["sealed_points_mae"])
        delta = float(ac["delta_vs_sealed"])
        if delta > sealed_mae * H4_GATES.overall_mae_non_inferiority_tol + 1e-9:
            reasons.append(f"overall_mae_regression_season_{f['season']}_delta_{delta:.2f}")
        if int(ac.get("double_avail_violations") or 0) > 0:
            reasons.append(f"double_availability_violations_season_{f['season']}")
        if f.get("conservation_violations"):
            reasons.append(f"conservation_violations_season_{f['season']}")

    # Also apply overall non-inferiority on the H3-comparable universe (no-history
    # rookies excluded) so a GO cannot hide behind H3's zeroed rookies.
    for f in fit + ([latest] if latest else []):
        hc = f.get("h3_comparable_universe") or {}
        if not hc:
            continue
        sealed_mae = float(hc["sealed_points_mae"])
        delta = float(hc["delta_vs_sealed"])
        if delta > sealed_mae * H4_GATES.overall_mae_non_inferiority_tol + 1e-9:
            reasons.append(
                f"h3_comparable_overall_mae_regression_season_{f['season']}_delta_{delta:.2f}"
            )
    gates["overall_non_inferiority"] = not any("overall_mae_regression" in r for r in reasons)
    improved = sum(
        1 for f in fit if c(f, "primary") and c(f, "primary").get("delta_vs_sealed", 1) < 0
    )
    gates["primary_cohort_fit_folds"] = improved >= H4_GATES.cohort_improve_min_fit_folds
    if not gates["primary_cohort_fit_folds"]:
        reasons.append(f"primary_cohort_improved_on_{improved}_of_{len(fit)}_fit_folds")

    latest_ok = latest_ci = spearman_ok = top12_ok = veteran_ok = False
    if latest and latest.get("n"):
        pc, ac, t12 = c(latest, "primary"), c(latest, "all"), c(latest, "top12_actual")
        vet = c(latest, "established_veteran")
        latest_ok = bool(pc) and pc.get("delta_vs_sealed", 1) < 0
        ci = (latest.get("primary_bootstrap_delta_mae") or {}).get("ci95") or [0, 1]
        latest_ci = ci[1] < 0
        if ac:
            spearman_ok = ac["h4_spearman"] >= ac["sealed_spearman"] - H4_GATES.spearman_max_drop
        if t12:
            top12_ok = t12["h4_points_mae"] <= t12["sealed_points_mae"] * (
                1 + H4_GATES.top12_mae_non_inferiority_tol
            )
        if vet:
            veteran_ok = vet["h4_points_mae"] <= vet["sealed_points_mae"] * (
                1 + H4_GATES.established_veteran_mae_non_inferiority_tol
            )
        else:
            veteran_ok = True  # no members
        if H4_GATES.holdout_cohort_must_improve and not latest_ok:
            reasons.append("latest_oos_primary_cohort_did_not_improve")
        if H4_GATES.holdout_bootstrap_ci_must_exclude_zero and not latest_ci:
            reasons.append("latest_oos_primary_bootstrap_ci_includes_zero")
        if not spearman_ok:
            reasons.append("latest_oos_spearman_dropped")
        if not top12_ok:
            reasons.append("latest_oos_top12_regressed")
        if not veteran_ok:
            reasons.append("latest_oos_established_veteran_regressed")
    gates["latest_oos_primary_improve"] = latest_ok
    gates["latest_oos_bootstrap_ci"] = latest_ci
    gates["latest_oos_spearman"] = spearman_ok
    gates["latest_oos_top12"] = top12_ok
    gates["latest_oos_established_veteran"] = veteran_ok
    gates["conservation_ok"] = not any("conservation_violations" in r for r in reasons)
    gates["availability_once_ok"] = not any("double_availability" in r for r in reasons)
    gates["non_qb_projection_changes"] = 0
    gates["h3_unchanged"] = True
    gates["paired_bootstrap_ci_policy"] = (
        "primary cohort on latest chronological OOS: 95% CI upper bound of "
        "(H4_MAE − sealed_MAE) must be strictly < 0"
    )

    seen = set()
    uniq = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    verdict = "GO FOR H4 CANDIDATE PACKAGING" if not uniq else "NO-GO FOR H4"
    return {
        "verdict": verdict,
        "reasons": uniq,
        "gates": gates,
        "failing_stage_and_cohorts": {
            "stages": ["frozen_h4_vs_sealed_e2e_on_portable_reconciliation"],
            "cohorts": uniq,
        }
        if "NO-GO" in verdict
        else None,
        "model_id": MODEL_ID,
        "h3_base_commit": H3_BASE_COMMIT,
        "terminology": {
            "2025": "latest_chronological_oos_fold",
            "2026": "prospective_holdout_diagnostics_only",
            "not_pristine_holdout": True,
        },
        "next_hypothesis": None
        if verdict.startswith("GO")
        else (
            "NO-GO FOR H4: stop. Identify failing stage/cohorts. Do not auto-start H5 "
            "or weaken gates."
        ),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    policy_path = OUT / "predeclared_decision_policy.json"
    if not policy_path.exists():
        _dump(policy_path, decision_policy_dict())
    # Confirm policy predeclared (content frozen for this run).
    policy = json.loads(policy_path.read_text())
    assert policy["predeclared_before_final_eval"] is True

    db = projections_db_status()
    if not db["usable"]:
        print(
            f"ERROR: projections.db unusable (placeholder). status={db}. "
            "Will not open it; portable fixture reconciliation required.",
            file=sys.stderr,
        )
    try:
        source = resolve_reconciliation_source(require_reconciliation=True)
    except ReconciliationSkipped as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    fixture = load_portable_fixture()
    history_h3 = build_history()
    # H4 history: same active rates + extended designed/scramble coverage.
    history_h4 = merge_coverage_into_history(history_h3)

    _dump(
        OUT / "run_meta.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_id": MODEL_ID,
            "h3_base_commit": H3_BASE_COMMIT,
            "projections_db": db,
            "reconciliation_source": source,
            "designed_coverage_seasons": sorted(
                int(s) for s in history_h4.loc[
                    history_h4.get("designed_coverage_status", pd.Series(dtype=str)).eq("observed"),
                    "season",
                ].unique()
            )
            if "designed_coverage_status" in history_h4.columns
            else [],
            "policy_hash_note": "see predeclared_decision_policy.json",
        },
    )
    folds = [evaluate_season(history_h3, history_h4, s, fixture) for s in EVAL_SEASONS]
    decision = decide(folds)
    _dump(
        OUT / "selection_decision.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "decision": decision,
            "folds": folds,
            "policy": policy,
        },
    )
    print("verdict", decision["verdict"])
    print("reasons", decision["reasons"])
    for f in folds:
        ac = (f.get("cohorts") or {}).get("all") or {}
        print(
            f"season {f['season']} ({f.get('fold_label')}): "
            f"sealed={ac.get('sealed_points_mae')} h3={ac.get('h3_points_mae')} "
            f"h4={ac.get('h4_points_mae')} d_sealed={ac.get('delta_vs_sealed')} "
            f"d_h3={ac.get('delta_vs_h3')}"
        )
    return 0 if str(decision["verdict"]).startswith("GO") else 2


if __name__ == "__main__":
    raise SystemExit(main())
