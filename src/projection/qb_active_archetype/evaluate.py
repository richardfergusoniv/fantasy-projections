"""Chronological evaluation for the active-start / archetype candidate."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.projection.fantasy_points import SCORING
from src.projection.qb_active_archetype.active_rates import (
    expected_availability,
    pooled_active_rate,
)
from src.projection.qb_active_archetype.archetypes import (
    classify_archetype,
    hierarchical_rush_priors,
)
from src.projection.qb_active_archetype.thresholds import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    EVAL_SEASONS,
    FIT_SEASONS,
    GATES,
    HOLDOUT_SEASON,
    MIN_EVAL_GAMES,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
RNG = np.random.default_rng(BOOTSTRAP_SEED)


def _score_rates(rates: dict, games: float) -> float:
    ppg = 0.0
    for stat, pts in SCORING.items():
        if stat in rates and rates[stat] is not None and np.isfinite(rates[stat]):
            ppg += float(rates[stat]) * float(pts)
    return ppg * float(games)


def _load_eval(season: int) -> pd.DataFrame:
    path = REPO_ROOT / "output" / f"fantasy_evaluation_{season}.csv"
    if not path.exists():
        return pd.DataFrame()
    ev = pd.read_csv(path)
    qb = ev[ev["preseason_position"].astype(str).eq("QB")].copy()
    qb = qb[pd.to_numeric(qb["actual_games_played"], errors="coerce").fillna(0) >= MIN_EVAL_GAMES]
    return qb


def predict_player(
    history: pd.DataFrame,
    *,
    player_id: str,
    target_season: int,
    mode: str,
) -> dict:
    """mode: baseline_conflated | candidate_active_archetype"""
    hist = history[
        (history["player_id"].astype(str) == str(player_id))
        & (history["season"] < int(target_season))
    ]
    if hist.empty:
        return {"ok": False}
    if mode == "baseline_conflated":
        # Last season conflated per-game rates × last season games as availability proxy
        # shrunk toward league — mirrors injury-diluted rate carry-forward.
        last = hist.sort_values("season").iloc[-1]
        rates = {}
        for stat in (
            "attempts",
            "completions",
            "passing_yards",
            "passing_tds",
            "interceptions",
            "carries",
            "rushing_yards",
            "rushing_tds",
        ):
            col = f"{stat}_per_game_conflated"
            rates[stat] = float(last[col]) if col in last.index and pd.notna(last[col]) else None
        games = float(last.get("active_starts") or last.get("weeks_rostered_proxy") or 14.0)
        # Carry-forward availability (conflates injury)
        return {
            "ok": True,
            "rates": rates,
            "expected_games": games,
            "archetype": None,
            "season_points": _score_rates(rates, games),
            "ppg": _score_rates(rates, games) / games if games else None,
        }

    # Candidate: active rates + availability + archetype rush priors
    avail = expected_availability(history, player_id=player_id, target_season=target_season)
    rush = hierarchical_rush_priors(history, player_id=player_id, target_season=target_season)
    rates = {}
    for stat in ("attempts", "completions", "passing_yards", "passing_tds", "interceptions"):
        pooled = pooled_active_rate(
            history, player_id=player_id, target_season=target_season, rate_col=f"{stat}_per_active"
        )
        rates[stat] = pooled["value"]
    rates["carries"] = rush["priors"].get("carries_per_active")
    rates["rushing_yards"] = rush["priors"].get("rushing_yards_per_active")
    rates["rushing_tds"] = rush["priors"].get("rushing_tds_per_active")
    # Fall back to active pool if archetype prior missing
    for stat in ("carries", "rushing_yards", "rushing_tds"):
        if rates[stat] is None:
            rates[stat] = pooled_active_rate(
                history, player_id=player_id, target_season=target_season, rate_col=f"{stat}_per_active"
            )["value"]
    games = float(avail["expected_active_starts"])
    return {
        "ok": True,
        "rates": rates,
        "expected_games": games,
        "archetype": rush["archetype"],
        "season_points": _score_rates(rates, games),
        "ppg": _score_rates(rates, games) / games if games else None,
        "designed_carries_per_active": rush["priors"].get("designed_carries_per_active"),
        "scramble_per_dropback": rush["priors"].get("scramble_per_dropback"),
    }


def _cohort_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    arch = frame["archetype"].astype(str)
    dual = arch.isin(["designed_runner", "mobile_scrambler"])
    pocket = arch.eq("pocket_passer")
    returning = frame["returning_injury"].astype(bool)
    return {
        "all": pd.Series(True, index=frame.index),
        "dual_threat": dual,
        "pocket_passer": pocket,
        "returning_injury": returning,
        "primary": dual | returning,
        "top12_actual": frame["actual_rank"] <= 12,
    }


def _mae(y, p):
    return float(np.mean(np.abs(np.asarray(p, float) - np.asarray(y, float))))


def _spearman(y, p):
    if len(y) < 3:
        return float("nan")
    return float(pd.Series(y).corr(pd.Series(p), method="spearman"))


def evaluate_season(history: pd.DataFrame, season: int) -> dict:
    ev = _load_eval(season)
    if ev.empty:
        return {"season": season, "n": 0}
    rows = []
    for _, r in ev.iterrows():
        pid = str(r["player_id"])
        actual_games = float(r["actual_games_played"])
        actual_pts = float(r["actual_points"])
        actual_ppg = actual_pts / actual_games if actual_games else np.nan
        # Returning-injury: prior season active_starts < 12 and earlier pool had >= 14
        prior = history[
            (history.player_id.astype(str) == pid)
            & (history.season < season)
            & (history.season >= season - 4)
        ]
        last = prior.sort_values("season").iloc[-1] if not prior.empty else None
        earlier = prior[prior.season < (last.season if last is not None else season)] if last is not None else prior
        returning = bool(
            last is not None
            and float(last.get("active_starts") or 0) < 12
            and (earlier.empty or float(earlier["active_starts"].max()) >= 14)
        )
        arch = classify_archetype(history, player_id=pid, target_season=season)["archetype"]
        base = predict_player(history, player_id=pid, target_season=season, mode="baseline_conflated")
        cand = predict_player(history, player_id=pid, target_season=season, mode="candidate_active_archetype")
        if not base["ok"] or not cand["ok"]:
            continue
        actual_att_pa = float(r["attempts"]) / actual_games
        actual_car_pa = float(r["carries"]) / actual_games
        rows.append(
            {
                "player_id": pid,
                "display_name": r.get("display_name"),
                "actual_points": actual_pts,
                "actual_ppg": actual_ppg,
                "actual_games": actual_games,
                "actual_attempts_pa": actual_att_pa,
                "actual_carries_pa": actual_car_pa,
                "base_points": base["season_points"],
                "cand_points": cand["season_points"],
                "base_ppg": base["ppg"],
                "cand_ppg": cand["ppg"],
                "base_games": base["expected_games"],
                "cand_games": cand["expected_games"],
                "base_attempts": base["rates"].get("attempts"),
                "cand_attempts": cand["rates"].get("attempts"),
                "base_carries": base["rates"].get("carries"),
                "cand_carries": cand["rates"].get("carries"),
                "archetype": arch,
                "returning_injury": returning,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {"season": season, "n": 0}
    frame = frame.sort_values("actual_points", ascending=False).reset_index(drop=True)
    frame["actual_rank"] = frame.index + 1
    frame["base_rank"] = frame["base_points"].rank(ascending=False, method="min")
    frame["cand_rank"] = frame["cand_points"].rank(ascending=False, method="min")

    cohorts = _cohort_masks(frame)
    metrics = {"season": season, "n": int(len(frame)), "cohorts": {}}
    for name, mask in cohorts.items():
        sub = frame[mask]
        if sub.empty:
            continue
        metrics["cohorts"][name] = {
            "n": int(len(sub)),
            "base_points_mae": _mae(sub.actual_points, sub.base_points),
            "cand_points_mae": _mae(sub.actual_points, sub.cand_points),
            "delta_points_mae": _mae(sub.actual_points, sub.cand_points)
            - _mae(sub.actual_points, sub.base_points),
            "base_ppg_mae": _mae(sub.actual_ppg, sub.base_ppg),
            "cand_ppg_mae": _mae(sub.actual_ppg, sub.cand_ppg),
            "base_spearman": _spearman(sub.actual_points, sub.base_points),
            "cand_spearman": _spearman(sub.actual_points, sub.cand_points),
            "base_attempts_mae": _mae(
                sub.actual_attempts_pa, sub.base_attempts.fillna(sub.actual_attempts_pa)
            ),
            "cand_attempts_mae": _mae(
                sub.actual_attempts_pa, sub.cand_attempts.fillna(sub.actual_attempts_pa)
            ),
            "base_carries_mae": _mae(
                sub.actual_carries_pa, sub.base_carries.fillna(sub.actual_carries_pa)
            ),
            "cand_carries_mae": _mae(
                sub.actual_carries_pa, sub.cand_carries.fillna(sub.actual_carries_pa)
            ),
            "base_games_mae": _mae(sub.actual_games, sub.base_games),
            "cand_games_mae": _mae(sub.actual_games, sub.cand_games),
        }
    # Bootstrap on primary cohort points MAE delta
    primary = frame[cohorts["primary"]]
    if len(primary) >= 5:
        deltas = []
        n = len(primary)
        y = primary.actual_points.to_numpy()
        b = primary.base_points.to_numpy()
        c = primary.cand_points.to_numpy()
        for _ in range(BOOTSTRAP_DRAWS):
            idx = RNG.integers(0, n, n)
            deltas.append(_mae(y[idx], c[idx]) - _mae(y[idx], b[idx]))
        deltas = np.asarray(deltas)
        metrics["primary_bootstrap_delta_mae"] = {
            "mean": float(deltas.mean()),
            "ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
        }
    metrics["rows"] = frame.to_dict("records")
    return metrics


def decide(fold_results: list[dict]) -> dict:
    """Apply predeclared gates; never weaken after seeing results."""
    reasons = []
    gates_hit = {}
    fit = [f for f in fold_results if f.get("season") in FIT_SEASONS and f.get("n", 0) > 0]
    hold = next((f for f in fold_results if f.get("season") == HOLDOUT_SEASON), None)

    def cohort(f, name="all"):
        return (f.get("cohorts") or {}).get(name) or {}

    # Overall non-inferiority on each fit fold + holdout
    overall_ok = True
    for f in fit + ([hold] if hold and hold.get("n") else []):
        c = cohort(f, "all")
        if not c:
            continue
        tol = GATES.overall_mae_non_inferiority_tol
        if c["cand_points_mae"] > c["base_points_mae"] * (1.0 + tol):
            overall_ok = False
            reasons.append(f"overall_mae_regression_season_{f['season']}")
    gates_hit["overall_non_inferiority"] = overall_ok

    # Cohort improvements on fit folds
    improved_fit = 0
    for f in fit:
        c = cohort(f, "primary")
        if c and c["delta_points_mae"] < 0:
            improved_fit += 1
    cohort_fit_ok = improved_fit >= GATES.cohort_improve_min_fit_folds
    gates_hit["primary_cohort_fit_folds"] = cohort_fit_ok
    if not cohort_fit_ok:
        reasons.append(
            f"primary_cohort_improved_on_{improved_fit}_fit_folds_need_{GATES.cohort_improve_min_fit_folds}"
        )

    hold_cohort_ok = False
    hold_ci_ok = False
    top12_ok = False
    spearman_ok = False
    if hold and hold.get("n"):
        pc = cohort(hold, "primary")
        ac = cohort(hold, "all")
        t12 = cohort(hold, "top12_actual")
        if pc:
            hold_cohort_ok = pc["delta_points_mae"] < 0
            if not hold_cohort_ok and GATES.holdout_cohort_must_improve:
                reasons.append("holdout_primary_cohort_did_not_improve")
        boot = hold.get("primary_bootstrap_delta_mae") or {}
        ci = boot.get("ci95") or [0, 1]
        hold_ci_ok = ci[1] < 0
        if GATES.holdout_bootstrap_ci_must_exclude_zero and not hold_ci_ok:
            reasons.append("holdout_primary_bootstrap_ci_includes_zero")
        if t12 and ac:
            top12_ok = t12["cand_points_mae"] <= t12["base_points_mae"] * (
                1.0 + GATES.top12_mae_non_inferiority_tol
            )
            if not top12_ok:
                reasons.append("holdout_top12_mae_regressed_beyond_tol")
            spearman_ok = (ac["cand_spearman"] >= ac["base_spearman"] - GATES.spearman_max_drop) or (
                np.isnan(ac["base_spearman"])
            )
            if not spearman_ok:
                reasons.append("holdout_spearman_dropped_beyond_tol")
    gates_hit["holdout_primary_improve"] = hold_cohort_ok
    gates_hit["holdout_bootstrap_ci"] = hold_ci_ok
    gates_hit["holdout_top12_non_inferior"] = top12_ok
    gates_hit["holdout_spearman"] = spearman_ok
    gates_hit["use_2026_for_selection"] = GATES.use_2026_for_selection

    verdict = "GO" if not reasons else "NO-GO"
    return {
        "verdict": verdict,
        "production_promotion": "NO",
        "reasons": reasons,
        "gates": gates_hit,
        "selected_configuration": None
        if verdict == "NO-GO"
        else "active_start_rates+archetype_priors+joint_v2",
        "predeclared_thresholds": {
            "overall_mae_non_inferiority_tol": GATES.overall_mae_non_inferiority_tol,
            "cohort_improve_min_fit_folds": GATES.cohort_improve_min_fit_folds,
            "holdout_cohort_must_improve": GATES.holdout_cohort_must_improve,
            "holdout_bootstrap_ci_must_exclude_zero": GATES.holdout_bootstrap_ci_must_exclude_zero,
            "top12_mae_non_inferiority_tol": GATES.top12_mae_non_inferiority_tol,
            "spearman_max_drop": GATES.spearman_max_drop,
        },
        "note": (
            "GO means chronological experiment gates vs conflated-rate baseline passed. "
            "Production promotion remains NO: sealed release and active pointer are not modified."
        ),
        "next_falsifiable_hypothesis": (
            None
            if verdict == "GO"
            else (
                "H3: Team pass-game script (dropbacks/play and play-action rate) and OL/skill "
                "supporting cast must be modeled as parents of QB active-start attempt rate; "
                "player-only active-rate pooling cannot recover healthy-starter attempt volume "
                "when the sealed team anchor and backup residual still under-allocate the room."
            )
        ),
    }
