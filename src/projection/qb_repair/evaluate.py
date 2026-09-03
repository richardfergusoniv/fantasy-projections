"""Rolling-origin evaluation for QB repair arms."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

from src.projection.fantasy_points import SCORING
from src.projection.qb_repair.allocation import estimate_starter_backup_shares
from src.projection.qb_repair.history import history_before, load_qb_season_history, per_game_rates
from src.projection.qb_repair.rate_prior import (
    apply_qb_rate_prior,
    build_qb_rate_priors,
    classify_qb_archetype,
)

HOLDOUT_SEASON = 2025
FIT_SEASONS = (2023, 2024)  # selection without using final holdout (2025)


@dataclass
class SegmentMetrics:
    segment: str
    n: int
    ppg_mae: float
    season_mae: float
    bias: float
    spearman: float
    top6_hit: float
    top12_hit: float
    fraction_improved: float
    calibration_slope: float


def _half_ppr_from_rates(row: pd.Series, games: float) -> tuple[float, float]:
    """Return (ppg, season_points) from per-game counting rates."""
    ppg = 0.0
    for stat, pts in SCORING.items():
        key = f"{stat}_pg" if f"{stat}_pg" in row.index else stat
        if key in row.index:
            ppg += float(pd.to_numeric(row.get(key), errors="coerce") or 0.0) * float(pts)
        elif stat in row.index and games:
            # season totals provided
            ppg += float(pd.to_numeric(row.get(stat), errors="coerce") or 0.0) / games * float(pts)
    return ppg, ppg * games


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    rho, _ = stats.spearmanr(a, b)
    return float(rho)


def _top_hit(actual: pd.Series, pred: pd.Series, k: int) -> float:
    if len(actual) < k:
        return float("nan")
    a = set(actual.nlargest(k).index)
    p = set(pred.nlargest(k).index)
    return float(len(a & p) / k)


def _calibration_slope(actual: np.ndarray, pred: np.ndarray) -> float:
    if len(actual) < 3 or np.allclose(pred, pred[0]):
        return float("nan")
    slope, _, _, _, _ = stats.linregress(pred, actual)
    return float(slope)


def _metrics(
    frame: pd.DataFrame,
    *,
    actual_col: str,
    pred_col: str,
    segment: str,
    baseline_err: pd.Series | None = None,
) -> SegmentMetrics:
    sub = frame.dropna(subset=[actual_col, pred_col]).copy()
    n = len(sub)
    if n == 0:
        return SegmentMetrics(segment, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    actual = sub[actual_col].to_numpy(dtype=float)
    pred = sub[pred_col].to_numpy(dtype=float)
    games = pd.to_numeric(sub.get("games"), errors="coerce").fillna(17.0).to_numpy(dtype=float)
    actual_ppg = actual / np.clip(games, 1.0, None)
    pred_ppg = pred / np.clip(games, 1.0, None)
    err = np.abs(pred - actual)
    improved = float("nan")
    if baseline_err is not None:
        aligned = baseline_err.reindex(sub.index)
        improved = float((err < aligned.to_numpy(dtype=float)).mean())
    return SegmentMetrics(
        segment=segment,
        n=n,
        ppg_mae=float(np.mean(np.abs(pred_ppg - actual_ppg))),
        season_mae=float(np.mean(err)),
        bias=float(np.mean(pred_ppg - actual_ppg)),
        spearman=_spearman(actual, pred),
        top6_hit=_top_hit(sub[actual_col], sub[pred_col], 6),
        top12_hit=_top_hit(sub[actual_col], sub[pred_col], 12),
        fraction_improved=improved,
        calibration_slope=_calibration_slope(actual_ppg, pred_ppg),
    )


def _load_eval_season(season: int) -> pd.DataFrame:
    path = f"output/fantasy_evaluation_{season}.csv"
    df = pd.read_csv(path)
    qb = df[df["preseason_position"].astype(str).eq("QB")].copy()
    qb["player_id"] = qb["player_id"].astype(str)
    qb["actual_points"] = pd.to_numeric(qb["actual_points"], errors="coerce")
    qb["model_points"] = pd.to_numeric(qb["model_points_end_to_end"], errors="coerce")
    qb["carry_forward_points"] = pd.to_numeric(qb["carry_forward_points"], errors="coerce")
    qb["games"] = pd.to_numeric(qb["actual_games_played"], errors="coerce").fillna(0.0)
    if "depth_tier" in qb.columns:
        qb["depth_tier"] = pd.to_numeric(qb["depth_tier"], errors="coerce")
    else:
        qb["depth_tier"] = np.nan
    return qb


def _prior_predict_season(
    *,
    target_season: int,
    population: pd.DataFrame,
    mobile_rushing_only: bool = False,
    blend_with_model: bool = True,
) -> pd.Series:
    """Predict season points via multi-season rate prior (+ optional model blend)."""
    hist = load_qb_season_history()
    priors = build_qb_rate_priors(
        target_season=target_season,
        player_ids=population["player_id"].astype(str).tolist(),
        history=hist,
        established_only=True,
    )
    # Build a synthetic long board from model component rates when present,
    # else from carry-forward season rates.
    rows = []
    for _, row in population.iterrows():
        games = float(row.get("games") or 17.0) or 17.0
        pid = str(row["player_id"])
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
            season_val = float(pd.to_numeric(row.get(stat), errors="coerce") or 0.0)
            # Evaluation CSVs store actual season totals; for prediction we need
            # a preseason baseline. Prefer model-implied rates from points only
            # when components are absent — use carry-forward via history.
            rows.append(
                {
                    "player_id": pid,
                    "position": "QB",
                    "stat": stat,
                    "pred_pg": np.nan,
                    "depth_tier": row.get("depth_tier", 1.0),
                    "projected_games": 17.0,
                }
            )
    long = pd.DataFrame(rows)
    # Seed pred_pg from history T-1 (carry-forward rates).
    prior_hist = per_game_rates(history_before(hist, target_season))
    last = (
        prior_hist.sort_values("season")
        .groupby("player_id", as_index=False)
        .tail(1)
        .set_index("player_id")
    )
    for idx, row in long.iterrows():
        pid = str(row["player_id"])
        stat = str(row["stat"])
        if pid in last.index and f"{stat}_pg" in last.columns:
            long.at[idx, "pred_pg"] = float(last.loc[pid, f"{stat}_pg"] or 0.0)
        else:
            long.at[idx, "pred_pg"] = 0.0

    long, _ = apply_qb_rate_prior(
        long,
        priors,
        only_tier1=False,
        mobile_rushing_only=mobile_rushing_only,
    )
    # Score season points at 17-game exposure (matches draft board convention).
    preds = {}
    for pid, group in long.groupby("player_id"):
        ppg = 0.0
        stats_map = dict(zip(group["stat"], group["pred_pg"]))
        for stat, pts in SCORING.items():
            ppg += float(stats_map.get(stat, 0.0) or 0.0) * float(pts)
        preds[str(pid)] = ppg * 17.0
    series = population["player_id"].astype(str).map(preds).astype(float)
    if blend_with_model and "model_points" in population.columns:
        model = pd.to_numeric(population["model_points"], errors="coerce")
        # Convex blend selected on fit seasons only by caller.
        return series
    return series


def evaluate_arm_on_season(
    *,
    season: int,
    arm: str,
    prior_blend: float = 0.5,
) -> dict:
    """Score one arm against a fantasy_evaluation season CSV."""
    pop = _load_eval_season(season)
    hist = load_qb_season_history()

    # Baselines
    pop["pred_baseline"] = pd.to_numeric(pop["model_points"], errors="coerce")
    pop["pred_carry"] = pd.to_numeric(pop["carry_forward_points"], errors="coerce")

    # Multi-season prior prediction
    prior_pts = _prior_predict_season(
        target_season=season,
        population=pop,
        mobile_rushing_only=(arm == "mobile_rush_prior"),
        blend_with_model=False,
    )
    pop["pred_prior"] = prior_pts
    pop["pred_prior_blend"] = (
        (1.0 - prior_blend) * pop["pred_baseline"].fillna(0.0)
        + prior_blend * pop["pred_prior"].fillna(pop["pred_baseline"])
    )

    # Allocation arm cannot fully re-compose without raw long boards per fold.
    # Proxy: boost tier-1 predictions toward historical starter share of the
    # position's predicted mass (diagnostic, reported separately).
    alloc = estimate_starter_backup_shares(target_season=season, history=hist)
    pop["pred_allocation"] = pop["pred_baseline"]
    tier1 = pop["depth_tier"].fillna(1.0).eq(1.0) | pop["depth_tier"].isna()
    # Re-scale: give tier-1 players alloc.starter_attempt_share of team-relative
    # mass approximated by boosting under-baseline players who are QB1s with
    # negative residual vs carry-forward.
    gap = pop["pred_carry"] - pop["pred_baseline"]
    pop.loc[tier1, "pred_allocation"] = pop.loc[tier1, "pred_baseline"] + 0.5 * gap.loc[tier1].clip(lower=0)

    pop["pred_allocation_plus_priors"] = (
        0.5 * pop["pred_allocation"].fillna(pop["pred_baseline"])
        + 0.5 * pop["pred_prior"].fillna(pop["pred_baseline"])
    )

    arm_col = {
        "baseline": "pred_baseline",
        "allocation": "pred_allocation",
        "multi_season_prior": "pred_prior_blend",
        "mobile_rush_prior": "pred_prior_blend",
        "allocation_plus_priors": "pred_allocation_plus_priors",
        "carry_forward": "pred_carry",
        "simple_multi_season_rate": "pred_prior",
    }[arm]

    # Segments
    pop["archetype"] = [
        classify_qb_archetype(hist, pid, target_season=season) for pid in pop["player_id"]
    ]
    pop["is_qb1"] = pop["depth_tier"].fillna(99).eq(1.0) | (
        pop["depth_tier"].isna() & (pop["actual_points"] > 0)
    )
    # High-confidence starters: tier-1 with 8+ actual games.
    pop["high_conf"] = pop["is_qb1"] & pop["games"].ge(8)
    pop["partial_prior"] = False
    prior_hist = history_before(hist, season)
    for i, row in pop.iterrows():
        ph = prior_hist[prior_hist["player_id"].astype(str).eq(str(row["player_id"]))]
        if not ph.empty and pd.to_numeric(ph["games"], errors="coerce").iloc[-1] < 12:
            pop.at[i, "partial_prior"] = True

    baseline_err = (pop["pred_baseline"] - pop["actual_points"]).abs()
    segments = {
        "all_qb": pop,
        "qb1": pop[pop["is_qb1"]],
        "mobile_qb": pop[pop["archetype"].eq("mobile")],
        "pocket_qb": pop[pop["archetype"].eq("pocket")],
        "partial_prior_season": pop[pop["partial_prior"]],
        "high_confidence_starter": pop[pop["high_conf"]],
    }
    metrics = []
    for name, frame in segments.items():
        metrics.append(
            asdict(
                _metrics(
                    frame,
                    actual_col="actual_points",
                    pred_col=arm_col,
                    segment=name,
                    baseline_err=baseline_err.reindex(frame.index),
                )
            )
        )
    return {
        "season": season,
        "arm": arm,
        "allocation": alloc.__dict__,
        "metrics": metrics,
        "n": int(len(pop)),
    }


def select_arm_from_fit_seasons(
    arms: list[str],
    *,
    fit_seasons: tuple[int, ...] = FIT_SEASONS,
) -> dict:
    """Select architecture on fit seasons only; holdout stays untouched."""
    table = []
    for season in fit_seasons:
        for arm in arms:
            table.append(evaluate_arm_on_season(season=season, arm=arm))
    # Aggregate starter MAE / Spearman across fit seasons.
    scores = []
    for arm in arms:
        rows = [r for r in table if r["arm"] == arm]
        starter = []
        all_qb = []
        for r in rows:
            by = {m["segment"]: m for m in r["metrics"]}
            starter.append(by.get("high_confidence_starter") or by.get("qb1"))
            all_qb.append(by["all_qb"])
        starter = [s for s in starter if s and s["n"] > 0]
        all_qb = [s for s in all_qb if s and s["n"] > 0]
        if not starter:
            continue
        scores.append(
            {
                "arm": arm,
                "starter_ppg_mae": float(np.nanmean([s["ppg_mae"] for s in starter])),
                "starter_spearman": float(np.nanmean([s["spearman"] for s in starter])),
                "all_qb_ppg_mae": float(np.nanmean([s["ppg_mae"] for s in all_qb])),
                "all_qb_spearman": float(np.nanmean([s["spearman"] for s in all_qb])),
            }
        )
    scores = sorted(
        scores,
        key=lambda r: (r["starter_ppg_mae"], -r["starter_spearman"], r["all_qb_ppg_mae"]),
    )
    selected = scores[0]["arm"] if scores else "baseline"
    return {"selected_arm": selected, "fit_scores": scores, "fit_raw": table}


def evaluate_holdout(arm: str, *, holdout_season: int = HOLDOUT_SEASON) -> dict:
    return evaluate_arm_on_season(season=holdout_season, arm=arm)
