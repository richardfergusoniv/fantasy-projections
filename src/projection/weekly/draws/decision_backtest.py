"""Decision-level backtests on out-of-fold joint draws."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class DecisionReadinessThresholds:
    """Predeclared before reviewing outer-fold results."""

    max_lineup_regret: float = 3.0
    min_matchup_brier_improvement: float = 0.0
    max_matchup_brier: float = 0.25
    min_recommendation_stability: float = 0.85
    max_monte_carlo_stderr: float = 0.02


DEFAULT_DECISION_THRESHOLDS = DecisionReadinessThresholds()


def lineup_points(draws: dict[str, np.ndarray], player_ids: Sequence[str]) -> np.ndarray:
    ids = [pid for pid in player_ids if pid in draws]
    if not ids:
        return np.zeros(0, dtype=float)
    n = min(len(draws[pid]) for pid in ids)
    total = np.zeros(n, dtype=float)
    for pid in ids:
        total += np.asarray(draws[pid], dtype=float)[:n]
    return total


def optimal_lineup_regret(
    *,
    candidate_lineups: Sequence[Sequence[str]],
    draws: dict[str, np.ndarray],
    actual_points: dict[str, float],
) -> dict[str, Any]:
    """Compare recommended (mean-optimal under draws) vs ex-post best legal lineup."""
    if not candidate_lineups:
        return {"regret": float("nan"), "recommended": [], "optimal_actual": []}
    mean_scores = []
    for lineup in candidate_lineups:
        pts = lineup_points(draws, lineup)
        mean_scores.append(float(pts.mean()) if pts.size else -1e9)
    rec_idx = int(np.argmax(mean_scores))
    recommended = list(candidate_lineups[rec_idx])
    actual_scores = [sum(float(actual_points.get(pid, 0.0)) for pid in lineup) for lineup in candidate_lineups]
    opt_idx = int(np.argmax(actual_scores))
    regret = float(actual_scores[opt_idx] - actual_scores[rec_idx])
    return {
        "regret": regret,
        "recommended": recommended,
        "optimal_actual": list(candidate_lineups[opt_idx]),
        "recommended_actual_points": float(actual_scores[rec_idx]),
        "optimal_actual_points": float(actual_scores[opt_idx]),
    }


def matchup_win_probability(
    my_lineup: Sequence[str],
    opp_lineup: Sequence[str],
    draws: dict[str, np.ndarray],
) -> dict[str, float]:
    mine = lineup_points(draws, my_lineup)
    opp = lineup_points(draws, opp_lineup)
    n = min(mine.size, opp.size)
    if n == 0:
        return {"win_prob": float("nan"), "stderr": float("nan"), "n": 0}
    wins = mine[:n] > opp[:n]
    p = float(np.mean(wins))
    stderr = float(np.sqrt(max(p * (1 - p), 1e-9) / n))
    return {"win_prob": p, "stderr": stderr, "n": float(n)}


def matchup_brier(
    win_probs: Sequence[float],
    outcomes: Sequence[int],
) -> float:
    p = np.asarray(win_probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    return float(np.mean((p - y) ** 2))


def recommendation_stability(
    rankings_by_draw_count: dict[int, Sequence[str]],
) -> dict[str, Any]:
    """Agreement of top recommendation as draw count increases."""
    counts = sorted(rankings_by_draw_count)
    if len(counts) < 2:
        return {"stability": float("nan"), "pairs": []}
    pairs = []
    for a, b in zip(counts, counts[1:], strict=False):
        ra, rb = list(rankings_by_draw_count[a]), list(rankings_by_draw_count[b])
        if not ra or not rb:
            continue
        pairs.append({"from": a, "to": b, "same_top": ra[0] == rb[0]})
    stability = float(np.mean([1.0 if p["same_top"] else 0.0 for p in pairs])) if pairs else float("nan")
    return {"stability": stability, "pairs": pairs}


def evaluate_decision_gates(
    *,
    regret: float,
    matchup_brier_score: float,
    stability: float,
    mc_stderr: float,
    thresholds: DecisionReadinessThresholds = DEFAULT_DECISION_THRESHOLDS,
) -> dict[str, Any]:
    failures = []
    if not (regret == regret) or regret > thresholds.max_lineup_regret:
        failures.append("lineup_regret")
    if not (matchup_brier_score == matchup_brier_score) or matchup_brier_score > thresholds.max_matchup_brier:
        failures.append("matchup_brier")
    if not (stability == stability) or stability < thresholds.min_recommendation_stability:
        failures.append("recommendation_stability")
    if not (mc_stderr == mc_stderr) or mc_stderr > thresholds.max_monte_carlo_stderr:
        failures.append("monte_carlo_stderr")
    return {
        "pass": not failures,
        "failures": failures,
        "thresholds": asdict(thresholds),
        "observed": {
            "regret": regret,
            "matchup_brier": matchup_brier_score,
            "stability": stability,
            "mc_stderr": mc_stderr,
        },
    }
