"""Component-level rate evaluation for QB multi-season priors."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.projection.qb_repair.history import (
    history_before,
    load_qb_season_history,
    per_game_rates,
)
from src.projection.qb_repair.rate_prior import build_qb_rate_priors, classify_qb_archetype


def evaluate_rate_prior_components(
    *,
    holdout_season: int = 2025,
    min_games: float = 8.0,
) -> dict:
    """Score prior vs carry-forward T-1 on holdout per-game rates.

    This does not replace fantasy-point gates; it diagnoses whether the prior
    recovers volume/efficiency components leakage-safely.
    """
    hist = load_qb_season_history()
    hold = per_game_rates(
        hist[hist["season"].eq(holdout_season) & (hist["games"] >= min_games)].copy()
    )
    if hold.empty:
        return {"n": 0}
    priors = build_qb_rate_priors(
        target_season=holdout_season,
        player_ids=hold["player_id"].astype(str).tolist(),
        history=hist,
        established_only=True,
    )
    prior_hist = per_game_rates(history_before(hist, holdout_season))
    last = (
        prior_hist.sort_values("season")
        .groupby("player_id", as_index=False)
        .tail(1)
        .set_index("player_id")
    )

    components = [
        "attempts_pg",
        "passing_yards_pg",
        "carries_pg",
        "rushing_yards_pg",
        "designed_carries_pg",
        "scramble_carries_pg",
    ]
    rows = []
    for _, row in hold.iterrows():
        pid = str(row["player_id"])
        arch = classify_qb_archetype(hist, pid, target_season=holdout_season)
        rec = priors.get(pid)
        for col in components:
            actual = row.get(col)
            if actual is None or (isinstance(actual, float) and np.isnan(actual)):
                continue
            cf = float(last.loc[pid, col]) if pid in last.index and col in last.columns else np.nan
            prior = (
                float(rec.components.get(col, np.nan))
                if rec and rec.applied
                else np.nan
            )
            rows.append(
                {
                    "player_id": pid,
                    "archetype": arch,
                    "component": col,
                    "actual": float(actual),
                    "carry_forward": cf,
                    "prior": prior,
                }
            )
    frame = pd.DataFrame(rows)
    summary = []
    for (arch, comp), grp in frame.groupby(["archetype", "component"]):
        sub = grp.dropna(subset=["actual", "prior", "carry_forward"])
        if sub.empty:
            continue
        summary.append(
            {
                "archetype": arch,
                "component": comp,
                "n": int(len(sub)),
                "prior_mae": float((sub["prior"] - sub["actual"]).abs().mean()),
                "carry_forward_mae": float(
                    (sub["carry_forward"] - sub["actual"]).abs().mean()
                ),
                "prior_bias": float((sub["prior"] - sub["actual"]).mean()),
                "improved_vs_cf_frac": float(
                    (
                        (sub["prior"] - sub["actual"]).abs()
                        < (sub["carry_forward"] - sub["actual"]).abs()
                    ).mean()
                ),
            }
        )
    return {
        "holdout_season": holdout_season,
        "n_player_components": int(len(frame)),
        "summary": summary,
    }
