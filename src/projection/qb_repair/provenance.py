"""Lamar / mobile-QB rushing feature provenance diagnostics."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.projection.qb_repair.history import (
    history_before,
    load_qb_season_history,
    per_game_rates,
)
from src.projection.qb_repair.rate_prior import build_qb_rate_priors, classify_qb_archetype

REPO_ROOT = Path(__file__).resolve().parents[3]
LAMAR_ID = "00-0034796"


def explain_mobile_rush_provenance(
    *,
    player_id: str = LAMAR_ID,
    target_season: int = 2026,
    sealed_fantasy_path: Path | None = None,
    raw_path: Path | None = None,
) -> dict:
    """Explain why a mobile QB's projected rush line sits where it does."""
    sealed_fantasy_path = sealed_fantasy_path or (
        REPO_ROOT / "output" / "accuracy_first_2026" / "fantasy_points_2026.csv"
    )
    raw_path = raw_path or (REPO_ROOT / "output" / "projections_2026_raw.csv")

    hist = load_qb_season_history()
    prior = per_game_rates(history_before(hist, target_season))
    player = prior[prior["player_id"].astype(str).eq(str(player_id))].sort_values("season")
    archetype = classify_qb_archetype(hist, player_id, target_season=target_season)
    priors = build_qb_rate_priors(
        target_season=target_season, player_ids=[player_id], history=hist
    )
    prior_rec = priors.get(str(player_id))

    raw = pd.read_csv(raw_path)
    raw_p = raw[raw["player_id"].astype(str).eq(str(player_id))]
    raw_rates = {
        str(r.stat): float(pd.to_numeric(r.pred_pg, errors="coerce") or 0.0)
        for _, r in raw_p.iterrows()
    }

    sealed = pd.read_csv(sealed_fantasy_path)
    sealed_p = sealed[sealed["player_id"].astype(str).eq(str(player_id))]
    sealed_rates = {}
    if not sealed_p.empty:
        row = sealed_p.iloc[0]
        for col in (
            "pg_carries",
            "pg_rushing_yards",
            "pg_rushing_tds",
            "pg_attempts",
            "pg_passing_yards",
            "fantasy_pts",
            "fantasy_pts_season",
        ):
            if col in row.index:
                sealed_rates[col] = float(pd.to_numeric(row[col], errors="coerce") or 0.0)

    # Games-weighted lookback excluding target season.
    lookback = player.tail(4)
    w = lookback["games"].clip(lower=0)
    def wmean(col: str) -> float | None:
        if col not in lookback.columns or not w.gt(0).any():
            return None
        vals = lookback[col]
        mask = vals.notna() & w.gt(0)
        if not mask.any():
            return None
        return float((vals[mask] * w[mask]).sum() / w[mask].sum())

    hist_profile = {
        "seasons": lookback[["season", "games", "carries_pg", "rushing_yards_pg", "rushing_tds", "designed_carries_pg", "scramble_carries_pg"]].to_dict("records")
        if not lookback.empty
        else [],
        "games_weighted_carries_pg": wmean("carries_pg"),
        "games_weighted_rushing_yards_pg": wmean("rushing_yards_pg"),
        "games_weighted_designed_carries_pg": wmean("designed_carries_pg"),
        "games_weighted_scramble_carries_pg": wmean("scramble_carries_pg"),
    }

    # Diagnosis against sealed rates.
    sealed_car = sealed_rates.get("pg_carries")
    hist_car = hist_profile["games_weighted_carries_pg"]
    raw_car = raw_rates.get("carries")
    causes = []
    if raw_car is not None and hist_car is not None and raw_car < 0.7 * hist_car:
        causes.append(
            "raw_model_underpredicts_rush_volume: veteran rate model output is "
            f"{raw_car:.2f} carries/g vs games-weighted prior {hist_car:.2f}"
        )
    last = player.tail(1)
    if not last.empty and float(last.iloc[0]["games"]) < 12:
        causes.append(
            "partial_source_season_overweighted: most recent prior season has "
            f"{float(last.iloc[0]['games']):.0f} games; ship path uses T-1 "
            "role prior without multi-season shrink (QB_PARTIAL_PRIOR_SHRINK disabled)"
        )
    if (
        hist_profile["games_weighted_designed_carries_pg"] is not None
        and hist_profile["games_weighted_designed_carries_pg"] > 3
        and (raw_car or 0) < 6
    ):
        causes.append(
            "designed_run_profile_not_preserved: historical designed-rush usage "
            f"~{hist_profile['games_weighted_designed_carries_pg']:.2f}/g is not "
            "reflected in the raw carries forecast"
        )
    if sealed_car is not None and raw_car is not None and abs(sealed_car - raw_car) < 0.5:
        causes.append(
            "compose_does_not_restore_rushing: sealed carries remain near the "
            "raw forecast; team-volume reconcile does not target QB rush stats"
        )

    return {
        "player_id": str(player_id),
        "archetype": archetype,
        "historical_profile": hist_profile,
        "raw_model_rates": raw_rates,
        "sealed_rates": sealed_rates,
        "multi_season_prior": None
        if prior_rec is None
        else {
            "applied": prior_rec.applied,
            "reason": prior_rec.reason,
            "input_seasons": prior_rec.input_seasons,
            "sample_games": prior_rec.sample_games,
            "weight": prior_rec.weight,
            "components": prior_rec.components,
        },
        "causes": causes,
        "verdict": (
            "mobile_rushing_lost_in_raw_rate_model_and_unrepaired_by_compose"
            if causes
            else "no_material_rush_gap"
        ),
    }
