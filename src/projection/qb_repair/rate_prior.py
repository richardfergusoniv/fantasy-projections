"""Multi-season established-QB rate priors (leakage-safe)."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.projection.qb_repair.history import (
    RATE_STATS,
    history_before,
    load_qb_season_history,
    per_game_rates,
)

# Dimensions modeled separately (passing volume / efficiency / rush splits / TD).
PRIOR_COMPONENTS = (
    "attempts_pg",
    "passing_yards_pg",
    "yards_per_attempt",
    "pass_td_rate",
    "int_rate",
    "designed_carries_pg",
    "scramble_carries_pg",
    "carries_pg",
    "rushing_yards_pg",
    "rush_td_rate",
)

MOBILE_CARRIES_PG_THRESHOLD = 5.5
ESTABLISHED_MIN_PRIOR_GAMES = 16
ESTABLISHED_MIN_SEASONS = 2
LOOKBACK_SEASONS = 4
PARTIAL_GAMES_FULL_WEIGHT = 12.0


@dataclass
class PlayerPriorRecord:
    player_id: str
    archetype: str
    input_seasons: list[int]
    sample_games: float
    weight: float
    applied: bool
    reason: str
    components: dict[str, float]
    adjustments: dict[str, float]


def classify_qb_archetype(
    history: pd.DataFrame,
    player_id: str,
    *,
    target_season: int,
) -> str:
    """Classify mobile vs pocket from prior-season carries/game only."""
    prior = history_before(history, target_season)
    prior = prior[prior["player_id"].astype(str).eq(str(player_id))]
    if prior.empty:
        return "unknown"
    rates = per_game_rates(prior)
    # Games-weighted mean carries/game over lookback.
    rates = rates.sort_values("season").tail(LOOKBACK_SEASONS)
    w = pd.to_numeric(rates["games"], errors="coerce").clip(lower=0.0)
    car = pd.to_numeric(rates["carries_pg"], errors="coerce")
    mask = w.gt(0) & car.notna()
    if not mask.any():
        return "unknown"
    mean_car = float(np.average(car[mask], weights=w[mask]))
    return "mobile" if mean_car >= MOBILE_CARRIES_PG_THRESHOLD else "pocket"


def _season_weights(games: pd.Series) -> pd.Series:
    """Partial seasons keep per-game info but down-weight by evidence mass."""
    g = pd.to_numeric(games, errors="coerce").fillna(0.0).clip(lower=0.0)
    return (g / PARTIAL_GAMES_FULL_WEIGHT).clip(upper=1.0) * g


def _archetype_means(rates: pd.DataFrame, archetype: str) -> dict[str, float]:
    sub = rates.copy()
    if "archetype" in sub.columns:
        pooled = sub[sub["archetype"].eq(archetype)]
        if len(pooled) < 20:
            pooled = sub
    else:
        pooled = sub
    out = {}
    for col in PRIOR_COMPONENTS:
        if col not in pooled.columns:
            continue
        vals = pd.to_numeric(pooled[col], errors="coerce")
        w = _season_weights(pooled["games"])
        mask = vals.notna() & w.gt(0)
        if mask.any():
            out[col] = float(np.average(vals[mask], weights=w[mask]))
    return out


def _fill_rush_splits(rates: pd.DataFrame) -> pd.DataFrame:
    """Impute designed/scramble splits when pbp coverage is missing."""
    out = rates.copy()
    car = pd.to_numeric(out.get("carries_pg"), errors="coerce")
    des = pd.to_numeric(out.get("designed_carries_pg"), errors="coerce")
    scr = pd.to_numeric(out.get("scramble_carries_pg"), errors="coerce")
    missing = des.isna() | scr.isna()
    # League split among rows with both observed.
    both = des.notna() & scr.notna() & car.notna() & car.gt(0)
    if both.any():
        des_share = float((des[both] / car[both]).clip(0, 1).median())
    else:
        des_share = 0.55
    out.loc[missing, "designed_carries_pg"] = car[missing] * des_share
    out.loc[missing, "scramble_carries_pg"] = car[missing] * (1.0 - des_share)
    return out


def build_qb_rate_priors(
    *,
    target_season: int,
    player_ids: list[str] | None = None,
    history: pd.DataFrame | None = None,
    established_only: bool = True,
) -> dict[str, PlayerPriorRecord]:
    """Build leakage-safe multi-season priors for projected QB1 candidates."""
    hist = history if history is not None else load_qb_season_history()
    prior_hist = history_before(hist, target_season)
    rates = _fill_rush_splits(per_game_rates(prior_hist))
    # Label archetypes on historical rows for pooling.
    arch_map = {}
    for pid in rates["player_id"].astype(str).unique():
        arch_map[pid] = classify_qb_archetype(hist, pid, target_season=target_season)
    rates["archetype"] = rates["player_id"].astype(str).map(arch_map)
    arch_priors = {
        "mobile": _archetype_means(rates, "mobile"),
        "pocket": _archetype_means(rates, "pocket"),
        "unknown": _archetype_means(rates, "unknown"),
    }

    ids = player_ids or sorted(rates["player_id"].astype(str).unique())
    records: dict[str, PlayerPriorRecord] = {}
    for pid in ids:
        pid = str(pid)
        rows = rates[rates["player_id"].astype(str).eq(pid)].sort_values("season")
        rows = rows[rows["season"] >= int(target_season) - LOOKBACK_SEASONS]
        archetype = arch_map.get(pid, "unknown")
        if rows.empty:
            records[pid] = PlayerPriorRecord(
                player_id=pid,
                archetype=archetype,
                input_seasons=[],
                sample_games=0.0,
                weight=0.0,
                applied=False,
                reason="no_prior_seasons",
                components={},
                adjustments={},
            )
            continue
        games = pd.to_numeric(rows["games"], errors="coerce").fillna(0.0)
        sample_games = float(games.sum())
        n_seasons = int(rows["season"].nunique())
        established = (
            sample_games >= ESTABLISHED_MIN_PRIOR_GAMES and n_seasons >= ESTABLISHED_MIN_SEASONS
        )
        if established_only and not established:
            records[pid] = PlayerPriorRecord(
                player_id=pid,
                archetype=archetype,
                input_seasons=[int(s) for s in rows["season"].tolist()],
                sample_games=sample_games,
                weight=0.0,
                applied=False,
                reason="insufficient_history",
                components={},
                adjustments={},
            )
            continue
        w = _season_weights(games)
        components = {}
        for col in PRIOR_COMPONENTS:
            if col not in rows.columns:
                continue
            vals = pd.to_numeric(rows[col], errors="coerce")
            mask = vals.notna() & w.gt(0)
            if not mask.any():
                continue
            player_mean = float(np.average(vals[mask], weights=w[mask]))
            pool = arch_priors.get(archetype) or arch_priors["unknown"]
            # Credibility: more games → trust player mean; sparse → archetype.
            cred = float(min(1.0, sample_games / 40.0))
            prior_val = pool.get(col, player_mean)
            components[col] = cred * player_mean + (1.0 - cred) * prior_val
        weight = float(min(1.0, sample_games / 48.0))
        records[pid] = PlayerPriorRecord(
            player_id=pid,
            archetype=archetype,
            input_seasons=[int(s) for s in rows["season"].tolist()],
            sample_games=sample_games,
            weight=weight,
            applied=True,
            reason="established_multi_season",
            components=components,
            adjustments={},
        )
    return records


def _component_to_stat_updates(components: dict[str, float]) -> dict[str, float]:
    """Map prior components onto long-board counting-stat rates."""
    updates: dict[str, float] = {}
    if "attempts_pg" in components:
        updates["attempts"] = components["attempts_pg"]
    if "passing_yards_pg" in components:
        updates["passing_yards"] = components["passing_yards_pg"]
    elif "attempts_pg" in components and "yards_per_attempt" in components:
        updates["passing_yards"] = components["attempts_pg"] * components["yards_per_attempt"]
    if "attempts_pg" in components and "pass_td_rate" in components:
        updates["passing_tds"] = components["attempts_pg"] * components["pass_td_rate"]
    if "attempts_pg" in components and "int_rate" in components:
        updates["interceptions"] = components["attempts_pg"] * components["int_rate"]
    if "attempts_pg" in components and "yards_per_attempt" in components:
        # Completions from a mild completion% proxy if available elsewhere.
        pass
    # Rushing: prefer designed+scramble sum when both exist.
    if "designed_carries_pg" in components and "scramble_carries_pg" in components:
        updates["carries"] = (
            components["designed_carries_pg"] + components["scramble_carries_pg"]
        )
    elif "carries_pg" in components:
        updates["carries"] = components["carries_pg"]
    if "rushing_yards_pg" in components:
        updates["rushing_yards"] = components["rushing_yards_pg"]
    if "carries" in updates and "rush_td_rate" in components:
        updates["rushing_tds"] = updates["carries"] * components["rush_td_rate"]
    return updates


def apply_qb_rate_prior(
    long_df: pd.DataFrame,
    priors: dict[str, PlayerPriorRecord],
    *,
    only_tier1: bool = True,
    mobile_rushing_only: bool = False,
) -> tuple[pd.DataFrame, list[dict]]:
    """Blend established-QB priors into long-form rate rows.

    ``mobile_rushing_only`` limits the prior to rush components for mobile
    archetypes (experimental arm). Otherwise all prior components apply.
    """
    out = long_df.copy()
    audit: list[dict] = []
    if out.empty or not priors:
        return out, audit

    qb = out["position"].astype(str).eq("QB")
    if only_tier1:
        if "depth_tier" in out.columns:
            qb = qb & pd.to_numeric(out["depth_tier"], errors="coerce").eq(1.0)
        elif "depth_rank" in out.columns:
            qb = qb & pd.to_numeric(out["depth_rank"], errors="coerce").eq(1.0)

    for pid, record in priors.items():
        if not record.applied or record.weight <= 0:
            audit.append(asdict(record))
            continue
        if mobile_rushing_only and record.archetype != "mobile":
            rec = asdict(record)
            rec["applied"] = False
            rec["reason"] = "not_mobile"
            audit.append(rec)
            continue
        mask = qb & out["player_id"].astype(str).eq(str(pid))
        if not mask.any():
            continue
        updates = _component_to_stat_updates(record.components)
        if mobile_rushing_only:
            updates = {k: v for k, v in updates.items() if k in ("carries", "rushing_yards", "rushing_tds")}
        adjustments = {}
        w = float(record.weight)
        for stat, target in updates.items():
            smask = mask & out["stat"].astype(str).eq(stat)
            if not smask.any():
                continue
            current = float(pd.to_numeric(out.loc[smask, "pred_pg"], errors="coerce").iloc[0])
            blended = (1.0 - w) * current + w * float(target)
            adjustments[stat] = {
                "before": current,
                "prior": float(target),
                "after": blended,
                "weight": w,
            }
            for col in ("pred_pg", "pred_pg_low", "pred_pg_high"):
                if col not in out.columns:
                    continue
                vals = pd.to_numeric(out.loc[smask, col], errors="coerce")
                # Preserve interval width around the new point for pred_pg_*.
                if col == "pred_pg":
                    out.loc[smask, col] = blended
                else:
                    delta = blended - current
                    out.loc[smask, col] = vals + delta
        # Completion% consistency if attempts moved.
        if "attempts" in adjustments and "completions" in out.loc[mask, "stat"].astype(str).values:
            att_after = adjustments["attempts"]["after"]
            comp_mask = mask & out["stat"].astype(str).eq("completions")
            comp_before = float(pd.to_numeric(out.loc[comp_mask, "pred_pg"], errors="coerce").iloc[0])
            att_before = adjustments["attempts"]["before"]
            if att_before > 0:
                comp_rate = comp_before / att_before
                comp_after = att_after * comp_rate
                adjustments["completions"] = {
                    "before": comp_before,
                    "prior": comp_after,
                    "after": comp_after,
                    "weight": w,
                }
                out.loc[comp_mask, "pred_pg"] = comp_after
        record.adjustments = adjustments
        audit.append(asdict(record))
    return out, audit
