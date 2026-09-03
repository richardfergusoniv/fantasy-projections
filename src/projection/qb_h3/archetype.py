"""H3 archetype classifier — frozen H1/H2 thresholds, fixed None-fallback order.

Does not change threshold constants. Only fixes control-flow so missing
designed/scramble features fall through to carries-based dual-threat check
instead of defaulting to pocket_passer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.projection.qb_active_archetype.thresholds import (
    ARCHETYPE_LOOKBACK,
    ARCHETYPE_MIN_ACTIVE_STARTS,
    ARCHETYPE_PRIOR_STRENGTH_STARTS,
    DESIGNED_RUNNER_DESIGNED_PER_START,
    MOBILE_SCRAMBLER_SCRAMBLE_PER_DB,
    POCKET_MAX_DESIGNED_PER_START,
    POCKET_MAX_SCRAMBLE_PER_DB,
)

RUSH_PRIOR_COLS = (
    "designed_carries_per_active",
    "scramble_per_dropback",
    "designed_ypc",
    "scramble_ypa",
    "carries_per_active",
    "rushing_yards_per_active",
    "rushing_tds_per_active",
)


def classify_archetype_h3(history: pd.DataFrame, *, player_id: str, target_season: int) -> dict:
    hist = history[
        (history["player_id"].astype(str) == str(player_id))
        & (history["season"] < int(target_season))
        & (history["season"] >= int(target_season) - ARCHETYPE_LOOKBACK)
    ].copy()
    sample = float(pd.to_numeric(hist.get("active_starts"), errors="coerce").fillna(0).sum())
    if hist.empty or sample < ARCHETYPE_MIN_ACTIVE_STARTS:
        return {
            "archetype": "insufficient_history",
            "sample_active_starts": sample,
            "input_seasons": [int(s) for s in hist["season"].tolist()],
            "features": {},
        }

    def wmean(col: str):
        if col not in hist.columns:
            return None
        vals = pd.to_numeric(hist[col], errors="coerce")
        w = pd.to_numeric(hist["active_starts"], errors="coerce").fillna(0.0)
        mask = vals.notna() & w.gt(0)
        if not mask.any():
            return None
        return float(np.average(vals[mask], weights=w[mask]))

    designed = wmean("designed_carries_per_active")
    scramble_db = wmean("scramble_per_dropback")
    carries = wmean("carries_per_active")
    features = {
        "designed_carries_per_active": designed,
        "scramble_per_dropback": scramble_db,
        "carries_per_active": carries,
    }
    # Order: designed → scramble → carries dual-threat → pocket only when
    # rush features are observed and low (None does NOT imply pocket).
    if designed is not None and designed >= DESIGNED_RUNNER_DESIGNED_PER_START:
        arch = "designed_runner"
    elif scramble_db is not None and scramble_db >= MOBILE_SCRAMBLER_SCRAMBLE_PER_DB:
        arch = "mobile_scrambler"
    elif carries is not None and carries >= 5.5:
        arch = "mobile_scrambler"
    elif (
        designed is not None
        and scramble_db is not None
        and designed <= POCKET_MAX_DESIGNED_PER_START
        and scramble_db <= POCKET_MAX_SCRAMBLE_PER_DB
    ):
        arch = "pocket_passer"
    elif carries is not None and carries < 5.5:
        arch = "pocket_passer"
    else:
        arch = "insufficient_history"
    return {
        "archetype": arch,
        "sample_active_starts": sample,
        "input_seasons": [int(s) for s in hist["season"].tolist()],
        "features": features,
    }


def hierarchical_rush_priors_h3(history: pd.DataFrame, *, player_id: str, target_season: int) -> dict:
    from src.projection.qb_active_archetype.archetypes import _archetype_means

    meta = classify_archetype_h3(history, player_id=player_id, target_season=target_season)
    arch = meta["archetype"]
    hist = history[
        (history["player_id"].astype(str) == str(player_id))
        & (history["season"] < int(target_season))
        & (history["season"] >= int(target_season) - ARCHETYPE_LOOKBACK)
    ].copy()
    w = pd.to_numeric(hist.get("active_starts"), errors="coerce").fillna(0.0)
    sample = float(w.sum())
    shrink = sample / (sample + ARCHETYPE_PRIOR_STRENGTH_STARTS)
    peer_arch = arch if arch != "insufficient_history" else "pocket_passer"
    peer = _archetype_means(history, target_season=target_season, archetype=peer_arch)
    if arch == "insufficient_history":
        shrink = min(shrink, 0.35)
    priors = {}
    for col in RUSH_PRIOR_COLS:
        player_val = None
        if col in hist.columns and sample > 0:
            vals = pd.to_numeric(hist[col], errors="coerce")
            mask = vals.notna() & w.gt(0)
            if mask.any():
                player_val = float(np.average(vals[mask], weights=w[mask]))
        peer_val = peer.get(col)
        if player_val is None and peer_val is None:
            priors[col] = None
        elif player_val is None:
            priors[col] = peer_val
        elif peer_val is None:
            priors[col] = player_val
        else:
            priors[col] = shrink * player_val + (1.0 - shrink) * peer_val
    return {"archetype": arch, "shrink": shrink, "priors": priors, "input_seasons": meta["input_seasons"]}
