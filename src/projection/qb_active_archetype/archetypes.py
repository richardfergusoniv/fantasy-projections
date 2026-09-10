"""H2: leakage-safe QB rushing archetypes and hierarchical priors."""
from __future__ import annotations

from typing import Literal

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

Archetype = Literal["designed_runner", "mobile_scrambler", "pocket_passer", "insufficient_history"]

RUSH_PRIOR_COLS = (
    "designed_carries_per_active",
    "scramble_per_dropback",
    "designed_ypc",
    "scramble_ypa",
    "carries_per_active",
    "rushing_yards_per_active",
    "rushing_tds_per_active",
)


def classify_archetype(
    history: pd.DataFrame,
    *,
    player_id: str,
    target_season: int,
) -> dict:
    """Classify using only seasons < target_season (no names, no future labels)."""
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

    def wmean(col: str) -> float | None:
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
    # Hierarchy: designed runner first, then scrambler, then pocket.
    if designed is not None and designed >= DESIGNED_RUNNER_DESIGNED_PER_START:
        arch: Archetype = "designed_runner"
    elif scramble_db is not None and scramble_db >= MOBILE_SCRAMBLER_SCRAMBLE_PER_DB:
        arch = "mobile_scrambler"
    elif (
        (designed is None or designed <= POCKET_MAX_DESIGNED_PER_START)
        and (scramble_db is None or scramble_db <= POCKET_MAX_SCRAMBLE_PER_DB)
    ):
        arch = "pocket_passer"
    elif carries is not None and carries >= 5.5:
        arch = "mobile_scrambler"
    else:
        arch = "pocket_passer"
    return {
        "archetype": arch,
        "sample_active_starts": sample,
        "input_seasons": [int(s) for s in hist["season"].tolist()],
        "features": features,
    }


def archetype_panel(history: pd.DataFrame, target_season: int) -> pd.DataFrame:
    """One row per player with pre-season archetype for ``target_season``."""
    players = history[history["season"] < target_season]["player_id"].astype(str).unique()
    rows = []
    for pid in players:
        meta = classify_archetype(history, player_id=pid, target_season=target_season)
        rows.append({"player_id": pid, "target_season": target_season, **meta, **{
            f"feat_{k}": v for k, v in meta.get("features", {}).items()
        }})
    return pd.DataFrame(rows)


def _row_archetype_from_rates(row: pd.Series) -> str:
    """Label a historical season-row from its own rates (no future data)."""
    designed = row.get("designed_carries_per_active")
    scramble_db = row.get("scramble_per_dropback")
    carries = row.get("carries_per_active")
    designed_f = float(designed) if pd.notna(designed) else None
    scramble_f = float(scramble_db) if pd.notna(scramble_db) else None
    carries_f = float(carries) if pd.notna(carries) else None
    if designed_f is not None and designed_f >= DESIGNED_RUNNER_DESIGNED_PER_START:
        return "designed_runner"
    if scramble_f is not None and scramble_f >= MOBILE_SCRAMBLER_SCRAMBLE_PER_DB:
        return "mobile_scrambler"
    if (
        (designed_f is None or designed_f <= POCKET_MAX_DESIGNED_PER_START)
        and (scramble_f is None or scramble_f <= POCKET_MAX_SCRAMBLE_PER_DB)
    ):
        return "pocket_passer"
    if carries_f is not None and carries_f >= 5.5:
        return "mobile_scrambler"
    return "pocket_passer"


def _archetype_means(
    history: pd.DataFrame,
    *,
    target_season: int,
    archetype: str,
) -> dict[str, float]:
    """Peer means from other players' seasons < target, matching archetype."""
    means: dict[str, float] = {}
    prior = history[history["season"] < int(target_season)].copy()
    if prior.empty:
        return means
    prior = prior.copy()
    prior["_arch"] = prior.apply(_row_archetype_from_rates, axis=1)
    peers = prior[prior["_arch"] == archetype]
    if peers.empty:
        return means
    w = pd.to_numeric(peers["active_starts"], errors="coerce").fillna(0.0)
    for col in RUSH_PRIOR_COLS:
        if col not in peers.columns:
            continue
        vals = pd.to_numeric(peers[col], errors="coerce")
        mask = vals.notna() & w.gt(0)
        if mask.any():
            means[col] = float(np.average(vals[mask], weights=w[mask]))
    return means


def hierarchical_rush_priors(
    history: pd.DataFrame,
    *,
    player_id: str,
    target_season: int,
) -> dict:
    """Player multi-season pool shrunk toward archetype means (not cross-archetype)."""
    meta = classify_archetype(history, player_id=player_id, target_season=target_season)
    arch = meta["archetype"]
    hist = history[
        (history["player_id"].astype(str) == str(player_id))
        & (history["season"] < int(target_season))
        & (history["season"] >= int(target_season) - ARCHETYPE_LOOKBACK)
    ].copy()
    w = pd.to_numeric(hist.get("active_starts"), errors="coerce").fillna(0.0)
    sample = float(w.sum())
    shrink = sample / (sample + ARCHETYPE_PRIOR_STRENGTH_STARTS)

    peer = _archetype_means(history, target_season=target_season, archetype=arch) if arch != "insufficient_history" else {}
    # For insufficient history, shrink toward pocket (conservative rush) means.
    if arch == "insufficient_history":
        peer = _archetype_means(history, target_season=target_season, archetype="pocket_passer")
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

    return {
        "archetype": arch,
        "shrink": shrink,
        "sample_active_starts": sample,
        "input_seasons": meta["input_seasons"],
        "priors": priors,
        "peer_means": peer,
        "player_means": {
            col: (
                float(np.average(
                    pd.to_numeric(hist[col], errors="coerce")[
                        pd.to_numeric(hist[col], errors="coerce").notna() & w.gt(0)
                    ],
                    weights=w[pd.to_numeric(hist[col], errors="coerce").notna() & w.gt(0)],
                ))
                if col in hist.columns
                and (pd.to_numeric(hist[col], errors="coerce").notna() & w.gt(0)).any()
                else None
            )
            for col in RUSH_PRIOR_COLS
        },
    }
