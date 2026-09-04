"""H4 experience / cohort taxonomy (preseason-available information only)."""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from src.projection.qb_h4.decision_policy import (
    ESTABLISHED_MIN_PRIOR_ACTIVE_STARTS,
    LIMITED_MAX_PRIOR_ACTIVE_STARTS,
    LIMITED_MIN_PRIOR_ACTIVE_STARTS,
)

ExperienceClass = Literal[
    "established_veteran",
    "limited_history",
    "rookie",
    "insufficient_history",
    "missing_identity",
]


def prior_active_starts_sum(
    history: pd.DataFrame, *, player_id: str, target_season: int
) -> float:
    if history is None or history.empty or not player_id:
        return 0.0
    hist = history[
        (history["player_id"].astype(str) == str(player_id))
        & (history["season"] < int(target_season))
    ]
    if hist.empty or "active_starts" not in hist.columns:
        return 0.0
    return float(pd.to_numeric(hist["active_starts"], errors="coerce").fillna(0.0).sum())


def classify_experience(
    *,
    player_id: str | None,
    target_season: int,
    history: pd.DataFrame,
    is_rookie_at_cutoff: bool = False,
    prior_active_starts: float | None = None,
) -> dict:
    """Classify using only information available before ``target_season``.

    Rules (documented, leakage-safe):
    - missing_identity: blank / NaN player_id
    - rookie: preseason ``is_rookie_at_cutoff`` flag (fantasy_evaluation population)
    - established_veteran: non-rookie with ≥ ESTABLISHED_MIN_PRIOR_ACTIVE_STARTS
      prior active starts (seasons < target)
    - limited_history: non-rookie with 1..LIMITED_MAX prior active starts
    - insufficient_history: non-rookie with 0 prior active starts

    Never uses future starts, snaps, depth outcomes, or same-season labels.
    """
    pid = None if player_id is None or (isinstance(player_id, float) and np.isnan(player_id)) else str(player_id).strip()
    if not pid:
        return {
            "experience_class": "missing_identity",
            "prior_active_starts_sum": 0.0,
            "is_rookie_at_cutoff": bool(is_rookie_at_cutoff),
            "rule": "empty_player_id",
        }
    starts = (
        float(prior_active_starts)
        if prior_active_starts is not None
        else prior_active_starts_sum(history, player_id=pid, target_season=target_season)
    )
    if bool(is_rookie_at_cutoff):
        cls: ExperienceClass = "rookie"
        rule = "preseason_is_rookie_at_cutoff"
    elif starts >= ESTABLISHED_MIN_PRIOR_ACTIVE_STARTS:
        cls = "established_veteran"
        rule = f"prior_active_starts>={ESTABLISHED_MIN_PRIOR_ACTIVE_STARTS}"
    elif starts >= LIMITED_MIN_PRIOR_ACTIVE_STARTS:
        cls = "limited_history"
        rule = f"prior_active_starts_in_[{LIMITED_MIN_PRIOR_ACTIVE_STARTS},{LIMITED_MAX_PRIOR_ACTIVE_STARTS}]"
    else:
        cls = "insufficient_history"
        rule = "non_rookie_zero_prior_active_starts"
    return {
        "experience_class": cls,
        "prior_active_starts_sum": starts,
        "is_rookie_at_cutoff": bool(is_rookie_at_cutoff),
        "rule": rule,
        "player_id": pid,
        "target_season": int(target_season),
    }
