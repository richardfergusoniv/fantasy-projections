"""Missingness-safe QB archetype classification (evaluation infrastructure).

Missing designed-run or scramble history is never treated as pocket-passer
evidence. Labels:
- observed_pocket / pocket_passer
- observed_mobile / mobile_scrambler / designed_runner
- insufficient_history
- missing_identity
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ARCHETYPE_LOOKBACK = 4
ARCHETYPE_MIN_STARTS = 8.0
MOBILE_CARRIES_PER_START = 5.5
DESIGNED_RUNNER_DESIGNED_PER_START = 4.0
MOBILE_SCRAMBLER_SCRAMBLE_PER_DB = 0.08
POCKET_MAX_DESIGNED_PER_START = 2.0
POCKET_MAX_SCRAMBLE_PER_DB = 0.05


def _starts_col(frame: pd.DataFrame) -> str | None:
    for col in ("active_starts", "games"):
        if col in frame.columns:
            return col
    return None


def _wmean(hist: pd.DataFrame, col: str, weight_col: str) -> float | None:
    if col not in hist.columns:
        return None
    vals = pd.to_numeric(hist[col], errors="coerce")
    w = pd.to_numeric(hist[weight_col], errors="coerce").fillna(0.0)
    mask = vals.notna() & w.gt(0)
    if not mask.any():
        return None
    return float(np.average(vals[mask], weights=w[mask]))


def classify_qb_archetype_safe(
    history: pd.DataFrame,
    *,
    player_id: str,
    target_season: int,
) -> dict:
    """Classify with explicit missingness; never pocket on null designed/scramble."""
    if player_id is None or str(player_id).strip() == "" or str(player_id).lower() == "nan":
        return {
            "archetype": "missing_identity",
            "sample_starts": 0.0,
            "input_seasons": [],
            "features": {},
            "reason": "missing_player_id",
        }
    if history is None or history.empty or "player_id" not in history.columns:
        return {
            "archetype": "missing_identity",
            "sample_starts": 0.0,
            "input_seasons": [],
            "features": {},
            "reason": "missing_history",
        }
    hist = history[
        (history["player_id"].astype(str) == str(player_id))
        & (history["season"] < int(target_season))
        & (history["season"] >= int(target_season) - ARCHETYPE_LOOKBACK)
    ].copy()
    if (hist["season"] >= int(target_season)).any():
        raise AssertionError("future-season rows entered archetype classification")
    starts_col = _starts_col(hist)
    if hist.empty or starts_col is None:
        return {
            "archetype": "insufficient_history",
            "sample_starts": 0.0,
            "input_seasons": [int(s) for s in hist["season"].tolist()] if not hist.empty else [],
            "features": {},
            "reason": "no_prior_rows" if hist.empty else "no_starts_column",
        }
    sample = float(pd.to_numeric(hist[starts_col], errors="coerce").fillna(0).sum())
    if sample < ARCHETYPE_MIN_STARTS:
        return {
            "archetype": "insufficient_history",
            "sample_starts": sample,
            "input_seasons": [int(s) for s in hist["season"].tolist()],
            "features": {},
            "reason": "below_min_starts",
        }

    # Prefer explicit per-start columns; otherwise derive from season totals.
    designed = _wmean(hist, "designed_carries_per_active", starts_col)
    scramble_db = _wmean(hist, "scramble_per_dropback", starts_col)
    carries = _wmean(hist, "carries_per_active", starts_col)

    if designed is None and "designed_carries" in hist.columns:
        des_tot = pd.to_numeric(hist["designed_carries"], errors="coerce")
        w = pd.to_numeric(hist[starts_col], errors="coerce").fillna(0.0)
        mask = des_tot.notna() & w.gt(0)
        if mask.any():
            designed = float(des_tot[mask].sum() / w[mask].sum())
    if scramble_db is None and "scramble_carries" in hist.columns and "attempts" in hist.columns:
        scr = pd.to_numeric(hist["scramble_carries"], errors="coerce")
        att = pd.to_numeric(hist["attempts"], errors="coerce")
        mask = scr.notna() & att.notna() & att.gt(0)
        if mask.any():
            scramble_db = float(scr[mask].sum() / att[mask].sum())
    if carries is None and "carries" in hist.columns:
        car = pd.to_numeric(hist["carries"], errors="coerce")
        w = pd.to_numeric(hist[starts_col], errors="coerce").fillna(0.0)
        mask = car.notna() & w.gt(0)
        if mask.any():
            carries = float(car[mask].sum() / w[mask].sum())

    features = {
        "designed_carries_per_start": designed,
        "scramble_per_dropback": scramble_db,
        "carries_per_start": carries,
    }

    # Order: designed → scramble → carries dual-threat evidence.
    # Pocket requires *observed* low designed AND scramble — null is not pocket.
    if designed is not None and designed >= DESIGNED_RUNNER_DESIGNED_PER_START:
        arch = "designed_runner"
        reason = "observed_designed_runner"
    elif scramble_db is not None and scramble_db >= MOBILE_SCRAMBLER_SCRAMBLE_PER_DB:
        arch = "mobile_scrambler"
        reason = "observed_scramble"
    elif carries is not None and carries >= MOBILE_CARRIES_PER_START:
        arch = "mobile_scrambler"
        reason = "observed_carries_mobile"
    elif (
        designed is not None
        and scramble_db is not None
        and designed <= POCKET_MAX_DESIGNED_PER_START
        and scramble_db <= POCKET_MAX_SCRAMBLE_PER_DB
    ):
        arch = "pocket_passer"
        reason = "observed_pocket"
    else:
        arch = "insufficient_history"
        reason = "missing_designed_or_scramble_not_pocket"

    return {
        "archetype": arch,
        "sample_starts": sample,
        "input_seasons": [int(s) for s in hist["season"].tolist()],
        "features": features,
        "reason": reason,
    }
