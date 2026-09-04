"""H4 role- and experience-conditioned priors for rookies / insufficient history.

Does not retune H3 availability coefficients or archetype thresholds.
Uses only seasons < target_season. Empirical-Bayes shrink toward peer means
stratified by experience class × preseason role when sample allows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.projection.qb_active_archetype.active_rates import pooled_active_rate
from src.projection.qb_active_archetype.thresholds import (
    ARCHETYPE_PRIOR_STRENGTH_STARTS,
    AVAIL_LOOKBACK_SEASONS,
)
from src.projection.qb_h3.archetype import classify_archetype_h3, hierarchical_rush_priors_h3
from src.projection.qb_h4.experience import classify_experience

RATE_COLS = (
    "attempts_per_active",
    "completions_per_active",
    "passing_yards_per_active",
    "passing_tds_per_active",
    "interceptions_per_active",
    "carries_per_active",
    "rushing_yards_per_active",
    "rushing_tds_per_active",
)

# Shrinkage strength for thin experience classes (starts). Narrow + interpretable.
ROOKIE_PRIOR_STRENGTH = 16.0
INSUFFICIENT_PRIOR_STRENGTH = 20.0
LIMITED_PRIOR_STRENGTH = 12.0

# League fallback active rates when peer cells are empty (from healthy starters,
# not tuned on 2025). These are structural NFL-ish anchors, not fold-fit.
LEAGUE_FALLBACK = {
    "attempts_per_active": 34.0,
    "completions_per_active": 22.0,
    "passing_yards_per_active": 240.0,
    "passing_tds_per_active": 1.5,
    "interceptions_per_active": 0.7,
    "carries_per_active": 3.0,
    "rushing_yards_per_active": 12.0,
    "rushing_tds_per_active": 0.15,
}

# Backup residual rates stay lower than starter (role-conditioned).
ROLE_RATE_SCALE = {
    "starter": 1.0,
    "rookie_starter": 0.95,
    "backup": 0.85,
    "rookie_backup": 0.80,
    "package": 0.70,
    "rookie_package": 0.65,
}


def _peer_pool(
    history: pd.DataFrame,
    *,
    target_season: int,
    experience_class: str,
    role: str | None,
) -> pd.DataFrame:
    """Historical peer seasons strictly before target_season."""
    hist = history[history["season"] < int(target_season)].copy()
    if hist.empty:
        return hist
    # Tag each historical player-season with the experience class that would
    # have been assigned *entering that season* (prior starts before that season).
    rows = []
    for _, r in hist.iterrows():
        pid = str(r["player_id"])
        season = int(r["season"])
        # Approximate rookie flag: first NFL season with any active starts in history.
        prior = history[
            (history.player_id.astype(str) == pid) & (history.season < season)
        ]
        prior_starts = float(pd.to_numeric(prior.get("active_starts"), errors="coerce").fillna(0).sum()) if len(prior) else 0.0
        is_rookie = prior_starts <= 0 and float(r.get("active_starts") or 0) >= 0
        # For historical peer labeling we only use prior_starts; treat first
        # observed season as rookie-analog when prior_starts==0.
        exp = classify_experience(
            player_id=pid,
            target_season=season,
            history=history,
            is_rookie_at_cutoff=is_rookie and prior_starts <= 0,
            prior_active_starts=prior_starts,
        )["experience_class"]
        if experience_class in ("rookie", "insufficient_history") and exp not in (
            "rookie",
            "insufficient_history",
            "limited_history",
        ):
            continue
        if experience_class == "limited_history" and exp not in (
            "limited_history",
            "rookie",
            "insufficient_history",
        ):
            continue
        if experience_class == "established_veteran" and exp != "established_veteran":
            continue
        rows.append({**r.to_dict(), "_peer_exp": exp})
    if not rows:
        return pd.DataFrame()
    peers = pd.DataFrame(rows)
    # Prefer same experience; fall back to all thin-history peers.
    same = peers[peers["_peer_exp"] == experience_class]
    if len(same) >= 8:
        return same
    return peers


def peer_rate_means(
    history: pd.DataFrame,
    *,
    target_season: int,
    experience_class: str,
    role: str | None,
) -> dict:
    peers = _peer_pool(
        history, target_season=target_season, experience_class=experience_class, role=role
    )
    means = {}
    if peers.empty:
        means = dict(LEAGUE_FALLBACK)
        means["_peer_n"] = 0
        means["_peer_starts"] = 0.0
        return means
    w = pd.to_numeric(peers.get("active_starts"), errors="coerce").fillna(0.0)
    for col in RATE_COLS:
        if col not in peers.columns:
            means[col] = LEAGUE_FALLBACK.get(col)
            continue
        vals = pd.to_numeric(peers[col], errors="coerce")
        mask = vals.notna() & w.gt(0)
        if not mask.any():
            means[col] = LEAGUE_FALLBACK.get(col)
        else:
            means[col] = float(np.average(vals[mask], weights=w[mask]))
    scale = ROLE_RATE_SCALE.get(role or "starter", 1.0)
    for col in RATE_COLS:
        if means.get(col) is not None:
            means[col] = float(means[col]) * scale
    means["_peer_n"] = int(len(peers))
    means["_peer_starts"] = float(w.sum())
    return means


def blend_rates(
    *,
    player_rates: dict,
    peer_means: dict,
    sample_starts: float,
    experience_class: str,
) -> dict:
    if experience_class == "rookie":
        strength = ROOKIE_PRIOR_STRENGTH
    elif experience_class == "insufficient_history":
        strength = INSUFFICIENT_PRIOR_STRENGTH
    elif experience_class == "limited_history":
        strength = LIMITED_PRIOR_STRENGTH
    else:
        strength = ARCHETYPE_PRIOR_STRENGTH_STARTS
    shrink = float(sample_starts) / (float(sample_starts) + strength)
    # Rookies / zero-history: force low shrink so peer prior dominates.
    if experience_class in ("rookie", "insufficient_history", "missing_identity"):
        shrink = min(shrink, 0.25)
    out = {}
    for col in RATE_COLS:
        key = col  # already *_per_active
        short = col.replace("_per_active", "")
        pval = player_rates.get(short)
        peer = peer_means.get(col)
        if pval is None and peer is None:
            out[short] = None
        elif pval is None:
            out[short] = peer
        elif peer is None:
            out[short] = pval
        else:
            out[short] = shrink * float(pval) + (1.0 - shrink) * float(peer)
    out["_shrink"] = shrink
    out["_prior_strength"] = strength
    return out


def h4_active_rates(
    history: pd.DataFrame,
    *,
    player_id: str,
    target_season: int,
    experience_class: str,
    preseason_role: str | None,
) -> dict:
    """Return per-active rates with experience-conditioned priors when needed."""
    player = {}
    sample = 0.0
    for col in RATE_COLS:
        short = col.replace("_per_active", "")
        got = pooled_active_rate(
            history,
            player_id=player_id,
            target_season=target_season,
            rate_col=col,
            lookback=AVAIL_LOOKBACK_SEASONS,
        )
        player[short] = got["value"]
        sample = max(sample, float(got.get("sample_active_starts") or 0.0))

    use_special_prior = experience_class in (
        "rookie",
        "insufficient_history",
        "limited_history",
        "missing_identity",
    )
    if not use_special_prior:
        # Established veterans: H3-style player pooling + archetype rush priors.
        rush = hierarchical_rush_priors_h3(
            history, player_id=player_id, target_season=target_season
        )
        for stat in ("carries", "rushing_yards", "rushing_tds"):
            if rush["priors"].get(f"{stat}_per_active") is not None:
                player[stat] = rush["priors"][f"{stat}_per_active"]
        arch = classify_archetype_h3(history, player_id=player_id, target_season=target_season)
        return {
            "rates": player,
            "shrink": 1.0,
            "peer": {},
            "archetype": arch["archetype"],
            "method": "veteran_player_pool",
            "sample_active_starts": sample,
        }

    peer = peer_rate_means(
        history,
        target_season=target_season,
        experience_class=experience_class if experience_class != "missing_identity" else "insufficient_history",
        role=preseason_role,
    )
    blended = blend_rates(
        player_rates=player,
        peer_means=peer,
        sample_starts=sample,
        experience_class=experience_class,
    )
    shrink = blended.pop("_shrink")
    strength = blended.pop("_prior_strength")
    # Rush archetype: never pocket on null designed; use H3 classifier on covered history.
    arch = classify_archetype_h3(history, player_id=player_id, target_season=target_season)
    rush = hierarchical_rush_priors_h3(history, player_id=player_id, target_season=target_season)
    # For thin history, blend rush toward peer carries if player missing.
    for stat in ("carries", "rushing_yards", "rushing_tds"):
        prior_val = rush["priors"].get(f"{stat}_per_active")
        if prior_val is not None and blended.get(stat) is not None:
            blended[stat] = shrink * float(blended[stat]) + (1.0 - shrink) * float(prior_val)
        elif prior_val is not None:
            blended[stat] = prior_val
    return {
        "rates": blended,
        "shrink": shrink,
        "prior_strength": strength,
        "peer": {k: v for k, v in peer.items() if not str(k).startswith("_")},
        "peer_n": peer.get("_peer_n"),
        "archetype": arch["archetype"],
        "method": f"experience_prior:{experience_class}",
        "sample_active_starts": sample,
    }
