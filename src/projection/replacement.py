"""Replacement-level baselines for curated players with no other projection path.

Does not import predict.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.projection.contracts import (
    REPLACEMENT_DEPTH_BANDS,
    REPLACEMENT_MIN_CELL,
    REPLACEMENT_POSITIONS,
)
from src.projection.depth_history import attach_depth_rank
from src.projection.features import TARGET_STATS


# QB is deliberately excluded from the replacement-level path — see contracts.
# REPLACEMENT_* constants live in contracts.py.


def _replacement_depth_band(rank):
    if rank is None or (isinstance(rank, float) and np.isnan(rank)):
        return "off_chart"
    for limit, label in REPLACEMENT_DEPTH_BANDS:
        if rank <= limit:
            return label
    return "rank_3_plus"


def fit_replacement_level_baselines(conn, feat, seasons):
    """Per-game rates for a player the models cannot see, by position and
    preseason depth band.

    A curated depth-chart player with no source-season production has no
    veteran feature row and is not in the rookie class, so nothing in the
    pipeline projects him at all - he simply does not exist in the output.
    That is the silent-drop failure this project has already been burned by
    twice, and it has a second cost here: his share of the team is not held
    open for him, so the reconcilers hand it to whoever is present.

    Fit the same way the rookie baselines and the Gate B ladder are - group
    historical outcomes and take the conditional mean - rather than asserting
    a constant. Rates are conditional on playing (games_played > 0), because
    that is what a per-game rate means everywhere else in this pipeline;
    availability is a separate question answered separately.

    Banding is on the nflverse preseason rank rather than a curated `role`
    because no curated table exists for historical seasons - the same reason
    Gate B is keyed that way. Cells thinner than REPLACEMENT_MIN_CELL fall
    back to the position's own pooled mean rather than trusting a handful of
    rows.
    """
    hist = feat[feat["season"].isin(seasons) & (feat["games_played"] > 0)].copy()
    if hist.empty:
        return pd.DataFrame()
    ranked = []
    for season in sorted(hist["season"].unique()):
        ranked.append(attach_depth_rank(hist[hist["season"] == season], int(season), conn=conn))
    hist = pd.concat(ranked, ignore_index=True)
    hist["depth_band"] = [_replacement_depth_band(r) for r in hist["nfl_depth_rank"]]

    pg_cols = sorted({f"{s}_pg" for stats in TARGET_STATS.values() for s in stats}
                     & set(hist.columns))
    banded = hist.groupby(["position", "depth_band"])[pg_cols].mean()
    counts = hist.groupby(["position", "depth_band"]).size().rename("n")
    pooled = hist.groupby("position")[pg_cols].mean()

    out = banded.join(counts)
    thin = out["n"] < REPLACEMENT_MIN_CELL
    for position, _ in out.index[thin]:
        out.loc[thin & (out.index.get_level_values("position") == position), pg_cols] = \
            pooled.loc[position, pg_cols].to_numpy()
    out["replacement_cell_thin"] = thin
    return out


def build_replacement_level_rows(conn, feat, depth_chart, present_ids, target_season,
                                 seasons, roster_map=None):
    """Synthesize rows for curated depth-chart players nothing else projects.

    Returns long-form rows shaped like the veteran output, marked
    `source='replacement_level'` and `low_confidence=True` so a reader can
    always tell a floor prior from a modeled projection.
    """
    if depth_chart.empty:
        return pd.DataFrame()
    missing = depth_chart[
        ~depth_chart["gsis_id"].isin(present_ids)
        & depth_chart["position"].isin(REPLACEMENT_POSITIONS)
    ].dropna(subset=["gsis_id"])
    if missing.empty:
        return pd.DataFrame()

    baselines = fit_replacement_level_baselines(conn, feat, seasons)
    if baselines.empty:
        return pd.DataFrame()

    availability = _replacement_availability(conn, feat, seasons)
    rows = []
    for _, r in missing.iterrows():
        position = r["position"]
        stats = TARGET_STATS.get(position, [])
        band = _replacement_depth_band(r.get("depth_rank"))
        key = (position, band)
        if key not in baselines.index:
            key = (position, "rank_3_plus")
        if key not in baselines.index:
            continue
        base = baselines.loc[key]
        games = availability.get((position, band), availability.get((position, "rank_3_plus"), np.nan))
        for stat in stats:
            col = f"{stat}_pg"
            if col not in base.index or pd.isna(base[col]):
                continue
            rows.append({
                "player_id": r["gsis_id"], "team": r["team"], "position": position,
                "stat": stat, "pred_pg": float(base[col]),
                "pred_pg_low": 0.0, "pred_pg_high": float(base[col]) * 2.0,
                "season": target_season, "source": "replacement_level",
                "low_confidence": True, "interval_low_n_flag": True,
                "team_changed": False, "roster_status": np.nan,
                "depth_rank": r.get("depth_rank"), "role": r.get("role"),
                "depth_chart_status": "replacement_level",
                "role_discount_applied": False, "role_discount_factor": 1.0,
                "nfl_depth_rank": np.nan,
                "projected_games": games, "projected_games_raw": games,
                "replacement_cell_thin": bool(base.get("replacement_cell_thin", False)),
            })
    return pd.DataFrame(rows)


def _replacement_availability(conn, feat, seasons):
    """Mean games played by position and preseason depth band, over the full
    cohort including players who never appeared - the same full-cohort
    treatment rookies.py uses, so a floor prior is not silently given a
    starter's exposure."""
    hist = feat[feat["season"].isin(seasons)].copy()
    if hist.empty:
        return {}
    ranked = []
    for season in sorted(hist["season"].unique()):
        ranked.append(attach_depth_rank(hist[hist["season"] == season], int(season), conn=conn))
    hist = pd.concat(ranked, ignore_index=True)
    hist["depth_band"] = [_replacement_depth_band(r) for r in hist["nfl_depth_rank"]]
    means = hist.groupby(["position", "depth_band"])["games_played"].mean()
    return means.to_dict()
