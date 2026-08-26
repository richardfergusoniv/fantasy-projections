"""Dirichlet-style compositional opportunity share models."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.projection.contracts import V3_MODELS_DIR

OPPORTUNITY_TYPES = ("target_share", "carry_share", "air_yards_share")


def _room_key(team: str, position: str, opp_type: str) -> str:
    return f"{team}:{position}:{opp_type}"


def fit_opportunity_shares(long_board: pd.DataFrame) -> dict:
    """Fit softmax-normal share models from historical season totals."""
    out_dir = Path(V3_MODELS_DIR) / "opportunity_shares"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"rooms": {}}
    if long_board.empty:
        (out_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return manifest

    frame = long_board.copy()
    frame["season_total"] = pd.to_numeric(frame.get("pred_season"), errors="coerce").fillna(0.0)
    for (team, position), grp in frame.groupby(["team", "position"], observed=True):
        total = grp.groupby("stat")["season_total"].sum()
        if total.sum() <= 0:
            continue
        shares = (total / total.sum()).to_dict()
        for stat, share in shares.items():
            key = _room_key(str(team), str(position), stat)
            manifest["rooms"][key] = {
                "share": float(share),
                "concentration": max(len(grp) * 2.0, 5.0),
            }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def draw_dirichlet_shares(
    room_players: pd.DataFrame,
    concentration: float,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw compositional shares that sum to 1."""
    n = len(room_players)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])
    prior = pd.to_numeric(room_players.get("pred_pg"), errors="coerce").fillna(0.0).to_numpy()
    if prior.sum() <= 0:
        prior = np.ones(n)
    alpha = prior / prior.sum() * concentration
    return rng.dirichlet(alpha)


def allocate_opportunities(
    players: pd.DataFrame,
    team_volume: float,
    *,
    rng: np.random.Generator,
    manifest: dict | None = None,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Allocate team volume across players with simplex-constrained shares.

    ``group_cols`` names the room that competes for ``team_volume``; each
    group receives the FULL volume. The default splits by position, which is
    right for a resource only one position claims (QB attempts, RB carries)
    and WRONG for one they share: passing WR, TE and RB target rows in a
    single call hands each position group a whole team's targets, allocating
    the team 3x over. Pass ``["team", "stat"]`` for a shared pool.
    """
    out = players.copy()
    out["allocated_volume"] = 0.0
    group_cols = group_cols or ["team", "position", "stat"]
    for keys, grp in out.groupby(group_cols, observed=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        lookup = dict(zip(group_cols, keys))
        key = _room_key(
            str(lookup.get("team", "")),
            str(lookup.get("position", grp["position"].iloc[0] if "position" in grp else "")),
            str(lookup.get("stat", "")),
        )
        concentration = 10.0
        if manifest:
            concentration = manifest.get("rooms", {}).get(key, {}).get("concentration", 10.0)
        shares = draw_dirichlet_shares(grp, concentration, rng=rng)
        out.loc[grp.index, "allocated_volume"] = shares * team_volume
    return out
