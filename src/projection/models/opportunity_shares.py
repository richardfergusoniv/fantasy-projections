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
    prior_col: str | None = None,
    allow_replacement_sink: bool = False,
) -> np.ndarray:
    """Draw compositional shares that sum to 1."""
    n = len(room_players)
    if n == 0:
        return np.array([])
    if n == 1:
        if allow_replacement_sink and prior_col and prior_col in room_players.columns:
            value = pd.to_numeric(room_players[prior_col], errors="coerce").fillna(0.0).iloc[0]
            if value <= 0:
                return np.array([0.0])
        return np.array([1.0])
    # Prefer the exposure-weighted season prior. Shares are claims on a
    # SEASON of team volume, so a per-game rate over-states a player who will
    # not be there for the season: on the 2026 board 18 players carry reduced
    # exposure and would otherwise claim a full share of the room. This is the
    # same weighting transitions.receiving_share_scale applies on the v1 path.
    prior = None
    if prior_col and prior_col in room_players.columns:
        candidate = pd.to_numeric(
            room_players[prior_col], errors="coerce").fillna(0.0).to_numpy()
        if candidate.sum() > 0 or allow_replacement_sink:
            prior = candidate
    if prior is None and "pred_season" in room_players.columns:
        candidate = pd.to_numeric(
            room_players["pred_season"], errors="coerce").fillna(0.0).to_numpy()
        if candidate.sum() > 0:
            prior = candidate
    if prior is None:
        pred_pg = room_players.get("pred_pg", pd.Series(0.0, index=room_players.index))
        prior = pd.to_numeric(pred_pg, errors="coerce").fillna(0.0).to_numpy()
    if prior.sum() <= 0:
        if allow_replacement_sink:
            return np.zeros(n)
        prior = np.ones(n)
    # Draw only over nonzero exposure. Dirichlet requires positive alpha, but
    # adding epsilon would revive a player whose availability draw is zero.
    active = prior > 0
    shares = np.zeros(n)
    alpha = prior[active] / prior[active].sum() * concentration
    shares[active] = rng.dirichlet(alpha)
    return shares


def allocate_opportunities(
    players: pd.DataFrame,
    team_volume: float,
    *,
    rng: np.random.Generator,
    manifest: dict | None = None,
    group_cols: list[str] | None = None,
    pool_key: str | None = None,
    prior_col: str | None = None,
    allow_replacement_sink: bool = False,
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
            concentration = (
                manifest.get("opportunity_shares", {})
                .get("pools", {})
                .get(pool_key or "", {})
                .get("concentration")
                or manifest.get("pools", {}).get(pool_key or "", {}).get("concentration")
                or manifest.get("rooms", {}).get(key, {}).get("concentration", 10.0)
            )
        shares = draw_dirichlet_shares(
            grp,
            float(concentration),
            rng=rng,
            prior_col=prior_col,
            allow_replacement_sink=allow_replacement_sink,
        )
        out.loc[grp.index, "allocated_volume"] = shares * team_volume
    out.attrs["replacement_sink_volume"] = float(
        max(float(team_volume) - float(out["allocated_volume"].sum()), 0.0)
    )
    return out
