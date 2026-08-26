"""V3 generative composition: team draw -> shares -> conversions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.projection.models.opportunity_shares import allocate_opportunities
from src.projection.models.receiving import draw_receiving_line
from src.projection.models.rushing import draw_rushing_line
from src.projection.models.passing import draw_passing_line
from src.projection.transitions import SEASON_GAMES


def reconcile_v3_generative(
    players: pd.DataFrame,
    team_environment: pd.DataFrame,
    *,
    rng: np.random.Generator,
    share_manifest: dict | None = None,
) -> pd.DataFrame:
    """One simulation draw through the v3 opportunity + conversion graph."""
    rows = []
    env = team_environment.set_index("team") if "team" in team_environment.columns else team_environment
    for team, room in players.groupby("team", observed=True):
        team_row = env.loc[team] if team in env.index else None
        pass_att = float(team_row.get("team_pass_attempts_mean", 600)) / SEASON_GAMES if team_row is not None else 35.0
        rush_att = float(team_row.get("team_carries_mean", 400)) / SEASON_GAMES if team_row is not None else 25.0
        qb = room[room["position"].eq("QB")]
        for _, qb_row in qb.iterrows():
            line = draw_passing_line(pass_att, rng=rng)
            line.update({"player_id": qb_row["player_id"], "position": "QB", "team": team})
            rows.append(line)
        recv = allocate_opportunities(
            room[room["position"].isin(["WR", "TE", "RB"]) & room["stat"].eq("targets")],
            pass_att * SEASON_GAMES,
            rng=rng,
            manifest=share_manifest,
        )
        for _, pl in recv.iterrows():
            if pl["stat"] != "targets":
                continue
            line = draw_receiving_line(pl["allocated_volume"] / SEASON_GAMES, rng=rng)
            line.update({"player_id": pl["player_id"], "position": pl["position"], "team": team})
            rows.append(line)
        rush = allocate_opportunities(
            room[room["position"].eq("RB") & room["stat"].eq("carries")],
            rush_att * SEASON_GAMES,
            rng=rng,
            manifest=share_manifest,
        )
        for _, pl in rush.iterrows():
            line = draw_rushing_line(pl["allocated_volume"] / SEASON_GAMES, rng=rng)
            line.update({"player_id": pl["player_id"], "position": "RB", "team": team})
            rows.append(line)
    return pd.DataFrame(rows)
