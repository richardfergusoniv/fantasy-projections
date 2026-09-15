"""Load sealed season projections as team volume + player role shares.

Uses the public sealed long board (default ``v2_baseline_20260830``). Does not
mutate the namespace, pointer, or freeze knobs. ADP and season-long Vegas are
not read.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.projection.weekly_latent.constants import (
    CONVERSION_RATES,
    GAMES_PER_SEASON,
    PLAYER_SHARE_POOLS,
    SKILL_POSITIONS,
    TEAM_VOLUME_PG_COLUMNS,
    normalize_team_abbr,
)


def default_sealed_projections_path(repo_root: str | Path) -> Path:
    return (
        Path(repo_root)
        / "draft_assistant"
        / "data"
        / "releases"
        / "v2_baseline_20260830"
        / "projections_2026.csv"
    )


def load_long_projections(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    need = {"player_id", "team", "position", "stat", "pred_season"}
    missing = need - set(frame.columns)
    if missing:
        raise ValueError(f"projections missing columns: {missing}")
    frame = frame.copy()
    frame["team"] = frame["team"].map(normalize_team_abbr)
    frame["position"] = frame["position"].astype(str).str.upper()
    frame = frame[frame["position"].isin(SKILL_POSITIONS)].copy()
    return frame


def team_season_volume(long_board: pd.DataFrame) -> pd.DataFrame:
    """Season team volume = sealed per-game team pred × 17 scheduled games."""
    cols = ["team", *TEAM_VOLUME_PG_COLUMNS.values()]
    present = [c for c in cols if c in long_board.columns]
    if "team" not in present:
        raise ValueError("projections have no team column")
    teams = long_board[present].drop_duplicates("team").copy()
    for out_name, pg_col in TEAM_VOLUME_PG_COLUMNS.items():
        if pg_col not in teams.columns:
            raise ValueError(f"projections missing team volume column {pg_col}")
        teams[out_name] = (
            pd.to_numeric(teams[pg_col], errors="coerce").fillna(0.0) * GAMES_PER_SEASON
        )
    return teams[["team", *TEAM_VOLUME_PG_COLUMNS.keys()]].reset_index(drop=True)


def player_season_table(long_board: pd.DataFrame) -> pd.DataFrame:
    """Wide player-season totals from the long sealed board."""
    identity_cols = ["player_id", "display_name", "team", "position"]
    frame = long_board.copy()
    if "display_name" not in frame.columns:
        frame["display_name"] = pd.NA
    extras = [
        c
        for c in ("depth_rank", "role", "projected_games", "projected_games_raw")
        if c in frame.columns
    ]
    ident = frame[identity_cols + extras].drop_duplicates("player_id").reset_index(drop=True)
    wide = frame.pivot_table(
        index="player_id",
        columns="stat",
        values="pred_season",
        aggfunc="sum",
    ).reset_index()
    wide.columns = [str(c) for c in wide.columns]
    out = ident.merge(wide, on="player_id", how="left")
    needed = set(PLAYER_SHARE_POOLS) | {num for num, _opp in CONVERSION_RATES.items()}
    needed.update(opp for _num, opp in CONVERSION_RATES.values())
    for stat in needed:
        if stat not in out.columns:
            out[stat] = 0.0
        out[stat] = pd.to_numeric(out[stat], errors="coerce").fillna(0.0)
    return out


def role_shares(
    players: pd.DataFrame,
    team_volume: pd.DataFrame,
) -> pd.DataFrame:
    """Player share of each team pool, plus an explicit other residual.

    If named shares sum to more than 1, they are rescaled onto the simplex
    (other = 0). That preserves team-volume conservation and within-week share
    conservation; player-season totals then match the sealed board only when
    named volume was already ≤ the team pool.
    """
    vol = team_volume.rename(
        columns={name: f"pool_{name}" for name in TEAM_VOLUME_PG_COLUMNS}
    )
    merged = players.merge(vol, on="team", how="left")
    chunks: list[pd.DataFrame] = []
    for team, grp in merged.groupby("team", dropna=False):
        for player_stat, pool in PLAYER_SHARE_POOLS.items():
            pool_col = f"pool_{pool}"
            pool_total = float(grp[pool_col].iloc[0] or 0.0) if pool_col in grp.columns else 0.0
            raw = grp[player_stat].astype(float).clip(lower=0.0)
            named_sum = float(raw.sum())
            if pool_total <= 0:
                shares = raw * 0.0
                other = 1.0
                mode = "empty_pool"
            elif named_sum <= pool_total + 1e-12:
                shares = raw / pool_total
                other = max(0.0, 1.0 - float(shares.sum()))
                mode = "exact_plus_other"
            else:
                shares = raw / named_sum
                other = 0.0
                mode = "rescaled_overflow"
            tmp = pd.DataFrame(
                {
                    "player_id": grp["player_id"].to_numpy(),
                    "team": team,
                    "stat": player_stat,
                    "share": shares.to_numpy(),
                    "other_share": other,
                    "share_mode": mode,
                    "pool": pool,
                    "pool_total": pool_total,
                    "named_pred_sum": named_sum,
                }
            )
            chunks.append(tmp)
    return pd.concat(chunks, ignore_index=True)
