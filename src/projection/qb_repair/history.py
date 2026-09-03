"""Leakage-safe QB season-rate history for priors and allocation fitting."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = REPO_ROOT / "data" / "raw" / "weekly_qb_repair_cache"
OUTPUT_CACHE = REPO_ROOT / "output" / "qb_repair" / "history"
EVAL_DIR = REPO_ROOT / "output"

RATE_STATS = (
    "attempts",
    "completions",
    "passing_yards",
    "passing_tds",
    "interceptions",
    "carries",
    "rushing_yards",
    "rushing_tds",
)


def _empty_season_frame() -> pd.DataFrame:
    cols = [
        "player_id",
        "season",
        "display_name",
        "team",
        "games",
        *RATE_STATS,
        "designed_carries",
        "scramble_carries",
        "designed_rushing_yards",
        "scramble_rushing_yards",
    ]
    return pd.DataFrame(columns=cols)


def _load_weekly_seasons() -> pd.DataFrame:
    path = CACHE_DIR / "qb_weekly.parquet"
    if not path.exists():
        return _empty_season_frame()
    weekly = pd.read_parquet(path)
    weekly = weekly[pd.to_numeric(weekly["week"], errors="coerce").between(1, 18)]
    name_col = (
        "player_display_name"
        if "player_display_name" in weekly.columns
        else "player_name"
    )
    team_col = "recent_team" if "recent_team" in weekly.columns else "team"
    agg = (
        weekly.groupby(["player_id", "season"], as_index=False)
        .agg(
            display_name=(name_col, "first"),
            team=(team_col, "first"),
            games=("week", "nunique"),
            **{stat: (stat, "sum") for stat in RATE_STATS if stat in weekly.columns},
        )
    )
    return agg


def _append_eval_season(frame: pd.DataFrame, season: int) -> pd.DataFrame:
    path = EVAL_DIR / f"fantasy_evaluation_{season}.csv"
    if not path.exists():
        return frame
    eval_df = pd.read_csv(path)
    qb = eval_df[eval_df["preseason_position"].astype(str).eq("QB")].copy()
    if qb.empty:
        return frame
    rows = []
    for _, row in qb.iterrows():
        games = float(pd.to_numeric(row.get("actual_games_played"), errors="coerce") or 0.0)
        if games <= 0:
            continue
        pid = str(row["player_id"])
        if ((frame["player_id"] == pid) & (frame["season"] == season)).any():
            continue
        entry = {
            "player_id": pid,
            "season": int(season),
            "display_name": row.get("display_name"),
            "team": row.get("team") or row.get("preseason_team"),
            "games": games,
        }
        for stat in RATE_STATS:
            entry[stat] = float(pd.to_numeric(row.get(stat), errors="coerce") or 0.0)
        rows.append(entry)
    if not rows:
        return frame
    return pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)


def _attach_designed_scramble(frame: pd.DataFrame) -> pd.DataFrame:
    path = CACHE_DIR / "pbp_rush_2023_2024.parquet"
    out = frame.copy()
    for col in (
        "designed_carries",
        "scramble_carries",
        "designed_rushing_yards",
        "scramble_rushing_yards",
    ):
        if col not in out.columns:
            out[col] = np.nan
    if not path.exists() or out.empty:
        return out
    pbp = pd.read_parquet(path)
    qb_ids = set(out["player_id"].astype(str))
    scramble = pbp[
        (pbp["qb_scramble"] == 1) & (pbp["rusher_player_id"].astype(str).isin(qb_ids))
    ]
    designed = pbp[
        (pbp["rush_attempt"] == 1)
        & (pbp["qb_scramble"] != 1)
        & (pbp["rusher_player_id"].astype(str).isin(qb_ids))
    ]
    sc = (
        scramble.groupby(["rusher_player_id", "season"], as_index=False)
        .agg(
            scramble_carries=("rush_attempt", "sum"),
            scramble_rushing_yards=("rushing_yards", "sum"),
        )
        .rename(columns={"rusher_player_id": "player_id"})
    )
    des = (
        designed.groupby(["rusher_player_id", "season"], as_index=False)
        .agg(
            designed_carries=("rush_attempt", "sum"),
            designed_rushing_yards=("rushing_yards", "sum"),
        )
        .rename(columns={"rusher_player_id": "player_id"})
    )
    out = out.drop(
        columns=[
            c
            for c in (
                "designed_carries",
                "scramble_carries",
                "designed_rushing_yards",
                "scramble_rushing_yards",
            )
            if c in out.columns
        ],
        errors="ignore",
    )
    out = out.merge(sc, on=["player_id", "season"], how="left")
    out = out.merge(des, on=["player_id", "season"], how="left")
    return out


def load_qb_season_history(*, refresh: bool = False) -> pd.DataFrame:
    """Return one row per QB-season with counting stats and optional rush splits."""
    OUTPUT_CACHE.mkdir(parents=True, exist_ok=True)
    cached = OUTPUT_CACHE / "qb_season_rates.parquet"
    if cached.exists() and not refresh:
        return pd.read_parquet(cached)

    frame = _load_weekly_seasons()
    if frame.empty and (CACHE_DIR / "qb_season_rates.parquet").exists():
        frame = pd.read_parquet(CACHE_DIR / "qb_season_rates.parquet")
        rename = {"player_display_name": "display_name", "recent_team": "team"}
        frame = frame.rename(columns={k: v for k, v in rename.items() if k in frame.columns})
    for season in (2023, 2024, 2025):
        frame = _append_eval_season(frame, season)
    frame = _attach_designed_scramble(frame)
    for stat in RATE_STATS:
        if stat not in frame.columns:
            frame[stat] = 0.0
        frame[stat] = pd.to_numeric(frame[stat], errors="coerce").fillna(0.0)
    frame["games"] = pd.to_numeric(frame["games"], errors="coerce").fillna(0.0)
    frame["player_id"] = frame["player_id"].astype(str)
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce").astype(int)
    frame.to_parquet(cached, index=False)
    return frame


def per_game_rates(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ``*_pg`` columns using games as the denominator."""
    out = frame.copy()
    games = pd.to_numeric(out["games"], errors="coerce").replace(0, np.nan)
    for stat in RATE_STATS:
        out[f"{stat}_pg"] = pd.to_numeric(out[stat], errors="coerce") / games
    for stat in (
        "designed_carries",
        "scramble_carries",
        "designed_rushing_yards",
        "scramble_rushing_yards",
    ):
        if stat in out.columns:
            out[f"{stat}_pg"] = pd.to_numeric(out[stat], errors="coerce") / games
    # Efficiency / rate constructs
    attempts = pd.to_numeric(out["attempts"], errors="coerce").replace(0, np.nan)
    carries = pd.to_numeric(out["carries"], errors="coerce").replace(0, np.nan)
    out["completion_pct"] = pd.to_numeric(out["completions"], errors="coerce") / attempts
    out["yards_per_attempt"] = pd.to_numeric(out["passing_yards"], errors="coerce") / attempts
    out["pass_td_rate"] = pd.to_numeric(out["passing_tds"], errors="coerce") / attempts
    out["int_rate"] = pd.to_numeric(out["interceptions"], errors="coerce") / attempts
    out["yards_per_carry"] = pd.to_numeric(out["rushing_yards"], errors="coerce") / carries
    out["rush_td_rate"] = pd.to_numeric(out["rushing_tds"], errors="coerce") / carries
    return out


def history_before(frame: pd.DataFrame, season: int) -> pd.DataFrame:
    """Strict leakage boundary: only seasons strictly before ``season``."""
    return frame[pd.to_numeric(frame["season"], errors="coerce") < int(season)].copy()
