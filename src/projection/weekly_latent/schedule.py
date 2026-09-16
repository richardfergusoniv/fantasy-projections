"""2026 regular-season schedule → team-weeks, including bye rows.

Pandas analogue of ``explode_schedules_to_team_weeks`` that does **not**
require Vegas spread/total and does **not** join same-week ``team_attempts`` /
``team_carries``. Those realized columns are the PR #70 defect in
``add_team_pass_rate``; M1 never calls that helper.

Schedule scaffolding here has no cutoff column yet. ``allocate_team_weeks``
is the single M1 writer that stamps ``available_at`` (preseason snapshot).
M2 must overwrite that cutoff when it attaches lagged opponent priors or
in-season updates.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.projection.weekly_latent.constants import (
    GAMES_PER_SEASON,
    OPPONENT_FACTOR_MAX,
    OPPONENT_FACTOR_MIN,
    REG_WEEKS,
    TEAM_DIVISIONS,
    normalize_team_abbr,
    opponent_shrinkage_lambda,
)

SCHEDULE_COLUMNS = (
    "game_id",
    "season",
    "week",
    "gameday",
    "weekday",
    "gametime",
    "away_team",
    "home_team",
    "location",
    "away_rest",
    "home_rest",
    "div_game",
    "roof",
    "surface",
    "stadium_id",
    "stadium",
)


def load_schedule_csv(path: str | Path) -> pd.DataFrame:
    """Load a slim REG schedule. Scores, lines, and QB names are not required."""
    frame = pd.read_csv(path)
    missing = [c for c in ("season", "week", "home_team", "away_team") if c not in frame.columns]
    if missing:
        raise ValueError(f"schedule missing columns: {missing}")
    if "game_type" in frame.columns:
        frame = frame[frame["game_type"].astype(str).str.upper().eq("REG")].copy()
    frame["home_team"] = frame["home_team"].map(normalize_team_abbr)
    frame["away_team"] = frame["away_team"].map(normalize_team_abbr)
    frame["week"] = frame["week"].astype(int)
    frame["season"] = frame["season"].astype(int)
    if "location" not in frame.columns:
        frame["location"] = "Home"
    return frame.reset_index(drop=True)


def _side_rows(schedule: pd.DataFrame, *, home: bool) -> pd.DataFrame:
    team_col = "home_team" if home else "away_team"
    opp_col = "away_team" if home else "home_team"
    rest_col = "home_rest" if home else "away_rest"
    rows = pd.DataFrame(
        {
            "season": schedule["season"],
            "week": schedule["week"],
            "game_id": schedule["game_id"] if "game_id" in schedule.columns else None,
            "team": schedule[team_col],
            "opponent": schedule[opp_col],
            "is_home": 1 if home else 0,
            "is_listed_home": 1 if home else 0,
            "location": schedule["location"],
            "rest_days": schedule[rest_col] if rest_col in schedule.columns else pd.NA,
            "div_game": schedule["div_game"] if "div_game" in schedule.columns else pd.NA,
            "roof": schedule["roof"] if "roof" in schedule.columns else pd.NA,
            "surface": schedule["surface"] if "surface" in schedule.columns else pd.NA,
            "stadium": schedule["stadium"] if "stadium" in schedule.columns else pd.NA,
            "stadium_id": schedule["stadium_id"] if "stadium_id" in schedule.columns else pd.NA,
            "gameday": schedule["gameday"] if "gameday" in schedule.columns else pd.NA,
            "weekday": schedule["weekday"] if "weekday" in schedule.columns else pd.NA,
            "gametime": schedule["gametime"] if "gametime" in schedule.columns else pd.NA,
        }
    )
    return rows


def explode_team_weeks(
    schedule: pd.DataFrame,
    *,
    require_full_season: bool = True,
) -> pd.DataFrame:
    """One row per team-week including bye weeks (active=0)."""
    if schedule.empty:
        raise ValueError("schedule is empty")
    played = pd.concat(
        [_side_rows(schedule, home=True), _side_rows(schedule, home=False)],
        ignore_index=True,
    )
    season = int(played["season"].iloc[0])
    teams = sorted(set(played["team"].dropna().tolist()))
    grid = pd.MultiIndex.from_product(
        [teams, list(REG_WEEKS)], names=["team", "week"]
    ).to_frame(index=False)
    grid["season"] = season
    merged = grid.merge(played, on=["season", "team", "week"], how="left")
    merged["is_bye"] = merged["opponent"].isna().astype(int)
    merged["active"] = (1 - merged["is_bye"]).astype(int)
    loc = merged["location"].fillna("Home").astype(str)
    merged["is_neutral"] = loc.str.lower().ne("home").astype(int)
    # 2026 neutrals in the nflverse slate are international sites.
    merged["is_international"] = merged["is_neutral"]
    merged["is_home_for_mult"] = (
        (merged["is_home"].fillna(0).astype(int).eq(1)) & merged["is_neutral"].eq(0)
    ).astype(int)
    merged["shrinkage_lambda"] = merged["week"].map(opponent_shrinkage_lambda)
    conf_div = pd.DataFrame(
        [
            {"team": abbr, "conference": conf, "division": div}
            for abbr, (conf, div) in TEAM_DIVISIONS.items()
        ]
    )
    merged = merged.merge(conf_div, on="team", how="left")
    opp_meta = conf_div.rename(
        columns={
            "team": "opponent",
            "conference": "opp_conference",
            "division": "opp_division",
        }
    )
    merged = merged.merge(opp_meta, on="opponent", how="left")
    computed_div = (
        merged["conference"].notna()
        & merged["opp_conference"].notna()
        & merged["conference"].eq(merged["opp_conference"])
        & merged["division"].eq(merged["opp_division"])
    )
    if "div_game" in merged.columns:
        merged["is_division"] = (
            pd.to_numeric(merged["div_game"], errors="coerce").fillna(0).astype(int).eq(1)
            | computed_div
        ).astype(int)
    else:
        merged["is_division"] = computed_div.astype(int)
    merged["A_team_w"] = merged["active"].astype(float)
    n_games = merged.groupby("team")["active"].sum()
    if require_full_season:
        bad = n_games[n_games != GAMES_PER_SEASON]
        if not bad.empty:
            raise ValueError(
                f"teams without exactly {GAMES_PER_SEASON} scheduled games: "
                + bad.to_dict().__repr__()
            )
    return merged.sort_values(["team", "week"]).reset_index(drop=True)


def attach_opponent_priors(
    team_weeks: pd.DataFrame,
    priors: pd.DataFrame | None,
) -> pd.DataFrame:
    """Join optional opponent pass/rush factors centered at 1.0.

    Priors must be lagged / preseason (prior-season defense). Never same-week
    realized ``team_attempts`` / ``team_carries``.
    """
    out = team_weeks.copy()
    out["opp_pass_factor"] = 1.0
    out["opp_rush_factor"] = 1.0
    if priors is None or priors.empty:
        return out
    need = {"opponent", "pass_factor", "rush_factor"}
    frame = priors.rename(columns={"team": "opponent"} if "team" in priors.columns else {})
    if not {"opponent"}.issubset(frame.columns):
        raise ValueError("opponent priors need an opponent/team column")
    frame["opponent"] = frame["opponent"].map(normalize_team_abbr)
    if "pass_factor" not in frame.columns:
        frame["pass_factor"] = 1.0
    if "rush_factor" not in frame.columns:
        frame["rush_factor"] = 1.0
    extra = set(frame.columns) & need
    if extra != need:
        pass
    slim = (
        frame[["opponent", "pass_factor", "rush_factor"]]
        .drop_duplicates("opponent")
        .rename(columns={"pass_factor": "opp_pass_factor", "rush_factor": "opp_rush_factor"})
    )
    out = out.drop(columns=["opp_pass_factor", "opp_rush_factor"])
    out = out.merge(slim, on="opponent", how="left")
    out["opp_pass_factor"] = (
        pd.to_numeric(out["opp_pass_factor"], errors="coerce")
        .fillna(1.0)
        .clip(OPPONENT_FACTOR_MIN, OPPONENT_FACTOR_MAX)
    )
    out["opp_rush_factor"] = (
        pd.to_numeric(out["opp_rush_factor"], errors="coerce")
        .fillna(1.0)
        .clip(OPPONENT_FACTOR_MIN, OPPONENT_FACTOR_MAX)
    )
    return out
