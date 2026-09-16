"""Load lagged / preseason opponent defense priors for Milestone 2.

Never joins same-week ``team_attempts`` / ``team_carries`` / ``team_targets`` /
``team_air_yards``. Prefers prior-season defensive EPA (pbp or committed
fixture). ``projections.db`` is optional; the cloud VM usually lacks it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.projection.weekly_latent.constants import (
    DEFAULT_OPP_EPA_PRIOR_REL,
    FORBIDDEN_SAME_WEEK_TRAINING_FEATURES,
    PRIOR_SEASON_EPA_AVAILABLE_AT,
    SEASON_DEFAULT,
    normalize_team_abbr,
)
from src.projection.weekly_latent.environment import (
    factors_from_def_epa,
    refuse_forbidden_m2_columns,
)


def load_epa_prior_fixture(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    need = {"team", "def_pass_epa_allowed", "def_rush_epa_allowed"}
    missing = need - set(frame.columns)
    if missing:
        raise ValueError(f"EPA prior fixture missing columns: {missing}")
    refuse_forbidden_m2_columns(frame, where="EPA prior fixture")
    frame = frame.copy()
    frame["team"] = frame["team"].map(normalize_team_abbr)
    if "available_at" not in frame.columns:
        frame["available_at"] = PRIOR_SEASON_EPA_AVAILABLE_AT
    frame["available_at"] = frame["available_at"].fillna(PRIOR_SEASON_EPA_AVAILABLE_AT)
    return frame


def _pbp_season_defense_epa(conn: Any, season_observed: int) -> pd.DataFrame | None:
    """Prior-season defensive EPA/play from pbp, keyed by defending team.

    Same grain idea as ``team_season_defense_epa`` in data_prep.py, inlined so
    this shadow module does not import the polars/pbp pipeline. Regular season
    only. Does not select team_attempts.
    """
    tables = {
        row[0]
        for row in conn.execute("select name from sqlite_master where type='table'").fetchall()
    }
    if "pbp" not in tables:
        return None
    weekly = pd.read_sql_query(
        "select season, defteam as team, pass_attempt, rush_attempt, epa "
        "from pbp where season = ? and season_type = 'REG' "
        "and defteam is not null and epa is not null "
        "and (pass_attempt = 1 or rush_attempt = 1)",
        conn,
        params=(int(season_observed),),
    )
    if weekly.empty:
        return None
    leaked = set(weekly.columns) & FORBIDDEN_SAME_WEEK_TRAINING_FEATURES
    if leaked:
        raise ValueError(f"pbp prior query returned forbidden columns: {sorted(leaked)}")
    pass_epa = (
        weekly[weekly["pass_attempt"].eq(1)]
        .groupby("team")["epa"]
        .mean()
        .rename("def_pass_epa_allowed")
        .reset_index()
    )
    rush_epa = (
        weekly[weekly["rush_attempt"].eq(1)]
        .groupby("team")["epa"]
        .mean()
        .rename("def_rush_epa_allowed")
        .reset_index()
    )
    out = pass_epa.merge(rush_epa, on="team", how="outer")
    out["season_observed"] = int(season_observed)
    out["available_at"] = PRIOR_SEASON_EPA_AVAILABLE_AT
    out["source"] = f"projections.db pbp REG {season_observed} mean epa by defteam"
    return out


def try_load_db_def_epa(db_path: str | Path, *, season_observed: int) -> tuple[pd.DataFrame | None, str]:
    path = Path(db_path)
    if not path.is_file():
        return None, f"no projections.db at {path}"
    try:
        import sqlite3

        conn = sqlite3.connect(str(path))
        try:
            frame = _pbp_season_defense_epa(conn, season_observed)
        finally:
            conn.close()
    except Exception as exc:
        return None, f"db defensive EPA skipped: {exc}"
    if frame is None or frame.empty:
        return None, "projections.db has no usable pbp defensive EPA; using fixture"
    refuse_forbidden_m2_columns(frame, where="db defensive EPA")
    return frame, f"loaded {len(frame)} team priors from pbp season {season_observed}"


def load_m2_opponent_priors(
    *,
    repo_root: str | Path,
    db_path: str | Path,
    fixture_path: str | Path | None = None,
    season: int = SEASON_DEFAULT,
    override: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, str]:
    """Return opponent pass/rush factors + available_at. Fixture if DB missing."""
    if override is not None:
        refuse_forbidden_m2_columns(override, where="override opponent priors")
        if {"pass_factor", "rush_factor"}.issubset(override.columns):
            frame = override.copy()
            if "opponent" not in frame.columns and "team" in frame.columns:
                frame = frame.rename(columns={"team": "opponent"})
            if "available_at" not in frame.columns:
                frame["available_at"] = PRIOR_SEASON_EPA_AVAILABLE_AT
            return frame, "caller-supplied opponent priors"
        factors = factors_from_def_epa(override)
        return factors, "caller-supplied defensive EPA converted to factors"

    observed = int(season) - 1
    db_frame, db_note = try_load_db_def_epa(db_path, season_observed=observed)
    if db_frame is not None:
        return factors_from_def_epa(db_frame), db_note

    fixture = Path(fixture_path or (Path(repo_root) / DEFAULT_OPP_EPA_PRIOR_REL))
    if not fixture.is_file():
        raise FileNotFoundError(
            f"M2 opponent EPA fixture not found at {fixture}. "
            "On Windows, set FANTASY_PROJECTIONS_DB_PATH="
            "D:\\fantasy-projections-data\\projections.db to build priors "
            "from pbp, or pass --opp-epa-priors."
        )
    loaded = load_epa_prior_fixture(fixture)
    factors = factors_from_def_epa(loaded)
    return factors, f"{db_note}; fixture {fixture.as_posix()}"


def build_as_of_priors(
    weekly_epa: pd.DataFrame,
    *,
    as_of_week: int,
    prior_season: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Rolling-origin opponent EPA: weeks strictly before ``as_of_week``.

    Same-week rows in ``weekly_epa`` are dropped, not averaged. If a caller
    marks ``include_current_week`` somehow by passing week >= as_of_week as
    the only history, we fall back to prior-season rather than leak.
    """
    refuse_forbidden_m2_columns(weekly_epa, where="as-of weekly EPA")
    if "week" not in weekly_epa.columns:
        raise ValueError("as-of weekly EPA needs a week column")
    past = weekly_epa[pd.to_numeric(weekly_epa["week"], errors="coerce") < int(as_of_week)].copy()
    if past.empty:
        if prior_season is None or prior_season.empty:
            raise ValueError(
                f"no lagged EPA before week {as_of_week} and no prior-season fallback"
            )
        out = factors_from_def_epa(prior_season)
        out["as_of_week"] = int(as_of_week)
        out["prior_source"] = "prior_season_fallback"
        return out
    team_col = "opponent" if "opponent" in past.columns else "team"
    pass_src = "def_pass_epa_allowed" if "def_pass_epa_allowed" in past.columns else "pass_epa"
    rush_src = "def_rush_epa_allowed" if "def_rush_epa_allowed" in past.columns else "rush_epa"
    if pass_src not in past.columns or rush_src not in past.columns:
        raise ValueError("as-of weekly EPA needs pass/rush EPA columns")
    collapsed = (
        past.groupby(team_col, as_index=False)
        .agg({pass_src: "mean", rush_src: "mean"})
        .rename(
            columns={
                team_col: "team",
                pass_src: "def_pass_epa_allowed",
                rush_src: "def_rush_epa_allowed",
            }
        )
    )
    if "gameday" in past.columns:
        collapsed["available_at"] = (
            pd.to_datetime(past["gameday"]).max().strftime("%Y-%m-%dT00:00:00+00:00")
        )
    elif "available_at" in past.columns:
        collapsed["available_at"] = past["available_at"].max()
    else:
        collapsed["available_at"] = PRIOR_SEASON_EPA_AVAILABLE_AT
    out = factors_from_def_epa(collapsed)
    out["as_of_week"] = int(as_of_week)
    out["prior_source"] = "lagged_weeks_before_as_of"
    return out


def assert_priors_are_as_of(priors: pd.DataFrame, *, as_of_week: int) -> None:
    """Fail closed if a prior row is labeled with week >= the target week."""
    if "week" not in priors.columns:
        return
    weeks = pd.to_numeric(priors["week"], errors="coerce")
    leaked = priors[weeks >= int(as_of_week)]
    if len(leaked):
        raise ValueError(
            f"opponent priors include week >= as_of_week={as_of_week}; "
            "same-week outcomes cannot enter M2 features"
        )
