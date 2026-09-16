"""Lagged opponent / schedule-environment multipliers for Milestone 2.

These may move season team volume. They are not M1 renormalized weights.
Vegas lines, ADP, and same-week box scores are not inputs.
"""
from __future__ import annotations

import pandas as pd

from src.projection.weekly_latent.constants import (
    EPA_VOLUME_KAPPA,
    ENV_FACTOR_MAX,
    ENV_FACTOR_MIN,
    FORBIDDEN_MARKET_DRIVERS,
    FORBIDDEN_SAME_WEEK_TRAINING_FEATURES,
    INTL_TRAVEL_MULT,
    M2_SCHEDULE_ENV_AVAILABLE_AT,
    REST_LONG_DAYS,
    REST_LONG_MULT,
    REST_NORMAL_MULT,
    REST_SHORT_DAYS,
    REST_SHORT_MULT,
    max_available_at,
    normalize_team_abbr,
)


def refuse_forbidden_m2_columns(frame: pd.DataFrame, *, where: str) -> None:
    """Fail closed if same-week volume or market drivers are on the frame."""
    cols = set(frame.columns)
    leaked = sorted(cols & FORBIDDEN_SAME_WEEK_TRAINING_FEATURES)
    if leaked:
        raise ValueError(
            f"same-week realized volume leaked onto {where}: {leaked}"
        )
    market = sorted(cols & FORBIDDEN_MARKET_DRIVERS)
    if market:
        raise ValueError(
            f"ADP / season-long Vegas / weekly lines cannot drive M2 ({where}): {market}"
        )


def rest_multiplier(rest_days: object) -> float:
    """Short week trims volume; extra rest is a small boost; missing → 1.0."""
    if rest_days is None or (isinstance(rest_days, float) and pd.isna(rest_days)):
        return REST_NORMAL_MULT
    try:
        days = float(rest_days)
    except (TypeError, ValueError):
        return REST_NORMAL_MULT
    if pd.isna(days):
        return REST_NORMAL_MULT
    if days < REST_SHORT_DAYS:
        return REST_SHORT_MULT
    if days >= REST_LONG_DAYS:
        return REST_LONG_MULT
    return REST_NORMAL_MULT


def travel_multiplier(is_international: object) -> float:
    try:
        flag = int(is_international or 0)
    except (TypeError, ValueError):
        flag = 0
    return INTL_TRAVEL_MULT if flag == 1 else 1.0


def epa_to_volume_factor(
    epa: float,
    league_mean: float,
    league_sd: float,
    *,
    kappa: float = EPA_VOLUME_KAPPA,
) -> float:
    """Higher EPA allowed (worse defense) → factor > 1 for the offense facing them."""
    sd = float(league_sd)
    if sd <= 1e-12:
        return 1.0
    z = (float(epa) - float(league_mean)) / sd
    return float(max(ENV_FACTOR_MIN, min(ENV_FACTOR_MAX, 1.0 + kappa * z)))


def _league_moments(values: pd.Series) -> tuple[float, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return 0.0, 1.0
    mean = float(numeric.mean())
    sd = float(numeric.std(ddof=0))
    return mean, (sd if sd > 1e-12 else 1.0)


def factors_from_def_epa(priors: pd.DataFrame) -> pd.DataFrame:
    """Convert prior-season defensive EPA allowed into opponent pass/rush factors."""
    refuse_forbidden_m2_columns(priors, where="defensive EPA priors")
    frame = priors.copy()
    if "opponent" not in frame.columns:
        if "team" in frame.columns:
            frame = frame.rename(columns={"team": "opponent"})
        else:
            raise ValueError("defensive EPA priors need team/opponent")
    frame["opponent"] = frame["opponent"].map(normalize_team_abbr)
    pass_col = (
        "def_pass_epa_allowed"
        if "def_pass_epa_allowed" in frame.columns
        else "pass_epa_allowed"
    )
    rush_col = (
        "def_rush_epa_allowed"
        if "def_rush_epa_allowed" in frame.columns
        else "rush_epa_allowed"
    )
    if pass_col not in frame.columns or rush_col not in frame.columns:
        raise ValueError("defensive EPA priors need pass and rush EPA allowed columns")
    pass_mean, pass_sd = _league_moments(frame[pass_col])
    rush_mean, rush_sd = _league_moments(frame[rush_col])
    frame["pass_factor"] = [
        epa_to_volume_factor(v, pass_mean, pass_sd) if pd.notna(v) else 1.0
        for v in pd.to_numeric(frame[pass_col], errors="coerce")
    ]
    frame["rush_factor"] = [
        epa_to_volume_factor(v, rush_mean, rush_sd) if pd.notna(v) else 1.0
        for v in pd.to_numeric(frame[rush_col], errors="coerce")
    ]
    if "available_at" not in frame.columns:
        frame["available_at"] = None
    return frame[
        [c for c in ("opponent", "pass_factor", "rush_factor", "available_at", pass_col, rush_col) if c in frame.columns]
    ].drop_duplicates("opponent")


def attach_environment(team_weeks: pd.DataFrame) -> pd.DataFrame:
    """Add rest × international environment multipliers. Not Vegas, not box scores."""
    refuse_forbidden_m2_columns(team_weeks, where="M2 team-weeks before env attach")
    out = team_weeks.copy()
    rest_src = (
        out["rest_days"] if "rest_days" in out.columns else pd.Series(pd.NA, index=out.index)
    )
    intl_src = (
        out["is_international"] if "is_international" in out.columns else pd.Series(0, index=out.index)
    )
    out["rest_mult"] = [rest_multiplier(v) for v in rest_src]
    out["travel_mult"] = [travel_multiplier(v) for v in intl_src]
    bye = out["is_bye"].eq(1) if "is_bye" in out.columns else pd.Series(False, index=out.index)
    out.loc[bye, "rest_mult"] = 0.0
    out.loc[bye, "travel_mult"] = 0.0
    out["env_mult"] = out["rest_mult"].astype(float) * out["travel_mult"].astype(float)
    out["env_available_at"] = M2_SCHEDULE_ENV_AVAILABLE_AT
    return out


def _stamp_or_none(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text in {"", "nan", "None", "<NA>", "NaT"}:
        return None
    return text


def stamp_available_at(
    frame: pd.DataFrame,
    *cutoffs: str | None,
    prior_column: str = "prior_available_at",
) -> pd.DataFrame:
    """Advance ``available_at`` to the latest attached feature vintage.

    Takes the max across board, schedule-env and per-row prior: a row is
    knowable only once its last input is. Never moves a stamp earlier.
    """
    out = frame.copy()
    row_priors = (
        out[prior_column] if prior_column in out.columns else pd.Series([None] * len(out), index=out.index)
    )
    env_stamp = (
        out["env_available_at"]
        if "env_available_at" in out.columns
        else pd.Series([None] * len(out), index=out.index)
    )
    stamped = []
    for prior, env in zip(row_priors, env_stamp):
        stamped.append(
            max_available_at(*cutoffs, _stamp_or_none(prior), _stamp_or_none(env))
        )
    out["available_at"] = stamped
    out["available_at_board"] = cutoffs[0] if cutoffs else None
    return out
