"""Week-varying player availability A_i,w for Milestone 3.

Bye forces A=0. A_base is 1.0: season exposure (projected_games) is
already in role shares from pred_season. Same-week volume and ADP/Vegas
are not inputs. Overrides fail closed without a pre-kickoff available_at.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.projection.weekly_latent.constants import (
    AVAIL_BACKUP_MIN_DEPTH_RANK,
    AVAIL_REST_NORMAL,
    AVAIL_REST_SHORT_BACKUP,
    AVAIL_REST_SHORT_STARTER,
    REST_SHORT_DAYS,
    format_available_at,
    max_available_at,
    parse_available_at,
)
from src.projection.weekly_latent.environment import (
    refuse_forbidden_m2_columns,
    _stamp_or_none,
)


def base_availability(projected_games: object) -> float:
    """Week-level baseline A. Always 1.0.

    Role shares are built from ``pred_season`` (``pred_pg × projected_games``).
    Re-applying ``projected_games / 17`` here would square the exposure
    discount. Sit, rest, and bye adjust A on top of this baseline.
    ``projected_games`` is accepted and ignored so callers can keep passing it.
    """
    _ = projected_games
    return 1.0


def _parse_cutoff(value: object) -> datetime | None:
    stamp = _stamp_or_none(value)
    if stamp is None:
        return None
    try:
        return parse_available_at(stamp)
    except (TypeError, ValueError):
        return None


def kickoff_at_from_row(
    *,
    kickoff_at: object = None,
    gameday: object = None,
    gametime: object = None,
) -> datetime | None:
    """Kickoff cutoff for an availability override. Prefer ``kickoff_at``."""
    parsed = _parse_cutoff(kickoff_at)
    if parsed is not None:
        return parsed
    day = _stamp_or_none(gameday)
    if day is None:
        return None
    time_text = _stamp_or_none(gametime)
    if time_text and "T" not in day:
        if len(time_text) == 5:
            time_text = f"{time_text}:00"
        combined = f"{day}T{time_text}"
    else:
        combined = day
    return _parse_cutoff(combined)


def rest_availability_mult(rest_days: object, *, depth_rank: object = 1) -> float:
    """Short rest raises sit risk; backups more than starters. Not team volume."""
    if rest_days is None or (isinstance(rest_days, float) and pd.isna(rest_days)):
        return AVAIL_REST_NORMAL
    try:
        days = float(rest_days)
    except (TypeError, ValueError):
        return AVAIL_REST_NORMAL
    if pd.isna(days) or days >= REST_SHORT_DAYS:
        return AVAIL_REST_NORMAL
    rank = 1
    try:
        if depth_rank is not None and not (isinstance(depth_rank, float) and pd.isna(depth_rank)):
            rank = int(depth_rank)
    except (TypeError, ValueError):
        rank = 1
    if rank >= AVAIL_BACKUP_MIN_DEPTH_RANK:
        return AVAIL_REST_SHORT_BACKUP
    return AVAIL_REST_SHORT_STARTER


def attach_player_availability(
    weekly: pd.DataFrame,
    *,
    overrides: pd.DataFrame | None = None,
    board_available_at: str | None = None,
) -> pd.DataFrame:
    """Add A_i,w. Bye wins over any override. Overrides must be lagged / as-of."""
    refuse_forbidden_m2_columns(weekly, where="M3 player-weeks before availability")
    out = weekly.copy()
    rest = (
        out["rest_days"] if "rest_days" in out.columns else pd.Series(pd.NA, index=out.index)
    )
    depth = (
        out["depth_rank"] if "depth_rank" in out.columns else pd.Series(1, index=out.index)
    )
    # Exposure is in the role share (pred_season). Do not scale A by games/17.
    out["A_base"] = 1.0
    out["avail_rest_mult"] = [
        rest_availability_mult(r, depth_rank=d) for r, d in zip(rest, depth)
    ]
    out["A_i_w"] = (out["A_base"] * out["avail_rest_mult"]).clip(lower=0.0, upper=1.0)
    board_stamp = board_available_at
    if board_stamp is None and "available_at_board" in out.columns:
        present = [s for s in out["available_at_board"].tolist() if _stamp_or_none(s)]
        board_stamp = present[0] if present else None
    env_stamp_col = (
        out["env_available_at"]
        if "env_available_at" in out.columns
        else pd.Series([None] * len(out), index=out.index)
    )
    out["avail_available_at"] = [
        max_available_at(*[s for s in (board_stamp, _stamp_or_none(env)) if s])
        if board_stamp or _stamp_or_none(env)
        else None
        for env in env_stamp_col
    ]
    if overrides is not None and not overrides.empty:
        refuse_forbidden_m2_columns(overrides, where="M3 availability overrides")
        need = {"player_id", "week", "A_i_w", "available_at"}
        missing = need - set(overrides.columns)
        if missing:
            raise ValueError(f"availability overrides missing columns: {sorted(missing)}")
        slim = overrides[["player_id", "week", "A_i_w", "available_at"]].copy()
        slim["player_id"] = slim["player_id"].astype(str)
        slim["week"] = slim["week"].astype(int)
        slim["A_i_w"] = pd.to_numeric(slim["A_i_w"], errors="coerce").clip(0.0, 1.0)
        slim = slim.rename(columns={"A_i_w": "A_override", "available_at": "override_available_at"})
        out["player_id"] = out["player_id"].astype(str)
        out["week"] = out["week"].astype(int)
        merge_on = ["player_id", "week"]
        if "season" in out.columns:
            out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
            if "season" in overrides.columns:
                slim["season"] = pd.to_numeric(overrides["season"], errors="coerce").astype("Int64")
            else:
                seasons = out["season"].dropna().unique()
                if len(seasons) > 1:
                    raise ValueError(
                        "availability overrides missing season; weekly frame has multiple seasons"
                    )
                if len(seasons) == 1:
                    slim["season"] = int(seasons[0])
            if "season" in slim.columns:
                merge_on = ["player_id", "season", "week"]
        if "override_available_at" in out.columns:
            out = out.drop(columns=["override_available_at"])
        if "A_override" in out.columns:
            out = out.drop(columns=["A_override"])
        out = out.merge(slim, on=merge_on, how="left")
        has_override = out["A_override"].notna()
        kickoff_src = (
            out["kickoff_at"] if "kickoff_at" in out.columns else pd.Series(pd.NA, index=out.index)
        )
        gameday_src = (
            out["gameday"] if "gameday" in out.columns else pd.Series(pd.NA, index=out.index)
        )
        gametime_src = (
            out["gametime"] if "gametime" in out.columns else pd.Series(pd.NA, index=out.index)
        )
        for idx in out.index[has_override]:
            as_of = _parse_cutoff(out.at[idx, "override_available_at"])
            if as_of is None:
                raise ValueError("availability override is missing available_at")
            kickoff = kickoff_at_from_row(
                kickoff_at=kickoff_src.at[idx],
                gameday=gameday_src.at[idx],
                gametime=gametime_src.at[idx],
            )
            if kickoff is None:
                raise ValueError(
                    "availability override cannot be validated: missing kickoff_at/gameday"
                )
            if as_of > kickoff:
                raise ValueError(
                    f"available_at {format_available_at(as_of)} is after "
                    f"kickoff_at {format_available_at(kickoff)}"
                )
        out.loc[has_override, "A_i_w"] = out.loc[has_override, "A_override"].astype(float)
        override_stamp = [
            _stamp_or_none(s) for s in out.get("override_available_at", pd.Series(pd.NA, index=out.index))
        ]
        mixed = []
        for current, over, has in zip(out["avail_available_at"], override_stamp, has_override):
            if has and over:
                mixed.append(over)
            else:
                mixed.append(current)
        out["avail_available_at"] = mixed
    bye = out["is_bye"].eq(1) if "is_bye" in out.columns else pd.Series(False, index=out.index)
    out.loc[bye, "A_i_w"] = 0.0
    out.loc[bye, "avail_rest_mult"] = 0.0
    team_active = (
        out["A_team_w"].astype(float) if "A_team_w" in out.columns else pd.Series(1.0, index=out.index)
    )
    out["A_i_w"] = out["A_i_w"].astype(float) * team_active.clip(lower=0.0, upper=1.0)
    out["A_i_w"] = out["A_i_w"].clip(lower=0.0, upper=1.0)
    return out


def stamp_player_available_at(frame: pd.DataFrame) -> pd.DataFrame:
    """Advance row available_at via max of attached vintages. Never earlier."""
    out = frame.copy()
    stamped = []
    for rec in out.itertuples(index=False):
        parts = [
            _stamp_or_none(getattr(rec, "available_at", None)),
            _stamp_or_none(getattr(rec, "available_at_board", None)),
            _stamp_or_none(getattr(rec, "env_available_at", None)),
            _stamp_or_none(getattr(rec, "prior_available_at", None)),
            _stamp_or_none(getattr(rec, "avail_available_at", None)),
            _stamp_or_none(getattr(rec, "conv_available_at", None)),
        ]
        present = [p for p in parts if p]
        if not present:
            raise ValueError("player-week row has no available_at vintage")
        stamped.append(max_available_at(*present))
    out["available_at"] = stamped
    return out
