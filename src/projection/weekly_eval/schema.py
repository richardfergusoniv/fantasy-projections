"""Minimal timestamped Vegas weekly prop snapshot schema (Role 2 only)."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.projection.weekly_eval.errors import (
    MissingAsOfError,
    MissingKickoffError,
    PostKickoffSnapshotError,
)
from src.projection.weekly_eval.leakage import assert_prediction_frame_has_no_outcomes

ALLOWED_MARKETS: frozenset[str] = frozenset(
    {
        "pass_yards",
        "pass_tds",
        "pass_attempts",
        "pass_completions",
        "pass_ints",
        "rush_yards",
        "rush_tds",
        "rush_attempts",
        "rec_yards",
        "rec_tds",
        "receptions",
        "targets",
    }
)

SNAPSHOT_REQUIRED: tuple[str, ...] = (
    "player_id",
    "season",
    "week",
    "market",
    "as_of",
    "kickoff_at",
)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return isinstance(value, str) and (
        not value.strip() or value.strip().lower() in {"nan", "nat", "none", "null"}
    )


def _parse_datetime(value: Any, *, field: str) -> datetime | None:
    if _is_missing(value):
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _as_float(value: Any) -> float | None:
    if _is_missing(value):
        return None
    return float(value)


def _as_int(value: Any, *, field: str) -> int:
    if _is_missing(value):
        raise ValueError(f"snapshot is missing {field}")
    return int(value)


def _as_str(value: Any, *, field: str) -> str:
    if _is_missing(value):
        raise ValueError(f"snapshot is missing {field}")
    text = str(value).strip()
    if not text:
        raise ValueError(f"snapshot is missing {field}")
    return text


@dataclass(frozen=True)
class PropSnapshot:
    player_id: str
    season: int
    week: int
    market: str
    as_of: datetime
    kickoff_at: datetime
    line: float | None = None
    implied_p_over: float | None = None
    implied_mean: float | None = None
    player_name: str | None = None
    source: str | None = None

    def implied_location(self) -> float:
        if self.implied_mean is not None:
            return float(self.implied_mean)
        if self.line is not None:
            return float(self.line)
        raise ValueError(
            f"snapshot {self.player_id} {self.season}w{self.week} {self.market} "
            "needs line or implied_mean"
        )


def prop_snapshot_from_row(row: Mapping[str, Any]) -> PropSnapshot:
    """Parse one snapshot row and fail closed on missing as_of / post-kickoff leak."""
    as_of = _parse_datetime(row.get("as_of"), field="as_of")
    if as_of is None:
        raise MissingAsOfError(
            "prop snapshot is missing as_of; later-season market must not leak into earlier rows"
        )
    kickoff_at = _parse_datetime(row.get("kickoff_at"), field="kickoff_at")
    if kickoff_at is None:
        raise MissingKickoffError(
            "prop snapshot is missing kickoff_at; cannot prove as_of is pre-kickoff"
        )
    if as_of > kickoff_at:
        raise PostKickoffSnapshotError(
            f"as_of {as_of.isoformat()} is after kickoff_at {kickoff_at.isoformat()}"
        )
    market = _as_str(row.get("market"), field="market")
    if market not in ALLOWED_MARKETS:
        raise ValueError(f"unsupported market {market!r}; allowed={sorted(ALLOWED_MARKETS)}")
    line = _as_float(row.get("line"))
    implied_mean = _as_float(row.get("implied_mean"))
    if line is None and implied_mean is None:
        raise ValueError("prop snapshot needs line or implied_mean")
    return PropSnapshot(
        player_id=_as_str(row.get("player_id"), field="player_id"),
        season=_as_int(row.get("season"), field="season"),
        week=_as_int(row.get("week"), field="week"),
        market=market,
        as_of=as_of,
        kickoff_at=kickoff_at,
        line=line,
        implied_p_over=_as_float(row.get("implied_p_over")),
        implied_mean=implied_mean,
        player_name=None if _is_missing(row.get("player_name")) else str(row.get("player_name")),
        source=None if _is_missing(row.get("source")) else str(row.get("source")),
    )


def _read_frame(source: Path | str | pd.DataFrame) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    return pd.read_csv(source)


def load_prop_snapshots(source: Path | str | pd.DataFrame) -> list[PropSnapshot]:
    frame = _read_frame(source)
    if frame.empty:
        return []
    missing_cols = [c for c in SNAPSHOT_REQUIRED if c not in frame.columns]
    if "as_of" in missing_cols:
        raise MissingAsOfError(
            "prop snapshot is missing as_of; later-season market must not leak into earlier rows"
        )
    if "kickoff_at" in missing_cols:
        raise MissingKickoffError(
            "prop snapshot is missing kickoff_at; cannot prove as_of is pre-kickoff"
        )
    return [prop_snapshot_from_row(row) for row in frame.to_dict(orient="records")]


def load_shadow_board(source: Path | str | pd.DataFrame) -> pd.DataFrame:
    frame = _read_frame(source)
    required = {"player_id", "season", "week", "market", "model_mean"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"shadow board is missing columns: {sorted(missing)}")
    assert_prediction_frame_has_no_outcomes(tuple(frame.columns))
    out = frame.copy()
    out["player_id"] = out["player_id"].astype(str)
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    out["market"] = out["market"].astype(str)
    out["model_mean"] = pd.to_numeric(out["model_mean"], errors="coerce")
    return out


def load_outcomes(source: Path | str | pd.DataFrame) -> pd.DataFrame:
    frame = _read_frame(source)
    required = {"player_id", "season", "week", "market", "actual"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"outcomes are missing columns: {sorted(missing)}")
    out = frame.copy()
    out["player_id"] = out["player_id"].astype(str)
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    out["market"] = out["market"].astype(str)
    out["actual"] = pd.to_numeric(out["actual"], errors="coerce")
    return out
