"""Durable tracker for live Role 2 shadow weeks toward the 6–8 promotion bar.

Fixture / M3 dry-run compares demonstrate the harness. They do not credit a
week. Still shadow; never a promotion.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

TRACKER_REL = "output/shadow_weekly_schedule_m3/live_shadow_weeks.json"
SCHEMA_VERSION = "live_shadow_week_tracker_v1"
TARGET_MIN = 6
TARGET_MAX = 8

LIVE_CREDIT_SOURCES = frozenset({"live_timestamped"})

CREDIT_RULE = (
    "A week is credited only when snapshot_source is live_timestamped "
    "(allow-listed; supplied / fixture / synthetic / dry-run never credit), "
    "timestamped Vegas snapshot rows for that week have as_of <= kickoff_at, "
    "and n_matched > 0. as_of is derived from the snapshot rows, not a caller "
    "bool. Fixture and synthetic weeks demonstrate the harness; they do not "
    "count toward 6-8. Outcomes are required for vs-actual metrics but not "
    "for the first credit of a week (model-vs-market). Re-credit is "
    "idempotent per (season, week). Gate remains not_promoting until the "
    "decision-note bar clears."
)


def default_tracker() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "credited_weeks": 0,
        "target_min": TARGET_MIN,
        "target_max": TARGET_MAX,
        "gate_verdict": "not_promoting",
        "credit_rule": CREDIT_RULE,
        "weeks": [],
    }


def load_tracker(path: str | Path) -> dict[str, Any]:
    loc = Path(path)
    if not loc.is_file():
        return default_tracker()
    payload = json.loads(loc.read_text(encoding="utf-8"))
    tracker = default_tracker()
    tracker.update(payload)
    tracker["weeks"] = list(payload.get("weeks") or [])
    tracker["credited_weeks"] = len(tracker["weeks"])
    tracker["gate_verdict"] = "not_promoting"
    return tracker


def write_tracker(path: str | Path, tracker: dict[str, Any]) -> Path:
    loc = Path(path)
    loc.parent.mkdir(parents=True, exist_ok=True)
    loc.write_text(json.dumps(tracker, indent=2) + "\n", encoding="utf-8")
    return loc


def _parse_stamp(value: object) -> datetime | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none", "null"}:
        return None
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _snapshot_records(
    snapshots: pd.DataFrame | Sequence[Mapping[str, Any]] | None,
) -> list[Mapping[str, Any]]:
    if snapshots is None:
        return []
    if isinstance(snapshots, pd.DataFrame):
        if snapshots.empty:
            return []
        return list(snapshots.to_dict(orient="records"))
    return list(snapshots)


def _as_of_ok_from_snapshots(
    snapshots: pd.DataFrame | Sequence[Mapping[str, Any]] | None,
    *,
    season: int,
    week: int,
) -> bool:
    rows = _snapshot_records(snapshots)
    matched: list[Mapping[str, Any]] = []
    for rec in rows:
        try:
            rec_season = int(rec.get("season"))
            rec_week = int(rec.get("week"))
        except (TypeError, ValueError):
            continue
        if rec_season == int(season) and rec_week == int(week):
            matched.append(rec)
    if not matched:
        return False
    for rec in matched:
        as_of = _parse_stamp(rec.get("as_of"))
        kickoff = _parse_stamp(rec.get("kickoff_at"))
        if as_of is None or kickoff is None or as_of > kickoff:
            return False
    return True


def credit_live_week(
    tracker: dict[str, Any],
    *,
    season: int,
    week: int,
    n_matched: int,
    source: str,
    as_of_ok: bool | None = None,
    snapshots: pd.DataFrame | Sequence[Mapping[str, Any]] | None = None,
) -> tuple[bool, str]:
    """Mutate ``tracker`` when a live week clears the credit rule.

    Returns ``(accepted, reason)``. Only ``live_timestamped`` credits.
    ``as_of_ok`` is derived from snapshot rows; a caller bool is not enough.
    """
    source_key = str(source or "").strip().lower()
    if source_key not in LIVE_CREDIT_SOURCES:
        return False, "only live_timestamped source credits the 6-8 live bar"
    if not _as_of_ok_from_snapshots(snapshots, season=season, week=week):
        return False, "as_of <= kickoff_at required from snapshot rows"
    if as_of_ok is False:
        return False, "as_of <= kickoff_at required"
    if int(n_matched) <= 0:
        return False, "n_matched must be > 0"
    for existing in tracker["weeks"]:
        if int(existing.get("season")) == int(season) and int(existing.get("week")) == int(week):
            return True, "already credited"
    tracker["weeks"].append(
        {
            "season": int(season),
            "week": int(week),
            "n_matched": int(n_matched),
            "source": str(source),
        }
    )
    tracker["credited_weeks"] = len(tracker["weeks"])
    tracker["gate_verdict"] = "not_promoting"
    return True, "credited"
