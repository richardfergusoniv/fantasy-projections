"""Durable tracker for live Role 2 shadow weeks toward the 6–8 promotion bar.

Fixture / M3 dry-run compares demonstrate the harness. They do not credit a
week. Still shadow; never a promotion.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TRACKER_REL = "output/shadow_weekly_schedule_m3/live_shadow_weeks.json"
SCHEMA_VERSION = "live_shadow_week_tracker_v1"
TARGET_MIN = 6
TARGET_MAX = 8

CREDIT_RULE = (
    "A week is credited only when a live (not fixture or M3 dry-run) Role 2 "
    "compare runs against timestamped Vegas snapshots with as_of <= kickoff_at "
    "and n_matched > 0. Fixture and synthetic weeks demonstrate the harness; "
    "they do not count toward 6-8. Outcomes are required for vs-actual metrics "
    "but not for the first credit of a week (model-vs-market). Re-credit is "
    "idempotent per (season, week). Gate remains not_promoting until the "
    "decision-note bar clears."
)

_FIXTURE_MARKERS = (
    "fixture",
    "synthetic",
    "dry_run",
    "dry-run",
    "m3_dry_run",
    "weekly_eval_fixture",
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


def _is_fixture_source(source: str) -> bool:
    text = str(source or "").strip().lower()
    return any(marker in text for marker in _FIXTURE_MARKERS)


def credit_live_week(
    tracker: dict[str, Any],
    *,
    season: int,
    week: int,
    n_matched: int,
    source: str,
    as_of_ok: bool,
) -> tuple[bool, str]:
    """Mutate ``tracker`` when a live week clears the credit rule.

    Returns ``(accepted, reason)``. Fixture weeks and n_matched=0 are rejected.
    """
    if _is_fixture_source(source):
        return False, "fixture/synthetic weeks do not credit the 6-8 live bar"
    if not as_of_ok:
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
