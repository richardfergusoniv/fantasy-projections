"""Live shadow-week tracker: start at 0; fixture weeks do not credit the 6–8 bar."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.projection.weekly_eval.live_weeks import (
    TRACKER_REL,
    credit_live_week,
    load_tracker,
)

ROOT = Path(__file__).resolve().parents[2]
COMMITTED = ROOT / TRACKER_REL


def test_committed_tracker_starts_at_zero():
    assert COMMITTED.is_file()
    payload = json.loads(COMMITTED.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "live_shadow_week_tracker_v1"
    assert payload["credited_weeks"] == 0
    assert payload["target_min"] == 6
    assert payload["target_max"] == 8
    assert payload["gate_verdict"] == "not_promoting"
    assert payload["weeks"] == []
    assert "as_of" in payload["credit_rule"].lower()
    assert "kickoff" in payload["credit_rule"].lower()
    assert "n_matched" in payload["credit_rule"].lower()


def _live_snaps(*, as_of: str = "2026-09-09T18:00:00+00:00") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_id": "00-0034857",
                "season": 2026,
                "week": 1,
                "market": "pass_yards",
                "as_of": as_of,
                "kickoff_at": "2026-09-10T20:20:00+00:00",
                "line": 265.5,
            }
        ]
    )


def test_credit_rejects_fixture_and_unmatched_weeks():
    tracker = load_tracker(COMMITTED)
    rejected, reason = credit_live_week(
        tracker,
        season=2026,
        week=1,
        n_matched=4,
        source="m3_synthetic_fixture",
        snapshots=_live_snaps(),
    )
    assert rejected is False
    assert "live" in reason.lower() or "fixture" in reason.lower() or "synthetic" in reason.lower()
    assert tracker["credited_weeks"] == 0

    rejected, reason = credit_live_week(
        tracker,
        season=2026,
        week=1,
        n_matched=0,
        source="live_timestamped",
        snapshots=_live_snaps(),
    )
    assert rejected is False
    assert "n_matched" in reason.lower()
    assert tracker["credited_weeks"] == 0


def test_credit_rejects_supplied_source_even_with_as_of_ok():
    tracker = load_tracker(COMMITTED)
    rejected, reason = credit_live_week(
        tracker,
        season=2026,
        week=1,
        n_matched=12,
        source="supplied",
        as_of_ok=True,
        snapshots=_live_snaps(),
    )
    assert rejected is False
    assert tracker["credited_weeks"] == 0
    assert "live_timestamped" in reason.lower() or "allow" in reason.lower()


def test_credit_rejects_caller_as_of_ok_when_rows_are_post_kickoff():
    tracker = load_tracker(COMMITTED)
    rejected, reason = credit_live_week(
        tracker,
        season=2026,
        week=1,
        n_matched=12,
        source="live_timestamped",
        as_of_ok=True,
        snapshots=_live_snaps(as_of="2026-09-11T18:00:00+00:00"),
    )
    assert rejected is False
    assert "as_of" in reason.lower()
    assert tracker["credited_weeks"] == 0


def test_credit_accepts_live_pre_kickoff_matched_week(tmp_path: Path):
    tracker = load_tracker(COMMITTED)
    accepted, reason = credit_live_week(
        tracker,
        season=2026,
        week=1,
        n_matched=12,
        source="live_timestamped",
        snapshots=_live_snaps(),
    )
    assert accepted is True
    assert tracker["credited_weeks"] == 1
    assert tracker["weeks"][0]["season"] == 2026
    assert tracker["weeks"][0]["week"] == 1
    assert tracker["gate_verdict"] == "not_promoting"
    out = tmp_path / "live_shadow_weeks.json"
    out.write_text(json.dumps(tracker, indent=2) + "\n", encoding="utf-8")
    reloaded = load_tracker(out)
    assert reloaded["credited_weeks"] == 1
    # The committed tracker stays at zero until Richard credits a live week.
    committed = json.loads(COMMITTED.read_text(encoding="utf-8"))
    assert committed["credited_weeks"] == 0
    assert reason is None or "credited" in reason.lower()
