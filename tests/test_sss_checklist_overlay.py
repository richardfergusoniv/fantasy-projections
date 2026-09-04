"""Stdlib checks for the Sunday Sports Society checklist overlay."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHECKLIST = REPO / "draft_assistant" / "data" / "draft_checklist_2026.json"
OVERRIDE = REPO / "scripts" / "apply_sss_checklist_override.py"


def test_committed_checklist_uses_sss_checks_for_chase() -> None:
    payload = json.loads(CHECKLIST.read_text(encoding="utf-8"))
    assert payload["meta"]["ol_included"] is True
    assert payload["meta"]["ol_source"] == "ol_unit_rating_chart_32_teams"
    assert "ol_top16" in payload["criteria_by_position"]["QB"]
    assert "ol_top16" in payload["criteria_by_position"]["RB"]
    assert "ol_top16" not in payload["criteria_by_position"]["WR"]

    den = next(t for t in payload["teams"] if t["abbr"] == "DEN")
    cin = next(t for t in payload["teams"] if t["abbr"] == "CIN")
    assert den["ol_unit_rank"] == 1
    assert cin["ol_unit_rank"] == 30

    chase = next(p for p in payload["players"] if p["player_id"] == "00-0036900")
    assert chase["name"] == "Ja'Marr Chase"
    assert chase["pos_market_rank"] == 1
    assert chase["checks"] == {
        "target_leader_in_group": True,
        "qb_top16": True,
        "offense_top16": True,
        "sos_top16": True,
    }
    # CIN OL is #30 on the unit chart → not top-16; Burrow/Chase teammates on QB/RB boards
    # should reflect that. Chase is WR so no ol_top16 key.
    allen = next(p for p in payload["players"] if p["name"] == "Josh Allen")
    assert allen["checks"]["ol_top16"] is True  # BUF #4
    browning = next(
        (p for p in payload["players"] if p["name"] == "Joe Burrow"),
        None,
    )
    if browning is not None:
        assert browning["checks"]["ol_top16"] is False  # CIN #30

    board_counts = {"QB": 0, "RB": 0, "WR": 0, "TE": 0}
    for player in payload["players"]:
        if player.get("sss_board"):
            board_counts[str(player["position"])] += 1
    assert board_counts == {"QB": 24, "RB": 48, "WR": 60, "TE": 24}


def test_override_script_dry_run_matches_full_board() -> None:
    result = subprocess.run(
        [sys.executable, str(OVERRIDE), "--dry-run"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "matched 156 / 156" in result.stdout
    assert "Ja'Marr Chase checks:" in result.stdout
    assert "'qb_top16': True" in result.stdout
    assert "'offense_top16': True" in result.stdout
