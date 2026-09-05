"""Reject book O/U lines that conflict with the same source's projection."""

from __future__ import annotations

from src.draft_assistant.vegas_consensus import _line_value, _robust_median, build_consensus


def test_line_value_drops_book_line_that_conflicts_with_source_projection():
    # Cooper Kupp RotoWire shape: Caesars 1499.5 vs RotoWire proj 364.
    assert (
        _line_value(
            {
                "line": 1499.5,
                "rotowire_proj": 364.0,
                "books": {"caesars": {"line": 1499.5}},
            }
        )
        is None
    )


def test_line_value_keeps_book_line_when_projection_agrees():
    assert _line_value({"line": 999.5, "rotowire_proj": 980.0}) == 999.5


def test_line_value_falls_back_to_projection_when_no_line():
    assert _line_value({"rotowire_proj": 364.0}) == 364.0


def test_robust_median_drops_far_outlier_with_three_quotes():
    assert _robust_median([800.0, 820.0, 1500.0]) == 810.0


def test_build_consensus_kupp_receiving_yards_not_inflated_by_caesars():
    payload = build_consensus(season=2026)
    kupp = next(player for player in payload["players"] if player["name"] == "Cooper Kupp")
    yards = kupp["markets"]["rec_yards"]
    # NumberFire ~798; the 1499.5 Caesars line must not pull this toward 1100+.
    assert 700 <= yards <= 900
