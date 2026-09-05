"""Reject polluted Vegas quotes and merge nickname splits."""

from __future__ import annotations

from src.draft_assistant.market_adp import canonicalize_player_name
from src.draft_assistant.vegas_consensus import (
    _extract_quote,
    _line_value,
    _robust_median,
    build_consensus,
)


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


def test_extract_quote_rejects_kalshi_only_skewed_threshold():
    # Kenny Gainwell BettingPros shape: Kalshi 749.5 at +355/-567.
    assert (
        _extract_quote(
            {
                "line": 749.5,
                "over_odds": 355,
                "under_odds": -567,
                "books": {
                    "Kalshi": {
                        "line": 749.5,
                        "over_odds": 355,
                        "under_odds": -567,
                    }
                },
            }
        )
        is None
    )


def test_extract_quote_prefers_traditional_books_over_kalshi():
    value, kind = _extract_quote(
        {
            "line": 900.5,
            "over_odds": -110,
            "under_odds": -110,
            "books": {
                "DraftKings": {
                    "line": 799.5,
                    "over_odds": -110,
                    "under_odds": -110,
                },
                "Kalshi": {
                    "line": 999.5,
                    "over_odds": 250,
                    "under_odds": -400,
                },
            },
        }
    )
    assert kind == "book"
    assert value == 799.5


def test_canonicalize_merges_common_nicknames():
    assert canonicalize_player_name("Kenny Gainwell") == "kenneth gainwell"
    assert canonicalize_player_name("Chig Okonkwo") == "chigoziem okonkwo"
    assert canonicalize_player_name("Cam Skattebo") == "cameron skattebo"
    assert canonicalize_player_name("Cam Ward") == "cameron ward"


def test_build_consensus_kupp_receiving_yards_not_inflated_by_caesars():
    payload = build_consensus(season=2026)
    kupp = next(player for player in payload["players"] if player["name"] == "Cooper Kupp")
    yards = kupp["markets"]["rec_yards"]
    # NumberFire ~798; the 1499.5 Caesars line must not pull this toward 1100+.
    assert 700 <= yards <= 900


def test_build_consensus_gainwell_not_above_bucky_from_kalshi_junk():
    payload = build_consensus(season=2026)
    by_norm = {player["name_norm"]: player for player in payload["players"]}
    assert "kenny gainwell" not in by_norm
    gainwell = by_norm["kenneth gainwell"]
    bucky = by_norm["bucky irving"]
    # Kalshi 749.5 / 5.5 must not survive; NF ~317 rush yards remains.
    assert gainwell["markets"]["rush_yards"] < 400
    assert bucky["markets"]["rush_yards"] > 750
    assert gainwell["markets"]["rush_yards"] < bucky["markets"]["rush_yards"]


def test_build_consensus_prefers_books_over_numberfire_when_both_exist():
    payload = build_consensus(season=2026)
    bucky = next(player for player in payload["players"] if player["name"] == "Bucky Irving")
    # NF ~990 rush yards must not blend into the sportsbook ~800 median.
    assert 780 <= bucky["markets"]["rush_yards"] <= 830
