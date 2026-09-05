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


def test_extract_quote_drops_juiced_td_line_vs_projection():
    # Josh Downs Caesars 9.5 rec TDs vs RotoWire proj 4.0.
    assert (
        _extract_quote(
            {
                "line": 9.5,
                "rotowire_proj": 4.0,
                "books": {"caesars": {"line": 9.5}},
            }
        )
        is None
    )


def test_extract_quote_drops_juiced_yard_line_vs_projection():
    # Josh Downs 999.5 (+220) vs RotoWire proj 770.
    assert (
        _extract_quote(
            {
                "line": 999.5,
                "over_odds": 220,
                "rotowire_proj": 770.0,
                "books": {
                    "draftkings": {"line": 999.5, "over_odds": 220},
                    "caesars": {"line": 999.5},
                },
            }
        )
        is None
    )


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
    assert gainwell["markets"]["rush_yards"] < 400
    assert bucky["markets"]["rush_yards"] > 750
    assert gainwell["markets"]["rush_yards"] < bucky["markets"]["rush_yards"]


def test_build_consensus_prefers_books_over_numberfire_when_both_exist():
    payload = build_consensus(season=2026)
    bucky = next(player for player in payload["players"] if player["name"] == "Bucky Irving")
    assert 780 <= bucky["markets"]["rush_yards"] <= 830


def test_build_consensus_downs_not_inflated_by_caesars_tds():
    payload = build_consensus(season=2026)
    downs = next(player for player in payload["players"] if player["name"] == "Josh Downs")
    assert downs["markets"]["rec_tds"] < 5.0
    assert downs["markets"]["rec_yards"] < 900


def test_extract_quote_prefers_two_sided_books_over_odds_less_alts():
    # Fernando Mendoza RotoWire: FanDuel 1950.5 (-114/-114) vs DK/Caesars 3499.5.
    value, kind = _extract_quote(
        {
            "line": 1950.5,
            "over_odds": -114,
            "rotowire_proj": 2894.0,
            "books": {
                "fanduel": {
                    "line": 1950.5,
                    "over_odds": -114,
                    "under_odds": -114,
                },
                "draftkings": {"line": 3499.5},
                "caesars": {"line": 3499.5},
            },
        }
    )
    assert kind == "book"
    assert value == 1950.5


def test_extract_quote_rejects_longshot_alt_and_keeps_real_line():
    # Khalil Shakir: DK 999.5 at +400 vs Caesars 700.5.
    value, kind = _extract_quote(
        {
            "line": 999.5,
            "over_odds": 400,
            "rotowire_proj": 747.0,
            "books": {
                "draftkings": {"line": 999.5, "over_odds": 400},
                "caesars": {"line": 700.5},
            },
        }
    )
    assert kind == "book"
    assert value == 700.5
