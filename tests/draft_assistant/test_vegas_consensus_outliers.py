"""Reject polluted Vegas quotes and merge nickname splits."""

from __future__ import annotations

from src.draft_assistant.market_adp import canonicalize_player_name
from src.draft_assistant.vegas_consensus import (
    _conflicts_with_projection,
    _extract_quote,
    _line_value,
    _one_sided_longshot,
    _prop_coverage,
    _robust_median,
    build_consensus,
)


def test_prop_coverage_classifies_book_vs_projection_scoring_markets():
    assert _prop_coverage({"rec_yards": "book", "rec_tds": "book"}) == "books"
    assert _prop_coverage({"rec_yards": "projection", "receptions": "projection"}) == (
        "projection"
    )
    assert _prop_coverage({"rec_yards": "book", "rec_tds": "projection"}) == "mixed"
    assert _prop_coverage({"targets": "projection"}) == "none"
    assert _prop_coverage({}) == "none"


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
    # Josh Downs 999.5 (+220) vs RotoWire proj 770 — reject the juice and keep
    # the same-source projection (milder than Kupp's full-source drop).
    value, kind = _extract_quote(
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
    assert kind == "projection"
    assert value == 770.0


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
    # After juice drop, remaining scoring markets are numberFire-only — not Vegas.
    assert kupp["prop_coverage"] == "projection"
    assert set(kupp["market_kinds"].values()) == {"projection"}


def test_build_consensus_star_has_book_backed_scoring_markets():
    payload = build_consensus(season=2026)
    chase = next(player for player in payload["players"] if player["name"] == "Ja'Marr Chase")
    assert chase["prop_coverage"] in ("books", "mixed")
    assert any(kind == "book" for kind in chase["market_kinds"].values())


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


def test_extract_quote_drops_pierce_style_juiced_yards_and_tds():
    # Alec Pierce RotoWire: DK/Caesars 999.5 (+125) + odds-less Caesars overlay
    # vs FanDuel 925.5 and RotoWire proj 872.
    value, kind = _extract_quote(
        {
            "line": 999.5,
            "over_odds": 125,
            "rotowire_proj": 872.0,
            "books": {
                "draftkings": {"line": 999.5, "over_odds": 125},
                "fanduel": {"line": 925.5},
                "caesars": {"line": 999.5},
            },
        }
    )
    assert kind == "book"
    assert value == 925.5

    # Caesars-only 7.5 rec TDs vs RotoWire proj 6.0 — trust the projection.
    value, kind = _extract_quote(
        {
            "line": 7.5,
            "rotowire_proj": 6.0,
            "books": {"caesars": {"line": 7.5}},
        }
    )
    assert kind == "projection"
    assert value == 6.0


def test_build_consensus_pierce_not_inflated_by_caesars_juice():
    payload = build_consensus(season=2026)
    pierce = next(player for player in payload["players"] if player["name"] == "Alec Pierce")
    assert pierce["markets"]["rec_yards"] < 950
    assert pierce["markets"]["rec_tds"] <= 6.0
    assert pierce["market_kinds"]["rec_yards"] == "book"
    assert pierce["market_kinds"]["rec_tds"] == "projection"


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


def test_one_sided_even_money_is_not_a_longshot():
    # theScore Bet posts main season numbers with an over price only. +100 is
    # even money, so rejecting it as a juice ladder threw away a player's only
    # sportsbook quote and left a model projection ranking as "Vegas".
    assert _one_sided_longshot(100, None) is False
    assert _one_sided_longshot(None, 100) is False
    assert _one_sided_longshot(125, None) is True
    assert _one_sided_longshot(None, 400) is True


def test_two_sided_books_outrank_a_conflicting_source_projection():
    # Aaron Rodgers passing yards: three books priced on both sides, all above
    # one 2700 projection. Three books agreeing is the market.
    value, kind = _extract_quote(
        {
            "line": 3099.5,
            "rotowire_proj": 2700.0,
            "books": {
                "draftkings": {"line": 3099.5, "over_odds": -110, "under_odds": -110},
                "fanduel": {"line": 3050.5, "over_odds": -114, "under_odds": -114},
                "circasports": {"line": 3075.5, "over_odds": -115, "under_odds": -115},
            },
        },
        market="passing_yards",
    )
    assert kind == "book"
    assert value == 3075.5


def test_two_sided_count_consensus_is_not_replaced_by_the_projection():
    # Javonte Williams rushing TDs: DK/FanDuel/Caesars all priced at 9.5 vs a
    # 7.0 projection. The median-stage override must not demote this to model.
    value, kind = _extract_quote(
        {
            "line": 9.5,
            "rotowire_proj": 7.0,
            "books": {
                "draftkings": {"line": 9.5, "over_odds": -105, "under_odds": -120},
                "fanduel": {"line": 9.5, "over_odds": -128, "under_odds": -104},
                "caesars": {"line": 9.5, "over_odds": -110, "under_odds": -110},
            },
        },
        market="rushing_tds",
    )
    assert kind == "book"
    assert value == 9.5


def test_conflict_gate_never_fires_on_a_book_below_the_projection():
    # A book quoting under a model is market disagreement, not a juice ladder;
    # dropping it would bias the consensus upward.
    assert _conflicts_with_projection(400.0, 900.0) is False
    assert _conflicts_with_projection(2.0, 9.0) is False
    assert _conflicts_with_projection(1499.5, 364.0) is True


def test_count_gate_is_chosen_by_market_not_magnitude():
    # An elite quarterback projected for 27 passing TDs is still a count
    # market. Sizing the branch by magnitude sent exactly those players to the
    # loose yardage gate, where a 34.5 alt rung sails through ``delta > 80``.
    assert _conflicts_with_projection(34.5, 27.0, market="passing_tds") is True
    # Without a market name the magnitude fallback still applies.
    assert _conflicts_with_projection(34.5, 27.0) is False


def test_build_consensus_recovers_book_quotes_priced_at_even_money():
    payload = build_consensus(season=2026)
    tyson = next(p for p in payload["players"] if p["name"] == "Jordyn Tyson")
    # theScore Bet 749.5 at +100 is a real book line, not a longshot rung.
    assert tyson["market_kinds"]["rec_yards"] == "book"
    assert tyson["markets"]["rec_yards"] == 749.5
    assert tyson["prop_coverage"] in ("books", "mixed")
