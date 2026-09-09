"""The player card must be able to explain the Vegas FP printed above it."""

from __future__ import annotations

from src.app.decisions.draft_checklist import load_checklist_payload
from src.draft_assistant.checklist_prepare import (
    CARD_MARKETS,
    _card_market_kinds,
    _card_markets,
    vegas_fantasy_points,
)


def test_card_markets_drop_absent_and_non_scoring_entries():
    row = {
        "markets": {
            "rec_yards": 787.5,
            "rec_tds": 0.0,          # consensus stores 0.0 for "no line"
            "receptions": None,
            "targets": 108.0,        # not a scoring market; not on the card
            "fantasy_points": 153.5,  # the source's own total, not a component
        },
        "market_kinds": {"rec_yards": "book", "targets": "projection"},
    }
    assert _card_markets(row) == {"rec_yards": 787.5}
    assert _card_market_kinds(row) == {"rec_yards": "book"}


def test_card_markets_are_the_ones_that_produce_vegas_fp():
    row = {
        "markets": {"rec_yards": 787.5, "rec_tds": 2.89, "receptions": 36.21},
        "market_kinds": {
            "rec_yards": "book",
            "rec_tds": "model",
            "receptions": "model",
        },
    }
    card = _card_markets(row)
    assert vegas_fantasy_points(card) == vegas_fantasy_points(row["markets"])
    # Model fallbacks keep their provenance so the card explains the total.
    assert _card_market_kinds(row)["rec_tds"] == "model"


def test_published_checklist_cards_reconcile_with_their_vegas_fp():
    """Every published Vegas FP is the sum of the rows the card renders.

    Previously the entry shipped the raw consensus markets while ``vegas_fp``
    was computed from a model-augmented map, so Jayden Higgins' card showed one
    787.5 rec-yard line (78.75 points) under a "Vegas FP 114.2" header, and 161
    of 163 priced cards rendered target / attempt rows that feed no total.
    """
    payload = load_checklist_payload(2026)
    assert payload is not None, "draft_checklist_2026.json must be published"

    checked = 0
    for entry in payload["players"]:
        vegas_fp = entry.get("vegas_fp")
        if vegas_fp is None:
            continue
        rebuilt = vegas_fantasy_points(entry.get("markets") or {})
        assert rebuilt is not None, entry["name"]
        assert abs(rebuilt - vegas_fp) <= 0.02, (
            f"{entry['name']}: card rows sum to {rebuilt:.2f}, header says {vegas_fp}"
        )
        checked += 1
    assert checked > 100, f"expected a populated board, only {checked} priced entries"


def test_published_cards_carry_provenance_for_every_rendered_market():
    payload = load_checklist_payload(2026)
    assert payload is not None
    seen_kinds: set[str] = set()
    for entry in payload["players"]:
        markets = entry.get("markets") or {}
        kinds = entry.get("market_kinds") or {}
        assert set(markets) <= set(CARD_MARKETS), entry["name"]
        assert set(kinds) <= set(markets), entry["name"]
        assert not any(value == 0 for value in markets.values()), entry["name"]
        seen_kinds.update(kinds.values())
    assert seen_kinds <= {"book", "projection", "model"}
    assert {"book", "model"} <= seen_kinds
