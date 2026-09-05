"""Vegas-FP-backed VORP on the draft board."""

from __future__ import annotations

from src.app.decisions.draft_board import DraftBoardService, _load_vegas_fp_by_player


def test_load_vegas_fp_map_covers_checklist_players():
    mapping = _load_vegas_fp_by_player(2026)
    assert len(mapping) >= 700
    assert all(isinstance(value, float) for value in mapping.values())


def test_sealed_draft_board_ranks_by_vegas_vorp():
    board = DraftBoardService().load_board(2026, limit=25)
    if not board["entries"]:
        return
    assert board.get("points_source") == "vegas_fp"
    assert board.get("ranking_basis") == "vegas_vorp"
    assert board["entries"][0]["rank"] == 1
    assert board["entries"][0]["vorp"] is not None
    # Ranks are dense and sorted by VORP descending.
    vorps = [entry["vorp"] for entry in board["entries"]]
    assert vorps == sorted(vorps, reverse=True)
    assert [entry["rank"] for entry in board["entries"]] == list(
        range(1, len(board["entries"]) + 1)
    )


def test_sealed_draft_board_assigns_usable_vegas_vorp_tiers():
    board = DraftBoardService().load_board(2026, limit=120)
    if not board["entries"]:
        return
    tiers = [entry["tier"] for entry in board["entries"] if entry.get("tier") is not None]
    assert tiers
    assert tiers == sorted(tiers)
    # Absolute cliffs from the VORP anchor should yield a handful of draftable
    # groups among the top ~100, not one-player micro-tiers.
    positive = [entry for entry in board["entries"] if (entry.get("vorp") or 0) > 0]
    positive_tiers = {entry["tier"] for entry in positive}
    assert 4 <= len(positive_tiers) <= 20
    assert board["entries"][0]["tier"] == 1
