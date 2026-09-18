"""Join diagnostics fail closed on missing keys and count distinct key tuples."""
from __future__ import annotations

import pandas as pd
import pytest

from src.projection.weekly_eval.match import describe_match, empty_match


def _row(player_id: str, market: str = "pass_yards") -> dict[str, object]:
    return {
        "player_id": player_id,
        "season": 2026,
        "week": 1,
        "market": market,
        "model_mean": 1.0,
    }


def test_describe_match_raises_when_join_keys_missing():
    board = pd.DataFrame([{"player_id": "00-1", "season": 2026, "week": 1, "model_mean": 1.0}])
    snaps = pd.DataFrame(
        [_row("00-1") | {"as_of": "2026-09-09T18:00:00+00:00", "kickoff_at": "2026-09-10T20:20:00+00:00"}]
    )
    with pytest.raises(ValueError, match="market"):
        describe_match(board, snaps, board.iloc[0:0])


def test_empty_match_counts_distinct_keys_not_rows():
    board = pd.DataFrame(
        [
            _row("a"),
            _row("a"),
            _row("b", market="rec_yards"),
        ]
    )
    match = empty_match(board=board)
    assert match["n_board_unmatched"] == 2
    assert match["n_snapshot_unmatched"] == 0
    assert set(match["unmatched_board_ids_sample"]) == {"a", "b"}
