"""Tests for weekly stat-draw partition generation."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from src.app.projections.weekly_draws import (
    generate_player_stat_draws,
    verify_weekly_draw_partition,
    write_weekly_draw_partition,
)


def test_weekly_draw_partition_is_deterministic_and_verifiable(tmp_path: Path):
    frame = pl.DataFrame(
        {
            "gsis_id": ["00-test-1", "00-test-2"],
            "position": ["RB", "WR"],
            "fantasy_points": [12.0, 18.0],
            "floor": [8.0, 12.0],
            "ceiling": [16.0, 24.0],
            "carries": [15.0, 0.0],
            "rushing_yards": [70.0, 0.0],
            "rushing_tds": [1.0, 0.0],
            "receptions": [3.0, 6.0],
            "receiving_yards": [20.0, 90.0],
            "receiving_tds": [0.0, 1.0],
        }
    )
    partition = write_weekly_draw_partition(
        frame,
        tmp_path,
        draw_count=50,
        seed_salt="test-seed",
    )
    assert partition.player_count == 2
    assert verify_weekly_draw_partition(partition.path, expected_sha256=partition.sha256)

    payload = json.loads(partition.path.read_text(encoding="utf-8"))
    draws_a = generate_player_stat_draws(
        payload["players"][0], draw_count=50, seed_salt="test-seed"
    )
    draws_b = generate_player_stat_draws(
        payload["players"][0], draw_count=50, seed_salt="test-seed"
    )
    assert draws_a == draws_b
    assert len(draws_a) == 50
    assert all(draw["rush_yards"] >= 0 for draw in draws_a)
