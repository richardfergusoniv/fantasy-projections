"""Scoring compiler regression tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.app.scoring.compiler import compile_sleeper_scoring, score_stat_draw

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "scoring"


@pytest.mark.parametrize(
    "fixture_name",
    [
        "standard.json",
        "superflex.json",
        "ppfd.json",
        "yardage_bonus.json",
        "k_dst.json",
        "dynasty.json",
    ],
)
def test_league_contracts_compile_without_unsupported_keys(fixture_name: str):
    payload = json.loads((FIXTURE_DIR / fixture_name).read_text(encoding="utf-8"))
    contract = compile_sleeper_scoring(payload["scoring_settings"], payload["roster_positions"])
    assert contract.unsupported_keys == []
    assert contract.contract_hash


def test_yardage_bonus_applies_on_draw_not_mean():
    payload = json.loads((FIXTURE_DIR / "yardage_bonus.json").read_text(encoding="utf-8"))
    contract = compile_sleeper_scoring(payload["scoring_settings"], payload["roster_positions"])
    below = score_stat_draw({"rush_yards": 99}, contract)
    at = score_stat_draw({"rush_yards": 100}, contract)
    assert at - below == pytest.approx(3.1)


def test_ppfd_first_down_scoring():
    payload = json.loads((FIXTURE_DIR / "ppfd.json").read_text(encoding="utf-8"))
    contract = compile_sleeper_scoring(payload["scoring_settings"], payload["roster_positions"])
    points = score_stat_draw(
        {"rec_first_downs": 4, "rec_yards": 40, "receptions": 5},
        contract,
    )
    assert points == pytest.approx(2.0 + 4.0 + 2.5)


def test_live_sleeper_kicker_distance_buckets_compile():
    contract = compile_sleeper_scoring(
        {"fgm_50_59": 5.0, "fgm_60p": 6.0, "rec": 1.0},
        ["K", "BN"],
    )
    assert contract.unsupported_keys == []
    points = score_stat_draw({"fgm_50_59": 1, "fgm_60p": 1}, contract)
    assert points == pytest.approx(11.0)


def test_live_sleeper_idp_dst_bonus_keys_compile():
    contract = compile_sleeper_scoring(
        {"def_pass_def": 1.0, "tkl_loss": 2.0},
        ["DEF", "BN"],
    )
    assert contract.unsupported_keys == []
