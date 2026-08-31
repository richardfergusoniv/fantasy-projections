"""Parity checks for ported team-first weekly v2 modules.

Frozen expectations in ``tests/fixtures/weekly_v2_parity.json`` were captured from
the ported accounting/availability/leakage helpers at port time (logic matches
fantasy-projections-2 source files listed in docs/WEEKLY_V2_PORT_PROVENANCE.md).
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from src.projection.weekly.features.leakage import filter_as_of
from src.projection.weekly.pipeline.accounting import apply_accounting
from src.projection.weekly.pipeline.availability import estimate_projected_games

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "weekly_v2_parity.json"


@pytest.fixture(scope="module")
def fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_accounting_parity(fixture: dict) -> None:
    players = pl.DataFrame(fixture["accounting_input_players"])
    team_totals = pl.DataFrame(fixture["accounting_input_team_totals"])
    expected = pl.DataFrame(fixture["accounting_output"]).sort("gsis_id")

    out = (
        apply_accounting(players, team_totals)
        .select(
            [
                "gsis_id",
                "attempts",
                "carries",
                "targets",
                "receptions",
                "passing_yards",
                "receiving_yards",
            ]
        )
        .sort("gsis_id")
    )

    for col in expected.columns:
        assert out[col].to_list() == pytest.approx(expected[col].to_list(), rel=1e-6, abs=1e-6)


def test_availability_parity(fixture: dict) -> None:
    panel = pl.DataFrame(fixture["availability_panel"])
    players = pl.DataFrame({"gsis_id": ["p1", "rookie"]})
    expected = {r["gsis_id"]: r["projected_games_estimate"] for r in fixture["availability_output"]}

    out = estimate_projected_games(panel, players, target_season=2026)
    by_id = {r["gsis_id"]: r["projected_games_estimate"] for r in out.iter_rows(named=True)}

    assert by_id["p1"] == pytest.approx(expected["p1"], abs=0.01)
    assert by_id["rookie"] == pytest.approx(expected["rookie"], abs=0.01)


def test_leakage_filter_excludes_target_week(fixture: dict) -> None:
    hist = pl.DataFrame(fixture["leakage_input"])
    filtered = filter_as_of(hist, season=2024, week=4)
    assert sorted(int(w) for w in filtered["week"].to_list()) == fixture["leakage_filtered_weeks"]
    assert 4 not in filtered["week"].to_list()


def test_weekly_package_exports() -> None:
    from src.projection.weekly import (
        apply_accounting,
        project_veterans_week,
        project_week_with_rookies,
    )

    assert callable(apply_accounting)
    assert callable(project_veterans_week)
    assert callable(project_week_with_rookies)
