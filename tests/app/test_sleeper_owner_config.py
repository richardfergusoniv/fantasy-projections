"""Tests for owner-confirmed Sleeper league configuration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.app.league.sleeper.owner_config import (
    LeagueSelectionError,
    SleeperOwnerConfig,
    load_owner_config,
    validate_league_selection,
)

EXAMPLE_PATH = Path(__file__).resolve().parents[2] / "config" / "sleeper_owner.example.json"


def _six_league_config() -> dict:
    return json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))


def test_example_config_has_six_leagues_and_validates():
    config = SleeperOwnerConfig.model_validate(_six_league_config())
    assert len(config.leagues) == 6
    assert config.allowed_league_ids == {
        entry.league_id for entry in config.leagues
    }


def test_duplicate_league_ids_fail_fast():
    payload = _six_league_config()
    payload["leagues"].append(dict(payload["leagues"][0]))
    with pytest.raises(ValidationError, match="duplicate league_id"):
        SleeperOwnerConfig.model_validate(payload)


def test_dynasty_without_rule_fails():
    payload = _six_league_config()
    payload["leagues"][2]["rookie_pick_rule"] = None
    with pytest.raises(ValidationError, match="require rookie_pick_rule"):
        SleeperOwnerConfig.model_validate(payload)


def test_redraft_with_rule_fails():
    payload = _six_league_config()
    payload["leagues"][0]["rookie_pick_rule"] = "max_pf"
    with pytest.raises(ValidationError, match="not allowed on redraft"):
        SleeperOwnerConfig.model_validate(payload)


def test_unknown_rule_fails():
    payload = _six_league_config()
    payload["leagues"][2]["rookie_pick_rule"] = "playoff_seed"
    with pytest.raises(ValidationError, match="unknown rookie_pick_rule"):
        SleeperOwnerConfig.model_validate(payload)


def test_extra_sleeper_leagues_are_reported_not_imported():
    config = SleeperOwnerConfig.model_validate(_six_league_config())
    discovered = [
        {"league_id": entry.league_id, "settings": {"type": 2 if entry.league_type == "dynasty" else 0}}
        for entry in config.leagues
    ]
    discovered.append({"league_id": "9999999999999999999", "settings": {"type": 0}})
    summary = validate_league_selection(config, discovered)
    assert summary["extra_leagues_ignored"] == 1
    assert summary["configured_count"] == 6


def test_missing_configured_league_fails():
    config = SleeperOwnerConfig.model_validate(_six_league_config())
    discovered = [{"league_id": config.leagues[0].league_id, "settings": {"type": 0}}]
    with pytest.raises(LeagueSelectionError, match="not returned by Sleeper"):
        validate_league_selection(config, discovered)


def test_league_type_conflict_fails():
    config = SleeperOwnerConfig.model_validate(_six_league_config())
    discovered = [
        {"league_id": entry.league_id, "settings": {"type": 0}}
        for entry in config.leagues
    ]
    with pytest.raises(LeagueSelectionError, match="type conflicts"):
        validate_league_selection(config, discovered)


def test_load_owner_config_from_example_file():
    config = load_owner_config(EXAMPLE_PATH)
    assert config.username == "your_sleeper_username"
    assert len(config.leagues) == 6
