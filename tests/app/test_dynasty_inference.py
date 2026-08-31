"""Dynasty state and rookie-pick order come from league data, not constants.

The `/dynasty/{roster_id}` route used to pass fixed numbers into the inference,
so every roster in every league returned "contender, 56%" and the same projected
pick. These tests assert the properties that catches that: two rosters of
different strength must not agree, the two rookie-pick rules must be able to
disagree, and a feature the data cannot support must be reported rather than
invented.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("APP_ENABLE_DEV_AUTH", "true")
os.environ.setdefault("TEST_DATABASE_URL", "sqlite+pysqlite:///:memory:?cache=shared")

DYNASTY_LEAGUE = "fixture-dynasty"  # reverse_standings
MAX_PF_LEAGUE = "fixture-superflex"  # max_pf
REDRAFT_LEAGUE = "fixture-standard"


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("APP_ENABLE_DEV_AUTH", "true")
    from src.app.config import get_settings
    from src.app.factory import create_app
    from src.app.middleware.rate_limit import limiter
    from src.app.persistence.database import get_session, init_db
    from src.app.seed import seed_development_data

    get_settings.cache_clear()
    limiter.reset()
    init_db()
    with get_session() as session:
        seed_development_data(session, email="owner@example.com")
    test_client = TestClient(create_app())
    link = test_client.post(
        "/api/v1/auth/magic-link", json={"email": "owner@example.com"}
    ).json()["development_link"]
    test_client.post("/api/v1/auth/verify", json={"token": link.split("token=")[-1]})
    try:
        yield test_client
    finally:
        limiter.reset()


def _state(client: TestClient, league_id: str, roster_id: int) -> dict:
    response = client.get(f"/api/v1/leagues/{league_id}/dynasty/{roster_id}")
    assert response.status_code == 200, response.text
    return response.json()


def test_two_rosters_do_not_receive_the_same_inference(client: TestClient):
    one = _state(client, DYNASTY_LEAGUE, 1)["manager_state"]
    two = _state(client, DYNASTY_LEAGUE, 2)["manager_state"]

    assert one["features"] != two["features"], "both rosters got identical features"
    assert one["probabilities"] != two["probabilities"]
    # The stronger roster must be the more likely contender.
    stronger, weaker = (
        (one, two) if one["features"]["ros_win_prob"] > two["features"]["ros_win_prob"] else (two, one)
    )
    assert stronger["probabilities"]["contender"] > weaker["probabilities"]["contender"]


def test_features_are_league_relative_and_bounded(client: TestClient):
    for roster_id in (1, 2):
        features = _state(client, DYNASTY_LEAGUE, roster_id)["manager_state"]["features"]
        assert set(features) == {
            "lineup_strength",
            "ros_win_prob",
            "multi_year_value",
            "pick_capital",
        }
        for name, value in features.items():
            assert 0.0 <= value <= 1.0, f"{name}={value}"


def test_a_feature_the_data_cannot_support_is_declared_not_invented(client: TestClient):
    """No birthdates are stored, so the value is not claimed to be age-adjusted."""
    state = _state(client, DYNASTY_LEAGUE, 1)["manager_state"]

    assert "age_adjustment" in state["unavailable_features"]
    assert "age_adjusted_value" not in state["features"]
    assert 0.0 < state["feature_coverage"] <= 1.0


def test_the_two_rookie_pick_rules_are_separate_orderings(client: TestClient):
    reverse = _state(client, DYNASTY_LEAGUE, 1)["rookie_pick_projection"]
    max_pf = _state(client, MAX_PF_LEAGUE, 1)["rookie_pick_projection"]

    assert reverse["rule"] == "reverse_standings"
    assert reverse["basis"] == "projected_final_standings"
    assert max_pf["rule"] == "max_pf"
    assert max_pf["basis"] == "simulated_optimal_and_potential_points"
    # Every roster in a league gets a distinct slot.
    slots = {
        _state(client, MAX_PF_LEAGUE, roster_id)["rookie_pick_projection"]["projected_pick"]
        for roster_id in (1, 2)
    }
    assert len(slots) == 2


def test_evaluating_state_persists_it_for_the_trade_engine(client: TestClient):
    """Trade output carries contender context only if the state was stored."""
    _state(client, DYNASTY_LEAGUE, 1)
    _state(client, DYNASTY_LEAGUE, 2)

    csrf = client.post(
        "/api/v1/auth/magic-link", json={"email": "owner@example.com"}
    ).json()["development_link"]
    token = csrf.split("token=")[-1]
    headers = {
        "X-CSRF-Token": client.post("/api/v1/auth/verify", json={"token": token}).json()[
            "csrf_token"
        ],
        "Idempotency-Key": "dynasty-context-trade",
    }
    response = client.post(
        f"/api/v1/leagues/{DYNASTY_LEAGUE}/trades/evaluate",
        json={
            "side_a": {"roster_id": 1, "player_ids": ["00-0034857"]},
            "side_b": {"roster_id": 2, "player_ids": ["00-0033280"]},
        },
        headers=headers,
    )
    assert response.status_code == 200, response.text
    context = response.json()["objective"]["context"]
    assert context["side_a_state"] is not None
    assert context["side_b_state"] is not None
    assert context["picks_permitted"] is True


def test_pick_capital_follows_traded_pick_ownership(client: TestClient):
    """A roster that acquired a first-rounder must out-rank the one that sent it."""
    from src.app.decisions.dynasty import DynastyService
    from src.app.persistence.database import get_session
    from src.app.persistence.models import TradedPick

    with get_session() as session:
        service = DynastyService(session)
        rosters = [1, 2]
        even = service.pick_capital("pick-capital-league", 2026, rosters)
        assert even == {1: 1.0, 2: 1.0}, even

        session.add(
            TradedPick(
                league_id="pick-capital-league",
                season=2027,
                round=1,
                original_roster_id=2,
                owner_roster_id=1,
            )
        )
        session.flush()
        skewed = service.pick_capital("pick-capital-league", 2026, rosters)

    assert skewed[1] == 1.0
    assert skewed[2] < 1.0


def test_no_usable_feature_produces_an_uncommitted_state(client: TestClient):
    """An unknown league must not yield a confident label from nothing."""
    from src.app.decisions.dynasty import DynastyService
    from src.app.persistence.database import get_session

    with get_session() as session:
        result = DynastyService(session).evaluate_roster("league-that-does-not-exist", 1)

    assert result.feature_coverage == 0.0
    assert result.features == {}
    assert result.probabilities == {
        "contender": 0.25,
        "fringe": 0.25,
        "retooling": 0.25,
        "rebuilding": 0.25,
    }
