"""Tests for tendencies, assistant gateway, operations, and release gates."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

os.environ["APP_ENV"] = "test"
os.environ["APP_ENABLE_DEV_AUTH"] = "true"
os.environ["TEST_DATABASE_URL"] = "sqlite+pysqlite:///:memory:?cache=shared"


def _login(client: TestClient) -> str:
    response = client.post("/api/v1/auth/magic-link", json={"email": "owner@example.com"})
    token = response.json()["development_link"].split("token=")[-1]
    verify = client.post("/api/v1/auth/verify", json={"token": token})
    return verify.json()["csrf_token"]


def test_dynasty_promotion_pointer(db_session: Session):
    from src.app.projections.weekly_run import WeeklyProjectionService
    from src.app.releases.bridge import ReleaseBridge
    from src.app.persistence.repositories import ProjectionRepository

    if ReleaseBridge(db_session).sync_preseason_pointer(2026) is None:
        pytest.skip("no active release bundle")
    service = WeeklyProjectionService(db_session)
    run_id = service.promote_dynasty(2026)
    assert run_id is not None
    run = ProjectionRepository(db_session).active_run(mode="dynasty", season=2026, week=None)
    assert run is not None
    assert run.id == run_id


def test_manager_tendency_learning(db_session: Session):
    from src.app.decisions.tendencies import ManagerTendencyService
    from src.app.persistence.models import TradeProposal
    from src.app.seed import seed_development_data

    seed = seed_development_data(db_session, email="owner@example.com")
    league_id = seed["leagues"][0]
    db_session.add(
        TradeProposal(
            league_id=league_id,
            created_by_roster_id=1,
            sides_json={"offered": ["00-0034857"], "received": ["00-0033280"]},
            direction="outgoing",
            status="accepted",
        )
    )
    db_session.flush()
    service = ManagerTendencyService(db_session)
    results = service.rebuild(league_id)
    assert results
    features = service.get(league_id, 1)
    assert features.sample_size >= 1


def test_assistant_gateway_routes_lineup(db_session: Session):
    from src.app.assistant.gateway import AssistantGateway
    from src.app.persistence.models import AppUser
    from src.app.seed import seed_development_data

    seed_development_data(db_session, email="owner@example.com")
    user = db_session.query(AppUser).filter(AppUser.email == "owner@example.com").one()
    gateway = AssistantGateway(db_session)
    response = gateway.respond(user, "help with my lineup starters", league_id="fixture-standard", week=1)
    assert response["tools_called"] == ["recommend_lineup"]
    assert "tool_result" in response


def test_release_gate_rejects_small_player_set():
    from src.app.projections.loader import PlayerSummary
    from src.app.releases.gates import validate_matchup_probabilities, validate_promotion

    players = {
        "p1": PlayerSummary(
            player_id="p1",
            name="Test",
            position="QB",
            team="BUF",
            mean_points=10.0,
            availability_probability=0.9,
            quantiles={"p50": 10.0},
        )
    }
    gate = validate_promotion(mode="weekly", players=players, min_players=100)
    assert gate.passed is False
    assert gate.failures

    prob_gate = validate_matchup_probabilities({"win": 0.6, "tie": 0.1, "loss": 0.3})
    assert prob_gate.passed is True
    bad_gate = validate_matchup_probabilities({"win": 0.9, "tie": 0.1, "loss": 0.2})
    assert bad_gate.passed is False


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("APP_ENABLE_DEV_AUTH", "true")
    monkeypatch.setenv("WEEKLY_V2_MODELS_DIR", str(tmp_path / "empty_models"))
    monkeypatch.setenv("WEEKLY_V2_OUTPUTS_DIR", str(tmp_path / "empty_outputs"))
    from src.app.config import get_settings
    from src.app.factory import create_app
    from src.app.persistence.database import get_session, init_db
    from src.app.seed import seed_development_data

    get_settings.cache_clear()
    init_db()
    with get_session() as session:
        seed_development_data(session, email="owner@example.com")
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_operations_status_endpoint(client: TestClient):
    _login(client)
    response = client.get("/api/v1/operations/status")
    assert response.status_code == 200
    body = response.json()
    assert "failed_gates" in body
    assert body.get("active_projection_run_id")


def test_trade_tendencies_endpoint(client: TestClient):
    _login(client)
    response = client.get("/api/v1/leagues/fixture-standard/managers/1/tendencies")
    assert response.status_code == 200
    assert "features" in response.json()


def test_assistant_endpoint(client: TestClient):
    csrf = _login(client)
    response = client.post(
        "/api/v1/assistant/responses",
        json={"message": "who should I start this week?", "league_id": "fixture-standard", "week": 1},
        headers={"X-CSRF-Token": csrf, "Idempotency-Key": "assistant-test-1"},
    )
    assert response.status_code == 200
    assert response.json().get("tools_called") == ["recommend_lineup"]


def test_projection_rollback_endpoint(client: TestClient, db_session: Session):
    from src.app.projections.weekly_run import WeeklyProjectionService
    from src.app.releases.bridge import ReleaseBridge

    if ReleaseBridge(db_session).sync_preseason_pointer(2026) is None:
        pytest.skip("no active release bundle")
    WeeklyProjectionService(db_session).promote_week(2026, week=1)
    csrf = _login(client)
    response = client.post(
        "/api/v1/operations/projections/rollback?mode=weekly&season=2026&week=1",
        headers={"X-CSRF-Token": csrf, "Idempotency-Key": "rollback-test-1"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"rolled_back", "unchanged"}
