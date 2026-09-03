"""Draft board and rate limit tests."""

from __future__ import annotations

import os
import uuid

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("APP_ENABLE_DEV_AUTH", "true")
os.environ.setdefault("TEST_DATABASE_URL", "sqlite+pysqlite:///:memory:?cache=shared")


def test_draft_board_returns_release_players(db_session):
    from src.app.decisions.draft_board import DraftBoardService

    service = DraftBoardService()
    board = service.load_board(2026, limit=20)
    if not board["entries"]:
        pytest.skip("no release bundle available")
    assert len(board["entries"]) <= 20
    assert board["entries"][0]["rank"] == 1
    assert board["entries"][0]["name"]


def test_rate_limit_blocks_excess_auth_requests():
    from src.app.middleware.rate_limit import RateLimiter

    limiter = RateLimiter()
    bucket = f"auth:test-{uuid.uuid4().hex}"
    for _ in range(3):
        limiter.check(bucket, limit=3)
    with pytest.raises(Exception) as exc:
        limiter.check(bucket, limit=3)
    assert exc.value.status_code == 429


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("APP_ENABLE_DEV_AUTH", "true")
    monkeypatch.setenv("APP_ALLOWED_EMAIL", "owner@example.com")
    monkeypatch.setenv("EMAIL_PROVIDER", "development")
    monkeypatch.setenv("AUTH_RATE_LIMIT_PER_MINUTE", "100")
    monkeypatch.setenv("TRUSTED_HOSTS", "testserver,localhost,127.0.0.1")
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


def test_draft_board_endpoint(client: TestClient):
    token = (
        client.post("/api/v1/auth/magic-link", json={"email": "owner@example.com"})
        .json()["development_link"]
        .split("token=")[-1]
    )
    client.post("/api/v1/auth/verify", json={"token": token})
    response = client.get("/api/v1/leagues/fixture-standard/draft/board")
    assert response.status_code == 200
    body = response.json()
    if body.get("entries"):
        assert body["entries"][0]["player_id"]
        assert len(body["entries"]) > 15
        assert body["league_specific"] is True
        assert body["replacement_ranks"]["WR"] > 1
        assert body["entries"][0]["replacement_rank"] > 1


def test_draft_checklist_endpoint(client: TestClient):
    token = (
        client.post("/api/v1/auth/magic-link", json={"email": "owner@example.com"})
        .json()["development_link"]
        .split("token=")[-1]
    )
    client.post("/api/v1/auth/verify", json={"token": token})
    response = client.get("/api/v1/leagues/fixture-standard/draft/checklist")
    assert response.status_code == 200
    body = response.json()
    assert body.get("available") is True
    assert body["entries"]
    assert "vorp" not in body["entries"][0]
    assert body["entries"][0]["rank_tier"] in {"adp", "ecr", "prior_pts", "none"}
    assert body["meta"]["market_as_of"]["scoring"] == "half-ppr"
    assert body["meta"]["market_as_of"]["teams"] == 12
    assert "checks" in body["entries"][0]


def test_draft_board_superflex_zero_vorp_qb_not_first():
    """A QB with 0 VORP must not outrank skill players on season-points fallback."""
    from src.app.decisions.draft_board import DraftBoardService, _draft_sort_value

    players = [
        {
            "player_id": "qb-zero",
            "display_name": "Rookie QB",
            "position": "QB",
            "vorp": 0.0,
            "fantasy_pts_season": 320.0,
        },
        {
            "player_id": "rb-plus",
            "display_name": "Elite RB",
            "position": "RB",
            "vorp": 18.5,
            "fantasy_pts_season": 290.0,
        },
        {
            "player_id": "wr-plus",
            "display_name": "Elite WR",
            "position": "WR",
            "vorp": 14.0,
            "fantasy_pts_season": 275.0,
        },
    ]
    ranked = sorted(players, key=_draft_sort_value, reverse=True)
    assert ranked[0]["player_id"] == "rb-plus"
    assert ranked[-1]["player_id"] == "qb-zero"

    service = DraftBoardService()
    board = service.load_board(2026, limit=50)
    if not board["entries"]:
        pytest.skip("no release bundle available")
    first = board["entries"][0]
    if first.get("position") == "QB" and first.get("vorp") == 0.0:
        pytest.fail("zero-VORP QB ranked first on production board")


def test_draft_board_zero_vorp_sorts_above_season_points_fallback():
    from src.app.decisions.draft_board import DraftBoardService, _draft_sort_value

    assert _draft_sort_value({"vorp": 0.0, "fantasy_pts_season": 320.0}) == 0.0
    assert _draft_sort_value({"vorp": 12.5, "fantasy_pts_season": 400.0}) == 12.5

    service = DraftBoardService()
    board = service.load_board(2026, limit=500)
    if not board["entries"]:
        pytest.skip("no release bundle available")
    zero_vorp = [row for row in board["entries"] if row.get("vorp") == 0.0]
    if not zero_vorp:
        pytest.skip("no zero-vorp players in release board")
    first_zero = zero_vorp[0]
    assert first_zero["name"] != board["entries"][0]["name"] or first_zero["rank"] > 1


def _format_rows() -> list[dict]:
    """Synthetic values make every FLEX and SUPER_FLEX assignment deterministic."""

    rows: list[dict] = []
    for position, values in {
        "QB": [120, 119, 118, 117, 116, 115],
        "RB": [90, 89, 50, 49, 20, 19, 10, 9],
        "WR": [110, 109, 108, 107, 106, 105, 104, 103, 102, 101],
        "TE": [70, 69, 30, 29, 10, 9],
    }.items():
        rows.extend(
            {
                "player_id": f"{position.lower()}-{index}",
                "name": f"{position} {index}",
                "position": position,
                "league_points": float(value),
            }
            for index, value in enumerate(values, start=1)
        )
    return rows


def test_replacement_rank_changes_for_three_wr_vs_two_flex():
    from src.app.decisions.draft_board import _league_wide_replacement_ranks
    from src.app.scoring.compiler import compile_sleeper_scoring

    three_wr = compile_sleeper_scoring(
        {},
        ["QB", "RB", "RB", "WR", "WR", "WR", "TE"],
    )
    two_flex = compile_sleeper_scoring(
        {},
        ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX"],
    )

    three_wr_ranks = _league_wide_replacement_ranks(
        _format_rows(), three_wr, team_count=2
    )
    two_flex_ranks = _league_wide_replacement_ranks(
        _format_rows(), two_flex, team_count=2
    )

    assert three_wr_ranks["WR"] == 7  # 3 WR starters x 2 teams, then replacement
    assert (
        two_flex_ranks["WR"] == 9
    )  # both FLEX seats per team go to the strong WR pool
    assert three_wr_ranks != two_flex_ranks


def test_replacement_rank_changes_for_superflex():
    from src.app.decisions.draft_board import _league_wide_replacement_ranks
    from src.app.scoring.compiler import compile_sleeper_scoring

    one_qb = compile_sleeper_scoring({}, ["QB", "RB", "WR", "TE"])
    superflex = compile_sleeper_scoring(
        {},
        ["QB", "RB", "WR", "TE", "SUPER_FLEX"],
    )

    one_qb_ranks = _league_wide_replacement_ranks(_format_rows(), one_qb, team_count=2)
    superflex_ranks = _league_wide_replacement_ranks(
        _format_rows(), superflex, team_count=2
    )

    assert one_qb_ranks["QB"] == 3
    assert superflex_ranks["QB"] == 5


def test_league_vorp_is_signed_instead_of_clipped_to_zero():
    from src.app.decisions.draft_board import _apply_league_vorp
    from src.app.scoring.compiler import compile_sleeper_scoring

    contract = compile_sleeper_scoring({}, ["QB", "RB", "WR", "TE"])
    ranked, _replacement_ranks, _replacement_points = _apply_league_vorp(
        _format_rows(),
        contract,
        team_count=2,
    )

    assert any(row["vorp"] > 0 for row in ranked)
    assert any(row["vorp"] < 0 for row in ranked)
    assert sum(row["vorp"] == 0 for row in ranked) == 4
