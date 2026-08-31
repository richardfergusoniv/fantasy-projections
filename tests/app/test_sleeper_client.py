"""Sleeper client hardening: retries, rate limits, staleness, and read-only guarantee."""

from __future__ import annotations

import logging
import random
from dataclasses import replace
from datetime import timedelta

import httpx
import pytest


def _client(handler, **overrides):
    from src.app.league.sleeper.client import SleeperClient

    slept: list[float] = []
    options = {
        "transport": httpx.MockTransport(handler),
        "sleep": slept.append,
        "backoff_base_seconds": 0.01,
        "backoff_max_seconds": 0.05,
        "rng": random.Random(1234),
    }
    options.update(overrides)
    return SleeperClient(**options), slept


def _players_payload(count: int) -> dict:
    return {
        f"sl-{index}": {"player_id": f"sl-{index}", "full_name": f"Player {index}", "injury_status": None}
        for index in range(count)
    }


def test_transient_server_errors_are_retried_then_succeed():
    attempts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request.method)
        if len(attempts) < 3:
            return httpx.Response(500)
        return httpx.Response(200, json={"user_id": "u-1", "username": "someone"})

    client, slept = _client(handler)
    try:
        data = client.get_user("someone")
    finally:
        client.close()

    assert data["user_id"] == "u-1"
    assert len(attempts) == 3
    assert set(attempts) == {"GET"}
    assert len(slept) == 2
    assert all(0.0 <= delay <= 0.05 for delay in slept)


def test_rate_limit_is_surfaced_after_bounded_retries():
    from src.app.league.sleeper.client import SleeperRateLimited

    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(429, headers={"Retry-After": "600"}, json={"error": "slow down"})

    client, slept = _client(handler, max_attempts=3)
    try:
        with pytest.raises(SleeperRateLimited) as excinfo:
            client.get_league("league-1")
    finally:
        client.close()

    assert len(attempts) == 3
    assert "429" in str(excinfo.value)
    assert "league/league-1" in str(excinfo.value)
    # Retry-After is honored but clamped so a hostile header cannot stall a job.
    assert slept == [0.05, 0.05]


def test_timeout_is_bounded_and_surfaced():
    from src.app.league.sleeper.client import SleeperClient, SleeperTimeout

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    client, slept = _client(handler, max_attempts=2)
    try:
        assert client.timeout == 30.0
        with pytest.raises(SleeperTimeout, match="timeout after 30.0s"):
            client.get_nfl_state()
    finally:
        client.close()
    assert len(slept) == 1

    bounded = SleeperClient(timeout=10_000)
    assert bounded.timeout == 120.0


def test_client_is_read_only():
    from src.app.league.sleeper.client import SleeperClient, SleeperError

    methods: list[str] = []
    payloads = {
        "/v1/user/someone": {"user_id": "u-1"},
        "/v1/user/u-1/leagues/nfl/2026": [{"league_id": "l-1", "season": 2026}],
        "/v1/league/l-1": {"league_id": "l-1", "scoring_settings": {"rec": 1}},
        "/v1/league/l-1/rosters": [{"roster_id": 1, "owner_id": "u-1"}],
        "/v1/league/l-1/users": [{"user_id": "u-1"}],
        "/v1/league/l-1/matchups/1": [{"roster_id": 1, "matchup_id": 1}],
        "/v1/league/l-1/transactions/1": [],
        "/v1/league/l-1/traded_picks": [],
        "/v1/league/l-1/drafts": [],
        "/v1/state/nfl": {"week": 1},
        "/v1/players/nfl/trending/add": [{"player_id": "sl-1", "count": 5}],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        return httpx.Response(200, json=payloads[request.url.path])

    client, _ = _client(handler)
    try:
        client.get_user("someone")
        client.get_leagues("u-1", 2026)
        client.get_league("l-1")
        client.get_rosters("l-1")
        client.get_users("l-1")
        client.get_matchups("l-1", 1)
        client.get_transactions("l-1", 1)
        client.get_traded_picks("l-1")
        client.get_drafts("l-1")
        client.get_nfl_state()
        client.get_trending_players()
    finally:
        client.close()

    assert methods
    assert set(methods) == {"GET"}
    assert not [name for name in dir(SleeperClient) if name.startswith(("post", "put", "patch", "delete"))]
    with pytest.raises(SleeperError, match="read-only"):
        SleeperClient._assert_read_only("POST")


def test_completeness_classification_is_endpoint_aware():
    from src.app.league.sleeper.client import SleeperClient

    live = SleeperClient()
    fixtures = SleeperClient(use_fixtures=True)

    # An empty transaction week is a legitimate answer, not a truncated payload.
    assert live.classify_completeness("league/l-1/transactions/3", []) is True
    assert live.classify_completeness("league/l-1/traded_picks", []) is True
    # Rosters and users are never legitimately empty for a real league.
    assert live.classify_completeness("league/l-1/rosters", []) is False
    assert live.classify_completeness("league/l-1/users", []) is False
    # A four-player `players/nfl` response is a truncated payload live, but the
    # expected size for a fixture.
    assert live.classify_completeness("players/nfl", _players_payload(4)) is False
    assert live.classify_completeness("players/nfl", _players_payload(1_500)) is True
    assert fixtures.classify_completeness("players/nfl", _players_payload(4)) is True
    assert live.classify_completeness("state/nfl", {"week": 1}) is True
    assert live.classify_completeness("state/nfl", {}) is False
    assert live.classify_completeness("players/nfl", None) is False


def test_failed_refresh_serves_payload_marked_stale():
    from src.app.league.sleeper.client import SleeperUnavailable

    responses = [httpx.Response(200, json=_players_payload(1_200))]

    def handler(request: httpx.Request) -> httpx.Response:
        if responses:
            return responses.pop()
        return httpx.Response(503)

    client, _ = _client(handler, max_attempts=1, players_ttl_seconds=3600)
    try:
        fresh = client.get_players_with_metadata()
        assert fresh.stale is False
        assert fresh.error is None
        assert len(fresh.data) == 1_200

        # Age the cache past its TTL so the next read must refresh.
        client._player_cache = replace(fresh, fetched_at=fresh.fetched_at - timedelta(hours=12))
        stale = client.get_players_with_metadata()

        assert stale.stale is True
        assert stale.error is not None
        assert len(stale.data) == 1_200
        assert stale.to_metadata()["stale"] is True
        assert stale.age_seconds > 3600

        meta = client.last_snapshot_meta
        assert meta["health_verdict"] == "stale"

        client._player_cache = None
        with pytest.raises(SleeperUnavailable):
            client.get_players_with_metadata()
    finally:
        client.close()


def test_player_cache_refreshes_after_ttl():
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, json=_players_payload(1_100 + len(calls)))

    client, _ = _client(handler, players_ttl_seconds=3600)
    try:
        first = client.get_players_with_metadata()
        client.get_players_with_metadata()
        assert len(calls) == 1

        client._player_cache = replace(first, fetched_at=first.fetched_at - timedelta(hours=2))
        refreshed = client.get_players_with_metadata()
        assert len(calls) == 2
        assert refreshed.stale is False
        assert refreshed.fetched_at > first.fetched_at
    finally:
        client.close()


class _RecordingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def test_retry_logs_do_not_leak_usernames_or_payloads():
    from src.app.league.sleeper.client import redact_endpoint

    secret_payload = {"email": "owner@example.com", "user_id": "u-1"}
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) < 2:
            return httpx.Response(500)
        return httpx.Response(200, json=secret_payload)

    logger = logging.getLogger("src.app.league.sleeper.client")
    recorder = _RecordingHandler()
    logger.addHandler(recorder)
    previous_level = logger.level
    logger.setLevel(logging.DEBUG)
    client, _ = _client(handler)
    try:
        client.get_user("private-username")
    finally:
        client.close()
        logger.removeHandler(recorder)
        logger.setLevel(previous_level)

    assert recorder.records
    logged = "\n".join(
        f"{record.getMessage()} {record.__dict__.get('endpoint', '')}" for record in recorder.records
    )
    assert "private-username" not in logged
    assert "owner@example.com" not in logged
    assert "<redacted>" in logged
    assert redact_endpoint("user/private-username/leagues/nfl/2026") == "user/<redacted>/leagues/nfl/2026"


def test_fixture_mode_never_falls_through_to_network():
    from src.app.league.sleeper.client import SleeperClient, SleeperFixtureMissing

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        raise AssertionError("fixture mode must not perform network calls")

    client = SleeperClient(use_fixtures=True, transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(SleeperFixtureMissing):
            client.get_league("league-that-has-no-fixture")
    finally:
        client.close()
