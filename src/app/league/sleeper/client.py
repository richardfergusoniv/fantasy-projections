"""Read-only Sleeper API client with fixture support.

The client issues GET requests only. Every request path runs through
:meth:`SleeperClient._request`, which asserts the method before touching the
network, so the read-only guarantee is enforced in code rather than by
convention.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import httpx

from src.app.artifacts.store import get_artifact_store

logger = logging.getLogger(__name__)

SLEEPER_BASE = "https://api.sleeper.app/v1"
FIXTURE_ROOT = Path(__file__).resolve().parents[4] / "tests" / "fixtures" / "sleeper"

READ_ONLY_METHOD = "GET"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BACKOFF_BASE_SECONDS = 0.5
DEFAULT_BACKOFF_MAX_SECONDS = 8.0
DEFAULT_PLAYERS_TTL_SECONDS = 6 * 60 * 60

#: Smallest plausible `players/nfl` payload from the live API.
LIVE_MIN_PLAYER_ROWS = 1_000
FIXTURE_MIN_PLAYER_ROWS = 1

_USER_SEGMENT = re.compile(r"(?<=user/)[^/]+")


class SleeperError(RuntimeError):
    """A Sleeper request could not be completed."""


class SleeperRateLimited(SleeperError):
    """Sleeper returned 429."""


class SleeperUnavailable(SleeperError):
    """Sleeper returned 5xx or the transport failed."""


class SleeperTimeout(SleeperError):
    """Sleeper did not answer within the bounded timeout."""


class SleeperFixtureMissing(SleeperError):
    """Fixture mode was asked for an endpoint with no recorded fixture."""


@dataclass(frozen=True)
class CachedPayload:
    """A cached payload that always carries its own freshness."""

    data: Any
    fetched_at: datetime
    ttl_seconds: int
    stale: bool = False
    error: str | None = None

    @property
    def age_seconds(self) -> float:
        return (datetime.now(UTC) - self.fetched_at).total_seconds()

    def to_metadata(self) -> dict[str, Any]:
        return {
            "fetched_at": self.fetched_at.isoformat(),
            "ttl_seconds": self.ttl_seconds,
            "stale": self.stale,
            "age_seconds": round(self.age_seconds, 3),
            "error": self.error,
        }


def redact_endpoint(endpoint: str) -> str:
    """Usernames appear in Sleeper paths; keep them out of logs."""

    return _USER_SEGMENT.sub("<redacted>", endpoint)


class SleeperClient:
    def __init__(
        self,
        *,
        use_fixtures: bool = False,
        fixture_root: Path | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        backoff_base_seconds: float = DEFAULT_BACKOFF_BASE_SECONDS,
        backoff_max_seconds: float = DEFAULT_BACKOFF_MAX_SECONDS,
        players_ttl_seconds: int = DEFAULT_PLAYERS_TTL_SECONDS,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.use_fixtures = use_fixtures
        self.fixture_root = fixture_root or FIXTURE_ROOT
        self.store = get_artifact_store()
        self.timeout = max(0.1, min(float(timeout), 120.0))
        self.max_attempts = max(1, int(max_attempts))
        self.backoff_base_seconds = max(0.0, float(backoff_base_seconds))
        self.backoff_max_seconds = max(self.backoff_base_seconds, float(backoff_max_seconds))
        self.players_ttl_seconds = max(1, int(players_ttl_seconds))
        self._transport = transport
        self._sleep = sleep or time.sleep
        self._rng = rng or random.Random()
        self._http: httpx.Client | None = None
        self._player_cache: CachedPayload | None = None
        self.last_snapshot_meta: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    def _client(self) -> httpx.Client:
        if self._http is None:
            self._http = httpx.Client(timeout=self.timeout, transport=self._transport)
        return self._http

    def close(self) -> None:
        if self._http is not None:
            self._http.close()
            self._http = None

    def __enter__(self) -> SleeperClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @staticmethod
    def _assert_read_only(method: str) -> None:
        if method.upper() != READ_ONLY_METHOD:
            raise SleeperError(f"Sleeper client is read-only; refused {method.upper()}")

    def _backoff_delay(self, attempt: int, retry_after: float | None = None) -> float:
        if retry_after is not None:
            return min(max(retry_after, 0.0), self.backoff_max_seconds)
        window = min(self.backoff_base_seconds * (2 ** (attempt - 1)), self.backoff_max_seconds)
        # Full jitter keeps retries from synchronizing across jobs.
        return self._rng.uniform(0.0, window)

    @staticmethod
    def _retry_after_seconds(response: httpx.Response) -> float | None:
        raw = response.headers.get("retry-after")
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    def _request(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        self._assert_read_only(READ_ONLY_METHOD)
        url = f"{SLEEPER_BASE}/{endpoint.lstrip('/')}"
        safe_endpoint = redact_endpoint(endpoint)
        last_error: SleeperError | None = None
        for attempt in range(1, self.max_attempts + 1):
            retry_after: float | None = None
            try:
                response = self._client().request(READ_ONLY_METHOD, url, params=params)
            except httpx.TimeoutException as exc:
                last_error = SleeperTimeout(
                    f"sleeper timeout after {self.timeout}s on {safe_endpoint} (attempt {attempt})"
                )
                logger.warning(
                    "sleeper_timeout",
                    extra={"endpoint": safe_endpoint, "attempt": attempt, "timeout": self.timeout},
                )
                _ = exc
            except httpx.HTTPError as exc:
                last_error = SleeperUnavailable(
                    f"sleeper transport error on {safe_endpoint} (attempt {attempt}): {type(exc).__name__}"
                )
                logger.warning(
                    "sleeper_transport_error",
                    extra={"endpoint": safe_endpoint, "attempt": attempt, "error": type(exc).__name__},
                )
            else:
                status = response.status_code
                if status == 429:
                    retry_after = self._retry_after_seconds(response)
                    last_error = SleeperRateLimited(
                        f"sleeper rate limited (429) on {safe_endpoint} after {attempt} attempt(s)"
                        + (f"; retry-after {retry_after}s" if retry_after else "")
                    )
                    logger.warning(
                        "sleeper_rate_limited",
                        extra={"endpoint": safe_endpoint, "attempt": attempt, "retry_after": retry_after},
                    )
                elif 500 <= status < 600:
                    last_error = SleeperUnavailable(
                        f"sleeper server error {status} on {safe_endpoint} after {attempt} attempt(s)"
                    )
                    logger.warning(
                        "sleeper_server_error",
                        extra={"endpoint": safe_endpoint, "attempt": attempt, "status": status},
                    )
                elif status >= 400:
                    raise SleeperError(f"sleeper client error {status} on {safe_endpoint}")
                else:
                    return response.json()
            if attempt >= self.max_attempts:
                break
            self._sleep(self._backoff_delay(attempt, retry_after))
        raise last_error or SleeperError(f"sleeper request failed on {safe_endpoint}")

    def _fixture_path(self, endpoint: str) -> Path:
        safe = endpoint.strip("/").replace("/", "__")
        return self.fixture_root / f"{safe}.json"

    def _fetch(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        if self.use_fixtures:
            path = self._fixture_path(endpoint)
            if not path.exists():
                # Fixture mode must never silently fall through to the network.
                raise SleeperFixtureMissing(f"no fixture for {redact_endpoint(endpoint)}")
            return json.loads(path.read_text(encoding="utf-8"))
        return self._request(endpoint, params)

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def _min_player_rows(self) -> int:
        return FIXTURE_MIN_PLAYER_ROWS if self.use_fixtures else LIVE_MIN_PLAYER_ROWS

    def record_count(self, payload: Any) -> int | None:
        if isinstance(payload, (list, dict)):
            return len(payload)
        return None

    def classify_completeness(self, endpoint: str, payload: Any) -> bool:
        """Endpoint-aware completeness.

        An empty list is a legitimate answer for transactions, traded picks and
        trending adds; it is not a legitimate answer for rosters or users, and a
        tiny `players/nfl` payload means the fetch was truncated.
        """

        normalized = endpoint.strip("/")
        if payload is None:
            return False
        if normalized == "players/nfl":
            return isinstance(payload, dict) and len(payload) >= self._min_player_rows()
        if normalized.endswith(("/rosters", "/users")):
            return isinstance(payload, list) and len(payload) >= 1
        if isinstance(payload, list):
            # matchups, transactions, traded_picks, drafts, leagues, trending adds
            return True
        if isinstance(payload, dict):
            return len(payload) > 0
        return False

    def persist_snapshot(
        self,
        endpoint: str,
        payload: Any,
        params: dict | None = None,
        *,
        stale: bool = False,
        fetched_at: datetime | None = None,
    ) -> dict:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        body_hash = hashlib.sha256(encoded).hexdigest()
        uri = self.store.put_json(payload)
        count = self.record_count(payload)
        request_params = dict(params or {})
        if count is not None:
            request_params.setdefault("record_count", count)
        if stale:
            verdict = "stale"
        elif isinstance(payload, (dict, list)):
            verdict = "healthy"
        else:
            verdict = "unhealthy"
        meta = {
            "endpoint": endpoint,
            "request_params": request_params,
            "fetched_at": (fetched_at or datetime.now(UTC)).isoformat(),
            "body_hash": body_hash,
            "artifact_uri": uri,
            "health_verdict": verdict,
            "is_complete": self.classify_completeness(endpoint, payload),
            "record_count": count,
            "stale": stale,
        }
        self.last_snapshot_meta = meta
        return meta

    # ------------------------------------------------------------------
    # Endpoints (GET only)
    # ------------------------------------------------------------------

    def get_user(self, username: str) -> dict:
        data = self._fetch(f"user/{username}")
        self.persist_snapshot(f"user/{username}", data)
        return data

    def get_leagues(self, user_id: str, season: int) -> list[dict]:
        data = self._fetch(f"user/{user_id}/leagues/nfl/{season}")
        self.persist_snapshot(f"user/{user_id}/leagues/nfl/{season}", data)
        return data

    def get_league(self, league_id: str) -> dict:
        data = self._fetch(f"league/{league_id}")
        self.persist_snapshot(f"league/{league_id}", data)
        return data

    def get_rosters(self, league_id: str) -> list[dict]:
        data = self._fetch(f"league/{league_id}/rosters")
        self.persist_snapshot(f"league/{league_id}/rosters", data)
        return data

    def get_users(self, league_id: str) -> list[dict]:
        data = self._fetch(f"league/{league_id}/users")
        self.persist_snapshot(f"league/{league_id}/users", data)
        return data

    def get_matchups(self, league_id: str, week: int) -> list[dict]:
        data = self._fetch(f"league/{league_id}/matchups/{week}")
        self.persist_snapshot(f"league/{league_id}/matchups/{week}", data, {"week": week})
        return data

    def get_transactions(self, league_id: str, week: int) -> list[dict]:
        data = self._fetch(f"league/{league_id}/transactions/{week}")
        self.persist_snapshot(f"league/{league_id}/transactions/{week}", data, {"week": week})
        return data

    def get_traded_picks(self, league_id: str) -> list[dict]:
        data = self._fetch(f"league/{league_id}/traded_picks")
        self.persist_snapshot(f"league/{league_id}/traded_picks", data)
        return data

    def get_drafts(self, league_id: str) -> list[dict]:
        data = self._fetch(f"league/{league_id}/drafts")
        self.persist_snapshot(f"league/{league_id}/drafts", data)
        return data

    def get_nfl_state(self) -> dict:
        data = self._fetch("state/nfl")
        self.persist_snapshot("state/nfl", data)
        return data

    def get_players_with_metadata(self) -> CachedPayload:
        """Return the player payload with explicit freshness.

        A failed refresh returns the previous payload marked ``stale`` so callers
        never treat cached data as current.
        """

        cached = self._player_cache
        if cached is not None and not cached.stale and cached.age_seconds < self.players_ttl_seconds:
            self.persist_snapshot("players/nfl", cached.data, fetched_at=cached.fetched_at)
            return cached
        try:
            data = self._fetch("players/nfl")
        except SleeperError as exc:
            if cached is None:
                raise
            stale = replace(cached, stale=True, error=str(exc))
            self._player_cache = stale
            logger.warning(
                "sleeper_players_stale",
                extra={"age_seconds": round(stale.age_seconds, 1), "error": type(exc).__name__},
            )
            self.persist_snapshot("players/nfl", stale.data, stale=True, fetched_at=stale.fetched_at)
            return stale
        payload = CachedPayload(
            data=data,
            fetched_at=datetime.now(UTC),
            ttl_seconds=self.players_ttl_seconds,
            stale=False,
        )
        self._player_cache = payload
        self.persist_snapshot("players/nfl", data, fetched_at=payload.fetched_at)
        return payload

    def get_players(self) -> dict[str, Any]:
        payload = self.get_players_with_metadata()
        if payload.stale:
            logger.warning("sleeper_players_served_stale", extra={"age_seconds": round(payload.age_seconds, 1)})
        return payload.data

    def get_trending_players(self, *, lookback_hours: int = 24, limit: int = 25) -> list[dict]:
        params = {"lookback_hours": lookback_hours, "limit": limit}
        data = self._fetch("players/nfl/trending/add", params=params)
        self.persist_snapshot("players/nfl/trending/add", data, params)
        return data

    def players_cache_expires_at(self) -> datetime | None:
        if self._player_cache is None:
            return None
        return self._player_cache.fetched_at + timedelta(seconds=self.players_ttl_seconds)
