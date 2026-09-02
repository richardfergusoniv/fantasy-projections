"""Shared rate limiting with optional Postgres backing for serverless."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from datetime import UTC, datetime, timedelta

from fastapi import HTTPException, Request, status


class RateLimiter:
    def __init__(self) -> None:
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str, *, limit: int, window_seconds: int = 60) -> None:
        if self._check_postgres(key, limit=limit, window_seconds=window_seconds):
            return
        now = time.monotonic()
        bucket = self._events[key]
        while bucket and now - bucket[0] > window_seconds:
            bucket.popleft()
        if len(bucket) >= limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
            )
        bucket.append(now)

    def _check_postgres(self, key: str, *, limit: int, window_seconds: int) -> bool:
        try:
            from sqlalchemy import text

            from src.app.persistence.database import get_session

            now = datetime.now(UTC)
            window_start = now - timedelta(seconds=window_seconds)
            with get_session() as session:
                bind = session.get_bind()
                if bind.dialect.name not in {"postgresql", "sqlite"}:
                    return False
                row = session.execute(
                    text(
                        "SELECT bucket_key, window_start, event_count FROM rate_limit_bucket "
                        "WHERE bucket_key = :key"
                    ),
                    {"key": key},
                ).fetchone()
                if row is None:
                    session.execute(
                        text(
                            "INSERT INTO rate_limit_bucket (bucket_key, window_start, event_count, updated_at) "
                            "VALUES (:key, :window_start, 1, :updated_at)"
                        ),
                        {"key": key, "window_start": now, "updated_at": now},
                    )
                    return True
                row_window = row.window_start
                if row_window.tzinfo is None:
                    row_window = row_window.replace(tzinfo=UTC)
                count = int(row.event_count)
                if row_window < window_start:
                    session.execute(
                        text(
                            "UPDATE rate_limit_bucket SET window_start = :window_start, "
                            "event_count = 1, updated_at = :updated_at WHERE bucket_key = :key"
                        ),
                        {"key": key, "window_start": now, "updated_at": now},
                    )
                    return True
                if count >= limit:
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail="Rate limit exceeded",
                    )
                session.execute(
                    text(
                        "UPDATE rate_limit_bucket SET event_count = event_count + 1, "
                        "updated_at = :updated_at WHERE bucket_key = :key"
                    ),
                    {"key": key, "updated_at": now},
                )
                return True
        except HTTPException:
            raise
        except Exception:
            return False

    def reset(self, key: str | None = None) -> None:
        if key is None:
            self._events.clear()
        else:
            self._events.pop(key, None)


limiter = RateLimiter()


def client_key(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"
