"""Simple in-memory rate limiting for auth and assistant routes."""

from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request, status


class RateLimiter:
    def __init__(self) -> None:
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str, *, limit: int, window_seconds: int = 60) -> None:
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

    def reset(self, key: str | None = None) -> None:
        """Clear one bucket, or all buckets when no key is given.

        Buckets live in process memory, so this is also what a process restart
        does implicitly; tests use it to isolate the shared limiter instance.
        """
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
