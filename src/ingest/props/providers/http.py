"""HTTP helpers for live sportsbook fetches.

DraftKings (and sometimes other books) sit behind TLS fingerprinting / bot
mitigation that plain ``httpx``/``requests`` cannot clear from typical cloud
egress. Prefer ``curl_cffi`` Chrome impersonation when installed; fall back to
``httpx`` for FanDuel and other permissive hosts.
"""

from __future__ import annotations

from typing import Any, Mapping

import httpx

DEFAULT_BROWSER_HEADERS: dict[str, str] = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
}


class LiveFetchError(RuntimeError):
    """Raised when a live provider HTTP call fails."""


def fetch_json(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout: float = 45.0,
    prefer_curl_cffi: bool = True,
    impersonate: str = "chrome131",
) -> Any:
    merged = dict(DEFAULT_BROWSER_HEADERS)
    if headers:
        merged.update(dict(headers))

    if prefer_curl_cffi:
        try:
            from curl_cffi import requests as cffi_requests
        except ImportError:
            cffi_requests = None  # type: ignore[assignment]
        else:
            response = cffi_requests.get(
                url,
                headers=merged,
                timeout=timeout,
                impersonate=impersonate,
            )
            if response.status_code >= 400:
                raise LiveFetchError(
                    f"HTTP {response.status_code} for {url}: {response.text[:240]}"
                )
            try:
                return response.json()
            except Exception as exc:  # noqa: BLE001
                raise LiveFetchError(f"invalid JSON from {url}: {exc}") from exc

    with httpx.Client(timeout=timeout, follow_redirects=True, headers=merged) as client:
        response = client.get(url)
        if response.status_code >= 400:
            raise LiveFetchError(
                f"HTTP {response.status_code} for {url}: {response.text[:240]}"
            )
        try:
            return response.json()
        except Exception as exc:  # noqa: BLE001
            raise LiveFetchError(f"invalid JSON from {url}: {exc}") from exc
