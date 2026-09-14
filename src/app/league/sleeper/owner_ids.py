"""Resolve configured Sleeper owner ids without importing the decision engine.

``GET /leagues`` needs these candidates on every home load. Keeping the helpers
here (instead of ``decisions.services``) lets that route avoid numpy/draws.
"""

from __future__ import annotations

USERNAME_TO_USER_ID_CACHE: dict[str, str | None] = {}


def configured_sleeper_username(settings) -> str | None:
    username = (settings.sleeper_username or "").strip() or None
    if username:
        return username
    if settings.sleeper_owner_config or settings.sleeper_owner_json:
        try:
            from src.app.league.sleeper.owner_config import load_owner_config

            loaded = (load_owner_config().username or "").strip()
            return loaded or None
        except (FileNotFoundError, OSError, ValueError):
            return None
    return None


def lookup_user_id_for_username(username: str, *, use_fixtures: bool) -> str | None:
    """Resolve a Sleeper username to user_id (cached; read-only GET)."""
    cache_key = username.strip().lower()
    if cache_key in USERNAME_TO_USER_ID_CACHE:
        return USERNAME_TO_USER_ID_CACHE[cache_key]
    try:
        from src.app.league.sleeper.client import SleeperClient

        payload = SleeperClient(use_fixtures=use_fixtures).get_user(username.strip())
        user_id = str(payload.get("user_id") or "").strip() or None
    except Exception:
        user_id = None
    USERNAME_TO_USER_ID_CACHE[cache_key] = user_id
    return user_id


def owner_user_id_candidates(settings) -> list[str]:
    """Ordered Sleeper user ids that may identify the configured owner.

    Prefer the configured ``SLEEPER_USER_ID``. When that is missing or does not
    match league memberships (quoted/stale id, etc.), also try resolving
    ``SLEEPER_USERNAME`` / owner-config username via the Sleeper user endpoint.
    """
    candidates: list[str] = []
    seen: set[str] = set()

    def _add(value: str | None) -> None:
        if not value:
            return
        key = str(value).strip()
        if not key or key in seen:
            return
        seen.add(key)
        candidates.append(key)

    _add(settings.sleeper_user_id)
    username = configured_sleeper_username(settings)
    if username:
        _add(lookup_user_id_for_username(username, use_fixtures=settings.use_sleeper_fixtures))
    return candidates
