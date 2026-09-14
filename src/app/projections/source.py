"""Projection source modes and configuration contract.

The application never silently switches projection sources. Every production path
defaults to ``sealed_release``; experimental weekly-v2 requires explicit opt-in.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Literal

from src.app.config import WEEKLY_PROPS_SHADOW_ENV_KEYS, get_settings

ProjectionSourceName = Literal[
    "sealed_release",
    "status_adjusted_release",
    "weekly_v2_rnd",
    "weekly_props",
]


class ProjectionSource(str, Enum):
    """Typed projection-source contract."""

    SEALED_RELEASE = "sealed_release"
    STATUS_ADJUSTED_RELEASE = "status_adjusted_release"
    WEEKLY_V2_RND = "weekly_v2_rnd"
    WEEKLY_PROPS = "weekly_props"

    @classmethod
    def parse(cls, raw: str | None) -> ProjectionSource:
        if raw is None or not str(raw).strip():
            return cls.SEALED_RELEASE
        normalized = str(raw).strip().lower()
        try:
            return cls(normalized)
        except ValueError as exc:
            raise ValueError(
                f"unknown projection source {raw!r}; "
                f"allowed: {', '.join(s.value for s in cls)}"
            ) from exc

    @property
    def is_production(self) -> bool:
        return self in {
            self.SEALED_RELEASE,
            self.STATUS_ADJUSTED_RELEASE,
            self.WEEKLY_PROPS,
        }

    @property
    def is_experimental(self) -> bool:
        return self == self.WEEKLY_V2_RND


def configured_projection_source() -> ProjectionSource:
    """Resolve the configured default projection source (fail closed on unknown)."""
    settings = get_settings()
    return ProjectionSource.parse(settings.app_projection_source)


def weekly_rnd_enabled() -> bool:
    """True only when weekly-v2 R&D source is explicitly enabled."""
    settings = get_settings()
    return bool(settings.weekly_rnd_enabled)


def parse_env_bool(raw: str) -> bool | None:
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def weekly_props_shadow_env_aliases_in_use() -> list[str]:
    """Non-canonical env keys that are set (usually a mistyped dashboard name)."""
    found: list[str] = []
    for key in WEEKLY_PROPS_SHADOW_ENV_KEYS:
        if key == "WEEKLY_PROPS_SHADOW_ONLY":
            continue
        raw = os.getenv(key)
        if raw is not None and str(raw).strip():
            found.append(key)
    return found


def weekly_props_shadow_only() -> bool:
    """When true, weekly_props jobs evaluate candidates but never swap pointers.

    Safe default is False: production jobs auto-promote after quality gates pass.
    Advanced escape hatch is ``WEEKLY_PROPS_FORCE_SHADOW=true``. Historical
    ``WEEKLY_PROPS_SHADOW_ONLY=true`` (and dashboard typos) are ignored so a
    leftover secret from older examples does not require an env flip.
    """
    settings = get_settings()
    raw = os.getenv("WEEKLY_PROPS_FORCE_SHADOW")
    if raw is not None and str(raw).strip():
        parsed = parse_env_bool(str(raw))
        if parsed is not None:
            return parsed
    return bool(getattr(settings, "weekly_props_force_shadow", False))


def resolve_effective_source(requested: ProjectionSource | None = None) -> ProjectionSource:
    """Return the source that will actually be used for a request.

    ``weekly_v2_rnd`` is returned only when both configured/enabled and explicitly
    requested; it is never selected implicitly. ``weekly_props`` is a production
    source and may be configured as the default.
    """
    configured = configured_projection_source()
    if requested is None:
        return configured
    if requested == ProjectionSource.WEEKLY_V2_RND:
        if not weekly_rnd_enabled():
            raise ValueError("weekly_v2_rnd requires WEEKLY_RND_ENABLED=true")
        return requested
    if requested == ProjectionSource.WEEKLY_PROPS:
        return requested
    if requested == ProjectionSource.STATUS_ADJUSTED_RELEASE:
        return requested
    return ProjectionSource.SEALED_RELEASE
