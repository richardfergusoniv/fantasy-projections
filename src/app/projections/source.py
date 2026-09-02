"""Projection source modes and configuration contract.

The application never silently switches projection sources. Every production path
defaults to ``sealed_release``; experimental weekly-v2 requires explicit opt-in.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from src.app.config import get_settings

ProjectionSourceName = Literal["sealed_release", "status_adjusted_release", "weekly_v2_rnd"]


class ProjectionSource(str, Enum):
    """Typed projection-source contract."""

    SEALED_RELEASE = "sealed_release"
    STATUS_ADJUSTED_RELEASE = "status_adjusted_release"
    WEEKLY_V2_RND = "weekly_v2_rnd"

    @classmethod
    def parse(cls, raw: str | None) -> "ProjectionSource":
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
        return self in {self.SEALED_RELEASE, self.STATUS_ADJUSTED_RELEASE}

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


def resolve_effective_source(requested: ProjectionSource | None = None) -> ProjectionSource:
    """Return the source that will actually be used for a request.

  ``weekly_v2_rnd`` is returned only when both configured/enabled and explicitly
  requested; it is never selected implicitly.
    """
    configured = configured_projection_source()
    if requested is None:
        return configured
    if requested == ProjectionSource.WEEKLY_V2_RND:
        if not weekly_rnd_enabled():
            raise ValueError("weekly_v2_rnd requires WEEKLY_RND_ENABLED=true")
        return requested
    if requested == ProjectionSource.STATUS_ADJUSTED_RELEASE:
        return requested
    return ProjectionSource.SEALED_RELEASE
