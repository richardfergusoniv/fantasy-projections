"""Injury research providers.

Two modes exist and they are never interchangeable: ``fixture`` produces clearly
synthetic evidence for local development, ``live`` performs real cited research
and requires explicit configuration. When live research is requested but not
configured the caller must report research as unavailable — inventing fixture
evidence in a production path is what this module exists to prevent.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime

from src.app.availability.service import (
    RESEARCH_MODE_FIXTURE,
    RESEARCH_MODE_LIVE,
    EvidenceClaim,
)

MODE_FIXTURE = RESEARCH_MODE_FIXTURE
MODE_LIVE = RESEARCH_MODE_LIVE
MODE_DISABLED = "disabled"
VALID_MODES = {MODE_FIXTURE, MODE_LIVE, MODE_DISABLED}

RESEARCH_MODE_ENV = "INJURY_RESEARCH_MODE"


class ResearchUnavailable(RuntimeError):
    """Raised when the requested research mode has no usable provider."""


@dataclass
class ResearchResult:
    claims: list[EvidenceClaim]
    model_id: str | None
    citations: list[dict[str, str]]
    mode: str = MODE_FIXTURE
    synthetic: bool = True


class InjuryResearchProvider(ABC):
    mode: str = MODE_FIXTURE
    synthetic: bool = True

    @abstractmethod
    def research(self, player_id: str, *, as_of: str | None = None) -> ResearchResult:
        raise NotImplementedError


class FixtureInjuryResearchProvider(InjuryResearchProvider):
    """Deterministic synthetic evidence.

    Citations use the ``fixture://`` scheme so they can never be mistaken for a
    news source, and every claim carries ``synthetic: true``.
    """

    mode = MODE_FIXTURE
    synthetic = True

    def research(self, player_id: str, *, as_of: str | None = None) -> ResearchResult:
        observed_at = as_of or datetime.now(UTC).isoformat()
        sources = [
            {
                "url": f"fixture://synthetic-injury-report/{player_id}",
                "title": f"SYNTHETIC fixture injury note for {player_id}",
                "published_at": observed_at,
                "publisher": "synthetic-fixture",
            }
        ]
        claim = EvidenceClaim(
            player_id=player_id,
            status="questionable",
            reported_injury="fixture injury (synthetic)",
            expected_return_min=None,
            expected_return_max=None,
            claim_confidence=0.2,
            sources=sources,
            mode=MODE_FIXTURE,
            synthetic=True,
            publisher="synthetic-fixture",
            source_reliability=0.0,
            published_at=observed_at,
            retrieved_at=observed_at,
        )
        return ResearchResult(
            claims=[claim],
            model_id="fixture",
            citations=sources,
            mode=MODE_FIXTURE,
            synthetic=True,
        )


class OpenAIInjuryResearchProvider(InjuryResearchProvider):
    """Live web-search research. Requires a configured API key at runtime."""

    mode = MODE_LIVE
    synthetic = False

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ResearchUnavailable("live injury research requires an OpenAI API key")
        self.api_key = api_key
        self.model = model

    def research(self, player_id: str, *, as_of: str | None = None) -> ResearchResult:
        raise ResearchUnavailable(
            "live injury research is configured but the web-search integration is not implemented; "
            "no evidence will be fabricated"
        )


def resolve_research_mode(settings, env: dict[str, str] | None = None) -> str:
    """Explicit env override wins; otherwise fixtures outside production."""

    environ = env if env is not None else os.environ
    requested = (environ.get(RESEARCH_MODE_ENV) or "").strip().lower()
    if requested:
        if requested not in VALID_MODES:
            raise ResearchUnavailable(f"unknown {RESEARCH_MODE_ENV}: {requested}")
        return requested
    if getattr(settings, "app_env", "development") == "production":
        return MODE_LIVE
    return MODE_FIXTURE


def build_provider(settings, *, mode: str) -> InjuryResearchProvider:
    """Return the provider for ``mode``, never a silent substitute."""

    if mode == MODE_FIXTURE:
        return FixtureInjuryResearchProvider()
    if mode == MODE_LIVE:
        api_key = getattr(settings, "openai_api_key", None)
        if not api_key:
            raise ResearchUnavailable("live_research_not_configured")
        return OpenAIInjuryResearchProvider(api_key, getattr(settings, "openai_balanced_model", "gpt-4.1"))
    raise ResearchUnavailable(f"research mode is {mode}")
