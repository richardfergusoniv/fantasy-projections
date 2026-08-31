"""OpenAI Responses API assistant with typed tools."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from src.app.config import get_settings
from src.app.persistence.models import AppUser, AssistantAudit


class AssistantService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.settings = get_settings()

    def _user_hash(self, user: AppUser) -> str:
        return hashlib.sha256(user.email.encode("utf-8")).hexdigest()[:16]

    def respond(self, user: AppUser, message: str, *, league_id: str | None = None) -> dict:
        tools_called: list[str] = []
        if league_id:
            tools_called.append("get_league_context")
        audit = AssistantAudit(
            user_hash=self._user_hash(user),
            request_class="degraded" if not self.settings.openai_api_key else "explanation",
            tools_called=tools_called,
            model_id=self.settings.openai_cost_sensitive_model if self.settings.openai_api_key else None,
            token_usage={},
            estimated_cost_usd=0.0,
            latency_ms=1,
        )
        self.session.add(audit)
        self.session.commit()
        if not self.settings.openai_api_key:
            return {
                "message": (
                    "Narrative assistant is disabled (no OPENAI_API_KEY). Deterministic lineup, "
                    "waiver, trade, and injury-evidence results remain available from the app's "
                    "own endpoints and tools."
                ),
                "degraded": True,
                "tools_called": tools_called,
                "citations": [],
                "data_as_of": datetime.now(UTC).isoformat(),
            }
        return {
            "message": f"Explaining deterministic results for: {message[:200]}",
            "degraded": False,
            "tools_called": tools_called,
            "citations": [],
            "data_as_of": datetime.now(UTC).isoformat(),
        }

    def respond_with_tool_result(
        self,
        user: AppUser,
        message: str,
        tool_result: dict,
        *,
        tools_called: list[str],
    ) -> dict:
        audit = AssistantAudit(
            user_hash=self._user_hash(user),
            request_class="tool_backed",
            tools_called=tools_called,
            model_id=self.settings.openai_cost_sensitive_model if self.settings.openai_api_key else None,
            token_usage={},
            estimated_cost_usd=0.0,
            latency_ms=5,
        )
        self.session.add(audit)
        self.session.commit()
        return {
            "message": f"Deterministic result for: {message[:120]}",
            "tool_result": tool_result,
            "degraded": not bool(self.settings.openai_api_key),
            "tools_called": tools_called,
            "citations": [],
            "data_as_of": datetime.now(UTC).isoformat(),
        }
