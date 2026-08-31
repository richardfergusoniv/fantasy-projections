"""Structured JSON logging with correlation IDs and secret redaction."""

from __future__ import annotations

import logging
import re
import sys
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, TypeVar

import structlog

correlation_id_var: ContextVar[str | None] = ContextVar("correlation_id", default=None)

REDACTED = "[REDACTED]"
REDACTED_EMAIL = "[REDACTED_EMAIL]"

#: Keys whose *values* are never safe to render, regardless of nesting depth.
SECRET_KEY_PATTERN = re.compile(
    r"(token|secret|password|api[_-]?key|apikey|authorization|cookie|session)",
    re.IGNORECASE,
)
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

_MAX_REDACT_DEPTH = 6

T = TypeVar("T")


def new_correlation_id() -> str:
    return uuid.uuid4().hex


def bind_correlation_id(correlation_id: str | None = None) -> str:
    cid = correlation_id or new_correlation_id()
    correlation_id_var.set(cid)
    structlog.contextvars.bind_contextvars(correlation_id=cid)
    return cid


def current_correlation_id() -> str | None:
    return correlation_id_var.get()


@contextmanager
def correlation_scope(correlation_id: str | None = None) -> Iterator[str]:
    """Bind a correlation id for the duration of a block, then restore.

    Use this to carry a request's correlation id into a background job or
    thread so job logs can be joined back to the originating request.
    """
    previous = correlation_id_var.get()
    cid = bind_correlation_id(correlation_id)
    try:
        yield cid
    finally:
        correlation_id_var.set(previous)
        if previous is None:
            structlog.contextvars.unbind_contextvars("correlation_id")
        else:
            structlog.contextvars.bind_contextvars(correlation_id=previous)


def run_with_correlation(correlation_id: str | None, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run ``fn`` with ``correlation_id`` bound to the logging context."""
    with correlation_scope(correlation_id):
        return fn(*args, **kwargs)


def redact_text(value: str) -> str:
    """Mask email addresses inside a rendered string value."""
    return EMAIL_PATTERN.sub(REDACTED_EMAIL, value)


def _redact_value(value: Any, *, depth: int) -> Any:
    if depth > _MAX_REDACT_DEPTH:
        return value
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return _redact_mapping(value, depth=depth + 1)
    if isinstance(value, (list, tuple, set)):
        rendered = [_redact_value(item, depth=depth + 1) for item in value]
        return type(value)(rendered) if isinstance(value, (list, tuple)) else set(rendered)
    return value


def _redact_mapping(mapping: dict[Any, Any], *, depth: int) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key, value in mapping.items():
        if isinstance(key, str) and SECRET_KEY_PATTERN.search(key):
            result[key] = REDACTED
            continue
        result[key] = _redact_value(value, depth=depth)
    return result


def redact_processor(_logger: Any, _method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """structlog processor that drops secret-ish values and masks emails.

    The correlation id is preserved verbatim because it is generated (or
    validated) server-side and is required for joining logs to requests.
    """
    correlation_id = event_dict.get("correlation_id")
    redacted = _redact_mapping(event_dict, depth=0)
    if correlation_id is not None:
        redacted["correlation_id"] = correlation_id
    return redacted


def configure_logging(*, json_logs: bool = True, level: str = "INFO") -> None:
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]
    if json_logs:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.format_exc_info,
            redact_processor,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level.upper())


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
