"""FastAPI application factory."""

from __future__ import annotations

import re
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import JSONResponse

from src.app.api.v1.router import api_v1_router
from src.app.config import get_settings
from src.app.logging import (
    bind_correlation_id,
    configure_logging,
    current_correlation_id,
    get_logger,
)

logger = get_logger(__name__)

#: Only the methods and headers the browser client actually uses.
ALLOWED_CORS_METHODS = ["GET", "POST", "PUT", "OPTIONS"]
ALLOWED_CORS_HEADERS = ["Content-Type", "X-CSRF-Token", "X-Correlation-ID", "Idempotency-Key"]

#: Correlation ids are echoed into response headers and structured logs, so an
#: inbound value is only trusted when it is uuid/hex shaped and length capped.
CORRELATION_ID_PATTERN = re.compile(r"^[0-9a-fA-F][0-9a-fA-F-]{7,63}$")


def sanitize_correlation_id(raw: str | None) -> str:
    if raw and CORRELATION_ID_PATTERN.match(raw):
        return raw
    return uuid.uuid4().hex


def _error_envelope(code: str, message: str, correlation_id: str, **extra: Any) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "correlation_id": correlation_id, **extra}}


def _correlation_id_of(request: Request) -> str:
    return (
        getattr(request.state, "correlation_id", None)
        or current_correlation_id()
        or "unknown"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(json_logs=settings.log_json, level=settings.log_level)
    # Fail closed: never boot a production process with unsafe configuration.
    settings.validate_production()
    logger.info("app_starting", env=settings.app_env)
    yield
    logger.info("app_stopping")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Fantasy Decision App",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.trusted_host_list,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=ALLOWED_CORS_METHODS,
        allow_headers=ALLOWED_CORS_HEADERS,
    )

    @app.middleware("http")
    async def correlation_middleware(request: Request, call_next):
        cid = sanitize_correlation_id(request.headers.get("X-Correlation-ID"))
        bind_correlation_id(cid)
        request.state.correlation_id = cid
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = cid
        return response

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        cid = _correlation_id_of(request)
        detail = exc.detail
        if isinstance(detail, dict):
            code = str(detail.get("code", f"http_{exc.status_code}"))
            message = str(detail.get("message", ""))
        else:
            code = f"http_{exc.status_code}"
            message = str(detail)
        # ``detail`` is retained for backward compatibility with existing
        # clients; ``error`` is the canonical envelope.
        body = {"detail": detail, **_error_envelope(code, message, cid)}
        return JSONResponse(status_code=exc.status_code, content=body, headers=exc.headers)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        cid = _correlation_id_of(request)
        logger.exception(
            "unhandled_exception",
            path=request.url.path,
            method=request.method,
            exception_type=type(exc).__name__,
        )
        message = "Internal server error. Quote the correlation id when reporting this."
        extra: dict[str, Any] = {}
        if get_settings().is_development:
            # Development only: the message aids local debugging. Never a
            # traceback, and never in production.
            extra["debug_message"] = f"{type(exc).__name__}: {exc}"
        return JSONResponse(
            status_code=500,
            content=_error_envelope("internal_error", message, cid, **extra),
        )

    @app.get("/health/live")
    async def health_live():
        return {"status": "ok"}

    @app.get("/health/ready")
    async def health_ready(request: Request):
        cid = _correlation_id_of(request)
        try:
            from src.app.persistence.database import get_engine

            with get_engine().connect() as conn:
                conn.exec_driver_sql("SELECT 1")
            return {"status": "ready"}
        except Exception:  # noqa: BLE001
            # Database errors routinely contain hostnames, users and DSNs.
            logger.exception("readiness_check_failed", correlation_id=cid)
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    **_error_envelope("dependency_unavailable", "Readiness check failed.", cid),
                },
            )

    app.include_router(api_v1_router, prefix="/api/v1")
    return app


__all__ = ["create_app", "lifespan", "sanitize_correlation_id"]
