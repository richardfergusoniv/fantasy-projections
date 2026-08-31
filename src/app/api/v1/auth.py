"""Authentication endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, EmailStr, Field

from src.app.api.deps import get_auth_service, get_current_user, require_csrf
from src.app.auth.service import AuthService
from src.app.config import get_settings
from src.app.logging import get_logger
from src.app.middleware.rate_limit import client_key, limiter
from src.app.persistence.models import AppUser

logger = get_logger(__name__)

router = APIRouter()

SESSION_COOKIE = "session"
SESSION_MAX_AGE_SECONDS = 60 * 60 * 24 * 30


class MagicLinkRequest(BaseModel):
    email: EmailStr


class MagicLinkVerify(BaseModel):
    token: str = Field(min_length=16, max_length=512)


def _set_session_cookie(response: Response, raw_token: str) -> None:
    settings = get_settings()
    response.set_cookie(
        key=SESSION_COOKIE,
        value=raw_token,
        httponly=True,
        samesite="lax",
        secure=settings.session_cookie_secure,
        max_age=SESSION_MAX_AGE_SECONDS,
        path="/",
    )


@router.post("/auth/magic-link")
def request_magic_link(payload: MagicLinkRequest, request: Request, auth: AuthService = Depends(get_auth_service)):
    settings = get_settings()
    limiter.check(f"auth:{client_key(request)}", limit=settings.auth_rate_limit_per_minute)
    return auth.request_magic_link(payload.email)


@router.post("/auth/verify")
def verify_magic_link(
    payload: MagicLinkVerify,
    request: Request,
    response: Response,
    auth: AuthService = Depends(get_auth_service),
):
    # Verification is the brute-force surface for magic-link tokens, so it gets
    # its own bucket rather than sharing the request bucket.
    settings = get_settings()
    limiter.check(f"auth-verify:{client_key(request)}", limit=settings.auth_rate_limit_per_minute)
    try:
        session = auth.verify_magic_link(payload.token)
    except ValueError:
        logger.warning("magic_link_verify_failed")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "invalid_or_expired_token", "message": "Login link is invalid or expired."},
        ) from None
    _set_session_cookie(response, session._raw_session_token)  # type: ignore[attr-defined]
    return {"status": "ok", "csrf_token": session.csrf_token}


@router.post("/auth/logout")
def logout(
    response: Response,
    user: AppUser = Depends(require_csrf),
    session_token: str | None = Cookie(default=None, alias=SESSION_COOKIE),
    auth: AuthService = Depends(get_auth_service),
):
    revoked = auth.revoke_session(session_token or "")
    response.delete_cookie(key=SESSION_COOKIE, path="/")
    logger.info("session_logout", revoked=revoked)
    return {"status": "logged_out", "revoked": revoked}


@router.get("/me")
def me(user: AppUser = Depends(get_current_user)):
    return {"id": user.id, "email": user.email}
