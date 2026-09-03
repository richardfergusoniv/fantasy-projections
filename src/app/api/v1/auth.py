"""Authentication endpoints."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, EmailStr, Field

from src.app.api.deps import get_auth_service, get_current_user, require_csrf
from src.app.auth.service import SESSION_MAX_AGE_SECONDS, AuthService
from src.app.config import get_settings
from src.app.logging import get_logger
from src.app.middleware.rate_limit import client_key, limiter
from src.app.persistence.models import AppUser

logger = get_logger(__name__)

router = APIRouter()

SESSION_COOKIE = "session"
SESSION_COOKIE_SAMESITE = "lax"


class MagicLinkRequest(BaseModel):
    email: EmailStr


class MagicLinkVerify(BaseModel):
    token: str = Field(min_length=16, max_length=512)


def _cookie_secure() -> bool:
    return get_settings().session_cookie_secure


def _set_session_cookie(response: Response, raw_token: str) -> None:
    """Persist the session on iOS standalone PWAs.

    iOS treats cookies that only set Max-Age (no Expires) as session cookies and
    drops them when the Home Screen app is fully closed. Both attributes are
    required for a long-lived private beta session.
    """
    expires = datetime.now(UTC) + timedelta(seconds=SESSION_MAX_AGE_SECONDS)
    response.set_cookie(
        key=SESSION_COOKIE,
        value=raw_token,
        httponly=True,
        samesite=SESSION_COOKIE_SAMESITE,
        secure=_cookie_secure(),
        max_age=SESSION_MAX_AGE_SECONDS,
        expires=expires,
        path="/",
    )


def _clear_session_cookie(response: Response) -> None:
    response.delete_cookie(
        key=SESSION_COOKIE,
        path="/",
        secure=_cookie_secure(),
        httponly=True,
        samesite=SESSION_COOKIE_SAMESITE,
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
    _clear_session_cookie(response)
    logger.info("session_logout", revoked=revoked)
    return {"status": "logged_out", "revoked": revoked}


@router.get("/me")
def me(
    user: AppUser = Depends(get_current_user),
    session_token: str | None = Cookie(default=None, alias=SESSION_COOKIE),
    auth: AuthService = Depends(get_auth_service),
):
    """Return the signed-in user plus CSRF so a PWA reopen can restore mutations.

    The session cookie is HTTP-only. The matching CSRF token used to live only
    in sessionStorage, which iOS deletes when the installed PWA is killed.
    Echoing it here lets the client rebuild the double-submit header from the
    still-valid cookie without prompting for another magic link.
    """
    record = auth.get_session_record(session_token or "")
    payload: dict[str, object] = {"id": user.id, "email": user.email}
    if record is not None:
        payload["csrf_token"] = record.csrf_token
        expires = record.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=UTC)
        payload["session_expires_at"] = expires.isoformat()
    return payload
