"""FastAPI auth dependencies."""

from __future__ import annotations

import secrets

from fastapi import Cookie, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from src.app.auth.service import AuthService
from src.app.logging import get_logger
from src.app.persistence.database import SessionLocal
from src.app.persistence.models import AppUser

logger = get_logger(__name__)

UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_auth_service(db: Session = Depends(get_db)) -> AuthService:
    return AuthService(db)


def get_current_user(
    request: Request,
    session_token: str | None = Cookie(default=None, alias="session"),
    db: Session = Depends(get_db),
) -> AppUser:
    if not session_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    user = AuthService(db).get_user_for_session(session_token)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session")
    return user


def require_csrf(
    request: Request,
    user: AppUser = Depends(get_current_user),
    session_token: str | None = Cookie(default=None, alias="session"),
    db: Session = Depends(get_db),
) -> AppUser:
    """Double-submit CSRF check bound to the caller's server-side session.

    The header must match the ``csrf_token`` stored on the session row, so a
    cross-site form post cannot succeed just by inventing a header value.
    """
    if request.method not in UNSAFE_METHODS:
        return user
    header = request.headers.get("X-CSRF-Token")
    if not header:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Missing CSRF token")
    record = AuthService(db).get_session_record(session_token or "")
    if record is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid CSRF token")
    if not secrets.compare_digest(header.encode("utf-8"), (record.csrf_token or "").encode("utf-8")):
        logger.warning("csrf_mismatch", path=request.url.path, method=request.method)
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid CSRF token")
    return user


def require_idempotency_key(request: Request) -> str:
    key = request.headers.get("Idempotency-Key")
    if not key:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing Idempotency-Key header")
    return key
