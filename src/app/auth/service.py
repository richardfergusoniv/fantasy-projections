"""Email magic-link authentication."""

from __future__ import annotations

import hashlib
import secrets
from abc import ABC, abstractmethod
from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session

from src.app.config import Settings, get_settings
from src.app.logging import get_logger
from src.app.persistence.models import AppUser, MagicLinkToken, SessionRecord

logger = get_logger(__name__)

RESEND_ENDPOINT = "https://api.resend.com/emails"

#: Long-lived owner session. Cookie Max-Age/Expires and the server row must
#: share this TTL so iOS cannot drop the cookie while the row is still valid
#: (or keep a cookie after the row has expired).
SESSION_TTL = timedelta(days=30)
SESSION_MAX_AGE_SECONDS = int(SESSION_TTL.total_seconds())
MAGIC_LINK_TTL = timedelta(minutes=15)

#: Dialects that support ``SELECT ... FOR UPDATE``. SQLite does not.
ROW_LOCK_DIALECTS = frozenset({"postgresql", "mysql", "mariadb", "oracle", "mssql"})


class EmailProviderConfigError(RuntimeError):
    """Raised when the selected email provider is missing required settings."""


class EmailProvider(ABC):
    @abstractmethod
    def send_magic_link(self, email: str, link: str) -> str | None:
        """Return dev-visible link when applicable."""


class DevelopmentEmailProvider(EmailProvider):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def send_magic_link(self, email: str, link: str) -> str | None:
        if self.settings.app_enable_dev_auth:
            return link
        raise RuntimeError("Development auth is disabled")


class SmtpEmailProvider(EmailProvider):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def send_magic_link(self, email: str, link: str) -> str | None:
        import smtplib
        from email.message import EmailMessage

        msg = EmailMessage()
        msg["Subject"] = "Fantasy app login link"
        msg["From"] = self.settings.email_from
        msg["To"] = email
        msg.set_content(f"Sign in: {link}")
        with smtplib.SMTP(self.settings.smtp_host, self.settings.smtp_port) as smtp:
            if self.settings.smtp_user and self.settings.smtp_password:
                smtp.starttls()
                smtp.login(self.settings.smtp_user, self.settings.smtp_password)
            smtp.send_message(msg)
        return None


class ResendEmailProvider(EmailProvider):
    """Deliver magic links through the Resend HTTP API.

    Constructed eagerly by :func:`get_email_provider` so a misconfigured
    provider fails at startup rather than silently swallowing login emails.
    """

    def __init__(self, settings: Settings) -> None:
        if not settings.resend_api_key:
            raise EmailProviderConfigError(
                "EMAIL_PROVIDER='resend' requires RESEND_API_KEY to be set"
            )
        self.settings = settings

    def send_magic_link(self, email: str, link: str) -> str | None:
        import httpx

        response = httpx.post(
            RESEND_ENDPOINT,
            headers={
                "Authorization": f"Bearer {self.settings.resend_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "from": self.settings.email_from,
                "to": [email],
                "subject": "Fantasy app login link",
                "text": f"Sign in: {link}",
            },
            timeout=10.0,
        )
        response.raise_for_status()
        return None


def get_email_provider(settings: Settings | None = None) -> EmailProvider:
    settings = settings or get_settings()
    if settings.email_provider == "development":
        return DevelopmentEmailProvider(settings)
    if settings.email_provider == "smtp":
        return SmtpEmailProvider(settings)
    if settings.email_provider == "resend":
        return ResendEmailProvider(settings)
    raise EmailProviderConfigError(f"Unsupported EMAIL_PROVIDER: {settings.email_provider!r}")


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


class AuthService:
    def __init__(self, session: Session, settings: Settings | None = None) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.email_provider = get_email_provider(self.settings)

    def _is_allowed_email(self, email: str | None) -> bool:
        allowed = (self.settings.app_allowed_email or "").strip().lower()
        candidate = (email or "").strip().lower()
        if not allowed or not candidate:
            return False
        return secrets.compare_digest(candidate.encode("utf-8"), allowed.encode("utf-8"))

    def _supports_row_locks(self) -> bool:
        try:
            return self.session.get_bind().dialect.name in ROW_LOCK_DIALECTS
        except Exception:  # noqa: BLE001 - unbound session in unit tests
            return False

    def request_magic_link(self, email: str) -> dict:
        # The response shape is identical for allowlisted and non-allowlisted
        # addresses so the endpoint cannot be used to enumerate the owner.
        if not self._is_allowed_email(email):
            logger.info("magic_link_rejected", reason="not_allowlisted")
            return {"status": "sent"}
        token = secrets.token_urlsafe(32)
        record = MagicLinkToken(
            email=email.lower(),
            token_hash=_hash_token(token),
            expires_at=datetime.now(UTC) + MAGIC_LINK_TTL,
        )
        self.session.add(record)
        self.session.flush()
        # Hash fragment keeps the token out of the initial GET so mail scanners
        # cannot consume a one-time link before the owner opens it in a browser.
        public_url = self.settings.effective_app_public_url.rstrip("/")
        if public_url.rstrip("/") != (self.settings.app_public_url or "").rstrip("/"):
            logger.warning(
                "magic_link_public_url_remapped",
                configured=self.settings.app_public_url,
                effective=public_url,
            )
        link = f"{public_url}/auth/callback#token={token}"
        dev_link = self.email_provider.send_magic_link(email, link)
        payload = {"status": "sent"}
        if dev_link:
            payload["development_link"] = dev_link
        return payload

    def verify_magic_link(self, token: str) -> SessionRecord:
        token_hash = _hash_token(token)
        query = self.session.query(MagicLinkToken).filter(MagicLinkToken.token_hash == token_hash)
        if self._supports_row_locks():
            query = query.with_for_update()
        record = query.one_or_none()
        if record is None or record.used_at is not None:
            raise ValueError("Invalid token")
        if _as_utc(record.expires_at) < datetime.now(UTC):
            raise ValueError("Expired token")
        # Re-check the allowlist at verification time: the owner address may
        # have been rotated after this token was minted.
        if not self._is_allowed_email(record.email):
            logger.warning("magic_link_email_no_longer_allowlisted")
            raise ValueError("Invalid token")

        # Compare-and-swap on used_at so two concurrent verifications of the
        # same token can never both mint a session.
        claimed = (
            self.session.query(MagicLinkToken)
            .filter(MagicLinkToken.id == record.id, MagicLinkToken.used_at.is_(None))
            .update({MagicLinkToken.used_at: datetime.now(UTC)}, synchronize_session="fetch")
        )
        if claimed != 1:
            raise ValueError("Invalid token")

        user = self.session.query(AppUser).filter(AppUser.email == record.email).one_or_none()
        if user is None:
            user = AppUser(email=record.email)
            self.session.add(user)
            self.session.flush()
        session_token = secrets.token_urlsafe(32)
        session_record = SessionRecord(
            user_id=user.id,
            session_hash=_hash_token(session_token),
            csrf_token=secrets.token_urlsafe(16),
            expires_at=datetime.now(UTC) + SESSION_TTL,
        )
        self.session.add(session_record)
        self.session.flush()
        session_record._raw_session_token = session_token  # type: ignore[attr-defined]
        return session_record

    def get_session_record(self, session_token: str) -> SessionRecord | None:
        """Return the live session row for a raw cookie value, or None.

        Expired rows are deleted rather than merely ignored so a stale cookie
        can never be resurrected by a clock change or a restored backup.
        """
        if not session_token:
            return None
        record = (
            self.session.query(SessionRecord)
            .filter(SessionRecord.session_hash == _hash_token(session_token))
            .one_or_none()
        )
        if record is None:
            return None
        if _as_utc(record.expires_at) < datetime.now(UTC):
            self.session.delete(record)
            self.session.flush()
            return None
        return record

    def get_user_for_session(self, session_token: str) -> AppUser | None:
        record = self.get_session_record(session_token)
        if record is None:
            return None
        user = self.session.query(AppUser).filter(AppUser.id == record.user_id).one_or_none()
        if user is None:
            # Orphaned session: the backing user row is gone.
            self.session.delete(record)
            self.session.flush()
            return None
        if not self._is_allowed_email(user.email):
            logger.warning("session_user_no_longer_allowlisted")
            return None
        return user

    def revoke_session(self, session_token: str) -> bool:
        """Delete the server-side session row. Returns True when one was removed.

        SessionRecord has no revocation column, so removal of the row *is* the
        revocation; there is no server-side state left to accept the cookie.
        """
        if not session_token:
            return False
        record = (
            self.session.query(SessionRecord)
            .filter(SessionRecord.session_hash == _hash_token(session_token))
            .one_or_none()
        )
        if record is None:
            return False
        self.session.delete(record)
        self.session.flush()
        return True
