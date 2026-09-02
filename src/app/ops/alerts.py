"""Operational alerting for missed jobs and readiness failures."""

from __future__ import annotations

import smtplib
from email.message import EmailMessage

import httpx

from src.app.auth.service import RESEND_ENDPOINT, get_email_provider
from src.app.config import get_settings
from src.app.logging import get_logger

logger = get_logger(__name__)


def send_ops_alert(subject: str, body: str) -> bool:
    """Best-effort email alert when ALERT_EMAIL_TO is configured."""
    settings = get_settings()
    recipient = settings.alert_email_to
    if not recipient:
        logger.warning("ops_alert_skipped", subject=subject, reason="alert_email_to_unset")
        return False

    if settings.email_provider == "development":
        get_email_provider(settings)
        logger.warning("ops_alert", subject=subject, body=body, recipient=recipient)
        return True

    try:
        if settings.email_provider == "smtp":
            msg = EmailMessage()
            msg["Subject"] = subject
            msg["From"] = settings.email_from
            msg["To"] = recipient
            msg.set_content(body)
            with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as smtp:
                if settings.smtp_user and settings.smtp_password:
                    smtp.starttls()
                    smtp.login(settings.smtp_user, settings.smtp_password)
                smtp.send_message(msg)
            return True

        if settings.email_provider == "resend":
            response = httpx.post(
                RESEND_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {settings.resend_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "from": settings.email_from,
                    "to": [recipient],
                    "subject": subject,
                    "text": body,
                },
                timeout=10.0,
            )
            response.raise_for_status()
            return True
    except Exception:  # noqa: BLE001
        logger.exception("ops_alert_failed", subject=subject)
        return False

    logger.warning("ops_alert_unsupported_provider", provider=settings.email_provider)
    return False
