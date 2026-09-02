"""Vercel ASGI entrypoint — re-exports the FastAPI app."""

from src.app.main import app  # noqa: F401
