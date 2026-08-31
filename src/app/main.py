"""Uvicorn entrypoint."""

from src.app.factory import create_app

app = create_app()
