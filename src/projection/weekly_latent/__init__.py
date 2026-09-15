"""Research/shadow weekly latent model — Milestone 1."""

from __future__ import annotations

from typing import Any

__all__ = ["run_milestone1"]


def __getattr__(name: str) -> Any:
    if name == "run_milestone1":
        from src.projection.weekly_latent.run import run_milestone1

        return run_milestone1
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
