"""Research/shadow weekly latent model — Milestones 1–2."""

from __future__ import annotations

from typing import Any

__all__ = ["run_milestone1", "run_milestone2"]


def __getattr__(name: str) -> Any:
    if name == "run_milestone1":
        from src.projection.weekly_latent.run import run_milestone1

        return run_milestone1
    if name == "run_milestone2":
        from src.projection.weekly_latent.run import run_milestone2

        return run_milestone2
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
