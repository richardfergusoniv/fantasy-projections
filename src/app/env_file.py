#!/usr/bin/env python3
"""Load production settings from an env file for infrastructure audits."""
from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import PydanticBaseSettingsSource

from src.app.config import Settings, get_settings


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def _field_kwargs(values: dict[str, str]) -> dict[str, str]:
    return {key.lower(): value for key, value in values.items()}


def settings_from_env_file(path: Path) -> Settings:
    """Build production Settings from a dotenv file without process env or .env bleed."""
    values = _field_kwargs(load_env_file(path))
    values["app_env"] = "production"
    values["app_enable_dev_auth"] = "false"

    class AuditSettings(Settings):
        @classmethod
        def settings_customise_sources(
            cls,
            settings_cls: type[Settings],
            init_settings: PydanticBaseSettingsSource,
            env_settings: PydanticBaseSettingsSource,
            dotenv_settings: PydanticBaseSettingsSource,
            file_secret_settings: PydanticBaseSettingsSource,
        ) -> tuple[PydanticBaseSettingsSource, ...]:
            return (init_settings,)

    return AuditSettings.model_validate(values)


def apply_env_file(path: Path) -> None:
    """Apply dotenv values to os.environ and clear settings cache."""
    for key, value in load_env_file(path).items():
        os.environ[key] = value
    get_settings.cache_clear()
