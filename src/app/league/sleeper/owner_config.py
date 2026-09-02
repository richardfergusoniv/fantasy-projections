"""Owner-confirmed Sleeper league selection and dynasty draft-order rules.

League names are display metadata only; ``league_id`` is the stable key.
Owner-specific files belong outside git — see ``config/sleeper_owner.example.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

KNOWN_ROOKIE_PICK_RULES = frozenset({"max_pf", "reverse_standings"})
LeagueType = Literal["redraft", "dynasty"]


class SleeperLeagueEntry(BaseModel):
    league_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    league_type: LeagueType
    rookie_pick_rule: str | None = None

    @field_validator("league_id")
    @classmethod
    def _strip_league_id(cls, value: str) -> str:
        return value.strip()

    @field_validator("rookie_pick_rule")
    @classmethod
    def _normalize_rule(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if not normalized:
            return None
        return normalized

    @model_validator(mode="after")
    def _validate_rule_for_type(self) -> SleeperLeagueEntry:
        if self.league_type == "redraft" and self.rookie_pick_rule is not None:
            raise ValueError(
                f"league {self.league_id}: rookie_pick_rule is not allowed on redraft leagues"
            )
        if self.league_type == "dynasty" and self.rookie_pick_rule is None:
            raise ValueError(
                f"league {self.league_id}: dynasty leagues require rookie_pick_rule "
                f"({', '.join(sorted(KNOWN_ROOKIE_PICK_RULES))})"
            )
        if self.rookie_pick_rule is not None and self.rookie_pick_rule not in KNOWN_ROOKIE_PICK_RULES:
            raise ValueError(
                f"league {self.league_id}: unknown rookie_pick_rule {self.rookie_pick_rule!r}"
            )
        return self


class SleeperOwnerConfig(BaseModel):
    schema_version: int = 1
    username: str = Field(min_length=1)
    season: int = Field(ge=2000, le=2100)
    leagues: list[SleeperLeagueEntry] = Field(min_length=1)

    @field_validator("username")
    @classmethod
    def _strip_username(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def _validate_unique_leagues(self) -> SleeperOwnerConfig:
        seen: set[str] = set()
        for entry in self.leagues:
            if entry.league_id in seen:
                raise ValueError(f"duplicate league_id: {entry.league_id}")
            seen.add(entry.league_id)
        return self

    @property
    def allowed_league_ids(self) -> frozenset[str]:
        return frozenset(entry.league_id for entry in self.leagues)

    def entry_for(self, league_id: str) -> SleeperLeagueEntry | None:
        for entry in self.leagues:
            if entry.league_id == league_id:
                return entry
        return None

    def log_summary(self) -> list[dict[str, str]]:
        """Privacy-safe configuration summary for reports and logs."""
        return [
            {
                "league_id": entry.league_id,
                "display_name": entry.display_name,
                "league_type": entry.league_type,
                "rookie_pick_rule": entry.rookie_pick_rule or "",
            }
            for entry in self.leagues
        ]


class LeagueSelectionError(ValueError):
    """Configured leagues do not match Sleeper discovery."""


def load_owner_config(path: str | Path | None = None) -> SleeperOwnerConfig:
    from src.app.config import get_settings

    settings = get_settings()
    if settings.sleeper_owner_json:
        payload = json.loads(settings.sleeper_owner_json)
        return SleeperOwnerConfig.model_validate(payload)
    if path is None:
        path = settings.sleeper_owner_config
    if not path:
        raise FileNotFoundError("owner config not configured")
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"owner config not found: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return SleeperOwnerConfig.model_validate(payload)


def sleeper_league_type(league_data: dict[str, Any]) -> LeagueType:
    """Derive dynasty vs redraft from Sleeper league settings."""
    settings = league_data.get("settings") or {}
    return "dynasty" if int(settings.get("type", 0) or 0) == 2 else "redraft"


def validate_league_selection(
    config: SleeperOwnerConfig,
    discovered: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare owner config against Sleeper's league list for the season.

  Raises :class:`LeagueSelectionError` when configured leagues are missing.
  Extra Sleeper leagues are reported and must not be imported.
    """
    discovered_by_id = {
        str(row["league_id"]): row for row in discovered if row.get("league_id")
    }
    configured_ids = config.allowed_league_ids
    discovered_ids = frozenset(discovered_by_id)
    missing = sorted(configured_ids - discovered_ids)
    extra = sorted(discovered_ids - configured_ids)
    if missing:
        raise LeagueSelectionError(
            f"configured leagues not returned by Sleeper: {', '.join(missing)}"
        )

    conflicts: list[str] = []
    for entry in config.leagues:
        payload = discovered_by_id.get(entry.league_id)
        if payload is None:
            continue
        actual_type = sleeper_league_type(payload)
        if actual_type != entry.league_type:
            conflicts.append(
                f"{entry.league_id}: configured={entry.league_type}, sleeper={actual_type}"
            )

    if conflicts:
        raise LeagueSelectionError(
            "league type conflicts between owner config and Sleeper: "
            + "; ".join(conflicts)
        )

    return {
        "configured_count": len(configured_ids),
        "discovered_count": len(discovered_ids),
        "imported_count": len(configured_ids),
        "extra_league_ids": extra,
        "extra_leagues_ignored": len(extra),
        "leagues": config.log_summary(),
    }
