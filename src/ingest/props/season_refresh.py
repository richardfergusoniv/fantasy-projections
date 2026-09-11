"""Season-long Vegas O/U refresh from live DraftKings + FanDuel boards.

Daily market-close jobs write closing-line snapshots and optionally refresh the
Draft Assistant vegas_raw inputs. Sealed ``vegas_consensus_{season}.json`` is
only overwritten when explicitly enabled — default is a shadow consensus under
``data/props/season_consensus/``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SNAPSHOT_ROOT = REPO_ROOT / "data" / "props" / "snapshots" / "season_ou"
DEFAULT_SHADOW_CONSENSUS_ROOT = REPO_ROOT / "data" / "props" / "season_consensus"
DRAFT_VEGAS_RAW = REPO_ROOT / "draft_assistant" / "data" / "vegas_raw"


@dataclass(frozen=True)
class SeasonRefreshResult:
    season: int
    written: tuple[str, ...]
    success_count: int
    failure_count: int
    consensus_path: str | None
    updated_vegas_raw: tuple[str, ...]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "season": self.season,
            "written": list(self.written),
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "consensus_path": self.consensus_path,
            "updated_vegas_raw": list(self.updated_vegas_raw),
            "errors": list(self.errors),
        }


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def refresh_season_ou(
    *,
    season: int,
    providers: tuple[str, ...] = ("draftkings", "fanduel"),
    snapshot_root: Path | None = None,
    update_vegas_raw: bool = True,
    rebuild_consensus: bool = True,
    write_sealed_consensus: bool = False,
    now: datetime | None = None,
) -> SeasonRefreshResult:
    clock = now or datetime.now(timezone.utc)
    stamp = clock.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = snapshot_root or DEFAULT_SNAPSHOT_ROOT
    written: list[str] = []
    updated_raw: list[str] = []
    errors: list[str] = []
    success = 0

    payloads: dict[str, dict[str, Any]] = {}
    for name in providers:
        try:
            if name == "draftkings":
                from src.ingest.props.providers.draftkings_live import fetch_season_raw

                payload = fetch_season_raw(season=season, now=clock)
            elif name == "fanduel":
                from src.ingest.props.providers.fanduel_live import fetch_season_raw

                payload = fetch_season_raw(season=season, now=clock)
            else:
                errors.append(f"{name}: unsupported_season_provider")
                continue
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc}")
            continue
        players = payload.get("players") or []
        if not players:
            errors.append(f"{name}: no_players")
            continue
        success += 1
        payloads[name] = payload
        path = _write_json(root / f"source={name}" / f"fetch={stamp}.json", payload)
        written.append(str(path))
        # Also keep a stable "latest" pointer for ops.
        latest = _write_json(root / f"source={name}" / "latest.json", payload)
        written.append(str(latest))
        if update_vegas_raw:
            raw_path = _write_json(DRAFT_VEGAS_RAW / f"{name}.json", payload)
            updated_raw.append(str(raw_path))

    consensus_path: str | None = None
    if rebuild_consensus and success > 0:
        try:
            from src.draft_assistant.vegas_consensus import export_consensus

            if write_sealed_consensus:
                out = export_consensus(season=season)
            else:
                out = export_consensus(
                    season=season,
                    out_path=DEFAULT_SHADOW_CONSENSUS_ROOT
                    / f"vegas_consensus_{season}.json",
                )
            consensus_path = str(out)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"consensus: {exc}")

    return SeasonRefreshResult(
        season=season,
        written=tuple(written),
        success_count=success,
        failure_count=len(providers) - success,
        consensus_path=consensus_path,
        updated_vegas_raw=tuple(updated_raw),
        errors=tuple(errors),
    )
