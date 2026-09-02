"""Versioned joint weekly draw partition schema (schema_version >= 2)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from src.projection.weekly.draws.contracts import DrawModeLabel

JOINT_PARTITION_SCHEMA_VERSION = 2

ScoringFidelity = Literal[
    "exact_joint",
    "legacy_scaled_components",
    "legacy_points_independent",
    "mixed_fallback",
]


@dataclass
class JointPartitionManifest:
    schema_version: int = JOINT_PARTITION_SCHEMA_VERSION
    season: int = 0
    week: int = 0
    as_of_cutoff: str = ""
    draw_count: int = 0
    global_seed: int = 0
    seed_salt: str = ""
    partition_id: str = ""
    model_hash: str = ""
    manifest_hash: str = ""
    feature_hash: str = ""
    evaluation_hash: str = ""
    contract_version: str = ""
    draw_mode: str = DrawModeLabel.JOINT_STAT_MIXTURE_CANDIDATE.value
    scoring_fidelity: ScoringFidelity = "exact_joint"
    ppfd_ready: bool = False
    kicker_ready: bool = False
    dst_ready: bool = False
    conservation_ok: bool = False
    probabilistic_gates_ok: bool = False
    games: list[dict[str, Any]] = field(default_factory=list)
    unavailable_rules: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def partition_content_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def write_joint_partition(
    manifest: JointPartitionManifest,
    output_dir: Path,
    *,
    filename: str = "joint_stat_partition.json",
) -> tuple[Path, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    payload = manifest.to_dict()
    digest = partition_content_hash(payload)
    payload["partition_hash"] = digest
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path, digest


def load_joint_partition(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_joint_partition(
    path: Path,
    *,
    expected_hash: str | None = None,
) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    payload = load_joint_partition(path)
    if int(payload.get("schema_version") or 0) != JOINT_PARTITION_SCHEMA_VERSION:
        return False, "schema_mismatch"
    stored = payload.get("partition_hash")
    check = dict(payload)
    check.pop("partition_hash", None)
    digest = partition_content_hash(check)
    if stored and stored != digest:
        return False, "hash_mismatch"
    if expected_hash and digest != expected_hash:
        return False, "expected_hash_mismatch"
    if not payload.get("games"):
        return False, "empty_games"
    return True, digest


def aligned_player_draws_by_index(
    partition: dict[str, Any],
) -> dict[str, list[dict[str, float]]]:
    """Load player component draws aligned on the shared simulation index."""
    out: dict[str, list[dict[str, float]]] = {}
    for game in partition.get("games") or []:
        for team in game.get("teams") or []:
            for player in team.get("players") or []:
                pid = str(player.get("player_id") or "")
                if not pid:
                    continue
                out[pid] = list(player.get("draws") or [])
    return out


def detect_partial_or_corrupt(partition: dict[str, Any]) -> list[str]:
    """Failure-injection helpers: reasons a partition must not advance pointers."""
    reasons: list[str] = []
    if int(partition.get("schema_version") or 0) != JOINT_PARTITION_SCHEMA_VERSION:
        reasons.append("schema")
    draw_count = int(partition.get("draw_count") or 0)
    if draw_count <= 0:
        reasons.append("draw_count")
    games = partition.get("games") or []
    if not games:
        reasons.append("no_games")
    for game in games:
        for team in game.get("teams") or []:
            for player in team.get("players") or []:
                draws = player.get("draws") or []
                if len(draws) != draw_count:
                    reasons.append(f"mismatched_draws:{player.get('player_id')}")
    if partition.get("draw_mode") == DrawModeLabel.JOINT_STAT_MIXTURE_VALIDATED.value:
        if not partition.get("conservation_ok") or not partition.get("probabilistic_gates_ok"):
            reasons.append("validated_without_gates")
    return reasons
