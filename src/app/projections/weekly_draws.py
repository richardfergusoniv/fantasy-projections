"""Deterministic weekly stat-level draw partitions for trained inference.

Legacy mode (schema v1)
-----------------------
``generate_player_stat_draws`` samples an independent split-normal fantasy-point
value per player and proportionally scales component stats. That path is retained
explicitly as ``legacy_scaled_components`` and must not be described as joint or
exact-league when conservation/PPFD gates have not passed.

Joint mixture mode (schema v2)
------------------------------
``write_joint_weekly_draw_partition`` builds game-level correlated draws via
``src.projection.weekly.draws.game_engine`` and persists a versioned joint
partition. Decision loaders must consume aligned simulation indices from that
artifact rather than reseeding per request.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl

from src.app.decisions.draws import stable_seed
from src.app.projections.weekly_stat_draw import weekly_row_to_stat_draw
from src.projection.weekly.draws.contracts import DrawModeLabel
from src.projection.weekly.draws.game_engine import (
    build_game_input_from_projection_rows,
    generate_game_draws,
)
from src.projection.weekly.draws.partition_schema import (
    JOINT_PARTITION_SCHEMA_VERSION,
    JointPartitionManifest,
    write_joint_partition,
)

SKILL_POSITIONS = frozenset({"QB", "RB", "WR", "TE"})
PARTITION_SCHEMA_VERSION = 1  # legacy independent scaled-component partitions
DrawPath = Literal["legacy_scaled_components", "joint_stat_mixture_candidate"]


@dataclass(frozen=True)
class WeeklyDrawPartition:
    path: Path
    sha256: str
    draw_count: int
    player_count: int
    seed_salt: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "draw_count": self.draw_count,
            "player_count": self.player_count,
            "seed_salt": self.seed_salt,
        }


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def build_partition_manifest(
    frame: pl.DataFrame,
    *,
    draw_count: int,
    seed_salt: str,
) -> dict[str, Any]:
    """Build a compact manifest that deterministically reconstructs stat draws."""
    players: list[dict[str, Any]] = []
    for record in frame.iter_rows(named=True):
        position = str(record.get("position") or "")
        if position not in SKILL_POSITIONS:
            continue
        player_id = str(record.get("gsis_id") or record.get("player_id") or "")
        if not player_id:
            continue
        fp = float(record.get("fantasy_points") or 0.0)
        floor = float(record.get("floor") or max(0.0, fp * 0.7))
        ceiling = float(record.get("ceiling") or max(fp, fp * 1.3))
        players.append(
            {
                "player_id": player_id,
                "position": position,
                "team": record.get("team"),
                "fantasy_points": fp,
                "floor": floor,
                "ceiling": ceiling,
                "components": weekly_row_to_stat_draw(record),
            }
        )
    return {
        "schema_version": PARTITION_SCHEMA_VERSION,
        "draw_count": int(draw_count),
        "seed_salt": seed_salt,
        "player_count": len(players),
        "players": players,
    }


def generate_player_stat_draws(
    player: dict[str, Any],
    *,
    draw_count: int,
    seed_salt: str,
) -> list[dict[str, float]]:
    """Reconstruct component-stat draws for one player from a partition entry."""
    components = dict(player.get("components") or {})
    if not components:
        return []

    fp = float(player.get("fantasy_points") or 0.0)
    floor = float(player.get("floor") or max(0.0, fp * 0.7))
    ceiling = float(player.get("ceiling") or max(fp, fp * 1.3))
    p10 = max(0.0, floor)
    p50 = max(0.0, fp)
    p90 = max(p50, ceiling)

    rng = np.random.default_rng(stable_seed(player["player_id"], seed_salt, "weekly_stat_draws"))
    z90 = 1.2815515655446004
    lower_scale = max((p50 - p10) / z90, 1e-9)
    upper_scale = max((p90 - p50) / z90, 1e-9)
    z = rng.standard_normal(draw_count)
    fp_draws = np.where(z < 0, p50 + z * lower_scale, p50 + z * upper_scale)
    fp_draws = np.maximum(fp_draws, 0.0)

    comp_sum = sum(max(0.0, float(v)) for v in components.values()) or 1.0
    draws: list[dict[str, float]] = []
    for fp_draw in fp_draws:
        scale = float(fp_draw) / comp_sum if comp_sum > 0 else 0.0
        draws.append({stat: max(0.0, float(value) * scale) for stat, value in components.items()})
    return draws


def write_weekly_draw_partition(
    frame: pl.DataFrame,
    output_dir: Path,
    *,
    draw_count: int,
    seed_salt: str,
) -> WeeklyDrawPartition:
    """Persist a weekly stat-draw partition manifest beside weekly output."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_partition_manifest(frame, draw_count=draw_count, seed_salt=seed_salt)
    path = output_dir / "stat_draw_partition.json"
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path.write_bytes(payload)
    return WeeklyDrawPartition(
        path=path,
        sha256=_sha256_bytes(payload),
        draw_count=draw_count,
        player_count=int(manifest["player_count"]),
        seed_salt=seed_salt,
    )


def load_weekly_draw_partition(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def weekly_draw_partition_from_file(path: Path, *, seed_salt: str) -> WeeklyDrawPartition:
    payload = load_weekly_draw_partition(path)
    content = path.read_bytes()
    return WeeklyDrawPartition(
        path=path,
        sha256=_sha256_bytes(content),
        draw_count=int(payload.get("draw_count") or 0),
        player_count=int(payload.get("player_count") or 0),
        seed_salt=seed_salt,
    )


def verify_weekly_draw_partition(path: Path, *, expected_sha256: str | None = None) -> bool:
    if not path.exists():
        return False
    digest = _sha256_bytes(path.read_bytes())
    if expected_sha256 and digest != expected_sha256:
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return int(payload.get("schema_version", 0)) == PARTITION_SCHEMA_VERSION


def write_joint_weekly_draw_partition(
    frame: pl.DataFrame,
    output_dir: Path,
    *,
    draw_count: int,
    seed_salt: str,
    season: int,
    week: int,
    as_of_cutoff: str = "",
    model_hash: str = "",
    feature_hash: str = "",
    evaluation_hash: str = "",
    draw_mode: str = DrawModeLabel.JOINT_STAT_MIXTURE_CANDIDATE.value,
) -> tuple[Path, str, JointPartitionManifest]:
    """Persist a schema-v2 joint game partition (does not overwrite legacy v1)."""
    records = list(frame.iter_rows(named=True))
    # Group into games via game_id when present; otherwise synthesize one game per matchup pair.
    games_meta: dict[str, dict[str, Any]] = {}
    for row in records:
        gid = str(row.get("game_id") or "")
        team = str(row.get("team") or "")
        opp = str(row.get("opponent") or row.get("opponent_team") or "")
        if not gid and team and opp:
            gid = f"synth-{season}-w{week}-{'-'.join(sorted([team, opp]))}"
        if not gid:
            continue
        meta = games_meta.setdefault(gid, {"teams": set(), "rows": []})
        meta["teams"].add(team)
        if opp:
            meta["teams"].add(opp)
        meta["rows"].append(row)

    game_payloads: list[dict[str, Any]] = []
    global_seed = stable_seed(seed_salt, season, week, "joint_weekly")
    for offset, (gid, meta) in enumerate(sorted(games_meta.items())):
        teams = sorted(t for t in meta["teams"] if t)
        if len(teams) < 2:
            # Bye / incomplete — skip rather than invent a failed participation event.
            continue
        home_team, away_team = teams[0], teams[1]
        game_input = build_game_input_from_projection_rows(
            meta["rows"],
            game_id=gid,
            season=season,
            week=week,
            home_team=home_team,
            away_team=away_team,
        )
        game_payloads.append(
            generate_game_draws(
                game_input,
                draw_count=draw_count,
                seed=int(global_seed + offset),
            )
        )

    player_count = sum(
        len(team.get("players") or [])
        for game in game_payloads
        for team in game.get("teams") or []
    )
    manifest = JointPartitionManifest(
        schema_version=JOINT_PARTITION_SCHEMA_VERSION,
        season=season,
        week=week,
        as_of_cutoff=as_of_cutoff,
        draw_count=draw_count,
        global_seed=int(global_seed),
        seed_salt=seed_salt,
        partition_id=f"joint-{season}-w{week}-{seed_salt}",
        model_hash=model_hash,
        feature_hash=feature_hash,
        evaluation_hash=evaluation_hash,
        contract_version="weekly_mixture_contract_v1",
        draw_mode=draw_mode,
        scoring_fidelity="exact_joint",
        ppfd_ready=True,
        kicker_ready=True,
        dst_ready=True,
        conservation_ok=False,
        probabilistic_gates_ok=False,
        games=game_payloads,
        notes=[
            "Candidate joint partition; validated label requires conservation + probabilistic gates.",
            "Legacy v1 scaled-component partitions remain available separately.",
        ],
    )
    path, digest = write_joint_partition(manifest, output_dir)
    manifest.manifest_hash = digest
    return path, digest, manifest
