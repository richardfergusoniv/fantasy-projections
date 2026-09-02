"""Load player projections from sealed release bundles."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.projection.active_release import (
    ActiveReleaseError,
    frozen_load_session,
    read_active_pointer,
    validate_active_pointer,
)
from src.app.config import get_settings
from src.app.storage.release_bundle import ReleaseBundleResolver
from src.projection.contracts import MODEL_V3_DIR, REPO_ROOT
from src.projection.release_bundle import (
    MANIFEST_FILENAME,
    ReleaseBundleError,
    load_sealed_manifest,
    public_release_dir,
    sha256_file,
    verify_provenance_identities,
)


class ReleaseBundleLoadError(RuntimeError):
    """Pointer, manifest, or artifact validation failed."""


@dataclass(frozen=True)
class PlayerSummary:
    player_id: str
    name: str
    position: str
    team: str | None
    mean_points: float
    quantiles: dict[str, float]
    availability_probability: float = 1.0


@dataclass
class ReleaseBundleSnapshot:
    """Immutable loaded view of one sealed release bundle."""

    season: int
    namespace: str
    release_id: str
    manifest_sha256: str
    manifest_path: Path
    bundle_root: Path
    players: dict[str, PlayerSummary]
    meta: dict[str, Any]
    artifact_hashes: dict[str, str] = field(default_factory=dict)
    simulation_summary: dict[str, Any] | None = None
    component_projections_path: Path | None = None
    validation_passed: bool = False
    caveats: list[str] = field(default_factory=list)
    model_id: str = "unknown"
    generated_at: str | None = None
    draw_count: int | None = None
    source_commit: str | None = None

    def cache_key(self) -> str:
        return f"{self.season}:{self.namespace}:{self.release_id}:{self.manifest_sha256}"

    def provenance(self) -> dict[str, Any]:
        return {
            "projection_source": "sealed_release",
            "namespace": self.namespace,
            "release_id": self.release_id,
            "manifest_sha256": self.manifest_sha256,
            "model_id": self.model_id,
            "generated_at": self.generated_at,
            "draw_count": self.draw_count,
            "source_commit": self.source_commit,
            "artifact_hashes": dict(self.artifact_hashes),
            "validation_passed": self.validation_passed,
            "caveats": list(self.caveats),
        }


def _resolve_bundle_roots(namespace: str, season: int) -> list[Path]:
    return [
        public_release_dir(namespace),
        Path(MODEL_V3_DIR) / "release_bundles" / f"season={season}" / f"namespace={namespace}",
    ]


def _artifact_entry(manifest: dict[str, Any], role: str) -> dict[str, Any] | None:
    for entry in manifest.get("artifacts", []):
        if entry.get("role") == role:
            return entry
    return None


def _artifact_path(bundle_root: Path, manifest: dict[str, Any], role: str) -> Path | None:
    entry = _artifact_entry(manifest, role)
    if entry is None:
        return None
    rel = str(entry.get("path") or "")
    if not rel or ".." in rel.replace("\\", "/"):
        raise ReleaseBundleLoadError(f"path traversal rejected for role {role}")
    return bundle_root / rel


def _resolve_artifact_path(
    namespace: str,
    season: int,
    manifest: dict[str, Any],
    role: str,
    *,
    primary_root: Path,
) -> Path | None:
    """Resolve a manifest artifact across public and full bundle roots."""
    entry = _artifact_entry(manifest, role)
    if entry is None:
        return None
    rel = str(entry.get("path") or "")
    if not rel or ".." in rel.replace("\\", "/"):
        raise ReleaseBundleLoadError(f"path traversal rejected for role {role}")
    for root in _resolve_bundle_roots(namespace, season):
        path = root / rel
        if path.is_file():
            return path
    return primary_root / rel if (primary_root / rel).is_file() else None


def _load_simulation_summary(
    namespace: str,
    season: int,
    bundle_root: Path,
    manifest: dict[str, Any],
) -> dict[str, Any] | None:
    path = _resolve_artifact_path(
        namespace,
        season,
        manifest,
        "simulation_summary",
        primary_root=bundle_root,
    )
    if path is None or not path.is_file():
        legacy = bundle_root / f"simulation_summary_{season}.csv"
        if not legacy.is_file():
            return None
        path = legacy
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        return {"row_count": len(rows), "path": str(path)}
    except OSError:
        return None


class ReleaseBundleLoader:
    """Production integration seam for sealed release bundles.

    Validates pointer schema, manifest hash, release identity, and every consumed
    artifact hash. Caches by pointer/manifest identity and reloads after atomic
    pointer swaps.
    """

    def __init__(self, season: int = 2026) -> None:
        self.season = season
        self._bundle: ReleaseBundleSnapshot | None = None
        self._cache_key: str | None = None
        self._players: dict[str, PlayerSummary] | None = None
        self._meta: dict | None = None

    def _current_cache_key(self) -> str | None:
        try:
            pointer = read_active_pointer(self.season)
        except ActiveReleaseError:
            return None
        if pointer is None:
            return None
        return (
            f"{self.season}:{pointer['namespace']}:{pointer['release_id']}:"
            f"{pointer['manifest_sha256']}"
        )

    def invalidate(self) -> None:
        self._bundle = None
        self._cache_key = None
        self._players = None
        self._meta = None

    def _resolve_bundle_root(self, namespace: str) -> Path | None:
        for root in _resolve_bundle_roots(namespace, self.season):
            manifest = root / MANIFEST_FILENAME
            if manifest.is_file():
                return root
        return None

    def load_bundle(self, *, verify_hashes: bool = True) -> ReleaseBundleSnapshot | None:
        cache_key = self._current_cache_key()
        if self._bundle is not None and cache_key == self._cache_key:
            return self._bundle

        try:
            pointer = read_active_pointer(self.season)
        except ActiveReleaseError as exc:
            raise ReleaseBundleLoadError(str(exc)) from exc
        if pointer is None:
            self.invalidate()
            return None

        validate_active_pointer(pointer, season=self.season)
        session = frozen_load_session(pointer)
        namespace = session["namespace"]
        resolver = ReleaseBundleResolver(
            season=self.season,
            namespace=namespace,
            manifest_storage_uri=pointer.get("manifest_storage_uri"),
        )
        try:
            manifest, manifest_digest, bundle_root = resolver.load_manifest()
        except ReleaseBundleError as exc:
            raise ReleaseBundleLoadError(str(exc)) from exc

        if manifest_digest != pointer["manifest_sha256"]:
            raise ReleaseBundleLoadError(
                "pointer manifest_sha256 does not match sealed manifest file"
            )
        verify_provenance_identities(
            manifest,
            expected={
                "season": self.season,
                "namespace": namespace,
                "release_id": session["release_id"],
            },
        )

        settings = get_settings()
        caveats: list[str] = []
        artifact_hashes: dict[str, str] = {}
        consumed_roles = ("players", "selected_board", "projections", "simulation_summary")
        required_roles = ("players",)
        if settings.app_env == "production":
            required_roles = ("players", "projections")

        for role in consumed_roles:
            content, storage_uri, local_path = resolver.resolve_artifact(
                manifest, role, primary_root=bundle_root
            )
            if content is None:
                if role in required_roles:
                    raise ReleaseBundleLoadError(f"missing artifact {role}")
                caveats.append(f"missing_optional_artifact:{role}")
                continue
            digest = hashlib.sha256(content).hexdigest()
            entry = next(row for row in manifest.get("artifacts", []) if row.get("role") == role)
            if digest != entry["sha256"]:
                raise ReleaseBundleLoadError(
                    f"hash mismatch for role {role}: manifest={entry['sha256']} file={digest}"
                )
            artifact_hashes[role] = digest

        players_content, _, players_path = resolver.resolve_artifact(
            manifest, "players", primary_root=bundle_root
        )
        if players_content is None:
            public_urls = pointer.get("public_urls") or {}
            players_rel = public_urls.get("players")
            if players_rel:
                players_path = Path(REPO_ROOT) / "draft_assistant" / players_rel.lstrip("/")
                if players_path.is_file():
                    players_content = players_path.read_bytes()
        if players_content is None:
            raise ReleaseBundleLoadError("players artifact missing from sealed bundle")

        payload = json.loads(players_content.decode("utf-8"))
        meta = payload.get("meta", {})
        index: dict[str, PlayerSummary] = {}
        for row in payload.get("players", []):
            player_id = str(row.get("player_id", ""))
            if not player_id:
                continue
            games = float(row.get("projected_games") or 17.0)
            season_pts = float(row.get("fantasy_pts_season") or row.get("fantasy_pts", 0.0) * games)
            per_game = season_pts / games if games else float(row.get("fantasy_pts", 0.0))
            index[player_id] = PlayerSummary(
                player_id=player_id,
                name=str(row.get("display_name", player_id)),
                position=str(row.get("position", "RB")),
                team=row.get("team"),
                mean_points=per_game,
                quantiles={
                    "0.1": float(row.get("fantasy_pts_p10", per_game * 0.7)) / games
                    if games
                    else per_game * 0.7,
                    "0.5": float(row.get("fantasy_pts_p50", season_pts)) / games
                    if games
                    else per_game,
                    "0.9": float(row.get("fantasy_pts_p90", per_game * 1.3)) / games
                    if games
                    else per_game * 1.3,
                },
                availability_probability=min(1.0, games / 17.0),
            )

        projections_content, _, component_path = resolver.resolve_artifact(
            manifest, "projections", primary_root=bundle_root
        )
        if projections_content is None:
            if settings.app_env == "production":
                raise ReleaseBundleLoadError("missing required projections artifact in production")
            caveats.append("missing_sealed_component_projections")
        elif component_path is None and projections_content is not None:
            component_path = resolver.materialize_to_temp(projections_content, suffix=".csv")

        sim = manifest.get("simulation") or {}
        git = manifest.get("git") or {}
        bundle = manifest.get("bundle") or {}

        snapshot = ReleaseBundleSnapshot(
            season=self.season,
            namespace=namespace,
            release_id=str(bundle.get("release_id") or session["release_id"]),
            manifest_sha256=manifest_digest,
            manifest_path=(bundle_root / MANIFEST_FILENAME) if bundle_root else Path("remote"),
            bundle_root=bundle_root or Path("remote"),
            players=index,
            meta=meta,
            artifact_hashes=artifact_hashes,
            simulation_summary=_load_simulation_summary(namespace, self.season, bundle_root or Path("."), manifest)
            if bundle_root
            else None,
            component_projections_path=component_path,
            validation_passed=True,
            caveats=caveats,
            model_id=str(bundle.get("model_id") or meta.get("model_id") or "accuracy_first_ensemble"),
            generated_at=str(meta.get("generated_at") or bundle.get("created_at")),
            draw_count=int(sim.get("draw_count")) if sim.get("draw_count") else None,
            source_commit=str(git.get("source_commit")) if git.get("source_commit") else None,
        )
        self._bundle = snapshot
        self._cache_key = snapshot.cache_key()
        self._players = index
        self._meta = meta
        return snapshot

    def players_path(self) -> Path | None:
        bundle = self.load_bundle()
        if bundle is None:
            return None
        path = _artifact_path(
            bundle.bundle_root,
            json.loads(bundle.manifest_path.read_text(encoding="utf-8")),
            "players",
        )
        return path

    def load(self) -> dict[str, PlayerSummary]:
        bundle = self.load_bundle()
        if bundle is None:
            return {}
        return bundle.players

    @property
    def meta(self) -> dict:
        if self._meta is None:
            self.load()
        return self._meta or {}

    def get(self, player_id: str) -> PlayerSummary | None:
        return self.load().get(player_id)

    def available_pool(self, rostered_ids: set[str], positions: set[str] | None = None) -> list[PlayerSummary]:
        pool = []
        for player_id, summary in self.load().items():
            if player_id in rostered_ids:
                continue
            if positions and summary.position not in positions:
                continue
            pool.append(summary)
        pool.sort(key=lambda p: p.mean_points, reverse=True)
        return pool[:100]

    def as_of(self) -> str:
        bundle = self.load_bundle()
        if bundle and bundle.generated_at:
            return bundle.generated_at
        generated = self.meta.get("generated_at")
        return generated or datetime.now(UTC).isoformat()

    def provenance(self) -> dict[str, Any]:
        bundle = self.load_bundle()
        if bundle is None:
            return {"projection_source": "sealed_release", "validation_passed": False}
        return bundle.provenance()


@lru_cache
def get_bundle_loader(season: int = 2026) -> ReleaseBundleLoader:
    return ReleaseBundleLoader(season=season)


def invalidate_bundle_loader_cache(season: int = 2026) -> None:
    get_bundle_loader(season).invalidate()
    get_bundle_loader.cache_clear()
