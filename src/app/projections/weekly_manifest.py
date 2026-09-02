"""Weekly-v2 manifest schema, hash verification, and readiness contract."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.projection.weekly.config.paths import MODELS_DIR, OUTPUTS_DIR

MANIFEST_SCHEMA_V2 = 2
SUPPORTED_SCHEMA_VERSIONS = frozenset({1, 2})

REQUIRED_MODEL_ARTIFACTS: tuple[str, ...] = (
    "team_totals.joblib",
    "volume_QB.joblib",
    "volume_RB.joblib",
    "volume_WR.joblib",
    "volume_TE.joblib",
    "efficiency_QB.joblib",
    "efficiency_RB.joblib",
    "efficiency_WR.joblib",
    "efficiency_TE.joblib",
)

OPTIONAL_ARTIFACTS: tuple[str, ...] = (
    "calibration.json",
    "tuning_selection.json",
    "rookie_QB.joblib",
    "rookie_RB.joblib",
    "rookie_WR.joblib",
    "rookie_TE.joblib",
)

_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _models_dir() -> Path:
    override = os.getenv("WEEKLY_V2_MODELS_DIR")
    return Path(override) if override else MODELS_DIR


def _outputs_dir() -> Path:
    override = os.getenv("WEEKLY_V2_OUTPUTS_DIR")
    return Path(override) if override else OUTPUTS_DIR


def manifest_path(season: int) -> Path:
    return _models_dir() / f"season={season}" / "manifest.json"


def safe_artifact_path(models_root: Path, name: str) -> Path | None:
    if not _SAFE_NAME.match(name):
        return None
    resolved = (models_root / name).resolve()
    try:
        resolved.relative_to(models_root.resolve())
    except ValueError:
        return None
    return resolved


@dataclass(frozen=True)
class ManifestValidation:
    season: int
    valid: bool
    schema_version: int | None
    model_version: str | None
    trained_through_season: int | None
    missing_artifacts: tuple[str, ...]
    hash_mismatches: tuple[str, ...]
    load_failures: tuple[str, ...]
    failures: tuple[str, ...] = ()
    artifact_hashes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "season": self.season,
            "valid": self.valid,
            "schema_version": self.schema_version,
            "model_version": self.model_version,
            "trained_through_season": self.trained_through_season,
            "missing_artifacts": list(self.missing_artifacts),
            "hash_mismatches": list(self.hash_mismatches),
            "load_failures": list(self.load_failures),
            "failures": list(self.failures),
            "artifact_hashes": dict(self.artifact_hashes),
        }


@dataclass(frozen=True)
class OutputProvenance:
    season: int
    week: int
    path: Path
    sha256: str
    model_hashes: dict[str, str]
    observed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "season": self.season,
            "week": self.week,
            "path": str(self.path),
            "sha256": self.sha256,
            "model_hashes": dict(self.model_hashes),
            "observed_at": self.observed_at,
        }


def _artifact_entries(manifest: dict) -> dict[str, dict[str, Any]]:
    if manifest.get("schema_version") == MANIFEST_SCHEMA_V2:
        entries = dict(manifest.get("artifacts") or {})
        entries.update(manifest.get("optional_artifacts") or {})
        return {str(k): v for k, v in entries.items() if isinstance(v, dict)}
    legacy: dict[str, dict[str, Any]] = {}
    for row in manifest.get("artifacts") or []:
        if not isinstance(row, dict):
            continue
        path = Path(str(row.get("path", "")))
        name = path.name
        if name:
            legacy[name] = {
                "fingerprint": row.get("fingerprint"),
                "size": row.get("size"),
            }
    return legacy


def verify_models_loadable(models_root: Path, names: tuple[str, ...]) -> list[str]:
    failures: list[str] = []
    import joblib

    for name in names:
        path = safe_artifact_path(models_root, name)
        if path is None or not path.exists():
            failures.append(f"unsafe_or_missing:{name}")
            continue
        try:
            joblib.load(path)
        except Exception as exc:
            failures.append(f"unloadable:{name}:{type(exc).__name__}")
    return failures


def validate_manifest(season: int, *, models_dir: Path | None = None) -> ManifestValidation:
    root = models_dir or _models_dir()
    season_dir = root / f"season={season}"
    manifest_file = season_dir / "manifest.json"
    failures: list[str] = []
    missing: list[str] = []
    hash_mismatches: list[str] = []
    artifact_hashes: dict[str, str] = {}

    if not manifest_file.exists():
        return ManifestValidation(
            season=season,
            valid=False,
            schema_version=None,
            model_version=None,
            trained_through_season=None,
            missing_artifacts=REQUIRED_MODEL_ARTIFACTS,
            hash_mismatches=(),
            load_failures=(),
            failures=("manifest_missing",),
        )

    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ManifestValidation(
            season=season,
            valid=False,
            schema_version=None,
            model_version=None,
            trained_through_season=None,
            missing_artifacts=REQUIRED_MODEL_ARTIFACTS,
            hash_mismatches=(),
            load_failures=(),
            failures=("manifest_unreadable",),
        )

    schema_version = manifest.get("schema_version")
    try:
        schema_version = int(schema_version)
    except (TypeError, ValueError):
        failures.append("manifest_schema_version_invalid")
        schema_version = None
    if schema_version is not None and schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        failures.append(f"manifest_schema_version_unsupported:{schema_version}")

    model_version = manifest.get("model_version")
    if not model_version:
        failures.append("manifest_missing_model_version")

    trained_through = manifest.get("trained_through_season")
    if trained_through is None and manifest.get("train_seasons"):
        trained_through = max(int(x) for x in manifest["train_seasons"])
    if trained_through is not None and int(trained_through) >= season:
        failures.append(f"training_cutoff_not_before_target:{trained_through}>={season}")

    entries = _artifact_entries(manifest)
    search_roots = (season_dir, root)

    for name in REQUIRED_MODEL_ARTIFACTS:
        path = None
        for candidate_root in search_roots:
            candidate = safe_artifact_path(candidate_root, name)
            if candidate is not None and candidate.exists():
                path = candidate
                break
        if path is None:
            missing.append(name)
            continue
        actual_hash = sha256_file(path)
        artifact_hashes[name] = actual_hash
        expected = (entries.get(name) or {}).get("sha256") or (entries.get(name) or {}).get(
            "fingerprint"
        )
        if expected and expected != actual_hash:
            hash_mismatches.append(name)

    load_failures: tuple[str, ...] = ()
    if not missing and not hash_mismatches and schema_version == MANIFEST_SCHEMA_V2:
        load_failures = tuple(verify_models_loadable(season_dir if season_dir.exists() else root, REQUIRED_MODEL_ARTIFACTS))
        if load_failures:
            failures.extend(load_failures)

    valid = not failures and not missing and not hash_mismatches and not load_failures
    return ManifestValidation(
        season=season,
        valid=valid,
        schema_version=schema_version,
        model_version=str(model_version) if model_version else None,
        trained_through_season=int(trained_through) if trained_through is not None else None,
        missing_artifacts=tuple(missing),
        hash_mismatches=tuple(hash_mismatches),
        load_failures=load_failures,
        failures=tuple(failures),
        artifact_hashes=artifact_hashes,
    )


def output_provenance_path(season: int, week: int) -> Path:
    return (
        _outputs_dir()
        / f"season={season}"
        / f"week={week:02d}"
        / "output_provenance.json"
    )


def load_output_provenance(season: int, week: int) -> OutputProvenance | None:
    path = output_provenance_path(season, week)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return OutputProvenance(
        season=int(payload["season"]),
        week=int(payload["week"]),
        path=Path(payload.get("output_path", "")),
        sha256=str(payload.get("sha256", "")),
        model_hashes={str(k): str(v) for k, v in (payload.get("model_hashes") or {}).items()},
        observed_at=payload.get("observed_at"),
    )


def verify_output_provenance(
    season: int,
    week: int,
    *,
    manifest_validation: ManifestValidation,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    provenance = load_output_provenance(season, week)
    if provenance is None:
        return False, ["output_provenance_missing"]
    if provenance.season != season or provenance.week != week:
        failures.append("output_provenance_season_week_mismatch")
    if provenance.path.exists():
        actual = sha256_file(provenance.path)
        if provenance.sha256 and provenance.sha256 != actual:
            failures.append("output_hash_mismatch")
    else:
        failures.append("output_file_missing")
    for name, expected in manifest_validation.artifact_hashes.items():
        recorded = provenance.model_hashes.get(name)
        if recorded != expected:
            failures.append(f"model_output_link_broken:{name}")
    extras = _read_provenance_extras(season, week)
    partition_path = provenance.path.parent / "stat_draw_partition.json"
    if not partition_path.exists():
        failures.append("stat_draw_partition_missing")
    else:
        recorded_partition = extras.get("partition_sha256")
        if recorded_partition:
            from src.app.projections.weekly_draws import verify_weekly_draw_partition

            if not verify_weekly_draw_partition(
                partition_path, expected_sha256=str(recorded_partition)
            ):
                failures.append("partition_hash_mismatch")
    return not failures, failures


def _read_provenance_extras(season: int, week: int) -> dict[str, Any]:
    path = output_provenance_path(season, week)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def build_manifest_v2(
    *,
    target_season: int,
    train_seasons: list[int],
    models_dir: Path,
    data_inputs: list[tuple[str, Path]],
    code_revision: str | None = None,
    library_versions: dict[str, str] | None = None,
) -> dict[str, Any]:
    season_dir = models_dir / f"season={target_season}"
    artifacts: dict[str, dict[str, Any]] = {}
    optional: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_MODEL_ARTIFACTS + OPTIONAL_ARTIFACTS:
        path = safe_artifact_path(season_dir, name) or safe_artifact_path(models_dir, name)
        if path is None or not path.exists():
            continue
        entry = {"sha256": sha256_file(path), "size": path.stat().st_size}
        if name in REQUIRED_MODEL_ARTIFACTS:
            artifacts[name] = entry
        else:
            optional[name] = entry
    missing_required = [n for n in REQUIRED_MODEL_ARTIFACTS if n not in artifacts]
    if missing_required:
        raise ValueError(f"cannot build manifest; missing required artifacts: {missing_required}")

    content_hash = hashlib.sha256(
        json.dumps(artifacts, sort_keys=True).encode()
    ).hexdigest()[:16]
    model_version = f"weekly_v2_{target_season}_{content_hash}"

    payload: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_V2,
        "target_season": target_season,
        "model_version": model_version,
        "trained_through_season": max(train_seasons),
        "train_seasons": sorted(int(x) for x in train_seasons),
        "artifacts": artifacts,
        "optional_artifacts": optional,
        "data_inputs": [
            {
                "relative_path": rel,
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }
            for rel, path in data_inputs
            if path.exists()
        ],
    }
    if code_revision:
        payload["code_revision"] = code_revision
    if library_versions:
        payload["library_versions"] = library_versions
    return payload


def write_manifest_v2(
    *,
    target_season: int,
    train_seasons: list[int],
    models_dir: Path | None = None,
    data_inputs: list[tuple[str, Path]] | None = None,
) -> Path:
    root = models_dir or _models_dir()
    season_dir = root / f"season={target_season}"
    season_dir.mkdir(parents=True, exist_ok=True)
    payload = build_manifest_v2(
        target_season=target_season,
        train_seasons=train_seasons,
        models_dir=root,
        data_inputs=data_inputs or [],
    )
    path = season_dir / "manifest.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
