"""Detect and classify weekly-v2 artifacts before they are used for publication.

The bridge previously answered a single boolean question ("are weekly v2
artifacts available?") and answered ``True`` whenever *any* candidate path
existed — including the checked-in test fixture manifest, which ships in the
repository. Season 2026 therefore always looked "available" even though no
trained model weights exist anywhere, and the resulting run still carried a
``fixture://`` manifest URI. This module now reports an explicit state so the
publication pipeline can refuse to auto-publish untrained output.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from src.projection.weekly.config.paths import MODELS_DIR, OUTPUTS_DIR

FIXTURE_MANIFEST_ROOT = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "weekly_v2"
)

#: Weight files the weekly v2 inference path loads through
#: :mod:`src.projection.weekly.models.registry`. Every one of these must exist
#: before a run may be labelled ``trained``.
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

#: Manifest ``schema_version`` values this application knows how to consume.
SUPPORTED_MANIFEST_SCHEMA_VERSIONS: frozenset[int] = frozenset({1})

STATE_TRAINED = "trained"
STATE_FALLBACK = "fallback"
STATE_FIXTURE = "fixture"
STATE_ABSENT = "absent"

DEFAULT_MODEL_VERSION = "weekly_fixture_v1"


@dataclass(frozen=True)
class WeeklyV2Readiness:
    """Classification of the weekly-v2 artifacts backing one season/week."""

    season: int
    week: int
    state: str
    model_version: str
    manifest_uri: str
    manifest_path: str | None
    missing_artifacts: tuple[str, ...]
    reasons: tuple[str, ...]

    @property
    def is_trained(self) -> bool:
        return self.state == STATE_TRAINED

    def to_dict(self) -> dict:
        return {
            "season": self.season,
            "week": self.week,
            "state": self.state,
            "model_version": self.model_version,
            "manifest_uri": self.manifest_uri,
            "manifest_path": self.manifest_path,
            "missing_artifacts": list(self.missing_artifacts),
            "reasons": list(self.reasons),
        }


def _models_dir() -> Path:
    override = os.getenv("WEEKLY_V2_MODELS_DIR")
    return Path(override) if override else MODELS_DIR


def _outputs_dir() -> Path:
    override = os.getenv("WEEKLY_V2_OUTPUTS_DIR")
    return Path(override) if override else OUTPUTS_DIR


def _fixture_root() -> Path:
    override = os.getenv("WEEKLY_V2_FIXTURE_ROOT")
    return Path(override) if override else FIXTURE_MANIFEST_ROOT


def _fixture_manifest(season: int) -> Path:
    return _fixture_root() / f"season={season}" / "manifest.json"


def _models_manifest(season: int) -> Path:
    return _models_dir() / f"season={season}" / "manifest.json"


def _artifact_search_roots(season: int) -> tuple[Path, ...]:
    models = _models_dir()
    return (models / f"season={season}", models)


def _missing_model_artifacts(season: int) -> tuple[str, ...]:
    roots = _artifact_search_roots(season)
    return tuple(
        name for name in REQUIRED_MODEL_ARTIFACTS if not any((root / name).exists() for root in roots)
    )


def _real_output_paths(season: int, week: int) -> tuple[Path, ...]:
    outputs = _outputs_dir()
    week_dir = outputs / f"season={season}" / f"week={week:02d}"
    return (
        week_dir / "weekly_projections.parquet",
        week_dir / "weekly_projections.json",
        outputs / f"projections_{season}_w{week:02d}.csv",
    )


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _manifest_incompatibility(manifest: dict | None) -> str | None:
    if manifest is None:
        return "manifest_unreadable"
    if "model_version" not in manifest:
        return "manifest_missing_model_version"
    schema_version = manifest.get("schema_version")
    if schema_version is None:
        return "manifest_missing_schema_version"
    try:
        schema_version = int(schema_version)
    except (TypeError, ValueError):
        return "manifest_schema_version_invalid"
    if schema_version not in SUPPORTED_MANIFEST_SCHEMA_VERSIONS:
        return f"manifest_schema_version_unsupported:{schema_version}"
    return None


def weekly_v2_readiness(season: int, week: int) -> WeeklyV2Readiness:
    """Classify the weekly-v2 artifacts for ``season``/``week``.

    ``trained``  every required weight file exists and the models manifest is
                 schema-compatible.
    ``fallback`` real weekly output exists (or a models manifest does) but the
                 weights are missing or the manifest schema is not supported.
    ``fixture``  only the checked-in test fixture manifest is present.
    ``absent``   nothing weekly-v2 related exists for this season.
    """
    reasons: list[str] = []
    models_manifest_path = _models_manifest(season)
    models_manifest = _read_json(models_manifest_path) if models_manifest_path.exists() else None
    missing = _missing_model_artifacts(season)

    if not missing:
        incompatibility = _manifest_incompatibility(models_manifest)
        if incompatibility is None:
            return WeeklyV2Readiness(
                season=season,
                week=week,
                state=STATE_TRAINED,
                model_version=str(models_manifest.get("model_version")),
                manifest_uri=models_manifest_path.resolve().as_uri(),
                manifest_path=str(models_manifest_path),
                missing_artifacts=(),
                reasons=(),
            )
        reasons.append(incompatibility)
    else:
        reasons.append(f"missing_model_artifacts:{len(missing)}")

    real_outputs = [path for path in _real_output_paths(season, week) if path.exists()]
    if real_outputs or models_manifest_path.exists():
        source = real_outputs[0] if real_outputs else models_manifest_path
        model_version = DEFAULT_MODEL_VERSION
        if models_manifest is not None:
            model_version = str(models_manifest.get("model_version", DEFAULT_MODEL_VERSION))
        return WeeklyV2Readiness(
            season=season,
            week=week,
            state=STATE_FALLBACK,
            model_version=model_version,
            manifest_uri=f"weekly-v2-fallback://{source.as_posix()}",
            manifest_path=str(models_manifest_path) if models_manifest_path.exists() else None,
            missing_artifacts=missing,
            reasons=tuple(reasons),
        )

    fixture_path = _fixture_manifest(season)
    if fixture_path.exists():
        fixture_manifest = _read_json(fixture_path) or {}
        reasons.append("fixture_manifest_only")
        return WeeklyV2Readiness(
            season=season,
            week=week,
            state=STATE_FIXTURE,
            model_version=str(fixture_manifest.get("model_version", DEFAULT_MODEL_VERSION)),
            manifest_uri=f"fixture://weekly-v2/{fixture_path.as_posix()}",
            manifest_path=str(fixture_path),
            missing_artifacts=missing,
            reasons=tuple(reasons),
        )

    reasons.append("no_weekly_v2_artifacts")
    return WeeklyV2Readiness(
        season=season,
        week=week,
        state=STATE_ABSENT,
        model_version=DEFAULT_MODEL_VERSION,
        manifest_uri=f"derived://preseason-bundle/{season}/{week:02d}",
        manifest_path=None,
        missing_artifacts=missing,
        reasons=tuple(reasons),
    )


def weekly_v2_artifacts_available(season: int, week: int) -> bool:
    """Backwards-compatible presence check.

    Presence is *not* readiness: a ``True`` here still covers the ``fixture`` and
    ``fallback`` states. Callers deciding whether to publish must use
    :func:`weekly_v2_readiness` and the artifact-readiness gate instead.
    """
    return weekly_v2_readiness(season, week).state != STATE_ABSENT


def load_weekly_v2_manifest(season: int) -> dict | None:
    for manifest_path in (_models_manifest(season), _fixture_manifest(season)):
        if manifest_path.exists():
            return _read_json(manifest_path)
    return None


def weekly_v2_model_version(season: int) -> str:
    manifest = load_weekly_v2_manifest(season)
    if manifest is None:
        return DEFAULT_MODEL_VERSION
    return str(manifest.get("model_version", "weekly_v2"))
