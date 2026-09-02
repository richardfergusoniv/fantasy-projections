"""Detect and classify weekly-v2 artifacts before they are used for publication.

Readiness is a verified contract, not mere file presence. A run may be labelled
``trained`` only when manifest hashes validate, required models load in-process,
and a real weekly output exists whose provenance links back to those exact model
hashes. Presence of nine joblib files without output provenance is ``fallback``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from src.app.projections.weekly_manifest import (
    REQUIRED_MODEL_ARTIFACTS,
    validate_manifest,
    verify_output_provenance,
)
from src.projection.weekly.config.paths import MODELS_DIR, OUTPUTS_DIR

FIXTURE_MANIFEST_ROOT = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "weekly_v2"
)

SUPPORTED_MANIFEST_SCHEMA_VERSIONS = frozenset({1, 2})

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
    auto_publish_allowed: bool = False

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
            "auto_publish_allowed": self.auto_publish_allowed,
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


def _evaluation_promotion_passes(season: int) -> tuple[bool, list[str]]:
    report_path = _outputs_dir() / "preseason_backtest.json"
    if not report_path.exists():
        return False, ["evaluation_report_missing"]
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, ["evaluation_report_unreadable"]
    promotion = payload.get("promotion") or {}
    if promotion.get("promote"):
        return True, []
    failures = promotion.get("failures") or []
    return False, [f"evaluation_promotion_failed:{failure}" for failure in failures]


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def weekly_v2_readiness(season: int, week: int) -> WeeklyV2Readiness:
    """Classify weekly-v2 artifacts for ``season``/``week``.

    ``trained``  manifest v2 validates, models load, output provenance links to
                 the same model hashes, and no fallback derivation is in use.
    ``fallback`` weights or manifest exist but the full contract is incomplete.
    ``fixture``  only the checked-in test fixture manifest is present.
    ``absent``   nothing weekly-v2 related exists for this season.
    """
    reasons: list[str] = []
    models_manifest_path = _models_manifest(season)
    validation = validate_manifest(season)

    if validation.valid:
        output_ok, output_reasons = verify_output_provenance(
            season, week, manifest_validation=validation
        )
        if output_ok:
            eval_ok, eval_reasons = _evaluation_promotion_passes(season)
            return WeeklyV2Readiness(
                season=season,
                week=week,
                state=STATE_TRAINED,
                model_version=str(validation.model_version),
                manifest_uri=models_manifest_path.resolve().as_uri(),
                manifest_path=str(models_manifest_path),
                missing_artifacts=(),
                reasons=tuple(eval_reasons),
                auto_publish_allowed=eval_ok,
            )
        reasons.extend(output_reasons)
        reasons.append("trained_artifacts_without_verified_output")
    else:
        reasons.extend(validation.failures)
        if validation.missing_artifacts:
            reasons.append(f"missing_model_artifacts:{len(validation.missing_artifacts)}")
        if validation.hash_mismatches:
            reasons.append(f"hash_mismatches:{len(validation.hash_mismatches)}")

    if models_manifest_path.exists() or validation.artifact_hashes:
        model_version = validation.model_version or DEFAULT_MODEL_VERSION
        return WeeklyV2Readiness(
            season=season,
            week=week,
            state=STATE_FALLBACK,
            model_version=str(model_version),
            manifest_uri=f"weekly-v2-fallback://{models_manifest_path.as_posix()}",
            manifest_path=str(models_manifest_path) if models_manifest_path.exists() else None,
            missing_artifacts=validation.missing_artifacts or REQUIRED_MODEL_ARTIFACTS,
            reasons=tuple(reasons),
            auto_publish_allowed=False,
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
            missing_artifacts=validation.missing_artifacts or REQUIRED_MODEL_ARTIFACTS,
            reasons=tuple(reasons),
            auto_publish_allowed=False,
        )

    reasons.append("no_weekly_v2_artifacts")
    return WeeklyV2Readiness(
        season=season,
        week=week,
        state=STATE_ABSENT,
        model_version=DEFAULT_MODEL_VERSION,
        manifest_uri=f"derived://preseason-bundle/{season}/{week:02d}",
        manifest_path=None,
        missing_artifacts=validation.missing_artifacts or REQUIRED_MODEL_ARTIFACTS,
        reasons=tuple(reasons),
        auto_publish_allowed=False,
    )


def weekly_v2_artifacts_available(season: int, week: int) -> bool:
    """Backwards-compatible presence check (not readiness)."""
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
