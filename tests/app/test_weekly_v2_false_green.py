"""Regression tests: dummy artifacts cannot produce trained/publishable weekly output."""

from __future__ import annotations

import json
from pathlib import Path

from pathlib import Path

import joblib
import pytest
from sklearn.linear_model import Ridge

from src.app.projections.weekly_manifest import REQUIRED_MODEL_ARTIFACTS, validate_manifest
from src.app.projections.weekly_v2_bridge import STATE_FALLBACK, STATE_FIXTURE, STATE_TRAINED, WeeklyV2Readiness, weekly_v2_readiness
from src.app.releases.gates import validate_artifact_readiness, validate_inference_provenance
from src.app.releases.publication import Candidate, CandidateRow


def _write_dummy_joblibs(models_dir: Path) -> None:
    model = Ridge()
    model.fit([[1.0], [2.0]], [1.0, 2.0])
    for name in REQUIRED_MODEL_ARTIFACTS:
        joblib.dump(model, models_dir / name)


def test_nine_dummy_joblibs_without_provenance_are_not_trained(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    season_dir = models_dir / "season=2026"
    season_dir.mkdir(parents=True)
    _write_dummy_joblibs(season_dir)

    manifest = {
        "schema_version": 2,
        "target_season": 2026,
        "model_version": "dummy_test",
        "trained_through_season": 2025,
        "train_seasons": list(range(2016, 2026)),
        "artifacts": {},
    }
    for name in REQUIRED_MODEL_ARTIFACTS:
        path = season_dir / name
        from src.app.projections.weekly_manifest import sha256_file

        manifest["artifacts"][name] = {"sha256": sha256_file(path), "size": path.stat().st_size}

    (season_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setenv("WEEKLY_V2_MODELS_DIR", str(models_dir))
    monkeypatch.setenv("WEEKLY_V2_OUTPUTS_DIR", str(tmp_path / "outputs"))

    validation = validate_manifest(2026)
    assert validation.missing_artifacts == ()
    readiness = weekly_v2_readiness(2026, 1)
    assert readiness.state != STATE_TRAINED
    assert readiness.state == STATE_FALLBACK
    assert readiness.auto_publish_allowed is False
    assert "output_provenance_missing" in readiness.reasons


def test_fixture_manifest_stays_fixture(monkeypatch):
    readiness = weekly_v2_readiness(2026, 1)
    if readiness.state != STATE_FIXTURE:
        pytest.skip("fixture manifest not present in this environment")
    assert readiness.auto_publish_allowed is False


def test_hash_scaled_derivation_blocked_for_trained_mode():
    candidate = Candidate(
        mode="weekly",
        season=2026,
        week=1,
        run_id="weekly-test",
        model_version="fake",
        input_hash="abc",
        manifest_uri="file://fake",
        artifact_mode="trained",
        partition_mode="weekly",
        rows=(
            CandidateRow(
                player_id="00-0034857",
                team="BUF",
                opponent=None,
                availability_probability=1.0,
                mean_json={"points": 12.0, "derivation": "preseason_bundle_scaled"},
                quantiles_json={"0.5": 12.0},
            ),
        ),
        metadata={"derivation": "preseason_bundle_scaled"},
    )

    class _Readiness:
        state = STATE_TRAINED

    gate = validate_inference_provenance(candidate, _Readiness())
    assert gate.passed is False
    assert any("derivation" in failure for failure in gate.failures)


def test_trained_readiness_requires_output_provenance_link(tmp_path, monkeypatch):
    """Manifest-valid models without linked output cannot auto-publish."""
    models_dir = tmp_path / "models"
    season_dir = models_dir / "season=2026"
    season_dir.mkdir(parents=True)
    _write_dummy_joblibs(season_dir)
    monkeypatch.setenv("WEEKLY_V2_MODELS_DIR", str(models_dir))
    monkeypatch.setenv("WEEKLY_V2_OUTPUTS_DIR", str(tmp_path / "outputs"))

    readiness = weekly_v2_readiness(2026, 1)
    assert readiness.is_trained is False
    assert readiness.auto_publish_allowed is False


def test_automatic_promotion_blocked_when_eval_gate_fails():
    readiness = WeeklyV2Readiness(
        season=2026,
        week=1,
        state=STATE_TRAINED,
        model_version="weekly_v2_test",
        manifest_uri="file://test",
        manifest_path="/tmp/manifest.json",
        missing_artifacts=(),
        reasons=("evaluation_promotion_failed:2023: dispersion outside policy",),
        auto_publish_allowed=False,
    )
    gate = validate_artifact_readiness(readiness, app_env="production", automatic=True)
    assert gate.passed is False
    assert any("auto_publish_blocked" in failure for failure in gate.failures)

    manual = validate_artifact_readiness(readiness, app_env="production", automatic=False)
    assert manual.passed is True
    assert any("auto_publish_blocked" in warning for warning in manual.warnings)


@pytest.mark.skipif(
    not Path("output/weekly_v2/models/season=2026/manifest.json").exists(),
    reason="trained weekly artifacts missing",
)
def test_automatic_trained_promotion_does_not_swap_pointer(db_session, monkeypatch):
    """Production automatic runs must not promote when evaluation gate failed."""
    from src.app.projections.weekly_run import WeeklyProjectionService
    from src.app.releases.bridge import ReleaseBridge
    from src.app.releases.publication import active_pointer

    monkeypatch.delenv("WEEKLY_V2_MODELS_DIR", raising=False)
    monkeypatch.delenv("WEEKLY_V2_OUTPUTS_DIR", raising=False)
    monkeypatch.setenv("APP_ENV", "production")

    from src.app.config import get_settings

    get_settings.cache_clear()

    bridge = ReleaseBridge(db_session)
    if bridge.sync_preseason_pointer(2026) is None:
        pytest.skip("no active release bundle")

    service = WeeklyProjectionService(db_session)
    manual_id = service.promote_week(2026, week=1, automatic=False)
    assert manual_id is not None

    before = active_pointer(db_session, mode="weekly", season=2026, week=1)
    blocked = service.promote_week(2026, week=1, automatic=True)
    after = active_pointer(db_session, mode="weekly", season=2026, week=1)

    assert blocked is None
    assert before is not None and after is not None
    assert before.run_id == after.run_id == manual_id
