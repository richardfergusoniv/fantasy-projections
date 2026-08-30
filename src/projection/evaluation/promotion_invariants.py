"""Six mandatory promotion invariants — all must pass before pointer movement."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from src.projection.evaluation.accuracy_first import sha256_file
from src.projection.git_provenance import GitProvenanceError, verify_promotion_git_state
from src.projection.inference.recenter import board_points_series
from src.projection.overlay_coverage import OVERLAY_COVERAGE_FIELDS, overlay_coverage_alignment
from src.projection.release_bundle import (
    MANIFEST_FILENAME,
    ReleaseBundleError,
    bundle_root,
    load_sealed_manifest,
    promotion_eligible,
    public_release_dir,
    selected_points_vector_hash,
    sha256_bytes,
    verify_artifact_hashes,
)
from src.projection.simulation_profile_resolver import (
    PROFILE_IDENTITY_FIELDS,
    resolve_simulation_profile_identity,
)

PROMOTION_INVARIANTS_VERSION = "promotion_invariants_v2"

INVARIANT_NAMES = (
    "overlay_coverage_alignment",
    "selected_board_hash_alignment",
    "simulation_profile_identity",
    "ensemble_source_provenance",
    "browser_artifact_completeness",
    "git_provenance",
)


def _check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), **details}


def _artifact_by_role(manifest: Mapping[str, Any], role: str) -> dict[str, Any] | None:
    return next((entry for entry in manifest["artifacts"] if entry["role"] == role), None)


def _require_artifact(manifest: Mapping[str, Any], role: str) -> tuple[dict[str, Any] | None, list[str]]:
    entry = _artifact_by_role(manifest, role)
    if entry is None:
        return None, [f"missing {role} artifact"]
    return entry, []


def _check_overlay_coverage_alignment(
    manifest: Mapping[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    players_entry, errors = _require_artifact(manifest, "players")
    report_entry, report_errors = _require_artifact(manifest, "release_report")
    errors.extend(report_errors)
    if errors:
        return _check("overlay_coverage_alignment", False, mismatches=errors)
    players_doc = json.loads((root / players_entry["path"]).read_text(encoding="utf-8"))
    report = json.loads((root / report_entry["path"]).read_text(encoding="utf-8"))
    report_coverage = report.get("overlay_coverage")
    if report_coverage is None:
        return _check("overlay_coverage_alignment", False, mismatches=["release report missing overlay_coverage"])
    manifest_coverage = manifest.get("overlay_coverage")
    if manifest_coverage is None:
        return _check("overlay_coverage_alignment", False, mismatches=["manifest missing overlay_coverage"])
    ok, details = overlay_coverage_alignment(
        players_doc=players_doc,
        manifest_coverage=manifest_coverage,
        report_coverage=report_coverage,
    )
    return _check("overlay_coverage_alignment", ok, **details)


def _check_selected_board_hash_alignment(
    manifest: Mapping[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    mismatches: list[str] = []
    board = manifest.get("board")
    if not isinstance(board, Mapping):
        return _check("selected_board_hash_alignment", False, mismatches=["manifest missing board block"])
    for key in ("selected_board_sha256", "selected_board_file_hash", "selected_points_vector_hash"):
        if key not in board or not str(board.get(key) or "").strip():
            mismatches.append(f"manifest.board missing {key}")

    selected, selected_errors = _require_artifact(manifest, "selected_board")
    mismatches.extend(selected_errors)
    sim_entry, sim_errors = _require_artifact(manifest, "simulation_manifest")
    mismatches.extend(sim_errors)
    report_entry, report_errors = _require_artifact(manifest, "release_report")
    mismatches.extend(report_errors)
    players_entry, players_errors = _require_artifact(manifest, "players")
    mismatches.extend(players_errors)
    if mismatches:
        return _check("selected_board_hash_alignment", False, mismatches=mismatches)

    board_sha = str(board["selected_board_sha256"])
    board_file_hash = str(board["selected_board_file_hash"])
    if board_sha != board_file_hash:
        mismatches.append("selected_board_sha256 != selected_board_file_hash")
    if selected["sha256"] != board_sha:
        mismatches.append("selected_board artifact sha256 != selected_board_sha256")

    sim_doc = json.loads((root / sim_entry["path"]).read_text(encoding="utf-8"))
    for key in ("selected_board_sha256", "selected_board_hash"):
        if key not in sim_doc or str(sim_doc[key]) != board_sha:
            mismatches.append(f"simulation_manifest missing or mismatched {key}")

    report = json.loads((root / report_entry["path"]).read_text(encoding="utf-8"))
    report_board = report.get("board")
    if not isinstance(report_board, Mapping) or report_board.get("selected_board_sha256") != board_sha:
        mismatches.append("release_report.board.selected_board_sha256 missing or mismatched")

    players_doc = json.loads((root / players_entry["path"]).read_text(encoding="utf-8"))
    meta = players_doc.get("meta")
    if not isinstance(meta, Mapping) or meta.get("selected_board_sha256") != board_sha:
        mismatches.append("players.meta.selected_board_sha256 missing or mismatched")

    try:
        import pandas as pd

        board_df = pd.read_csv(root / selected["path"])
        vector_hash = selected_points_vector_hash(board_points_series(board_df))
        if vector_hash != board["selected_points_vector_hash"]:
            mismatches.append("selected_points_vector_hash mismatch")
    except Exception as exc:
        mismatches.append(f"selected_points_vector_hash recompute failed: {exc}")

    return _check(
        "selected_board_hash_alignment",
        not mismatches,
        selected_board_sha256=board_sha,
        mismatches=mismatches,
    )


def _identity_block(doc: Mapping[str, Any], *, label: str) -> tuple[dict[str, Any] | None, list[str]]:
    simulation = doc.get("simulation")
    if not isinstance(simulation, Mapping):
        return None, [f"{label} missing simulation block"]
    mismatches = []
    for field in PROFILE_IDENTITY_FIELDS:
        if field not in simulation or simulation[field] in (None, ""):
            mismatches.append(f"{label}.simulation missing {field}")
    if mismatches:
        return None, mismatches
    return dict(simulation), mismatches


def _check_simulation_profile_identity(
    manifest: Mapping[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    mismatches: list[str] = []
    simulation = manifest.get("simulation")
    if not isinstance(simulation, Mapping):
        return _check("simulation_profile_identity", False, mismatches=["manifest missing simulation block"])
    for field in PROFILE_IDENTITY_FIELDS:
        if field not in simulation or simulation[field] in (None, ""):
            mismatches.append(f"manifest.simulation missing {field}")

    config_entry, config_errors = _require_artifact(manifest, "simulation_config")
    mismatches.extend(config_errors)
    rollout_entry, rollout_errors = _require_artifact(manifest, "draw_count_rollout_decision")
    mismatches.extend(rollout_errors)
    sim_entry, sim_errors = _require_artifact(manifest, "simulation_manifest")
    mismatches.extend(sim_errors)
    report_entry, report_errors = _require_artifact(manifest, "release_report")
    mismatches.extend(report_errors)
    if mismatches:
        return _check("simulation_profile_identity", False, mismatches=mismatches)

    try:
        expected = resolve_simulation_profile_identity(
            profile_key=str(simulation["profile_key"]),
            rollout_path=root / rollout_entry["path"],
            simulation_config_path_arg=root / config_entry["path"],
        )
    except Exception as exc:
        return _check("simulation_profile_identity", False, error=str(exc))

    for field in PROFILE_IDENTITY_FIELDS:
        if str(simulation.get(field)) != str(expected.get(field)):
            mismatches.append(f"manifest.simulation.{field} mismatch")

    if sha256_file(root / rollout_entry["path"]) != expected["rollout_decision_hash"]:
        mismatches.append("sealed rollout decision artifact hash mismatch")
    if sha256_file(root / config_entry["path"]) != expected["simulation_config_hash"]:
        mismatches.append("sealed simulation_config artifact hash mismatch")

    sim_doc = json.loads((root / sim_entry["path"]).read_text(encoding="utf-8"))
    sim_block, sim_block_errors = _identity_block(
        {"simulation": {key: sim_doc.get(key) for key in PROFILE_IDENTITY_FIELDS}},
        label="simulation_manifest",
    )
    mismatches.extend(sim_block_errors)
    if sim_block:
        for field in PROFILE_IDENTITY_FIELDS:
            if str(sim_block.get(field)) != str(expected.get(field)):
                mismatches.append(f"simulation_manifest.{field} mismatch")

    report = json.loads((root / report_entry["path"]).read_text(encoding="utf-8"))
    report_block, report_block_errors = _identity_block(report, label="release_report")
    mismatches.extend(report_block_errors)
    if report_block:
        for field in PROFILE_IDENTITY_FIELDS:
            if str(report_block.get(field)) != str(expected.get(field)):
                mismatches.append(f"release_report.simulation.{field} mismatch")

    return _check("simulation_profile_identity", not mismatches, mismatches=mismatches)


def _check_ensemble_source_provenance(
    manifest: Mapping[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    season = int(manifest["bundle"]["season"])
    ensemble = manifest.get("ensemble")
    if not isinstance(ensemble, Mapping):
        return _check("ensemble_source_provenance", False, mismatches=["manifest missing ensemble block"])

    contract_entry, contract_errors = _require_artifact(manifest, "application_contract")
    v2_entry, v2_errors = _require_artifact(manifest, "v2_points")
    adp_entry, adp_errors = _require_artifact(manifest, "adp_source")
    weights_entry, weights_errors = _require_artifact(manifest, "ensemble_weights")
    mismatches = contract_errors + v2_errors + adp_errors + weights_errors
    if mismatches:
        return _check("ensemble_source_provenance", False, mismatches=mismatches)

    contract = json.loads((root / contract_entry["path"]).read_text(encoding="utf-8"))
    source_hashes = contract.get("source_hashes")
    if not isinstance(source_hashes, Mapping):
        return _check(
            "ensemble_source_provenance",
            False,
            mismatches=["application_contract missing source_hashes"],
        )

    required_keys = {
        f"v2_points_{season}": v2_entry,
        f"consensus_{season}": adp_entry,
        "ensemble_weights": weights_entry,
    }
    for key, entry in required_keys.items():
        if key not in source_hashes:
            mismatches.append(f"application_contract.source_hashes missing required key {key!r}")
            continue
        actual = sha256_file(root / entry["path"])
        expected = str(source_hashes[key])
        if actual != expected:
            mismatches.append(f"{key} hash != contract source_hashes")
        if key == f"v2_points_{season}" and actual != ensemble.get("v2_points_hash"):
            mismatches.append("ensemble.v2_points_hash mismatch")
        if key == f"consensus_{season}" and actual != ensemble.get("adp_source_hash"):
            mismatches.append("ensemble.adp_source_hash mismatch")
        if key == "ensemble_weights" and actual != ensemble.get("ensemble_weights_hash"):
            mismatches.append("ensemble.ensemble_weights_hash mismatch")

    contract_hash = str(contract.get("contract_hash") or "")
    if contract_hash != ensemble.get("contract_hash"):
        mismatches.append("ensemble.contract_hash mismatch")
    if contract_hash != manifest["application"]["contract_hash"]:
        mismatches.append("application.contract_hash mismatch")

    return _check("ensemble_source_provenance", not mismatches, mismatches=mismatches)


def validate_browser_artifacts_in_directory(
    manifest: Mapping[str, Any],
    public_dir: Path,
) -> dict[str, Any]:
    """Validate browser_consumed artifacts in one public namespace directory."""
    if not public_dir.exists():
        return _check("browser_artifact_completeness", False, error=f"public namespace missing: {public_dir}")

    browser_entries = [entry for entry in manifest["artifacts"] if entry.get("browser_consumed")]
    mismatches: list[str] = []
    listed_browser_paths = {entry["path"] for entry in browser_entries}
    listed_browser_paths.add(MANIFEST_FILENAME)
    present: set[str] = set()
    for path in public_dir.rglob("*"):
        if path.is_file():
            present.add(path.relative_to(public_dir).as_posix())
    unlisted = sorted(present - listed_browser_paths)
    missing = sorted(listed_browser_paths - present)
    if unlisted:
        mismatches.append(f"undeclared public files: {unlisted}")
    if missing:
        mismatches.append(f"missing public browser files: {missing}")
    for entry in browser_entries:
        path = public_dir / entry["path"]
        if not path.is_file():
            mismatches.append(f"missing browser artifact {entry['role']}")
            continue
        actual = sha256_file(path)
        size = path.stat().st_size
        if actual != entry["sha256"]:
            mismatches.append(f"browser artifact hash mismatch for {entry['role']}")
        if int(size) != int(entry["byte_size"]):
            mismatches.append(f"browser artifact byte_size mismatch for {entry['role']}")
    manifest_path = public_dir / MANIFEST_FILENAME
    if manifest_path.is_file():
        for entry in manifest["artifacts"]:
            if entry.get("browser_consumed"):
                continue
            path = public_dir / entry["path"]
            if path.exists():
                mismatches.append(f"non-browser artifact present in public namespace: {entry['path']}")
    return _check("browser_artifact_completeness", not mismatches, mismatches=mismatches)


def _check_git_provenance(manifest: Mapping[str, Any]) -> dict[str, Any]:
    git = manifest.get("git")
    if not isinstance(git, Mapping):
        return _check("git_provenance", False, error="manifest missing git block")
    try:
        verify_promotion_git_state(git)
        return _check("git_provenance", True, source_commit=git.get("source_commit"))
    except GitProvenanceError as exc:
        return _check("git_provenance", False, error=str(exc))


def _validate_loaded_manifest(
    manifest: Mapping[str, Any],
    *,
    season: int,
    namespace: str,
    manifest_hash: str | None,
    public_dir: Path | None,
    include_browser: bool,
    include_git: bool = True,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    root = bundle_root(season, namespace)

    eligible = promotion_eligible(manifest)
    checks.append(
        _check("promotion_eligible_schema", eligible, schema_version=manifest.get("schema_version"))
    )
    if not eligible:
        return _result(season, namespace, checks, manifest_hash, eligible=False)

    checks.append(_check_overlay_coverage_alignment(manifest, root=root))
    checks.append(_check_selected_board_hash_alignment(manifest, root=root))
    checks.append(_check_simulation_profile_identity(manifest, root=root))
    checks.append(_check_ensemble_source_provenance(manifest, root=root))
    if include_browser:
        target = public_dir or public_release_dir(namespace)
        checks.append(validate_browser_artifacts_in_directory(manifest, target))
    if include_git:
        checks.append(_check_git_provenance(manifest))
    return _result(season, namespace, checks, manifest_hash, eligible=True)


def validate_promotion_invariants(
    *,
    season: int,
    namespace: str,
    public_dir: Path | None = None,
    include_git: bool = True,
) -> dict[str, Any]:
    """Run all six promotion invariants. Any failure blocks promotion."""
    root = bundle_root(season, namespace)
    try:
        manifest, manifest_hash = load_sealed_manifest(root)
        verify_artifact_hashes(manifest, root=root)
    except ReleaseBundleError as exc:
        checks = [_check("bundle_readable", False, error=str(exc))]
        return _result(season, namespace, checks, None, eligible=False)

    return _validate_loaded_manifest(
        manifest,
        season=season,
        namespace=namespace,
        manifest_hash=manifest_hash,
        public_dir=public_dir,
        include_browser=True,
        include_git=include_git,
    )


def validate_sealed_promotion_invariants(
    *,
    season: int,
    namespace: str,
) -> dict[str, Any]:
    """Validate sealed-bundle invariants before public browser copies exist."""
    root = bundle_root(season, namespace)
    try:
        manifest, manifest_hash = load_sealed_manifest(root)
        verify_artifact_hashes(manifest, root=root)
    except ReleaseBundleError as exc:
        checks = [_check("bundle_readable", False, error=str(exc))]
        return _result(season, namespace, checks, None, eligible=False)

    return _validate_loaded_manifest(
        manifest,
        season=season,
        namespace=namespace,
        manifest_hash=manifest_hash,
        public_dir=None,
        include_browser=False,
    )


def _result(
    season: int,
    namespace: str,
    checks: list[dict[str, Any]],
    manifest_hash: str | None,
    *,
    eligible: bool,
) -> dict[str, Any]:
    invariant_checks = [
        check
        for check in checks
        if check["check"] in INVARIANT_NAMES or check["check"] == "promotion_eligible_schema"
    ]
    passed = eligible and all(check.get("passed") for check in invariant_checks)
    return {
        "schema_version": PROMOTION_INVARIANTS_VERSION,
        "season": int(season),
        "namespace": namespace,
        "manifest_sha256": manifest_hash,
        "promotion_eligible": eligible,
        "verdict": "pass" if passed else "fail",
        "checks": checks,
    }


def copy_and_validate_public_browser_artifacts(
    manifest: Mapping[str, Any],
    *,
    source_root: Path,
    manifest_sha256: str,
) -> Path:
    """Copy browser artifacts to a fresh public namespace and rehash copies."""
    import shutil

    namespace = manifest["bundle"]["namespace"]
    public = public_release_dir(namespace)
    temp = public.parent / f".{namespace}.promote_tmp"
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_root / MANIFEST_FILENAME, temp / MANIFEST_FILENAME)
    if sha256_bytes((temp / MANIFEST_FILENAME).read_bytes()) != manifest_sha256:
        shutil.rmtree(temp, ignore_errors=True)
        raise ReleaseBundleError("temporary public manifest copy hash mismatch")
    for entry in manifest["artifacts"]:
        if not entry.get("browser_consumed"):
            continue
        src = source_root / entry["path"]
        dest = temp / entry["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        if sha256_file(dest) != entry["sha256"]:
            shutil.rmtree(temp, ignore_errors=True)
            raise ReleaseBundleError(f"temporary public copy hash mismatch for {entry['role']}")
    browser_check = validate_browser_artifacts_in_directory(manifest, temp)
    if not browser_check.get("passed"):
        shutil.rmtree(temp, ignore_errors=True)
        raise ReleaseBundleError(f"browser artifact validation failed on temp copy: {browser_check}")
    if public.exists():
        shutil.rmtree(public)
    temp.rename(public)
    return public
