"""Sealed release-bundle manifest: identity, hashed artifacts, no mutable status.

``release_bundle_manifest_v1`` describes one immutable namespace. Active vs
inactive is a property of the active pointer (and of validation output), never
a field rewritten on this document.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.projection.contracts import MODEL_V3_DIR, REPO_ROOT


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


SCHEMA_VERSION = "release_bundle_manifest_v1"
SCHEMA_VERSION_V2 = "release_bundle_manifest_v2"
BUNDLE_SCHEMA_VERSION = 1
BUNDLE_SCHEMA_VERSION_V2 = 2
PROMOTION_ELIGIBLE_SCHEMAS = frozenset({SCHEMA_VERSION_V2})
MANIFEST_FILENAME = "release_bundle_manifest.json"
VALIDATION_FILENAME = "release_bundle_validation.json"
SIDECAR_FILENAMES = frozenset({MANIFEST_FILENAME, VALIDATION_FILENAME})

FORBIDDEN_MUTABLE_KEYS = frozenset({"status", "active", "inactive"})

NAMESPACE_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")

ARTIFACT_FIELDS = (
    "role",
    "path",
    "sha256",
    "byte_size",
    "media_type",
    "required",
    "browser_consumed",
)

TREATMENT_KEYS = ("selected", "incumbent", "new_player_v1_only")

MEDIA_TYPES = {
    ".json": "application/json",
    ".csv": "text/csv",
    ".parquet": "application/vnd.apache.parquet",
    ".txt": "text/plain",
}

REQUIRED_IDENTITY_SECTIONS = (
    "bundle",
    "application",
    "runs",
    "board",
    "simulation",
    "overlay",
    "artifacts",
    "contract_treatments",
)

REQUIRED_IDENTITY_SECTIONS_V2 = REQUIRED_IDENTITY_SECTIONS + (
    "overlay_coverage",
    "ensemble",
    "git",
)


class ReleaseBundleError(ValueError):
    """Invalid sealed bundle, path, or identity."""


def bundle_root(season: int, namespace: str) -> Path:
    return (
        Path(MODEL_V3_DIR)
        / "release_bundles"
        / f"season={int(season)}"
        / f"namespace={namespace}"
    )


def public_release_dir(namespace: str) -> Path:
    return Path(REPO_ROOT) / "draft_assistant" / "data" / "releases" / namespace


def canonical_dumps(payload: Mapping[str, Any]) -> bytes:
    """Deterministic JSON bytes. Key order is sorted; the payload must not include its own hash."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_dumps(payload)).hexdigest()


def confine_namespace_path(rel: str) -> str:
    """Reject absolute, parent, empty, and Windows-escaped paths."""
    if not isinstance(rel, str) or not rel.strip():
        raise ReleaseBundleError("artifact path must be a non-empty string")
    path = rel.replace("\\", "/")
    if path != rel:
        raise ReleaseBundleError(f"artifact path must use posix separators: {rel!r}")
    if path.startswith("/") or path.startswith("~"):
        raise ReleaseBundleError(f"artifact path must be namespace-relative: {rel!r}")
    if ":" in path.split("/")[0]:
        raise ReleaseBundleError(f"artifact path must not be absolute: {rel!r}")
    parts = path.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ReleaseBundleError(f"unsafe artifact path: {rel!r}")
    if any(part in SIDECAR_FILENAMES for part in parts[:-1]):
        raise ReleaseBundleError(f"sidecar filenames cannot be used as directories: {rel!r}")
    return path


def media_type_for(path: str) -> str:
    suffix = Path(path).suffix.lower()
    return MEDIA_TYPES.get(suffix, "application/octet-stream")


def player_id_set_hash(player_ids: Iterable[str]) -> str:
    ordered = sorted({str(pid) for pid in player_ids})
    encoded = "\n".join(ordered).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def selected_points_vector_hash(rows: Mapping[str, float] | Iterable[tuple[str, float]]) -> str:
    if hasattr(rows, "items"):
        items = [(str(pid), float(value)) for pid, value in rows.items()]
    else:
        items = [(str(pid), float(value)) for pid, value in rows]
    items.sort(key=lambda item: item[0])
    payload = {"player_id": [pid for pid, _ in items], "points": [round(value, 10) for _, value in items]}
    return canonical_json_hash(payload)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def artifact_record(
    *,
    role: str,
    path: str,
    file_path: Path,
    required: bool = True,
    browser_consumed: bool = False,
    media_type: str | None = None,
) -> dict[str, Any]:
    confined = confine_namespace_path(path)
    if not file_path.is_file():
        raise ReleaseBundleError(f"missing artifact file for role {role!r}: {file_path}")
    digest = sha256_file(file_path)
    return {
        "role": str(role),
        "path": confined,
        "sha256": digest,
        "byte_size": int(file_path.stat().st_size),
        "media_type": media_type or media_type_for(confined),
        "required": bool(required),
        "browser_consumed": bool(browser_consumed),
    }


def _require_keys(section: Mapping[str, Any], keys: Iterable[str], *, name: str) -> None:
    missing = [key for key in keys if key not in section or section[key] in (None, "")]
    if missing:
        raise ReleaseBundleError(f"{name} missing required fields: {missing}")


def _assert_no_mutable_status(payload: Mapping[str, Any]) -> None:
    overlap = FORBIDDEN_MUTABLE_KEYS & set(payload)
    if overlap:
        raise ReleaseBundleError(
            f"immutable bundle manifest must not contain mutable status keys: {sorted(overlap)}"
        )


def validate_namespace(namespace: str) -> str:
    if not NAMESPACE_RE.match(str(namespace or "")):
        raise ReleaseBundleError(f"invalid artifact namespace: {namespace!r}")
    return str(namespace)


def normalize_artifact(entry: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in ARTIFACT_FIELDS if field not in entry]
    if missing:
        raise ReleaseBundleError(f"artifact missing fields {missing}: {entry}")
    path = confine_namespace_path(str(entry["path"]))
    role = str(entry["role"]).strip()
    if not role:
        raise ReleaseBundleError("artifact role must be non-empty")
    digest = str(entry["sha256"]).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ReleaseBundleError(f"artifact {role!r} sha256 is not a 64-char hex digest")
    size = int(entry["byte_size"])
    if size < 0:
        raise ReleaseBundleError(f"artifact {role!r} byte_size must be >= 0")
    return {
        "role": role,
        "path": path,
        "sha256": digest,
        "byte_size": size,
        "media_type": str(entry["media_type"]),
        "required": bool(entry["required"]),
        "browser_consumed": bool(entry["browser_consumed"]),
    }


def normalize_treatments(treatments: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in TREATMENT_KEYS:
        block = treatments.get(key)
        if not isinstance(block, Mapping):
            raise ReleaseBundleError(f"contract_treatments.{key} must be an object")
        _require_keys(block, ("count", "player_id_hash"), name=f"contract_treatments.{key}")
        count = int(block["count"])
        if count < 0:
            raise ReleaseBundleError(f"contract_treatments.{key}.count must be >= 0")
        digest = str(block["player_id_hash"]).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ReleaseBundleError(f"contract_treatments.{key}.player_id_hash is invalid")
        empty_hash = player_id_set_hash([])
        if count == 0 and digest != empty_hash:
            raise ReleaseBundleError(
                f"contract_treatments.{key} count is 0 but player_id_hash is not the empty-set hash"
            )
        out[key] = {"count": count, "player_id_hash": digest}
    extra = set(treatments) - set(TREATMENT_KEYS)
    if extra:
        raise ReleaseBundleError(f"unknown contract_treatments keys: {sorted(extra)}")
    return out


def enumerate_namespace_files(root: Path) -> list[str]:
    """Namespace-relative posix paths, excluding sealed sidecars."""
    if not root.exists():
        raise ReleaseBundleError(f"bundle root does not exist: {root}")
    found: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel in SIDECAR_FILENAMES:
            continue
        found.append(rel)
    return found


def validate_artifact_enumeration(
    artifacts: Iterable[Mapping[str, Any]],
    *,
    root: Path | None = None,
) -> list[dict[str, Any]]:
    normalized = [normalize_artifact(entry) for entry in artifacts]
    if not normalized:
        raise ReleaseBundleError("artifacts[] must not be empty")
    roles = [entry["role"] for entry in normalized]
    paths = [entry["path"] for entry in normalized]
    if len(roles) != len(set(roles)):
        dupes = sorted({role for role in roles if roles.count(role) > 1})
        raise ReleaseBundleError(f"duplicate artifact roles: {dupes}")
    if len(paths) != len(set(paths)):
        dupes = sorted({path for path in paths if paths.count(path) > 1})
        raise ReleaseBundleError(f"duplicate artifact paths: {dupes}")
    if root is not None:
        listed = set(paths)
        present = set(enumerate_namespace_files(root))
        missing = sorted(listed - present)
        unlisted = sorted(present - listed)
        if missing:
            raise ReleaseBundleError(f"manifest lists missing files: {missing}")
        if unlisted:
            raise ReleaseBundleError(f"namespace contains unlisted files: {unlisted}")
    return normalized


def promotion_eligible(manifest: Mapping[str, Any]) -> bool:
    schema = manifest.get("schema_version")
    if schema not in PROMOTION_ELIGIBLE_SCHEMAS:
        return False
    if manifest.get("promotion_eligible") is not True:
        return False
    return True


def _validate_sha256_field(value: Any, *, name: str) -> str:
    digest = str(value).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ReleaseBundleError(f"{name} is not a sha256 digest")
    return digest


def _validate_manifest_schema_v1(payload: Mapping[str, Any]) -> dict[str, Any]:
    for section in REQUIRED_IDENTITY_SECTIONS:
        if section not in payload:
            raise ReleaseBundleError(f"manifest missing section {section!r}")

    bundle = payload["bundle"]
    _require_keys(
        bundle,
        ("season", "namespace", "release_id", "created_at", "model_id", "schema_version"),
        name="bundle",
    )
    if int(bundle["schema_version"]) != BUNDLE_SCHEMA_VERSION:
        raise ReleaseBundleError("bundle.schema_version must be 1")
    validate_namespace(str(bundle["namespace"]))
    if int(bundle["season"]) < 2000:
        raise ReleaseBundleError("bundle.season is implausible")

    application = payload["application"]
    _require_keys(application, ("contract_version", "contract_hash"), name="application")
    _validate_sha256_field(application["contract_hash"], name="application.contract_hash")

    runs = payload["runs"]
    _require_keys(runs, ("projection_run_id", "simulation_run_id"), name="runs")

    board = payload["board"]
    _require_keys(
        board,
        ("selected_board_file_hash", "selected_points_vector_hash"),
        name="board",
    )
    _validate_sha256_field(board["selected_board_file_hash"], name="board.selected_board_file_hash")
    _validate_sha256_field(board["selected_points_vector_hash"], name="board.selected_points_vector_hash")
    if board.get("selected_board_sha256") is not None:
        if board["selected_board_sha256"] != board["selected_board_file_hash"]:
            raise ReleaseBundleError(
                "board.selected_board_sha256 must equal board.selected_board_file_hash"
            )

    simulation = payload["simulation"]
    _require_keys(
        simulation,
        ("profile", "draw_count", "configuration_hash", "calibration_hashes", "joint_donor_hash"),
        name="simulation",
    )
    if int(simulation["draw_count"]) <= 0:
        raise ReleaseBundleError("simulation.draw_count must be positive")
    if not isinstance(simulation["calibration_hashes"], Mapping):
        raise ReleaseBundleError("simulation.calibration_hashes must be an object")

    overlay = payload["overlay"]
    _require_keys(
        overlay,
        ("simulated_player_population_hash", "simulated_player_count"),
        name="overlay",
    )
    if int(overlay["simulated_player_count"]) < 0:
        raise ReleaseBundleError("overlay.simulated_player_count must be >= 0")

    artifacts = validate_artifact_enumeration(payload["artifacts"])
    treatments = normalize_treatments(payload["contract_treatments"])

    normalized = json.loads(canonical_dumps(payload).decode("utf-8"))
    normalized["artifacts"] = artifacts
    normalized["contract_treatments"] = treatments
    _assert_no_mutable_status(normalized)
    return normalized


def _validate_manifest_schema_v2(payload: Mapping[str, Any]) -> dict[str, Any]:
    for section in REQUIRED_IDENTITY_SECTIONS_V2:
        if section not in payload:
            raise ReleaseBundleError(f"manifest missing section {section!r}")
    if payload.get("promotion_eligible") is not True:
        raise ReleaseBundleError("v2 manifest must set promotion_eligible=true")

    bundle = payload["bundle"]
    _require_keys(
        bundle,
        ("season", "namespace", "release_id", "created_at", "model_id", "schema_version"),
        name="bundle",
    )
    if int(bundle["schema_version"]) != BUNDLE_SCHEMA_VERSION_V2:
        raise ReleaseBundleError("bundle.schema_version must be 2")
    validate_namespace(str(bundle["namespace"]))
    if int(bundle["season"]) < 2000:
        raise ReleaseBundleError("bundle.season is implausible")

    application = payload["application"]
    _require_keys(application, ("contract_version", "contract_hash"), name="application")
    _validate_sha256_field(application["contract_hash"], name="application.contract_hash")

    runs = payload["runs"]
    _require_keys(runs, ("projection_run_id", "simulation_run_id"), name="runs")

    board = payload["board"]
    _require_keys(
        board,
        ("selected_board_sha256", "selected_board_file_hash", "selected_points_vector_hash"),
        name="board",
    )
    board_sha = _validate_sha256_field(board["selected_board_sha256"], name="board.selected_board_sha256")
    file_sha = _validate_sha256_field(board["selected_board_file_hash"], name="board.selected_board_file_hash")
    if board_sha != file_sha:
        raise ReleaseBundleError("board.selected_board_sha256 must equal board.selected_board_file_hash")
    _validate_sha256_field(board["selected_points_vector_hash"], name="board.selected_points_vector_hash")

    simulation = payload["simulation"]
    _require_keys(
        simulation,
        (
            "profile_key",
            "profile_label",
            "profile",
            "draw_count",
            "chunk_size",
            "configuration_hash",
            "policy_hash",
            "calibration_hashes",
            "joint_donor_hash",
        ),
        name="simulation",
    )
    if int(simulation["draw_count"]) <= 0:
        raise ReleaseBundleError("simulation.draw_count must be positive")
    if int(simulation["chunk_size"]) <= 0:
        raise ReleaseBundleError("simulation.chunk_size must be positive")
    if not isinstance(simulation["calibration_hashes"], Mapping):
        raise ReleaseBundleError("simulation.calibration_hashes must be an object")
    _validate_sha256_field(simulation["configuration_hash"], name="simulation.configuration_hash")
    _validate_sha256_field(simulation["policy_hash"], name="simulation.policy_hash")

    overlay = payload["overlay"]
    _require_keys(
        overlay,
        ("simulated_player_population_hash", "simulated_player_count"),
        name="overlay",
    )
    if int(overlay["simulated_player_count"]) < 0:
        raise ReleaseBundleError("overlay.simulated_player_count must be >= 0")

    overlay_coverage = payload["overlay_coverage"]
    _require_keys(overlay_coverage, ("total_players", "fields"), name="overlay_coverage")
    if not isinstance(overlay_coverage["fields"], Mapping):
        raise ReleaseBundleError("overlay_coverage.fields must be an object")

    ensemble = payload["ensemble"]
    _require_keys(
        ensemble,
        ("contract_hash", "ensemble_weights_hash", "v2_points_hash", "adp_source_hash"),
        name="ensemble",
    )
    for key in ("contract_hash", "ensemble_weights_hash", "v2_points_hash", "adp_source_hash"):
        _validate_sha256_field(ensemble[key], name=f"ensemble.{key}")

    git = payload["git"]
    _require_keys(git, ("source_commit", "source_dirty"), name="git")
    if git.get("source_dirty") is not False:
        raise ReleaseBundleError("git.source_dirty must be false")
    if not str(git.get("source_commit") or "").strip():
        raise ReleaseBundleError("git.source_commit must be non-empty")

    artifacts = validate_artifact_enumeration(payload["artifacts"])
    treatments = normalize_treatments(payload["contract_treatments"])

    normalized = json.loads(canonical_dumps(payload).decode("utf-8"))
    normalized["artifacts"] = artifacts
    normalized["contract_treatments"] = treatments
    _assert_no_mutable_status(normalized)
    return normalized


def validate_manifest_schema(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ReleaseBundleError("manifest must be a JSON object")
    _assert_no_mutable_status(payload)
    schema = payload.get("schema_version")
    if schema == SCHEMA_VERSION:
        return _validate_manifest_schema_v1(payload)
    if schema == SCHEMA_VERSION_V2:
        return _validate_manifest_schema_v2(payload)
    raise ReleaseBundleError(f"unsupported manifest schema_version: {schema!r}")


def build_manifest(
    *,
    season: int,
    namespace: str,
    release_id: str,
    model_id: str,
    created_at: str | None = None,
    application: Mapping[str, Any],
    runs: Mapping[str, Any],
    board: Mapping[str, Any],
    simulation: Mapping[str, Any],
    overlay: Mapping[str, Any],
    artifacts: Iterable[Mapping[str, Any]],
    contract_treatments: Mapping[str, Any],
    overlay_coverage: Mapping[str, Any] | None = None,
    ensemble: Mapping[str, Any] | None = None,
    git: Mapping[str, Any] | None = None,
    schema_version: str = SCHEMA_VERSION,
) -> dict[str, Any]:
    board_payload = dict(board)
    if schema_version == SCHEMA_VERSION_V2:
        if "selected_board_sha256" not in board_payload:
            board_payload["selected_board_sha256"] = board_payload.get("selected_board_file_hash")
        if board_payload.get("selected_board_file_hash") != board_payload.get("selected_board_sha256"):
            raise ReleaseBundleError("board hashes must agree before sealing v2 manifest")
        bundle_schema_version = BUNDLE_SCHEMA_VERSION_V2
        if overlay_coverage is None or ensemble is None or git is None:
            raise ReleaseBundleError("v2 manifest requires overlay_coverage, ensemble, and git sections")
    else:
        bundle_schema_version = BUNDLE_SCHEMA_VERSION

    payload: dict[str, Any] = {
        "schema_version": schema_version,
        "bundle": {
            "season": int(season),
            "namespace": validate_namespace(namespace),
            "release_id": str(release_id),
            "created_at": created_at or datetime.now(timezone.utc).isoformat(),
            "model_id": str(model_id),
            "schema_version": bundle_schema_version,
        },
        "application": dict(application),
        "runs": dict(runs),
        "board": board_payload,
        "simulation": dict(simulation),
        "overlay": dict(overlay),
        "artifacts": [normalize_artifact(entry) for entry in artifacts],
        "contract_treatments": normalize_treatments(contract_treatments),
    }
    if schema_version == SCHEMA_VERSION_V2:
        payload["promotion_eligible"] = True
        payload["overlay_coverage"] = dict(overlay_coverage or {})
        payload["ensemble"] = dict(ensemble or {})
        payload["git"] = dict(git or {})
    return validate_manifest_schema(payload)


def seal_manifest(payload: Mapping[str, Any], *, root: Path) -> tuple[dict[str, Any], str]:
    """Validate against staged files, write canonical bytes, return (manifest, hash).

    The hash is of the canonical manifest body and is stored outside the file
    (attestation + active pointer) so hashing cannot recurse into itself.
    """
    manifest = validate_manifest_schema(payload)
    validate_artifact_enumeration(manifest["artifacts"], root=root)
    verify_artifact_hashes(manifest, root=root)
    encoded = canonical_dumps(manifest)
    digest = hashlib.sha256(encoded).hexdigest()
    dest = root / MANIFEST_FILENAME
    dest.write_bytes(encoded)
    return manifest, digest


def verify_artifact_hashes(manifest: Mapping[str, Any], *, root: Path) -> None:
    for entry in manifest["artifacts"]:
        path = root / entry["path"]
        if not path.is_file():
            raise ReleaseBundleError(f"missing artifact {entry['role']}: {entry['path']}")
        actual = sha256_file(path)
        size = path.stat().st_size
        if actual != entry["sha256"]:
            raise ReleaseBundleError(
                f"hash mismatch for role {entry['role']!r} path {entry['path']}: "
                f"manifest={entry['sha256']} file={actual}"
            )
        if int(size) != int(entry["byte_size"]):
            raise ReleaseBundleError(
                f"byte_size mismatch for role {entry['role']!r}: "
                f"manifest={entry['byte_size']} file={size}"
            )


def load_sealed_manifest(root: Path) -> tuple[dict[str, Any], str]:
    path = root / MANIFEST_FILENAME
    if not path.is_file():
        raise ReleaseBundleError(f"sealed manifest missing: {path}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ReleaseBundleError(f"sealed manifest is not valid JSON: {exc}") from exc
    manifest = validate_manifest_schema(payload)
    if canonical_dumps(manifest) != raw:
        raise ReleaseBundleError("sealed manifest is not in canonical deterministic form")
    return manifest, digest


def verify_provenance_identities(
    manifest: Mapping[str, Any],
    *,
    expected: Mapping[str, Any] | None = None,
) -> None:
    """Reject inconsistent identity fields across sections and optional expected values."""
    artifacts_by_role = {entry["role"]: entry for entry in manifest["artifacts"]}
    board_hash = manifest["board"]["selected_board_file_hash"]
    selected = artifacts_by_role.get("selected_board")
    if selected and selected["sha256"] != board_hash:
        raise ReleaseBundleError(
            "board.selected_board_file_hash does not match selected_board artifact sha256"
        )
    if expected:
        pairs = (
            ("season", manifest["bundle"]["season"], expected.get("season")),
            ("namespace", manifest["bundle"]["namespace"], expected.get("namespace")),
            ("release_id", manifest["bundle"]["release_id"], expected.get("release_id")),
            ("projection_run_id", manifest["runs"]["projection_run_id"], expected.get("projection_run_id")),
            ("simulation_run_id", manifest["runs"]["simulation_run_id"], expected.get("simulation_run_id")),
            (
                "application.contract_hash",
                manifest["application"]["contract_hash"],
                expected.get("contract_hash"),
            ),
            (
                "overlay.simulated_player_population_hash",
                manifest["overlay"]["simulated_player_population_hash"],
                expected.get("simulated_player_population_hash"),
            ),
        )
        mismatches = [
            f"{name}: manifest={actual!r} expected={want!r}"
            for name, actual, want in pairs
            if want is not None and str(actual) != str(want)
        ]
        if mismatches:
            raise ReleaseBundleError("inconsistent provenance identities: " + "; ".join(mismatches))


def treatment_block(player_ids: Iterable[str]) -> dict[str, Any]:
    ids = [str(pid) for pid in player_ids]
    return {"count": len(set(ids)), "player_id_hash": player_id_set_hash(ids)}
