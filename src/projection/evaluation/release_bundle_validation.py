"""Validate a sealed release bundle and write release_bundle_validation_v1."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.projection.active_release import (
    ActiveReleaseError,
    derived_bundle_status,
    read_active_pointer,
)
from src.projection.release_bundle import (
    SCHEMA_VERSION,
    VALIDATION_FILENAME,
    ReleaseBundleError,
    bundle_root,
    canonical_dumps,
    load_sealed_manifest,
    player_id_set_hash,
    sha256_file,
    verify_artifact_hashes,
    verify_provenance_identities,
)


VALIDATION_SCHEMA_VERSION = "release_bundle_validation_v1"
VALIDATION_POLICY_VERSION = "release_bundle_validation_policy_v1"


def _check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), **details}


def validate_release_bundle(
    *,
    season: int,
    namespace: str,
    require_active: bool = False,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    root = bundle_root(season, namespace)
    manifest = None
    manifest_hash = None
    derived = "inactive"

    try:
        manifest, manifest_hash = load_sealed_manifest(root)
        checks.append(_check("sealed_manifest_canonical", True, manifest_sha256=manifest_hash))
    except ReleaseBundleError as exc:
        checks.append(_check("sealed_manifest_canonical", False, error=str(exc)))
        return _attestation(
            season=season,
            namespace=namespace,
            checks=checks,
            manifest_hash=None,
            derived_status="inactive",
            require_active=require_active,
        )

    if int(manifest["bundle"]["season"]) != int(season):
        checks.append(_check("season_matches", False, manifest=manifest["bundle"]["season"]))
    else:
        checks.append(_check("season_matches", True))
    if manifest["bundle"]["namespace"] != namespace:
        checks.append(
            _check(
                "namespace_matches",
                False,
                manifest=manifest["bundle"]["namespace"],
                requested=namespace,
            )
        )
    else:
        checks.append(_check("namespace_matches", True))

    try:
        verify_artifact_hashes(manifest, root=root)
        checks.append(_check("artifact_hashes", True))
    except ReleaseBundleError as exc:
        checks.append(_check("artifact_hashes", False, error=str(exc)))

    try:
        from src.projection.release_bundle import validate_artifact_enumeration

        validate_artifact_enumeration(manifest["artifacts"], root=root)
        checks.append(_check("artifact_enumeration", True))
    except ReleaseBundleError as exc:
        checks.append(_check("artifact_enumeration", False, error=str(exc)))

    try:
        verify_provenance_identities(manifest, expected=expected)
        checks.append(_check("provenance_identities", True))
    except ReleaseBundleError as exc:
        checks.append(_check("provenance_identities", False, error=str(exc)))

    if manifest["bundle"]["model_id"] != "accuracy_first_ensemble":
        checks.append(
            _check(
                "accuracy_first_model_id",
                False,
                model_id=manifest["bundle"]["model_id"],
            )
        )
    else:
        checks.append(_check("accuracy_first_model_id", True))

    selected = next(
        (entry for entry in manifest["artifacts"] if entry["role"] == "selected_board"),
        None,
    )
    board_match = bool(
        selected and selected["sha256"] == manifest["board"]["selected_board_file_hash"]
    )
    checks.append(_check("selected_board_hash_alignment", board_match))
    if selected:
        try:
            import pandas as pd
            from src.projection.inference.recenter import board_points_series
            from src.projection.release_bundle import selected_points_vector_hash

            board_df = pd.read_csv(root / selected["path"])
            vector_hash = selected_points_vector_hash(board_points_series(board_df))
            checks.append(
                _check(
                    "selected_points_vector_hash",
                    vector_hash == manifest["board"]["selected_points_vector_hash"],
                    expected=manifest["board"]["selected_points_vector_hash"],
                    actual=vector_hash,
                )
            )
        except Exception as exc:
            checks.append(_check("selected_points_vector_hash", False, error=str(exc)))

    players_entry = next(
        (entry for entry in manifest["artifacts"] if entry["role"] == "players"),
        None,
    )
    if players_entry:
        try:
            players_doc = json.loads((root / players_entry["path"]).read_text(encoding="utf-8"))
            overlay_ids = [str(row.get("player_id")) for row in players_doc.get("players") or []]
            pop_hash = player_id_set_hash(overlay_ids)
            checks.append(
                _check(
                    "simulated_player_population_hash",
                    pop_hash == manifest["overlay"]["simulated_player_population_hash"]
                    and len(set(overlay_ids)) == int(manifest["overlay"]["simulated_player_count"]),
                    expected=manifest["overlay"]["simulated_player_population_hash"],
                    actual=pop_hash,
                )
            )
            model_id = (players_doc.get("meta") or {}).get("model_id")
            checks.append(
                _check(
                    "exported_board_model_id",
                    model_id == "accuracy_first_ensemble",
                    model_id=model_id,
                )
            )
        except Exception as exc:
            checks.append(_check("simulated_player_population_hash", False, error=str(exc)))

    contract_entry = next(
        (entry for entry in manifest["artifacts"] if entry["role"] == "application_contract"),
        None,
    )
    if contract_entry:
        try:
            contract = json.loads((root / contract_entry["path"]).read_text(encoding="utf-8"))
            checks.append(
                _check(
                    "application_contract_hash",
                    str(contract.get("contract_hash")) == str(manifest["application"]["contract_hash"]),
                    manifest=manifest["application"]["contract_hash"],
                    file=contract.get("contract_hash"),
                )
            )
        except Exception as exc:
            checks.append(_check("application_contract_hash", False, error=str(exc)))

        # The v2 board comes from a separate repository. Pinning its hash in the
        # contract only means something if the sealed copy is checked against it
        # -- otherwise the input that carries most of the RB/WR mean could be
        # swapped and every other hash in the chain would still validate.
        #
        # Bundles sealed before v2 was pinned have neither the hash nor the
        # artifact; those skip rather than fail. A contract that DOES pin v2
        # while the bundle omits the copy is a real gap and fails.
        try:
            contract = json.loads((root / contract_entry["path"]).read_text(encoding="utf-8"))
            v2_expected = (contract.get("source_hashes") or {}).get(f"v2_points_{season}")
            v2_entry = next(
                (entry for entry in manifest["artifacts"] if entry["role"] == "v2_points"),
                None,
            )
            if v2_expected is None and v2_entry is None:
                pass  # sealed before v2 was pinned
            elif v2_entry is None:
                checks.append(
                    _check(
                        "v2_points_source_hash",
                        False,
                        error="contract pins v2_points but bundle has no v2_points artifact",
                    )
                )
            elif v2_expected is None:
                checks.append(
                    _check(
                        "v2_points_source_hash",
                        False,
                        error="bundle seals a v2_points artifact the contract does not pin",
                    )
                )
            else:
                actual = sha256_file(root / v2_entry["path"])
                checks.append(
                    _check(
                        "v2_points_source_hash",
                        actual == str(v2_expected),
                        contract=v2_expected,
                        bundle=actual,
                    )
                )
        except Exception as exc:
            checks.append(_check("v2_points_source_hash", False, error=str(exc)))

    sim_entry = next(
        (entry for entry in manifest["artifacts"] if entry["role"] == "simulation_manifest"),
        None,
    )
    if sim_entry:
        try:
            sim_doc = json.loads((root / sim_entry["path"]).read_text(encoding="utf-8"))
            sim_run = sim_doc.get("simulation_run_id") or sim_doc.get("canonical_projection_run_id")
            checks.append(
                _check(
                    "simulation_run_id_alignment",
                    str(sim_run) == str(manifest["runs"]["simulation_run_id"]),
                    manifest=manifest["runs"]["simulation_run_id"],
                    file=sim_run,
                )
            )
        except Exception as exc:
            checks.append(_check("simulation_run_id_alignment", False, error=str(exc)))

    if int(manifest["simulation"]["draw_count"]) != 10000:
        checks.append(
            _check(
                "publish_draw_count",
                False,
                draw_count=manifest["simulation"]["draw_count"],
            )
        )
    else:
        checks.append(_check("publish_draw_count", True))
    if manifest["simulation"]["profile"] != "publish":
        checks.append(
            _check("publish_profile", False, profile=manifest["simulation"]["profile"])
        )
    else:
        checks.append(_check("publish_profile", True))

    pointer_error = None
    pointer = None
    try:
        pointer = read_active_pointer(season)
    except ActiveReleaseError as exc:
        pointer_error = str(exc)
        checks.append(_check("active_pointer_well_formed", False, error=pointer_error))
    else:
        checks.append(_check("active_pointer_well_formed", True, present=pointer is not None))

    derived = derived_bundle_status(
        season=season,
        namespace=namespace,
        manifest_sha256=manifest_hash,
        pointer=pointer,
    )
    checks.append(_check("derived_status", True, status=derived))

    if require_active:
        checks.append(_check("require_active", derived == "active", derived_status=derived))
        if pointer_error:
            checks.append(_check("require_active_pointer_readable", False, error=pointer_error))

    attestation = _attestation(
        season=season,
        namespace=namespace,
        checks=checks,
        manifest_hash=manifest_hash,
        derived_status=derived,
        require_active=require_active,
        extra={
            "release_id": manifest["bundle"]["release_id"],
            "schema_version_validated": SCHEMA_VERSION,
        },
    )
    dest = root / VALIDATION_FILENAME
    dest.write_bytes(canonical_dumps(attestation))
    return attestation


def _attestation(
    *,
    season: int,
    namespace: str,
    checks: list[dict[str, Any]],
    manifest_hash: str | None,
    derived_status: str,
    require_active: bool,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    passed = all(check.get("passed") for check in checks)
    payload = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "validation_policy_version": VALIDATION_POLICY_VERSION,
        "season": int(season),
        "namespace": namespace,
        "manifest_sha256": manifest_hash,
        "derived_status": derived_status,
        "require_active": bool(require_active),
        "verdict": "pass" if passed else "fail",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
    }
    if extra:
        payload.update(extra)
    return payload
