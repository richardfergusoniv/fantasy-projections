"""Frozen accuracy-first application contract: apply, never refit.

The sealed contract carries serialized market curves, position arms, source
hashes, the frozen eligibility set, and the frozen external-input universe.
Applying it to a fresh v1 board must not fit weights or reconstruct historical
ADP curves. Incumbent v1/v2 blending is a declared treatment for in-universe
non-selected rows, not a fallback for missing contract inputs.

v1-only is a local classification for a player absent from both the frozen
external-input universe and the frozen eligibility set. It cannot satisfy
missing inputs for a contract-eligible player.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.projection.contracts import OUTPUT_DIR
from src.projection.evaluation.accuracy_first import (
    POSITIONS,
    TOP_ADP,
    canonical_json_hash,
    incumbent_points,
    sha256_file,
)
from src.projection.release_bundle import player_id_set_hash, treatment_block


CONTRACT_SCHEMA_VERSION = "accuracy_application_contract_v1"
CONTRACT_VERSION = "accuracy_first_2026_v1"
MODEL_ID = "accuracy_first_ensemble"

ARM_REQUIRED_COLUMNS = {
    "incumbent": ("v1_pred", "v2_pred"),
    "market_no_v3": ("v1_pred", "v2_pred", "adp_points"),
    "full": ("v1_pred", "v2_pred", "v3_p50", "adp_points"),
    "model_only": ("v1_pred", "v2_pred", "v3_p50"),
}

ACCURACY_FIRST_DIR = Path(OUTPUT_DIR) / "accuracy_first_2026"
DEFAULT_CONTRACT_PATH = ACCURACY_FIRST_DIR / "application_contract.json"
DEFAULT_WEIGHTS_PATH = ACCURACY_FIRST_DIR / "ensemble_weights.json"
DEFAULT_FREEZE_PATH = ACCURACY_FIRST_DIR / "freeze_manifest.json"


class ApplicationContractError(ValueError):
    """Frozen application contract is invalid or cannot be applied."""


def serialize_isotonic(curve: IsotonicRegression) -> dict[str, Any]:
    return {
        "x_thresholds": [float(value) for value in curve.X_thresholds_],
        "y_thresholds": [float(value) for value in curve.y_thresholds_],
        "increasing": bool(getattr(curve, "increasing_", False)),
        "y_min": 0.0,
        "out_of_bounds": "clip",
    }


def apply_serialized_curve(adp: np.ndarray, spec: Mapping[str, Any]) -> np.ndarray:
    xp = np.asarray(spec["x_thresholds"], dtype=float)
    fp = np.asarray(spec["y_thresholds"], dtype=float)
    values = np.asarray(adp, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    if valid.any():
        out[valid] = np.interp(values[valid], xp, fp)
        if spec.get("y_min") is not None:
            out[valid] = np.maximum(out[valid], float(spec["y_min"]))
    return out


def apply_serialized_market_curves(frame: pd.DataFrame, curves: Mapping[str, Any]) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    adp = pd.to_numeric(frame.get("adp"), errors="coerce")
    position = frame["position"].astype(str)
    for pos, spec in curves.items():
        mask = position.eq(pos) & adp.notna()
        if mask.any():
            out.loc[mask] = apply_serialized_curve(adp.loc[mask].to_numpy(), spec)
    return out


def contract_hash(payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("contract_hash", None)
    return canonical_json_hash(body)


def _id_set(values: Iterable[str]) -> list[str]:
    return sorted({str(value) for value in values})


def validate_application_contract(
    payload: Mapping[str, Any],
    *,
    source_files: Mapping[str, Path] | None = None,
    require_source_files: bool = False,
) -> dict[str, Any]:
    if payload.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise ApplicationContractError(
            f"unsupported application contract schema: {payload.get('schema_version')!r}"
        )
    required = (
        "contract_version",
        "model_id",
        "source_hashes",
        "positions",
        "transforms",
        "eligibility",
        "external_input_universe",
        "reference_fixture",
        "incumbent_fallback",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ApplicationContractError(f"application contract missing fields: {missing}")
    if payload.get("incumbent_fallback") is not False:
        raise ApplicationContractError("application contract must disable incumbent fallback")
    if payload.get("model_id") != MODEL_ID:
        raise ApplicationContractError("application contract model_id must be accuracy_first_ensemble")
    transforms = payload["transforms"]
    curves = (transforms.get("market_curves") or {})
    if set(curves) != set(POSITIONS):
        raise ApplicationContractError("serialized market curves must cover QB/RB/WR/TE")
    eligibility_ids = _id_set(payload["eligibility"]["player_ids"])
    universe_ids = _id_set(payload["external_input_universe"]["player_ids"])
    if payload["eligibility"].get("player_id_hash") != player_id_set_hash(eligibility_ids):
        raise ApplicationContractError("eligibility player_id_hash does not match player_ids")
    if payload["external_input_universe"].get("player_id_hash") != player_id_set_hash(universe_ids):
        raise ApplicationContractError("external_input_universe player_id_hash does not match player_ids")
    expected_hash = payload.get("contract_hash")
    actual_hash = contract_hash(payload)
    if expected_hash and expected_hash != actual_hash:
        raise ApplicationContractError("application contract_hash does not match payload")
    if require_source_files or source_files:
        files = source_files or {}
        hashes = payload.get("source_hashes") or {}
        for name, expected in hashes.items():
            path = files.get(name)
            if path is None:
                if require_source_files:
                    raise ApplicationContractError(f"missing source file for hashed input {name!r}")
                continue
            if not Path(path).exists():
                raise ApplicationContractError(f"source file missing for {name}: {path}")
            actual = sha256_file(path)
            if actual != expected:
                raise ApplicationContractError(
                    f"source hash mismatch for {name}: expected {expected} actual {actual}"
                )
    replay_reference_fixture(payload)
    return dict(payload)


def replay_reference_fixture(contract: Mapping[str, Any]) -> None:
    fixture = contract["reference_fixture"]
    row = pd.DataFrame(
        [
            {
                "player_id": str(fixture["player_id"]),
                "position": str(fixture["position"]),
                "v1_pred": float(fixture["inputs"]["v1_pred"]),
                "v2_pred": fixture["inputs"].get("v2_pred"),
                "v3_p50": fixture["inputs"].get("v3_p50"),
                "adp": fixture["inputs"].get("adp"),
            }
        ]
    )
    applied, treatments = apply_application_contract(
        row,
        contract,
        v2_by_id={str(fixture["player_id"]): fixture["inputs"].get("v2_pred")},
        adp_by_id={str(fixture["player_id"]): fixture["inputs"].get("adp")},
        validate=False,
    )
    predicted = float(applied.loc[0, "fantasy_pts_season"])
    expected = float(fixture["expected_points"])
    if abs(predicted - expected) > 1e-9:
        raise ApplicationContractError(
            f"reference fixture replay failed: predicted={predicted} expected={expected}"
        )
    expected_treatment = str(fixture["expected_treatment"])
    actual_treatment = str(applied.loc[0, "contract_treatment"])
    if actual_treatment != expected_treatment:
        raise ApplicationContractError(
            f"reference fixture treatment {actual_treatment!r} != {expected_treatment!r}"
        )
    if expected_treatment == "new_player_v1_only" and treatments["new_player_v1_only"]["count"] != 1:
        raise ApplicationContractError("reference fixture did not record v1-only treatment")


def _arm_for_position(contract: Mapping[str, Any], position: str) -> dict[str, Any]:
    spec = (contract.get("positions") or {}).get(position)
    if not spec:
        raise ApplicationContractError(f"no frozen position spec for {position}")
    return spec


def _missing_required(row: pd.Series, columns: tuple[str, ...]) -> list[str]:
    missing: list[str] = []
    for column in columns:
        value = row.get(column)
        if value is None or (isinstance(value, float) and not np.isfinite(value)) or pd.isna(value):
            missing.append(column)
    return missing


def apply_application_contract(
    v1_board: pd.DataFrame,
    contract: Mapping[str, Any],
    *,
    v2_by_id: Mapping[str, float] | None = None,
    adp_by_id: Mapping[str, float] | None = None,
    v3_by_id: Mapping[str, float] | None = None,
    validate: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply frozen transforms to a fresh v1 board. Never fits. Never fills eligible holes."""
    if validate:
        contract = validate_application_contract(contract)
    eligibility = set(_id_set(contract["eligibility"]["player_ids"]))
    universe = set(_id_set(contract["external_input_universe"]["player_ids"]))
    v2_by_id = {str(key): value for key, value in (v2_by_id or {}).items()}
    adp_by_id = {str(key): value for key, value in (adp_by_id or {}).items()}
    v3_by_id = {str(key): value for key, value in (v3_by_id or {}).items()}

    out = v1_board.copy()
    out["player_id"] = out["player_id"].astype(str)
    if "v1_pred" not in out.columns:
        out["v1_pred"] = pd.to_numeric(out["fantasy_pts_season"], errors="coerce")
    else:
        out["v1_pred"] = pd.to_numeric(out["v1_pred"], errors="coerce")
    out["v2_pred"] = out["player_id"].map(v2_by_id)
    if "adp" not in out.columns:
        out["adp"] = out["player_id"].map(adp_by_id)
    else:
        out["adp"] = pd.to_numeric(out["adp"], errors="coerce")
        missing_adp = out["adp"].isna()
        out.loc[missing_adp, "adp"] = out.loc[missing_adp, "player_id"].map(adp_by_id)
    out["v3_p50"] = out["player_id"].map(v3_by_id) if v3_by_id else np.nan
    out["adp_points"] = apply_serialized_market_curves(out, contract["transforms"]["market_curves"])
    return _apply_rows(out, contract, eligibility, universe)


def _incumbent_weight_map(contract: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    stored = contract.get("incumbent_weights")
    if stored:
        return {pos: dict(weights) for pos, weights in stored.items()}
    weights: dict[str, dict[str, float]] = {}
    for position in POSITIONS:
        spec = contract["positions"][position]
        if spec["arm"] == "incumbent":
            weights[position] = dict(spec["weights"])
        else:
            weights[position] = {"v1_pred": 1.0, "v2_pred": 0.0}
    return weights


def _apply_rows(
    out: pd.DataFrame,
    contract: Mapping[str, Any],
    eligibility: set[str],
    universe: set[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    incumbent_weights = _incumbent_weight_map(contract)
    treatments = {"selected": [], "incumbent": [], "new_player_v1_only": []}
    points: list[float] = []
    arms: list[str] = []
    applied_flags: list[bool] = []
    labels: list[str] = []
    fatal: list[str] = []

    for _, row in out.iterrows():
        player_id = str(row["player_id"])
        position = str(row["position"])
        spec = _arm_for_position(contract, position)
        arm = str(spec["arm"])
        in_eligibility = player_id in eligibility
        in_universe = player_id in universe

        if in_eligibility:
            required = ARM_REQUIRED_COLUMNS.get(arm)
            if required is None:
                fatal.append(f"{player_id}: unknown arm {arm!r}")
                _append_row(points, arms, applied_flags, labels, row["v1_pred"], arm, False, "selected")
                continue
            missing = _missing_required(row, required)
            if missing:
                fatal.append(
                    f"{player_id}: missing contract inputs {missing} for eligible arm {arm}"
                )
                _append_row(points, arms, applied_flags, labels, row["v1_pred"], arm, False, "selected")
                continue
            total = sum(float(weight) * float(row[column]) for column, weight in spec["weights"].items())
            if arm == "incumbent":
                treatments["incumbent"].append(player_id)
                _append_row(points, arms, applied_flags, labels, total, arm, False, "incumbent")
            else:
                treatments["selected"].append(player_id)
                _append_row(points, arms, applied_flags, labels, total, arm, True, "selected")
            continue

        if not in_universe:
            if pd.isna(row["v1_pred"]):
                fatal.append(f"{player_id}: new player is missing v1_pred")
                _append_row(points, arms, applied_flags, labels, float("nan"), "new_player_v1_only", False, "new_player_v1_only")
            else:
                treatments["new_player_v1_only"].append(player_id)
                _append_row(
                    points, arms, applied_flags, labels,
                    float(row["v1_pred"]), "new_player_v1_only", False, "new_player_v1_only",
                )
            continue

        missing = _missing_required(row, ("v1_pred",))
        if missing:
            fatal.append(f"{player_id}: in-universe player missing {missing}")
            _append_row(points, arms, applied_flags, labels, float("nan"), "incumbent", False, "incumbent")
            continue
        work = pd.DataFrame([{"position": position, "v1_pred": row["v1_pred"], "v2_pred": row["v2_pred"]}])
        if pd.isna(work.loc[0, "v2_pred"]):
            work.loc[0, "v2_pred"] = work.loc[0, "v1_pred"]
        pred = float(incumbent_points(work, incumbent_weights).iloc[0])
        treatments["incumbent"].append(player_id)
        _append_row(points, arms, applied_flags, labels, pred, "incumbent", False, "incumbent")

    if fatal:
        raise ApplicationContractError("application failed:\n  " + "\n  ".join(fatal))

    result = out.copy()
    result["fantasy_pts_season"] = points
    if "projected_games" in result.columns:
        games = pd.to_numeric(result["projected_games"], errors="coerce").replace(0, np.nan)
    else:
        games = pd.Series(17.0, index=result.index)
    result["fantasy_pts"] = result["fantasy_pts_season"] / games
    result["fantasy_pts"] = result["fantasy_pts"].fillna(result["fantasy_pts_season"] / 17.0)
    result["accuracy_ensemble_pred"] = points
    result["accuracy_ensemble_applied"] = applied_flags
    result["accuracy_ensemble_arm"] = arms
    result["contract_treatment"] = labels
    treatment_payload = {key: treatment_block(values) for key, values in treatments.items()}
    treatment_payload["player_ids"] = {key: sorted(set(values)) for key, values in treatments.items()}
    return result, treatment_payload


def _append_row(points, arms, applied_flags, labels, value, arm, applied, label) -> None:
    points.append(float(value) if value is not None and pd.notna(value) else float("nan"))
    arms.append(arm)
    applied_flags.append(applied)
    labels.append(label)


def build_application_contract(
    *,
    positions: Mapping[str, Any],
    market_curves: Mapping[str, Any],
    eligibility_ids: Iterable[str],
    universe_ids: Iterable[str],
    reference_fixture: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    incumbent_weights: Mapping[str, Mapping[str, float]] | None = None,
    contract_version: str = CONTRACT_VERSION,
    top_adp: float = TOP_ADP,
) -> dict[str, Any]:
    eligibility = _id_set(eligibility_ids)
    universe = _id_set(universe_ids)
    payload = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "contract_version": contract_version,
        "model_id": MODEL_ID,
        "source_hashes": dict(source_hashes),
        "positions": json.loads(json.dumps(positions)),
        "incumbent_weights": json.loads(json.dumps(incumbent_weights or {})),
        "transforms": {
            "market_curves": json.loads(json.dumps(market_curves)),
            "top_adp": float(top_adp),
        },
        "eligibility": {
            "player_ids": eligibility,
            "player_id_hash": player_id_set_hash(eligibility),
        },
        "external_input_universe": {
            "player_ids": universe,
            "player_id_hash": player_id_set_hash(universe),
        },
        "reference_fixture": json.loads(json.dumps(reference_fixture)),
        "incumbent_fallback": False,
    }
    payload["contract_hash"] = contract_hash(payload)
    return validate_application_contract(payload)


def load_application_contract(path: Path | None = None) -> dict[str, Any]:
    contract_path = path or DEFAULT_CONTRACT_PATH
    payload = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    return validate_application_contract(payload)


def freeze_application_contract_from_artifacts(
    *,
    weights_path: Path | None = None,
    freeze_path: Path | None = None,
    consensus_dir: Path | None = None,
    incumbent_weights_path: Path | None = None,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """Serialize frozen transforms from the existing accuracy-first freeze.

    This path may fit curves once from hash-pinned historical snapshots. The
    resulting contract is what later publishes apply without refitting.
    """
    from scripts.evaluate_accuracy_first_ensemble import (
        CONSENSUS_DIR,
        INCUMBENT_WEIGHTS_PATH,
        _market_history,
    )
    from src.projection.evaluation.accuracy_first import fit_market_curves, load_consensus_snapshot

    weights_path = weights_path or DEFAULT_WEIGHTS_PATH
    freeze_path = freeze_path or DEFAULT_FREEZE_PATH
    weights = json.loads(Path(weights_path).read_text(encoding="utf-8"))
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    freeze_files = freeze.get("files") or {}
    actual_weights = sha256_file(weights_path)
    expected_weights = freeze_files.get("ensemble_weights.json")
    if expected_weights and actual_weights != expected_weights:
        raise ApplicationContractError("ensemble_weights.json does not match freeze_manifest")

    history, _ = _market_history(2026)
    curves = fit_market_curves(history)
    serialized = {position: serialize_isotonic(curves[position]) for position in POSITIONS}

    consensus_root = Path(consensus_dir or CONSENSUS_DIR)
    snapshot_path = consensus_root / "consensus_2026.json"
    consensus, _ = load_consensus_snapshot(snapshot_path, expected_season=2026)
    eligible = consensus[pd.to_numeric(consensus["adp"], errors="coerce").le(TOP_ADP)]
    eligibility_ids = eligible["player_id"].astype(str).tolist()

    v1_path = Path(OUTPUT_DIR) / "fantasy_points_2026.csv"
    v2_path = Path(OUTPUT_DIR) / "model_v2" / "fantasy_points_2026.csv"
    universe: set[str] = set(consensus["player_id"].astype(str))
    if v1_path.exists():
        universe.update(pd.read_csv(v1_path)["player_id"].astype(str))
    if v2_path.exists():
        universe.update(pd.read_csv(v2_path)["player_id"].astype(str))

    source_hashes = dict(weights.get("source_hashes") or {})
    source_hashes["ensemble_weights"] = actual_weights
    source_hashes["freeze_manifest"] = freeze.get("manifest_hash") or sha256_file(freeze_path)
    source_hashes["consensus_2026"] = sha256_file(snapshot_path)
    # v2 is the largest single input to the published WR mean (0.55) and is
    # produced by a SEPARATE repository, synced in as a CSV. Every other input
    # to the selected board was already hash-pinned; this one was not, so the
    # sealed bundle could not detect a swap of the file that most determines
    # RB/WR ranking. inputs_from_frozen_sources reads it unconditionally, so
    # its absence is an error rather than a skipped hash.
    if not v2_path.exists():
        raise ApplicationContractError(f"v2 points file missing: {v2_path}")
    source_hashes["v2_points_2026"] = sha256_file(v2_path)

    incumbent_path = Path(incumbent_weights_path or INCUMBENT_WEIGHTS_PATH)
    incumbent = json.loads(incumbent_path.read_text(encoding="utf-8")).get("weights") or {}

    # Build a selected-arm fixture from one eligible RB/WR with complete inputs.
    positions = weights["positions"]
    selected_pos = next(
        pos for pos, spec in positions.items() if spec["arm"] != "incumbent"
    )
    sample = eligible[eligible["position"].eq(selected_pos)].iloc[0]
    player_id = str(sample["player_id"])
    v1 = pd.read_csv(v1_path)
    v1["player_id"] = v1["player_id"].astype(str)
    v2 = pd.read_csv(v2_path)
    v2["player_id"] = v2["player_id"].astype(str)
    v1_pred = float(v1.loc[v1["player_id"].eq(player_id), "fantasy_pts_season"].iloc[0])
    v2_pred = float(v2.loc[v2["player_id"].eq(player_id), "fantasy_pts_season"].iloc[0])
    adp = float(sample["adp"])
    adp_points = float(apply_serialized_curve(np.array([adp]), serialized[selected_pos])[0])
    spec = positions[selected_pos]
    expected = 0.0
    inputs_map = {"v1_pred": v1_pred, "v2_pred": v2_pred, "adp": adp, "adp_points": adp_points}
    for column, weight in spec["weights"].items():
        expected += float(weight) * float(inputs_map[column] if column != "adp_points" else adp_points)

    fixture = {
        "player_id": player_id,
        "position": selected_pos,
        "inputs": {"v1_pred": v1_pred, "v2_pred": v2_pred, "adp": adp},
        "expected_points": expected,
        "expected_treatment": "selected",
    }
    contract = build_application_contract(
        positions=positions,
        market_curves=serialized,
        eligibility_ids=eligibility_ids,
        universe_ids=universe,
        reference_fixture=fixture,
        source_hashes=source_hashes,
        incumbent_weights=incumbent,
    )
    dest = out_path or DEFAULT_CONTRACT_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return contract


def inputs_from_frozen_sources(
    *,
    v2_path: Path,
    consensus_path: Path,
    v3_path: Path | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    v2 = pd.read_csv(v2_path)
    v2["player_id"] = v2["player_id"].astype(str)
    v2_by_id = {
        str(row.player_id): float(row.fantasy_pts_season)
        for row in v2.itertuples(index=False)
        if pd.notna(row.fantasy_pts_season)
    }
    consensus = json.loads(Path(consensus_path).read_text(encoding="utf-8"))
    adp_by_id = {
        str(row["player_id"]): float(row["adp"])
        for row in (consensus.get("rows") or [])
        if row.get("adp") is not None
    }
    v3_by_id: dict[str, float] = {}
    if v3_path and Path(v3_path).exists():
        v3 = pd.read_csv(v3_path)
        v3["player_id"] = v3["player_id"].astype(str)
        col = "p50" if "p50" in v3.columns else "fantasy_pts_season"
        v3_by_id = {
            str(row.player_id): float(getattr(row, col))
            for row in v3.itertuples(index=False)
            if pd.notna(getattr(row, col))
        }
    return v2_by_id, adp_by_id, v3_by_id
