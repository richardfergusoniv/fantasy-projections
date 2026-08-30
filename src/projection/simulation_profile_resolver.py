"""Resolve production simulation identity from sealed rollout decision and config."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from src.projection.contracts import MODEL_V3_DIR, REPO_ROOT
from src.projection.evaluation.accuracy_first import canonical_json_hash, sha256_file
from src.projection.inference.simulation_config import (
    CONFIG_PATH,
    profile_chunk_size,
    profile_draws,
)

DEFAULT_ROLLOUT_DECISION_PATH = Path(MODEL_V3_DIR) / "draw_count_rollout_decision.json"

PROFILE_IDENTITY_FIELDS = (
    "profile_key",
    "profile_label",
    "draw_count",
    "chunk_size",
    "configuration_hash",
    "policy_hash",
)


def rollout_decision_path(path: Path | None = None) -> Path:
    return path or DEFAULT_ROLLOUT_DECISION_PATH


def simulation_config_path(path: Path | None = None) -> Path:
    return path or CONFIG_PATH


def load_rollout_decision(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"draw count rollout decision missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_sealed_simulation_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"simulation config missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def policy_hash_from_rollout(decision: Mapping[str, Any]) -> str:
    """Stable hash of the closed rollout-decision artifact body."""
    body = dict(decision)
    for key in ("generated_at", "human_decision"):
        body.pop(key, None)
    return canonical_json_hash(body)


def configuration_hash_from_config(config: Mapping[str, Any], *, profile_key: str) -> str:
    return canonical_json_hash(
        {
            "random_seed": config.get("random_seed"),
            "profile_key": profile_key,
            "profile_config": (config.get("profiles") or {}).get(profile_key),
        }
    )


def resolve_simulation_profile_identity(
    *,
    profile_key: str = "publish",
    rollout_path: Path | None = None,
    simulation_config_path_arg: Path | None = None,
) -> dict[str, Any]:
    """Derive profile identity from sealed configuration and rollout-decision copies."""
    config_file = simulation_config_path(simulation_config_path_arg)
    rollout_file = rollout_decision_path(rollout_path)
    config = load_sealed_simulation_config(config_file)
    decision = load_rollout_decision(rollout_file)

    profile_label = str(
        decision.get("current_production_profile")
        or decision.get("operational_policy")
        or decision.get("promotion_gate_10k", {}).get("policy")
        or ""
    )
    if profile_key == "publish" and not profile_label:
        raise ValueError("rollout decision does not define current_production_profile")
    draw_count = int(
        decision.get("current_production_draw_count")
        or decision.get("chosen_production_draw_count")
        or 0
    )
    config_draws = profile_draws(config, profile_key)
    if config_draws is None:
        raise ValueError(f"sealed simulation config missing draws for profile {profile_key!r}")
    if profile_key == "publish":
        if draw_count <= 0:
            raise ValueError("rollout decision does not define a positive draw_count")
        if int(config_draws) != draw_count:
            raise ValueError(
                f"sealed simulation config profile {profile_key!r} draws={config_draws} "
                f"!= rollout decision draw_count={draw_count}"
            )
    else:
        draw_count = int(config_draws)
        profile_label = profile_key
    chunk_size = profile_chunk_size(config, profile_key)
    configuration_hash = configuration_hash_from_config(config, profile_key=profile_key)
    policy_hash = policy_hash_from_rollout(decision)
    try:
        rollout_rel = str(rollout_file.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        rollout_rel = rollout_file.as_posix()
    try:
        config_rel = str(config_file.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        config_rel = config_file.as_posix()
    return {
        "profile_key": profile_key,
        "profile_label": profile_label,
        "draw_count": draw_count,
        "chunk_size": chunk_size,
        "configuration_hash": configuration_hash,
        "policy_hash": policy_hash,
        "rollout_decision_path": rollout_rel,
        "rollout_decision_hash": sha256_file(rollout_file),
        "simulation_config_path": config_rel,
        "simulation_config_hash": sha256_file(config_file),
    }
