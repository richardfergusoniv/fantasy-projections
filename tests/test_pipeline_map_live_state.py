"""Guard that PIPELINE_MAP live-state table matches the checked-in control plane."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.projection.active_release import read_active_pointer
from src.projection.contracts import REPO_ROOT
from src.projection.evaluation.accuracy_first import sha256_file
from src.projection.release_bundle import MANIFEST_FILENAME, public_release_dir

LIVE_STATE_PATH = Path(REPO_ROOT) / "docs" / "PIPELINE_MAP.md"
BEGIN = "<!-- LIVE_STATE_TABLE_BEGIN -->"
END = "<!-- LIVE_STATE_TABLE_END -->"
REQUIRED_KEYS = (
    "namespace",
    "previous_namespace",
    "release_id",
    "manifest_sha256",
    "model_id",
    "draw_count",
    "overlay_population",
    "strict_gate_promotion",
)


def _parse_live_state_table(text: str) -> dict[str, str]:
    if BEGIN not in text or END not in text:
        raise AssertionError("PIPELINE_MAP.md missing LIVE_STATE_TABLE markers")
    block = text.split(BEGIN, 1)[1].split(END, 1)[0]
    rows: dict[str, str] = {}
    for line in block.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 2:
            continue
        key, value = cells
        if key in {"key", "---"} or set(key) <= {"-"}:
            continue
        rows[key] = value
    return rows


def test_pipeline_map_live_state_matches_control_plane():
    text = LIVE_STATE_PATH.read_text(encoding="utf-8")
    documented = _parse_live_state_table(text)
    missing = [key for key in REQUIRED_KEYS if key not in documented]
    assert not missing, f"live-state table missing keys: {missing}"

    pointer = read_active_pointer(2026)
    assert pointer is not None
    assert documented["namespace"] == pointer["namespace"]
    assert documented["release_id"] == pointer["release_id"]
    assert documented["manifest_sha256"] == pointer["manifest_sha256"]
    previous = pointer.get("previous") or {}
    assert documented["previous_namespace"] == previous.get("namespace")

    public_manifest = public_release_dir(pointer["namespace"]) / MANIFEST_FILENAME
    assert public_manifest.is_file()
    assert sha256_file(public_manifest) == pointer["manifest_sha256"]
    manifest = json.loads(public_manifest.read_text(encoding="utf-8"))
    assert documented["model_id"] == manifest["bundle"]["model_id"]
    assert int(documented["draw_count"]) == int(manifest["simulation"]["draw_count"])
    overlay_count = manifest.get("overlay", {}).get("simulated_player_count")
    if overlay_count is None:
        overlay_count = (manifest.get("overlay_coverage") or {}).get("total_players")
    assert int(documented["overlay_population"]) == int(overlay_count)

    rollout_path = Path(REPO_ROOT) / "output" / "model_v3" / "draw_count_rollout_decision.json"
    rollout = json.loads(rollout_path.read_text(encoding="utf-8"))
    documented_strict = documented["strict_gate_promotion"].strip().lower()
    assert documented_strict in {"true", "false"}
    assert (documented_strict == "true") is bool(rollout.get("strict_gate_promotion"))

    contract_path = (
        Path(REPO_ROOT)
        / "output"
        / "model_v3"
        / "release_bundles"
        / "season=2026"
        / f"namespace={pointer['namespace']}"
        / "application_contract.json"
    )
    if contract_path.exists():
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        assert contract.get("model_id") == documented["model_id"]
