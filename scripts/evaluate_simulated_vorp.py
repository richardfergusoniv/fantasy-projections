"""Evaluate simulated VORP replacement contract and write gate artifacts."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.draft_assistant.replacement_contract import (
    build_replacement_contract,
    contract_output_dir,
    default_selected_board_path,
    load_roster_configuration,
    load_selected_board,
    read_replacement_contract,
    write_replacement_contract,
)
from src.draft_assistant.simulated_vorp import (
    load_manifest_partitions,
    stream_simulated_vorp_summary,
)
from src.projection.contracts import MODEL_V3_DIR, OUTPUT_DIR
from src.projection.evaluation.finish_probability_gate import read_finish_probability_gate
from src.projection.evaluation.simulated_vorp_contract_tests import run_contract_tests
from src.projection.evaluation.simulated_vorp_gate import (
    build_simulated_vorp_gate,
    gate_output_dir,
    write_simulated_vorp_gate,
)
from src.projection.inference.recenter import sha256_file


def _load_manifest(path: Path | None, season: int) -> dict:
    manifest_path = path or Path(MODEL_V3_DIR) / f"simulation_manifest_{season}.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing simulation manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--board", type=Path, default=None)
    parser.add_argument("--simulation-manifest", type=Path, default=None)
    parser.add_argument("--finish-gate", type=Path, default=None)
    parser.add_argument("--roster-config", type=Path, default=None)
    parser.add_argument("--write-contract", action="store_true")
    parser.add_argument("--write-summary", action="store_true")
    parser.add_argument("--write-gate", action="store_true")
    args = parser.parse_args()

    manifest = _load_manifest(args.simulation_manifest, args.season)
    finish_gate = read_finish_probability_gate(args.finish_gate)
    board_path = args.board or default_selected_board_path(args.season)
    board = load_selected_board(args.season, board_path=board_path)
    board_hash = sha256_file(board_path)
    roster_config = load_roster_configuration(args.roster_config)
    run_id = manifest.get("canonical_projection_run_id") or manifest.get(
        "source_projection_run_id"
    )
    if not run_id:
        raise SystemExit("manifest missing canonical_projection_run_id")

    contract = build_replacement_contract(
        board,
        season=args.season,
        selected_board_hash=board_hash,
        selected_board_model_id=str(manifest.get("selected_board_model_id") or "accuracy_first_ensemble"),
        canonical_projection_run_id=str(run_id),
        roster_config=roster_config,
    )
    contract_tests = run_contract_tests()
    partitions = load_manifest_partitions(manifest, season=args.season)
    summary = stream_simulated_vorp_summary(partitions, replacement_contract=contract)
    gate = build_simulated_vorp_gate(
        season=args.season,
        manifest=manifest,
        replacement_contract=contract,
        contract_tests=contract_tests,
        finish_gate=finish_gate,
    )

    out_dir = gate_output_dir(args.season, board_hash)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sim-vorp-", dir=out_dir.parent) as tmp:
        stage = Path(tmp)
        if args.write_contract or args.write_summary or args.write_gate:
            write_replacement_contract(contract, stage / "replacement_contract.json")
        if args.write_summary:
            summary_path = stage / "simulated_vorp_summary.parquet"
            summary.to_parquet(summary_path, index=False)
        if args.write_gate:
            write_simulated_vorp_gate(gate, stage / "simulated_vorp_gate.json")
        final_dir = contract_output_dir(args.season, board_hash)
        if final_dir.exists():
            shutil.rmtree(final_dir)
        shutil.move(str(stage), str(final_dir))

    print(
        json.dumps(
            {
                "state": gate.get("state"),
                "publication_verdict": gate.get("publication_verdict"),
                "contract_tests_pass": contract_tests.get("passes"),
                "precondition_failures": gate.get("preconditions", {}).get("failures"),
                "output_dir": str(out_dir),
                "summary_players": int(len(summary)),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0 if gate.get("publication_verdict") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
