"""Fail-closed repair-track policy for shadow v1 RB/WR freeze."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.projection.evaluation.accuracy_first import canonical_json_hash
from src.projection.shadow.contracts import CLOSEOUT_SCHEMA
from src.projection.shadow.repair import (
    evaluate_repair_track_authorization,
    freeze_shadow_candidate,
)


def _seal_body(**overrides: object) -> dict:
    body: dict = {
        "schema_version": CLOSEOUT_SCHEMA,
        "verdict": "close_shadow_repair_track",
        "repair_track_status": "closed",
        "further_repair_authorized": False,
        "promotion_authorized": False,
        "production_weights_unchanged": True,
    }
    body.update(overrides)
    return body


def _write_policy(path: Path, body: dict) -> dict:
    payload = dict(body)
    payload["artifact_hash"] = canonical_json_hash(payload)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _freeze_kwargs(**overrides: object) -> dict:
    kwargs: dict = {
        "candidate_id": "gate_test",
        "code_identity": {"module": "tests"},
        "evidence": {"note": "synthetic"},
        "source_hashes": {},
        "fold_mae_relative_deltas": [-0.05, -0.02, -0.01],
        "pooled_top120_spearman_baseline": 0.4,
        "pooled_top120_spearman_candidate": 0.55,
        "attribution_status": "ok",
    }
    kwargs.update(overrides)
    return kwargs


class RepairTrackPolicyTests(unittest.TestCase):
    def test_absent_policy_not_authorized(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = evaluate_repair_track_authorization(Path(tmp) / "missing.json")
        self.assertFalse(result["authorized"])
        self.assertEqual(result["reason"], "repair_track_closed")
        self.assertEqual(result["policy_detail"], "policy_absent")

    def test_malformed_policy_not_authorized(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "repair_track_closed.json"
            path.write_text("{not-json", encoding="utf-8")
            result = evaluate_repair_track_authorization(path)
        self.assertFalse(result["authorized"])
        self.assertEqual(result["policy_detail"], "policy_malformed")

    def test_hash_mismatch_not_authorized(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "repair_track_closed.json"
            body = _seal_body()
            body["artifact_hash"] = "0" * 64
            path.write_text(json.dumps(body, indent=2), encoding="utf-8")
            result = evaluate_repair_track_authorization(path)
        self.assertFalse(result["authorized"])
        self.assertEqual(result["policy_detail"], "policy_hash_mismatch")

    def test_closed_seal_not_authorized(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "repair_track_closed.json"
            _write_policy(path, _seal_body(further_repair_authorized=False))
            result = evaluate_repair_track_authorization(path)
        self.assertFalse(result["authorized"])
        self.assertEqual(result["policy_detail"], "further_repair_not_authorized")

    def test_authorized_true_without_reopen_evidence_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "repair_track_closed.json"
            _write_policy(path, _seal_body(further_repair_authorized=True))
            result = evaluate_repair_track_authorization(path)
        self.assertFalse(result["authorized"])
        self.assertEqual(result["policy_detail"], "reopen_evidence_missing")

    def test_explicit_reopen_authorizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "repair_track_closed.json"
            _write_policy(
                path,
                _seal_body(
                    further_repair_authorized=True,
                    repair_track_status="reopened",
                    reopen={
                        "cutoff_available_defect": "synthetic cutoff defect",
                        "evidence_refs": ["tests/synthetic_evidence.json"],
                    },
                ),
            )
            result = evaluate_repair_track_authorization(path)
        self.assertTrue(result["authorized"])
        self.assertEqual(result["reason"], "repair_authorized")

    def test_freeze_blocked_when_policy_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "out"
            policy = Path(tmp) / "missing_policy.json"
            with mock.patch(
                "src.projection.shadow.repair.snapshot_production_artifacts",
                return_value={"ensemble_weights": {"sha256": "a"}},
            ), mock.patch(
                "src.projection.shadow.repair.assert_production_unchanged",
                return_value={"production_weights_unchanged": True},
            ):
                payload = freeze_shadow_candidate(
                    out_dir=dest,
                    policy_path=policy,
                    **_freeze_kwargs(),
                )
            self.assertEqual(payload["verdict"], "hold_v1_structural_role")
            self.assertEqual(payload["gate"]["reason"], "repair_track_closed")
            self.assertFalse((dest / "freeze_gate_test.json").exists())
            self.assertTrue((dest / "hold_v1_structural_role.json").is_file())

    def test_freeze_blocked_when_closed_even_if_gate_would_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "out"
            policy = Path(tmp) / "repair_track_closed.json"
            _write_policy(policy, _seal_body(further_repair_authorized=False))
            with mock.patch(
                "src.projection.shadow.repair.snapshot_production_artifacts",
                return_value={"ensemble_weights": {"sha256": "a"}},
            ), mock.patch(
                "src.projection.shadow.repair.assert_production_unchanged",
                return_value={"production_weights_unchanged": True},
            ):
                payload = freeze_shadow_candidate(
                    out_dir=dest,
                    policy_path=policy,
                    **_freeze_kwargs(),
                )
            self.assertEqual(payload["gate"]["reason"], "repair_track_closed")
            self.assertFalse((dest / "freeze_gate_test.json").exists())

    def test_freeze_allowed_with_explicit_reopen_when_gate_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "out"
            policy = Path(tmp) / "repair_track_closed.json"
            _write_policy(
                policy,
                _seal_body(
                    further_repair_authorized=True,
                    repair_track_status="reopened",
                    reopen={
                        "cutoff_available_defect": "synthetic cutoff defect",
                        "evidence_refs": ["tests/synthetic_evidence.json"],
                    },
                ),
            )
            with mock.patch(
                "src.projection.shadow.repair.snapshot_production_artifacts",
                return_value={"ensemble_weights": {"sha256": "a"}},
            ), mock.patch(
                "src.projection.shadow.repair.assert_production_unchanged",
                return_value={"production_weights_unchanged": True},
            ) as assert_unchanged:
                payload = freeze_shadow_candidate(
                    out_dir=dest,
                    policy_path=policy,
                    **_freeze_kwargs(),
                )
            self.assertTrue(payload["gate"]["passed"])
            self.assertTrue((dest / "freeze_gate_test.json").is_file())
            assert_unchanged.assert_called()


if __name__ == "__main__":
    unittest.main()
