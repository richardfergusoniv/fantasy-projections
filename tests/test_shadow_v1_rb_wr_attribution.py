"""Shadow v1 RB/WR attribution: consensus pin, error sum, isolation, compose parity."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from src.projection.composition import (
    COMPOSE_CHECKPOINT_NAMES,
    CompositionContext,
    compose_board,
    compose_board_stages,
    run_compose_stages,
)
from src.projection.shadow.consensus_pin import (
    ConsensusPinError,
    expected_consensus_hashes,
    load_pinned_consensus,
    membership_set_hash,
    persist_top120_membership,
)
from src.projection.shadow.decision_rules import classify_diagnosis, repair_gate
from src.projection.shadow.error_decomposition import decompose_prediction_error
from src.projection.shadow.forbidden import (
    ForbiddenDependencyError,
    ForbiddenImportGuard,
    assert_input_path_allowed,
    assert_no_forbidden_imports,
)
from src.projection.shadow.rb_wr_attribution import (
    SHADOW_ENTRYPOINTS,
    run_shadow_attribution,
)
from src.projection.shadow.repair import REPAIR_ENTRYPOINTS, freeze_shadow_candidate


def _tiny_long_board() -> pd.DataFrame:
    rows = []
    anchors = {
        "team_pass_attempts_pg_pred": 34.0,
        "team_passing_yards_pg_pred": 250.0,
        "team_carries_pg_pred": 26.0,
        "team_rushing_yards_pg_pred": 115.0,
        "team_total_pred": 250.0,
        "team_anchor_source_season": 2025,
        "team_anchor_lag_team": "AAA",
        "team_anchor_provenance": "test_fixture",
    }
    for pid, pos, team, yards_stat, yards in (
        ("rb1", "RB", "AAA", "rushing_yards", 80.0),
        ("wr1", "WR", "AAA", "receiving_yards", 70.0),
        ("qb1", "QB", "AAA", "passing_yards", 250.0),
    ):
        base = {
            "player_id": pid,
            "display_name": pid,
            "position": pos,
            "team": team,
            "stat": yards_stat,
            "pred_pg": yards,
            "pred_pg_low": yards * 0.8,
            "pred_pg_high": yards * 1.2,
            "projected_games": 16.0,
            "season": 2026,
            "interval_low_n_flag": False,
            **anchors,
            "team_anchor_lag_team": team,
        }
        rows.append(base)
        if pos == "WR":
            rows.append({
                **base,
                "stat": "receptions",
                "pred_pg": 5.0,
                "pred_pg_low": 4.0,
                "pred_pg_high": 6.0,
            })
        if pos == "QB":
            rows.append({
                **base,
                "stat": "attempts",
                "pred_pg": 34.0,
                "pred_pg_low": 30.0,
                "pred_pg_high": 38.0,
            })
        if pos == "RB":
            rows.append({
                **base,
                "stat": "carries",
                "pred_pg": 18.0,
                "pred_pg_low": 14.0,
                "pred_pg_high": 22.0,
            })
    return pd.DataFrame(rows)


def _ctx() -> CompositionContext:
    return CompositionContext(
        target_season=2026,
        depth_chart=pd.DataFrame(),
        status_overrides=pd.DataFrame(),
        artifact_provenance="test",
    )


class ComposeStageRunnerTests(unittest.TestCase):
    def test_final_checkpoint_equals_compose_board(self):
        import src.projection.composition as composition

        rows = _tiny_long_board()
        ctx = _ctx()

        def _passthrough(df, *args, **kwargs):
            return df

        def _add_season(df, *args, **kwargs):
            out = df.copy()
            games = pd.to_numeric(out.get("projected_games"), errors="coerce").fillna(0.0)
            out["pred_season"] = pd.to_numeric(out["pred_pg"], errors="coerce").fillna(0.0) * games
            return out

        patch_targets = {
            "propagate_team_anchors": _passthrough,
            "reconcile_team_volume": _passthrough,
            "apply_concentration": _passthrough,
            "reconcile_td_rate_constraints": _passthrough,
            "reconcile_pass_td_t1_lite": _passthrough,
            "reconcile_stat_constraints": _passthrough,
            "add_projected_season_totals": _add_season,
            "reconcile_team_season_identities": _passthrough,
            "apply_full_season_games_baseline": _passthrough,
            "apply_status_overrides": _passthrough,
        }
        with mock.patch.multiple(composition, **{
            name: mock.MagicMock(side_effect=fn) for name, fn in patch_targets.items()
        }):
            shipped = compose_board(rows.copy(), ctx)
            staged_final, boards = run_compose_stages(
                rows.copy(), ctx, capture_checkpoints=True
            )
        pd.testing.assert_frame_equal(
            shipped.reset_index(drop=True),
            staged_final.reset_index(drop=True),
            check_dtype=False,
        )
        self.assertEqual(set(COMPOSE_CHECKPOINT_NAMES), set(boards))
        with mock.patch.multiple(composition, **{
            name: mock.MagicMock(side_effect=fn) for name, fn in patch_targets.items()
        }):
            scored = compose_board_stages(rows.copy(), ctx)
        self.assertIn("rb1", scored["season_total_finalization"])
        self.assertIn("wr1", scored["season_total_finalization"])
        self.assertIn("qb1", scored["season_total_finalization"])
        self.assertIn("final_shipped", scored)
        self.assertEqual(
            scored["final_shipped"]["rb1"]["fantasy_ppg"],
            scored["season_total_finalization"]["rb1"]["fantasy_ppg"],
        )


class ErrorDecompositionTests(unittest.TestCase):
    def test_components_sum_to_total_error(self):
        frame = pd.DataFrame({
            "player_id": ["a", "b", "c"],
            "v1_pred": [100.0, 50.0, 0.0],
            "actual_points": [80.0, 60.0, 0.0],
            "raw_rate_ppg": [6.0, 4.0, 0.0],
            "composed_rate_ppg": [5.5, 4.5, 0.0],
            "projected_games": [16.0, 14.0, 17.0],
            "actual_games_played": [14.0, 12.0, 0.0],
        })
        out = decompose_prediction_error(frame)
        residual = (
            out["raw_rate_error"]
            + out["composition_rate_effect"]
            + out["availability_effect"]
            + out["finalization_remainder"]
            - out["total_error"]
        )
        self.assertTrue(np.allclose(residual, 0.0, atol=1e-9))

    def test_zero_game_row(self):
        frame = pd.DataFrame({
            "player_id": ["z"],
            "v1_pred": [40.0],
            "actual_points": [0.0],
            "raw_rate_ppg": [3.0],
            "composed_rate_ppg": [2.5],
            "projected_games": [16.0],
            "actual_games_played": [0.0],
        })
        out = decompose_prediction_error(frame)
        self.assertAlmostEqual(float(out.loc[0, "total_error"]), 40.0)
        self.assertTrue(np.isclose(float(out.loc[0, "decomposition_residual"]), 0.0))


class ConsensusPinTests(unittest.TestCase):
    def test_expected_hashes_from_freeze_manifest(self):
        hashes = expected_consensus_hashes()
        self.assertEqual(set(hashes), {2023, 2024, 2025})
        for season, digest in hashes.items():
            _, record = load_pinned_consensus(season, expected_hash=digest)
            self.assertTrue(record["hash_match"])
            self.assertEqual(record["actual_hash"], digest)

    def test_mismatch_fail_closed(self):
        with self.assertRaises(ConsensusPinError):
            load_pinned_consensus(2024, expected_hash="0" * 64)

    def test_membership_hash_stable(self):
        a = membership_set_hash(["b", "a", "a"])
        b = membership_set_hash(["a", "b"])
        self.assertEqual(a, b)

    def test_persist_membership_roundtrip(self):
        hashes = expected_consensus_hashes()
        rows, pin = load_pinned_consensus(2025, expected_hash=hashes[2025])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "top120_membership_2025.json"
            payload = persist_top120_membership(2025, rows, out_path=path, pin_record=pin)
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["top120_membership_hash"], loaded["top120_membership_hash"])
            self.assertEqual(
                membership_set_hash(loaded["player_ids"]),
                loaded["top120_membership_hash"],
            )


class ForbiddenIsolationTests(unittest.TestCase):
    def test_static_graph_rejects_sleeper_and_promotion(self):
        graph = assert_no_forbidden_imports(SHADOW_ENTRYPOINTS)
        self.assertIn("src.projection.shadow.rb_wr_attribution", graph)
        self.assertNotIn("src.comparison.sleeper_compare", graph)
        self.assertNotIn("src.projection.promote_release", graph)
        repair_graph = assert_no_forbidden_imports(REPAIR_ENTRYPOINTS)
        self.assertNotIn("src.comparison.spot_check", repair_graph)

    def test_runtime_guard_blocks_dynamic_import(self):
        with ForbiddenImportGuard():
            with self.assertRaises(ForbiddenDependencyError):
                __import__("src.comparison.sleeper_compare")

    def test_sleeper_input_path_rejected(self):
        with self.assertRaises(ForbiddenDependencyError):
            assert_input_path_allowed("output/sleeper_snapshots/players_nfl_abc.json")
        with self.assertRaises(ForbiddenDependencyError):
            assert_input_path_allowed("output/sleeper_comparison_2026.csv")


class DecisionRuleTests(unittest.TestCase):
    def test_diagnosis_is_exactly_one_label(self):
        label = classify_diagnosis(
            component_dominance={
                "raw_rate_error": -12.0,
                "composition_rate_effect": 1.0,
                "availability_effect": 3.0,
                "finalization_remainder": 0.5,
            }
        )
        self.assertEqual(label, "raw_rate_model_defect")

    def test_repair_gate_hold_when_insufficient(self):
        result = repair_gate(
            fold_mae_deltas=[0.02, -0.01, 0.0],
            pooled_top120_spearman_baseline=0.5,
            pooled_top120_spearman_candidate=0.49,
            all_eligible_ok=True,
            coverage_unchanged=True,
            team_identity_unchanged=True,
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["verdict"], "hold_v1_structural_role")


class ShadowAttributionIntegrationTests(unittest.TestCase):
    def test_run_writes_artifacts_and_keeps_production(self):
        eval_2023 = Path("output/fantasy_evaluation_2023.csv")
        eval_2024 = Path("output/fantasy_evaluation_2024.csv")
        eval_2025 = Path("output/fantasy_evaluation_2025.csv")
        if not (eval_2023.is_file() and eval_2024.is_file() and eval_2025.is_file()):
            self.skipTest("historical fantasy_evaluation frames unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "shadow_v1_rb_wr"
            manifest = run_shadow_attribution(out_dir=dest, n_boot=50)
            self.assertEqual(manifest["status"], "ok")
            self.assertTrue(manifest["production_weights_unchanged"])
            self.assertTrue((dest / "attribution_players.parquet").is_file())
            self.assertTrue((dest / "attribution_metrics.csv").is_file())
            self.assertTrue((dest / "stage_attribution.csv").is_file())
            self.assertTrue((dest / "attribution_summary.json").is_file())
            self.assertTrue((dest / "manifest.json").is_file())
            for season in (2023, 2024, 2025):
                self.assertTrue((dest / f"top120_membership_{season}.json").is_file())
            # Missing coverage must remain visible (nulls), not silently filled.
            players = pd.read_parquet(dest / "attribution_players.parquet")
            self.assertIn("v2_covered", players.columns)
            self.assertIn("adp_covered", players.columns)
            self.assertTrue(players["decomposition_residual"].abs().max() < 1e-5)

    def test_consensus_failure_writes_no_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "shadow_v1_rb_wr"
            with mock.patch(
                "src.projection.shadow.rb_wr_attribution.require_all_pinned_consensus",
                side_effect=ConsensusPinError("hash mismatch"),
            ):
                with self.assertRaises(ConsensusPinError):
                    run_shadow_attribution(out_dir=dest, n_boot=10)
            self.assertTrue((dest / "manifest.json").is_file())
            payload = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "fail_closed")
            self.assertFalse((dest / "attribution_players.parquet").exists())
            self.assertFalse((dest / "attribution_metrics.csv").exists())

    def test_freeze_hold_when_gate_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = freeze_shadow_candidate(
                candidate_id="noop",
                code_identity={"module": "none"},
                evidence={"note": "synthetic"},
                source_hashes={},
                fold_mae_relative_deltas=[0.05, 0.02, 0.01],
                pooled_top120_spearman_baseline=0.4,
                pooled_top120_spearman_candidate=0.3,
                out_dir=tmp,
            )
            self.assertEqual(payload["gate"]["verdict"], "hold_v1_structural_role")
            self.assertTrue((Path(tmp) / "hold_v1_structural_role.json").is_file())


if __name__ == "__main__":
    unittest.main()
