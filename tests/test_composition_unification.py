"""Guards that ship and measurement keep using ONE composition pipeline.

The failure these tests exist to prevent is not a wrong number, it is a silent
fork: a new allocation layer gets added to predict.py, the evaluation harness
keeps running the old stage list, and the gap between what ships and what is
measured widens without anything going red. Written as unittest.TestCase to
match tests/test_fantasy_evaluation.py, so both `pytest` and
`python -m unittest discover -s tests` collect them.
"""
import unittest
from unittest import mock

import pandas as pd

from src.projection import composition
from src.projection import fantasy_evaluation as fe
from src.projection import predict
from src.projection.composition import CompositionContext, compose_board

EXPECTED_STAGE_ORDER = [
    "apply_full_season_games_baseline",
    "apply_status_overrides",
    "propagate_team_anchors",
    "reconcile_stat_constraints",
    "add_projected_season_totals",
]

NON_NUMERIC_TAIL_STAGES = (
    "add_projected_season_totals",
)


def _context(**overrides):
    empty = pd.DataFrame()
    kwargs = dict(
        target_season=2026,
        depth_chart=empty,
        status_overrides=empty,
        artifact_provenance="test",
    )
    kwargs.update(overrides)
    return CompositionContext(**kwargs)


class CompositionUnificationTest(unittest.TestCase):
    def test_both_callers_bind_the_same_compose_board(self):
        self.assertIs(predict.compose_board, compose_board)
        self.assertIs(fe.compose_board, compose_board)

    def test_compose_board_runs_the_shipped_stages_in_order(self):
        calls = []

        def recorder(name):
            def stage(df, *args, **kwargs):
                calls.append(name)
                return df
            return stage

        patchers = [
            mock.patch.object(composition, name, recorder(name))
            for name in EXPECTED_STAGE_ORDER
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

        compose_board(pd.DataFrame({"player_id": ["a"], "projected_games": [10.0]}), _context())
        self.assertEqual(calls, EXPECTED_STAGE_ORDER)

    def test_last_numeric_stage_is_the_stat_constraint_guard(self):
        numeric = [
            stage for stage in EXPECTED_STAGE_ORDER
            if stage not in NON_NUMERIC_TAIL_STAGES
        ]
        self.assertEqual(numeric[-1], "reconcile_stat_constraints")

    def test_full_season_baseline_then_status_zero(self):
        frame = pd.DataFrame({
            "player_id": ["ok", "ir", "ok", "ir"],
            "position": ["WR", "WR", "WR", "WR"],
            "season": [2026, 2026, 2026, 2026],
            "team": ["A", "A", "A", "A"],
            "stat": ["targets", "targets", "receptions", "receptions"],
            "pred_pg": [5.0, 5.0, 3.0, 3.0],
            "pred_pg_low": [5.0, 5.0, 3.0, 3.0],
            "pred_pg_high": [5.0, 5.0, 3.0, 3.0],
            "projected_games": [12.0, 12.0, 12.0, 12.0],
        })
        patcher = mock.patch.object(
            composition, "propagate_team_anchors", lambda df, *a, **k: df)
        patcher.start()
        self.addCleanup(patcher.stop)

        out = compose_board(
            frame,
            _context(status_overrides=pd.DataFrame([
                {"gsis_id": "ir", "mode": "zero", "projected_games": None},
            ])),
        )
        healthy = out[out["player_id"].eq("ok")]
        self.assertTrue((healthy["projected_games"] == 17.0).all())
        self.assertTrue((out.loc[out["player_id"].eq("ir"), "projected_games"] == 0.0).all())
        self.assertTrue((healthy["pred_season"] == healthy["pred_pg"] * 17.0).all())
        self.assertTrue((healthy["projected_games_raw"] == 12.0).all())

    def test_stat_constraint_caps_child_above_parent(self):
        frame = pd.DataFrame({
            "player_id": ["p", "p"],
            "position": ["WR", "WR"],
            "season": [2026, 2026],
            "stat": ["receptions", "targets"],
            "pred_pg": [6.0, 5.0],
            "pred_pg_low": [6.0, 5.0],
            "pred_pg_high": [6.0, 5.0],
            "projected_games": [12.0, 12.0],
        })

        def passthrough(df, *args, **kwargs):
            return df

        for name in (
            "apply_full_season_games_baseline",
            "apply_status_overrides",
            "propagate_team_anchors",
            "add_projected_season_totals",
        ):
            patcher = mock.patch.object(composition, name, passthrough)
            patcher.start()
            self.addCleanup(patcher.stop)

        out = compose_board(frame, _context())
        receptions = float(out.loc[out["stat"].eq("receptions"), "pred_pg"].iloc[0])
        targets = float(out.loc[out["stat"].eq("targets"), "pred_pg"].iloc[0])
        self.assertLessEqual(receptions, targets + 1e-9)

    def test_stat_constraint_flag_is_sticky(self):
        from src.projection.team_reconcile import reconcile_stat_constraints

        frame = pd.DataFrame({
            "player_id": ["p", "p"],
            "position": ["QB", "QB"],
            "season": [2026, 2026],
            "stat": ["completions", "attempts"],
            "pred_pg": [30.0, 25.0],
        })
        once = reconcile_stat_constraints(frame)
        self.assertTrue(bool(once.loc[once["stat"].eq("completions"),
                                      "stat_constraint_applied"].iloc[0]))
        twice = reconcile_stat_constraints(once)
        self.assertTrue(bool(twice.loc[twice["stat"].eq("completions"),
                                       "stat_constraint_applied"].iloc[0]))
        pd.testing.assert_series_equal(once["pred_pg"], twice["pred_pg"])

    def test_absent_curated_research_is_reported_not_hidden(self):
        coverage = _context().describe_coverage()
        self.assertEqual(coverage["apply_full_season_games_baseline"], "active")
        self.assertIn("no-op", coverage["apply_status_overrides"])
        for stage in EXPECTED_STAGE_ORDER:
            self.assertIn(stage, coverage)

    def test_present_status_overrides_are_active(self):
        coverage = _context(
            status_overrides=pd.DataFrame({"gsis_id": ["x"], "mode": ["zero"]}),
        ).describe_coverage()
        self.assertEqual(coverage["apply_full_season_games_baseline"], "active")
        self.assertEqual(coverage["apply_status_overrides"], "active")


if __name__ == "__main__":
    unittest.main()
