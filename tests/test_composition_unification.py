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

# The shipped stage sequence, in order. Changing this list is a deliberate
# change to how every board - projected and evaluated - is composed.
EXPECTED_STAGE_ORDER = [
    "apply_deep_bench_games_cap",
    "apply_status_overrides",
    "propagate_team_anchors",
    "reconcile_qb_projected_volume_games",
    "apply_usage_share_prior",
    "attach_team_pass_mix",
    "apply_hierarchical_pass_distribution",
    "attach_team_rush_mix",
    "apply_hierarchical_rush_distribution",
    "normalize_team_passing_volume",
    "normalize_team_rushing_volume",
    "reconcile_stat_constraints",
    "reconcile_team_pass_receive_counts",
    # Trailing guard: reconcile_team_pass_receive_counts rescales receptions and
    # receiving TDs with a (team, position) factor their PARENT stats do not
    # share, so it can put a child back above its parent after the first guard
    # already ran. This second call makes the constraint hold at the output
    # boundary rather than merely somewhere in the middle.
    "reconcile_stat_constraints",
    "add_team_pass_catch_coherence_flag",
    "add_projected_season_totals",
]

# Stages after which no pred_pg / pred_pg_low / pred_pg_high value changes.
# Everything before the last of these is "numeric" for the purposes of the
# guard below.
NON_NUMERIC_TAIL_STAGES = (
    "add_team_pass_catch_coherence_flag",
    "add_projected_season_totals",
)


def _context(**overrides):
    empty = pd.DataFrame()
    kwargs = dict(
        target_season=2026,
        depth_chart=empty,
        status_overrides=empty,
        usage_share_priors=empty,
        pass_mix_profiles=empty,
        rush_mix_profiles=empty,
        artifact_provenance="test",
    )
    kwargs.update(overrides)
    return CompositionContext(**kwargs)


class CompositionUnificationTest(unittest.TestCase):
    def test_both_callers_bind_the_same_compose_board(self):
        """Neither module may grow a private copy of the pipeline."""
        self.assertIs(predict.compose_board, compose_board)
        self.assertIs(fe.compose_board, compose_board)

    def test_compose_board_runs_the_shipped_stages_in_order(self):
        """Pins the order in exactly one place, for both callers."""
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

        compose_board(pd.DataFrame({"player_id": ["a"]}), _context())
        self.assertEqual(calls, EXPECTED_STAGE_ORDER)

    def test_last_numeric_stage_is_the_stat_constraint_guard(self):
        """The output boundary, not the middle, is where the identity must hold.

        Pinned separately from the full order list so that reordering or
        inserting a new allocation stage at the end of the pipeline fails with
        a message about the guard rather than as one line of a long diff.
        """
        numeric = [
            stage for stage in EXPECTED_STAGE_ORDER
            if stage not in NON_NUMERIC_TAIL_STAGES
        ]
        self.assertEqual(numeric[-1], "reconcile_stat_constraints")
        # And it must run at least once BEFORE the count reconciler too, so a
        # violation introduced by the volume normalizers is not carried through
        # a stage that assumes coherent inputs.
        self.assertLess(
            numeric.index("reconcile_stat_constraints"),
            numeric.index("reconcile_team_pass_receive_counts"),
        )
        self.assertEqual(numeric.count("reconcile_stat_constraints"), 2)

    def test_trailing_guard_catches_a_violation_the_count_stage_introduces(self):
        """Behavioural counterpart: a child stat rescaled past its parent by the
        FINAL numeric stage must still be capped in the returned board."""
        frame = pd.DataFrame({
            "player_id": ["p", "p"],
            "position": ["WR", "WR"],
            "season": [2026, 2026],
            "stat": ["receptions", "targets"],
            "pred_pg": [4.0, 5.0],
            "pred_pg_low": [4.0, 5.0],
            "pred_pg_high": [4.0, 5.0],
        })

        def inflate_receptions(df, *args, **kwargs):
            out = df.copy()
            hit = out["stat"].eq("receptions")
            out.loc[hit, ["pred_pg", "pred_pg_low", "pred_pg_high"]] *= 2.0
            return out

        passthrough = [
            name for name in dict.fromkeys(EXPECTED_STAGE_ORDER)
            if name not in ("reconcile_stat_constraints",
                            "reconcile_team_pass_receive_counts")
        ]
        for name in passthrough:
            patcher = mock.patch.object(
                composition, name, lambda df, *a, **k: df)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch.object(
            composition, "reconcile_team_pass_receive_counts", inflate_receptions)
        patcher.start()
        self.addCleanup(patcher.stop)

        out = compose_board(frame, _context())
        receptions = float(out.loc[out["stat"].eq("receptions"), "pred_pg"].iloc[0])
        targets = float(out.loc[out["stat"].eq("targets"), "pred_pg"].iloc[0])
        self.assertLessEqual(receptions, targets + 1e-9)

    def test_stat_constraint_flag_survives_the_second_call(self):
        """Running the guard twice must not erase the first call's audit trail.

        `stat_constraint_applied` ships in OUTPUT_COLUMNS and is aggregated into
        fantasy_points' any_stat_constraint_applied. A non-sticky flag would
        report only what the LAST call capped - on the 2026 board that silently
        dropped 28 genuine caps to False.
        """
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
        # Second pass has nothing left to cap, and must keep the flag.
        twice = reconcile_stat_constraints(once)
        self.assertTrue(bool(twice.loc[twice["stat"].eq("completions"),
                                       "stat_constraint_applied"].iloc[0]))
        pd.testing.assert_series_equal(once["pred_pg"], twice["pred_pg"])

    def test_absent_curated_research_is_reported_not_hidden(self):
        """A stage with no input must say so, so coverage can't read as skill."""
        coverage = _context().describe_coverage()
        self.assertIn("no-op", coverage["apply_deep_bench_games_cap"])
        self.assertIn("no-op", coverage["apply_status_overrides"])
        self.assertIn("degraded", coverage["apply_hierarchical_pass_distribution"])
        self.assertEqual(set(coverage), set(EXPECTED_STAGE_ORDER))

    def test_present_curated_research_is_reported_as_active(self):
        chart = pd.DataFrame({
            "gsis_id": ["x"], "position": ["WR"], "team": ["A"],
            "depth_rank": [1], "role": ["starter"], "formation_role": ["LWR"],
        })
        priors = pd.DataFrame({"target_share": [0.1], "carry_share": [0.1]})
        mix = pd.DataFrame({"season": [2026], "team": ["A"]})
        coverage = _context(
            depth_chart=chart,
            status_overrides=pd.DataFrame({"gsis_id": ["x"], "mode": ["zero"]}),
            usage_share_priors=priors,
            pass_mix_profiles=mix,
            rush_mix_profiles=mix,
        ).describe_coverage()
        for stage in EXPECTED_STAGE_ORDER:
            self.assertEqual(coverage[stage], "active", stage)

    def test_mix_profile_builders_accept_a_bounded_history(self):
        """Leakage safety of the L2 mix layer rests on this parameter existing."""
        import inspect

        from src.projection.team_pass_mix import build_team_pass_mix_profiles
        from src.projection.team_rush_mix import build_team_rush_mix_profiles

        for builder in (build_team_pass_mix_profiles, build_team_rush_mix_profiles):
            params = inspect.signature(builder).parameters
            self.assertIn("history_seasons", params, builder.__name__)
            # None on the shipped path: every observed season is fair game there.
            self.assertIsNone(params["history_seasons"].default)


if __name__ == "__main__":
    unittest.main()
