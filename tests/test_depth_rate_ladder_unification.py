"""Guards that the Gate B depth-rate ladder is applied by ONE rule everywhere.

The defect these pin down: the ladder used to be applied three different ways.
``depth_gating.apply_depth_chart_gating`` gated it behind the existence of a
curated depth chart (which exists for 2026 and no other season), so the shipped
path applied it for exactly one season; ``fantasy_evaluation`` applied it
unconditionally; ``backtest.py`` never applied it, so
``models/interval_residuals.csv`` and the elite-shrinkage coefficients in
``models/corrections.joblib`` were fit against undiscounted predictions and then
consumed by a path that ships discounted ones.

Written as unittest.TestCase to match the rest of tests/, so both `pytest` and
`python -m unittest discover -s tests` collect them.
"""
import inspect
import unittest

import numpy as np
import pandas as pd

from src.projection import backtest
from src.projection import corrections
from src.projection import fantasy_evaluation as fe
from src.projection import veterans
from src.projection.contracts import (
    DEPTH_RATE_DEEP,
    DEPTH_RATE_LADDER,
    DEPTH_RATE_OFF_CHART,
)
from src.projection.depth_gating import apply_depth_chart_gating
from src.projection.depth_rates import (
    apply_depth_rate_ladder,
    attach_depth_rate_factor,
    depth_rate_factor,
    depth_rate_factors,
)

EMPTY_CHART = pd.DataFrame(
    columns=["team", "position", "depth_rank", "gsis_id", "role", "confidence"])


def _rows():
    """Four veterans spanning every rung class: on-ladder, deep, off-chart."""
    return pd.DataFrame({
        "player_id": ["qb1", "qb2", "wr5", "teX"],
        "position": ["QB", "QB", "WR", "TE"],
        "nfl_depth_rank": [1.0, 2.0, 5.0, np.nan],
        "pred_pg": [10.0, 10.0, 10.0, 10.0],
        "pred_pg_low": [8.0, 8.0, 8.0, 8.0],
        "pred_pg_high": [12.0, 12.0, 12.0, 12.0],
        "low_confidence": [False, False, False, False],
    })


class DepthRateLadderRule(unittest.TestCase):
    def test_factor_depends_only_on_position_and_rank(self):
        """The whole point of the unification: nothing else selects it."""
        self.assertEqual(depth_rate_factor("QB", 2), DEPTH_RATE_LADDER["QB"][2])
        self.assertEqual(depth_rate_factor("RB", 9), DEPTH_RATE_DEEP["RB"])
        self.assertEqual(depth_rate_factor("WR", float("nan")), DEPTH_RATE_OFF_CHART["WR"])
        self.assertEqual(depth_rate_factor("WR", None), DEPTH_RATE_OFF_CHART["WR"])
        # A position the ladder was never fit on gets 1.0, not a guess.
        self.assertEqual(depth_rate_factor("K", 1), 1.0)

    def test_vector_form_matches_scalar_form(self):
        pos = ["QB", "RB", "WR", "TE", "K"]
        ranks = [1.0, 3.0, np.nan, 2.0, 1.0]
        np.testing.assert_allclose(
            depth_rate_factors(pos, ranks),
            [depth_rate_factor(p, r) for p, r in zip(pos, ranks)])

    def test_vector_form_rejects_misaligned_inputs(self):
        with self.assertRaises(ValueError):
            depth_rate_factors(["QB", "RB"], [1.0])

    def test_missing_rank_column_raises_instead_of_defaulting(self):
        """NaN rank carries a real discount, so a missing column must not
        silently become the off-chart factor for every row."""
        with self.assertRaises(ValueError) as ctx:
            attach_depth_rate_factor(_rows().drop(columns=["nfl_depth_rank"]))
        self.assertIn("nfl_depth_rank", str(ctx.exception))

    def test_apply_scales_only_columns_that_exist(self):
        df = _rows().drop(columns=["pred_pg_low", "pred_pg_high"])
        out = apply_depth_rate_ladder(df)
        self.assertNotIn("pred_pg_low", out.columns)
        np.testing.assert_allclose(
            out["pred_pg"], 10.0 * out["role_discount_factor"])


class LadderIsNotGatedOnTheCuratedChart(unittest.TestCase):
    """The regression itself: an empty curated chart must not disable Gate B."""

    def test_gating_no_longer_scales_anything(self):
        """SUPERSEDED CONTRACT, kept as a guard rather than deleted.

        This class used to assert that an empty curated chart still applied
        the Gate B ladder - the regression the unification fixed. The ladder
        itself is now retired: depth reaches a projection as a model INPUT
        (the depth tier in ROLE_FEATURES), not as a multiplier on the model's
        output, because a post-hoc factor keyed on rank cannot condition on
        the player's own usage history and its rungs were calibrated
        rate-to-rate while being applied to a season total.

        What must stay true is that NOTHING here scales a prediction. A future
        edit that reintroduces a multiplier on this path fails here.
        """
        out = apply_depth_chart_gating(_rows(), EMPTY_CHART)
        np.testing.assert_allclose(out["pred_pg"], 10.0)
        np.testing.assert_allclose(out["pred_pg_low"], 8.0)
        np.testing.assert_allclose(out["pred_pg_high"], 12.0)
        np.testing.assert_allclose(out["role_discount_factor"], 1.0)
        self.assertFalse(out["role_discount_applied"].any())

    def test_empty_curated_chart_still_reports_no_curated_knowledge(self):
        """Only the CURATED half no-ops. Those fields must stay honest."""
        out = apply_depth_chart_gating(_rows(), EMPTY_CHART)
        self.assertTrue(out["depth_rank"].isna().all())
        self.assertTrue(out["role"].isna().all())
        self.assertTrue((out["depth_chart_status"] == "not_curated_no_table").all())

    def test_discount_flags_are_inert_on_the_empty_branch(self):
        out = apply_depth_chart_gating(_rows(), EMPTY_CHART)
        self.assertTrue((out["role_discount_factor"] == 1.0).all())
        self.assertFalse(out["role_discount_applied"].any())

    def test_curated_and_uncurated_seasons_agree_on_the_factor(self):
        """Same player: this stage must not scale him either way.

        NOTE the deliberate reversal elsewhere. The curated chart DOES now
        change a player's projection - it supplies the model's depth tier via
        depth_gating.apply_curated_depth_tier, which is the whole point of a
        hand-verified chart. What it must not do is select a post-hoc
        multiplier here, which is what this asserts.
        """
        chart = pd.DataFrame({
            "team": ["AAA"] * 4,
            "position": ["QB", "QB", "WR", "TE"],
            "depth_rank": [1, 2, 3, 1],
            "gsis_id": ["qb1", "qb2", "wr5", "teX"],
            "role": ["starter", "backup", "starter", "starter"],
        })
        curated = apply_depth_chart_gating(_rows(), chart)
        uncurated = apply_depth_chart_gating(_rows(), EMPTY_CHART)
        np.testing.assert_allclose(
            curated["role_discount_factor"], uncurated["role_discount_factor"])
        np.testing.assert_allclose(curated["pred_pg"], uncurated["pred_pg"])


class EveryPathGoesThroughTheSharedHelper(unittest.TestCase):
    """Source-level guards. A future edit that re-implements the ladder inline
    on one path is exactly how the three-way disagreement happened; these fail
    when a path stops importing the shared rule."""

    def test_depth_gating_applies_no_multiplier_on_either_branch(self):
        """The shared-helper guard, inverted for the retired ladder.

        Its purpose is unchanged: stop a future edit from re-implementing a
        depth multiplier inline on one path, which is how the original
        three-way disagreement happened. There is now no multiplier to share,
        so the guard is that neither branch calls one.
        """
        src = inspect.getsource(apply_depth_chart_gating)
        self.assertNotIn("apply_depth_rate_ladder", src)
        self.assertNotIn("depth_rate_factors", src)

    def test_fantasy_evaluation_applies_no_multiplier(self):
        """This harness used to be the ONLY one applying the ladder
        unconditionally - which is what made it the reference the other two
        were unified onto. Now none of the three apply one, and this fold
        must score the model that actually ships."""
        src = inspect.getsource(fe._veteran_forecasts)
        self.assertNotIn("depth_rate_factors", src)
        self.assertIn("ROLE_FEATURES", src)

    def test_backtest_applies_no_multiplier_on_any_prediction_path(self):
        """Inverted, same purpose: the harness must measure what ships.

        These four paths used to be REQUIRED to apply the ladder, because
        interval_residuals.csv and the elite-shrinkage beta were being fit
        against undiscounted predictions and consumed by a discounted one.
        The ladder is gone, so the mismatch now runs the other way - a path
        that still multiplied would be scoring a pipeline production no
        longer has.
        """
        for fn in (backtest.backtest_position_stat,
                   backtest._predict_all_reframed_receiving,
                   backtest.rolling_residual_rows,
                   backtest.backtest_season_totals):
            with self.subTest(fn=fn.__name__):
                src = inspect.getsource(fn)
                self.assertNotIn("depth_ladder_factors", src)
                self.assertNotIn("depth_rate_factors", src)

    def test_backtest_fits_on_the_role_basis(self):
        """The harness has to build its folds the way training does, or the
        held-out numbers describe a different model than the shipped one."""
        for fn in (backtest.backtest_position_stat,
                   backtest.rolling_residual_rows,
                   backtest._predict_all_reframed_receiving):
            with self.subTest(fn=fn.__name__):
                src = inspect.getsource(fn)
                self.assertIn("build_role_transition_pairs", src)
                self.assertIn("role_features_for", src)

    def test_ladder_calibration_artifact_is_gone(self):
        """depth_rate_calibration fit the ladder's own rungs. With no ladder
        there is nothing to calibrate, and leaving it would keep writing an
        authoritative-looking models/ artifact for a retired mechanism."""
        self.assertFalse(hasattr(backtest, "depth_rate_calibration"))
        self.assertFalse(hasattr(backtest, "depth_ladder_factors"))

    def test_corrections_fit_on_the_shipped_basis(self):
        """beta is an ADDITIVE yards/game term. It used to need the ladder
        applied here so the basis it was FIT on matched the basis it was ADDED
        to. With no multiplier on either side the two agree by construction -
        but the label and feature set still have to match production."""
        src = inspect.getsource(corrections.compute_loo_receiving_residuals)
        self.assertNotIn("depth_rate_factors", src)
        self.assertIn("RECEIVING_SHARE_ELIG_LABEL", src)
        self.assertIn("role_features_for", src)




class IntervalsAreBuiltOnTheDiscountedPrediction(unittest.TestCase):
    """backtest.py now fits interval_residuals.csv as (actual - discounted
    pred), so the band belongs on a discounted prediction and must not be
    scaled by the factor a second time."""

    def test_non_reframed_endpoints_add_the_raw_residual(self):
        combined = pd.DataFrame({
            "player_id": ["qb1", "qb2"],
            "position": ["QB", "QB"],
            "stat": ["attempts", "attempts"],
            "pred_pg": [30.0, 20.0],
            "pred_pg_low": [np.nan, np.nan],
            "pred_pg_high": [np.nan, np.nan],
            "interval_low_n_flag": [False, False],
        })
        resid = pd.DataFrame([{
            "position": "QB", "stat": "attempts",
            "resid_low": -5.0, "resid_high": 4.0, "low_n_flag": False,
        }])
        out = veterans._attach_veteran_intervals(combined, resid)
        np.testing.assert_allclose(out["pred_pg_low"], [25.0, 15.0])
        np.testing.assert_allclose(out["pred_pg_high"], [34.0, 24.0])

    def test_missing_residual_row_flags_rather_than_borrows(self):
        combined = pd.DataFrame({
            "player_id": ["qb1"], "position": ["QB"], "stat": ["carries"],
            "pred_pg": [3.0], "pred_pg_low": [np.nan], "pred_pg_high": [np.nan],
            "interval_low_n_flag": [False],
        })
        resid = pd.DataFrame(
            columns=["position", "stat", "resid_low", "resid_high", "low_n_flag"])
        out = veterans._attach_veteran_intervals(combined, resid)
        self.assertTrue(out["pred_pg_low"].isna().all())
        self.assertTrue(bool(out["interval_low_n_flag"].iloc[0]))

    def test_reframed_rows_are_left_to_the_composer(self):
        combined = pd.DataFrame({
            "player_id": ["wr1"], "position": ["WR"], "stat": ["receiving_yards"],
            "pred_pg": [0.2], "pred_pg_low": [np.nan], "pred_pg_high": [np.nan],
            "interval_low_n_flag": [False],
        })
        resid = pd.DataFrame([{
            "position": "WR", "stat": "receiving_yards",
            "resid_low": -10.0, "resid_high": 12.0, "low_n_flag": False,
        }])
        out = veterans._attach_veteran_intervals(combined, resid)
        self.assertTrue(out["pred_pg_low"].isna().all())
        self.assertTrue(out["pred_pg_high"].isna().all())

    def test_veteran_loop_defers_endpoints_until_after_gating(self):
        src = inspect.getsource(veterans.project_veterans)
        self.assertLess(
            src.index("apply_depth_chart_gating(combined"),
            src.index("_attach_veteran_intervals(combined"))


if __name__ == "__main__":
    unittest.main()
