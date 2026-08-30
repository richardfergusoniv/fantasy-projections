"""Tests for E2 QB context features and temporal invariance."""
import unittest

import numpy as np
import pandas as pd

from src.projection.qb_context import (
    QB_CONTEXT_FEATURES,
    assert_temporal_invariance,
    attach_qb_context,
    build_team_qb_context,
    features_for_model,
    model_artifact_manifest,
)
from src.projection.transitions import ROLE_FEATURES


class QbContextTest(unittest.TestCase):
    def test_qb_context_features_not_in_all_features(self):
        from src.projection.transitions import ALL_FEATURES

        overlap = set(QB_CONTEXT_FEATURES) & set(ALL_FEATURES)
        self.assertFalse(overlap)

    def test_features_for_model_appends_only_when_requested(self):
        base = list(ROLE_FEATURES)
        without = features_for_model("WR", "targets", qb_context=False, base_features=base)
        with_ctx = features_for_model("WR", "targets", qb_context=True, base_features=base)
        self.assertEqual(without, base)
        self.assertEqual(with_ctx[-len(QB_CONTEXT_FEATURES):], list(QB_CONTEXT_FEATURES))

    def test_attach_qb_context_merges_by_team(self):
        players = pd.DataFrame({
            "player_id": ["p1"],
            "preseason_team": ["NYG"],
            "preseason_position": ["WR"],
        })
        ctx = pd.DataFrame({
            "team": ["NYG"],
            **{col: [1.0 if col == "qb_changed" else 0.0] for col in QB_CONTEXT_FEATURES},
        })
        out = attach_qb_context(players.rename(columns={"preseason_team": "team"}), ctx)
        self.assertEqual(int(out["qb_changed"].iloc[0]), 1)

    def test_temporal_invariance_detects_drift(self):
        baseline = pd.DataFrame({"team": ["A"], "qb_changed": [0], "qb_prior_cpoe": [0.1]})
        mutated = baseline.copy()
        mutated["qb_prior_cpoe"] = [0.2]
        self.assertFalse(assert_temporal_invariance(baseline, mutated))

    def test_model_manifest_declares_consumption(self):
        manifest = model_artifact_manifest(consumes_qb_context=True)
        self.assertTrue(manifest["consumes_qb_context"])


if __name__ == "__main__":
    unittest.main()
