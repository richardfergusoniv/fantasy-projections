import unittest
from unittest.mock import patch

import pandas as pd

from src.projection.transitions import ALL_FEATURES, build_availability_pairs


class AvailabilityPositionChangeTest(unittest.TestCase):
    def test_target_depth_lookup_uses_target_position_then_restores_model_position(self):
        source = {c: 0.0 for c in ALL_FEATURES}
        source.update(
            season=2024,
            player_id="position_changer",
            team="AAA",
            position="WR",
            games_played=12.0,
        )
        target = {c: 0.0 for c in ALL_FEATURES}
        target.update(
            season=2025,
            player_id="position_changer",
            team="AAA",
            position="TE",
            games_played=9.0,
        )
        feat = pd.DataFrame([source, target])

        def fake_attach(frame, season):
            self.assertEqual(season, 2025)
            self.assertEqual(frame.loc[0, "position"], "TE")
            out = frame.copy()
            out["target_depth_rank"] = 2.0
            return out

        with patch(
            "src.projection.transitions.attach_availability_depth_rank",
            side_effect=fake_attach,
        ):
            got = build_availability_pairs(feat, "WR", [(2024, 2025)])

        self.assertEqual(got.loc[0, "games_played_to"], 9.0)
        self.assertEqual(got.loc[0, "target_position"], "TE")
        self.assertEqual(got.loc[0, "target_depth_rank"], 2.0)
        self.assertEqual(got.loc[0, "position"], "WR")


if __name__ == "__main__":
    unittest.main()
