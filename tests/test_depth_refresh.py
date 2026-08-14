"""Tests for dynamic injury-driven depth-chart refresh."""

import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.depth_chart.events import PUP_GAMES_CAP, detect_injury_events, policy_for_status
from src.depth_chart.live import build_live_depth_chart, load_curated_depth_chart
from src.depth_chart.refresh import events_to_override_rows, refresh_depth_chart
from src.projection.predict import load_status_overrides


class InjuryPolicyTests(unittest.TestCase):
    def test_ir_is_auto_safe_zero_and_remove(self):
        p = policy_for_status("IR")
        self.assertEqual(p["override_mode"], "zero")
        self.assertTrue(p["remove_from_chart"])
        self.assertTrue(p["promote_next"])
        self.assertTrue(p["auto_safe"])

    def test_pup_caps_without_promote(self):
        p = policy_for_status("PUP")
        self.assertEqual(p["override_mode"], "cap")
        self.assertEqual(p["override_games"], PUP_GAMES_CAP)
        self.assertFalse(p["remove_from_chart"])
        self.assertFalse(p["promote_next"])

    def test_out_is_flag_only(self):
        p = policy_for_status("Out")
        self.assertIsNone(p["override_mode"])
        self.assertEqual(p["bucket"], "short_term")


class DetectAndLiveDepthTests(unittest.TestCase):
    def _chart(self):
        return pd.DataFrame([
            dict(season=2026, team="SF", position="TE", depth_rank=1,
                 player_name="George Kittle", gsis_id="00-kittle",
                 role="starter", confidence="high", notes="",
                 usage_share_prior=0.1, usage_share_reviewed=False),
            dict(season=2026, team="SF", position="TE", depth_rank=2,
                 player_name="Backup Te", gsis_id="00-te2",
                 role="backup", confidence="high", notes="",
                 usage_share_prior=0.02, usage_share_reviewed=False),
            dict(season=2026, team="SEA", position="RB", depth_rank=1,
                 player_name="Zach Charbonnet", gsis_id="00-charb",
                 role="starter", confidence="high", notes="",
                 usage_share_prior=0.3, usage_share_reviewed=False),
            dict(season=2026, team="SEA", position="RB", depth_rank=2,
                 player_name="Other Back", gsis_id="00-rb2",
                 role="committee", confidence="high", notes="",
                 usage_share_prior=0.2, usage_share_reviewed=False),
            dict(season=2026, team="SF", position="WR", depth_rank=1,
                 player_name="Mike Evans", gsis_id="00-evans",
                 role="starter", confidence="high", notes="",
                 usage_share_prior=0.1554, usage_share_reviewed=True),
            dict(season=2026, team="SF", position="WR", depth_rank=2,
                 player_name="Deebo Samuel Sr.", gsis_id="00-deebo",
                 role="starter", confidence="high", notes="",
                 usage_share_prior=0.0667, usage_share_reviewed=True),
            dict(season=2026, team="SF", position="WR", depth_rank=3,
                 player_name="Injured Wr", gsis_id="00-irwr",
                 role="starter", confidence="high", notes="",
                 usage_share_prior=0.0386, usage_share_reviewed=True),
        ])

    def _status(self):
        return pd.DataFrame([
            dict(sleeper_id="1", gsis_id="00-irwr", display_name="Injured Wr",
                 name_key="injured wr", team="SF", position="WR",
                 injury_status="IR", injury_body_part="Knee"),
            dict(sleeper_id="2", gsis_id="00-charb", display_name="Zach Charbonnet",
                 name_key="zach charbonnet", team="SEA", position="RB",
                 injury_status="PUP", injury_body_part="Foot"),
            dict(sleeper_id="3", gsis_id=None, display_name="Ricky Pearsall",
                 name_key="ricky pearsall", team="SF", position="WR",
                 injury_status="IR", injury_body_part="Knee"),
            dict(sleeper_id="4", gsis_id="00-out", display_name="Out Guy",
                 name_key="out guy", team="SF", position="WR",
                 injury_status="Out", injury_body_part="Ankle"),
        ])

    def test_ir_on_chart_removes_and_promotes(self):
        events = detect_injury_events(self._status(), self._chart(), as_of="2026-08-14")
        ir = events[events["gsis_id"] == "00-irwr"].iloc[0]
        self.assertTrue(ir["remove_from_chart"])
        self.assertEqual(ir["override_mode"], "zero")
        self.assertTrue(ir["auto_safe"])

        live, applied = build_live_depth_chart(
            self._chart(), events, as_of_date="2026-08-14")
        sf_wr = live[(live.team == "SF") & (live.position == "WR")].sort_values("depth_rank")
        self.assertEqual(list(sf_wr["gsis_id"]), ["00-evans", "00-deebo"])
        self.assertEqual(list(sf_wr["depth_rank"]), [1, 2])
        self.assertAlmostEqual(sf_wr.iloc[0]["usage_share_prior"], 0.1554)
        self.assertAlmostEqual(sf_wr.iloc[1]["usage_share_prior"], 0.0667)
        self.assertIn("00-irwr", set(applied["gsis_id"]))

    def test_pup_caps_without_removing(self):
        events = detect_injury_events(self._status(), self._chart(), as_of="2026-08-14")
        pup = events[events["gsis_id"] == "00-charb"].iloc[0]
        self.assertFalse(pup["remove_from_chart"])
        self.assertEqual(pup["override_mode"], "cap")
        self.assertEqual(pup["override_games"], PUP_GAMES_CAP)

        live, applied = build_live_depth_chart(
            self._chart(), events, as_of_date="2026-08-14")
        sea_rb = live[(live.team == "SEA") & (live.position == "RB")]
        self.assertIn("00-charb", set(sea_rb["gsis_id"]))
        ov = events_to_override_rows(applied, 2026)
        charb = ov[ov.gsis_id == "00-charb"].iloc[0]
        self.assertEqual(charb["mode"], "cap")
        self.assertEqual(float(charb["projected_games"]), PUP_GAMES_CAP)

    def test_off_chart_ir_resolved_via_lookup(self):
        lookup = pd.DataFrame([{
            "gsis_id": "00-0039916",
            "display_name": "Ricky Pearsall",
            "name_key": "ricky pearsall",
            "position": "WR",
            "team": "SF",
        }])
        events = detect_injury_events(
            self._status(), self._chart(), as_of="2026-08-14", id_lookup=lookup)
        pear = events[events["player_name"] == "Ricky Pearsall"].iloc[0]
        self.assertEqual(pear["gsis_id"], "00-0039916")
        self.assertFalse(pear["on_curated_chart"])
        self.assertEqual(pear["action"], "override_zero")
        self.assertFalse(pear["remove_from_chart"])

    def test_dry_run_leaves_curated_unchanged(self):
        curated_before = load_curated_depth_chart(2026)
        with tempfile.TemporaryDirectory() as tmp:
            prop = os.path.join(tmp, "proposals.csv")
            status = self._status()
            with patch("src.depth_chart.refresh.ingest_sleeper_player_status", return_value=status), \
                 patch("src.depth_chart.refresh.load_curated_depth_chart", return_value=self._chart()), \
                 patch("src.depth_chart.refresh._players_id_lookup", return_value=pd.DataFrame()):
                result = refresh_depth_chart(2026, apply=False, proposals_file=prop)
            self.assertFalse(result["applied"])
            self.assertTrue(os.path.exists(prop))
            curated_after = load_curated_depth_chart(2026)
            pd.testing.assert_frame_equal(
                curated_before.reset_index(drop=True),
                curated_after.reset_index(drop=True),
            )


class StatusOverrideAsOfTests(unittest.TestCase):
    def test_as_of_filters_and_keeps_latest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "status_overrides_2026.csv")
            pd.DataFrame([
                dict(season=2026, gsis_id="p1", player_name="A", as_of_date="2026-08-01",
                     mode="cap", projected_games=10, reason="early"),
                dict(season=2026, gsis_id="p1", player_name="A", as_of_date="2026-08-10",
                     mode="cap", projected_games=6, reason="later"),
                dict(season=2026, gsis_id="p2", player_name="B", as_of_date="2026-08-20",
                     mode="zero", projected_games="", reason="future"),
            ]).to_csv(path, index=False)
            with patch("src.projection.predict.STATUS_OVERRIDES_PATH", path):
                all_rows = load_status_overrides(2026)
                self.assertEqual(len(all_rows), 2)  # latest per gsis+mode
                self.assertEqual(
                    float(all_rows.set_index("gsis_id").loc["p1", "projected_games"]), 6.0
                )
                asof = load_status_overrides(2026, as_of="2026-08-05")
                self.assertEqual(list(asof["gsis_id"]), ["p1"])
                self.assertEqual(float(asof.iloc[0]["projected_games"]), 10.0)


if __name__ == "__main__":
    unittest.main()
