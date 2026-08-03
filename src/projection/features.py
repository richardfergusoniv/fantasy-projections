"""Player-season feature table for QB/RB/WR/TE, 2016-2025.

Granularity: player-SEASON (not player-season-week). The project's target
stats are per-game rates and the opportunity/scheme signals (OC tendencies,
OL quality) are already season-level in Phase 2/3's output tables, so
season is the natural grain - a weekly grain would require re-deriving
week-level share/OL-snap features that don't exist upstream and would mostly
just add noise for a next-SEASON projection task.

Feature groups per player-season:
- opportunity = team volume (oc_tendency_profiles, observed pass_oe/pace/
  personnel/play-action for that team-season) x player share (carry/target
  share of team pbp totals, red-zone carry/target share, average snap %).
- efficiency conditioning = team-season OL quality (src/projection/ol_quality.py,
  2021+ only) and the same scheme features (pass_oe, personnel rates) reused
  from the opportunity block - scheme affects both how much opportunity a
  player gets and how efficiently they convert it, so there's no reason to
  build a second copy of those columns.
- targets = per-game rate for each position's counting stats (see TARGET_STATS).

2021-2025-only scope decision: OL quality (`ol_coefficients_pooled`, keyed
2021-2025) has no equivalent for 2016-2020, so this table is built across
the full 2016-2025 window (rows exist, OL columns are simply NaN pre-2021)
but `src/projection/train.py` restricts the actual train/predict pairs to
2021-2025 for consistency across ALL stat models, not just the OL-conditioned
ones - see PHASE4_REPORT.md for the reasoning and the alternative considered.
"""
import pandas as pd

from src.projection.data_prep import (
    SEASONS, load_weekly_usage, season_aggregate, team_season_pbp_totals,
    player_rz_usage, player_season_snap_pct,
)
from src.projection.ol_quality import team_season_ol_quality

TARGET_STATS = {
    "QB": ["attempts", "completions", "passing_yards", "passing_tds", "interceptions"],
    "RB": ["carries", "rushing_yards", "rushing_tds", "targets", "receptions", "receiving_yards", "receiving_tds"],
    "WR": ["targets", "receptions", "receiving_yards", "receiving_tds"],
    "TE": ["targets", "receptions", "receiving_yards", "receiving_tds"],
}

OC_METRICS = [
    "pass_oe", "pass_oe_neutral", "neutral_sec_per_play", "play_action_rate",
    "personnel_11_rate", "personnel_12_rate", "personnel_21_rate", "personnel_other_rate",
]

FEATURE_COLS = [
    "carry_share", "target_share", "rz_carry_share", "rz_target_share", "snap_pct",
    "ol_pass_protection_score", "ol_run_blocking_score", "ol_confidence_low_churn",
] + OC_METRICS


def build_player_season_features(conn, seasons=SEASONS):
    wu = load_weekly_usage(conn)
    wu = wu[wu["season"].isin(seasons)]
    base = season_aggregate(wu)

    team_totals = team_season_pbp_totals(conn, seasons)
    base = base.merge(team_totals, on=["season", "team"], how="left")

    rz = player_rz_usage(conn, seasons)
    base = base.merge(rz, on=["season", "player_id"], how="left")
    base[["rz_carries", "rz_targets"]] = base[["rz_carries", "rz_targets"]].fillna(0)

    snaps = player_season_snap_pct(conn, seasons)
    base = base.merge(snaps, on=["season", "player_id"], how="left")

    base["carry_share"] = base["carries"] / base["team_rush_attempts"]
    base["target_share"] = base["targets"] / base["team_pass_attempts"]
    base["rz_carry_share"] = base["rz_carries"] / base["team_rz_rush_attempts"]
    base["rz_target_share"] = base["rz_targets"] / base["team_rz_pass_attempts"]

    oc = pd.read_sql(f"select season, team, {', '.join(OC_METRICS)} from oc_tendency_profiles", conn)
    base = base.merge(oc, on=["season", "team"], how="left")

    olq = team_season_ol_quality(conn, seasons)
    base = base.merge(olq, on=["season", "team"], how="left")

    import numpy as np

    for stat_group in TARGET_STATS.values():
        for stat in stat_group:
            base[f"{stat}_pg"] = base[stat] / base["games_played"].replace(0, np.nan)

    return base
