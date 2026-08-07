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
    player_rz_usage, player_season_snap_pct, build_player_season_injury_durability,
    team_season_rz_position_totals, player_active_rz_position_opportunity,
    player_season_air_yards,
    player_season_designed_rushes, team_season_opponent_strength,
    player_active_team_opportunity, team_season_yardage_totals,
    player_season_receiving_yards_share, _team_season_game_count,
    player_season_age,
)
from src.projection.ol_quality import team_season_ol_quality

TARGET_STATS = {
    # rushing added post-launch: found via a Sleeper-projection comparison
    # that our QB fantasy points were pure-passing, systematically
    # underrating every mobile/dual-threat QB (Lamar Jackson, Josh Allen,
    # Jayden Daniels, Kyler Murray, Caleb Williams all showed the largest
    # Sleeper-higher deltas). carries/rushing_yards/rushing_tds were
    # already summed into every QB row's raw totals (STAT_COLS in
    # data_prep.py is position-agnostic) and carry_share/rz_carry_share
    # were already computed for QB rows too (build_player_season_features
    # doesn't filter FEATURE_COLS by position) - this was purely a missing
    # entry in this dict, not a missing upstream signal.
    "QB": ["attempts", "completions", "passing_yards", "passing_tds", "interceptions",
           "carries", "rushing_yards", "rushing_tds"],
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
    "injury_durability_rate", "age",  # see data_prep.player_season_age for the age x injury investigation this came from
    "peak_receiving_yards_share",  # max(own share, prior season's) - see build_player_season_features for the Nabers-case investigation this came from
    # Ceiling/concentration features, added to close the bell-cow/alpha
    # under-projection gap (Malik Nabers, Josh Allen, Lamar Jackson, Sam
    # LaPorta, etc. - see the task brief this was built for). None of the
    # existing share features distinguish "the clear #1 option at this
    # position" from "gets some volume, split among several similar
    # players" - these do:
    "rz_carry_monopoly", "rz_target_monopoly",  # concentration within the player's OWN position group's red-zone looks (see data_prep.team_season_rz_position_totals)
    "air_yards_share", "adot",  # true #1 receiving option vs. possession/checkdown role (pbp.air_yards)
    "qb_designed_run_rate",  # scheme-called rush usage, distinct from scramble production (pbp.qb_scramble)
    # Opponent/schedule-strength proxy, added to address the ~18%-high
    # systematic QB volume over-projection (no opponent-agnostic feature
    # existed anywhere in the model before this) - see
    # data_prep.team_season_opponent_strength for the full construction.
    "opp_def_pass_epa_prior", "opp_def_rush_epa_prior",
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

    # Share denominators use the team's attempts during the WEEKS THIS
    # PLAYER WAS ACTIVE, not the team's full-season totals (`team_totals`
    # above, still used for qb_designed_run_rate's team_plays denominator
    # and by other callers of team_season_pbp_totals) - see
    # data_prep.player_active_team_opportunity's docstring for the injury-
    # season dilution bug this fixes (Nabers/LaPorta/Reed 2025). A rare
    # player with zero active weeks in a season (shouldn't happen given
    # `base` is built from active-week aggregation already, but guarded
    # defensively) would divide by 0 -> NaN, correctly "no real share to
    # compute," not silently zeroed.
    active_opp = player_active_team_opportunity(conn, seasons)
    base = base.merge(active_opp, on=["season", "player_id"], how="left")

    base["carry_share"] = base["carries"] / base["team_rush_attempts_active"]
    base["target_share"] = base["targets"] / base["team_pass_attempts_active"]
    base["rz_carry_share"] = base["rz_carries"] / base["team_rz_rush_attempts_active"]
    base["rz_target_share"] = base["rz_targets"] / base["team_rz_pass_attempts_active"]

    oc = pd.read_sql(f"select season, team, {', '.join(OC_METRICS)} from oc_tendency_profiles", conn)
    base = base.merge(oc, on=["season", "team"], how="left")

    olq = team_season_ol_quality(conn, seasons)
    base = base.merge(olq, on=["season", "team"], how="left")

    injury = build_player_season_injury_durability(conn, seasons)
    base = base.merge(injury, on=["season", "player_id"], how="left")
    # Every row in `base` originates from load_weekly_usage/season_aggregate,
    # the exact same universe build_player_season_injury_durability is built
    # from - so this merge should always find a match. fillna(0) here is a
    # defensive backstop (0 = "no missed games, no injury-report weeks
    # found," a real meaningful value), not a silent cover for an expected
    # gap - if this ever actually fires it means the two functions'
    # universes have diverged and is worth investigating.
    base["injury_durability_rate"] = base["injury_durability_rate"].fillna(0)

    age = player_season_age(conn, seasons)
    base = base.merge(age, on=["season", "player_id"], how="left")
    # NOT filled - see player_season_age's docstring. A missing age is a
    # real, if uncommon, roster-data gap, not "age 0."

    import numpy as np

    # --- Red-zone MONOPOLY (concentration within the player's own position
    # group), distinct from rz_carry_share/rz_target_share which divide by
    # ALL of the team's red-zone plays across every position. See
    # data_prep.team_season_rz_position_totals's docstring for the full
    # reasoning. Denominators are ACTIVE-WEEKS position-group totals
    # (player_active_rz_position_opportunity - Phase 3 of the consensus-gap
    # work): the original full-season denominators diluted injury-shortened
    # alpha seasons exactly the way carry_share/target_share used to before
    # player_active_team_opportunity, and this diluted monopoly was
    # measured as the single largest driver of Malik Nabers' 2026
    # under-projection. The full-season totals (team_season_rz_position_
    # totals) are still merged for `base` readers/diagnostics, just no
    # longer the feature denominator.
    rz_pos_totals = team_season_rz_position_totals(conn, seasons)
    base = base.merge(rz_pos_totals, on=["season", "team", "position"], how="left")
    base[["team_rz_carries_pos", "team_rz_targets_pos"]] = base[
        ["team_rz_carries_pos", "team_rz_targets_pos"]
    ].fillna(0)
    rz_pos_active = player_active_rz_position_opportunity(conn, seasons)
    base = base.merge(rz_pos_active, on=["player_id", "season"], how="left")
    base[["team_rz_carries_pos_active", "team_rz_targets_pos_active"]] = base[
        ["team_rz_carries_pos_active", "team_rz_targets_pos_active"]
    ].fillna(0)
    base["rz_carry_monopoly"] = (base["rz_carries"] / base["team_rz_carries_pos_active"]).fillna(0)
    base["rz_target_monopoly"] = (base["rz_targets"] / base["team_rz_targets_pos_active"]).fillna(0)
    # Both ratios are 0/0 -> NaN exactly when the player's own count AND the
    # active-weeks position-group total are both 0 (a player's own red-zone
    # touch can only happen in a week they were active, so the denominator
    # is >= the numerator by construction) - filled to 0 deliberately:
    # "0 real red-zone looks existed for this position group in the weeks
    # this player was on the field" means there is nothing to be a monopoly
    # over, which is a real 0, not a missing value.

    # --- Air yards share / aDOT: true #1 receiving option vs. a
    # possession/checkdown role, which target_share/rz_target_share alone
    # can't separate (a low-aDOT receiver can still carry a healthy target
    # share). See data_prep.player_season_air_yards's docstring for the
    # sack/air_yards-null handling.
    player_ay, _team_ay_full_season = player_season_air_yards(conn, seasons)
    base = base.merge(player_ay, on=["season", "player_id"], how="left")
    # player_air_yards: a player with zero real (non-null-air_yards) targets
    # that season genuinely accumulated 0 air yards - fillna(0) is a real
    # value, not a cover for a failed join.
    base["player_air_yards"] = base["player_air_yards"].fillna(0)
    # Denominator is team_air_yards_active (from player_active_team_opportunity,
    # merged above alongside the attempt-share columns) - the same
    # games-played-aware fix as carry_share/target_share, NOT the
    # full-season team_air_yards this function also returns (kept unused
    # here, `_team_ay_full_season` - it has the identical injury-season
    # dilution bug carry_share/target_share had, found while validating
    # that fix; see player_active_team_opportunity's docstring).
    base["air_yards_share"] = (base["player_air_yards"] / base["team_air_yards_active"]).fillna(0)
    # `adot` (player_adot from player_season_air_yards) is deliberately LEFT
    # AS NaN when a player has zero targets with a real air_yards value that
    # season (e.g. a QB, or a RB/WR/TE who genuinely never saw a target) -
    # filling it to 0 would falsely claim "targeted exclusively behind the
    # line of scrimmage" for a player who was never targeted at all. This
    # is a real, reportable gap for non-receiving-relevant rows (QBs will be
    # essentially all-NaN here, by construction, since air_yards is credited
    # to the receiver not the passer), not a silent fill.
    base = base.rename(columns={"player_adot": "adot"})

    # --- QB designed-run-rate: scheme-called rush usage as a fraction of
    # team offensive plays, distinct from scramble production and from raw
    # rushing_yards volume. See data_prep.player_season_designed_rushes's
    # docstring for why this is only cleanly meaningful for QB rows (it's
    # highly collinear with carry_share for RB/WR/TE, included generically
    # anyway since FEATURE_COLS isn't filtered by position elsewhere).
    designed = player_season_designed_rushes(conn, seasons)
    base = base.merge(designed, on=["season", "player_id"], how="left")
    base["designed_rush_attempts"] = base["designed_rush_attempts"].fillna(0)
    # Denominator is the ACTIVE-WEEKS team play count (from
    # player_active_team_opportunity, merged above) - the full-season
    # team_pass_attempts + team_rush_attempts denominator had the same
    # injury-season dilution bug as the monopoly features (Phase 3 of the
    # consensus-gap work): an injury-shortened rushing QB's design-run rate
    # was diluted by team plays from weeks they never took a snap.
    team_plays_active = base["team_pass_attempts_active"] + base["team_rush_attempts_active"]
    base["qb_designed_run_rate"] = (base["designed_rush_attempts"] / team_plays_active).fillna(0)

    # --- Opponent/schedule-strength: a team-season proxy (not player-
    # specific) for how tough the team's actual schedule was, built from
    # each opponent's PRIOR-season pass/rush defensive EPA/play allowed. See
    # data_prep.team_season_opponent_strength's docstring for the full
    # construction and the stated NaN behavior for 2016 (no season-2015 data
    # exists in this DB).
    opp_strength = team_season_opponent_strength(conn, seasons)
    base = base.merge(opp_strength, on=["season", "team"], how="left")
    # Deliberately NOT filled - see team_season_opponent_strength's
    # docstring. Only 2016 team-seasons are affected, and training
    # (transitions.py) never uses season_from=2016, so this doesn't reach
    # the actual production/backtest models, but is left genuinely NaN
    # rather than silently zeroed for anyone reading `base` directly.

    # --- Joint/multi-output Phase A additions (team-total x player-share
    # decomposition - see the plan this was built from). Both are LABELS,
    # not input FEATURE_COLS: `receiving_yards_share` is what
    # WR_receiving_yards/TE_receiving_yards/RB_receiving_yards are trained
    # to predict instead of receiving_yards_pg directly;
    # `team_passing_yards_pg` is what the new team-season
    # team_passing_yards model (train.py) is trained to predict. Neither
    # belongs in FEATURE_COLS - a model can't be trained to predict its own
    # input.
    recv_share = player_season_receiving_yards_share(conn, seasons)
    base = base.merge(recv_share, on=["season", "player_id"], how="left")

    # --- peak_receiving_yards_share (FEATURE_COLS input, not a label):
    # max(this season's own receiving_yards_share, the immediately PRIOR
    # season's) - built from the age/injury investigation (Malik Nabers'
    # 2025: 4 games, torn ACL, diluted current-season share vs. a real
    # elite 2024 rookie share). Deliberately DIFFERENT from the multi-
    # season trend/slope feature already investigated and ruled out for
    # exactly this case (a slope from a monster prior season INTO an
    # injury-shortened one reads as decline, not proof of ceiling) - this
    # captures "this player has PROVEN they can do this," not a direction.
    # Tested directly before adding (not assumed): for the subgroup of
    # injury-shortened seasons (games_played<10) where the prior season's
    # share notably exceeds the diluted current one, adding this feature
    # reduced held-out reconstruction MAE for WR (10.40->10.22 overall,
    # 8.96->8.78 on the subgroup) and RB (5.72->5.52 overall, 4.40->4.23
    # subgroup); TE improved on the subgroup (6.23->5.73) but slightly
    # regressed overall (7.48->7.61 - likely small-sample noise given only
    # 97 held-out TE rows, reported honestly rather than hidden).
    prior_share = base[["player_id", "season", "receiving_yards_share"]].copy()
    prior_share["season"] = prior_share["season"] + 1
    prior_share = prior_share.rename(columns={"receiving_yards_share": "prior_receiving_yards_share"})
    base = base.merge(prior_share, on=["player_id", "season"], how="left")
    base["peak_receiving_yards_share"] = base[["receiving_yards_share", "prior_receiving_yards_share"]].max(axis=1)
    # No prior season on record (rookie, or a gap year) -> peak equals the
    # current season's own share (no extra information to add), not a
    # fabricated 0 or a silently dropped NaN.
    base["peak_receiving_yards_share"] = base["peak_receiving_yards_share"].fillna(base["receiving_yards_share"])
    base = base.drop(columns=["prior_receiving_yards_share"])

    # `team_totals` (team_season_pbp_totals, merged earlier) covers attempt
    # counts only, not yardage - team_season_yardage_totals is a separate
    # merge, normalized by the team's schedule length (same
    # _team_season_game_count helper injury_durability_rate already uses)
    # to get a per-game rate, consistent with every other `_pg` label.
    team_yds = team_season_yardage_totals(conn, seasons)
    base = base.merge(team_yds, on=["season", "team"], how="left")
    base["team_passing_yards_pg"] = base["team_passing_yards"] / base["season"].apply(_team_season_game_count)

    for stat_group in TARGET_STATS.values():
        for stat in stat_group:
            base[f"{stat}_pg"] = base[stat] / base["games_played"].replace(0, np.nan)

    return base
