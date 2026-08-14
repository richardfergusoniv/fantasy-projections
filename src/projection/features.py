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

2021-2025-only scope decision: OL quality (exact-season `ol_coefficients`, keyed
2021-2025) has no equivalent for 2016-2020, so this table is built across
the full 2016-2025 window (rows exist, OL columns are simply NaN pre-2021)
but `src/projection/train.py` restricts the actual train/predict pairs to
2021-2025 for consistency across ALL stat models, not just the OL-conditioned
ones - see PHASE4_REPORT.md for the reasoning and the alternative considered.
"""
import numpy as np
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

# Season-N observed rates are among the strongest honest predictors of the
# same player's season-N+1 rate.  They previously existed only as the naive
# benchmark in transitions.py and were withheld from every fitted model.
# Keep one column per stat (rather than a target-dependent placeholder) so
# every saved model has a stable schema that predict.py can construct before
# entering its per-stat loop.
LAG_RATE_FEATURES = [
    f"prior_{stat}_pg"
    for stat in sorted({stat for stats in TARGET_STATS.values() for stat in stats})
]

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
    # Experience only. `draft_round`/`draft_pick` were added here in Phase 5
    # of the consensus-gap work (the draft-capital signal died at the
    # rookie/veteran boundary, so year-2 breakouts were invisible) and
    # REMOVED again after a driver review: once a player has NFL snaps on
    # record, where he was drafted is not what predicts his next season -
    # his observed usage, role, and experience are. Draft capital is a
    # rookie-path input and belongs only there (src/projection/rookies.py
    # sources round/pick from `draft_picks` independently of this table, so
    # the rookie model is unaffected by this removal).
    # career_year = season - rookie_season stays: it is experience, not
    # draft slot, and is the honest carrier of the year-2/year-3 curve.
    "career_year",
] + OC_METRICS + LAG_RATE_FEATURES

# Player-grain rate/share/monopoly features stabilized by the
# games-weighted two-season blending at the end of
# build_player_season_features (Phase 4 of the consensus-gap work - see
# the long comment there for the mechanism and what deliberately stays
# raw). Everything here is "how was this player USED per unit of
# opportunity" - the feature family a 4-game sample measures correctly
# but noisily.
BLEND_FEATURES = [
    "carry_share", "target_share", "rz_carry_share", "rz_target_share", "snap_pct",
    "rz_carry_monopoly", "rz_target_monopoly", "air_yards_share", "adot",
    "qb_designed_run_rate",
]

# Games at which a season counts as fully self-evident: w = min(1,
# games_played / BLEND_GAMES_T). T=8 chosen from the Phase-4 gate's
# sensitivity backtest over {no-blend, 8, 12, 15}, not tuned silently: 8
# touches the fewest rows (only sub-8-game seasons change at all), gave
# the best injury-shortened-WR cohort MAE (9.36 -> 9.14 on gp<=8 WR
# holdout rows) and the best WR-targets/TE-receiving numbers, at a small
# stated cost to headline WR receiving MAE (10.27 -> 10.39) that TE/RB
# receiving gains offset in aggregate.
BLEND_GAMES_T = 8


def build_player_season_features(conn, seasons=SEASONS):
    wu = load_weekly_usage(conn)
    wu = wu[wu["season"].isin(seasons)]
    # The team-attempt model must predict the same official QB-attempt
    # quantity that the player projections emit. nflverse's PBP
    # ``pass_attempt`` flag is a play-family denominator and includes plays
    # that are not charged as an official quarterback attempt (most notably
    # sacks). Sum QB box-score attempts at the WEEK/TEAM grain before the
    # player-season aggregation: a traded QB's full-season attempts must not
    # all be assigned to the team on which he finished the year.
    team_qb_attempts = (
        wu[wu["position"].eq("QB")]
        .groupby(["season", "team"], as_index=False)["attempts"]
        .sum()
        .rename(columns={"attempts": "team_qb_attempts"})
    )
    # Official box-score rushing labels at the same week/team grain as the
    # modeled player universe.  Build before resolving a player to one season
    # team so traded-player production remains with the team that earned it.
    team_rushing = (
        wu[wu["position"].isin(TARGET_STATS)]
        .groupby(["season", "team"], as_index=False)[["carries", "rushing_yards"]]
        .sum()
        .rename(columns={
            "carries": "team_carries",
            "rushing_yards": "team_rushing_yards_modeled",
        })
    )
    schedule_games = team_rushing["season"].apply(_team_season_game_count)
    team_rushing["team_carries_pg"] = team_rushing["team_carries"] / schedule_games
    team_rushing["team_rushing_yards_pg"] = (
        team_rushing["team_rushing_yards_modeled"] / schedule_games
    )
    base = season_aggregate(wu)

    team_totals = team_season_pbp_totals(conn, seasons)
    base = base.merge(team_totals, on=["season", "team"], how="left")

    rz = player_rz_usage(conn, seasons)
    base = base.merge(rz, on=["season", "player_id"], how="left")
    base[["rz_carries", "rz_targets"]] = base[["rz_carries", "rz_targets"]].fillna(0)

    snaps = player_season_snap_pct(conn, seasons)
    base = base.merge(snaps, on=["season", "player_id"], how="left")

    # Share denominators use the team's attempts during the WEEKS THIS
    # PLAYER APPEARED ON OFFENSE, not the team's full-season totals (`team_totals`
    # above, still used for qb_designed_run_rate's team_plays denominator
    # and by other callers of team_season_pbp_totals) - see
    # data_prep.player_active_team_opportunity's docstring for the injury-
    # season dilution bug this fixes (Nabers/LaPorta/Reed 2025). A rare
    # player with zero appearance weeks in a season (shouldn't happen given
    # `base` is built from appearance-week aggregation already, but guarded
    # defensively) would divide by 0 -> NaN, correctly "no real share to
    # compute," not silently zeroed.
    active_opp = player_active_team_opportunity(conn, seasons)
    base = base.merge(active_opp, on=["season", "player_id"], how="left")

    base["carry_share"] = base["carries"] / base["team_rush_attempts_active"].replace(0, np.nan)
    base["target_share"] = base["targets"] / base["team_pass_attempts_active"].replace(0, np.nan)
    base["rz_carry_share"] = base["rz_carries"] / base["team_rz_rush_attempts_active"].replace(0, np.nan)
    base["rz_target_share"] = base["rz_targets"] / base["team_rz_pass_attempts_active"].replace(0, np.nan)

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

    # --- Experience. career_year = season - rookie_season (0 = rookie year,
    # 1 = sophomore); NaN when rookie_season itself is unknown. A static
    # per-player fact - deliberately NOT in BLEND_FEATURES.
    #
    # `draft_round`/`draft_pick` were read here too (Phase 5 of the
    # consensus-gap work) and fed to the player-level models. Removed: draft
    # slot is a rookie-path signal, and once a player has real NFL snaps his
    # own observed usage and experience are what carry the year-2/year-3
    # curve - see the FEATURE_COLS comment. `players.rookie_season` is read
    # from the master roster (not draft_picks), so undrafted and pre-2016
    # players still get a real career_year.
    draft = pd.read_sql(
        "select gsis_id as player_id, rookie_season from players", conn
    )
    base = base.merge(draft, on="player_id", how="left")
    base["career_year"] = base["season"] - base["rookie_season"]
    base = base.drop(columns=["rookie_season"])

    # --- Red-zone MONOPOLY (concentration within the player's own position
    # group), distinct from rz_carry_share/rz_target_share which divide by
    # ALL of the team's red-zone plays across every position. See
    # data_prep.team_season_rz_position_totals's docstring for the full
    # reasoning. Denominators are APPEARANCE-WEEK position-group totals
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
    base["rz_carry_monopoly"] = (
        base["rz_carries"] / base["team_rz_carries_pos_active"].replace(0, np.nan)
    ).fillna(0)
    base["rz_target_monopoly"] = (
        base["rz_targets"] / base["team_rz_targets_pos_active"].replace(0, np.nan)
    ).fillna(0)
    monopoly_cols = ["rz_carry_monopoly", "rz_target_monopoly"]
    impossible_monopoly = (base[monopoly_cols] > 1.0 + 1e-9).any(axis=1)
    if impossible_monopoly.any():
        bad = base.loc[impossible_monopoly, ["player_id", "season", "position"] + monopoly_cols]
        raise ValueError(
            "red-zone monopoly exceeded 1; season-position numerator/denominator mismatch:\n"
            + bad.to_string(index=False)
        )
    base[monopoly_cols] = base[monopoly_cols].clip(lower=0.0, upper=1.0)
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
    base["air_yards_share"] = (
        base["player_air_yards"]
        / base["team_air_yards_active"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0)
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
    # Denominator is the APPEARANCE-WEEK team play count (from
    # player_active_team_opportunity, merged above) - the full-season
    # team_pass_attempts + team_rush_attempts denominator had the same
    # injury-season dilution bug as the monopoly features (Phase 3 of the
    # consensus-gap work): an injury-shortened rushing QB's design-run rate
    # was diluted by team plays from weeks they never took a snap.
    team_plays_active = base["team_pass_attempts_active"] + base["team_rush_attempts_active"]
    base["qb_designed_run_rate"] = (
        base["designed_rush_attempts"] / team_plays_active.replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

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
    base = base.merge(team_qb_attempts, on=["season", "team"], how="left")
    base["team_pass_attempts_pg"] = base["team_qb_attempts"] / base["season"].apply(_team_season_game_count)
    base = base.merge(
        team_rushing[["season", "team", "team_carries_pg", "team_rushing_yards_pg"]],
        on=["season", "team"], how="left",
    )

    # --- Games-weighted two-season feature blending (Phase 4 of the
    # consensus-gap work): for a season with few active games, the
    # rate/share/monopoly features above are correct but computed on a tiny
    # sample - Phase 3 made Nabers' 4-game red-zone monopoly RIGHT (4 of
    # 10, not 4 of 36) at the cost of making it noisy. One mechanism
    # stabilizes all of them at once: blend each rate-shaped feature with
    # the player's own PRIOR season, weighted by how much evidence the
    # current season actually contains - w = min(1, games_played / T),
    # blended = w*f_N + (1-w)*f_{N-1}. Continuous (no threshold cliff),
    # and exactly 1 for any season with >= BLEND_GAMES_T games, so the
    # healthy majority of rows are bit-identical to their raw features.
    # Chosen over a credibility form g/(g+k) (never reaches 1, shifts
    # EVERY row's distribution) and over universal two-season pooling
    # (dilutes genuine role changes - actively harmful for sophomore
    # breakouts).
    #
    # Applied identically in train/backtest/predict by construction (this
    # is the shared feature builder), BEFORE the label loop below - labels
    # (`*_pg`, receiving_yards_share, team_passing_yards_pg) and
    # `naive_pred` are NEVER blended. Deliberately raw: games_played and
    # injury_durability_rate (they ARE the evidence-quantity signal the
    # model should still see), age, peak_receiving_yards_share (already a
    # max-with-prior construct), and every team-grain column (the team
    # played its full season regardless of this player's health). A player
    # with no prior-season row (rookie season, gap year) keeps raw values -
    # nothing is fabricated; per-column, a NaN on either side keeps the
    # current season's raw value (blending needs two real numbers).
    # QB rows are deliberately EXCLUDED from blending: a low-games QB
    # season is usually a benching or a lost QB competition, not an injury
    # - for that archetype the prior season's usage shape is anti-signal
    # (the depth chart decided, not health), and measured directly: with
    # QB rows blended, held-out QB attempts FLIPPED to a naive-baseline
    # loss (7.81 vs 7.43) and QB passing_yards degraded 49.8 -> 55.1;
    # models are strictly per-position, so excluding QB rows reverts the
    # QB models to their pre-blending state exactly while keeping the
    # RB/WR/TE gains.
    prior = base[["player_id", "season"] + BLEND_FEATURES].copy()
    prior["season"] = prior["season"] + 1
    prior = prior.rename(columns={c: f"_prior_{c}" for c in BLEND_FEATURES})
    base = base.merge(prior, on=["player_id", "season"], how="left")
    blend_w = np.minimum(1.0, base["games_played"] / BLEND_GAMES_T)
    blendable = base["position"].isin(["RB", "WR", "TE"])
    for c in BLEND_FEATURES:
        pc = f"_prior_{c}"
        both = blendable & base[c].notna() & base[pc].notna()
        base.loc[both, c] = blend_w[both] * base.loc[both, c] + (1 - blend_w[both]) * base.loc[both, pc]
    base = base.drop(columns=[f"_prior_{c}" for c in BLEND_FEATURES])

    for stat_group in TARGET_STATS.values():
        for stat in stat_group:
            base[f"{stat}_pg"] = base[stat] / base["games_played"].replace(0, np.nan)

    # These are named ``prior_*`` from the perspective of the target season:
    # a season-N feature row is paired with a season-N+1 label downstream.
    # Creating them here also guarantees live prediction gets the identical
    # columns without any special saved-model or predict-time code path.
    for stat in sorted({stat for stats in TARGET_STATS.values() for stat in stats}):
        base[f"prior_{stat}_pg"] = base[f"{stat}_pg"]

    return base
