"""Rookie-season projection path - deliberately separate from the veteran
LightGBM pipeline in train.py, per the hard project rule: a rookie has no
prior-NFL-season trailing features (the entire premise of the veteran
model), and must not use same-season NFL stats or college production as
inputs (none of which exist in this DB anyway - college production is
simply not modeled here).

Allowed inputs only:
- draft capital (round/pick from `draft_picks`)
- "vacated team opportunity": the target/carry share the rookie's new team
  lost from players who were on that team in season N-1 but are NOT active
  (no games with usage) for that same team in season N. This uses only
  season N-1 data plus who's on/off the roster in season N - never the
  rookie's or anyone's season-N production - so it doesn't leak forward
  information a real preseason projection wouldn't have.

Model: rule-based, not LightGBM. Rookie sample sizes per position x
draft-round-bucket are too small (see PHASE4_REPORT.md) for a tree model
to learn anything but noise, and a rookie's feature vector is structurally
incompatible with the veteran model's inputs anyway - this is exactly the
"distinct path" the spec calls for, not a shrunken version of the same
pipeline. Historical per-game rates for rookies in the same
position/draft-round bucket are averaged, then scaled by the ratio of this
player's team's vacated opportunity to the bucket's historical average
vacated opportunity (clipped to avoid small-sample blowups).
"""
import warnings
import re

import numpy as np
import pandas as pd

from src.projection.data_prep import SEASONS, load_weekly_usage, season_aggregate
from src.projection.depth_history import attach_availability_depth_rank, attach_depth_rank
from src.projection.features import TARGET_STATS

ROUND_BUCKETS = {1: "round_1", 2: "round_2_3", 3: "round_2_3", 4: "round_4_7", 5: "round_4_7",
                  6: "round_4_7", 7: "round_4_7"}
VACATED_CLIP = (0.3, 2.5)
# A curated listing alone does not make a rookie the player who absorbs an
# opening. Match the veteran vacancy rule: backups can be scaled down by a
# poor landing spot, but only confirmed starters/committee players may be
# scaled above their historical draft-bucket baseline.
ROOKIE_BOOST_ELIGIBLE_ROLES = frozenset({"starter", "committee"})
ROOKIE_AVAILABILITY_MIN_CELL = 5

# Modest multiplicative scale applied to a rookie's whole projected line
# based on a discrete combine-athleticism tier (Addendum 4, Part 3) - see
# load_combine_athletic_tier's docstring for the full reasoning. Deliberately
# small (+/-8%/6%, not a big swing) and applied as a THIRD discrete tier
# rather than a continuous score, per the spec's own guidance to prefer
# flags/tiers over precise continuous scaling on a sample this thin (11-132
# historical rookies per bucket, ~85% combine pfr_id join coverage on top of
# that) - fitting a continuous relationship between combine testing and NFL
# per-game production on these sample sizes would be noise-fitting, not
# signal. Chosen as a SCALE on the existing bucket-mean projection (not a
# new bucketing dimension for fit_rookie_baselines) specifically so it does
# NOT further fragment the already-thin (position, round_bucket) training
# samples the way adding a third grouping key would.
ATHLETIC_SCALE = {"above_median": 1.08, "below_median": 0.94, "no_data": 1.0}
ROOKIE_INTERVAL_QUANTILES = (0.10, 0.90)  # same width as the veteran empirical interval, for comparability
ROOKIE_INTERVAL_MIN_N = 20  # bucket sample sizes below this get interval_low_n_flag=True

# draft_picks.team ships PFR-style abbreviations, not nflverse's standard
# ones used everywhere else in this project (weekly/pbp/seasonal_rosters/
# oc_tendency_profiles) - found while validating identify_target_season_rookie_class's
# output: every 2026 draft pick's team fell back to draft_picks.team (their
# placeholder id isn't in seasonal_rosters at all - see that function's
# docstring), and 100% of those came through as GNB/KAN/LAR/LVR/NOR/NWE/SFO/TAM
# instead of GB/KC/LA/LV/NO/NE/SF/TB. Left unfixed, this both mislabels
# rookie rows in the final output AND corrupts team_vacated_opportunity's
# roster-fallback join (a Rams rookie tagged "LAR" would never match the
# team's actual vacated-share row, keyed "LA").
TEAM_ABBR_FIX = {
    "GNB": "GB", "KAN": "KC", "LAR": "LA", "LVR": "LV",
    "NOR": "NO", "NWE": "NE", "SFO": "SF", "TAM": "TB",
}


def _round_bucket(rnd):
    if pd.isna(rnd):
        return "undrafted"
    return ROUND_BUCKETS.get(int(rnd), "round_4_7")


def _normalized_name(value):
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _canonical_gsis_mask(values):
    return values.astype(str).str.match(r"^00-\d{7}$")


def _canonicalize_drafted_ids(drafted, roster, players):
    """Resolve draft-feed IDs without dropping null/placeholder rows.

    Resolution order is stable PFR id, then an exact normalized-name + pick
    + position roster match (team is a tie-breaker, not a hard condition,
    because drafted players can be traded). Every input row survives; a row
    that still cannot resolve receives an explicit deterministic placeholder.
    """
    out = drafted.copy()
    roster = roster.copy()
    players = players.copy()
    out["_input_order"] = np.arange(len(out))
    out["_norm_name"] = out["name"].map(_normalized_name)
    roster["_norm_name"] = roster["name"].map(_normalized_name)

    canonical_roster = roster[_canonical_gsis_mask(roster["player_id"])].copy()
    canonical_players = players[_canonical_gsis_mask(players["canonical_player_id"])].copy()
    roster_by_pfr = (
        canonical_roster.dropna(subset=["pfr_id", "player_id"])
        .drop_duplicates("pfr_id").set_index("pfr_id")["player_id"]
    )
    players_by_pfr = (
        canonical_players.dropna(subset=["pfr_id", "canonical_player_id"])
        .drop_duplicates("pfr_id").set_index("pfr_id")["canonical_player_id"]
    )
    canonical = out["pfr_id"].map(roster_by_pfr)
    canonical = canonical.fillna(out["pfr_id"].map(players_by_pfr))

    unresolved_idx = out.index[canonical.isna()]
    for idx in unresolved_idx:
        row = out.loc[idx]
        candidates = canonical_roster[
            canonical_roster["_norm_name"].eq(row["_norm_name"])
            & canonical_roster["position"].eq(row["position"])
        ]
        if "season" in out.columns and "season" in roster.columns:
            candidates = candidates[candidates["season"].eq(row["season"])]
        if pd.notna(row.get("pick")) and "draft_number" in candidates.columns:
            candidates = candidates[
                pd.to_numeric(candidates["draft_number"], errors="coerce").eq(float(row["pick"]))
            ]
        candidates = candidates.dropna(subset=["player_id"]).drop_duplicates("player_id")
        if len(candidates) > 1 and pd.notna(row.get("draft_team")):
            same_team = candidates[candidates["team"].eq(row["draft_team"])]
            if len(same_team) == 1:
                candidates = same_team
        if len(candidates) == 1:
            canonical.loc[idx] = candidates.iloc[0]["player_id"]

    original = out["player_id"] if "player_id" in out.columns else pd.Series(index=out.index, dtype=object)
    # Keep a feed placeholder only after every canonical route failed; this
    # makes the unresolved state explicit without discarding the source key.
    resolved = canonical.fillna(original.where(_canonical_gsis_mask(original)))
    deterministic = "UNRESOLVED:" + out["pfr_id"].fillna(out["_norm_name"]).astype(str)
    unresolved_fallback = original.fillna(deterministic)
    out["player_id"] = resolved.fillna(unresolved_fallback)
    out = out.sort_values("_input_order").drop(columns=["_input_order", "_norm_name"])
    if len(out) != len(drafted):
        raise AssertionError("drafted-input conservation failed during ID canonicalization")
    return out


def load_draft_capital(conn):
    dp = pd.read_sql(
        "select season as draft_season, gsis_id as player_id, round, pick, position, "
        "team, pfr_player_id as pfr_id, pfr_player_name as name "
        "from draft_picks where position in ('QB','RB','WR','TE')", conn,
    )
    dp["team"] = dp["team"].replace(TEAM_ABBR_FIX)
    roster = pd.read_sql(
        "select player_id, season, team, position, draft_number, player_name as name, pfr_id "
        "from seasonal_rosters", conn,
    ).rename(columns={"season": "draft_season"})
    players = pd.read_sql(
        "select gsis_id as canonical_player_id, pfr_id from players", conn,
    )
    canonical_input = dp.rename(columns={"draft_season": "season", "team": "draft_team"})
    canonical_roster = roster.rename(columns={"draft_season": "season"})
    dp = _canonicalize_drafted_ids(canonical_input, canonical_roster, players).rename(
        columns={"season": "draft_season", "draft_team": "team"}
    )
    dp["round_bucket"] = dp["round"].apply(_round_bucket)
    return dp


def identify_rookie_seasons(conn, seasons=SEASONS):
    """(player_id, season) pairs where `season` is the player's first
    season with any active week in `weekly` (2016-2025 window) AND matches
    their draft season - i.e. no prior-NFL-season row exists at all."""
    draft = load_draft_capital(conn)
    rookies = draft[draft["draft_season"].isin(seasons)].copy()
    rookies = rookies[["player_id", "draft_season", "round", "pick", "round_bucket", "position", "team", "pfr_id", "name"]].rename(
        columns={"draft_season": "season"}
    )
    rookies["rookie_tier"] = "drafted"
    return rookies


def identify_udfa_rookie_seasons(conn, seasons=SEASONS):
    """UDFA counterpart to identify_rookie_seasons: (player_id, season) pairs
    for players with NO draft round (`players.draft_round` is null) whose
    first active `weekly` season equals `players.rookie_season`.

    Deliberately NOT implemented as "absent from draft_picks in this
    season" - draft_picks only covers 2016+, so a veteran actually drafted
    before 2016 (data left-censored) would look falsely "undrafted" the
    moment their first active week in OUR window happens to land in 2016
    (spot-checked: 484 players' first-active-season lands in 2016 under
    that naive definition, nearly all of them established veterans per
    players.rookie_season/draft_year predating 2016, e.g. Tom Brady,
    Drew Brees - not real UDFA rookies). `players.rookie_season` and
    `draft_round` are nflverse master-roster fields, not windowed by this
    project's ingestion range, so they don't have this problem."""
    udfa = pd.read_sql(
        f"select player_id, season, team, position, player_name as name, pfr_id "
        f"from seasonal_rosters where season in ({','.join(map(str, seasons))}) "
        f"and years_exp = 0 and draft_number is null", conn,
    )
    udfa = udfa[udfa["position"].isin(["QB", "RB", "WR", "TE"])].copy()
    udfa = udfa.drop_duplicates(subset=["player_id", "season", "position"])
    udfa["round"] = np.nan
    udfa["pick"] = np.nan
    udfa["round_bucket"] = "undrafted"
    udfa["rookie_tier"] = "udfa"
    return udfa[["player_id", "season", "round", "pick", "round_bucket", "position", "rookie_tier", "team", "pfr_id", "name"]]


def load_combine_athletic_tier(conn):
    """(player_id) -> athletic_score, athletic_tier from `combine_data`
    (Addendum 4, Part 3): a rookie's discrete athletic-testing tier,
    joined via players.pfr_id per this project's established convention
    (players is the general master crosswalk; `ids` is fantasy-platform
    -scoped and known to drop many players - see Phase 1 findings, restated
    in PHASE4_REPORT.md's ingestion notes).

    Not fed into the veteran LightGBM path as a continuous regressor and not
    used to fit a new (position, round_bucket, tier) baseline - see
    ATHLETIC_SCALE's comment for why: the per-(position, round_bucket)
    rookie sample is already thin (11-132 rows, PHASE4_REPORT.md), and
    combine_data's ~85% pfr_id join coverage would shrink any tier-specific
    subgroup further. Instead this produces a simple discrete tier
    (`athletic_tier` in {'above_median','below_median','no_data'}) that
    predict_rookies applies as a modest multiplicative scale on the
    existing bucket-mean x vacated-opportunity projection - refining the
    point estimate without needing its own fitted sample.

    athletic_score = mean of two units-normalized percentile ranks, WITHIN
    POSITION (raw 40 times and vertical jumps aren't comparable across
    QB/RB/WR/TE - a 4.5 forty is elite for a guard-sized TE prospect and
    mediocre for a WR): 40-time percentile (faster => higher percentile) and
    vertical-jump percentile (higher => higher percentile). The percentile
    reference population is every PRIOR-SEASON combine tester at that
    position, not the held-out player's own cohort and not only players who
    made an NFL roster. This makes a historical/held-out score invariant to
    peers from its own or future cohorts while avoiding roster survivorship
    bias. The earliest available season uses its own cohort once because no
    prior reference exists. A player missing one of the
    two metrics still gets a score from whichever one they have (mean with
    skipna); missing both (or no combine_data row/pfr_id join at all)
    produces athletic_tier='no_data' - a real, explicit fallback tier, not
    a silently-dropped row or a NaN scale multiplier.

    Tier cutoff is a simple median split (>=0.5 combined percentile =
    'above_median') - deliberately the simplest possible discretization
    given the "keep it simple, don't overfit the rookie path" mandate,
    not a data-driven optimal cutpoint search on a sample this small.

    Returns the PLAYER_ID-keyed version (via players.pfr_id) - correct for
    any HISTORICAL rookie season, where player_id is always a real gsis_id.
    For the current target season's own drafted rookie class, use
    combine_athletic_scores_by_pfr_id instead and join on pfr_id directly -
    see that function's docstring for the placeholder-gsis_id bug this
    avoids."""
    scores = combine_athletic_scores_by_pfr_id(conn)
    crosswalk = pd.read_sql("select gsis_id as player_id, pfr_id from players where pfr_id is not null", conn)
    out = scores.merge(crosswalk, on="pfr_id", how="inner")
    return out[["player_id", "athletic_score", "athletic_tier"]]


def _load_combine_frame(conn):
    return pd.read_sql(
        "select season, pos, pfr_id, forty, vertical from combine_data "
        "where pos in ('QB','RB','WR','TE') and pfr_id is not null", conn,
    )


def _score_combine_rows(rows, reference):
    """Score rows against a strictly earlier empirical reference sample."""
    scored = rows.copy()
    scored["forty_pctile"] = np.nan
    scored["vertical_pctile"] = np.nan
    for pos, idx in scored.groupby("pos").groups.items():
        ref = reference[reference["pos"].eq(pos)]
        # The earliest available combine season has no prior reference. It
        # uses its own cohort once; every later/held-out season is strictly
        # invariant to the composition of its own cohort.
        if ref.empty:
            ref = scored.loc[idx]
        forty_ref = ref["forty"].dropna().to_numpy()
        vertical_ref = ref["vertical"].dropna().to_numpy()
        if len(forty_ref):
            scored.loc[idx, "forty_pctile"] = scored.loc[idx, "forty"].map(
                lambda value: np.nan if pd.isna(value) else float(np.mean(forty_ref >= value))
            )
        if len(vertical_ref):
            scored.loc[idx, "vertical_pctile"] = scored.loc[idx, "vertical"].map(
                lambda value: np.nan if pd.isna(value) else float(np.mean(vertical_ref <= value))
            )
    scored["athletic_score"] = scored[["forty_pctile", "vertical_pctile"]].mean(axis=1, skipna=True)
    scored = scored.dropna(subset=["athletic_score"])
    scored["athletic_tier"] = np.where(
        scored["athletic_score"] >= 0.5, "above_median", "below_median"
    )
    return scored


def combine_athletic_scores_by_pfr_id(conn, max_reference_season=None):
    """(pfr_id) -> athletic_score, athletic_tier - the pfr_id-keyed form of
    load_combine_athletic_tier's scoring logic, factored out so callers with
    their OWN real pfr_id in hand (e.g. draft_picks.pfr_player_id, which is
    unaffected by the 2026 draft class's placeholder-gsis_id bug - see
    identify_target_season_rookie_class's docstring) can join directly
    without going through the players.gsis_id crosswalk at all."""
    combine = _load_combine_frame(conn)
    if max_reference_season is not None:
        combine = combine[combine["season"] <= max_reference_season]
    pieces = []
    for season in sorted(combine["season"].dropna().unique()):
        rows = combine[combine["season"].eq(season)]
        reference = combine[combine["season"].lt(season)]
        pieces.append(_score_combine_rows(rows, reference))
    combine = pd.concat(pieces, ignore_index=True) if pieces else combine.assign(
        athletic_score=np.nan, athletic_tier=None
    )
    # No duplicate pfr_id rows found in combine_data as of this data pull
    # (spot-checked directly) - drop_duplicates here is a defensive
    # backstop against a future data refresh introducing one, not a known
    # active case.
    combine = combine.sort_values("season").drop_duplicates(subset=["pfr_id"], keep="last")
    return combine[["pfr_id", "athletic_score", "athletic_tier"]]


def team_vacated_opportunity(conn, seasons=SEASONS):
    """(season, team) -> vacated_carry_share, vacated_target_share: the
    fraction of the team's season-(N-1) carries/targets that belonged to
    players who did NOT have an active week for that same team in season N
    (retired, cut, signed elsewhere, or simply lost their role - this
    doesn't distinguish why, only that the volume is no longer theirs).

    Roster-fallback for a genuinely future `season` with zero played games
    (e.g. projecting a season that hasn't started): "active in season N" is
    structurally undeterminable from `weekly` there, so this falls back to
    `seasonal_rosters[season]` (available pre-season, unlike game logs) to
    determine which season-(N-1) players are still on the same roster. This
    is a weaker signal than confirmed game participation (a rostered player
    could still be cut/inactive all year), but it's the best pre-season
    proxy available - the alternative would be reporting zero vacated
    opportunity for every rookie in the target season, which is worse."""
    wu = load_weekly_usage(conn)
    agg = season_aggregate(wu)
    agg = agg[agg["opportunity_games"] > 0]

    # Use a preseason roster snapshot for every season. The former branch
    # used realized season-N participation whenever games existed, leaking
    # benching/injury outcomes into historical rookie features while live
    # forecasts used roster membership. Earliest weekly_rosters is the
    # closest common preseason proxy; seasonal_rosters is only the fallback
    # for a future season whose weekly snapshot is not published yet.
    roster_team = pd.read_sql(
        f"select player_id, season, week, team from weekly_rosters where season in "
        f"({','.join(map(str, seasons))})", conn,
    )
    if not roster_team.empty:
        first_week = roster_team.groupby("season")["week"].transform("min")
        roster_team = roster_team[roster_team["week"] == first_week]
        roster_team = roster_team.drop_duplicates(subset=["player_id", "season"])
    covered = set(roster_team["season"].unique()) if not roster_team.empty else set()
    missing = [s for s in seasons if s not in covered]
    if missing:
        fallback = pd.read_sql(
            f"select player_id, season, team from seasonal_rosters where season in "
            f"({','.join(map(str, missing))})", conn,
        ).drop_duplicates(subset=["player_id", "season"])
        roster_team = pd.concat([roster_team, fallback], ignore_index=True, sort=False)

    rows = []
    for season in seasons:
        prev = agg[agg["season"] == season - 1]
        if prev.empty:
            continue
        curr_team_of_player = roster_team[roster_team["season"] == season].set_index("player_id")["team"]
        prev = prev.copy()
        prev["returning_same_team"] = prev["player_id"].map(curr_team_of_player) == prev["team"]

        g = prev.groupby("team")
        team_totals = g[["carries", "targets", "attempts"]].sum().rename(
            columns={"carries": "prev_team_carries", "targets": "prev_team_targets", "attempts": "prev_team_attempts"}
        )
        returning = prev[prev["returning_same_team"]].groupby("team")[["carries", "targets", "attempts"]].sum().rename(
            columns={"carries": "returning_carries", "targets": "returning_targets", "attempts": "returning_attempts"}
        )
        merged = team_totals.join(returning, how="left").fillna(0)
        merged["vacated_carry_share"] = 1 - merged["returning_carries"] / merged["prev_team_carries"].replace(0, np.nan)
        merged["vacated_target_share"] = 1 - merged["returning_targets"] / merged["prev_team_targets"].replace(0, np.nan)
        # QB-specific proxy - see predict_rookies' docstring for the bug this
        # fixes: vacated_target_share reflects WR/TE/RB receiving-corps
        # turnover, which has nothing to do with whether a backup/rookie QB
        # gets snaps. vacated_attempts_share instead measures how much of
        # the team's PASSING volume belonged to QBs no longer on the roster
        # (attempts is ~exclusively a QB stat, so this isolates QB-room
        # turnover specifically) - the right signal for "is the starting job
        # actually open," not receiver churn.
        merged["vacated_attempts_share"] = 1 - merged["returning_attempts"] / merged["prev_team_attempts"].replace(0, np.nan)
        merged["season"] = season
        rows.append(merged.reset_index()[
            ["season", "team", "vacated_carry_share", "vacated_target_share", "vacated_attempts_share"]
        ])

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["season", "team", "vacated_carry_share", "vacated_target_share", "vacated_attempts_share"]
    )


def build_rookie_dataset(conn, feature_table, seasons=SEASONS):
    """Rookie player-seasons with draft capital + vacated opportunity +
    actual per-game rates (for fitting the historical bucket averages /
    for backtest evaluation) - NOT the veteran feature columns.

    Includes both drafted rookies and UDFA rookies (identify_udfa_rookie_seasons)
    in one combined table so fit_rookie_baselines' groupby('round_bucket')
    naturally produces an 'undrafted' bucket populated by real UDFA per-game
    rates, not the empty bucket PHASE4_REPORT.md flagged (drafted rows with
    round_bucket='undrafted' essentially never occur - that requires a
    draft_picks row with a null round, which doesn't happen in practice).

    Also merges the combine-athleticism tier (Addendum 4, Part 3,
    load_combine_athletic_tier) onto every rookie row - both for
    predict_rookies' target-season scaling AND so backtest.py's historical
    rookie evaluation exercises the exact same combine-scaled code path
    the real 2026 prediction uses, not a separate untested branch. Players
    with no combine match get athletic_tier='no_data' (the real, explicit
    fallback - not a dropped row) rather than NaN."""
    rookies = identify_rookie_seasons(conn, seasons)
    udfa = identify_udfa_rookie_seasons(conn, seasons)
    rookies = pd.concat([rookies, udfa], ignore_index=True, sort=False)
    rookies["_tier_priority"] = rookies["rookie_tier"].map({"drafted": 0, "udfa": 1})
    rookies = (
        rookies.sort_values("_tier_priority")
        .drop_duplicates(["player_id", "season"], keep="first")
        .drop(columns="_tier_priority")
    )
    vacated = team_vacated_opportunity(conn, seasons)
    athletic = load_combine_athletic_tier(conn)

    stat_cols = sorted({s for stats in TARGET_STATS.values() for s in stats})
    pg_cols = [f"{s}_pg" for s in stat_cols]
    actual_cols = ["player_id", "season", "team", "games_played"] + pg_cols
    if "opportunity_games" in feature_table.columns:
        actual_cols.append("opportunity_games")
    actuals = feature_table[actual_cols].rename(columns={"team": "actual_team"})

    # LEFT join keeps the full preseason rookie cohort, including drafted
    # players and camp UDFAs who never record an opportunity. Their rates are
    # undefined (NaN), but games_played is a real zero and must inform the
    # availability estimate.
    df = rookies.merge(actuals, on=["player_id", "season"], how="left")
    df["team"] = df["actual_team"].fillna(df["team"])
    df = df.drop(columns=["actual_team"])
    df["games_played"] = df["games_played"].fillna(0.0)
    if "opportunity_games" not in df.columns:
        df["opportunity_games"] = df["games_played"]
    else:
        df["opportunity_games"] = df["opportunity_games"].fillna(0.0)
    df = df.merge(vacated, on=["season", "team"], how="left")
    df = df.merge(athletic, on="player_id", how="left")
    df["athletic_tier"] = df["athletic_tier"].fillna("no_data")
    ranked = []
    for season, grp in df.groupby("season", sort=False):
        # Keep both meanings explicit. target_depth_rank is harmonized across
        # nflverse's old/new schemas and is the availability input;
        # nfl_depth_rank remains untruncated for the conditional-rate ladder.
        with_availability_rank = attach_availability_depth_rank(
            grp, int(season), conn=conn)
        ranked.append(attach_depth_rank(
            with_availability_rank, int(season), conn=conn))
    if ranked:
        df = pd.concat(ranked, ignore_index=True, sort=False)
    return df


def identify_target_season_rookie_class(conn, target_season):
    """Drafted + UDFA rookie class for `target_season`, built WITHOUT
    requiring any played game in that season. Historical and target cohorts
    are both defined from preseason draft/roster records so zero-game
    rookies remain represented; a genuinely future season simply has no
    outcomes to attach yet.

    Team is resolved from `seasonal_rosters[target_season]` (the most
    current post-draft/free-agency snapshot) with a fallback to
    draft_picks' own `team` column for a drafted player not found there
    (e.g. a crosswalk/ids gap).

    `pfr_id` (Addendum 4, Part 3): carried through here specifically so
    predict.py can join `load_combine_athletic_tier` WITHOUT going through
    `player_id` -> `players.gsis_id` for drafted rookies. Bug found and
    fixed while wiring up the combine feature for the 2026 class: every
    2026 draft_picks.gsis_id is the known placeholder id (not a real
    gsis_id - see the `name` column's docstring above for the same
    root cause), so joining combine data via player_id/players.pfr_id
    silently matched almost nothing for drafted rookies (spot-checked: Drew
    Allar, who DID test at the 2026 combine and has a real
    draft_picks.pfr_player_id of 'AllaDr00', still came back 'no_data' via
    the player_id path, because 'AllaDr00' only exists under his REAL
    gsis_id 00-0041565 in `players`, not under his placeholder draft_picks
    id). draft_picks.pfr_player_id is unaffected by the gsis_id placeholder
    bug and matches combine_data.pfr_id's format directly, so it's used
    here for drafted rookies; UDFA rookies still get pfr_id from
    seasonal_rosters (their player_id IS a real gsis_id, but seasonal_rosters
    already carries pfr_id directly, avoiding an extra crosswalk hop)."""
    drafted = pd.read_sql(
        f"select gsis_id as player_id, round, pick, position, team as draft_team, "
        f"pfr_player_name as name, pfr_player_id as pfr_id "
        f"from draft_picks where season = {target_season} "
        f"and position in ('QB','RB','WR','TE')", conn,
    )
    drafted["season"] = target_season
    drafted_input_n = len(drafted)
    drafted["draft_team"] = drafted["draft_team"].replace(TEAM_ABBR_FIX)
    drafted["round_bucket"] = drafted["round"].apply(_round_bucket)
    drafted["rookie_tier"] = "drafted"

    roster = pd.read_sql(
        f"select player_id, team, position, years_exp, draft_number, player_name as name, pfr_id "
        f"from seasonal_rosters where season = {target_season}", conn,
    ).drop_duplicates(subset=["player_id"])

    players = pd.read_sql(
        "select gsis_id as canonical_player_id, pfr_id from players "
        "where gsis_id is not null and pfr_id is not null", conn,
    )
    drafted = _canonicalize_drafted_ids(drafted, roster.assign(season=target_season), players)

    udfa = roster[
        (roster["years_exp"] == 0) & (roster["draft_number"].isna())
        & (roster["position"].isin(["QB", "RB", "WR", "TE"]))
        & (~roster["player_id"].isin(set(drafted["player_id"])))
    ].copy()
    udfa["round"] = np.nan
    udfa["pick"] = np.nan
    udfa["round_bucket"] = "undrafted"
    udfa["rookie_tier"] = "udfa"

    team_map = roster.set_index("player_id")["team"]
    drafted["team"] = drafted["player_id"].map(team_map).fillna(drafted["draft_team"])
    # Prefer the current roster name after canonicalization, while retaining
    # the draft feed's name as a fallback for players not rostered yet.
    name_map = roster.set_index("player_id")["name"]
    drafted["name"] = drafted["player_id"].map(name_map).fillna(drafted["name"])

    cols = ["player_id", "team", "round", "pick", "position", "round_bucket", "rookie_tier", "name", "pfr_id"]
    combined = pd.concat([drafted[cols], udfa[cols]], ignore_index=True)
    combined["season"] = target_season
    combined["rookie_id_unresolved"] = ~combined["player_id"].astype(str).str.match(r"^00-\d{7}$")
    unresolved = combined[combined["rookie_id_unresolved"]]
    if not unresolved.empty:
        detail = ", ".join(
            f"{row.name} ({row.player_id})" for row in unresolved.itertuples(index=False)
        )
        warnings.warn(
            f"{len(unresolved)} target rookies retain unresolved placeholder IDs: {detail}",
            RuntimeWarning,
        )
    accounted_drafted_n = int(combined["rookie_tier"].eq("drafted").sum())
    if accounted_drafted_n != drafted_input_n:
        raise AssertionError(
            f"drafted-input conservation failed: {drafted_input_n} eligible picks, "
            f"{accounted_drafted_n} drafted rows emitted"
        )
    return combined


def fit_rookie_baselines(rookie_df, train_seasons):
    """Historical (position, round_bucket) -> mean per-game rate + mean
    vacated_carry/target_share, fit ONLY on train_seasons (so the backtest
    holdout season's own rookies never inform their own baseline)."""
    train = rookie_df[rookie_df["season"].isin(train_seasons)].copy()
    pg_cols = [c for c in rookie_df.columns if c.endswith("_pg")]
    vacated_cols = ["vacated_carry_share", "vacated_target_share", "vacated_attempts_share"]
    # Pandas means skip NaN, so per-game rates remain conditional on recording
    # an opportunity while games_played is averaged over the full cohort,
    # including never-played rookies. This separates rate from availability
    # instead of shrinking rate with an external consensus probability.
    baselines = train.groupby(["position", "round_bucket"])[pg_cols + vacated_cols + ["games_played"]].mean()
    baselines = baselines.rename(columns={"games_played": "mean_games_played"})
    counts = train.groupby(["position", "round_bucket"]).size().rename("n_train_rookies")
    rate_counts = train[train["opportunity_games"] > 0].groupby(
        ["position", "round_bucket"]).size().rename("n_rate_rookies")

    if "target_depth_rank" in train.columns:
        train["_depth_band"] = train["target_depth_rank"].apply(
            lambda x: "off_chart" if pd.isna(x) else ("rank_1" if x <= 1 else ("rank_2" if x <= 2 else "rank_3_plus")))
        depth_games = train.pivot_table(
            index=["position", "round_bucket"], columns="_depth_band",
            values="games_played", aggfunc="mean")
        depth_games = depth_games.add_prefix("mean_games_")
        baselines = baselines.join(depth_games, how="left")
        depth_counts = train.pivot_table(
            index=["position", "round_bucket"], columns="_depth_band",
            values="games_played", aggfunc="count")
        baselines = baselines.join(depth_counts.add_prefix("n_games_"), how="left")
    return baselines.join(counts).join(rate_counts)


def predict_rookies(rookie_df, baselines, target_seasons, depth_chart=None):
    """Rule-based projection: bucket mean per-game rate, scaled by this
    player's team's vacated opportunity vs. the bucket's historical average
    vacated opportunity (RB: vacated carry share; WR/TE: vacated target
    share; QB: vacated ATTEMPTS share - see team_vacated_opportunity's
    docstring for why this must be QB-specific and not the receiving-corps
    target share used for WR/TE. Bug found and fixed here: QB rookies used
    to be scaled by vacated_target_share too, which measures WR/TE/RB
    turnover, not QB-room turnover - a team that lost a lot of receivers
    (nothing to do with the QB depth chart) would inflate a buried
    7th-round/UDFA QB's projection toward starter volume purely because of
    that unrelated churn. Verified: Athan Kaliakmanis (WAS, round_4_7,
    clearly a long-shot) was projected ~28 attempts/game before this fix -
    driven by WAS's 0.52 vacated_target_share (real WR/TE turnover) versus
    the round_4_7 QB bucket's historical average, with nothing capping a QB
    scale factor derived from a receiving-corps signal.
    Falls back to the unscaled bucket mean if the bucket has no historical
    rows or the vacated feature is null (e.g. an expansion-style edge case).

    pg_cols is read off `baselines.columns`, not `rookie_df.columns` - the
    target rows (a genuinely future season) have no actual per-game rates
    to speak of, so rookie_df itself may not carry any `*_pg` columns; the
    baselines (fit on historical data) always do.

    `depth_chart` (optional, Phase 6's src/depth_chart/starters_2026.csv -
    empty/None for any other season): caps the UPWARD half of the vacancy
    scale (>1.0) to a rookie the curated table actually lists for their
    (team, position) - same principle predict.py's reassign_team_changers
    already applies to team-changing veterans, applied here to fix its
    rookie-path sibling. Bug this fixes: a rookie/UDFA with zero real
    chance of playing could still get boosted toward starter volume purely
    because their team's ACTUAL new starter opened up a big vacancy (e.g. a
    random UDFA QB scaled toward 45 attempts/game because the team's
    veteran QB left for another team) - the vacancy is real, but nothing
    checked whether THIS specific long-shot player is who's stepping into
    it. Downward scaling (a below-average opportunity) still always
    applies - a below-average situation should reduce even an unconfirmed
    player's already-modest bucket-mean projection.
    """
    pg_cols = [c for c in baselines.columns if c.endswith("_pg")]
    target = rookie_df[rookie_df["season"].isin(target_seasons)].copy()
    boost_eligible_players = set()
    if depth_chart is not None and not depth_chart.empty:
        eligible_chart = depth_chart[
            depth_chart["role"].isin(ROOKIE_BOOST_ELIGIBLE_ROLES)]
        boost_eligible_players = set(zip(
            eligible_chart["gsis_id"], eligible_chart["position"]))

    def project_row(row):
        key = (row["position"], row["round_bucket"])
        if key not in baselines.index:
            return pd.Series({c: np.nan for c in pg_cols} | {"low_confidence": True, "baseline_n": 0})
        b = baselines.loc[key]
        if row["position"] == "RB":
            vac_col = "vacated_carry_share"
        elif row["position"] == "QB":
            vac_col = "vacated_attempts_share"
        else:
            vac_col = "vacated_target_share"
        player_vac, hist_vac = row.get(vac_col), b[vac_col]
        if pd.isna(player_vac) or pd.isna(hist_vac) or hist_vac == 0:
            scale = 1.0
        else:
            scale = np.clip(player_vac / hist_vac, *VACATED_CLIP)
            if (row["player_id"], row["position"]) not in boost_eligible_players:
                scale = min(scale, 1.0)
        preds = {c: b[c] * scale for c in pg_cols}
        preds["rookie_vacancy_scale"] = scale

        # Combine-athleticism scale (Addendum 4, Part 3) - a modest,
        # discrete-tier multiplier on top of the vacated-opportunity scale
        # above, same reasoning as ATHLETIC_SCALE's module-level comment.
        # row.get(...) rather than row["athletic_tier"] so this stays a
        # no-op (scale=1.0, tier reported as 'no_data') for any caller that
        # hasn't merged load_combine_athletic_tier onto its rookie frame -
        # defensive, but every real caller (build_rookie_dataset,
        # project_season's target-class path) does merge it.
        athletic_tier = row.get("athletic_tier", "no_data")
        if pd.isna(athletic_tier):
            athletic_tier = "no_data"
        athletic_scale = ATHLETIC_SCALE.get(athletic_tier, 1.0)
        for c in pg_cols:
            preds[c] = preds[c] * athletic_scale

        # Availability comes entirely from the internal full rookie cohort.
        # Draft bucket supplies the prior and preseason depth rank refines it.
        # target_depth_rank is the schema-harmonized availability feature.
        # Fall back only for older direct callers that predate the dedicated
        # column; when the column exists and is NaN, NaN correctly means
        # off-chart and must not fall through to the untruncated rank.
        rank = (
            row["target_depth_rank"]
            if "target_depth_rank" in row.index
            else row.get("nfl_depth_rank")
        )
        depth_band = "off_chart" if pd.isna(rank) else ("rank_1" if rank <= 1 else ("rank_2" if rank <= 2 else "rank_3_plus"))
        depth_col = f"mean_games_{depth_band}"
        depth_n = b.get(f"n_games_{depth_band}", 0)
        projected_games = b.get(depth_col, np.nan)
        fallback_used = (
            pd.isna(projected_games) or pd.isna(depth_n)
            or depth_n < ROOKIE_AVAILABILITY_MIN_CELL
        )
        if fallback_used:
            projected_games = b.get("mean_games_played", np.nan)
        preds["projected_games"] = np.clip(projected_games, 0, 17) if pd.notna(projected_games) else np.nan
        preds["rookie_depth_band"] = depth_band
        preds["rookie_availability_cell_n"] = depth_n
        preds["rookie_availability_fallback_used"] = fallback_used
        preds["athletic_tier"] = athletic_tier

        preds["low_confidence"] = True
        preds["baseline_n"] = b["n_train_rookies"]
        return pd.Series(preds)

    preds = target.apply(project_row, axis=1)
    identity_cols = [
        "player_id", "season", "team", "position", "round_bucket", "pick",
        "rookie_tier",
    ]
    identity_cols += [
        c for c in ["target_depth_rank", "nfl_depth_rank", "rookie_id_unresolved"]
        if c in target.columns
    ]
    out = pd.concat([target[identity_cols], preds], axis=1)
    return out


def rookie_interval_ratios(rookie_df, baselines, train_seasons, quantiles=ROOKIE_INTERVAL_QUANTILES):
    """Rookie prediction intervals: no naive-baseline backtest comparison
    exists for rookies (no prior season to carry forward as a baseline -
    see PHASE4_REPORT.md), so veteran-style additive residuals from a
    held-out backtest aren't available either. Fallback used instead: for
    each (position, round_bucket, stat), the empirical p10/p90 of
    actual_pg / bucket_mean_pg across all historical rookies in that bucket
    (train_seasons only, same rows fit_rookie_baselines used). Applied
    MULTIPLICATIVELY to a given player's point prediction
    (pred_pg_low = pred_pg * ratio_low) rather than additively, because the
    point prediction itself is bucket_mean * a vacated-opportunity scale
    factor - a multiplicative ratio stays consistent with that scaling
    logic instead of bolting on a flat additive band that ignores how much
    a specific player's prediction was scaled up/down.

    Sample sizes here are the bucket sizes already reported in
    PHASE4_REPORT.md (11-132) - smaller than the veteran backtest's 61-170,
    so buckets below ROOKIE_INTERVAL_MIN_N are flagged via
    interval_low_n_flag rather than silently presented at equal
    confidence to a well-sampled bucket."""
    train = rookie_df[rookie_df["season"].isin(train_seasons) & (rookie_df["games_played"] > 0)]
    pg_cols = [c for c in rookie_df.columns if c.endswith("_pg")]
    stat_names = [c[:-3] for c in pg_cols]

    rows = []
    for (position, bucket), grp in train.groupby(["position", "round_bucket"]):
        n = len(grp)
        if (position, bucket) not in baselines.index:
            continue
        b = baselines.loc[(position, bucket)]
        for stat, pg_col in zip(stat_names, pg_cols):
            mean = b[pg_col]
            vals = grp[pg_col].dropna()
            if pd.isna(mean) or mean == 0 or len(vals) < 3:
                continue
            ratio = vals / mean
            lo, hi = np.quantile(ratio, quantiles)
            rows.append({
                "position": position, "round_bucket": bucket, "stat": stat, "n": n,
                "ratio_low": float(lo), "ratio_high": float(hi),
                "interval_low_n_flag": n < ROOKIE_INTERVAL_MIN_N,
            })
    return pd.DataFrame(rows)
