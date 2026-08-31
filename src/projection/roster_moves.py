"""Target-season roster resolution, vacancy boosts, and team-changer reassignment.

May import team_vacated_opportunity from rookies. Must not be imported by rookies.
Does not import predict.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.projection.contracts import (
    BOOST_ELIGIBLE_ROLES,
    INCUMBENT_VACANCY_ALPHA,
    INCUMBENT_VACANCY_NET_CLIP,
    INCUMBENT_VACANCY_SCALE_CAP,
    TEAM_CHANGE_SHARE_CLIP,
    TEAM_CHANGE_VACANCY_ALPHA,
)
from src.projection.data_prep import team_season_opponent_strength
from src.projection.features import OC_METRICS
from src.projection.ol_quality import team_season_ol_quality
from src.projection.rookies import TEAM_ABBR_FIX, team_vacated_opportunity


def load_target_roster_map(conn, target_season):
    """player_id -> (team, status) from seasonal_rosters[target_season], the
    source of truth for "what team is this player actually on for the
    season being projected" (fixes the Cousins-on-ATL/Murray-on-MIN bug -
    the old code used the player's season_from RESOLVED team for
    everything). A player can have >1 roster row in a season (practice
    squad stint, in-season cut/re-sign); status='ACT' is preferred when
    present, per the spec's explicit guidance. Non-ACT statuses (RES/PUP,
    CUT, RET, E14) are NOT dropped or treated as "no longer relevant" -
    IR/PUP doesn't mean out for a season that hasn't started, and even a
    late cut is still worth surfacing rather than silently vanishing - they
    are kept with their roster team and flagged via the output's
    `roster_status` column so a reader can judge for themselves (e.g. a
    RET status probably means don't trust this row at all)."""
    df = pd.read_sql(f"select player_id, team, status from seasonal_rosters where season={target_season}", conn)
    df["team"] = df["team"].replace(TEAM_ABBR_FIX)
    df["is_act"] = (df["status"] == "ACT").astype(int)
    df = df.sort_values(["player_id", "is_act"], ascending=[True, False])
    df = df.drop_duplicates(subset=["player_id"], keep="first")
    return df.set_index("player_id")[["team", "status"]]


# opp_def_pass_epa_prior/opp_def_rush_epa_prior (added alongside the
# ceiling/concentration features, see features.py's FEATURE_COLS comment)
# are team-season schedule-strength context exactly like the OL/OC columns
# below - a team-changer's opponent slate for the target season depends on
# their NEW team's schedule, not their old one, so they belong in the same
# re-pointing list.
TEAM_CONTEXT_COLS = (
    ["ol_pass_protection_score", "ol_run_blocking_score", "ol_confidence_low_churn"]
    + OC_METRICS
    + ["opp_def_pass_epa_prior", "opp_def_rush_epa_prior"]
)


# Roles the curated depth chart confirms as boost-eligible for vacancy scale > 1.
# Constants live in contracts.py; comments retained in DEPTH_CHART_ALLOCATION notes.


def _incoming_volume_share(df, changed):
    """Per destination team, the source-season carry/target volume walking
    IN, expressed on team_vacated_opportunity's own raw-count basis (the
    arrivals' prior totals over the destination team's prior total) so the
    two are directly subtractable.

    Shared by both vacancy adjustments so they cannot drift apart: the
    incumbent boost subtracts this whole quantity (arrivals absorb the
    room, so returners shouldn't be credited with it), while the
    team-changer scale subtracts it MINUS the player's own contribution
    (each arrival should see the room net of its COMPETITORS, never net
    of itself). Assumes df["team"] is still the source-season team, i.e.
    it must be called before the team reassignment at the end of
    reassign_team_changers.

    Also returns per-position target shares under ``targets_by_position`` so
    hierarchical L2 vacancy can net WR arrivals against WR vacancy only.
    """
    prev_team = df.groupby("team")[["carries", "targets"]].sum()
    incoming = df[changed].groupby("team_target")[["carries", "targets"]].sum()
    out = {}
    for col in ["carries", "targets"]:
        share = (incoming[col] / prev_team[col]).replace([np.inf, -np.inf], np.nan)
        own = df[col] / df["team_target"].map(prev_team[col]).replace(0, np.nan)
        out[col] = (share.fillna(0.0), own.fillna(0.0))

    prev_pos = df.groupby(["team", "position"])["targets"].sum()
    inc_pos = df[changed].groupby(["team_target", "position"])["targets"].sum()
    by_pos = {}
    for position in ("WR", "TE", "RB"):
        prev_p = df[df["position"].eq(position)].groupby("team")["targets"].sum()
        inc_p = df[changed & df["position"].eq(position)].groupby("team_target")["targets"].sum()
        share_p = (inc_p / prev_p).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        own_p = pd.Series(0.0, index=df.index)
        mask = df["position"].eq(position)
        own_p.loc[mask] = (
            df.loc[mask, "targets"]
            / df.loc[mask, "team_target"].map(prev_p).replace(0, np.nan)
        ).fillna(0.0)
        by_pos[position] = (share_p, own_p)
    out["targets_by_position"] = by_pos
    return out


def _effective_vacated_target(df, vacated_frame):
    """Row-wise target vacancy: position-group when available, else team-wide."""
    base = df["team_target"].map(vacated_frame["vacated_target_share"])
    out = base.copy()
    for position, col in (
        ("WR", "vacated_target_share_wr"),
        ("TE", "vacated_target_share_te"),
        ("RB", "vacated_target_share_rb"),
    ):
        if col not in vacated_frame.columns:
            continue
        mapped = df["team_target"].map(vacated_frame[col])
        hit = df["position"].eq(position) & mapped.notna()
        out = out.where(~hit, mapped)
    return out


def _attach_rookie_residual_vacancy(conn, df, target_season, changed):
    """What fraction of a team's vacated opportunity is still unclaimed once
    the veteran paths have taken their cut.

    The rookie path is the only claimant on a team's vacancy that never nets
    itself against the others. Arrivals net out their competitors
    (reassign_team_changers) and incumbents net out arrivals
    (apply_incumbent_vacancy_boost), but a rookie is handed the team's ENTIRE
    gross vacancy - so the same opening is spent twice. Philadelphia loses
    A.J. Brown, DeVonta Smith is credited for absorbing part of it as a
    curated incumbent, and Makai Lemon is then scaled by the whole thing
    anyway, landing at 118.9 targets against Smith's 97.9.

    In share-of-team-volume units the accounting is:

        v_net    = gross_vacated - arrivals_incoming     (already the basis
                   apply_incumbent_vacancy_boost credits incumbents on)
        absorbed = INCUMBENT_VACANCY_ALPHA * v_net       (their measured cut)
        residual = v_net - absorbed = v_net * (1 - alpha)

    and this returns residual / gross - the share of the boost a rookie can
    still honestly claim. It falls out to 1.0 exactly when there are no
    arrivals and no eligible incumbents to credit, i.e. when the rookie
    really is the only claimant. Carries use the same residual formula with
    INCUMBENT_VACANCY_ALPHA['carry'] (measured 1.0; see
    RB_CARRY_VACANCY_2026-08-14.md).

    Computed here rather than recomputed later because `team_target` and
    `changed` are both fully resolved at this point, and re-deriving either
    one downstream is exactly how two code paths drift apart.
    """
    for kind in ("carry", "target"):
        df[f"rookie_residual_{kind}_fraction"] = 1.0
    vacated = team_vacated_opportunity(conn, [target_season])
    vacated = vacated[vacated["season"] == target_season].set_index("team")
    if vacated.empty:
        return df
    incoming = _incoming_volume_share(df, changed) if changed.any() else None
    for col, vac_col, kind in (
        ("carries", "vacated_carry_share", "carry"),
        ("targets", "vacated_target_share", "target"),
    ):
        gross = vacated[vac_col]
        if incoming is None:
            absorbed = pd.Series(0.0, index=gross.index)
        else:
            absorbed = incoming[col][0].reindex(gross.index).fillna(0.0)
        v_net = (gross - absorbed).clip(lower=0.0)
        residual = v_net * (1.0 - INCUMBENT_VACANCY_ALPHA[kind])
        fraction = (residual / gross.replace(0, np.nan)).clip(0.0, 1.0)
        df[f"rookie_residual_{kind}_fraction"] = (
            df["team_target"].map(fraction).fillna(1.0)
        )
    return df


def drop_players_absent_from_target_season(conn, df, depth_chart, target_season):
    """Drop players who have NO target_season roster row at all AND are
    not vouched for by the curated depth chart - players who have left the
    league, not players having a quiet season.

    The case that surfaced this: Philip Rivers, five years retired, came
    out of retirement for Indianapolis in weeks 15-17 of 2025 (28/37/32
    attempts, 544 yards, 4 TDs) during a QB emergency. That is a real
    event and the data recording it is correct - an earlier pass through
    this project wrongly wrote it off as an upstream nflverse ID
    mislabeling, which it is not. But a genuine 3-game emergency stint at
    age 44 is one of the great outliers in league history, not the basis
    for a 2026 projection, and he duly showed up in the deliverable with
    a 37.3 passing-yards-per-game line on a team he is not on.

    The leak is structural, not Rivers-specific, and this fixes the
    class: reassign_team_changers' `no_info` branch keeps a player's OLD
    team when they cannot be found in target_season's roster, which is
    the right call for a crosswalk gap but silently converts "out of the
    league" into "still on last year's team." 65 players reached the 2026
    output that way.

    Two guards keep this from becoming its own silent-deletion bug - the
    failure mode this project has already been burned by once (see
    project_veterans' docstring on the rookie-filter bug):

    1. The curated depth chart WINS over a missing roster row. A player
       our own hand research affirmatively places on a 2026 roster is
       kept even with no roster row, because the human signal is stronger
       than the absence of a machine one - the same precedence already
       used when a curated starter overrides the vacancy heuristic. This
       is load-bearing: Deebo Samuel and Stefon Diggs are both curated
       starters with no 2026 roster row, and a blanket rule would have
       deleted two legitimate starter projections.
    2. Every dropped player is PRINTED, with the count and the most
       significant names. A drop that is announced is auditable; a drop
       that is silent is the bug.

    Runs before the models, not as an output filter, so a departed player
    also stops consuming his old team's receiving-share budget."""
    if "roster_status" not in df.columns:
        return df
    absent = df["roster_status"].isna()
    if not absent.any():
        return df

    curated_ids = set()
    if not depth_chart.empty:
        curated_ids = set(depth_chart["gsis_id"].dropna())
    to_drop = absent & ~df["player_id"].isin(curated_ids)
    if not to_drop.any():
        return df

    names = pd.read_sql("select gsis_id as player_id, display_name from players", conn)
    names = names.drop_duplicates("player_id").set_index("player_id")["display_name"]

    dropped = df[to_drop].copy()
    dropped["display_name"] = dropped["player_id"].map(names).fillna(dropped["player_id"])
    kept_anyway = df[absent & df["player_id"].isin(curated_ids)].copy()
    kept_anyway["display_name"] = kept_anyway["player_id"].map(names).fillna(kept_anyway["player_id"])

    print(
        f"Dropped {len(dropped)} player(s) with no {target_season} roster row and no curated "
        f"depth-chart entry (out of the league, not merely low-usage):"
    )
    # Ranked by prior-season receiving/rushing volume, not games - the
    # ones worth a human second look are the ones who were PRODUCTIVE.
    dropped["_vol"] = dropped[["receiving_yards", "rushing_yards", "passing_yards"]].fillna(0).max(axis=1)
    for _, r in dropped.nlargest(min(10, len(dropped)), "_vol").iterrows():
        print(f"    {r['display_name']} ({r['position']}, last seen {r['team']}, "
              f"{r['games_played']:.0f} games in {target_season - 1})")
    if len(dropped) > 10:
        print(f"    ... and {len(dropped) - 10} more")
    if not kept_anyway.empty:
        print(f"  KEPT {len(kept_anyway)} player(s) with no roster row but a curated depth-chart entry "
              f"(human research outranks a missing roster row): {', '.join(sorted(kept_anyway['display_name']))}")
    return df[~to_drop]


def apply_incumbent_vacancy_boost(conn, df, target_season, depth_chart, changed):
    """Credit a team's RETURNING players with the opportunity its departed
    players left behind.

    The gap this closes (found by the project owner asking who absorbs
    Green Bay's work after Romeo Doubs and Dontayvion Wicks left):
    reassign_team_changers' vacated-opportunity scaling only ever fires
    for players who CHANGED TEAMS. A player who stays put has his share
    features read straight off the source season - a season in which the
    now-departed teammates were still there taking their cut - so the
    model gives their vacated volume to nobody at all.

    League-wide this hides, because most teams replace departures from
    outside and those arrivals DO get boosted: across the 2026 slate, a
    team's vacated target share correlates POSITIVELY (+0.48) with how
    much of its passing offense we allocate. It only surfaces on a team
    that replaces from within. Green Bay lost 132 of 462 targets (28.6%)
    and re-signed nobody of note, and ended up allocating 93.1% of its
    projected team passing yards - second-lowest in the league against a
    111.5% mean - with three of the five worst remaining WR consensus
    gaps (Reed, Golden, Watson) all on that one roster.

    Mechanism, with each piece measured rather than assumed:

    1. NET vacancy, not gross. `team_vacated_opportunity` measures volume
       that LEFT; from it we subtract the volume walking IN (the
       source-season carries/targets of players joining this team, over
       the team's own source-season total - the same raw-count basis
       vacated_* uses, so the two are subtractable). Without this a team
       that lost three starters and signed three would boost its
       incumbents AND its arrivals for the same opening, double-counting
       the room.
    2. Proportional redistribution, damped: scale = 1 + alpha * v_net /
       (1 - v_net), with alpha per opportunity type - see
       INCUMBENT_VACANCY_ALPHA for the fitted values and the evidence.
    3. Depth-chart gated, upward only. Only curated `starter`/`committee`
       incumbents receive it (BOOST_ELIGIBLE_ROLES - the same guard that
       stopped a confirmed backup from out-projecting his own starter in
       the Phase-6 Gainwell bug); everyone else keeps their share
       untouched. The boost never reduces a share, so a low-vacancy team
       is a strict no-op rather than a penalty - incumbent shares DO decay
       on stable teams (observed 0.907x median), but that is ordinary
       regression the models already learn, and re-applying it here would
       double-count it.

    LIMITATIONS, stated: (a) incoming ROOKIES are not subtracted in step
    1 - they have no prior NFL volume to measure and their predictions
    don't exist yet at this point in the pipeline; the share-sum cap at
    composition time is the backstop, and the team-allocation diagnostic
    is how it gets checked. (b) Requires a curated depth chart, so this
    is a no-op for any season other than 2026 - deliberately, since
    without role gating it would inflate every bench player on a
    high-turnover team. (c) Like every other adjustment in this function,
    it is applied at PREDICT time only and so is not exercised by
    backtest.py; its evidence is the historical validation above, not the
    2025 holdout."""
    if depth_chart.empty:
        return df

    incumbent = ~changed
    if not incumbent.any():
        return df

    vacated = team_vacated_opportunity(conn, [target_season])
    vacated = vacated[vacated["season"] == target_season].set_index("team")
    if vacated.empty:
        return df

    # Volume arriving from elsewhere, on team_vacated_opportunity's own
    # raw-count basis. Shared with the team-changer scale via
    # _incoming_volume_share so the two vacancy adjustments can never
    # disagree about how much room the arrivals are taking; the incumbent
    # side subtracts the WHOLE incoming share (arrivals absorb the room),
    # the arrival side subtracts it net of the player's own contribution.
    incoming = _incoming_volume_share(df, changed)
    incoming_carry = incoming["carries"][0]
    incoming_target = incoming["targets"][0]

    role_lookup = depth_chart[["position", "gsis_id", "role"]].rename(columns={"gsis_id": "player_id"})
    role_lookup = role_lookup.dropna(subset=["player_id"]).drop_duplicates(subset=["player_id", "position"])
    roles = df[["player_id", "position"]].merge(role_lookup, on=["player_id", "position"], how="left")["role"]
    # QB rows are excluded, for the same reason predict_rookies needs
    # vacated_attempts_share rather than vacated_target_share: receiving-
    # corps and backfield turnover say nothing about a QB's own workload,
    # and scaling a quarterback's carry_share by how many RB carries left
    # the building is a category error, not a small one. It is also
    # simply unvalidated - the historical fit behind
    # INCUMBENT_VACANCY_ALPHA covers RB/WR/TE only.
    boostable_position = df["position"].isin(["RB", "WR", "TE"]).to_numpy()
    eligible = incumbent & roles.isin(BOOST_ELIGIBLE_ROLES).to_numpy() & boostable_position
    if not eligible.any():
        return df

    def _scale(vac_col, incoming_share, kind):
        gross = df["team_target"].map(vacated[vac_col])
        absorbed = df["team_target"].map(incoming_share).fillna(0.0)
        v_net = (gross - absorbed).fillna(0.0).clip(lower=0.0, upper=INCUMBENT_VACANCY_NET_CLIP)
        lever = v_net / (1.0 - v_net)
        s = 1.0 + INCUMBENT_VACANCY_ALPHA[kind] * lever
        return s.clip(upper=INCUMBENT_VACANCY_SCALE_CAP)

    carry_scale = _scale("vacated_carry_share", incoming_carry, "carry")
    # Position-group target vacancy: WR arrivals/incumbents use WR vacated
    # share so Diggs-style WR turnover does not inflate TE/RB (and vice versa).
    target_scale = pd.Series(1.0, index=df.index)
    for position in ("WR", "TE", "RB", "QB"):
        pos_mask = df["position"].eq(position)
        if not pos_mask.any():
            continue
        if position in ("WR", "TE", "RB") and "targets_by_position" in incoming:
            share_by_team, _own = incoming["targets_by_position"][position]
            vac_col = f"vacated_target_share_{position.lower()}"
            if vac_col not in vacated.columns:
                vac_col = "vacated_target_share"
        else:
            share_by_team = incoming_target
            vac_col = "vacated_target_share"
        # Rebuild scale for this position using the group vacancy.
        gross = df["team_target"].map(vacated[vac_col] if vac_col in vacated.columns else vacated["vacated_target_share"])
        absorbed = df["team_target"].map(share_by_team).fillna(0.0)
        v_net = (gross - absorbed).fillna(0.0).clip(lower=0.0, upper=INCUMBENT_VACANCY_NET_CLIP)
        lever = v_net / (1.0 - v_net)
        s = (1.0 + INCUMBENT_VACANCY_ALPHA["target"] * lever).clip(upper=INCUMBENT_VACANCY_SCALE_CAP)
        target_scale = target_scale.where(~pos_mask, s)

    for c in ["carry_share", "rz_carry_share"]:
        df.loc[eligible, c] = (df.loc[eligible, c] * carry_scale[eligible]).clip(upper=1.0)
    for c in ["target_share", "rz_target_share"]:
        df.loc[eligible, c] = (df.loc[eligible, c] * target_scale[eligible]).clip(upper=1.0)

    # snap_pct follows the position's primary opportunity type, matching
    # reassign_team_changers' own treatment.
    snap_scale = pd.Series(1.0, index=df.index)
    is_rb = df["position"] == "RB"
    is_recv = df["position"].isin(["WR", "TE"])
    snap_scale[is_rb] = carry_scale[is_rb]
    snap_scale[is_recv] = target_scale[is_recv]
    df.loc[eligible, "snap_pct"] = (df.loc[eligible, "snap_pct"] * snap_scale[eligible]).clip(upper=1.0)

    return df


def reassign_team_changers(conn, df, target_season, depth_chart):
    """Task 1 fix. For every player-row (source_season features), resolve
    the player's ACTUAL target_season team from seasonal_rosters and, for
    players who changed teams, re-point every team-dependent feature at the
    NEW team instead of silently keeping the old (source_season) team's
    numbers - the exact bug the project owner found by eyeballing Cousins
    (shown on ATL, his 2025 team, instead of his real 2026 team LV) and
    Murray (shown on ARI instead of MIN).

    Three things get re-pointed for a team-changer, all justified in
    PHASE6_REPORT.md:

    1. Output team label -> target_season roster team (source of truth).
    2. Team-context features (oc_tendency_profiles + OL quality) -> the
       NEW team's most recently OBSERVED season (source_season, i.e.
       2025 for a 2026 target) instead of the old team's. This is the same
       judgment call train.py/transitions.py already make for the
       no-team-change case ("last observed season stands in for the
       unplayed next one") - just correctly re-pointed at the team the
       player is actually walking into, not the one they left.
    3. Player SHARE features (carry/target/rz share, snap_pct) -> the old
       team's share number does NOT carry over to a new team with a
       different depth chart and different available volume (that would be
       just as wrong as the original bug, in a subtler way). Estimated
       instead via a team-changer adaptation of rookies.py's
       "vacated opportunity" concept: this player's own established share
       at their OLD team is used as a "quality tier" signal (how much
       volume this player is capable of commanding when given a role), then
       scaled by how much MORE or LESS opportunity is open at the new team
       compared to a league-average team this season:
           scale = clip(new_team_vacated_share / league_avg_vacated_share, 0.3, 2.5)
           new_share = old_team_share * scale
       `team_vacated_opportunity` (rookies.py) already computes, per team,
       what fraction of last season's carries/targets belonged to players
       who are no longer on that team for target_season (roster-fallback
       already built in for a season with zero played games) - exactly the
       "how much room is actually open here" signal a real preseason
       projection needs. The league average (not the player's OLD team's
       own vacated share) is used as the baseline, mirroring how
       rookies.predict_rookies scales against the historical BUCKET
       average rather than the specific player's own prior situation.

    LIMITATION, stated plainly: this does not, and cannot, capture scheme
    fit ("the new team's offense throws far more to the slot than the old
    team's did") - it only reflects how much raw opportunity is open, not
    how a specific scheme will actually distribute it. That residual is a
    real, unaddressed source of error for every team-changer in this
    output, on top of whatever normal projection error already exists.

    BUG FOUND AND FIXED (post-Phase-6, spot-checked while building fantasy
    points): the scale above was originally applied uncapped to every
    team-changer independently, with no check on whether another player
    (the team's actual new starter) was already the one absorbing that
    vacated opportunity. Concretely: Kenny Gainwell's own PIT carry_share
    (0.28, a real committee share) times TB's vacated-carry scale (1.48x,
    since Bucky Irving's team lost a lot of 2025 carries) produced an
    implied 0.41 carry_share for a player the curated depth chart correctly
    lists as TB's RB2 BEHIND Irving - bell-cow volume for a backup, because
    the team's whole vacancy was being credited to every team-changer at
    once instead of primarily to whoever the depth chart says is actually
    stepping into it. Fixed: the UPWARD half of the scale (>1.0, i.e. "this
    team has more room than average") is only applied to players the
    curated depth chart (`depth_chart` param, Task 2's
    src/depth_chart/starters_2026.csv) confirms as `role in {'starter',
    'committee'}` for their new team+position - see BOOST_ELIGIBLE_ROLES.
    Everyone else (confirmed 'backup', or not in the curated table for
    their new team at all) still gets the DOWNWARD half of the scale (a
    worse-than-average opportunity should still reduce their share) but is
    capped at 1.0 on the upside, so a windfall at the team level can no
    longer inflate a confirmed backup past their own established volume.
    This only applies for target_season=2026 (the only season with a
    curated table); for any other season, depth_chart is empty and every
    team-changer gets the ORIGINAL uncapped scale (unchanged pre-fix
    behavior) - there's no curated role signal to gate on outside 2026.
    snap_pct is scaled by the same carry/target scale as the position's
    primary opportunity type (RB->carry scale, WR/TE->target scale, QB->
    left unchanged, since starter-QB snap_pct is ~100% regardless of new
    team and the depth-chart gating in Task 3, not this share model,
    is what actually distinguishes a new team's QB1 from its QB3)."""
    roster_map = load_target_roster_map(conn, target_season)
    # reset_index: every merge below (pd.merge always returns a fresh
    # RangeIndex, regardless of the input frames' index) needs to line up
    # positionally with boolean masks computed on this same row order -
    # without this, `changed`'s original (filtered, non-contiguous) index
    # silently fails to align against the post-merge frame's index.
    df = df.copy().reset_index(drop=True)
    df["team_target"] = df["player_id"].map(roster_map["team"])
    df["roster_status"] = df["player_id"].map(roster_map["status"])

    # The curated depth chart OVERRIDES seasonal_rosters for team
    # assignment, and every disagreement is printed.
    #
    # Why curated wins: seasonal_rosters is a cached upstream snapshot
    # that lags real transactions by days-to-weeks in the preseason, and
    # nothing in this pipeline can tell "the roster is right and the CSV
    # is stale" from "the CSV is right and the roster hasn't caught up."
    # The curated table is the surface a human actually edits, so if it
    # were silently outranked by a stale snapshot, correcting a player's
    # team by hand would appear to do nothing - the worst possible
    # failure mode for a manually-maintained override. Deebo Samuel
    # (re-signed SF) and Stefon Diggs (signed WAS) were both being
    # projected on their 2025 teams for exactly this reason.
    #
    # The safety valve is the printed reconciliation below: a stale
    # CURATED row is now the thing that can go wrong, so every case where
    # the two sources disagree is surfaced by name for review rather than
    # resolved silently in either direction.
    if not depth_chart.empty:
        curated_team = (
            depth_chart.dropna(subset=["gsis_id"])
            .drop_duplicates(subset=["gsis_id"])
            .set_index("gsis_id")["team"]
        )
        curated = df["player_id"].map(curated_team)
        disagree = curated.notna() & df["team_target"].notna() & (curated != df["team_target"])
        filled = curated.notna() & df["team_target"].isna()
        if disagree.any() or filled.any():
            names = pd.read_sql("select gsis_id as player_id, display_name from players", conn)
            names = names.drop_duplicates("player_id").set_index("player_id")["display_name"]
            if disagree.any():
                print(f"Curated depth chart OVERRODE seasonal_rosters on team for "
                      f"{int(disagree.sum())} player(s) - verify the CSV is not the stale side:")
                for _, r in df[disagree].iterrows():
                    print(f"    {names.get(r['player_id'], r['player_id'])}: curated "
                          f"{curated[r.name]} vs roster {r['team_target']}")
            if filled.any():
                print(f"Curated depth chart SUPPLIED a team for {int(filled.sum())} player(s) absent "
                      f"from the {target_season} roster snapshot: "
                      f"{', '.join(sorted(names.get(p, p) for p in df.loc[filled, 'player_id']))}")
        df.loc[curated.notna(), "team_target"] = curated[curated.notna()]

    no_info = df["team_target"].isna()
    if no_info.any():
        # Not found in target_season's roster at all and not curated
        # (retired, out of league, or a crosswalk gap) - keep the old team
        # rather than inventing one. drop_players_absent_from_target_season
        # is what decides whether such a player stays in the output.
        df.loc[no_info, "team_target"] = df.loc[no_info, "team"]
    changed = (df["team_target"] != df["team"]) & ~no_info
    df["team_changed"] = changed

    if changed.any():
        source_season = target_season - 1

        oc = pd.read_sql(
            f"select season, team, {', '.join(OC_METRICS)} from oc_tendency_profiles where season={source_season}", conn,
        )
        olq = team_season_ol_quality(
            conn, [source_season], trailing_for_seasons={source_season},
        )
        # opp_def_pass_epa_prior/opp_def_rush_epa_prior for the NEW team,
        # same "most recently observed season stands in for the unplayed
        # target season" logic already used for OC/OL above - the new
        # team's source_season schedule-strength value is the best proxy
        # available for their target_season schedule.
        # Bug found while integrating (Sleeper-comparison investigation):
        # team_season_opponent_strength internally shifts its defense-EPA
        # lookup to season-1 to align each opponent's PRIOR-season defense
        # against the schedule season - calling it with ONLY [source_season]
        # means that prior season's defense-EPA rows were never fetched at
        # all, so the shifted join always missed and came back NaN for
        # every team. Passing [source_season - 1, source_season] gives it
        # both halves of its own join.
        opp_strength = team_season_opponent_strength(conn, [source_season - 1, source_season])
        opp_strength = opp_strength[opp_strength["season"] == source_season]
        team_ctx = oc.merge(olq, on=["season", "team"], how="left")
        team_ctx = team_ctx.merge(opp_strength, on=["season", "team"], how="left").drop(columns=["season"])
        team_ctx = team_ctx.rename(columns={"team": "team_target"})

        df = df.merge(team_ctx, on="team_target", how="left", suffixes=("", "_new"))
        for c in TEAM_CONTEXT_COLS:
            new_c = f"{c}_new"
            if new_c in df.columns:
                df.loc[changed, c] = df.loc[changed, new_c]
        df = df.drop(columns=[c for c in df.columns if c.endswith("_new")])

        vacated = team_vacated_opportunity(conn, [target_season])
        vacated = vacated[vacated["season"] == target_season]
        league_avg_carry_vac = vacated["vacated_carry_share"].mean()
        league_avg_target_vac = vacated["vacated_target_share"].mean()
        vac_keep = ["team", "vacated_carry_share", "vacated_target_share",
                    "vacated_target_share_wr", "vacated_target_share_te",
                    "vacated_target_share_rb"]
        vacated = vacated[[c for c in vac_keep if c in vacated.columns]].rename(
            columns={"team": "team_target"})
        df = df.merge(vacated, on="team_target", how="left")

        # Net the vacancy across COMPETING ARRIVALS, then damp toward
        # carrying the player's own share forward. Before this, every
        # arrival was handed the team's ENTIRE vacancy independently -
        # Washington vacated 52.3% of its targets and so awarded a 2.18x
        # share boost to Stefon Diggs, Chig Okonkwo AND Rachaad White at
        # once, inflating Diggs (0.185 -> 0.403) past the incumbent
        # McLaurin and inverting the team's pecking order. It is the same
        # bug this function's own docstring already describes for the
        # Gainwell case - "no check on whether another player was already
        # absorbing that vacated opportunity" - which the role gate below
        # only partly contained, since a curated starter passes it.
        #
        # `others_incoming` deliberately excludes the player's own volume:
        # an arrival should see the room net of its competitors, never net
        # of itself. See TEAM_CHANGE_VACANCY_ALPHA for the measured
        # damping and for how badly the un-netted version scored.
        # Hierarchical L2: target vacancy is position-group-specific so a
        # WR arrival does not claim TE/RB vacated share.
        incoming = _incoming_volume_share(df, changed)
        share_by_team_c, own_c = incoming["carries"]
        others_c = (df["team_target"].map(share_by_team_c).fillna(0.0) - own_c).clip(lower=0.0)
        v_net_c = (df["vacated_carry_share"] - others_c).clip(lower=0.0)
        raw_c = (v_net_c / league_avg_carry_vac).clip(*TEAM_CHANGE_SHARE_CLIP).fillna(1.0)
        carry_scale = 1.0 + TEAM_CHANGE_VACANCY_ALPHA["carry"] * (raw_c - 1.0)

        target_scale = pd.Series(1.0, index=df.index)
        for position in ("WR", "TE", "RB"):
            pos_mask = changed & df["position"].eq(position)
            if not pos_mask.any():
                continue
            vac_col = f"vacated_target_share_{position.lower()}"
            if vac_col not in df.columns:
                vac_col = "vacated_target_share"
            share_by_team, own = incoming["targets_by_position"][position]
            others = (df["team_target"].map(share_by_team).fillna(0.0) - own).clip(lower=0.0)
            v_net = (df[vac_col] - others).clip(lower=0.0)
            raw = (v_net / league_avg_target_vac).clip(*TEAM_CHANGE_SHARE_CLIP).fillna(1.0)
            s = 1.0 + TEAM_CHANGE_VACANCY_ALPHA["target"] * (raw - 1.0)
            target_scale = target_scale.where(~pos_mask, s)
        # Non WR/TE/RB changers (e.g. QB) keep team-wide target vac if any.
        other_chg = changed & ~df["position"].isin(["WR", "TE", "RB"])
        if other_chg.any():
            share_by_team, own = incoming["targets"]
            others = (df["team_target"].map(share_by_team).fillna(0.0) - own).clip(lower=0.0)
            v_net = (df["vacated_target_share"] - others).clip(lower=0.0)
            raw = (v_net / league_avg_target_vac).clip(*TEAM_CHANGE_SHARE_CLIP).fillna(1.0)
            s = 1.0 + TEAM_CHANGE_VACANCY_ALPHA["target"] * (raw - 1.0)
            target_scale = target_scale.where(~other_chg, s)

        if not depth_chart.empty:
            role_lookup = depth_chart[["position", "gsis_id", "role"]].rename(columns={"gsis_id": "player_id"})
            role_lookup = role_lookup.dropna(subset=["player_id"]).drop_duplicates(subset=["player_id", "position"])
            df = df.merge(role_lookup, on=["player_id", "position"], how="left")
        else:
            df["role"] = None
        boost_eligible = df["role"].isin(BOOST_ELIGIBLE_ROLES)
        carry_scale = carry_scale.where(boost_eligible, carry_scale.clip(upper=1.0))
        target_scale = target_scale.where(boost_eligible, target_scale.clip(upper=1.0))

        # Bug found via Sleeper comparison (Waddle MIA->DEN, DJ Moore
        # CHI->BUF, both curated 'starter' at their new team but crushed to
        # ~0.06 target_share by the DOWNWARD half of the vacancy scale):
        # team_vacated_opportunity measures how much volume LEFT a team
        # (players no longer on the roster) - a trade acquisition into an
        # already-full room legitimately shows near-zero "vacated" share by
        # that definition even though the team specifically traded for and
        # is starting this player. The curated table confirming role=
        # 'starter' is a stronger, player-specific signal than the
        # team-level vacancy heuristic, so a confirmed starter's own
        # established share (their "quality tier" per this function's
        # class docstring) is never scaled BELOW what they already had -
        # only the upside (extra room beyond a normal team) still applies.
        # 'committee'/'backup' deliberately keep the original full
        # 0.3-2.5 range: a genuine committee/backup landing spot CAN mean
        # real volume loss; the availability and calibrated depth signals
        # handle that downstream rather than an asserted role multiplier.
        confirmed_starter = changed & (df["role"] == "starter")
        carry_scale = carry_scale.where(~confirmed_starter, carry_scale.clip(lower=1.0))
        target_scale = target_scale.where(~confirmed_starter, target_scale.clip(lower=1.0))

        df = df.drop(columns=["role"])

        for c in ["carry_share", "rz_carry_share"]:
            df.loc[changed, c] = (df.loc[changed, c] * carry_scale[changed]).clip(upper=1.0)
        for c in ["target_share", "rz_target_share"]:
            df.loc[changed, c] = (df.loc[changed, c] * target_scale[changed]).clip(upper=1.0)

        snap_scale = pd.Series(1.0, index=df.index)
        snap_scale[df["position"] == "RB"] = carry_scale[df["position"] == "RB"]
        snap_scale[df["position"].isin(["WR", "TE"])] = target_scale[df["position"].isin(["WR", "TE"])]
        df.loc[changed, "snap_pct"] = (df.loc[changed, "snap_pct"] * snap_scale[changed]).clip(upper=1.0)

        df = df.drop(columns=[c for c in [
            "vacated_carry_share", "vacated_target_share",
            "vacated_target_share_wr", "vacated_target_share_te",
            "vacated_target_share_rb",
        ] if c in df.columns])

    # Incumbents (everyone the block above did NOT touch) get the other
    # half of the same idea: credit for the opportunity their departing
    # teammates left behind. Runs here, while df["team"] is still the
    # source-season team and `changed` is available to net out arrivals.
    df = _attach_rookie_residual_vacancy(conn, df, target_season, changed)
    df = apply_incumbent_vacancy_boost(conn, df, target_season, depth_chart, changed)

    df["team"] = df["team_target"]
    df = df.drop(columns=["team_target"])
    return df

