"""Veteran projection path and curation tripwire warnings.

Does not import predict (imports stage modules instead).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.projection.artifacts import load_availability_models
from src.projection.depth_gating import (
    apply_curated_availability_override,
    apply_full_season_games_baseline,
    apply_curated_depth_tier,
    apply_depth_chart_gating,
    apply_status_overrides,
    enforce_availability_chart_review,
    load_depth_chart,
    load_status_overrides,
)
from src.projection.depth_history import (
    attach_availability_depth_rank,
    attach_depth_rank,
    attach_depth_tier,
)
from src.projection.features import TARGET_STATS
from src.projection.roster_moves import (
    drop_players_absent_from_target_season,
    reassign_team_changers,
)
from src.projection.team_reconcile import _attach_team_total_pred
from src.projection.transitions import (
    ALL_FEATURES,
    ROLE_PRIOR_FEATURE,
    role_label_for,
    REFRAMED_SHARE_STATS,
    SEASON_GAMES,
    age_shrunk_predict,
)


def _attach_veteran_intervals(combined, resid):
    """Empirical 80% endpoints for NON-reframed veteran rows, post-discount.

    Runs after ``apply_depth_chart_gating`` so the residual is added to the
    prediction that ships. ``models/interval_residuals.csv`` is fit by
    ``backtest.py`` as ``actual - depth-discounted pred``, so the band is
    already the band for a discounted player and must not be rescaled by the
    factor a second time.

    Reframed receiving rows are untouched here: their endpoints are built in
    ``_compose_reframed_receiving_predictions``, after the share has been
    composed into a rate, and on the same absolute-residual convention.

    A (position, stat) with no residual row keeps NaN endpoints and
    ``interval_low_n_flag = True`` — the pre-existing contract for "we have
    no calibration for this cell", never a silently borrowed band.
    """
    if combined.empty:
        return combined
    out = combined.copy()
    reframed = pd.MultiIndex.from_tuples(REFRAMED_SHARE_STATS, names=["position", "stat"])
    is_reframed = out.set_index(["position", "stat"]).index.isin(reframed)
    target = ~is_reframed
    if not target.any():
        return out
    r = resid[["position", "stat", "resid_low", "resid_high", "low_n_flag"]]
    keys = out.loc[target, ["position", "stat"]].merge(r, on=["position", "stat"], how="left")
    keys.index = out.index[target]
    has = keys["resid_low"].notna()
    idx = keys.index[has]
    out.loc[idx, "pred_pg_low"] = (
        out.loc[idx, "pred_pg"] + keys.loc[has, "resid_low"]).clip(lower=0)
    out.loc[idx, "pred_pg_high"] = out.loc[idx, "pred_pg"] + keys.loc[has, "resid_high"]
    out.loc[keys.index, "interval_low_n_flag"] = (
        keys["low_n_flag"].fillna(True).astype(bool))
    return out


def _prior_in_label_units(pos_df, position, stat):
    """Season N's own value of the label season N+1 is predicted on.

    ALL_FEATURES already carries `prior_{stat}_pg`, but that is a rate per
    APPEARANCE week - a different unit from the role-rate label. Handing the
    model the prior in the label's own units is what the prototype validated,
    and it is the same column build_role_transition_pairs supplies at fit
    time, so training and serving agree by construction.
    """
    label = role_label_for(position, stat)
    if label not in pos_df.columns:
        raise ValueError(
            f"{position} {stat}: source-season frame is missing {label!r}. "
            f"build_player_season_features supplies it; a frame built by an "
            f"older version of that function cannot score a role-rate model.")
    return pd.to_numeric(pos_df[label], errors="coerce").to_numpy()


def project_veterans(conn, feat, source_season, models, resid, target_season, as_of=None):
    """source_season's feature rows -> next-season per-game rate
    predictions, for every player with real source_season production.
    target_season's OWN incoming rookie class (drafted in target_season, or
    UDFA per identify_target_season_rookie_class) is projected separately
    below via the rule-based path instead - but that class is naturally
    ABSENT from source_season's features already (they have no NFL history
    before target_season), so no extra exclusion is needed here to avoid
    double-counting them.

    target_season is now required (previously predict.py effectively
    assumed the player's source_season team carried forward unchanged -
    see reassign_team_changers' docstring for the bug this fixes) and is
    also used to gate output against the curated depth chart (Task 3).

    Bug found and fixed here (Phase 6, while spot-checking a random 2025
    rookie per Task 4): this function used to additionally exclude every
    player whose OWN rookie season equaled source_season (via
    identify_rookie_seasons/identify_udfa_rookie_seasons([source_season])),
    on the stated reasoning "source_season IS their only season, so they
    have no real trailing features." That reasoning doesn't hold - a
    player's rookie season IS a complete, real season of production, and is
    exactly the trailing data the veteran model needs for their SOPHOMORE
    projection. The effect was silently dropping every player's entire
    second season from the output (verified: Ashton Jeanty, a real 2025
    rookie with a full 17-game/267-carry season, was completely absent from
    2026's output before this fix, despite having a valid 2025 feature row -
    the same class of silent-drop bug PHASE5_REPORT.md already found and
    fixed once for long-tenured veterans; this is its sibling for players
    exactly one year removed from being a rookie, not caught by that
    review's Allen/Chase/McCaffrey spot-checks since none of them were in
    their second season at the time). Removing the exclusion here is safe
    (see previous paragraph: target_season's real rookies can't appear in
    source_season's features regardless), and was re-verified end-to-end
    after the fix (Jeanty now appears; see PHASE6_REPORT.md's spot-checks)."""
    depth_chart = load_depth_chart(target_season)

    base = feat[(feat["season"] == source_season) & (feat["games_played"] > 0)]
    base = reassign_team_changers(conn, base, target_season, depth_chart)
    base = drop_players_absent_from_target_season(conn, base, depth_chart, target_season)
    # Team-level, resolved above while team_target and `changed` were both
    # in hand. Handed to the rookie path so it nets against the same
    # arithmetic the veteran paths used instead of re-deriving it.
    residual_cols = ["rookie_residual_carry_fraction", "rookie_residual_target_fraction"]
    rookie_residual = (
        base.drop_duplicates("team").set_index("team")[residual_cols]
        if set(residual_cols) <= set(base.columns) else pd.DataFrame(columns=residual_cols)
    )
    base = base.drop(columns=residual_cols, errors="ignore")

    # Availability (Phase 11): a per-game rate alone can't express season
    # value - two players at the same rate are worth very different
    # amounts if one plays 8 games and the other 16, which is exactly the
    # Mike Evans / Deebo Samuel case. Measured directly on the 2024->2025
    # holdout (backtest_season_totals, re-measured after Gate A): rate x a
    # fixed 17 games is WORSE than naive carry-forward at predicting season
    # totals (WR 252.3 vs 170.5 yards MAE, TE 152.8 vs 110.9, RB 295.1 vs
    # 191.3, QB 1420.4 vs 780.0); rate x predicted games beats both (WR
    # 151.8, TE 90.7, RB 172.7, QB 615.3). Attached per player here,
    # carried through to the output so the split stays visible rather than
    # being folded into the rate.
    #
    # The model reads target_season's preseason depth chart (Gate A) - see
    # depth_history.py and train.fit_availability. Attached here, on the
    # post-reassign_team_changers frame, so a player who changed teams is
    # looked up on his NEW team's chart, which is the whole point: Deebo at
    # SF WR-something and Deebo off WAS's chart are different availability
    # predictions.
    #
    # Note this is the nflverse chart, NOT the curated
    # src/depth_chart/starters_2026.csv one used for gating below. The two
    # are deliberately separate: the curated file is hand-verified and
    # authoritative for membership/team/role, but exists only for 2026, so
    # it can never be a trained-on feature. The nflverse chart exists for
    # every season, which is the only reason the availability model can be
    # honestly held out on it. At predict time only, curated membership
    # overrides target_depth_rank (see apply_curated_availability_override).
    avail_models = load_availability_models()
    base = base.copy()
    base = attach_availability_depth_rank(base, target_season, conn=conn, as_of=as_of)
    # The untruncated rank, for the Gate B volume ladder. Attached here so
    # it rides along into `combined` and is available to gating and to the
    # share renormalization without a second lookup.
    base = attach_depth_rank(base, target_season, conn=conn, as_of=as_of)
    base = attach_depth_tier(base, target_season, conn=conn, as_of=as_of)
    base = apply_curated_depth_tier(base, depth_chart)
    base = apply_curated_availability_override(base, depth_chart)
    status_overrides = load_status_overrides(target_season, as_of=as_of)
    enforce_availability_chart_review(
        base, depth_chart, status_overrides, target_season, conn=conn)
    base["projected_games"] = np.nan
    for position, am in avail_models.items():
        mask = base["position"] == position
        if mask.any():
            # am["features"], not ALL_FEATURES: the availability models
            # carry a wider schema than the rate models (AVAILABILITY_
            # FEATURES), and an older models/ directory predating Gate A
            # still carries the narrower one and must keep working.
            # Gate A still runs for audit (projected_games_raw); draft
            # exposure is a full season except IR/PUP/Sus overrides.
            base.loc[mask, "projected_games"] = np.clip(
                am["model"].predict(base.loc[mask, am["features"]]), 0, SEASON_GAMES)
    base = apply_full_season_games_baseline(base, season_games=SEASON_GAMES)
    base = apply_status_overrides(base, status_overrides)

    rows = []
    for position, stats in TARGET_STATS.items():
        pos_df = base[base["position"] == position]
        if pos_df.empty:
            continue
        for stat in stats:
            m = models[(position, stat)]
            # m["features"], not ALL_FEATURES: the role-rate models carry
            # ROLE_FEATURES (depth tier + the prior in the label's own units),
            # and reading the schema off the joblib is what stops a model fit
            # on one basis from being scored as if it were the other.
            features = m.get("features", ALL_FEATURES)
            X = pos_df.copy()
            X[ROLE_PRIOR_FEATURE] = _prior_in_label_units(pos_df, position, stat)
            preds = age_shrunk_predict(m["model"], X, position, features=features)
            out = pos_df[["player_id", "team", "position", "team_changed",
                          "roster_status", "projected_games", "projected_games_raw",
                          "nfl_depth_rank", "depth_tier", "depth_tier_source"]].copy()
            if "status_override_applied" in pos_df.columns:
                out["status_override_applied"] = pos_df["status_override_applied"].to_numpy()
            out["stat"] = stat
            out["source"] = "veteran_model"
            out["low_confidence"] = False

            if (position, stat) in REFRAMED_SHARE_STATS:
                # preds here is a SHARE (see train.py's fit_one), not a
                # rate - composed into a real pred_pg by
                # _compose_reframed_receiving_predictions AFTER this whole
                # loop finishes, since normalizing a team's shares needs
                # ALL THREE reframed positions' predictions at once, and
                # this loop produces them at different iterations. pred_pg
                # holds the raw share as a placeholder; low/high are
                # deferred entirely (computed in rate units, post-compose).
                out["pred_pg"] = np.clip(preds, 0, None)
                out["pred_pg_low"], out["pred_pg_high"], out["interval_low_n_flag"] = np.nan, np.nan, False
            else:
                out["pred_pg"] = np.clip(preds, 0, None)  # a per-game rate can't be negative; LightGBM isn't constrained
                # Endpoints are deferred to _attach_veteran_intervals below,
                # AFTER the Gate B ladder. backtest.py now fits
                # interval_residuals.csv as (actual - discounted pred), so
                # the residual belongs on a discounted prediction. Computing
                # it here and letting the ladder scale the whole interval
                # would apply the discount to the band a second time - a QB2
                # would carry an 0.77x-narrow version of a band already
                # calibrated for him. Reframed rows have always deferred
                # theirs for the same reason (see
                # _compose_reframed_receiving_predictions' interval note).
                out["pred_pg_low"], out["pred_pg_high"], out["interval_low_n_flag"] = np.nan, np.nan, False
            rows.append(out)
    combined = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if combined.empty:
        return combined, rookie_residual

    # Phase 2 of the consensus-gap work: the reframed receiving rows leave
    # this function with pred_pg still in (now role-discounted) SHARE
    # units plus a team_total_pred helper column - project_season composes
    # them into real rates only after the rookie path has produced the
    # incoming rookies' implied shares for the same denominator. Gating
    # multiplying a share by its role discount here is exactly what lets
    # the later renormalization see discounted shares.
    combined = _attach_team_total_pred(combined, feat, source_season, models)

    depth_chart = load_depth_chart(target_season)
    combined = apply_depth_chart_gating(combined, depth_chart)
    combined = _attach_veteran_intervals(combined, resid)
    # Status overrides re-applied after rate gating so mode=zero wins.
    combined = apply_status_overrides(
        combined, load_status_overrides(target_season, as_of=as_of))
    _warn_discounted_high_usage(conn, combined, base, source_season)
    return combined, rookie_residual

# Curation-tripwire thresholds: season-N usage above ANY of these makes a
# discounted projection suspicious enough to warrant a human second look at
# the curated depth chart. Calibrated on the two known misses these would
# have caught (Parker Washington: 97 targets/56.5 rec ypg; Wan'Dale
# Robinson: 141 targets/63.4 rec ypg, both absent from starters_2026.csv
# and silently treated as deep bench) while sitting above genuine
# deep-bench usage levels.
TRIPWIRE_SEASON_TARGETS = 70
TRIPWIRE_TARGETS_PG = 5.0
TRIPWIRE_REC_YPG = 40.0

# Passing and rushing equivalents. Their absence was a real hole, not a
# judgment that QB/RB curation matters less: the criteria above key on
# targets and receiving yards only, so a discounted QB or RB could not fire
# this tripwire at ALL regardless of volume. 2026 had 26 discounted rows
# with >=150 attempts or >=100 carries in 2025 - Flacco (436 att), Tua
# (417), Dowdle (237 car), Henderson (180) - none of which could be
# surfaced.
#
# Passing/rushing criteria are additionally gated on a severe net factor so
# ordinary calibrated depth shading does not bury real curation misses.
TRIPWIRE_SEASON_ATTEMPTS = 200
TRIPWIRE_ATTEMPTS_PG = 25.0
TRIPWIRE_SEASON_CARRIES = 100
TRIPWIRE_CARRIES_PG = 8.0
TRIPWIRE_SEVERE_FACTOR = 0.2

# The >= 4 game guard the receiving criteria use is too loose for a rushing
# RATE, and the reason is specific to the stat rather than general caution:
# carries per game over a 4-5 game sample is dominated by whether the
# player happened to start those weeks, so it reads as a starter's rate by
# construction and says nothing about his next-season role (Audric Estime,
# 46 carries in 5 games, rates 9.2/g). Half a season is the point where the
# rate reflects a role. Passing keeps the 4-game guard: a QB who threw 189
# times in 5 games was unambiguously starting them, and that IS the signal.
TRIPWIRE_RUSH_MIN_GAMES = 8


def _warn_availability_chart_disagreement(base, depth_chart):
    """Warn (stderr only, never changes a number - same contract as
    _warn_discounted_high_usage below) when the hand-curated depth chart
    lists a player that the nflverse preseason chart does not carry at
    availability depth.

    The two charts are independent and normally agree closely (2026:
    182/183 curated starters, 27/27 committee, 50/54 backups). Where they
    disagree, the nflverse absence is what the availability model sees, and
    it is a genuine signal - a player on nobody's published chart in early
    August really does play fewer games. But a curator who deliberately
    placed a player on a team (Diggs -> WAS, Deebo -> SF; see
    reassign_team_changers) deserves to be told that the games model
    disagrees, rather than discovering a halved projected_games by
    accident. Resolving it is a human call about which chart is right.

    Covers EVERY curated role, not just the ones where a games error costs
    the most fantasy points. A curated backup absent from the nflverse
    chart is the cheaper case in isolation, but it is the same underlying
    event - the two sources disagree about whether this player has a job -
    and suppressing the cheap half would make the warning's silence mean
    "no disagreement above a threshold I chose" instead of "no
    disagreement." Listed worst-role-first so triage is still ordered."""
    if depth_chart.empty:
        return
    curated = depth_chart[depth_chart["role"].notna()]
    off = base[base["player_id"].isin(curated["gsis_id"])
               & base["target_depth_rank"].isna()]
    if off.empty:
        return
    # `base` is the feature frame and carries no display_name; the curated
    # chart does, and is the table a reader would go fix.
    by_id = curated.drop_duplicates("gsis_id").set_index("gsis_id")
    name_of, role_of = by_id["player_name"], by_id["role"]
    order = {"starter": 0, "committee": 1, "backup": 2}
    listed = sorted(
        ((order.get(role_of.get(r.player_id), 9), name_of.get(r.player_id, r.player_id),
          r.team, r.position, role_of.get(r.player_id), r.projected_games)
         for r in off.itertuples()),
        key=lambda x: (x[0], -x[5]))
    names = ", ".join(f"{n} ({t} {p}, curated {role}, {g:.1f} g)"
                      for _, n, t, p, role, g in listed)
    counts = ", ".join(f"{c} {r}" for r, c in
                       sorted(((r, sum(1 for x in listed if x[4] == r))
                               for r in set(x[4] for x in listed)),
                              key=lambda kv: order.get(kv[0], 9)))
    print(f"AVAILABILITY CHART DISAGREEMENT: {len(off)} curated player(s) ({counts}) "
          f"are absent from the nflverse preseason chart, so the games model treated "
          f"them as off-chart: {names}", file=sys.stderr)


def _warn_discounted_high_usage(conn, combined, base, source_season):
    """Warn (stderr only - curation stays a human decision, this NEVER
    changes a number) when a player whose projection was discounted by
    depth-chart gating had clearly fantasy-relevant usage in the source
    season. A high-usage player marked outside relevant depth is more likely
    a curation gap than a real role collapse."""
    discounted = combined[
        (combined["depth_chart_status"] == "deep_bench_discounted")
        | combined["role_discount_applied"]
    ]
    if discounted.empty:
        return
    usage = base[["player_id", "targets", "targets_pg", "receiving_yards_pg",
                  "attempts", "attempts_pg", "carries", "carries_pg", "games_played"]]
    flagged = (
        discounted[["player_id", "team", "position", "role", "role_discount_factor"]]
        .drop_duplicates("player_id")
        .merge(usage, on="player_id")
    )
    # Per-game thresholds need a minimum sample (>= 4 games) so a 1-3 game
    # blip (e.g. 5 targets in a single spot appearance) doesn't fire; the
    # season-total threshold needs no such guard by construction. 4 keeps
    # genuinely relevant injury-shortened seasons (Tyreek Hill's 4-game
    # 66-ypg 2025) inside the net.
    enough_games = flagged["games_played"] >= 4
    receiving = (
        (flagged["targets"] >= TRIPWIRE_SEASON_TARGETS)
        | (enough_games & (flagged["targets_pg"] >= TRIPWIRE_TARGETS_PG))
        | (enough_games & (flagged["receiving_yards_pg"] >= TRIPWIRE_REC_YPG))
    )
    # Keyed on the net multiplier actually applied, not on role names, so a
    # future role tier is covered by its severity rather than by having
    # been added to a list here.
    severe = flagged["role_discount_factor"] <= TRIPWIRE_SEVERE_FACTOR
    enough_rush_games = flagged["games_played"] >= TRIPWIRE_RUSH_MIN_GAMES
    passing = severe & (
        (flagged["attempts"] >= TRIPWIRE_SEASON_ATTEMPTS)
        | (enough_games & (flagged["attempts_pg"] >= TRIPWIRE_ATTEMPTS_PG))
    )
    rushing = severe & (
        (flagged["carries"] >= TRIPWIRE_SEASON_CARRIES)
        | (enough_rush_games & (flagged["carries_pg"] >= TRIPWIRE_CARRIES_PG))
    )
    flagged = flagged[receiving | passing | rushing].copy()
    if flagged.empty:
        return
    # Which criterion fired, so a reader can tell a 400-attempt QB2 from a
    # 70-target WR without reading the thresholds out of this file.
    flagged["_why"] = np.select(
        [passing.reindex(flagged.index, fill_value=False),
         rushing.reindex(flagged.index, fill_value=False)],
        ["passing", "rushing"], default="receiving")
    flagged["_sort"] = np.where(
        flagged["_why"] == "passing", flagged["attempts"],
        np.where(flagged["_why"] == "rushing", flagged["carries"], flagged["targets"]))
    names = pd.read_sql("SELECT gsis_id AS player_id, display_name FROM players", conn)
    flagged = flagged.merge(names, on="player_id", how="left")
    print(
        f"CURATION TRIPWIRE: {len(flagged)} discounted player(s) had fantasy-relevant "
        f"season-{source_season} usage - verify their rows in the curated depth chart:",
        file=sys.stderr,
    )
    for _, r in flagged.sort_values(["_why", "_sort"], ascending=[True, False]).iterrows():
        if r["_why"] == "passing":
            detail = f"{r['attempts']:.0f} attempts, {r['attempts_pg']:.1f} att/g"
        elif r["_why"] == "rushing":
            detail = f"{r['carries']:.0f} carries, {r['carries_pg']:.1f} car/g"
        else:
            detail = (f"{r['targets']:.0f} targets, {r['targets_pg']:.1f} tgt/g, "
                      f"{r['receiving_yards_pg']:.1f} rec ypg")
        print(
            f"  {r['display_name']} ({r['team']} {r['position']}, role={r['role']}, "
            f"{r['role_discount_factor']:.2f}x): {detail} in {r['games_played']:.0f} games",
            file=sys.stderr,
        )
