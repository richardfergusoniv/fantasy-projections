"""Team anchor scoring, receiving composition, and volume reconciliation.

Does not import predict.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.projection.contracts import (
    NAMED_REC_RECEPTIONS_COVERAGE,
    NAMED_REC_TDS_COVERAGE,
    NAMED_REC_YARDS_COVERAGE,
    NAMED_RUSH_COVERAGE,
    PASS_CATCH_COHERENCE_BAND,
    QB_ATTEMPTS_PER_VOLUME_GAME_MAX,
    RUSH_ATTEMPTS_PER_APPEARANCE_MAX,
    RUSH_YARDS_PER_CARRY_MAX,
    TEAM_ANCHOR_OUTPUT_COLS,
    USAGE_SHARE_BLEND_W,
    USAGE_SHARE_CURATED_W,
    USAGE_SHARE_FAMILIES,
    USAGE_SHARE_MAX_RANK,
)
from src.projection.corrections import elite_shrinkage_adjustment
from src.projection.data_prep import load_weekly_usage
from src.projection.depth_history import attach_depth_rank
from src.projection.transitions import (
    REFRAMED_SHARE_STATS,
    SEASON_GAMES,
    TEAM_ATTEMPTS_LABEL,
    TEAM_CARRIES_LABEL,
    TEAM_FEATURES,
    TEAM_MODEL_FEATURES,
    TEAM_RUSH_YARDS_LABEL,
    TEAM_TOTAL_LABEL,
    receiving_share_scale,
)


TEAM_ANCHOR_SPECS = (
    (TEAM_TOTAL_LABEL, ("TEAM", "passing_yards"), "team_passing_yards_pg_pred"),
    (TEAM_ATTEMPTS_LABEL, ("TEAM", "pass_attempts"), "team_pass_attempts_pg_pred"),
    (TEAM_CARRIES_LABEL, ("TEAM", "carries"), "team_carries_pg_pred"),
    (TEAM_RUSH_YARDS_LABEL, ("TEAM", "rushing_yards"), "team_rushing_yards_pg_pred"),
)


def canonical_team_anchor_frame(source_feat, source_season, target_teams, models):
    """Score team models from one canonical source-season row per target team.

    Team anchors must never be derived from a reassigned player row.  A player
    arriving at MIA still carries his old team's observed team-total labels;
    selecting that row after changing only its ``team`` value silently makes
    the anchor depend on player row order.  This function starts from the
    unreassigned feature table, validates team-grain invariants, and returns a
    deterministic one-row-per-team scoring frame.
    """
    teams = pd.Index(pd.Series(target_teams).dropna().unique(), name="team")
    if len(teams) != 32:
        raise ValueError(
            f"Expected exactly 32 target teams for physical team anchors; found {len(teams)}: "
            f"{sorted(map(str, teams))}"
        )
    labels = [spec[0] for spec in TEAM_ANCHOR_SPECS]
    required = ["season", "team"] + TEAM_FEATURES + labels
    missing_cols = [c for c in required if c not in source_feat.columns]
    if missing_cols:
        raise KeyError(f"Canonical team anchor frame is missing required columns: {missing_cols}")

    raw = source_feat[(source_feat["season"] == source_season) & source_feat["team"].isin(teams)]
    if raw.empty:
        raise ValueError(f"No canonical team features found for source season {source_season}")
    invariant_cols = TEAM_FEATURES + labels
    nunique = raw.groupby("team")[invariant_cols].nunique(dropna=False)
    bad = nunique.columns[(nunique > 1).any()].tolist()
    if bad:
        offenders = nunique.index[(nunique[bad] > 1).any(axis=1)].tolist()
        raise ValueError(
            f"Team-grain source features are not invariant for {offenders}; conflicting columns: {bad}"
        )

    team_feat = raw.drop_duplicates("team").set_index("team").reindex(teams).reset_index()
    missing_teams = team_feat.loc[team_feat[invariant_cols].isna().all(axis=1), "team"].tolist()
    if missing_teams:
        raise ValueError(f"Missing canonical source-season anchors for teams: {missing_teams}")
    if team_feat[TEAM_FEATURES].isna().any(axis=None):
        cols = team_feat[TEAM_FEATURES].columns[team_feat[TEAM_FEATURES].isna().any()].tolist()
        affected = team_feat.loc[team_feat[cols].isna().any(axis=1), "team"].tolist()
        raise ValueError(f"NaN team-model inputs for teams {affected}; columns: {cols}")

    for label, model_key, pred_col in TEAM_ANCHOR_SPECS:
        artifact = models[model_key]
        if artifact.get("label") != label:
            raise ValueError(
                f"Team model {model_key} label mismatch: expected {label}, got {artifact.get('label')}"
            )
        x = team_feat[TEAM_FEATURES].copy()
        x["team_naive_pred"] = team_feat[label]
        if list(artifact["features"]) != list(TEAM_MODEL_FEATURES):
            raise ValueError(
                f"Team model {model_key} feature schema mismatch: {artifact['features']}"
            )
        team_feat[pred_col] = np.clip(
            artifact["model"].predict(x[artifact["features"]]), 0, None)

    team_feat["team_total_pred"] = team_feat["team_passing_yards_pg_pred"]
    team_feat["team_anchor_source_season"] = int(source_season)
    team_feat["team_anchor_lag_team"] = team_feat["team"]
    team_feat["team_anchor_provenance"] = "canonical_source_team_frame"
    pred_cols = [spec[2] for spec in TEAM_ANCHOR_SPECS]
    if team_feat[pred_cols].isna().any(axis=None):
        raise ValueError("At least one canonical team model emitted a missing anchor")
    return team_feat


def _attach_team_total_pred(combined, source_feat, source_season, models):
    """Attach deterministic, canonical team anchors before composition."""
    team_feat = canonical_team_anchor_frame(
        source_feat, source_season, combined["team"], models)
    anchor_cols = [
        "team", "team_total_pred", "team_passing_yards_pg_pred",
        "team_pass_attempts_pg_pred", "team_carries_pg_pred",
        "team_rushing_yards_pg_pred", "team_anchor_source_season",
        "team_anchor_lag_team", "team_anchor_provenance",
    ]
    combined = combined.merge(team_feat[anchor_cols], on="team", how="left", validate="many_to_one")
    if combined["team_anchor_provenance"].isna().any():
        missing = sorted(combined.loc[combined["team_anchor_provenance"].isna(), "team"].dropna().unique())
        raise ValueError(f"Projection rows missing canonical team anchors: {missing}")
    # The player's OWN observed season-N receiving rate, carried alongside
    # so the Phase-7 elite-shrinkage correction can key on it (see
    # corrections.py for why the observed rate and not the predicted one).
    observed = source_feat[source_feat["season"] == source_season][
        ["player_id", "receiving_yards_pg"]].rename(
        columns={"receiving_yards_pg": "_observed_recv_pg"})
    combined = combined.merge(observed.drop_duplicates("player_id"), on="player_id", how="left")
    return combined


# Intermediate columns dropped once composition is done. `role_discount_
# factor` is deliberately NOT among them any more: it is the actual
# multiplier applied to the row, and this project's rule is that a discount
# must be visible in the output table rather than buried in a module. The
# The explicit factor makes the magnitude auditable without reconstructing
# it from role and chart status.
_HELPER_COLS = ["team_total_pred", "_observed_recv_pg"]


# The receiving stats that ride along with a receiving_yards correction.
# receiving_tds is included on the same measured evidence as the other two
# (ratio 1.07/1.08) even though it is the noisiest of the three.
ELITE_COMPANION_STATS = ("receptions", "targets", "receiving_tds")


def _propagate_elite_correction(other, reframed, before, adj):
    """Scale a corrected player's companion receiving stats by the same
    proportion his receiving_yards moved - see the call site for why
    proportional is the measured answer.

    Guarded on `before > 0`: a player whose composed yards prediction is
    zero has no defined proportion, and scaling from zero would either
    divide by zero or invent volume from nothing. Those rows keep their
    uncorrected companion stats, which is the honest fallback."""
    moved = adj > 0
    if not moved.any():
        return other
    ratio = pd.Series(
        np.where(before > 0, (before + adj) / np.where(before > 0, before, 1.0), 1.0),
        index=reframed.index,
    )
    key = pd.MultiIndex.from_arrays(
        [reframed.loc[moved, "player_id"], reframed.loc[moved, "position"]])
    per_player = pd.Series(ratio[moved].to_numpy(), index=key)
    per_player = per_player[~per_player.index.duplicated()]

    hit = other["stat"].isin(ELITE_COMPANION_STATS)
    if not hit.any():
        return other
    other = other.copy()
    idx = pd.MultiIndex.from_arrays([other.loc[hit, "player_id"], other.loc[hit, "position"]])
    scale = pd.Series(per_player.reindex(idx).to_numpy(), index=other.index[hit]).fillna(1.0)
    for col in ["pred_pg", "pred_pg_low", "pred_pg_high"]:
        other.loc[hit, col] = other.loc[hit, col] * scale
    return other


def _compose_reframed_receiving_predictions(combined, resid, rookie_receiving=None, corrections=None):
    """Joint/multi-output Phase A: turns the SHARE predictions the main
    project_veterans loop produced for REFRAMED_SHARE_STATS rows
    (WR/TE/RB receiving_yards) into real pred_pg values, by composing them
    with the team_passing_yards_pg forecast _attach_team_total_pred left on
    the frame - see transitions.py's REFRAMED_SHARE_STATS/
    RECEIVING_SHARE_LABEL for the shared source of truth on which
    (position, stat) combos are reframed.

    This runs after depth-chart gating and after the rookie path exists.
    The exposure-weighted guard therefore sees shipped veteran shares and
    incoming rookies' implied shares on the same basis. `rookie_receiving`
    enters the denominator as implied shares - the
    user-diagnosed Robinson/Tate case: an incoming 1st-round WR consumes
    real target share the veteran share models can't see.

    Interval note: pred_pg_low/high = composed pred +/- empirical residual,
    with the residual in absolute rate units NOT scaled by any role
    discount. This used to be the least-bad reading of a mismatch (the
    residuals were calibrated on UNDISCOUNTED backtest predictions); since
    backtest.py started applying the Gate B ladder it is simply correct -
    the band is calibrated on discounted predictions and is added to one.
    veterans._attach_veteran_intervals now uses the same convention for the
    non-reframed rows, which previously took (pred + resid) * factor.

    Non-reframed rows pass through unchanged (minus the helper column)."""
    reframed_index = pd.MultiIndex.from_tuples(REFRAMED_SHARE_STATS, names=["position", "stat"])
    mask = combined.set_index(["position", "stat"]).index.isin(reframed_index)
    if not mask.any():
        return combined.drop(columns=_HELPER_COLS, errors="ignore")
    reframed = combined[mask].copy()
    other = combined[~mask].drop(columns=_HELPER_COLS, errors="ignore").copy()
    other["receiving_share_capped"] = np.nan
    other["receiving_share_normalized"] = np.nan

    extra_team_share = None
    if rookie_receiving is not None and not rookie_receiving.empty:
        team_totals = reframed.drop_duplicates("team").set_index("team")["team_total_pred"]
        rr = rookie_receiving.copy()
        rr["team_total_pred"] = rr["team"].map(team_totals)
        # A rookie on a team with no composable veteran total (or a 0
        # fallback total) contributes nothing rather than dividing by 0.
        rr = rr[rr["team_total_pred"] > 0]
        rr["weight"] = (
            pd.to_numeric(rr["projected_games"], errors="coerce") / SEASON_GAMES
        ).clip(0, 1).fillna(0.0)
        extra_team_share = (
            rr["pred_pg"] / rr["team_total_pred"] * rr["weight"]
        ).groupby(rr["team"]).sum()

    share_df = reframed[["team"]].copy()
    share_df["share"] = reframed["pred_pg"]
    # Participation weight for the cap denominator (Gate B). A player
    # projected for 2 games consumes 2/17 of the team's season share
    # budget, not all of it and not an arbitrary 0.15 of it. Missing
    # projected_games (an older models/ directory with no availability
    # model) falls back to 1.0 - the pre-Gate-B behaviour - rather than to
    # 0, which would drop that team's denominator to nothing and disable
    # the cap silently.
    exposure_col = "projected_games"
    share_df["weight"] = (
        (reframed[exposure_col] / SEASON_GAMES).clip(0, 1).fillna(1.0)
        if exposure_col in reframed.columns else 1.0
    )
    scale, over_cap = receiving_share_scale(share_df, extra_team_share=extra_team_share)
    reframed["receiving_share_capped"] = over_cap
    reframed["receiving_share_normalized"] = ~np.isclose(scale, 1.0)
    reframed["pred_pg"] = reframed["pred_pg"] * scale * reframed["team_total_pred"]

    # Phase 7: additive elite-shrinkage correction, applied AFTER
    # composition (it is fit in rate units on composed out-of-sample
    # residuals) and scaled by the row's own role discount so a
    # discounted player can't be handed an undiscounted bonus. Rows with
    # no observed season-N rate, or a position with no fitted parameters,
    # get exactly 0.0 - see corrections.elite_shrinkage_adjustment.
    reframed["elite_correction_pg"] = 0.0
    if corrections:
        adj = elite_shrinkage_adjustment(
            reframed["position"], reframed["_observed_recv_pg"], corrections)
        adj = adj * reframed["role_discount_factor"].fillna(1.0).to_numpy()
        reframed["elite_correction_pg"] = adj
        before = reframed["pred_pg"].to_numpy()
        reframed["pred_pg"] = before + adj
        # Carry the same proportional bump to the player's OTHER receiving
        # stats. Without this the correction adds yards to a tight end while
        # leaving his receptions and targets untouched, so the shipped row
        # says he gains 8 yards a game on exactly the same catches - an
        # internally inconsistent player in fantasy_points_<season>.csv,
        # where receptions are scored separately (0.5 each in this league).
        #
        # Proportional, and specifically NOT a separately-fit correction,
        # because the elite under-prediction is a uniform VOLUME effect
        # rather than a yards-per-catch one. Measured LOO over 2021-2025 on
        # the same above-knot cohort, actual/predicted per stat:
        #   WR (n=111): yards 1.04  receptions 1.06  targets 1.05  rec TDs 1.07
        #   TE (n=23):  yards 1.05  receptions 1.05  targets 1.05  rec TDs 1.08
        # The ratios are the same stat to stat, so holding the player's own
        # yards-per-reception fixed and scaling volume is what the data
        # says - and it needs no new fitted parameters, which matters given
        # the TE fit already rests on 23 rows.
        other = _propagate_elite_correction(other, reframed, before, adj)
    reframed = reframed.drop(columns=_HELPER_COLS, errors="ignore")

    r = resid[(resid["stat"] == "receiving_yards") & (resid["position"].isin(["WR", "TE", "RB"]))]
    reframed = reframed.merge(
        r[["position", "stat", "resid_low", "resid_high", "low_n_flag"]], on=["position", "stat"], how="left",
    )
    has_resid = reframed["resid_low"].notna()
    reframed.loc[has_resid, "pred_pg_low"] = (
        reframed.loc[has_resid, "pred_pg"] + reframed.loc[has_resid, "resid_low"]
    ).clip(lower=0)
    reframed.loc[has_resid, "pred_pg_high"] = reframed.loc[has_resid, "pred_pg"] + reframed.loc[has_resid, "resid_high"]
    reframed["interval_low_n_flag"] = reframed["low_n_flag"].fillna(True)
    reframed = reframed.drop(columns=["resid_low", "resid_high", "low_n_flag"])

    return pd.concat([other, reframed], ignore_index=True, sort=False)


# PASS_CATCH_COHERENCE_BAND and NAMED_REC_* live in contracts.py.


def reconcile_stat_constraints(df):
    """Enforce counting-stat identities at the final output boundary.

    The component models are intentionally fit independently, so small
    violations can occur even when every individual prediction is sensible.
    A completion cannot exist without an attempt and a reception cannot
    exist without a target. Cap the child stat at its matching parent for
    point and interval endpoints, and expose which rows were changed.
    """
    out = df.copy()
    # STICKY, not reset. compose_board calls this twice - once after the volume
    # normalizers and once as the trailing guard - and the column means "this
    # row was capped somewhere in composition", which is what OUTPUT_COLUMNS
    # ships and what fantasy_points.any_stat_constraint_applied aggregates.
    # Re-initialising it to False would let the second call erase the record of
    # every cap the first call made (28 rows on the 2026 board), turning an
    # audit trail into a report of the last call only.
    out["stat_constraint_applied"] = (
        out["stat_constraint_applied"].fillna(False).astype(bool)
        if "stat_constraint_applied" in out.columns else False
    )
    keys = [c for c in ["player_id", "position", "season"] if c in out.columns]
    if not keys:
        return out
    relations = (("completions", "attempts"), ("receptions", "targets"))
    value_cols = [c for c in ["pred_pg", "pred_pg_low", "pred_pg_high"] if c in out.columns]
    for child, parent in relations:
        child_mask = out["stat"] == child
        parent_rows = out[out["stat"] == parent]
        if not child_mask.any() or parent_rows.empty:
            continue
        if parent_rows.duplicated(keys).any():
            raise ValueError(f"duplicate {parent} rows prevent stat reconciliation")
        parent_rows = parent_rows.set_index(keys)
        child_index = pd.MultiIndex.from_frame(out.loc[child_mask, keys]) if len(keys) > 1 else pd.Index(out.loc[child_mask, keys[0]])
        child_positions = out.index[child_mask]
        for col in value_cols:
            parent_values = parent_rows[col].reindex(child_index).to_numpy()
            child_values = out.loc[child_positions, col].to_numpy(dtype=float)
            changed = np.isfinite(parent_values) & np.isfinite(child_values) & (child_values > parent_values)
            if changed.any():
                changed_index = child_positions[changed]
                out.loc[changed_index, col] = parent_values[changed]
                out.loc[changed_index, "stat_constraint_applied"] = True
    return out


def reconcile_qb_projected_volume_games(df, season_games=SEASON_GAMES):
    """Reconcile marginal QB appearances to a 17-game team exposure budget.

    ``projected_games`` and ``projected_games_raw`` retain the independently
    modeled, unconstrained appearance forecast. Quarterbacks are sequential,
    so that marginal forecast is not itself a season-volume allocation. This
    function creates a separate, mutually-exclusive
    ``projected_volume_games`` allocation:

    * non-QBs retain projected_games;
    * every resolved named-QB room is allocated exactly ``season_games``;
    * underfilled rooms water-fill upward using raw availability plus
      preseason role/depth priority, without erasing the raw forecast;
    * a team with exactly one curated starter preserves that starter's
      marginal exposure first when an overfull room must be reduced;
    * audit fields expose the raw room total, adjustment direction and each
      player's allocation scale.

    A named QB's allocated volume can exceed his raw marginal appearances in
    an underfilled room; that is the intended distinction between the two
    columns, not an overwrite of availability.
    """
    out = df.copy()
    current_games = pd.to_numeric(out.get("projected_games"), errors="coerce")
    if "projected_games_raw" in out.columns:
        out["projected_games_raw"] = pd.to_numeric(
            out["projected_games_raw"], errors="coerce"
        ).fillna(current_games)
    else:
        out["projected_games_raw"] = current_games
    out["projected_volume_games"] = pd.to_numeric(
        out.get("projected_games"), errors="coerce"
    )
    qb = out[out["position"] == "QB"]
    if qb.empty or "player_id" not in out.columns or "team" not in out.columns:
        return out

    player_cols = ["player_id", "team", "projected_games_raw"]
    for optional in ["role", "depth_chart_status", "depth_rank", "nfl_depth_rank"]:
        if optional in qb.columns:
            player_cols.append(optional)
    players = qb[player_cols].drop_duplicates(["player_id", "team"]).copy()
    players["raw"] = pd.to_numeric(
        players["projected_games_raw"], errors="coerce"
    ).clip(0, season_games).fillna(0.0)
    players["volume"] = players["raw"]
    players["priority"] = 1.0
    if "role" in players.columns:
        role_priority = players["role"].map({
            "starter": 4.0, "backup": 1.5, "deep_bench": 0.5,
        })
        players["priority"] = role_priority.fillna(players["priority"])
    # Depth is the fallback for rooms without a curated role.  Keep one
    # ordinal rather than multiplying role and depth together, which would
    # overstate noisy differences between feeds.
    depth = pd.Series(np.nan, index=players.index, dtype=float)
    for col in ["depth_rank", "nfl_depth_rank"]:
        if col in players.columns:
            depth = depth.fillna(pd.to_numeric(players[col], errors="coerce"))
    no_role = players["role"].isna() if "role" in players.columns else pd.Series(True, index=players.index)
    depth_priority = pd.cut(
        depth, bins=[-np.inf, 1, 2, 3, np.inf], labels=[4.0, 1.5, 0.75, 0.5]
    ).astype(float)
    players.loc[no_role & depth_priority.notna(), "priority"] = depth_priority[
        no_role & depth_priority.notna()
    ]
    players["priority"] = players["priority"].clip(lower=0.25)

    room_raw = players.groupby("team")["raw"].sum(min_count=1)
    directions = pd.Series(index=room_raw.index, dtype=object)
    resolved = players.assign(
        _resolved=players["player_id"].notna()
        & players["player_id"].astype(str).str.strip().ne("")
    ).groupby("team")["_resolved"].any()

    for team, idx in players.groupby("team").groups.items():
        idx = list(idx)
        vals = players.loc[idx, "raw"]
        total = vals.sum(min_count=1)
        if pd.isna(total):
            directions.at[team] = "unresolved"
            continue
        priority = players.loc[idx, "priority"]
        # A small base retains role/depth ordering even if the availability
        # model returned zero for every named player in a room.
        weights = (vals + 0.25) * priority
        starter = pd.Series(False, index=idx)
        if "role" in players.columns:
            starter = players.loc[idx, "role"].eq("starter")
            if "depth_chart_status" in players.columns:
                starter &= players.loc[idx, "depth_chart_status"].eq("curated")
        starter_idx = list(starter[starter].index)
        if abs(float(total) - float(season_games)) <= 1e-9:
            players.loc[idx, "volume"] = vals
            directions.at[team] = "exact"
        elif total < season_games:
            extra_capacity = (float(season_games) - vals).clip(lower=0.0)
            extra = _capped_proportional_allocation(
                weights, extra_capacity, float(season_games) - float(total)
            )
            players.loc[idx, "volume"] = vals.to_numpy(dtype=float) + extra
            directions.at[team] = "upward"
        elif len(starter_idx) == 1:
            sidx = starter_idx[0]
            starter_volume = min(float(players.at[sidx, "raw"]), float(season_games))
            others = [i for i in idx if i != sidx]
            remaining = max(0.0, season_games - starter_volume)
            if others and remaining > 0:
                players.at[sidx, "volume"] = starter_volume
                players.loc[others, "volume"] = _capped_proportional_allocation(
                    weights.reindex(others), players.loc[others, "raw"], remaining
                )
            else:
                players.loc[idx, "volume"] = 0.0
                players.at[sidx, "volume"] = season_games
            directions.at[team] = "downward"
        else:
            players.loc[idx, "volume"] = _capped_proportional_allocation(
                weights, vals, season_games
            )
            directions.at[team] = "downward"

    volume = players.set_index(["player_id", "team"])["volume"]
    key = pd.MultiIndex.from_frame(out[["player_id", "team"]])
    qb_volume = volume.reindex(key).to_numpy()
    is_qb = out["position"].eq("QB")
    out.loc[is_qb, "projected_volume_games"] = qb_volume[is_qb]
    assigned = players.groupby("team")["volume"].sum(min_count=1)
    residual = (float(season_games) - assigned).clip(lower=0.0)
    out["team_unmodeled_qb_volume_games"] = out["team"].map(residual)
    out["team_qb_raw_appearance_games"] = out["team"].map(room_raw)
    out["team_qb_volume_allocation_direction"] = out["team"].map(directions)
    out["team_qb_roster_resolved"] = out["team"].map(resolved)
    out["qb_volume_games_scale"] = np.nan
    raw_by_row = pd.to_numeric(out.loc[is_qb, "projected_games_raw"], errors="coerce")
    out.loc[is_qb, "qb_volume_games_scale"] = (
        pd.to_numeric(out.loc[is_qb, "projected_volume_games"], errors="coerce")
        / raw_by_row.replace(0, np.nan)
    )
    out["qb_volume_allocation_adjusted"] = False
    out.loc[is_qb, "qb_volume_allocation_adjusted"] = (
        pd.to_numeric(out.loc[is_qb, "projected_volume_games"], errors="coerce")
        - raw_by_row
    ).abs() > 1e-9
    resolved_assigned = assigned[resolved.reindex(assigned.index).fillna(False)]
    if not np.allclose(resolved_assigned.to_numpy(dtype=float), season_games, atol=1e-8):
        raise AssertionError("Resolved QB rooms must allocate exactly the team game budget")
    return out


# QB_ATTEMPTS_PER_VOLUME_GAME_MAX lives in contracts.py.


def _capped_proportional_allocation(raw, capacity, target):
    """Allocate ``target`` proportional to raw weights without exceeding caps."""
    raw = np.nan_to_num(np.asarray(raw, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    capacity = np.nan_to_num(np.asarray(capacity, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    raw, capacity = np.clip(raw, 0, None), np.clip(capacity, 0, None)
    target = min(max(float(target), 0.0), float(capacity.sum()))
    alloc = np.zeros(len(raw), dtype=float)
    active = capacity > 0
    remaining = target
    while remaining > 1e-10 and active.any():
        room = capacity - alloc
        weights = np.where(active, raw, 0.0)
        if weights.sum() <= 0:
            weights = np.where(active, room, 0.0)
        proposal = remaining * weights / weights.sum()
        hit = active & (proposal >= room - 1e-10)
        if not hit.any():
            alloc += proposal
            remaining = 0.0
            break
        alloc[hit] += room[hit]
        remaining -= float(room[hit].sum())
        active[hit] = False
    return np.minimum(alloc, capacity)


def _row_exposure(rows):
    volume = pd.to_numeric(
        rows["projected_volume_games"] if "projected_volume_games" in rows else
        pd.Series(np.nan, index=rows.index), errors="coerce")
    games = pd.to_numeric(
        rows["projected_games"] if "projected_games" in rows else
        pd.Series(np.nan, index=rows.index), errors="coerce")
    return volume.fillna(games).clip(lower=0.0)


def _map_team_field(out, values, column):
    out[column] = out["team"].map(values)


def normalize_team_passing_volume(df, season_games=SEASON_GAMES):
    """Allocate team passing anchors while preserving named-player support.

    Named QB attempt rates are water-filled only up to the historical-support
    ceiling.  Any remainder belongs to an explicit replacement-QB bucket.

    Named receivers are scaled toward ``NAMED_REC_YARDS_COVERAGE`` of the
    team pass-yardage anchor (proportional to current season volume): fill up
    when under-covered, cut only when over the hard 1.0 anchor. The residual
    share stays in ``team_unmodeled_receiving_yards_season`` for practice-
    squad / emergency / non-skill production.
    """
    out = df.copy()
    out["team_passing_volume_scale"] = np.nan
    out["team_pass_catch_ratio_pre_normalization"] = np.nan
    out["team_pass_catch_pre_normalization_flag"] = np.nan
    qb_attempts = out[out["position"].eq("QB") & out["stat"].eq("attempts")].copy()
    if qb_attempts.empty:
        raise ValueError("Cannot allocate team passing anchors without named QB attempt rows")

    qb_attempts["exposure"] = _row_exposure(qb_attempts)
    qb_attempts["raw_season"] = pd.to_numeric(qb_attempts["pred_pg"], errors="coerce").clip(lower=0) * qb_attempts["exposure"]
    qb_attempts["capacity"] = qb_attempts["exposure"] * QB_ATTEMPTS_PER_VOLUME_GAME_MAX
    qb_attempts["allocated"] = 0.0
    attempt_anchor = (
        qb_attempts.drop_duplicates("team").set_index("team")["team_pass_attempts_pg_pred"]
        * season_games
    )
    for team, idx in qb_attempts.groupby("team").groups.items():
        rows = qb_attempts.loc[idx]
        qb_attempts.loc[idx, "allocated"] = _capped_proportional_allocation(
            rows["raw_season"], rows["capacity"], attempt_anchor.at[team])

    key_cols = ["player_id", "team"]
    attempt_alloc = qb_attempts.set_index(key_cols)["allocated"]
    raw_attempt = qb_attempts.set_index(key_cols)["raw_season"]
    attempt_factor = (attempt_alloc / raw_attempt.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    attempt_stats = {"attempts", "completions", "passing_tds", "interceptions"}
    for idx in out.index[out["position"].eq("QB") & out["stat"].isin(attempt_stats)]:
        key = (out.at[idx, "player_id"], out.at[idx, "team"])
        factor = attempt_factor.get(key, np.nan)
        exposure = float(pd.to_numeric(pd.Series([out.at[idx, "projected_volume_games"]]), errors="coerce").iloc[0])
        if out.at[idx, "stat"] == "attempts" and exposure > 0:
            old = float(out.at[idx, "pred_pg"])
            new = float(attempt_alloc.get(key, 0.0)) / exposure
            endpoint_factor = new / old if old > 0 else 0.0
            out.loc[idx, ["pred_pg_low", "pred_pg_high"]] *= endpoint_factor
            out.at[idx, "pred_pg"] = new
            out.loc[idx, ["pred_pg", "pred_pg_low", "pred_pg_high"]] = out.loc[
                idx, ["pred_pg", "pred_pg_low", "pred_pg_high"]].clip(upper=QB_ATTEMPTS_PER_VOLUME_GAME_MAX)
            out.at[idx, "team_passing_volume_scale"] = endpoint_factor
        elif pd.notna(factor):
            out.loc[idx, ["pred_pg", "pred_pg_low", "pred_pg_high"]] *= factor
            out.at[idx, "team_passing_volume_scale"] = factor

    named_attempts = qb_attempts.groupby("team")["allocated"].sum(min_count=1)
    unmodeled_attempts = (attempt_anchor - named_attempts).clip(lower=0.0)
    if "team_qb_roster_resolved" in qb_attempts.columns:
        room_resolved = qb_attempts.drop_duplicates("team").set_index("team")[
            "team_qb_roster_resolved"
        ].fillna(False).astype(bool)
    else:
        room_resolved = qb_attempts.assign(
            _resolved=qb_attempts["player_id"].notna()
            & qb_attempts["player_id"].astype(str).str.strip().ne("")
        ).groupby("team")["_resolved"].any()
    impossible_resolved = unmodeled_attempts[
        room_resolved.reindex(unmodeled_attempts.index).fillna(False)
        & unmodeled_attempts.gt(1e-8)
    ]
    if not impossible_resolved.empty:
        detail = ", ".join(
            f"{team}={value:.2f}" for team, value in impossible_resolved.items()
        )
        raise ValueError(
            "Resolved QB rooms could not meet the team attempt anchor within "
            f"the {QB_ATTEMPTS_PER_VOLUME_GAME_MAX:.1f} attempts/game support cap: {detail}"
        )
    _map_team_field(out, unmodeled_attempts, "team_unmodeled_qb_attempts_season")
    _map_team_field(
        out, unmodeled_attempts.le(1e-8), "team_qb_attempt_anchor_fully_allocated"
    )

    qb_yards = out[out["position"].eq("QB") & out["stat"].eq("passing_yards")].copy()
    qb_yards["exposure"] = _row_exposure(qb_yards)
    current_yards = (
        pd.to_numeric(qb_yards["pred_pg"], errors="coerce").clip(lower=0) * qb_yards["exposure"]
    ).groupby(qb_yards["team"]).sum(min_count=1)
    yard_anchor = (
        qb_yards.drop_duplicates("team").set_index("team")["team_passing_yards_pg_pred"]
        * season_games
    )
    named_attempt_fraction = (named_attempts / attempt_anchor.replace(0, np.nan)).clip(0, 1).fillna(0)
    named_yard_target = yard_anchor * named_attempt_fraction
    yard_scale = (named_yard_target / current_yards.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for idx in out.index[out["position"].eq("QB") & out["stat"].eq("passing_yards")]:
        factor = yard_scale.get(out.at[idx, "team"], 0.0)
        out.loc[idx, ["pred_pg", "pred_pg_low", "pred_pg_high"]] *= factor
        out.at[idx, "team_passing_volume_scale"] = factor
    named_yards = named_yard_target.where(current_yards > 0, 0.0)
    unmodeled_qb_yards = (yard_anchor - named_yards).clip(lower=0.0)
    _map_team_field(out, unmodeled_qb_yards, "team_unmodeled_qb_passing_yards_season")

    recv_mask = out["position"].isin(["RB", "WR", "TE"]) & out["stat"].eq("receiving_yards")
    recv = out[recv_mask].copy()
    if recv.empty:
        raise ValueError("Cannot reconcile passing/receiving yards without receiver rows")
    recv["exposure"] = _row_exposure(recv)
    current_receiving = (
        pd.to_numeric(recv["pred_pg"], errors="coerce").clip(lower=0) * recv["exposure"]
    ).groupby(recv["team"]).sum(min_count=1)
    recv_anchor = recv.drop_duplicates("team").set_index("team")["team_passing_yards_pg_pred"] * season_games
    pre_ratio = current_receiving / recv_anchor.where(recv_anchor > 0)
    low, high = PASS_CATCH_COHERENCE_BAND
    pre_flag = (~pre_ratio.between(low, high)).astype(object)
    pre_flag.loc[pre_ratio.isna()] = np.nan
    _map_team_field(out, pre_ratio, "team_pass_catch_ratio_pre_normalization")
    _map_team_field(out, pre_flag, "team_pass_catch_pre_normalization_flag")

    mix_cols = ["wr_target_share", "te_target_share", "rb_target_share"]
    hierarchical = all(c in out.columns for c in mix_cols)
    named_receiving = pd.Series(0.0, index=current_receiving.index)

    if hierarchical:
        # Scale within (team, position) to the L2 slice of the pass-yard
        # anchor so a team-wide factor cannot wash out WR/TE/RB mix.
        group_share = {
            "WR": "wr_target_share", "TE": "te_target_share", "RB": "rb_target_share",
        }
        for idx in out.index[recv_mask]:
            team = out.at[idx, "team"]
            position = out.at[idx, "position"]
            gcol = group_share[position]
            gshare = float(out.at[idx, gcol]) if pd.notna(out.at[idx, gcol]) else np.nan
            anchor = float(recv_anchor.at[team]) if team in recv_anchor.index else np.nan
            # Season volume currently held by this position on this team.
            # Computed once per group below via groupby; placeholder fill.
            out.at[idx, "team_passing_volume_scale"] = 1.0

        pos_season = (
            pd.to_numeric(recv["pred_pg"], errors="coerce").clip(lower=0) * recv["exposure"]
        ).groupby([recv["team"], recv["position"]]).sum(min_count=1)
        scale_map = {}
        named_pos = {}
        for (team, position), raw in pos_season.items():
            raw = float(raw) if pd.notna(raw) else 0.0
            gcol = group_share[position]
            # Mix is constant within team; read from any row.
            team_rows = out[(out["team"] == team) & out[gcol].notna()]
            gshare = float(team_rows[gcol].iloc[0]) if not team_rows.empty else (1.0 / 3.0)
            anchor = float(recv_anchor.at[team]) * gshare if team in recv_anchor.index else np.nan
            if raw > 0 and np.isfinite(anchor):
                target = _named_supply_target(raw, anchor, NAMED_REC_YARDS_COVERAGE)
                scale_map[(team, position)] = target / raw
                named_pos[(team, position)] = target
            elif raw > 0:
                scale_map[(team, position)] = 1.0
                named_pos[(team, position)] = raw
            else:
                scale_map[(team, position)] = 0.0
                named_pos[(team, position)] = 0.0
        for idx in out.index[recv_mask]:
            key = (out.at[idx, "team"], out.at[idx, "position"])
            factor = float(scale_map.get(key, 0.0))
            out.loc[idx, ["pred_pg", "pred_pg_low", "pred_pg_high"]] *= factor
            out.at[idx, "team_passing_volume_scale"] = factor
        named_receiving = pd.Series(
            {team: sum(v for (t, _p), v in named_pos.items() if t == team)
             for team in current_receiving.index}
        )
    else:
        recv_scale = pd.Series(0.0, index=current_receiving.index)
        for team in current_receiving.index:
            raw = float(current_receiving.at[team])
            anchor = float(recv_anchor.at[team]) if team in recv_anchor.index else np.nan
            if raw > 0 and np.isfinite(anchor):
                target = _named_supply_target(raw, anchor, NAMED_REC_YARDS_COVERAGE)
                recv_scale.at[team] = target / raw
                named_receiving.at[team] = target
            elif raw > 0:
                recv_scale.at[team] = 1.0
                named_receiving.at[team] = raw
        for idx in out.index[recv_mask]:
            factor = float(recv_scale.get(out.at[idx, "team"], 0.0))
            out.loc[idx, ["pred_pg", "pred_pg_low", "pred_pg_high"]] *= factor
            out.at[idx, "team_passing_volume_scale"] = factor

    unmodeled_receiving = (
        recv_anchor.reindex(named_receiving.index).fillna(0.0) - named_receiving
    ).clip(lower=0.0)
    _map_team_field(out, unmodeled_receiving, "team_unmodeled_receiving_yards_season")
    return out


# RUSH_* ceilings, NAMED_RUSH_COVERAGE, USAGE_SHARE_* live in contracts.py.


def fit_usage_share_priors(conn, seasons):
    """Mean team share by position and preseason depth rank.

    Keyed on the nflverse preseason rank rather than the curated chart's
    `depth_rank` for two reasons. It is the only rank that exists for
    historical seasons, so it is the only one this can be fit and validated
    on - the same argument that keys Gate B's ladder. And the curated
    `depth_rank` is explicitly formation order: starters_2026.csv says so in
    its own notes column, which is exactly why it cannot be read as a volume
    ordering. Dallas is the case in point - the curated chart lists George
    Pickens ahead of CeeDee Lamb, while the nflverse chart has Lamb first.

    Rank is a real usage signal, not a proxy for one: Spearman between rank
    and target share is -0.62 for WR, with mean shares of 0.154 / 0.066 /
    0.038 across the top three rungs.
    """
    usage = load_weekly_usage(conn)
    usage = usage[usage["season"].isin(seasons)]
    if usage.empty:
        return pd.DataFrame()
    agg = usage.groupby(["season", "player_id", "team", "position"], as_index=False).agg(
        targets=("targets", "sum"), carries=("carries", "sum"))
    team = agg.groupby(["season", "team"])[["targets", "carries"]].sum().replace(0, np.nan)
    keys = pd.MultiIndex.from_frame(agg[["season", "team"]])
    agg["target_share"] = agg["targets"].to_numpy() / team["targets"].reindex(keys).to_numpy()
    agg["carry_share"] = agg["carries"].to_numpy() / team["carries"].reindex(keys).to_numpy()
    ranked = []
    for season in sorted(agg["season"].unique()):
        ranked.append(attach_depth_rank(agg[agg["season"] == season], int(season), conn=conn))
    agg = pd.concat(ranked, ignore_index=True)
    agg = agg[agg["nfl_depth_rank"].notna()]
    if agg.empty:
        return pd.DataFrame()
    agg["rank"] = agg["nfl_depth_rank"].clip(upper=USAGE_SHARE_MAX_RANK).astype(int)
    return agg.groupby(["position", "rank"])[["target_share", "carry_share"]].mean()


def apply_usage_share_prior(df, priors, depth_chart=None, weight=None):
    """Pull each room's modeled ordering toward what its depth ranks imply.

    The gap this closes: the per-(position, stat) models are independent and
    each carries a player's own prior-season usage forward, so nothing can
    correct an ordering the depth chart contradicts. George Pickens
    out-projects CeeDee Lamb on every receiving stat because his 2025 was
    better, and no part of the pipeline was able to say that Dallas's alpha
    receiver is Lamb - the curated chart's rank is formation order, and Gate
    B's ladder gives both a 1.00 multiplier, by construction: it was fit to
    calibrate a rate, not to rank a room.

    Group totals are preserved exactly. Each player's blended weight is
    renormalised within his (team, position) group, so this redistributes
    volume between teammates and never creates or destroys any. Every team
    anchor downstream still binds.

    Two priors feed this, at different strengths. A REVIEWED
    `usage_share_prior` on the curated chart carries USAGE_SHARE_CURATED_W,
    because research about one specific room beats a league average over a
    rank. The fitted rank prior carries USAGE_SHARE_BLEND_W, which ships at
    0.0 - it tested as a straight loss against real outcomes. An unreviewed
    curated value is a starting point for research, not a claim, and is
    ignored entirely.
    """
    if weight is None:
        weight = USAGE_SHARE_BLEND_W
    out = df.copy()
    out["usage_share_blend_factor"] = np.nan
    if priors is None or priors.empty:
        return out

    curated_prior = {}
    if depth_chart is not None and "usage_share_prior" in getattr(depth_chart, "columns", []):
        curated = depth_chart.dropna(subset=["gsis_id", "usage_share_prior"])
        if "usage_share_reviewed" in curated.columns:
            reviewed = curated["usage_share_reviewed"]
            curated = curated[
                reviewed.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))
            ]
        curated_prior = {
            (r["gsis_id"], r["position"]): float(r["usage_share_prior"])
            for _, r in curated.iterrows()
        }
    if weight <= 0 and not curated_prior:
        return out

    exposure = _row_exposure(out)
    rank = pd.to_numeric(out.get("nfl_depth_rank"), errors="coerce")
    banded = rank.clip(upper=USAGE_SHARE_MAX_RANK)

    for family in USAGE_SHARE_FAMILIES.values():
        prior_col = family["prior"]
        mask = out["position"].isin(family["positions"]) & out["stat"].isin(family["stats"])
        if not mask.any():
            continue
        volume_stat = family["stats"][0]
        vol_mask = mask & out["stat"].eq(volume_stat)
        vol = out[vol_mask].copy()
        vol["exposure"] = exposure[vol_mask]
        vol["season_volume"] = (
            pd.to_numeric(vol["pred_pg"], errors="coerce").clip(lower=0) * vol["exposure"])
        priors_and_weights = [
            (curated_prior[(pid, pos)], USAGE_SHARE_CURATED_W)
            if (pid, pos) in curated_prior
            else (priors[prior_col].get((pos, int(r)), np.nan) if pd.notna(r) else np.nan,
                  weight)
            for pid, pos, r in zip(vol["player_id"], vol["position"], banded[vol_mask])
        ]
        vol["prior"] = [p for p, _ in priors_and_weights]
        vol["prior_weight"] = [w for _, w in priors_and_weights]
        factors = {}
        for (team, position), group in vol.groupby(["team", "position"]):
            # Only players the preseason chart actually ranks are reordered,
            # among themselves; everyone else keeps exactly what the models
            # gave them. Requiring the WHOLE room to be ranked would disable
            # this entirely - charts run three deep and rooms do not - and
            # blending an unranked player toward a prior he has no rank for
            # would be inventing a number, not correcting one.
            rows = group[group["prior"].notna() & (group["prior_weight"] > 0)]
            if len(rows) < 2:
                continue
            total = rows["season_volume"].sum()
            prior = rows["prior"]
            if total <= 0 or prior.sum() <= 0:
                continue
            model_w = rows["season_volume"] / total
            prior_w = prior / prior.sum()
            # Per-row weight: a reviewed curated prior pulls harder than a
            # fitted rank one, so a room can mix a researched player with
            # rank-prior teammates without the research being diluted.
            w = rows["prior_weight"]
            blended = (1 - w) * model_w + w * prior_w
            blended = blended / blended.sum()
            new_volume = blended * total
            for idx, old, new in zip(rows.index, rows["season_volume"], new_volume):
                factors[(vol.at[idx, "player_id"], team, position)] = (
                    new / old if old > 0 else 1.0)
        if not factors:
            continue
        for idx in out.index[mask]:
            key = (out.at[idx, "player_id"], out.at[idx, "team"], out.at[idx, "position"])
            factor = factors.get(key)
            if factor is None:
                continue
            out.loc[idx, ["pred_pg", "pred_pg_low", "pred_pg_high"]] *= factor
            out.at[idx, "usage_share_blend_factor"] = factor
    return out


def _named_supply_target(raw_supply, anchor, coverage=NAMED_RUSH_COVERAGE):
    """How much team volume named players may be allocated.

    Their own raw supply, floored at the historically supported share of the
    anchor and capped by the anchor itself. A room already projecting above
    that share keeps what it projects; only a genuinely overfull one is cut.
    """
    if not np.isfinite(anchor):
        return float(raw_supply)
    return float(min(anchor, max(raw_supply, anchor * coverage)))


def normalize_team_rushing_volume(df, season_games=SEASON_GAMES):
    """Reconcile team rushing anchors against named-player supply.

    Post-role-gating raw projections are the allocation weights, preserving
    the modeled pecking order. Named players are allocated up to
    ``NAMED_RUSH_COVERAGE`` of the team anchor - the historically measured
    share a modelable roster actually accounts for - and never past the
    anchor itself. Whatever is left over stays an explicit residual in
    ``team_unmodeled_carries_season`` /
    ``team_unmodeled_rushing_yards_season`` instead of being pushed into
    whichever players happen to be modeled.

    The bug this fixes: pushing the FULL anchor into named players inflated
    a lead back to exactly his position capacity ceiling whenever his
    backfield was under-represented. Green Bay's charted committee back had
    no projection row, so Josh Jacobs absorbed the whole missing share and
    landed on the 25.0 carries/game cap at 1.72x his modeled rate. An
    anchor a modeled roster cannot account for is missing information, not
    opportunity belonging to the players who happen to be present - the
    same measured-coverage floor applied to receiving yards / TDs in
    normalize_team_passing_volume and reconcile_team_pass_receive_counts.

    Position capacity (RUSH_ATTEMPTS_PER_APPEARANCE_MAX /
    RUSH_YARDS_PER_CARRY_MAX) still binds each player independently, so an
    overfull room is redistributed within real single-player support.
    """
    out = df.copy()
    out["team_rushing_volume_scale"] = np.nan
    carry_mask = out["stat"].eq("carries") & out["position"].isin(RUSH_ATTEMPTS_PER_APPEARANCE_MAX)
    carries = out[carry_mask].copy()
    if carries.empty:
        raise ValueError("Cannot reconcile team carries: no modeled carry rows")
    carries["exposure"] = _row_exposure(carries)
    carries["raw_season"] = pd.to_numeric(carries["pred_pg"], errors="coerce").clip(lower=0) * carries["exposure"]
    carries["capacity"] = carries["exposure"] * carries["position"].map(RUSH_ATTEMPTS_PER_APPEARANCE_MAX)
    carries["allocated"] = 0.0
    carry_anchor = carries.drop_duplicates("team").set_index("team")["team_carries_pg_pred"] * season_games
    for team, idx in carries.groupby("team").groups.items():
        rows = carries.loc[idx]
        carries.loc[idx, "allocated"] = _capped_proportional_allocation(
            rows["raw_season"], rows["capacity"],
            _named_supply_target(rows["raw_season"].sum(), carry_anchor.at[team]))
    keys = ["player_id", "team"]
    carry_alloc = carries.set_index(keys)["allocated"]
    raw_carries = carries.set_index(keys)["raw_season"]
    carry_factor = (carry_alloc / raw_carries.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    for idx in out.index[carry_mask]:
        key = (out.at[idx, "player_id"], out.at[idx, "team"])
        exposure = float(out.at[idx, "projected_volume_games"])
        old = float(out.at[idx, "pred_pg"])
        new = float(carry_alloc.get(key, 0.0)) / exposure if exposure > 0 else 0.0
        factor = new / old if old > 0 else 0.0
        out.loc[idx, ["pred_pg_low", "pred_pg_high"]] *= factor
        out.at[idx, "pred_pg"] = new
        ceiling = RUSH_ATTEMPTS_PER_APPEARANCE_MAX[out.at[idx, "position"]]
        out.loc[idx, ["pred_pg", "pred_pg_low", "pred_pg_high"]] = out.loc[
            idx, ["pred_pg", "pred_pg_low", "pred_pg_high"]].clip(upper=ceiling)
        out.at[idx, "team_rushing_volume_scale"] = factor
    named_carries = carries.groupby("team")["allocated"].sum(min_count=1)
    _map_team_field(out, (carry_anchor - named_carries).clip(lower=0.0),
                    "team_unmodeled_carries_season")

    yard_mask = out["stat"].eq("rushing_yards") & out["position"].isin(RUSH_YARDS_PER_CARRY_MAX)
    yards = out[yard_mask].copy()
    yards["exposure"] = _row_exposure(yards)
    yards["raw_season"] = pd.to_numeric(yards["pred_pg"], errors="coerce").clip(lower=0) * yards["exposure"]
    yard_key = pd.MultiIndex.from_frame(yards[keys])
    yards["allocated_carries"] = carry_alloc.reindex(yard_key).to_numpy()
    yards["capacity"] = yards["allocated_carries"] * yards["position"].map(RUSH_YARDS_PER_CARRY_MAX)
    yards["allocated"] = 0.0
    yard_anchor = yards.drop_duplicates("team").set_index("team")["team_rushing_yards_pg_pred"] * season_games
    for team, idx in yards.groupby("team").groups.items():
        rows = yards.loc[idx]
        yards.loc[idx, "allocated"] = _capped_proportional_allocation(
            rows["raw_season"], rows["capacity"],
            _named_supply_target(rows["raw_season"].sum(), yard_anchor.at[team]))
    yard_alloc = yards.set_index(keys)["allocated"]
    raw_yards = yards.set_index(keys)["raw_season"]
    yard_factor = (yard_alloc / raw_yards.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    for idx in out.index[yard_mask]:
        key = (out.at[idx, "player_id"], out.at[idx, "team"])
        factor = yard_factor.get(key, 0.0)
        out.loc[idx, ["pred_pg", "pred_pg_low", "pred_pg_high"]] *= factor
        out.at[idx, "team_rushing_volume_scale"] = factor
    named_yards = yards.groupby("team")["allocated"].sum(min_count=1)
    _map_team_field(out, (yard_anchor - named_yards).clip(lower=0.0),
                    "team_unmodeled_rushing_yards_season")

    td_mask = out["stat"].eq("rushing_tds") & out["position"].isin(RUSH_ATTEMPTS_PER_APPEARANCE_MAX)
    for idx in out.index[td_mask]:
        key = (out.at[idx, "player_id"], out.at[idx, "team"])
        factor = carry_factor.get(key, 0.0)
        out.loc[idx, ["pred_pg", "pred_pg_low", "pred_pg_high"]] *= factor
        out.at[idx, "team_rushing_volume_scale"] = factor
    return out


def reconcile_team_pass_receive_counts(df, season_games=SEASON_GAMES):
    """Make completion/reception and pass-TD/rec-TD identities auditable.

    Named receivers are filled up to the measured coverage floors
    (``NAMED_REC_RECEPTIONS_COVERAGE`` / ``NAMED_REC_TDS_COVERAGE``) of the
    pass-side totals and cut only when they exceed those totals. Leftover
    production stays in explicit residual buckets for practice-squad /
    emergency / non-skill scores.
    """
    out = df.copy()
    out["team_pass_receive_count_scale"] = np.nan

    def season_sum(position_mask, stat):
        rows = out[position_mask & out["stat"].eq(stat)].copy()
        exposure = _row_exposure(rows)
        return (
            pd.to_numeric(rows["pred_pg"], errors="coerce").clip(lower=0) * exposure
        ).groupby(rows["team"]).sum(min_count=1)

    qb_mask = out["position"].eq("QB")
    recv_position = out["position"].isin(["RB", "WR", "TE"])
    named_attempts = season_sum(qb_mask, "attempts")
    named_completions = season_sum(qb_mask, "completions")
    named_pass_tds = season_sum(qb_mask, "passing_tds")
    unmodeled_attempts = out.drop_duplicates("team").set_index("team")[
        "team_unmodeled_qb_attempts_season"].reindex(named_attempts.index).fillna(0.0)
    completion_rate = (named_completions / named_attempts.replace(0, np.nan)).clip(0, 1).fillna(0.0)
    pass_td_rate = (named_pass_tds / named_attempts.replace(0, np.nan)).clip(lower=0).fillna(0.0)
    unmodeled_completions = unmodeled_attempts * completion_rate
    unmodeled_pass_tds = unmodeled_attempts * pass_td_rate
    total_completions = named_completions + unmodeled_completions
    total_pass_tds = named_pass_tds + unmodeled_pass_tds
    _map_team_field(out, unmodeled_completions, "team_unmodeled_qb_completions_season")
    _map_team_field(out, unmodeled_pass_tds, "team_unmodeled_qb_passing_tds_season")

    for stat, total, residual_col, coverage in (
        ("receptions", total_completions, "team_unmodeled_receptions_season",
         NAMED_REC_RECEPTIONS_COVERAGE),
        ("receiving_tds", total_pass_tds, "team_unmodeled_receiving_tds_season",
         NAMED_REC_TDS_COVERAGE),
    ):
        mix_cols = ["wr_target_share", "te_target_share", "rb_target_share"]
        hierarchical = all(c in out.columns for c in mix_cols)
        if hierarchical:
            group_share = {
                "WR": "wr_target_share", "TE": "te_target_share", "RB": "rb_target_share",
            }
            rows = out[recv_position & out["stat"].eq(stat)].copy()
            exposure = _row_exposure(rows)
            pos_season = (
                pd.to_numeric(rows["pred_pg"], errors="coerce").clip(lower=0) * exposure
            ).groupby([rows["team"], rows["position"]]).sum(min_count=1)
            scale_map = {}
            named_after_team = {}
            for (team, position), raw in pos_season.items():
                raw = float(raw) if pd.notna(raw) else 0.0
                gcol = group_share[position]
                team_rows = out[(out["team"] == team) & out[gcol].notna()]
                gshare = float(team_rows[gcol].iloc[0]) if not team_rows.empty else (1.0 / 3.0)
                anchor = float(total.at[team]) * gshare if team in total.index else np.nan
                if raw > 0 and np.isfinite(anchor):
                    target = _named_supply_target(raw, anchor, coverage)
                    scale_map[(team, position)] = target / raw
                    named_after_team[team] = named_after_team.get(team, 0.0) + target
                elif raw > 0:
                    scale_map[(team, position)] = 1.0
                    named_after_team[team] = named_after_team.get(team, 0.0) + raw
            mask = recv_position & out["stat"].eq(stat)
            for idx in out.index[mask]:
                key = (out.at[idx, "team"], out.at[idx, "position"])
                factor = float(scale_map.get(key, 0.0))
                out.loc[idx, ["pred_pg", "pred_pg_low", "pred_pg_high"]] *= factor
                out.at[idx, "team_pass_receive_count_scale"] = factor
            named_after = pd.Series(named_after_team)
            residual = (
                total.reindex(named_after.index).fillna(0.0) - named_after
            ).clip(lower=0.0)
            _map_team_field(out, residual, residual_col)
            continue

        named = season_sum(recv_position, stat)
        scale = pd.Series(0.0, index=named.index)
        named_after = pd.Series(0.0, index=named.index)
        for team in named.index:
            raw = float(named.at[team])
            anchor = float(total.at[team]) if team in total.index else np.nan
            if raw > 0 and np.isfinite(anchor):
                target = _named_supply_target(raw, anchor, coverage)
                scale.at[team] = target / raw
                named_after.at[team] = target
            elif raw > 0:
                scale.at[team] = 1.0
                named_after.at[team] = raw
        mask = recv_position & out["stat"].eq(stat)
        for idx in out.index[mask]:
            factor = float(scale.get(out.at[idx, "team"], 0.0))
            out.loc[idx, ["pred_pg", "pred_pg_low", "pred_pg_high"]] *= factor
            out.at[idx, "team_pass_receive_count_scale"] = factor
        residual = (
            total.reindex(named_after.index).fillna(0.0) - named_after
        ).clip(lower=0.0)
        _map_team_field(out, residual, residual_col)
    return out


def add_projected_season_totals(df):
    """Expose the canonical season totals used by downstream comparisons.

    Keeping both appearance and mutually-exclusive QB volume exposure is
    intentional, but it makes ``pred_pg * projected_games`` an invalid team
    total for quarterbacks. These explicit columns prevent consumers from
    accidentally recreating the original double-counting bug.
    """
    out = df.copy()
    exposure = pd.to_numeric(
        out.get("projected_volume_games"), errors="coerce"
    ).fillna(pd.to_numeric(out.get("projected_games"), errors="coerce"))
    for rate_col, season_col in (
        ("pred_pg", "pred_season"),
        ("pred_pg_low", "pred_season_low"),
        ("pred_pg_high", "pred_season_high"),
    ):
        out[season_col] = pd.to_numeric(out[rate_col], errors="coerce") * exposure
    return out


def add_team_pass_catch_coherence_flag(df, depth_chart=None):
    """Assert the post-normalization passing/receiving accounting identity.

    This computes expected receiving and passing yards per team game after
    ``normalize_team_passing_volume`` has forced both sides to the shared
    team-yardage anchor. It is therefore an output invariant, not independent
    model validation. The genuinely diagnostic, pre-adjustment values ship as
    ``team_pass_catch_ratio_pre_normalization`` and
    ``team_pass_catch_pre_normalization_flag``.

    Receivers are weighted by projected offensive appearances and QBs by
    reconciled volume-games. Explicit unmodeled QB/receiver residuals are then
    included on their respective side so the accounting identity holds after
    the measured coverage floors leave a small intentional residual.

    A missing exposure value falls back to weight 1.0 for backward
    compatibility with pre-availability model artifacts. A missing or zero
    expected QB denominator remains NaN, not a false claim of coherence."""
    df = df.copy()
    qb_rows = df[(df["position"] == "QB") & (df["stat"] == "passing_yards")]
    recv = df[(df["position"].isin(["WR", "TE", "RB"])) & (df["stat"] == "receiving_yards")]

    def _expected_per_team_game(rows, name, exposure_col):
        if rows.empty:
            return pd.Series(dtype=float, name=name)
        if exposure_col in rows.columns:
            weight = (pd.to_numeric(rows[exposure_col], errors="coerce") / SEASON_GAMES).clip(0, 1).fillna(1.0)
        else:
            weight = 1.0
        expected = pd.to_numeric(rows["pred_pg"], errors="coerce") * weight
        return expected.groupby(rows["team"]).sum(min_count=1).rename(name)

    qb_exposure = "projected_volume_games" if "projected_volume_games" in qb_rows.columns else "projected_games"
    recv_exposure = "projected_games"
    anchor = _expected_per_team_game(qb_rows, "qb_anchor_pg", qb_exposure)
    recv_sum = _expected_per_team_game(recv, "team_receiving_sum_pg", recv_exposure)

    per_team = df.drop_duplicates("team").set_index("team")
    if "team_unmodeled_qb_passing_yards_season" in per_team:
        anchor = anchor.add(
            pd.to_numeric(per_team["team_unmodeled_qb_passing_yards_season"], errors="coerce")
            / SEASON_GAMES,
            fill_value=0,
        ).rename("qb_anchor_pg")
    if "team_unmodeled_receiving_yards_season" in per_team:
        recv_sum = recv_sum.add(
            pd.to_numeric(per_team["team_unmodeled_receiving_yards_season"], errors="coerce")
            / SEASON_GAMES,
            fill_value=0,
        ).rename("team_receiving_sum_pg")

    team_ratio = pd.concat([anchor, recv_sum], axis=1)
    valid_qb = team_ratio["qb_anchor_pg"].where(team_ratio["qb_anchor_pg"] > 0)
    team_ratio["team_pass_catch_ratio"] = team_ratio["team_receiving_sum_pg"] / valid_qb
    low, high = PASS_CATCH_COHERENCE_BAND
    # object dtype (not bool) so the NaN assigned below for an unresolved
    # ratio is representable - a plain bool column can't hold NaN and
    # .between() on a NaN input silently evaluates to False, which would
    # otherwise misrepresent "not computable" as "confirmed coherent."
    team_ratio["team_pass_catch_coherence_flag"] = (
        ~team_ratio["team_pass_catch_ratio"].between(low, high)
    ).astype(object)
    # NaN (not False) when either side of the ratio isn't resolvable (no
    # expected QB passing volume, or no WR/TE/RB rows at all) -
    # "not computable" is a distinct, honestly-reported state from
    # "confirmed coherent."
    team_ratio.loc[team_ratio["team_pass_catch_ratio"].isna(), "team_pass_catch_coherence_flag"] = np.nan
    team_ratio = team_ratio.reset_index().rename(columns={"index": "team"})

    df = df.merge(
        team_ratio[["team", "team_pass_catch_ratio", "team_pass_catch_coherence_flag"]],
        on="team", how="left",
    )
    df["coherence_receiver_exposure_basis"] = "projected_games"
    return df


def _apply_rookie_depth_rate_gating(rookie_long):
    """Keep rookie conditional rates neutral to the veteran-only ladder.

    DEPTH_RATE_LADDER is fitted on veteran transition pairs. Applying it to
    structurally excluded rookies was neutral-to-harmful in the dedicated
    rookie backtest, so depth affects rookie availability only until a rookie
    calibration clears its own folds.
    """
    out = rookie_long.copy()
    out["role_discount_factor"] = 1.0
    out["role_discount_applied"] = False
    out["low_confidence"] = True
    return out


# TEAM_ANCHOR_OUTPUT_COLS lives in contracts.py.


def propagate_team_anchors(df):
    """Map the canonical veteran-built team frame onto rookie rows as well."""
    out = df.copy()
    for col in TEAM_ANCHOR_OUTPUT_COLS:
        if col not in out.columns:
            raise KeyError(f"Missing canonical team anchor column {col}")
        per_team = out.dropna(subset=[col]).drop_duplicates("team").set_index("team")[col]
        out[col] = out[col].fillna(out["team"].map(per_team))
    teams = out["team"].dropna().unique()
    if len(teams) != 32:
        raise ValueError(f"Expected projection rows for 32 anchored teams; found {len(teams)}")
    missing = out.loc[out[TEAM_ANCHOR_OUTPUT_COLS].isna().any(axis=1), "team"].dropna().unique()
    if len(missing):
        raise ValueError(f"Rows missing canonical team anchor provenance: {sorted(missing)}")
    return out
