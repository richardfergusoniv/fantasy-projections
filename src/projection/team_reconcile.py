"""Team-anchor attach, elite receiving correction, and output hygiene.

Attaches Ridge team-total anchors, applies the elite receiving-yards
correction, pulls each team's summed volume partway toward its own anchor
(reconcile_team_volume), enforces counting-stat identities, and adds season
totals. It does not redistribute volume BETWEEN players - every scale is a
single factor applied to a whole team-position group, so within-team ordering
is untouched.

Does not import predict.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.projection.artifacts import load_reconcile_calibration, reconcile_alpha_for
from src.projection.contracts import (
    TEAM_ANCHOR_OUTPUT_COLS,
    QB_PASS_TD_RATE_CLIP,
    QB_RUSH_TD_PER_CARRY_CLIP,
    TEAM_RECONCILE_ALPHA,
    TEAM_RECONCILE_BENCH_FLOOR,
    TEAM_RECONCILE_CLIP,
    TEAM_RECONCILE_PROTECT_STARTER,
    TEAM_VOLUME_SHARES,
    TEAM_VOLUME_SIBLINGS,
)
from src.projection.corrections import elite_shrinkage_adjustment
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

    if resid is not None and not resid.empty and "stat" in resid.columns:
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
        reframed = reframed.drop(columns=["resid_low", "resid_high", "low_n_flag"], errors="ignore")

    return pd.concat([other, reframed], ignore_index=True, sort=False)



def reconcile_stat_constraints(df):
    """Enforce counting-stat identities at the final output boundary.

    The component models are intentionally fit independently, so small
    violations can occur even when every individual prediction is sensible.
    A completion cannot exist without an attempt and a reception cannot
    exist without a target. Cap the child stat at its matching parent for
    point and interval endpoints, and expose which rows were changed.
    """
    out = df.copy()
    # Sticky across any prior composition call that already set the flag.
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


def reconcile_td_rate_constraints(df, rush_td_hi=None, pass_td_clip=None):
    """Clip QB TD rates to plausible efficiency bands (T2/T3).

    ``passing_tds`` and ``rushing_tds`` are fit as independent counting models
    then scaled with attempts/carries siblings. When the implied TD rate exits
    historical starter bands, cap the child stat and expose the adjustment.

    ``rush_td_hi`` overrides the upper bound of ``QB_RUSH_TD_PER_CARRY_CLIP``
    for ablations (T3b). ``pass_td_clip`` overrides ``QB_PASS_TD_RATE_CLIP``.
    """
    out = df.copy()
    out["td_rate_clip_applied"] = (
        out["td_rate_clip_applied"].fillna(False).astype(bool)
        if "td_rate_clip_applied" in out.columns else False
    )
    keys = [c for c in ["player_id", "position", "season"] if c in out.columns]
    if len(keys) < 1 or out.empty:
        return out
    value_cols = [c for c in ["pred_pg", "pred_pg_low", "pred_pg_high"] if c in out.columns]
    pass_lo, pass_hi = pass_td_clip or QB_PASS_TD_RATE_CLIP
    rush_lo, rush_hi = QB_RUSH_TD_PER_CARRY_CLIP
    if rush_td_hi is not None:
        rush_hi = float(rush_td_hi)

    qb = out[out["position"].eq("QB")].copy()
    if qb.empty:
        return out

    def _apply_clip(group, parent_stat, child_stat, rate_lo, rate_hi):
        parent = group[group["stat"].eq(parent_stat)]
        child = group[group["stat"].eq(child_stat)]
        if parent.empty or child.empty:
            return
        parent_idx = parent.index[0]
        child_idx = child.index[0]
        for col in value_cols:
            parent_val = pd.to_numeric(out.loc[parent_idx, col], errors="coerce")
            child_val = pd.to_numeric(out.loc[child_idx, col], errors="coerce")
            if not np.isfinite(parent_val) or parent_val <= 0 or not np.isfinite(child_val):
                continue
            rate = child_val / parent_val
            if rate < rate_lo or rate > rate_hi:
                out.loc[child_idx, col] = parent_val * np.clip(rate, rate_lo, rate_hi)
                out.loc[child_idx, "td_rate_clip_applied"] = True

    for _, group in qb.groupby(keys, dropna=False):
        _apply_clip(group, "attempts", "passing_tds", pass_lo, pass_hi)
        _apply_clip(group, "carries", "rushing_tds", rush_lo, rush_hi)
    return out


def reconcile_pass_td_t1_lite(df, pass_td_clip=None):
    """T1-lite: re-derive passing_tds from attempts x clipped pass-TD rate.

    Post-compose hygiene only - does not retrain models. Keeps rush-TD handling
    separate so T3b ablations can stack with this pass-TD pass.
    """
    out = df.copy()
    pass_lo, pass_hi = pass_td_clip or QB_PASS_TD_RATE_CLIP
    keys = [c for c in ["player_id", "position", "season"] if c in out.columns]
    value_cols = [c for c in ["pred_pg", "pred_pg_low", "pred_pg_high"] if c in out.columns]
    qb = out[out["position"].eq("QB")]
    if qb.empty:
        return out
    for _, group in qb.groupby(keys, dropna=False):
        parent = group[group["stat"].eq("attempts")]
        child = group[group["stat"].eq("passing_tds")]
        if parent.empty or child.empty:
            continue
        parent_idx = parent.index[0]
        child_idx = child.index[0]
        for col in value_cols:
            parent_val = pd.to_numeric(out.loc[parent_idx, col], errors="coerce")
            child_val = pd.to_numeric(out.loc[child_idx, col], errors="coerce")
            if not np.isfinite(parent_val) or parent_val <= 0:
                continue
            rate = child_val / parent_val if np.isfinite(child_val) else np.nan
            if not np.isfinite(rate):
                continue
            clipped = parent_val * np.clip(rate, pass_lo, pass_hi)
            if not np.isclose(clipped, child_val):
                out.loc[child_idx, col] = clipped
                out.loc[child_idx, "td_rate_clip_applied"] = True
    return out


def reconcile_team_volume(
    df, alpha=None, calibration=None, volume_shares=None, volume_siblings=None
):
    """Pull each team's summed volume toward its own team anchor.

    ``volume_shares`` / ``volume_siblings`` default to the shipped contracts
    and exist so an ablation can vary ONE of them per run (see
    scripts/ablate_qb_volume_share.py). They are not ship knobs.

    The player models are fit independently, so nothing ties a team's summed
    output to what that team will run. Nothing downstream did either -
    compose_board was explicitly hygiene-only - and the result was 22 of 32
    teams projecting more QB pass attempts than any NFL team recorded in
    2021-2024.

    Partial by design, and the reason is measured rather than cautious:
    forcing the sum to the anchor is the best setting when the projected
    population covers the whole team and the WORST when it does not (+2.93%
    MAE on backtest folds averaging 85% coverage, -7.31% on folds above 90%),
    so the shipped value is the strongest setting that improves in both
    regimes.

    That value is now FITTED, not the constant. With ``alpha=None`` the
    per-cell alpha comes from models/reconcile_calibration.json (currently a
    0.75 default, chosen by rolling origin on player-level season-total MAE);
    `TEAM_RECONCILE_ALPHA` = 0.5 is only the fallback when that artifact is
    missing. Passing ``alpha`` explicitly overrides both and is for
    ablations. See contracts.TEAM_RECONCILE_ALPHA and
    scripts/fit_reconcile_alpha.py.

    This is deliberately NOT the "rescale receivers to match the QB's
    predicted volume" idea that backtested worse (8.94 -> 9.47 MAE) earlier in
    this project. That propagated one player model's error across a whole
    team; this scales a position group toward a separately fitted TEAM-grain
    forecast, and it is scored on player-level error, never on the team sum
    that alpha=1 would fix by construction.

    Runs on rates, before season totals are materialized, so pred_season
    follows automatically. Interval endpoints take the same scale as the
    point estimate.
    """
    calibration = calibration or load_reconcile_calibration()
    use_per_cell = alpha is None
    default_alpha = float(calibration.get("default_alpha", TEAM_RECONCILE_ALPHA))
    shares = TEAM_VOLUME_SHARES if volume_shares is None else volume_shares
    siblings = TEAM_VOLUME_SIBLINGS if volume_siblings is None else volume_siblings
    out = df.copy()
    out["team_volume_scale"] = 1.0
    if out.empty:
        return out
    if not use_per_cell and not alpha:
        return out

    exposure = _row_exposure(out)
    lo, hi = TEAM_RECONCILE_CLIP
    for (position, stat), (anchor_col, share) in shares.items():
        cell_alpha = (
            reconcile_alpha_for(position, stat, calibration)
            if use_per_cell
            else float(alpha)
        )
        if not cell_alpha:
            continue
        if anchor_col not in out.columns:
            continue
        group = [stat] + list(siblings.get((position, stat), ()))
        anchored = out["position"].eq(position) & out["stat"].eq(stat)
        if not anchored.any():
            continue
        rows = out[anchored]
        season = pd.to_numeric(rows["pred_pg"], errors="coerce") * exposure[anchored]
        summed = season.groupby(rows["team"]).sum()
        target = (
            pd.to_numeric(rows.drop_duplicates("team").set_index("team")[anchor_col],
                          errors="coerce")
            * SEASON_GAMES * share
        )
        touched = out["position"].eq(position) & out["stat"].isin(group)

        if TEAM_RECONCILE_PROTECT_STARTER.get(position, False) and "depth_tier" in rows.columns:
            # Bench absorbs the deficit first; the tier-1 starter is touched
            # only if flooring the bench still can't reach the target. See
            # contracts.TEAM_RECONCILE_PROTECT_STARTER for why this is a QB-
            # only shape: a QB room has one dominant starter whose own rate
            # is usually well calibrated, while an RB "starter" is often a
            # real committee member and correcting him alongside the room is
            # the right call, not a workaround for a miscalibrated bench.
            is_starter = rows["depth_tier"].eq(1.0)
            starter_pred = season[is_starter].groupby(rows.loc[is_starter, "team"]).sum().reindex(
                summed.index).fillna(0.0)
            bench_pred = season[~is_starter].groupby(rows.loc[~is_starter, "team"]).sum().reindex(
                summed.index).fillna(0.0)
            tgt = target.reindex(summed.index)
            needs_downward_reconcile = tgt < summed
            ordinary_scale = (
                (tgt / summed.replace(0, np.nan)).clip(lo, hi) ** cell_alpha
            ).fillna(1.0)
            bench_floor_amt = bench_pred * TEAM_RECONCILE_BENCH_FLOOR
            bench_alone_suffices = (starter_pred + bench_floor_amt) <= tgt
            bench_scale = pd.Series(np.where(
                bench_alone_suffices,
                ((tgt - starter_pred) / bench_pred.replace(0, np.nan)).clip(lo, hi),
                TEAM_RECONCILE_BENCH_FLOOR), index=summed.index).fillna(1.0)
            starter_scale = pd.Series(np.where(
                bench_alone_suffices, 1.0,
                ((tgt - bench_floor_amt) / starter_pred.replace(0, np.nan)).clip(lo, hi)),
                index=summed.index).fillna(1.0)

            starter_ids = set(rows.loc[is_starter, "player_id"])
            touched_team = out.loc[touched, "team"]
            touched_is_starter = out.loc[touched, "player_id"].isin(starter_ids)
            touched_needs_downward = touched_team.map(needs_downward_reconcile).fillna(False)
            protected_factor = np.where(
                touched_is_starter,
                touched_team.map(starter_scale).astype(float),
                touched_team.map(bench_scale).astype(float),
            )
            factor = pd.Series(
                np.where(
                    touched_needs_downward,
                    protected_factor,
                    touched_team.map(ordinary_scale).astype(float),
                ),
                index=out.loc[touched].index).fillna(1.0)
        else:
            ratio = (target.reindex(summed.index) / summed.replace(0, np.nan)).clip(lo, hi)
            scale = (ratio ** cell_alpha).reindex(summed.index)
            factor = out.loc[touched, "team"].map(scale).astype(float).fillna(1.0)
        for col in ("pred_pg", "pred_pg_low", "pred_pg_high"):
            if col in out.columns:
                out.loc[touched, col] = (
                    pd.to_numeric(out.loc[touched, col], errors="coerce") * factor)
        out.loc[touched, "team_volume_scale"] = factor
    return out


def _row_exposure(rows):
    volume = pd.to_numeric(
        rows["projected_volume_games"] if "projected_volume_games" in rows else
        pd.Series(np.nan, index=rows.index), errors="coerce")
    games = pd.to_numeric(
        rows["projected_games"] if "projected_games" in rows else
        pd.Series(np.nan, index=rows.index), errors="coerce")
    return volume.fillna(games).clip(lower=0.0)



def add_projected_season_totals(df):
    """Expose the canonical season totals used by downstream comparisons.

    Season totals are ``pred_pg × projected_volume_games``, with
    ``projected_volume_games`` falling back to ``projected_games`` when unset.
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


# Hard pass/catch identities for season totals. targets/attempts is not a 1.0
# identity (throwaways exist). 0.952 is the attempt-weighted 2021-2024 ratio
# after excluding 2025, whose target feed fails the source-coverage check
# (0.882 versus 0.951-0.959 in validated seasons).
TARGETS_PER_ATTEMPT = 0.952
TEAM_IDENTITY_PAIRS: list[tuple[str, str, float | None]] = [
    ("receiving_yards", "passing_yards", 1.0),
    ("receptions", "completions", 1.0),
    ("receiving_tds", "passing_tds", 1.0),
    ("targets", "attempts", TARGETS_PER_ATTEMPT),
]
RECEIVING_POSITIONS = ("RB", "WR", "TE")


def reconcile_team_season_identities(
    long: pd.DataFrame,
    *,
    pairs: list[tuple[str, str, float | None]] | None = None,
) -> pd.DataFrame:
    """Make each team's season totals satisfy the pass/catch identities.

    A team's season receiving yards are its season passing yards. Season totals
    are formed as each player's rate times *his own* ``projected_games``, and
    nothing upstream makes those games add up the same way across two position
    groups, so the identity has to be restored here.

    Three properties are deliberate:

    * **The target is the identity**, not whatever the rates currently imply.
    * **``targets`` / ``attempts``** reconciles to ``TARGETS_PER_ATTEMPT`` rather
      than 1.0 (throwaways are attempts with no target).
    * **The split is symmetric** (geometric): receivers and passers each move
      half the log-gap. Forcing receivers onto the QB volume backtested worse.

    Per-game rates (``pred_pg``) are deliberately untouched; only season columns
    move. Downstream season fantasy points must score from ``pred_season``.
    """
    if long.empty or "pred_season" not in long.columns:
        return long
    pairs = pairs or TEAM_IDENTITY_PAIRS
    out = long.copy()
    is_recv = out["position"].isin(RECEIVING_POSITIONS)
    is_qb = out["position"] == "QB"
    season_cols = [
        c for c in ("pred_season", "pred_season_low", "pred_season_high")
        if c in out.columns
    ]

    for recv_stat, pass_stat, target in pairs:
        recv_rows = is_recv & out["stat"].eq(recv_stat)
        pass_rows = is_qb & out["stat"].eq(pass_stat)

        def totals(rows: pd.Series) -> pd.Series:
            return out.loc[rows].groupby("team")["pred_season"].sum()

        recv_season = totals(recv_rows)
        pass_season = totals(pass_rows)

        teams = recv_season.index.union(pass_season.index)
        recv_season = recv_season.reindex(teams).fillna(0.0)
        pass_season = pass_season.reindex(teams).fillna(0.0)

        usable = (recv_season > 0) & (pass_season > 0)
        if not usable.any():
            continue

        if target is None:
            league = float(recv_season[usable].sum() / pass_season[usable].sum())
            wanted = pd.Series(league, index=recv_season[usable].index)
        else:
            wanted = pd.Series(float(target), index=recv_season[usable].index)

        have = recv_season[usable] / pass_season[usable]
        gap = (have / wanted).astype(float)
        adj = gap ** 0.5
        recv_scale = (1.0 / adj).to_dict()
        pass_scale = adj.to_dict()

        for col in season_cols:
            out.loc[recv_rows, col] = (
                pd.to_numeric(out.loc[recv_rows, col], errors="coerce")
                * out.loc[recv_rows, "team"].map(recv_scale).fillna(1.0)
            )
            out.loc[pass_rows, col] = (
                pd.to_numeric(out.loc[pass_rows, col], errors="coerce")
                * out.loc[pass_rows, "team"].map(pass_scale).fillna(1.0)
            )
    return out


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
