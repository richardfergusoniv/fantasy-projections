"""Compose-stage wrappers and experimental QB repair arms."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from src.projection.composition import CompositionContext, run_compose_stages
from src.projection.qb_repair.allocation import (
    AllocationParams,
    estimate_starter_backup_shares,
    reconcile_qb_volume_with_allocation,
)
from src.projection.qb_repair.rate_prior import (
    apply_qb_rate_prior,
    build_qb_rate_priors,
)
from src.projection.team_reconcile import reconcile_team_volume


ARM_BASELINE = "baseline"
ARM_ALLOCATION = "allocation"
ARM_MULTI_SEASON_PRIOR = "multi_season_prior"
ARM_MOBILE_RUSH_PRIOR = "mobile_rush_prior"
ARM_ALLOCATION_PLUS_PRIORS = "allocation_plus_priors"

ALL_ARMS = (
    ARM_BASELINE,
    ARM_ALLOCATION,
    ARM_MULTI_SEASON_PRIOR,
    ARM_MOBILE_RUSH_PRIOR,
    ARM_ALLOCATION_PLUS_PRIORS,
)


@dataclass
class ArmResult:
    arm: str
    board: pd.DataFrame
    provenance: dict = field(default_factory=dict)


def _replace_team_volume_stage(
    rows: pd.DataFrame,
    ctx: CompositionContext,
    *,
    volume_fn: Callable[[pd.DataFrame], tuple[pd.DataFrame, dict]],
) -> tuple[pd.DataFrame, dict]:
    """Run compose stages but swap team-volume reconciliation."""
    from src.projection.concentration import apply_concentration
    from src.projection.depth_gating import (
        apply_full_season_games_baseline,
        apply_status_overrides,
    )
    from src.projection.team_reconcile import (
        add_projected_season_totals,
        propagate_team_anchors,
        reconcile_pass_td_t1_lite,
        reconcile_stat_constraints,
        reconcile_td_rate_constraints,
        reconcile_team_season_identities,
    )

    provenance: dict = {}
    out = apply_full_season_games_baseline(
        rows,
        season_games=ctx.season_games,
        blend_alpha=ctx.exposure_blend_alpha,
    )
    out = apply_status_overrides(out, ctx.status_overrides)
    out = propagate_team_anchors(out)
    out["projected_volume_games"] = pd.to_numeric(out.get("projected_games"), errors="coerce")

    out, vol_report = volume_fn(out)
    provenance["team_volume"] = vol_report

    out = apply_concentration(out)
    out = reconcile_td_rate_constraints(out, rush_td_hi=ctx.qb_rush_td_clip_hi)
    if ctx.qb_pass_td_t1_lite:
        out = reconcile_pass_td_t1_lite(out)
    out = reconcile_stat_constraints(out)
    out = add_projected_season_totals(out)
    out = reconcile_team_season_identities(out)
    return out, provenance


def _baseline_volume(ctx: CompositionContext):
    def _fn(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        out = reconcile_team_volume(
            frame,
            alpha=ctx.reconcile_alpha,
            volume_shares=ctx.team_volume_shares,
            volume_siblings=ctx.team_volume_siblings,
        )
        return out, {"mode": "shipped_reconcile_team_volume"}

    return _fn


def _allocation_volume(ctx: CompositionContext, allocation: AllocationParams):
    def _fn(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        # Preserve shipped RB (and any non-QB) reconciliation exactly, then
        # replace only the QB passing-volume allocation.
        base = reconcile_team_volume(
            frame,
            alpha=ctx.reconcile_alpha,
            volume_shares=ctx.team_volume_shares,
            volume_siblings=ctx.team_volume_siblings,
        )
        # Reset QB scales before re-allocating so we do not compound factors.
        qb = base["position"].astype(str).eq("QB")
        if qb.any():
            # Revert QB pred_pg to pre-reconcile rates using recorded scales.
            scale = pd.to_numeric(base.loc[qb, "team_volume_scale"], errors="coerce").replace(0, np.nan).fillna(1.0)
            for col in ("pred_pg", "pred_pg_low", "pred_pg_high"):
                if col in base.columns:
                    base.loc[qb, col] = pd.to_numeric(base.loc[qb, col], errors="coerce") / scale
            base.loc[qb, "team_volume_scale"] = 1.0
        return reconcile_qb_volume_with_allocation(
            base,
            allocation=allocation,
            alpha=ctx.reconcile_alpha,
            volume_shares=ctx.team_volume_shares,
            volume_siblings=ctx.team_volume_siblings,
        )

    return _fn


def run_arm(
    raw_long: pd.DataFrame,
    ctx: CompositionContext,
    arm: str,
    *,
    target_season: int,
) -> ArmResult:
    """Execute one experimental arm on a pre-compose long board."""
    if arm not in ALL_ARMS:
        raise ValueError(f"Unknown arm {arm!r}; expected one of {ALL_ARMS}")

    working = raw_long.copy()
    provenance: dict = {"arm": arm, "target_season": int(target_season)}
    allocation = estimate_starter_backup_shares(target_season=target_season)
    provenance["allocation_params"] = allocation.__dict__

    tier1_ids = []
    qb = working[working["position"].astype(str).eq("QB")]
    if "depth_tier" in qb.columns:
        tier1_ids = (
            qb[pd.to_numeric(qb["depth_tier"], errors="coerce").eq(1.0)]["player_id"]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
    elif "depth_rank" in qb.columns:
        tier1_ids = (
            qb[pd.to_numeric(qb["depth_rank"], errors="coerce").eq(1.0)]["player_id"]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

    use_prior = arm in {
        ARM_MULTI_SEASON_PRIOR,
        ARM_MOBILE_RUSH_PRIOR,
        ARM_ALLOCATION_PLUS_PRIORS,
    }
    if use_prior:
        priors = build_qb_rate_priors(
            target_season=target_season,
            player_ids=tier1_ids or None,
        )
        working, prior_audit = apply_qb_rate_prior(
            working,
            priors,
            only_tier1=True,
            mobile_rushing_only=(arm == ARM_MOBILE_RUSH_PRIOR),
        )
        provenance["rate_prior_audit"] = prior_audit

    if arm in {ARM_ALLOCATION, ARM_ALLOCATION_PLUS_PRIORS}:
        volume_fn = _allocation_volume(ctx, allocation)
    else:
        volume_fn = _baseline_volume(ctx)

    # Arms that only change the prior still use shipped reconcile.
    if arm == ARM_BASELINE:
        final, _ = run_compose_stages(working, ctx, capture_checkpoints=False)
        provenance["team_volume"] = {"mode": "shipped_reconcile_team_volume"}
        return ArmResult(arm=arm, board=final, provenance=provenance)

    final, stage_prov = _replace_team_volume_stage(working, ctx, volume_fn=volume_fn)
    provenance.update(stage_prov)
    return ArmResult(arm=arm, board=final, provenance=provenance)
