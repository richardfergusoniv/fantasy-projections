"""The one post-forecast board pipeline, shared by ship and by measurement.

After veterans, rookies and replacement rows are concatenated, this module
does post-forecast hygiene plus ONE volume step - a partial top-down pull of
each team's summed output toward that team's own anchor (see
team_reconcile.reconcile_team_volume). It does not invent or redistribute
volume between players. Specifically it:

  * sets draft exposure to a full season (Gate A stays in projected_games_raw)
  * applies IR / PUP / suspension status overrides
  * fans team-anchor metadata onto every row
  * enforces child ≤ parent counting-stat identities
  * materializes ``pred_season = pred_pg × projected_games``

Forecast-stage work (models, Gate A/B, roster moves, replacement construction)
stays with each caller. Artifact provenance still differs between ship
(``models/`` + curated research) and leakage-safe eval (refit through
source_season); ``compose_board`` itself is identical.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.projection.depth_gating import (
    apply_full_season_games_baseline,
    apply_status_overrides,
    load_depth_chart,
    load_status_overrides,
)
from src.projection.contracts import EXPOSURE_BLEND_ALPHA
from src.projection.concentration import apply_concentration
from src.projection.team_reconcile import (
    add_projected_season_totals,
    propagate_team_anchors,
    reconcile_pass_td_t1_lite,
    reconcile_stat_constraints,
    reconcile_td_rate_constraints,
    reconcile_team_season_identities,
    reconcile_team_volume,
)
from src.projection.transitions import SEASON_GAMES

SHIPPED_ARTIFACTS = "shipped_models_and_curated_research"
LEAKAGE_SAFE_ARTIFACTS = "refit_on_history_through_source_season"


@dataclass
class CompositionContext:
    """Artifacts ``compose_board`` needs, plus how they were obtained.

    ``depth_chart`` and ``status_overrides`` are hand-curated research files
    that exist for 2026 only. For any other season they arrive EMPTY and the
    stages that read them become pass-throughs; that is recorded in
    ``stage_coverage`` rather than hidden.
    """

    target_season: int
    depth_chart: pd.DataFrame
    status_overrides: pd.DataFrame
    artifact_provenance: str
    season_games: float = SEASON_GAMES
    exposure_blend_alpha: float = EXPOSURE_BLEND_ALPHA
    # TD-architecture ablation toggles (ship defaults keep these None/False).
    qb_rush_td_clip_hi: float | None = None
    qb_pass_td_t1_lite: bool = False
    stage_coverage: dict = field(default_factory=dict)

    def describe_coverage(self):
        """Per-stage 'ran on real inputs' / 'degraded, and why'."""
        overrides = self.status_overrides is not None and not self.status_overrides.empty
        coverage = {
            "apply_full_season_games_baseline": "active",
            "apply_status_overrides": (
                "active" if overrides else
                f"no-op: no status_overrides_{self.target_season}.csv"),
            "propagate_team_anchors": "active",
            "reconcile_team_volume": "active",
            "apply_concentration": "active",
            "reconcile_td_rate_constraints": "active",
            "reconcile_stat_constraints": "active",
            "add_projected_season_totals": "active",
            "reconcile_team_season_identities": "active",
        }
        coverage.update(self.stage_coverage)
        return coverage


def shipped_context(conn, target_season, usage_prior_seasons=None, as_of=None):
    """Context built from curated research files (ship provenance).

    ``usage_prior_seasons`` is accepted but ignored — usage-share priors were
    retired with the volume-composition path. Kept so existing call sites do
    not break mid-refactor.
    """
    del conn, usage_prior_seasons  # no fitted mix/priors on the slim path
    depth_chart = load_depth_chart(target_season)
    return CompositionContext(
        target_season=target_season,
        depth_chart=depth_chart,
        status_overrides=load_status_overrides(target_season, as_of=as_of),
        artifact_provenance=SHIPPED_ARTIFACTS,
    )


def leakage_safe_context(conn, target_season, source_season):
    """Context for a held-out fold: curated files only, same loaders as ship.

    Mix profiles and usage priors no longer exist. The curated depth chart and
    status overrides still degrade to pass-throughs when absent for historical
    seasons, and say so in ``stage_coverage``.
    """
    del conn, source_season
    depth_chart = load_depth_chart(target_season)
    ctx = CompositionContext(
        target_season=target_season,
        depth_chart=depth_chart,
        status_overrides=load_status_overrides(target_season),
        artifact_provenance=LEAKAGE_SAFE_ARTIFACTS,
    )
    if depth_chart.empty:
        ctx.stage_coverage["_curated_depth_chart"] = (
            f"absent: src/depth_chart/starters_{target_season}.csv does not exist. "
            f"Curated membership, roles, formation roles and replacement-level "
            f"rows are therefore unmeasurable on this fold.")
    return ctx


def compose_board(rows, ctx):
    """Post-forecast hygiene from concatenated player rows to finished board.

    Draft exposure is a full season except IR / PUP / suspension overrides.
    ``projected_volume_games`` equals ``projected_games``.
    """
    out = apply_full_season_games_baseline(
        rows,
        season_games=ctx.season_games,
        blend_alpha=ctx.exposure_blend_alpha,
    )
    out = apply_status_overrides(out, ctx.status_overrides)
    out = propagate_team_anchors(out)
    out["projected_volume_games"] = pd.to_numeric(out.get("projected_games"), errors="coerce")
    # Top-down: pull each team's summed volume toward its own anchor before
    # the counting-stat identities and the season totals are materialised.
    out = reconcile_team_volume(out)
    out = apply_concentration(out)
    out = reconcile_td_rate_constraints(out, rush_td_hi=ctx.qb_rush_td_clip_hi)
    if ctx.qb_pass_td_t1_lite:
        out = reconcile_pass_td_t1_lite(out)
    out = reconcile_stat_constraints(out)
    out = add_projected_season_totals(out)
    # Season totals use each player's own projected_games, so QB rooms and
    # receiver rooms can diverge even when rates were coherent. Restore the
    # hard pass/catch identities on season columns only (rates untouched).
    return reconcile_team_season_identities(out)


def compose_board_stages(rows, ctx):
    """Return fantasy-relevant compose checkpoints for stage attribution."""
    from src.projection.fantasy_points import SCORING

    def _qb_ppg(frame):
        if frame.empty:
            return {}
        qb = frame[frame["position"].eq("QB")].copy()
        if qb.empty:
            return {}
        wide = qb.pivot_table(
            index="player_id",
            columns="stat",
            values="pred_pg",
            aggfunc="first",
        )
        score = pd.Series(0.0, index=wide.index)
        for stat, weight in SCORING.items():
            if stat in wide.columns:
                score = score + pd.to_numeric(wide[stat], errors="coerce").fillna(0.0) * weight
        names = (
            qb.drop_duplicates("player_id").set_index("player_id")["display_name"]
            if "display_name" in qb.columns
            else pd.Series(index=wide.index, dtype=str)
        )
        teams = qb.drop_duplicates("player_id").set_index("player_id")["team"]
        out = {}
        for pid, ppg in score.items():
            out[pid] = {
                "fantasy_ppg": round(float(ppg), 3),
                "display_name": str(names.get(pid, pid)),
                "team": str(teams.get(pid, "")),
            }
        return out

    baseline = apply_full_season_games_baseline(
        rows,
        season_games=ctx.season_games,
        blend_alpha=ctx.exposure_blend_alpha,
    )
    baseline = apply_status_overrides(baseline, ctx.status_overrides)
    baseline = propagate_team_anchors(baseline)
    baseline["projected_volume_games"] = pd.to_numeric(
        baseline.get("projected_games"), errors="coerce"
    )
    post_reconcile = reconcile_team_volume(baseline.copy())
    post_concentration = apply_concentration(post_reconcile.copy())
    post_td = reconcile_td_rate_constraints(
        post_concentration.copy(), rush_td_hi=ctx.qb_rush_td_clip_hi
    )
    if ctx.qb_pass_td_t1_lite:
        post_td = reconcile_pass_td_t1_lite(post_td)
    final = reconcile_stat_constraints(post_td.copy())
    final = add_projected_season_totals(final)
    final = reconcile_team_season_identities(final)
    return {
        "raw_model": _qb_ppg(baseline),
        "post_team_volume_reconcile": _qb_ppg(post_reconcile),
        "post_concentration": _qb_ppg(post_concentration),
        "post_td_clip": _qb_ppg(post_td),
        "final_shipped": _qb_ppg(final),
    }
