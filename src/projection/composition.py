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
from src.projection.team_reconcile import (
    add_projected_season_totals,
    propagate_team_anchors,
    reconcile_stat_constraints,
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
            "reconcile_stat_constraints": "active",
            "add_projected_season_totals": "active",
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
    out = apply_full_season_games_baseline(rows, season_games=ctx.season_games)
    out = apply_status_overrides(out, ctx.status_overrides)
    out = propagate_team_anchors(out)
    out["projected_volume_games"] = pd.to_numeric(out.get("projected_games"), errors="coerce")
    # Top-down: pull each team's summed volume toward its own anchor before
    # the counting-stat identities and the season totals are materialised.
    out = reconcile_team_volume(out)
    out = reconcile_stat_constraints(out)
    return add_projected_season_totals(out)
