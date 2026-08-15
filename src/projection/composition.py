"""The one composition/allocation pipeline, shared by ship and by measurement.

Why this module exists
----------------------
There used to be two implementations of "turn per-player model output into a
reconciled board": ``predict.project_season`` (16 sequential stages, the thing
that ships) and ``fantasy_evaluation._compose_and_reconcile`` (7 stages, the
only thing scored against real fantasy outcomes). They drifted. Every new
allocation layer widened the gap, and the layers were added on the side that is
never measured.

They could not simply be merged, because they legitimately differ in ARTIFACT
PROVENANCE:

  * ``project_season`` loads model binaries from ``models/``, trained by
    train.py on every available season including the target's source season and
    everything before it. That is correct for shipping.
  * the evaluation harness deliberately REFITS its models on history sliced to
    ``season <= source_season`` so a 2024 -> 2025 fold stays leakage-safe. That
    is the entire value of the harness.

This module separates the two concerns. ``CompositionContext`` carries the
artifacts (depth chart, status overrides, usage priors, L2 mix profiles) and the
CALLER decides how those artifacts were produced — fitted leakage-safely for a
held-out fold, or loaded from ``models/`` and the curated 2026 research files.
``compose_board`` then runs the identical stage sequence over them.

The boundary is deliberate: ``compose_board`` starts at the point where veteran,
rookie and replacement rows have already been concatenated into one long frame,
and runs to the finished board. Everything before that (fitting/loading rate
models, availability, the veteran depth-rate ladder, roster reassignment,
replacement-row construction) is FORECAST-stage work that necessarily differs
between provenances and stays with each caller.

Honest-coverage rule
--------------------
A stage whose input does not exist for a given season must NO-OP VISIBLY, never
be faked and never be quietly dropped. ``CompositionContext.stage_coverage``
records, per stage, whether it ran with real inputs or degraded to a pass-through
— and the evaluation harness copies that map into its metadata so a coverage
number can never be mistaken for a performance number.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.projection.depth_gating import (
    apply_deep_bench_games_cap,
    apply_status_overrides,
    load_depth_chart,
    load_status_overrides,
)
from src.projection.team_pass_mix import (
    apply_hierarchical_pass_distribution,
    attach_team_pass_mix,
    build_team_pass_mix_profiles,
)
from src.projection.team_reconcile import (
    add_projected_season_totals,
    add_team_pass_catch_coherence_flag,
    apply_usage_share_prior,
    fit_usage_share_priors,
    normalize_team_passing_volume,
    normalize_team_rushing_volume,
    propagate_team_anchors,
    reconcile_qb_projected_volume_games,
    reconcile_stat_constraints,
    reconcile_team_pass_receive_counts,
)
from src.projection.team_rush_mix import (
    apply_hierarchical_rush_distribution,
    attach_team_rush_mix,
    build_team_rush_mix_profiles,
)
from src.projection.transitions import SEASON_GAMES

# Provenance labels for the artifacts a context carries. These describe HOW the
# inputs were produced, never WHAT composition does with them.
SHIPPED_ARTIFACTS = "shipped_models_and_curated_research"
LEAKAGE_SAFE_ARTIFACTS = "refit_on_history_through_source_season"


@dataclass
class CompositionContext:
    """Everything ``compose_board`` needs, plus how it was obtained.

    ``depth_chart`` and ``status_overrides`` are hand-curated research files
    that exist for 2026 only. For any other season they arrive EMPTY and the
    stages that read them become pass-throughs; that is recorded in
    ``stage_coverage`` rather than hidden.
    """

    target_season: int
    depth_chart: pd.DataFrame
    status_overrides: pd.DataFrame
    usage_share_priors: pd.DataFrame
    pass_mix_profiles: pd.DataFrame
    rush_mix_profiles: pd.DataFrame
    artifact_provenance: str
    season_games: float = SEASON_GAMES
    stage_coverage: dict = field(default_factory=dict)

    def _mix_covers_target(self, profiles):
        return (
            profiles is not None
            and not profiles.empty
            and bool((profiles["season"] == self.target_season).any())
        )

    def describe_coverage(self):
        """Per-stage 'ran on real inputs' / 'degraded, and why'.

        Computed from the artifacts actually present, so it cannot claim
        coverage a context does not have.
        """
        curated = self.depth_chart is not None and not self.depth_chart.empty
        overrides = self.status_overrides is not None and not self.status_overrides.empty
        priors = self.usage_share_priors is not None and not self.usage_share_priors.empty
        formation = curated and "formation_role" in getattr(
            self.depth_chart, "columns", [])
        coverage = {
            "apply_deep_bench_games_cap": (
                "active" if curated else
                f"no-op: no curated depth chart for {self.target_season}, so no row is "
                f"marked deep_bench_discounted"),
            "apply_status_overrides": (
                "active" if overrides else
                f"no-op: no status_overrides_{self.target_season}.csv"),
            "propagate_team_anchors": "active",
            "reconcile_qb_projected_volume_games": (
                "active" if curated else
                "active, degraded: QB room priority falls back to nflverse depth rank "
                "because no curated role column exists for this season"),
            "apply_usage_share_prior": (
                "active" if (curated and priors) else
                "no-op: the fitted rank prior ships at weight 0 and there is no "
                "reviewed curated usage_share_prior for this season"),
            "attach_team_pass_mix": (
                "active" if self._mix_covers_target(self.pass_mix_profiles) else
                "degraded: no L2 pass-mix row for the target season; league-mean "
                "fallback applies"),
            "apply_hierarchical_pass_distribution": (
                "active" if formation else
                "active, degraded: no curated formation_role column, so the within-WR "
                "split is fungible instead of LWR/RWR/SWR two-stage"),
            "attach_team_rush_mix": (
                "active" if self._mix_covers_target(self.rush_mix_profiles) else
                "degraded: no L2 rush-mix row for the target season; league-mean "
                "fallback applies"),
            "apply_hierarchical_rush_distribution": "active",
            "normalize_team_passing_volume": "active",
            "normalize_team_rushing_volume": "active",
            "reconcile_stat_constraints": "active",
            "reconcile_team_pass_receive_counts": "active",
            "add_team_pass_catch_coherence_flag": "active",
            "add_projected_season_totals": "active",
        }
        coverage.update(self.stage_coverage)
        return coverage


def shipped_context(conn, target_season, usage_prior_seasons, as_of=None):
    """Context built from ``models/`` artifacts and the curated research files.

    This is the provenance that ships. Every artifact may see every completed
    season, which is correct for a real forward projection and is exactly what
    makes it unusable for a held-out fold.
    """
    depth_chart = load_depth_chart(target_season)
    pass_mix, _ = build_team_pass_mix_profiles(conn, target_season=target_season)
    rush_mix, _ = build_team_rush_mix_profiles(conn, target_season=target_season)
    return CompositionContext(
        target_season=target_season,
        depth_chart=depth_chart,
        status_overrides=load_status_overrides(target_season, as_of=as_of),
        usage_share_priors=fit_usage_share_priors(conn, usage_prior_seasons),
        pass_mix_profiles=pass_mix,
        rush_mix_profiles=rush_mix,
        artifact_provenance=SHIPPED_ARTIFACTS,
    )


def leakage_safe_context(conn, target_season, source_season):
    """Context for a held-out fold: nothing here may see ``target_season``.

    Every fitted artifact is bounded at ``source_season``:

    * the L2 pass/rush mix profiles fit and lag on observed seasons
      <= source_season (``history_seasons``), and score the target season from
      ``source_season`` scheme features — the same framing the shipped path uses
      for its own target season;
    * the usage-share rank prior is fit on seasons <= source_season;
    * the curated depth chart and status overrides are loaded through the SAME
      loader the shipped path uses, which returns empty for any season without a
      research file. They are NOT substituted with anything, so the stages that
      consume them degrade to pass-throughs and say so in ``stage_coverage``.

    OC mix inheritance keys on ``(season, team)`` in oc_assignments.csv, whose
    rows are preseason-known coaching hires, so it is applied at its real
    historical value rather than suppressed.
    """
    history = list(range(2016, source_season + 1))
    depth_chart = load_depth_chart(target_season)
    pass_mix, _ = build_team_pass_mix_profiles(
        conn, target_season=target_season, history_seasons=history)
    rush_mix, _ = build_team_rush_mix_profiles(
        conn, target_season=target_season, history_seasons=history)
    ctx = CompositionContext(
        target_season=target_season,
        depth_chart=depth_chart,
        status_overrides=load_status_overrides(target_season),
        usage_share_priors=fit_usage_share_priors(conn, history),
        pass_mix_profiles=pass_mix,
        rush_mix_profiles=rush_mix,
        artifact_provenance=LEAKAGE_SAFE_ARTIFACTS,
    )
    if depth_chart.empty:
        # Named rather than inferred, because "the curated chart does not exist
        # for this season" is a different statement from "the chart exists and
        # lists nobody", and only the first is true here.
        ctx.stage_coverage["_curated_depth_chart"] = (
            f"absent: src/depth_chart/starters_{target_season}.csv does not exist. "
            f"Curated membership, roles, formation roles, reviewed usage priors and "
            f"replacement-level rows are therefore unmeasurable on this fold.")
    return ctx


def compose_board(rows, ctx):
    """Composition/allocation, from concatenated player rows to finished board.

    ``rows`` must already contain every player row that competes for team
    volume — veterans, rookies and any replacement-level rows — with team
    anchors attached, because several stages allocate a fixed team budget and a
    missing row silently hands its share to whoever is present.

    The stage order is load-bearing and is the single source of truth for it.
    Notably: the room reordering (``apply_usage_share_prior``) runs BEFORE the
    hierarchical mix so it settles who gets volume while the reconcilers settle
    how much there is; the L2/L3 mix runs BEFORE the volume normalizers so those
    can preserve group budgets; and ``reconcile_stat_constraints`` runs TWICE,
    once after the volume normalizers and once as the final numeric stage,
    because every stage that rescales a child stat independently of its parent
    can reintroduce a completion without an attempt — including the last one.
    """
    out = apply_deep_bench_games_cap(rows)
    out = apply_status_overrides(out, ctx.status_overrides)
    out = propagate_team_anchors(out)
    out = reconcile_qb_projected_volume_games(out, season_games=ctx.season_games)
    # Reorder each room toward what its depth ranks imply, before the anchors
    # bind - the blend preserves group totals.
    out = apply_usage_share_prior(out, ctx.usage_share_priors, ctx.depth_chart)
    # L2 team WR/TE/RB mix, then L3 within-group composition against team pass
    # attempts. Must run before the pass-volume reconcilers.
    out = attach_team_pass_mix(out, ctx.pass_mix_profiles, ctx.target_season)
    out = apply_hierarchical_pass_distribution(out, season_games=ctx.season_games)
    # L2 rush mix (RB/QB/OTHER) then L3 within-group composition against team
    # carries, before rush reconcile preserves named coverage residuals.
    out = attach_team_rush_mix(out, ctx.rush_mix_profiles, ctx.target_season)
    out = apply_hierarchical_rush_distribution(out, season_games=ctx.season_games)
    out = normalize_team_passing_volume(out, season_games=ctx.season_games)
    out = normalize_team_rushing_volume(out, season_games=ctx.season_games)
    out = reconcile_stat_constraints(out)
    out = reconcile_team_pass_receive_counts(out, season_games=ctx.season_games)
    # Second, TRAILING call - not a redundant repeat. The first call runs after
    # the volume normalizers, which is where a completion-without-an-attempt is
    # introduced. But reconcile_team_pass_receive_counts is itself a numeric
    # stage: it rescales receptions and receiving TDs against the pass-side
    # totals with a (team, position) factor that receptions and targets do NOT
    # share, so it can put receptions back above targets after the guard has
    # already run. Without this call the last stage that changes a number is
    # unguarded and the identity holds everywhere except at the output boundary.
    # Idempotent by construction (it only ever caps a child at its parent), so
    # running it twice cannot move a row the first call already settled.
    out = reconcile_stat_constraints(out)
    out = add_team_pass_catch_coherence_flag(out, ctx.depth_chart)
    return add_projected_season_totals(out)
