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
    # Team-volume ablation toggles. None means "use the shipped contracts".
    # These exist so the QB 0.941/0.942 -> 1.000 change and the detaching of
    # TDs from volume scaling can each be measured on their own, rather than
    # inferred from a board where both already moved together.
    team_volume_shares: dict | None = None
    team_volume_siblings: dict | None = None
    reconcile_alpha: float | None = None
    # Experimental joint QB-room allocation (starter share first, backups
    # residual on expected missed games). Shipped default False keeps the
    # existing reconcile_team_volume path bit-identical.
    qb_joint_room_allocation: bool = False
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


# Shared stage names for production composition and shadow diagnostics.
COMPOSE_CHECKPOINT_NAMES = (
    "raw_forecast",
    "exposure_status_baseline",
    "team_volume_reconcile",
    "concentration",
    "td_constraints",
    "counting_stat_constraints",
    "season_total_finalization",
)

# Legacy aliases kept for QB ablation scripts that keyed off earlier names.
COMPOSE_CHECKPOINT_ALIASES = {
    "raw_model": "exposure_status_baseline",
    "post_team_volume_reconcile": "team_volume_reconcile",
    "post_concentration": "concentration",
    "post_td_clip": "td_constraints",
    "final_shipped": "season_total_finalization",
}


def _score_checkpoint_fantasy(frame: pd.DataFrame) -> dict:
    """Score fantasy PPG (and season points when present) for every position."""
    from src.projection.fantasy_points import SCORING

    if frame.empty or "stat" not in frame.columns or "pred_pg" not in frame.columns:
        return {}
    work = frame.copy()
    work["player_id"] = work["player_id"].astype(str)
    wide = work.pivot_table(
        index="player_id",
        columns="stat",
        values="pred_pg",
        aggfunc="first",
    )
    score = pd.Series(0.0, index=wide.index, dtype=float)
    for stat, weight in SCORING.items():
        if stat in wide.columns:
            score = score + pd.to_numeric(wide[stat], errors="coerce").fillna(0.0) * weight

    season_score = None
    if "pred_season" in work.columns:
        season_wide = work.pivot_table(
            index="player_id",
            columns="stat",
            values="pred_season",
            aggfunc="first",
        )
        season_score = pd.Series(0.0, index=season_wide.index, dtype=float)
        for stat, weight in SCORING.items():
            if stat in season_wide.columns:
                season_score = season_score + (
                    pd.to_numeric(season_wide[stat], errors="coerce").fillna(0.0) * weight
                )

    meta = work.drop_duplicates("player_id").set_index("player_id")
    names = meta["display_name"] if "display_name" in meta.columns else pd.Series(dtype=str)
    teams = meta["team"] if "team" in meta.columns else pd.Series(dtype=str)
    positions = meta["position"] if "position" in meta.columns else pd.Series(dtype=str)
    games = (
        pd.to_numeric(meta["projected_games"], errors="coerce")
        if "projected_games" in meta.columns
        else pd.Series(dtype=float)
    )
    out = {}
    for pid, ppg in score.items():
        row = {
            "fantasy_ppg": round(float(ppg), 6),
            "display_name": str(names.get(pid, pid)),
            "team": str(teams.get(pid, "")),
            "position": str(positions.get(pid, "")),
        }
        if pid in games.index and pd.notna(games.get(pid)):
            row["projected_games"] = float(games.get(pid))
        if season_score is not None and pid in season_score.index:
            row["fantasy_pts_season"] = round(float(season_score.loc[pid]), 6)
        out[str(pid)] = row
    return out


def run_compose_stages(rows, ctx, *, capture_checkpoints: bool = False):
    """Run the shipped composition sequence; optionally capture long-form boards.

    Production and diagnostics share this runner so stage attribution cannot
    drift from ``compose_board``. When ``capture_checkpoints`` is False the
    path matches the historical in-place stage chain (no intermediate copies).
    """
    checkpoints: dict[str, pd.DataFrame] = {}

    def _capture(name: str, frame: pd.DataFrame) -> None:
        if capture_checkpoints:
            checkpoints[name] = frame.copy(deep=True)

    _capture("raw_forecast", rows)

    out = apply_full_season_games_baseline(
        rows,
        season_games=ctx.season_games,
        blend_alpha=ctx.exposure_blend_alpha,
    )
    out = apply_status_overrides(out, ctx.status_overrides)
    out = propagate_team_anchors(out)
    out["projected_volume_games"] = pd.to_numeric(out.get("projected_games"), errors="coerce")
    _capture("exposure_status_baseline", out)

    # Top-down: pull each team's summed volume toward its own anchor before
    # the counting-stat identities and the season totals are materialised.
    if ctx.qb_joint_room_allocation:
        from src.projection.qb_joint_allocation import reconcile_qb_joint_room

        # Non-QB positions keep the shipped reconciler; QB uses joint room.
        out = reconcile_team_volume(
            out,
            alpha=ctx.reconcile_alpha,
            volume_shares=ctx.team_volume_shares,
            volume_siblings=ctx.team_volume_siblings,
        )
        qb = out["position"].astype(str).eq("QB")
        if qb.any():
            scale = pd.to_numeric(out.loc[qb, "team_volume_scale"], errors="coerce").replace(0, pd.NA).fillna(1.0)
            for col in ("pred_pg", "pred_pg_low", "pred_pg_high"):
                if col in out.columns:
                    out.loc[qb, col] = pd.to_numeric(out.loc[qb, col], errors="coerce") / scale
            out.loc[qb, "team_volume_scale"] = 1.0
            out, _qb_report = reconcile_qb_joint_room(
                out,
                target_season=ctx.target_season,
                alpha=ctx.reconcile_alpha,
                volume_shares=ctx.team_volume_shares,
                volume_siblings=ctx.team_volume_siblings,
            )
            ctx.stage_coverage["qb_joint_room_allocation"] = "active"
    else:
        out = reconcile_team_volume(
            out,
            alpha=ctx.reconcile_alpha,
            volume_shares=ctx.team_volume_shares,
            volume_siblings=ctx.team_volume_siblings,
        )
    _capture("team_volume_reconcile", out)

    out = apply_concentration(out)
    _capture("concentration", out)

    out = reconcile_td_rate_constraints(out, rush_td_hi=ctx.qb_rush_td_clip_hi)
    if ctx.qb_pass_td_t1_lite:
        out = reconcile_pass_td_t1_lite(out)
    _capture("td_constraints", out)

    out = reconcile_stat_constraints(out)
    _capture("counting_stat_constraints", out)

    out = add_projected_season_totals(out)
    # Season totals use each player's own projected_games, so QB rooms and
    # receiver rooms can diverge even when rates were coherent. Restore the
    # hard pass/catch identities on season columns only (rates untouched).
    out = reconcile_team_season_identities(out)
    _capture("season_total_finalization", out)
    return out, checkpoints


def compose_board(rows, ctx):
    """Post-forecast hygiene from concatenated player rows to finished board.

    Draft exposure is a full season except IR / PUP / suspension overrides.
    ``projected_volume_games`` equals ``projected_games``.
    """
    final, _ = run_compose_stages(rows, ctx, capture_checkpoints=False)
    return final


def compose_board_stages(rows, ctx, *, return_boards: bool = False):
    """Return fantasy-relevant compose checkpoints for stage attribution.

    Scores every skill position. Legacy alias keys remain for older QB
    ablation scripts; prefer ``COMPOSE_CHECKPOINT_NAMES``.
    """
    final, boards = run_compose_stages(rows, ctx, capture_checkpoints=True)
    # Final board is the same object compose_board would return.
    del final
    scored = {
        name: _score_checkpoint_fantasy(boards[name])
        for name in COMPOSE_CHECKPOINT_NAMES
        if name in boards
    }
    for alias, canonical in COMPOSE_CHECKPOINT_ALIASES.items():
        if canonical in scored:
            scored[alias] = scored[canonical]
    if return_boards:
        scored["_boards"] = boards
    return scored


def checkpoint_boards_long(stage_payload: dict) -> dict[str, pd.DataFrame]:
    """Extract long-form checkpoint boards from ``compose_board_stages`` output."""
    boards = stage_payload.get("_boards") or {}
    return {name: boards[name] for name in COMPOSE_CHECKPOINT_NAMES if name in boards}
