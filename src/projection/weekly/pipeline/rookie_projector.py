"""Rookie projection track and blend with veteran projections."""
# Ported from fantasy-projections-2 (team-first weekly v2). See docs/WEEKLY_V2_PORT_PROVENANCE.md.

from __future__ import annotations

import logging

import polars as pl

from src.projection.weekly.config.scoring import ScoringConfig
from src.projection.weekly.features.injuries import apply_injury_haircut
from src.projection.weekly.models.rookie import predict_rookie_fp_pg
from src.projection.weekly.pipeline.veteran_projector import project_veterans_week

logger = logging.getLogger(__name__)


def blend_weight(games_played_prior: float, *, full_blend_games: float = 4.0) -> float:
    """Weight on veteran model; 0 for no NFL sample, 1 after full_blend_games."""
    return float(min(1.0, max(0.0, games_played_prior) / full_blend_games))


def capped_rookie_prior(
    vet_fp: float,
    rook_fp: float,
    *,
    max_lift: float = 7.0,
) -> float:
    """Bound how far a zero-NFL-sample prior may outrun the volume track.

    ``pred_rookie_fp_pg`` is trained on fantasy points for rookies who played.
    When the veteran/volume track still sees committee crumbs (common for
    depth-chart RB1/WR1 rookies before shares are seeded), using the prior at
    face value creates elite fantasy lines with almost no carries/targets.
    Allow a real draft-capital lift, but keep the board volume-consistent.
    """
    vet = float(vet_fp)
    rook = float(rook_fp)
    if not (rook > vet):
        return rook
    return vet + min(rook - vet, float(max_lift))


def project_week_with_rookies(
    panel: pl.DataFrame,
    *,
    season: int,
    week: int,
    scoring: ScoringConfig | None = None,
    train_seasons: list[int] | None = None,
    team_totals_model=None,
    volume_models: dict | None = None,
    efficiency_models: dict | None = None,
    rookie_models: dict | None = None,
) -> pl.DataFrame:
    """Project a week, blending rookie priors for first-year players."""
    scoring = scoring or ScoringConfig()
    vet = project_veterans_week(
        panel,
        season=season,
        week=week,
        scoring=scoring,
        train_seasons=train_seasons,
        team_totals_model=team_totals_model,
        volume_models=volume_models,
        efficiency_models=efficiency_models,
    )

    week_panel = panel.filter((pl.col("season") == season) & (pl.col("week") == week))
    rookies = week_panel.filter(pl.col("is_rookie") == 1)
    if rookies.is_empty() or "gsis_id" not in vet.columns:
        return vet.with_columns(
            [
                pl.lit(1.0).alias("veteran_weight"),
                pl.col("fantasy_points").alias("fantasy_points_veteran"),
            ]
        )

    # Predict rookie per-game prior
    rook_pred = predict_rookie_fp_pg(rookies, models=rookie_models)
    rook_cols = rook_pred.select(
        [
            c
            for c in ("gsis_id", "pred_rookie_fp_pg", "games_played_prior")
            if c in rook_pred.columns
        ]
    ).unique(subset=["gsis_id"], keep="first")
    if "games_played_prior" not in rook_cols.columns:
        # pull from panel
        gp = week_panel.select(["gsis_id", "games_played_prior"]) if "games_played_prior" in week_panel.columns else None
        if gp is not None:
            rook_cols = rook_cols.join(gp, on="gsis_id", how="left")
        else:
            rook_cols = rook_cols.with_columns(pl.lit(0.0).alias("games_played_prior"))

    # Avoid suffix-dependent behavior when the veteran projector carries a
    # feature with the same name, and guarantee a malformed duplicate prospect
    # row cannot multiply the projection output.
    if "pred_rookie_fp_pg" in vet.columns:
        vet = vet.drop("pred_rookie_fp_pg")
    if "games_played_prior" in vet.columns and "games_played_prior" in rook_cols.columns:
        rook_cols = rook_cols.rename({"games_played_prior": "_rookie_games_played_prior"})
    merged = vet.join(rook_cols, on="gsis_id", how="left")
    if "_rookie_games_played_prior" in merged.columns:
        merged = merged.with_columns(
            pl.coalesce(
                [pl.col("_rookie_games_played_prior"), pl.col("games_played_prior")]
            ).alias("games_played_prior")
        ).drop("_rookie_games_played_prior")
    # Carry depth from week panel for damping deep-bench rookie priors
    if "depth_rank" in week_panel.columns and "depth_rank" not in merged.columns:
        merged = merged.join(
            week_panel.select(["gsis_id", "depth_rank"]).unique(subset=["gsis_id"]),
            on="gsis_id",
            how="left",
        )

    from src.projection.weekly.pipeline.accounting import rookie_role_confidence

    weights = []
    blended = []
    depth_ranks = (
        merged["depth_rank"].to_list() if "depth_rank" in merged.columns else [None] * merged.height
    )
    positions = (
        merged["position"].to_list() if "position" in merged.columns else [None] * merged.height
    )
    for is_rook, vet_fp, rook_fp, gp, depth, pos in zip(
        merged["is_rookie"].to_list() if "is_rookie" in merged.columns else [0] * merged.height,
        merged["fantasy_points"].to_list(),
        merged["pred_rookie_fp_pg"].to_list() if "pred_rookie_fp_pg" in merged.columns else [None] * merged.height,
        merged["games_played_prior"].to_list() if "games_played_prior" in merged.columns else [0] * merged.height,
        depth_ranks,
        positions,
        strict=False,
    ):
        if not is_rook or rook_fp is None:
            weights.append(1.0)
            blended.append(float(vet_fp))
            continue
        # Two independent reasons to trust the veteran track over the rookie
        # prior: NFL games already on tape, and holding a real depth-chart
        # role.  A buried rookie falls back to the depth-aware, accounting-
        # normalized veteran projection rather than a scaled-down starter one.
        w = blend_weight(float(gp or 0.0))
        role = rookie_role_confidence(depth, position=pos)
        vet_w = w + (1.0 - w) * (1.0 - role)
        rook_eff = capped_rookie_prior(float(vet_fp), float(rook_fp))
        fp = vet_w * float(vet_fp) + (1.0 - vet_w) * rook_eff
        weights.append(vet_w)
        blended.append(fp)

    out = merged.with_columns(
        [
            pl.Series("veteran_weight", weights),
            pl.col("fantasy_points").alias("fantasy_points_veteran"),
            pl.Series("fantasy_points", blended),
        ]
    )
    # Recompute floor/ceiling around blended mean if present
    if "floor" in out.columns and "ceiling" in out.columns:
        out = out.with_columns(
            [
                (pl.col("fantasy_points") - (pl.col("fantasy_points_veteran") - pl.col("floor")))
                .clip(lower_bound=0)
                .alias("floor"),
                (pl.col("fantasy_points") + (pl.col("ceiling") - pl.col("fantasy_points_veteran")))
                .clip(lower_bound=0)
                .alias("ceiling"),
            ]
        )

    # Re-apply after blend so rookie prior cannot resurrect Out players
    if "play_prob" not in out.columns or "is_out" not in out.columns:
        inj_cols = [
            c
            for c in ("gsis_id", "is_out", "is_doubtful", "is_questionable", "play_prob", "injury_status")
            if c in week_panel.columns
        ]
        if "gsis_id" in inj_cols and len(inj_cols) > 1:
            out = out.join(week_panel.select(inj_cols).unique(subset=["gsis_id"]), on="gsis_id", how="left")
    out = apply_injury_haircut(out, mode="stats")

    logger.info(
        "Blended %d rookies for %s week %s",
        rookies.height,
        season,
        week,
    )
    return out.sort(["position", "fantasy_points"], descending=[False, True])
