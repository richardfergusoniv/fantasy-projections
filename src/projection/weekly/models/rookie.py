"""Rookie translation model: draft capital + college + landing spot."""
# Ported from fantasy-projections-2 (team-first weekly v2). See docs/WEEKLY_V2_PORT_PROVENANCE.md.

from __future__ import annotations

import logging

import numpy as np
import polars as pl

from src.projection.weekly.models.base import MultiTargetModel, available_features, dataframe_to_matrix
from src.projection.weekly.models.registry import load_model, save_model

logger = logging.getLogger(__name__)

# Approximate Chase Stuart draft value chart (relative value by overall pick)
def stuart_draft_value(pick: float | None) -> float:
    if pick is None or (isinstance(pick, float) and np.isnan(pick)):
        return 0.0
    p = float(pick)
    if p <= 0:
        return 0.0
    # Smooth hyperbolic approximation of Stuart chart
    return 1000.0 / (p + 10.0)


COLLEGE_FEATURES = [
    "college_ppa_all",
    "college_ppa_last_season",
    "college_rec_yards",
    "college_rush_yards",
    "college_recruiting_stars",
    "college_recruiting_rating",
    "college_dominator",
    "college_rec_dominator",
    "college_rec_yard_share",
    "college_rec_td_share",
    "college_rush_yard_share",
    "college_rush_td_share",
    "college_rec_yards_per_game",
    "college_rush_yards_per_game",
    "rec_yards_final",
    "rush_yards_final",
    "college_rec_yards_per_game_final",
    "college_rush_yards_per_game_final",
    "college_rec_yards_career",
    "college_rush_yards_career",
    "college_dominator_career_weighted",
    "college_rec_dominator_career_weighted",
    "college_rec_yard_share_career_weighted",
    "college_rush_yard_share_career_weighted",
    "college_team_srs_career_weighted",
    "college_seasons_played",
    "college_breakout_season",
    "college_breakout_age",
]

ROOKIE_FEATURES = [
    "draft_pick",
    "draft_round",
    "draft_value",
    "age",
    "is_early_declare",
    "combine_forty",
    "combine_height",
    "combine_weight",
    "forty_missing",
    "team_pass_rate_prior_season",
    "vacated_target_share",
    "vacated_carry_share",
    "implied_team_total",
    *COLLEGE_FEATURES,
]


def _normalized_name_expr(column: str) -> pl.Expr:
    """A conservative player-name key used only when a stable ID is absent."""
    return (
        pl.col(column)
        .cast(pl.Utf8)
        .str.to_lowercase()
        .str.replace_all(r"[^a-z0-9 ]", " ")
        .str.replace_all(r"\b(jr|sr|ii|iii|iv)\b", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def build_vacated_opportunity(panel: pl.DataFrame) -> pl.DataFrame:
    """Return prior-year opportunity lost by each team in the following season.

    Shares are averaged to one row per player-season before summing.  A player
    counts as departing when they have no row for the same club next season.
    This avoids the old implementation summing weekly shares (which nearly
    always clipped both features to zero vacated opportunity).
    """
    required = {"gsis_id", "season", "team"}
    if panel.is_empty() or not required.issubset(panel.columns):
        return pl.DataFrame(
            schema={
                "season": pl.Int64,
                "team": pl.Utf8,
                "vacated_target_share": pl.Float64,
                "vacated_carry_share": pl.Float64,
            }
        )

    share_cols = [c for c in ("target_share", "carry_share") if c in panel.columns]
    if not share_cols:
        return pl.DataFrame(
            schema={
                "season": pl.Int64,
                "team": pl.Utf8,
                "vacated_target_share": pl.Float64,
                "vacated_carry_share": pl.Float64,
            }
        )

    identities = panel.select(["gsis_id", "season", "team"]).drop_nulls(
        ["gsis_id", "season", "team"]
    ).unique(subset=["gsis_id", "season", "team"])
    current = identities.with_columns(pl.lit(1).alias("_returned"))
    usage = panel.group_by(["gsis_id", "season", "team"]).agg(
        [pl.col(c).mean().fill_null(0.0).alias(c) for c in share_cols]
    )
    prior = usage.with_columns((pl.col("season") + 1).alias("season")).join(
        current, on=["gsis_id", "season", "team"], how="left"
    )
    departed = prior.filter(pl.col("_returned").is_null())
    aggs = []
    if "target_share" in share_cols:
        aggs.append(pl.col("target_share").sum().clip(0.0, 1.0).alias("vacated_target_share"))
    else:
        aggs.append(pl.lit(0.0).alias("vacated_target_share"))
    if "carry_share" in share_cols:
        aggs.append(pl.col("carry_share").sum().clip(0.0, 1.0).alias("vacated_carry_share"))
    else:
        aggs.append(pl.lit(0.0).alias("vacated_carry_share"))
    return departed.group_by(["season", "team"]).agg(aggs)


def build_target_vacated_opportunity(
    panel: pl.DataFrame,
    roster: pl.DataFrame,
    *,
    target_season: int,
    feature_season: int | None = None,
) -> pl.DataFrame:
    """Vacated shares for a future season using its actual roster identity."""
    feature_season = feature_season or target_season - 1
    required = {"gsis_id", "season", "team"}
    if panel.is_empty() or roster.is_empty() or not required.issubset(panel.columns) or "gsis_id" not in roster.columns:
        return pl.DataFrame(schema={"team": pl.Utf8, "vacated_target_share": pl.Float64, "vacated_carry_share": pl.Float64})
    share_cols = [c for c in ("target_share", "carry_share") if c in panel.columns]
    prior = panel.filter(pl.col("season") == feature_season).group_by(
        ["gsis_id", "team"]
    ).agg([pl.col(c).mean().fill_null(0.0).alias(c) for c in share_cols])
    roster_keys = (
        roster.select(["gsis_id", "team"])
        .drop_nulls(["gsis_id", "team"])
        .unique()
        .with_columns(pl.lit(1).alias("_returned"))
    )
    departed = prior.join(roster_keys, on=["gsis_id", "team"], how="left").filter(
        pl.col("_returned").is_null()
    )
    aggs = [
        (pl.col(c).sum().clip(0.0, 1.0).alias(f"vacated_{c}"))
        for c in share_cols
    ]
    out = departed.group_by("team").agg(aggs) if aggs else departed.select("team").unique()
    for col in ("vacated_target_share", "vacated_carry_share"):
        if col not in out.columns:
            out = out.with_columns(pl.lit(0.0).alias(col))
    return out.select(["team", "vacated_target_share", "vacated_carry_share"])


def attach_target_rookie_features(
    players: pl.DataFrame,
    *,
    target_season: int,
    draft: pl.DataFrame | None = None,
    combine: pl.DataFrame | None = None,
    college: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Attach target-class identity and prospect data without template leakage.

    Stable ``gsis_id`` is preferred. Name fallback is restricted to the target
    draft class and deduplicated, so a join cannot multiply player rows.
    """
    if players.is_empty():
        return players
    out = players
    name_col = next((c for c in ("player_name", "full_name", "display_name") if c in out.columns), None)
    if name_col:
        out = out.with_columns(_normalized_name_expr(name_col).alias("_rookie_name"))

    # Roster identity is authoritative even when nflverse has not published a
    # gsis-linked draft row yet.
    identity_parts = []
    if "rookie_year" in out.columns:
        identity_parts.append(pl.col("rookie_year").cast(pl.Int64, strict=False) == target_season)
    if "entry_year" in out.columns:
        identity_parts.append(pl.col("entry_year").cast(pl.Int64, strict=False) == target_season)

    d = draft if draft is not None else pl.DataFrame()
    if not d.is_empty() and "season" in d.columns:
        d = d.filter(pl.col("season").cast(pl.Int64, strict=False) == target_season)
        rename = {}
        if "pick" in d.columns and "draft_pick" not in d.columns:
            rename["pick"] = "draft_pick"
        if "round" in d.columns and "draft_round" not in d.columns:
            rename["round"] = "draft_round"
        d = d.rename(rename) if rename else d
        d_name = next((c for c in ("player_name", "pfr_player_name", "full_name") if c in d.columns), None)
        select_cols = [c for c in ("gsis_id", "draft_pick", "draft_round", "position") if c in d.columns]
        if d_name:
            d = d.with_columns(_normalized_name_expr(d_name).alias("_rookie_name"))
            select_cols.append("_rookie_name")
        d = d.select(list(dict.fromkeys(select_cols)))
        if "gsis_id" in out.columns and "gsis_id" in d.columns:
            by_id = d.filter(pl.col("gsis_id").is_not_null()).unique(subset=["gsis_id"], keep="first")
            attrs = [c for c in ("draft_pick", "draft_round") if c in by_id.columns]
            if attrs:
                out = out.join(
                    by_id.select(["gsis_id"] + attrs).rename({c: f"_target_{c}" for c in attrs}),
                    on="gsis_id", how="left",
                )
                for c in attrs:
                    base = pl.col(c) if c in out.columns else pl.lit(None)
                    out = out.with_columns(pl.coalesce([pl.col(f"_target_{c}"), base]).alias(c)).drop(f"_target_{c}")
            identity_parts.append(pl.col("gsis_id").is_in(by_id["gsis_id"].to_list()))
        if name_col and "_rookie_name" in d.columns:
            by_name = d.filter(pl.col("_rookie_name") != "").unique(subset=["_rookie_name"], keep="first")
            attrs = [c for c in ("draft_pick", "draft_round") if c in by_name.columns]
            if attrs:
                named = by_name.select(["_rookie_name"] + attrs).rename({c: f"_name_{c}" for c in attrs})
                out = out.join(named, on="_rookie_name", how="left")
                for c in attrs:
                    base = pl.col(c) if c in out.columns else pl.lit(None)
                    out = out.with_columns(pl.coalesce([base, pl.col(f"_name_{c}")]).alias(c)).drop(f"_name_{c}")
            identity_parts.append(
                pl.col("_rookie_name").is_in(by_name["_rookie_name"].to_list())
            )

    rookie_expr = identity_parts[0] if identity_parts else pl.lit(False)
    for expr in identity_parts[1:]:
        rookie_expr = rookie_expr | expr
    out = out.with_columns(
        [
            rookie_expr.fill_null(False).cast(pl.Int8).alias("is_rookie"),
            pl.when(rookie_expr.fill_null(False))
            .then(pl.lit(target_season))
            .otherwise(pl.col("rookie_season") if "rookie_season" in out.columns else pl.lit(None))
            .cast(pl.Int64)
            .alias("rookie_season"),
        ]
    )

    # Combine records must match the target class, not an arbitrary first row.
    c = combine if combine is not None else pl.DataFrame()
    if not c.is_empty():
        combine_year = next((x for x in ("season", "draft_year") if x in c.columns), None)
        if combine_year:
            c = c.filter(pl.col(combine_year).cast(pl.Int64, strict=False) == target_season)
        ren = {}
        for src, dst in (("forty", "combine_forty"), ("forty_yard", "combine_forty"), ("ht", "combine_height"), ("height", "combine_height"), ("wt", "combine_weight"), ("weight", "combine_weight")):
            if src in c.columns and dst not in c.columns:
                ren[src] = dst
        c = c.rename(ren) if ren else c
        attrs = [x for x in ("combine_forty", "combine_height", "combine_weight") if x in c.columns]
        if attrs and "gsis_id" in out.columns and "gsis_id" in c.columns:
            by_id = c.select(["gsis_id"] + attrs).filter(pl.col("gsis_id").is_not_null()).unique(subset=["gsis_id"], keep="first")
            out = out.join(by_id, on="gsis_id", how="left", suffix="_target_combine")
        combine_name = next((x for x in ("player_name", "full_name", "pfr_player_name") if x in c.columns), None)
        if attrs and name_col and combine_name:
            by_name = (
                c.with_columns(_normalized_name_expr(combine_name).alias("_rookie_name"))
                .select(["_rookie_name"] + attrs)
                .filter(pl.col("_rookie_name") != "")
                .unique(subset=["_rookie_name"], keep="first")
            )
            # Prefer an ID attachment when present, otherwise fill from name.
            named = by_name.rename({x: f"_name_{x}" for x in attrs})
            out = out.join(named, on="_rookie_name", how="left")
            for attr in attrs:
                base = pl.col(attr) if attr in out.columns else pl.lit(None)
                out = out.with_columns(pl.coalesce([base, pl.col(f"_name_{attr}")]).alias(attr)).drop(f"_name_{attr}")

    # College attachment is an as-of join: only seasons before the draft year.
    if college is not None and not college.is_empty() and name_col and "college_player" in college.columns:
        c2 = college.with_columns(_normalized_name_expr("college_player").alias("_rookie_name"))
        if "college_season" in c2.columns:
            c2 = c2.filter(pl.col("college_season") < target_season).sort("college_season").unique(subset=["_rookie_name"], keep="last")
        else:
            c2 = c2.unique(subset=["_rookie_name"], keep="last")
        mapping = {"ppa_all": "college_ppa_all", "rec_yards": "college_rec_yards", "rush_yards": "college_rush_yards"}
        c2 = c2.rename({k: v for k, v in mapping.items() if k in c2.columns and v not in c2.columns})
        attrs = [x for x in COLLEGE_FEATURES if x in c2.columns]
        if attrs:
            out = out.join(c2.select(["_rookie_name"] + attrs), on="_rookie_name", how="left", suffix="_target_college")
            if "college_ppa_all" in out.columns and "college_ppa_last_season" not in out.columns:
                out = out.with_columns(pl.col("college_ppa_all").alias("college_ppa_last_season"))

    return out.drop("_rookie_name") if "_rookie_name" in out.columns else out


def _add_draft_value(df: pl.DataFrame) -> pl.DataFrame:
    if "draft_pick" not in df.columns:
        return df.with_columns(pl.lit(0.0).alias("draft_value"))
    picks = df["draft_pick"].to_list()
    values = [stuart_draft_value(p) for p in picks]
    return df.with_columns(pl.Series("draft_value", values))


def build_rookie_training_frame(
    panel: pl.DataFrame,
    *,
    college: pl.DataFrame | None = None,
    combine: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """One row per rookie-season with season-total fantasy points as label.

    Weekly rows are aggregated; features use preseason / landing-spot signals.
    """
    rookies = panel.filter(pl.col("is_rookie") == 1)
    if rookies.is_empty():
        return rookies

    season_fp = (
        rookies.group_by(["gsis_id", "season", "position", "team", "player_name"])
        .agg(
            [
                pl.col("fantasy_points").sum().alias("rookie_season_fp"),
                pl.col("fantasy_points").mean().alias("rookie_fp_pg"),
                pl.col("week").n_unique().alias("games_played"),
                pl.col("draft_pick").first(),
                pl.col("draft_round").first(),
                pl.col("age").mean().alias("age"),
                pl.col("team_pass_rate_prior_season").first(),
                pl.col("implied_team_total").mean(),
            ]
        )
    )
    season_fp = _add_draft_value(season_fp)

    # Early declare proxy: age <= 21.5 at draft
    season_fp = season_fp.with_columns(
        (pl.col("age") <= 21.5).fill_null(False).cast(pl.Int8).alias("is_early_declare")
    )

    # Prior-season opportunity belonging to players absent from this club in
    # the rookie's target season.
    vacated = build_vacated_opportunity(panel)
    season_fp = season_fp.join(vacated, on=["season", "team"], how="left")

    # Combine
    if combine is not None and not combine.is_empty():
        c = combine
        # Best-effort column mapping
        rename = {}
        for src, dst in (
            ("forty", "combine_forty"),
            ("forty_yard", "combine_forty"),
            ("ht", "combine_height"),
            ("height", "combine_height"),
            ("wt", "combine_weight"),
            ("weight", "combine_weight"),
            ("season", "combine_season"),
        ):
            if src in c.columns and dst not in c.columns:
                rename[src] = dst
        c = c.rename(rename) if rename else c
        id_col = "gsis_id" if "gsis_id" in c.columns else None
        attrs = [x for x in ("combine_forty", "combine_height", "combine_weight") if x in c.columns]
        if id_col and attrs:
            if "combine_season" in c.columns:
                c = c.select([id_col, "combine_season"] + attrs).unique(
                    subset=[id_col, "combine_season"], keep="first"
                )
                season_fp = season_fp.join(
                    c, left_on=["gsis_id", "season"], right_on=[id_col, "combine_season"], how="left"
                )
            else:
                c = c.select([id_col] + attrs).unique(subset=[id_col], keep="first")
                season_fp = season_fp.join(c, left_on="gsis_id", right_on=id_col, how="left")
        elif attrs and "player_name" in c.columns:
            year_col = next((x for x in ("combine_season", "draft_year") if x in c.columns), None)
            c = c.with_columns(_normalized_name_expr("player_name").alias("_join_name"))
            season_fp = season_fp.with_columns(_normalized_name_expr("player_name").alias("_join_name"))
            if year_col:
                c = c.select(["_join_name", year_col] + attrs).unique(
                    subset=["_join_name", year_col], keep="first"
                )
                season_fp = season_fp.join(
                    c,
                    left_on=["_join_name", "season"],
                    right_on=["_join_name", year_col],
                    how="left",
                )
            else:
                c = c.select(["_join_name"] + attrs).unique(subset=["_join_name"], keep="first")
                season_fp = season_fp.join(c, on="_join_name", how="left")
            season_fp = season_fp.drop("_join_name")

    for col, default in (
        ("combine_forty", None),
        ("combine_height", None),
        ("combine_weight", None),
    ):
        if col not in season_fp.columns:
            season_fp = season_fp.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    season_fp = season_fp.with_columns(
        pl.col("combine_forty").is_null().cast(pl.Int8).alias("forty_missing")
    )

    # College features — join by normalized name and the latest season strictly
    # before the rookie's draft year.
    if college is not None and not college.is_empty():
        name_key = "college_player" if "college_player" in college.columns else None
        if name_key and "player_name" in season_fp.columns:
            c2 = college.with_columns(
                _normalized_name_expr(name_key).alias("_join_name")
            )
            pick = ["_join_name"] + (["college_season"] if "college_season" in c2.columns else [])
            for c in (
                "ppa_all",
                "rec_yards",
                "rush_yards",
                *COLLEGE_FEATURES,
            ):
                if c in c2.columns:
                    pick.append(c)
            last = c2.select(list(dict.fromkeys(pick)))
            rename = {}
            if "ppa_all" in last.columns:
                rename["ppa_all"] = "college_ppa_all"
            if "rec_yards" in last.columns:
                rename["rec_yards"] = "college_rec_yards"
            if "rush_yards" in last.columns:
                rename["rush_yards"] = "college_rush_yards"
            last = last.rename(rename) if rename else last
            if "college_ppa_all" in last.columns:
                last = last.with_columns(pl.col("college_ppa_all").alias("college_ppa_last_season"))
            season_fp = season_fp.with_columns(
                [
                    _normalized_name_expr("player_name").alias("_join_name"),
                    (pl.col("season") - 1).cast(pl.Int64).alias("_college_cutoff"),
                ]
            )
            if "college_season" in last.columns:
                # Match the latest college year known before each draft; never
                # attach a future prospect season to a historical rookie.
                season_fp = season_fp.sort(["_join_name", "_college_cutoff"]).join_asof(
                    last.sort(["_join_name", "college_season"]),
                    left_on="_college_cutoff",
                    right_on="college_season",
                    by="_join_name",
                    strategy="backward",
                    check_sortedness=False,
                )
            else:
                season_fp = season_fp.join(
                    last.unique(subset=["_join_name"], keep="last"), on="_join_name", how="left"
                )
            season_fp = season_fp.drop([c for c in ("_join_name", "_college_cutoff", "college_season") if c in season_fp.columns])

    for col in COLLEGE_FEATURES:
        if col not in season_fp.columns:
            season_fp = season_fp.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    return season_fp


def train_rookie_model(
    panel: pl.DataFrame,
    *,
    train_seasons: list[int],
    college: pl.DataFrame | None = None,
    combine: pl.DataFrame | None = None,
    model_type: str = "ridge",
    persist: bool = True,
) -> dict[str, MultiTargetModel]:
    frame = build_rookie_training_frame(panel, college=college, combine=combine)
    frame = frame.filter(pl.col("season").is_in(train_seasons))
    models: dict[str, MultiTargetModel] = {}

    for pos in ("QB", "RB", "WR", "TE"):
        pos_df = frame.filter(pl.col("position") == pos)
        if pos_df.height < 20:
            logger.warning("Not enough rookies for %s (%d); skipping", pos, pos_df.height)
            continue
        features = available_features(pos_df, ROOKIE_FEATURES)
        targets = ["rookie_fp_pg"]
        X = dataframe_to_matrix(pos_df, features)
        y = pos_df.select(targets)
        model = MultiTargetModel(targets=targets, feature_cols=features)
        model.fit(X, y, model_type=model_type)
        if persist:
            save_model(
                f"rookie_{pos}",
                model,
                meta={
                    "position": pos,
                    "features": features,
                    "targets": targets,
                    "n": pos_df.height,
                    "train_seasons": train_seasons,
                },
            )
        models[pos] = model
        logger.info("Trained rookie_%s on %d rookies", pos, pos_df.height)

    return models


def predict_rookie_fp_pg(
    rookies: pl.DataFrame,
    models: dict[str, MultiTargetModel] | None = None,
) -> pl.DataFrame:
    """Predict per-game fantasy points for rookies from draft/college features."""
    if rookies.is_empty():
        return rookies.with_columns(pl.lit(None).cast(pl.Float64).alias("pred_rookie_fp_pg"))

    out = _add_draft_value(rookies)
    if "is_early_declare" not in out.columns:
        out = out.with_columns(
            (pl.col("age") <= 21.5).fill_null(False).cast(pl.Int8).alias("is_early_declare")
            if "age" in out.columns
            else pl.lit(0).alias("is_early_declare")
        )
    for col in ROOKIE_FEATURES:
        if col not in out.columns:
            out = out.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    frames = []
    for pos in out["position"].unique().to_list():
        sub = out.filter(pl.col("position") == pos)
        try:
            model = (models or {}).get(pos) or load_model(f"rookie_{pos}")
        except FileNotFoundError:
            # Fallback: draft-value heuristic
            sub = sub.with_columns(
                (pl.col("draft_value") / 50.0).clip(0, 20).alias("pred_rookie_fp_pg")
            )
            frames.append(sub)
            continue
        frames.append(model.predict_frame(sub, prefix="pred_"))
    return pl.concat(frames, how="diagonal_relaxed")
