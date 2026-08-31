"""Preseason QB context features for E2 retraining.

``QB_CONTEXT_FEATURES`` is a separate contract from ``ALL_FEATURES``,
``ROLE_FEATURES``, and availability features.  Target-season outcomes never
enter these columns.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.projection.data_prep import get_conn
from src.projection.depth_history import load_preseason_depth_chart

QB_CONTEXT_FEATURES: tuple[str, ...] = (
    "qb_changed",
    "qb_prior_epa_per_dropback",
    "qb_prior_cpoe",
    "qb_prior_dropback_sample",
    "qb_epa_change_vs_prev",
    "qb_cpoe_change_vs_prev",
    "qb_rookie_or_unknown",
    "qb_low_sample",
)

QB_CONTEXT_MANIFEST_KEY = "consumes_qb_context"
LOW_SAMPLE_DROPBACKS = 100
ROOKIE_UNKNOWN_SAMPLE = 0

AFFECTED_POSITIONS = frozenset({"RB", "WR", "TE"})
AFFECTED_TEAM_LABELS = frozenset({
    "team_passing_yards_pg",
    "team_pass_attempts_pg",
    "team_carries_pg",
    "team_rushing_yards_pg",
})


@dataclass(frozen=True)
class QbContextConfig:
    low_sample_dropbacks: int = LOW_SAMPLE_DROPBACKS
    shrink_k: float = 50.0


def _league_prior(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.mean()) if not clean.empty else 0.0


def _shrink(value: float | None, sample: float, prior: float, *, k: float) -> float:
    if value is None or pd.isna(value):
        return prior
    n = max(float(sample), 0.0)
    w = n / (n + k) if (n + k) > 0 else 0.0
    return w * float(value) + (1.0 - w) * prior


def _primary_passer_by_team(feat: pd.DataFrame, season: int) -> pd.DataFrame:
    qb = feat[(feat["season"].eq(season)) & (feat["position"].eq("QB"))].copy()
    if qb.empty or "attempts" not in qb.columns:
        return pd.DataFrame(columns=["team", "player_id", "attempts"])
    qb["attempts"] = pd.to_numeric(qb["attempts"], errors="coerce").fillna(0.0)
    idx = qb.groupby("team")["attempts"].idxmax()
    return qb.loc[idx, ["team", "player_id", "attempts"]].rename(
        columns={"player_id": "prev_primary_passer_id", "attempts": "prev_primary_attempts"}
    )


def _preseason_qb1(chart: pd.DataFrame) -> pd.DataFrame:
    if chart.empty:
        return pd.DataFrame(columns=["team", "preseason_qb1_id"])
    qb = chart[chart["position"].eq("QB")].copy()
    rank_col = "depth_rank" if "depth_rank" in qb.columns else None
    if rank_col:
        qb = qb[pd.to_numeric(qb[rank_col], errors="coerce").eq(1)]
    else:
        qb = qb.iloc[0:0]
    if qb.empty:
        return pd.DataFrame(columns=["team", "preseason_qb1_id"])
    qb = qb.sort_values(["team", rank_col or "team"])
    qb = qb.drop_duplicates("team", keep="first")
    id_col = "player_id" if "player_id" in qb.columns else "gsis_id"
    return qb[["team", id_col]].rename(columns={id_col: "preseason_qb1_id"})


def _qb_passing_metrics(conn, season: int) -> pd.DataFrame:
    """Season-level EPA/dropback and CPOE for quarterbacks through ``season``."""
    try:
        pbp = pd.read_sql(
            "SELECT passer_player_id AS player_id, epa FROM pbp "
            "WHERE season = ? AND pass_attempt = 1 AND passer_player_id IS NOT NULL",
            conn,
            params=(int(season),),
        )
    except Exception:
        pbp = pd.DataFrame(columns=["player_id", "epa"])
    if pbp.empty:
        epa = pd.DataFrame(columns=["player_id", "epa_per_dropback", "dropbacks"])
    else:
        grouped = pbp.groupby("player_id", as_index=False).agg(
            epa_per_dropback=("epa", "mean"),
            dropbacks=("epa", "count"),
        )
        epa = grouped

    try:
        ngs = pd.read_sql(
            "SELECT player_gsis_id AS player_id, completion_percentage_above_expectation AS cpoe, "
            "attempts FROM ngs_passing WHERE season = ?",
            conn,
            params=(int(season),),
        )
    except Exception:
        ngs = pd.DataFrame(columns=["player_id", "cpoe", "attempts"])
    if not ngs.empty:
        ngs["attempts"] = pd.to_numeric(ngs["attempts"], errors="coerce").fillna(0.0)
        ngs["cpoe"] = pd.to_numeric(ngs["cpoe"], errors="coerce")

        def _weighted_cpoe(g: pd.DataFrame) -> pd.Series:
            if g["cpoe"].notna().any():
                cpoe = float(np.average(g["cpoe"], weights=g["attempts"].clip(lower=1)))
            else:
                cpoe = np.nan
            return pd.Series({"cpoe": cpoe, "ngs_attempts": g["attempts"].sum()})

        ngs = ngs.groupby("player_id", as_index=False).apply(_weighted_cpoe)
    else:
        ngs = pd.DataFrame(columns=["player_id", "cpoe", "ngs_attempts"])

    if epa.empty and ngs.empty:
        return pd.DataFrame(columns=["player_id", "epa_per_dropback", "dropbacks", "cpoe"])
    out = epa.merge(ngs, on="player_id", how="outer")
    if "dropbacks" not in out.columns:
        out["dropbacks"] = out.get("ngs_attempts", 0)
    out["dropbacks"] = pd.to_numeric(out["dropbacks"], errors="coerce").fillna(
        pd.to_numeric(out.get("ngs_attempts"), errors="coerce")
    ).fillna(0.0)
    return out


def build_team_qb_context(
    conn,
    feature_table: pd.DataFrame,
    *,
    source_season: int,
    target_season: int,
    config: QbContextConfig | None = None,
) -> pd.DataFrame:
    """Build team-grain QB context using information through ``source_season`` only."""
    cfg = config or QbContextConfig()
    chart = load_preseason_depth_chart(int(target_season), conn=conn)
    qb1 = _preseason_qb1(chart)
    prev_primary = _primary_passer_by_team(feature_table, int(source_season))
    metrics = _qb_passing_metrics(conn, int(source_season))

    teams = pd.DataFrame({"team": sorted(set(qb1["team"]).union(set(prev_primary["team"])))})
    if teams.empty:
        teams = pd.DataFrame({"team": chart["team"].dropna().unique()}) if not chart.empty else teams
    frame = teams.merge(qb1, on="team", how="left").merge(prev_primary, on="team", how="left")
    frame["qb_changed"] = (
        frame["preseason_qb1_id"].notna()
        & frame["prev_primary_passer_id"].notna()
        & frame["preseason_qb1_id"].astype(str).ne(frame["prev_primary_passer_id"].astype(str))
    ).astype(int)

    qb1_metrics = metrics.rename(columns={
        "player_id": "preseason_qb1_id",
        "epa_per_dropback": "qb1_epa_raw",
        "cpoe": "qb1_cpoe_raw",
        "dropbacks": "qb1_dropbacks",
    })
    prev_metrics = metrics.rename(columns={
        "player_id": "prev_primary_passer_id",
        "epa_per_dropback": "prev_epa_raw",
        "cpoe": "prev_cpoe_raw",
        "dropbacks": "prev_dropbacks",
    })
    frame = frame.merge(
        qb1_metrics[["preseason_qb1_id", "qb1_epa_raw", "qb1_cpoe_raw", "qb1_dropbacks"]],
        on="preseason_qb1_id",
        how="left",
    )
    frame = frame.merge(
        prev_metrics[["prev_primary_passer_id", "prev_epa_raw", "prev_cpoe_raw", "prev_dropbacks"]],
        on="prev_primary_passer_id",
        how="left",
    )

    epa_prior = _league_prior(metrics.get("epa_per_dropback", pd.Series(dtype=float)))
    cpoe_prior = _league_prior(metrics.get("cpoe", pd.Series(dtype=float)))

    frame["qb_rookie_or_unknown"] = (
        frame["preseason_qb1_id"].isna()
        | frame["qb1_dropbacks"].fillna(0).le(ROOKIE_UNKNOWN_SAMPLE)
    ).astype(int)
    frame["qb_low_sample"] = frame["qb1_dropbacks"].fillna(0).lt(cfg.low_sample_dropbacks).astype(int)

    frame["qb_prior_epa_per_dropback"] = [
        _shrink(row.get("qb1_epa_raw"), row.get("qb1_dropbacks", 0), epa_prior, k=cfg.shrink_k)
        for _, row in frame.iterrows()
    ]
    frame["qb_prior_cpoe"] = [
        _shrink(row.get("qb1_cpoe_raw"), row.get("qb1_dropbacks", 0), cpoe_prior, k=cfg.shrink_k)
        for _, row in frame.iterrows()
    ]
    frame["qb_prior_dropback_sample"] = pd.to_numeric(frame["qb1_dropbacks"], errors="coerce").fillna(0.0)
    frame["qb_epa_change_vs_prev"] = (
        pd.to_numeric(frame["qb1_epa_raw"], errors="coerce")
        - pd.to_numeric(frame["prev_epa_raw"], errors="coerce")
    ).fillna(0.0)
    frame["qb_cpoe_change_vs_prev"] = (
        pd.to_numeric(frame["qb1_cpoe_raw"], errors="coerce")
        - pd.to_numeric(frame["prev_cpoe_raw"], errors="coerce")
    ).fillna(0.0)

    for col in QB_CONTEXT_FEATURES:
        if col not in frame.columns:
            frame[col] = 0.0
    return frame[["team", *QB_CONTEXT_FEATURES]].drop_duplicates("team")


def augment_history_with_qb_context(
    conn,
    history: pd.DataFrame,
    pairs: list[tuple[int, int]],
) -> pd.DataFrame:
    """Attach preseason QB context for each transition pair's target season."""
    out = history.copy()
    for col in QB_CONTEXT_FEATURES:
        if col not in out.columns:
            out[col] = 0.0
    for source, target in pairs:
        ctx = build_team_qb_context(
            conn, history, source_season=int(source), target_season=int(target)
        )
        mask = out["season"].eq(int(source))
        if not mask.any() or ctx.empty:
            continue
        merged = out.loc[mask].merge(ctx, on="team", how="left", suffixes=("", "_new"))
        for col in QB_CONTEXT_FEATURES:
            new_col = f"{col}_new"
            if new_col in merged.columns:
                out.loc[mask, col] = merged[new_col].to_numpy()
    return out


def attach_qb_context(frame: pd.DataFrame, qb_context: pd.DataFrame) -> pd.DataFrame:
    """Merge team QB context onto a player or team frame."""
    if qb_context.empty:
        out = frame.copy()
        for col in QB_CONTEXT_FEATURES:
            out[col] = 0.0
        return out
    team_col = "team" if "team" in frame.columns else "preseason_team"
    return frame.merge(qb_context, on=team_col, how="left", suffixes=("", "_qbctx"))


def features_for_model(
    position: str,
    stat: str,
    *,
    qb_context: bool,
    base_features: list[str],
) -> list[str]:
    """Return feature list for a model, optionally appending QB context."""
    if not qb_context:
        return list(base_features)
    if position in AFFECTED_POSITIONS or stat in {"attempts", "passing_yards", "passing_tds", "completions", "interceptions"}:
        return list(base_features) + list(QB_CONTEXT_FEATURES)
    return list(base_features)


def model_artifact_manifest(*, consumes_qb_context: bool) -> dict:
    return {QB_CONTEXT_MANIFEST_KEY: bool(consumes_qb_context)}


def assert_temporal_invariance(
    baseline: pd.DataFrame,
    mutated: pd.DataFrame,
    *,
    target_season_outcome_cols: tuple[str, ...] = (
        "passing_yards",
        "passing_tds",
        "team_passing_yards_pg",
        "fantasy_pts",
    ),
) -> bool:
    """True when QB context is unchanged after mutating target-season outcomes."""
    join_cols = ["team"] if "team" in baseline.columns else ["preseason_team"]
    merged = baseline.merge(
        mutated,
        on=join_cols,
        suffixes=("_base", "_mut"),
    )
    for col in QB_CONTEXT_FEATURES:
        base_col = f"{col}_base" if f"{col}_base" in merged.columns else col
        mut_col = f"{col}_mut" if f"{col}_mut" in merged.columns else col
        if base_col not in merged.columns or mut_col not in merged.columns:
            continue
        if not np.allclose(
            pd.to_numeric(merged[base_col], errors="coerce").fillna(0.0),
            pd.to_numeric(merged[mut_col], errors="coerce").fillna(0.0),
            equal_nan=True,
        ):
            return False
    for col in target_season_outcome_cols:
        if col in mutated.columns and col in baseline.columns:
            # Outcome columns may change; QB context must not track them.
            pass
    return True
