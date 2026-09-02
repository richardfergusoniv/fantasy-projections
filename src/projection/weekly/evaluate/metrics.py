"""Evaluation metrics for projections."""
# Ported from fantasy-projections-2 (team-first weekly v2). See docs/WEEKLY_V2_PORT_PROVENANCE.md.

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

# Efficiency metrics derived from counting stats: (name, numerator, denominator, min_denom)
EFFICIENCY_SPECS: tuple[tuple[str, str, str, int], ...] = (
    ("ypa", "passing_yards", "attempts", 10),
    ("ypc", "rushing_yards", "carries", 5),
    ("ypr", "receiving_yards", "receptions", 2),
    ("catch_rate", "receptions", "targets", 3),
)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 3:
        return float("nan")
    # Rank correlation via average ranks
    def _rank(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(a), dtype=float)
        return ranks

    rt = _rank(y_true[mask])
    rp = _rank(y_pred[mask])
    rt = (rt - rt.mean()) / (rt.std() + 1e-12)
    rp = (rp - rp.mean()) / (rp.std() + 1e-12)
    return float(np.mean(rt * rp))


def _ratio_mae(
    frame: pl.DataFrame,
    *,
    pred_num: str,
    pred_den: str,
    act_num: str,
    act_den: str,
    min_denom: int,
) -> dict[str, Any] | None:
    """MAE of pred_num/pred_den vs act_num/act_den with volume threshold on actuals."""
    needed = [pred_num, pred_den, act_num, act_den]
    if any(c not in frame.columns for c in needed):
        return None
    sub = frame.filter(
        (pl.col(act_den) >= min_denom)
        & (pl.col(pred_den) > 0)
        & pl.col(pred_num).is_not_null()
        & pl.col(act_num).is_not_null()
    )
    if sub.is_empty():
        return None
    y_pred = (sub[pred_num] / sub[pred_den]).to_numpy().astype(float)
    y_true = (sub[act_num] / sub[act_den]).to_numpy().astype(float)
    return {"n": sub.height, "mae": mae(y_true, y_pred)}


def compute_efficiency_mae(
    projections: pl.DataFrame,
    actuals: pl.DataFrame,
    *,
    keys: tuple[str, ...] = ("gsis_id", "season", "week"),
) -> dict[str, Any]:
    """Efficiency MAE from projected vs actual counting stats.

    Prefers deriving ratios from box-score columns on each frame
    (e.g. ypa = passing_yards / attempts) rather than precomputed labels.
    Rows need sufficient actual volume (min denom) to reduce div-by-zero noise.
    """
    join_keys = [k for k in keys if k in projections.columns and k in actuals.columns]
    if not join_keys:
        return {}

    count_cols = sorted(
        {
            c
            for _, num, den, _ in EFFICIENCY_SPECS
            for c in (num, den)
            if c in projections.columns or c in actuals.columns
        }
    )
    pred_cols = [c for c in count_cols if c in projections.columns]
    act_cols = [c for c in count_cols if c in actuals.columns]
    if not pred_cols or not act_cols:
        return {}

    pred = projections.select(join_keys + pred_cols + (["position"] if "position" in projections.columns else []))
    # Disambiguate shared counting-stat names after join
    pred = pred.rename({c: f"pred_{c}" for c in pred_cols})
    act = actuals.select(join_keys + act_cols).rename({c: f"act_{c}" for c in act_cols})
    merged = pred.join(act, on=join_keys, how="inner")
    if merged.is_empty():
        return {}

    overall: dict[str, Any] = {}
    for name, num, den, min_den in EFFICIENCY_SPECS:
        result = _ratio_mae(
            merged,
            pred_num=f"pred_{num}",
            pred_den=f"pred_{den}",
            act_num=f"act_{num}",
            act_den=f"act_{den}",
            min_denom=min_den,
        )
        if result is not None:
            overall[name] = result

    by_position: dict[str, Any] = {}
    if "position" in merged.columns:
        for pos in merged["position"].unique().to_list():
            sub = merged.filter(pl.col("position") == pos)
            pos_report: dict[str, Any] = {}
            for name, num, den, min_den in EFFICIENCY_SPECS:
                result = _ratio_mae(
                    sub,
                    pred_num=f"pred_{num}",
                    pred_den=f"pred_{den}",
                    act_num=f"act_{num}",
                    act_den=f"act_{den}",
                    min_denom=min_den,
                )
                if result is not None and result["n"] >= 5:
                    pos_report[name] = result
            if pos_report:
                by_position[pos] = pos_report

    out: dict[str, Any] = {}
    if overall:
        out["overall"] = overall
    if by_position:
        out["by_position"] = by_position
    return out


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 3:
        return float("nan")
    yt = y_true[mask]
    yp = y_pred[mask]
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def dispersion_ratio(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 3:
        return float("nan")
    sd_true = float(np.std(y_true[mask]))
    if sd_true < 1e-12:
        return float("nan")
    return float(np.std(y_pred[mask]) / sd_true)


def top_n_hit_rate(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> float:
    """Fraction of actual top-n that appear in predicted top-n."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < n:
        return float("nan")
    yt = y_true[mask]
    yp = y_pred[mask]
    true_top = set(np.argsort(-yt)[:n].tolist())
    pred_top = set(np.argsort(-yp)[:n].tolist())
    return len(true_top & pred_top) / float(n)


def _score_vectors(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    return {
        "n": int(np.sum(np.isfinite(y_true) & np.isfinite(y_pred))),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "rank_corr": spearman_corr(y_true, y_pred),
        "r2": r_squared(y_true, y_pred),
        "dispersion_ratio": dispersion_ratio(y_pred, y_true),
    }


def evaluate_season_level(
    projections: pl.DataFrame,
    actuals: pl.DataFrame,
    *,
    keys: tuple[str, ...] = ("gsis_id", "season", "week"),
    pred_col: str = "fantasy_points",
    actual_col: str = "fantasy_points",
    top_ns: tuple[int, ...] = (12, 24, 36),
) -> dict[str, Any]:
    """Aggregate player-season totals and score MAE / rank / top-N / dispersion."""
    join_keys = [k for k in keys if k in projections.columns and k in actuals.columns]
    if not join_keys or "gsis_id" not in join_keys:
        return {}

    pred_cols = join_keys + [pred_col]
    if "position" in projections.columns:
        pred_cols = pred_cols + ["position"]
    pred = projections.select([c for c in pred_cols if c in projections.columns]).rename(
        {pred_col: "pred_fp"}
    )
    act = actuals.select(join_keys + [actual_col]).rename({actual_col: "actual_fp"})
    merged = pred.join(act, on=join_keys, how="inner")
    if merged.is_empty():
        return {"n": 0}

    group_keys = ["gsis_id"]
    if "season" in merged.columns:
        group_keys.append("season")
    agg_exprs = [
        pl.col("pred_fp").sum().alias("pred_fp"),
        pl.col("actual_fp").sum().alias("actual_fp"),
        pl.len().alias("n_weeks"),
    ]
    if "position" in merged.columns:
        agg_exprs.append(pl.col("position").first().alias("position"))
    season_df = merged.group_by(group_keys).agg(agg_exprs)

    y_true = season_df["actual_fp"].to_numpy().astype(float)
    y_pred = season_df["pred_fp"].to_numpy().astype(float)
    report: dict[str, Any] = _score_vectors(y_true, y_pred)
    report["top_n_hit_rate"] = {
        f"top_{n}": top_n_hit_rate(y_true, y_pred, n) for n in top_ns
    }

    by_position: dict[str, Any] = {}
    if "position" in season_df.columns:
        for pos in season_df["position"].unique().to_list():
            sub = season_df.filter(pl.col("position") == pos)
            yt = sub["actual_fp"].to_numpy().astype(float)
            yp = sub["pred_fp"].to_numpy().astype(float)
            pos_rep = _score_vectors(yt, yp)
            pos_rep["top_n_hit_rate"] = {
                f"top_{n}": top_n_hit_rate(yt, yp, n) for n in top_ns if sub.height >= n
            }
            by_position[str(pos)] = pos_rep
    if by_position:
        report["by_position"] = by_position
    return report


def build_last5_baseline(
    panel: pl.DataFrame,
    *,
    season: int,
    fp_col: str = "fantasy_points",
) -> pl.DataFrame:
    """Lagged 5-game rolling mean of fantasy points (no current-week leakage)."""
    hist = panel.filter(pl.col("season") == season).sort(["gsis_id", "week"])
    if hist.is_empty() or fp_col not in hist.columns:
        return pl.DataFrame()
    out = hist.with_columns(
        pl.col(fp_col)
        .shift(1)
        .rolling_mean(window_size=5, min_samples=1)
        .over("gsis_id")
        .alias("fantasy_points")
    )
    keep = ["gsis_id", "season", "week", "fantasy_points"]
    if "position" in out.columns:
        keep.append("position")
    return out.select([c for c in keep if c in out.columns]).filter(
        pl.col("fantasy_points").is_not_null()
    )


def build_prior_season_ppg_baseline(
    panel: pl.DataFrame,
    *,
    season: int,
    fp_col: str = "fantasy_points",
) -> pl.DataFrame:
    """Prior-season points-per-game broadcast to every week of ``season``."""
    prior = panel.filter(pl.col("season") == season - 1)
    if prior.is_empty() or fp_col not in prior.columns:
        return pl.DataFrame()
    ppg = prior.group_by("gsis_id").agg(
        [
            pl.col(fp_col).mean().alias("fantasy_points"),
            pl.col("position").first().alias("position")
            if "position" in prior.columns
            else pl.lit(None).alias("position"),
        ]
    )
    weeks = (
        panel.filter(pl.col("season") == season)
        .select(["gsis_id", "season", "week"] + (["position"] if "position" in panel.columns else []))
        .unique()
    )
    if weeks.is_empty():
        return pl.DataFrame()
    out = weeks.join(ppg.drop([c for c in ("position",) if c in ppg.columns and c in weeks.columns]), on="gsis_id", how="inner")
    if "position" not in out.columns and "position" in ppg.columns:
        out = out.join(ppg.select(["gsis_id", "position"]), on="gsis_id", how="left")
    return out.filter(pl.col("fantasy_points").is_not_null())


def _depth_bucket(rank: float | None) -> str:
    if rank is None or (isinstance(rank, float) and not np.isfinite(rank)):
        return "null"
    r = int(round(float(rank)))
    if r >= 6:
        return "6+"
    if r < 1:
        return "null"
    return str(r)


def compute_share_mae(
    projections: pl.DataFrame,
    actuals: pl.DataFrame,
    *,
    keys: tuple[str, ...] = ("gsis_id", "season", "week"),
) -> dict[str, Any]:
    """Share MAE overall, by position, and by depth_rank bucket."""
    join_keys = [k for k in keys if k in projections.columns and k in actuals.columns]
    if not join_keys:
        return {}

    out: dict[str, Any] = {"overall": {}, "by_position": {}, "by_depth_rank": {}}

    for share_pred, share_act in (
        ("pred_target_share", "target_share"),
        ("pred_carry_share", "carry_share"),
    ):
        if share_pred not in projections.columns or share_act not in actuals.columns:
            continue

        pred_cols = join_keys + [share_pred]
        for extra in ("position", "depth_rank"):
            if extra in projections.columns and extra not in pred_cols:
                pred_cols.append(extra)
        spred = projections.select(pred_cols)
        act_cols = join_keys + [share_act]
        if "depth_rank" not in spred.columns and "depth_rank" in actuals.columns:
            act_cols = act_cols + ["depth_rank"]
        if "position" not in spred.columns and "position" in actuals.columns:
            act_cols = act_cols + ["position"]
        sact = actuals.select([c for c in act_cols if c in actuals.columns])
        sm = spred.join(sact, on=join_keys, how="inner")
        # Prefer projection depth_rank; fall back to actuals suffix
        if "depth_rank" not in sm.columns and "depth_rank_right" in sm.columns:
            sm = sm.rename({"depth_rank_right": "depth_rank"})
        elif "depth_rank" in sm.columns and "depth_rank_right" in sm.columns:
            sm = sm.with_columns(
                pl.coalesce([pl.col("depth_rank"), pl.col("depth_rank_right")]).alias("depth_rank")
            ).drop("depth_rank_right")
        sm = sm.drop_nulls(subset=[share_pred, share_act])
        if sm.is_empty():
            continue

        yt = sm[share_act].to_numpy().astype(float)
        yp = sm[share_pred].to_numpy().astype(float)
        out["overall"][share_act] = {
            "n": sm.height,
            "mae": mae(yt, yp),
            "bias": float(np.nanmean(yp - yt)),
        }

        if "position" in sm.columns:
            pos_map: dict[str, Any] = out["by_position"].setdefault(share_act, {})
            for pos in sm["position"].unique().to_list():
                if pos is None:
                    continue
                sub = sm.filter(pl.col("position") == pos)
                if sub.height < 5:
                    continue
                yt_p = sub[share_act].to_numpy().astype(float)
                yp_p = sub[share_pred].to_numpy().astype(float)
                pos_map[str(pos)] = {
                    "n": sub.height,
                    "mae": mae(yt_p, yp_p),
                    "bias": float(np.nanmean(yp_p - yt_p)),
                }

        if "depth_rank" in sm.columns:
            buckets = [_depth_bucket(r) for r in sm["depth_rank"].to_list()]
            sm = sm.with_columns(pl.Series("_depth_bucket", buckets))
            depth_map: dict[str, Any] = out["by_depth_rank"].setdefault(share_act, {})
            for bucket in sorted(set(buckets), key=lambda b: (b == "null", b == "6+", b)):
                sub = sm.filter(pl.col("_depth_bucket") == bucket)
                if sub.height < 5:
                    continue
                yt_d = sub[share_act].to_numpy().astype(float)
                yp_d = sub[share_pred].to_numpy().astype(float)
                depth_map[bucket] = {
                    "n": sub.height,
                    "mae": mae(yt_d, yp_d),
                    "bias": float(np.nanmean(yp_d - yt_d)),
                }

    # Flatten overall to keep backward-compatible top-level keys used by old reports
    flat = {k: v["mae"] for k, v in out["overall"].items()}
    if flat:
        out["target_share"] = flat.get("target_share")
        out["carry_share"] = flat.get("carry_share")
    return out if out["overall"] else {}


def evaluate_projections(
    projections: pl.DataFrame,
    actuals: pl.DataFrame,
    *,
    keys: tuple[str, ...] = ("gsis_id", "season", "week"),
    pred_col: str = "fantasy_points",
    actual_col: str = "fantasy_points",
) -> dict[str, Any]:
    """Score projections against actual fantasy points."""
    join_keys = [k for k in keys if k in projections.columns and k in actuals.columns]
    pred_extra = [pred_col]
    if "position" in projections.columns:
        pred_extra.append("position")
    pred_extra.extend(c for c in ("floor", "ceiling") if c in projections.columns)
    pred = projections.select(join_keys + pred_extra)
    act = actuals.select(join_keys + [actual_col]).rename({actual_col: "actual_fp"})
    if pred_col in pred.columns:
        pred = pred.rename({pred_col: "pred_fp"})
    merged = pred.join(act, on=join_keys, how="inner")
    if merged.is_empty():
        return {"n": 0, "mae": None, "rmse": None, "rank_corr": None, "by_position": {}}

    y_true = merged["actual_fp"].to_numpy().astype(float)
    y_pred = merged["pred_fp"].to_numpy().astype(float)
    report: dict[str, Any] = {
        "n": merged.height,
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "rank_corr": spearman_corr(y_true, y_pred),
        "dispersion_ratio": dispersion_ratio(y_pred, y_true),
        "by_position": {},
    }

    if {"floor", "ceiling"}.issubset(merged.columns):
        lo = merged["floor"].to_numpy().astype(float)
        hi = merged["ceiling"].to_numpy().astype(float)
        valid = np.isfinite(y_true) & np.isfinite(lo) & np.isfinite(hi) & (hi >= lo)
        if valid.any():
            covered = (y_true[valid] >= lo[valid]) & (y_true[valid] <= hi[valid])
            report["interval"] = {
                "n": int(valid.sum()),
                "coverage": float(np.mean(covered)),
                "mean_width": float(np.mean(hi[valid] - lo[valid])),
            }

    if "position" in merged.columns:
        for pos in merged["position"].unique().to_list():
            sub = merged.filter(pl.col("position") == pos)
            yt = sub["actual_fp"].to_numpy().astype(float)
            yp = sub["pred_fp"].to_numpy().astype(float)
            report["by_position"][pos] = {
                "n": sub.height,
                "mae": mae(yt, yp),
                "rmse": rmse(yt, yp),
                "rank_corr": spearman_corr(yt, yp),
                "dispersion_ratio": dispersion_ratio(yp, yt),
            }
            if {"floor", "ceiling"}.issubset(sub.columns):
                lo_p = sub["floor"].to_numpy().astype(float)
                hi_p = sub["ceiling"].to_numpy().astype(float)
                valid_p = np.isfinite(yt) & np.isfinite(lo_p) & np.isfinite(hi_p) & (hi_p >= lo_p)
                if valid_p.any():
                    report["by_position"][pos]["interval_coverage"] = float(
                        np.mean((yt[valid_p] >= lo_p[valid_p]) & (yt[valid_p] <= hi_p[valid_p]))
                    )

    share_report = compute_share_mae(projections, actuals, keys=keys)
    if share_report:
        report["share_mae"] = share_report

    efficiency = compute_efficiency_mae(projections, actuals, keys=keys)
    if efficiency:
        report["efficiency_mae"] = efficiency

    # Rookie subset
    if "is_rookie" in projections.columns:
        rook_ids = projections.filter(pl.col("is_rookie") == 1).select(join_keys)
        rook = merged.join(rook_ids, on=join_keys, how="inner")
        if rook.height:
            report["rookies"] = {
                "n": rook.height,
                "mae": mae(
                    rook["actual_fp"].to_numpy().astype(float),
                    rook["pred_fp"].to_numpy().astype(float),
                ),
                "rmse": rmse(
                    rook["actual_fp"].to_numpy().astype(float),
                    rook["pred_fp"].to_numpy().astype(float),
                ),
            }

    season_report = evaluate_season_level(
        projections, actuals, keys=keys, pred_col=pred_col, actual_col=actual_col
    )
    if season_report:
        report["season_level"] = season_report

    return report


def format_report(report: dict[str, Any]) -> str:
    lines = [
        f"n={report.get('n')}",
        f"MAE={report.get('mae'):.3f}" if report.get("mae") is not None else "MAE=n/a",
        f"RMSE={report.get('rmse'):.3f}" if report.get("rmse") is not None else "RMSE=n/a",
        f"RankCorr={report.get('rank_corr'):.3f}"
        if report.get("rank_corr") is not None
        else "RankCorr=n/a",
    ]
    if report.get("dispersion_ratio") is not None and np.isfinite(report["dispersion_ratio"]):
        lines.append(f"Dispersion={report['dispersion_ratio']:.3f}")
    by_pos = report.get("by_position") or {}
    for pos, m in by_pos.items():
        lines.append(
            f"  {pos}: n={m['n']} MAE={m['mae']:.3f} RMSE={m['rmse']:.3f} RankCorr={m['rank_corr']:.3f}"
        )
    if report.get("share_mae"):
        sm = report["share_mae"]
        overall = sm.get("overall") or {}
        if overall:
            parts = [
                f"{k}=n={v['n']} MAE={v['mae']:.4f} bias={v['bias']:+.4f}"
                for k, v in overall.items()
            ]
            lines.append("Share MAE: " + ", ".join(parts))
        else:
            lines.append(f"Share MAE: {sm}")
        by_depth = sm.get("by_depth_rank") or {}
        for share_name, buckets in by_depth.items():
            parts = [
                f"{b}=n={m['n']} MAE={m['mae']:.4f}"
                for b, m in buckets.items()
            ]
            if parts:
                lines.append(f"  Share {share_name} by depth: " + ", ".join(parts))
    eff = report.get("efficiency_mae") or {}
    overall_eff = eff.get("overall") or {}
    if overall_eff:
        parts = [
            f"{name}=n={m['n']} MAE={m['mae']:.3f}" for name, m in overall_eff.items()
        ]
        lines.append("Efficiency MAE: " + ", ".join(parts))
    for pos, metrics in (eff.get("by_position") or {}).items():
        parts = [
            f"{name}=n={m['n']} MAE={m['mae']:.3f}" for name, m in metrics.items()
        ]
        lines.append(f"  Efficiency {pos}: " + ", ".join(parts))
    if report.get("rookies"):
        r = report["rookies"]
        lines.append(f"Rookies: n={r['n']} MAE={r['mae']:.3f} RMSE={r['rmse']:.3f}")
    season = report.get("season_level") or {}
    if season.get("n"):
        lines.append(
            f"Season: n={season['n']} MAE={season.get('mae'):.3f} "
            f"RankCorr={season.get('rank_corr'):.3f} "
            f"R2={season.get('r2'):.3f} Dispersion={season.get('dispersion_ratio'):.3f}"
        )
        hit = season.get("top_n_hit_rate") or {}
        if hit:
            parts = [f"{k}={v:.3f}" for k, v in hit.items() if v == v]
            if parts:
                lines.append("  Top-N hit: " + ", ".join(parts))
    for name in ("baseline_last5", "baseline_prior_ppg"):
        base = report.get(name)
        if not base:
            continue
        lines.append(
            f"{name}: MAE={base.get('mae'):.3f} RankCorr={base.get('rank_corr'):.3f}"
            if base.get("mae") is not None
            else f"{name}: n/a"
        )
    return "\n".join(lines)
