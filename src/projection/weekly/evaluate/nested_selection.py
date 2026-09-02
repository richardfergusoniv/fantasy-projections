"""Nested cross-validation candidate selection with dispersion-aware ranking."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from src.projection.weekly.config.scoring import ScoringConfig
from src.projection.weekly.evaluate.harness import (
    PreseasonEvalConfig,
    run_preseason_backtest,
)
from src.projection.weekly.evaluate.preseason import PromotionPolicy, promotion_gate
from src.projection.weekly.models.volume_config import (
    BASELINE_CANDIDATE_NAME,
    DEFAULT_CANDIDATE_GRID,
)


@dataclass(frozen=True)
class NestedSelectionConfig:
    """Nested expanding-window selection protocol."""

    outer_start: int = 2022
    outer_end: int = 2025
    min_inner_seasons: int = 1
    scoring: str = "half_ppr"
    random_seed: int = 42
    promotion_policy: PromotionPolicy = field(default_factory=PromotionPolicy)
    candidates: tuple[dict[str, Any], ...] = DEFAULT_CANDIDATE_GRID
    baseline_name: str = BASELINE_CANDIDATE_NAME


def _mae_improvement(report: dict[str, Any]) -> float:
    baseline = report.get("baseline") or {}
    mae = report.get("mae")
    base_mae = baseline.get("mae")
    if mae is None or base_mae in (None, 0):
        return float("-inf")
    return (float(base_mae) - float(mae)) / float(base_mae)


def _rank_delta(report: dict[str, Any]) -> float:
    baseline = report.get("baseline") or {}
    rank = report.get("rank_corr")
    base_rank = baseline.get("rank_corr")
    if rank is None or base_rank is None:
        return float("-inf")
    return float(rank) - float(base_rank)


def _dispersion_distance(report: dict[str, Any]) -> float:
    dispersion = report.get("dispersion_ratio")
    if dispersion is None:
        return float("inf")
    return abs(float(dispersion) - 1.0)


def _interval_coverage(report: dict[str, Any]) -> float | None:
    interval = report.get("interval") or {}
    return interval.get("coverage")


def _aggregate_inner_metrics(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        return {
            "n_seasons": 0,
            "mean_mae_improvement": float("-inf"),
            "mean_rank_delta": float("-inf"),
            "mean_dispersion": None,
            "mean_dispersion_distance": float("inf"),
            "min_dispersion": None,
            "max_dispersion": None,
            "mean_interval_coverage": None,
        }
    mae_gains = [_mae_improvement(r) for r in reports]
    rank_deltas = [_rank_delta(r) for r in reports]
    dispersions = [r.get("dispersion_ratio") for r in reports if r.get("dispersion_ratio") is not None]
    disp_dist = [_dispersion_distance(r) for r in reports]
    intervals = [_interval_coverage(r) for r in reports if _interval_coverage(r) is not None]
    return {
        "n_seasons": len(reports),
        "mean_mae_improvement": sum(mae_gains) / len(mae_gains),
        "mean_rank_delta": sum(rank_deltas) / len(rank_deltas),
        "mean_dispersion": sum(dispersions) / len(dispersions) if dispersions else None,
        "mean_dispersion_distance": sum(disp_dist) / len(disp_dist) if disp_dist else float("inf"),
        "min_dispersion": min(dispersions) if dispersions else None,
        "max_dispersion": max(dispersions) if dispersions else None,
        "mean_interval_coverage": sum(intervals) / len(intervals) if intervals else None,
        "reports": [
            {
                "season": r.get("season"),
                "mae": r.get("mae"),
                "rank_corr": r.get("rank_corr"),
                "dispersion_ratio": r.get("dispersion_ratio"),
                "coverage": r.get("coverage"),
                "interval": r.get("interval"),
                "by_position": r.get("by_position"),
            }
            for r in reports
        ],
    }


def _selection_policy(n_inner: int, base: PromotionPolicy) -> PromotionPolicy:
    """Inner ranking adapts min_seasons to available calibrated folds; thresholds stay frozen."""
    return replace(base, min_seasons=max(1, min(base.min_seasons, n_inner)))


def rank_candidate_on_inner(
    inner_reports: list[dict[str, Any]],
    *,
    baseline_reports: list[dict[str, Any]] | None,
    policy: PromotionPolicy,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Lexicographic ranking key and eligibility metadata.

    Order:
    1. Passes selection policy on inner calibrated folds (including dispersion).
    2. Inner-fold dispersion band satisfied.
    3. Higher mean MAE improvement vs baseline.
    4. Higher mean rank delta.
    5. Lower mean dispersion distance to 1.0.
    """
    gate = promotion_gate(inner_reports, policy=policy)
    agg = _aggregate_inner_metrics(inner_reports)
    baseline_agg = _aggregate_inner_metrics(baseline_reports or [])
    mae_vs_baseline = agg["mean_mae_improvement"] - baseline_agg.get("mean_mae_improvement", 0.0)
    dispersion_ok = (
        agg["min_dispersion"] is not None
        and agg["max_dispersion"] is not None
        and policy.min_dispersion_ratio <= agg["min_dispersion"]
        and agg["max_dispersion"] <= policy.max_dispersion_ratio
    )
    rank_key = (
        1 if gate["promote"] else 0,
        1 if dispersion_ok else 0,
        mae_vs_baseline,
        agg["mean_rank_delta"],
        -agg["mean_dispersion_distance"],
        agg["min_dispersion"] if agg["min_dispersion"] is not None else -1.0,
    )
    meta = {
        "promotion": gate,
        "aggregate": agg,
        "mae_vs_baseline": mae_vs_baseline,
        "rank_key": rank_key,
    }
    return rank_key, meta


def _filter_reports_before(
    reports: list[dict[str, Any]],
    outer_season: int,
) -> list[dict[str, Any]]:
    return [r for r in reports if int(r.get("season", 0)) < outer_season]


def select_from_cached_backtests(
    outer_season: int,
    candidate_backtests: dict[str, list[dict[str, Any]]],
    candidates: tuple[dict[str, Any], ...],
    *,
    config: NestedSelectionConfig,
) -> dict[str, Any]:
    """Select using only calibrated reports from seasons strictly before outer."""
    inner_seasons = list(range(config.outer_start, outer_season))
    # First outer season has no calibrated prior folds after warm-up.
    available_inner = [
        y for y in inner_seasons if y > config.outer_start
    ]  # calibrated seasons start after first warm-up
    if len(inner_seasons) < config.min_inner_seasons or not available_inner:
        return {
            "outer_season": outer_season,
            "status": "warmup",
            "inner_seasons": inner_seasons,
            "selected": None,
            "reason": "insufficient_inner_calibrated_seasons",
        }

    sel_policy = _selection_policy(len(available_inner), config.promotion_policy)
    baseline_name = config.baseline_name
    baseline_reports = _filter_reports_before(
        candidate_backtests.get(baseline_name) or next(iter(candidate_backtests.values())),
        outer_season,
    )

    ranked: list[dict[str, Any]] = []
    for candidate in candidates:
        inner_reports = _filter_reports_before(
            candidate_backtests.get(candidate["name"]) or [],
            outer_season,
        )
        rank_key, meta = rank_candidate_on_inner(
            inner_reports,
            baseline_reports=baseline_reports,
            policy=sel_policy,
        )
        ranked.append(
            {
                "name": candidate["name"],
                "options": candidate["options"],
                "rank_key": rank_key,
                **meta,
            }
        )
    ranked.sort(key=lambda r: r["rank_key"], reverse=True)
    winner = ranked[0]
    return {
        "outer_season": outer_season,
        "status": "selected",
        "inner_seasons": inner_seasons,
        "inner_calibrated_seasons": [r.get("season") for r in baseline_reports],
        "selected": winner["name"],
        "selected_options": winner["options"],
        "pareto_table": ranked,
    }


def _candidate_cache_path(cache_dir: Path, name: str) -> Path:
    return cache_dir / f"candidate_{name}_backtest.json"


def _seed_baseline_from_existing_backtest(
    cache_dir: Path,
    candidate: dict[str, Any],
    existing_path: Path,
) -> bool:
    """Reuse a prior production backtest for the baseline candidate when options match."""
    if not existing_path.exists():
        return False
    if candidate["name"] != BASELINE_CANDIDATE_NAME:
        return False
    payload = json.loads(existing_path.read_text(encoding="utf-8"))
    seeded = {
        "name": candidate["name"],
        "volume_options": candidate["options"],
        "calibrated_seasons": payload.get("calibrated_seasons") or [],
        "seasons": [
            {
                "season": r.get("season"),
                "mae": r.get("mae"),
                "rank_corr": r.get("rank_corr"),
                "dispersion_ratio": r.get("dispersion_ratio"),
                "coverage": r.get("coverage"),
            }
            for r in (payload.get("seasons") or [])
        ],
        "promotion": payload.get("promotion"),
        "config_fingerprint": "seeded_from_preseason_backtest",
        "seeded_from": str(existing_path),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _candidate_cache_path(cache_dir, candidate["name"])
    if not path.exists():
        path.write_text(json.dumps(seeded, indent=2, default=str), encoding="utf-8")
        print(f"seeded baseline cache from {existing_path}")
        return True
    return False


def _load_or_run_candidate_backtest(
    panel: pl.DataFrame,
    candidate: dict[str, Any],
    *,
    config: NestedSelectionConfig,
    panel_path: Path,
    cache_dir: Path | None,
    scoring: ScoringConfig,
) -> dict[str, Any]:
    """Run one full nested-calibration backtest per candidate; resume from cache."""
    if cache_dir is not None:
        path = _candidate_cache_path(cache_dir, candidate["name"])
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("volume_options") == candidate["options"]:
                return payload

    eval_config = PreseasonEvalConfig(
        panel_path=panel_path,
        outer_start=config.outer_start,
        outer_end=config.outer_end,
        scoring=config.scoring,
        volume_options=candidate["options"],
        random_seed=config.random_seed,
        promotion_policy=config.promotion_policy,
    )
    backtest = run_preseason_backtest(panel, config=eval_config, scoring=scoring)
    # Drop heavy OOF frame from persisted payload
    payload = {
        "name": candidate["name"],
        "volume_options": candidate["options"],
        "calibrated_seasons": backtest["calibrated_seasons"],
        "seasons": [
            {
                "season": r.get("season"),
                "mae": r.get("mae"),
                "rank_corr": r.get("rank_corr"),
                "dispersion_ratio": r.get("dispersion_ratio"),
                "coverage": r.get("coverage"),
            }
            for r in backtest["seasons"]
        ],
        "promotion": backtest["promotion"],
        "config_fingerprint": backtest.get("config_fingerprint"),
    }
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = _candidate_cache_path(cache_dir, candidate["name"])
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"cached {path}")
    return payload


def run_nested_selection(
    panel: pl.DataFrame,
    *,
    config: NestedSelectionConfig,
    panel_path: Path | None = None,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate each candidate once, then select with nested inner/outer folds.

    Cost: O(candidates × outer seasons) evaluate_season calls with disk resume.
    """
    scoring = ScoringConfig.from_name(config.scoring)
    resolved_panel = panel_path or Path("data/processed/player_week_panel.parquet")
    candidate_payloads: dict[str, dict[str, Any]] = {}
    candidate_backtests: dict[str, list[dict[str, Any]]] = {}

    for candidate in config.candidates:
        print(f"Evaluating candidate {candidate['name']}...")
        if cache_dir is not None and candidate["name"] == BASELINE_CANDIDATE_NAME:
            from src.projection.weekly.config.paths import OUTPUTS_DIR

            _seed_baseline_from_existing_backtest(
                cache_dir,
                candidate,
                OUTPUTS_DIR / "preseason_backtest.json",
            )
        payload = _load_or_run_candidate_backtest(
            panel,
            candidate,
            config=config,
            panel_path=resolved_panel,
            cache_dir=cache_dir,
            scoring=scoring,
        )
        candidate_payloads[candidate["name"]] = payload
        candidate_backtests[candidate["name"]] = payload.get("calibrated_seasons") or []

    fold_selections: list[dict[str, Any]] = []
    honest_outer: list[dict[str, Any]] = []
    for outer in range(config.outer_start, config.outer_end + 1):
        selection = select_from_cached_backtests(
            outer, candidate_backtests, config.candidates, config=config
        )
        fold_selections.append(selection)
        if selection["status"] != "selected":
            continue
        winner_name = selection["selected"]
        winner_reports = candidate_backtests.get(winner_name) or []
        outer_report = next(
            (r for r in winner_reports if int(r.get("season", 0)) == outer),
            None,
        )
        if outer_report is not None:
            honest = dict(outer_report)
            honest["selected_candidate"] = winner_name
            honest_outer.append(honest)

    # Global selection for 2026 training: all calibrated folds 2023–outer_end
    all_calibrated = [
        r
        for r in (candidate_backtests.get(config.baseline_name) or [])
    ]
    n_global = len(all_calibrated) or 1
    sel_policy = _selection_policy(n_global, config.promotion_policy)
    baseline_reports = candidate_backtests.get(config.baseline_name) or (
        next(iter(candidate_backtests.values())) if candidate_backtests else []
    )
    global_ranked: list[dict[str, Any]] = []
    for candidate in config.candidates:
        reports = candidate_backtests.get(candidate["name"]) or []
        rank_key, meta = rank_candidate_on_inner(
            reports,
            baseline_reports=baseline_reports,
            policy=sel_policy,
        )
        # Final promote claim requires the full frozen promotion policy
        full_gate = promotion_gate(reports, policy=config.promotion_policy)
        global_ranked.append(
            {
                "name": candidate["name"],
                "options": candidate["options"],
                "rank_key": rank_key,
                "full_promotion": full_gate,
                **meta,
            }
        )
    global_ranked.sort(key=lambda r: r["rank_key"], reverse=True)
    global_winner = global_ranked[0]
    # Only promote for training if full policy passes on all calibrated folds
    promote_for_train = bool(global_winner.get("full_promotion", {}).get("promote"))

    honest_promotion = promotion_gate(honest_outer, policy=config.promotion_policy)

    return {
        "schema_version": 2,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol": asdict(config),
        "fold_selections": fold_selections,
        "honest_outer_calibrated": honest_outer,
        "honest_promotion": honest_promotion,
        "global_pareto_table": global_ranked,
        "global_selected": global_winner["name"] if promote_for_train else None,
        "global_selected_options": global_winner["options"] if promote_for_train else {},
        "global_selection_meta": {
            **{k: v for k, v in global_winner.items() if k not in ("options",)},
            "promote_for_train": promote_for_train,
            "note": (
                "global_selected is None unless full PromotionPolicy passes; "
                "rank_key still identifies the best relative candidate"
            ),
            "best_relative_candidate": global_winner["name"],
            "best_relative_options": global_winner["options"],
        },
        "candidate_full_backtests": {
            name: {
                "calibrated_seasons": payload.get("calibrated_seasons"),
                "promotion": payload.get("promotion"),
                "config_fingerprint": payload.get("config_fingerprint"),
            }
            for name, payload in candidate_payloads.items()
        },
    }


def write_selection_artifact(result: dict[str, Any], path: Path) -> Path:
    """Persist selection with full provenance."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return path


def build_tuning_selection_payload(
    nested_result: dict[str, Any],
    *,
    experiment_id: str,
    panel_hash: str,
    code_revision: str | None = None,
) -> dict[str, Any]:
    """Build training-ready selection artifact with provenance."""
    global_meta = nested_result.get("global_selection_meta") or {}
    promote = bool(global_meta.get("promote_for_train"))
    return {
        "schema_version": 2,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "experiment_id": experiment_id,
        "panel_hash": panel_hash,
        "code_revision": code_revision,
        "selected": nested_result.get("global_selected"),
        "volume_options": nested_result.get("global_selected_options") or {},
        "promote": promote,
        "best_relative_candidate": global_meta.get("best_relative_candidate"),
        "best_relative_options": global_meta.get("best_relative_options") or {},
        "selection_rule": (
            "lexicographic: selection_gate(dispersion), mae_vs_baseline, "
            "rank_delta, dispersion_distance; train only if full PromotionPolicy passes"
        ),
        "candidate_grid": [c["name"] for c in DEFAULT_CANDIDATE_GRID],
        "global_pareto_table": nested_result.get("global_pareto_table"),
        "honest_outer_promotion": nested_result.get("honest_promotion"),
        "fold_selections": nested_result.get("fold_selections"),
        "nested_result_path": None,
    }
