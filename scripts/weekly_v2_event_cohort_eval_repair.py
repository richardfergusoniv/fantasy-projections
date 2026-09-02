"""Event cohort + evaluation-integrity repair experiment (new namespace).

Does not modify joint_usage_draws_20260831 or volume_tune_20260831_v2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src.projection.weekly.draws.cohort_panel import (
    build_complete_roster_cohort,
    build_mixture_panel_v2,
    persist_cohort_panel,
    sha256_frame_full,
    summarize_cohort_exclusions,
)
from src.projection.weekly.draws.contracts import contract_fingerprint, DEFAULT_CONTRACT
from src.projection.weekly.draws.contracts_v2 import CONTRACT_VERSION_V2
from src.projection.weekly.draws.conservation import validate_partition_draws
from src.projection.weekly.draws.event_baselines import fit_training_baselines, save_baseline_bundle
from src.projection.weekly.draws.event_models import (
    evaluate_event_predictions,
    fit_event_models,
    save_event_bundle_meta,
)
from src.projection.weekly.draws.evaluate_dist import (
    correlation_diagnostic,
    summarize_distributional_eval,
)
from src.projection.weekly.draws.feature_outcome_split import split_prediction_outcome_frames
from src.projection.weekly.draws.game_engine import generate_game_draws, player_means_from_game_draws
from src.projection.weekly.draws.prediction_inputs import (
    build_scheduled_game_from_predictions,
    projection_rows_only,
)
from src.projection.weekly.draws.readiness import GateStatus, JointReadinessReport


EXPERIMENT_ID = "event_cohort_eval_repair_20260831"
OUTER_FOLDS = ((2023, 2022), (2024, 2023), (2025, 2024))
EVENT_GATE_MAJORITY = 0.6


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _attach_fold_event_predictions(
    test: pl.DataFrame,
    bundle,
    baselines,
) -> pl.DataFrame:
    """Add fold-specific event probabilities to test rows."""
    parts = []
    for pos in ("QB", "RB", "WR", "TE"):
        sub = test.filter(pl.col("position") == pos)
        if sub.is_empty():
            continue
        cols: dict[str, list[float]] = {}
        for event, out_col in (
            ("active_label", "p_active_model"),
            ("participated_label", "p_participates"),
            ("positive_usage_label", "p_positive_usage"),
        ):
            key = f"{event}:{pos}"
            if key in bundle.models:
                cols[out_col] = list(bundle.predict_proba(event, pos, sub))  # type: ignore[arg-type]
            elif key in baselines.cells:
                cols[out_col] = list(baselines.predict(event, pos, sub))  # type: ignore[arg-type]
        if cols:
            sub = sub.with_columns([pl.Series(k, v) for k, v in cols.items()])
        parts.append(sub)
    out = pl.concat(parts, how="vertical_relaxed") if parts else test
    out = out.with_columns(
        pl.coalesce([pl.col("p_active_model"), pl.col("play_prob"), pl.lit(1.0)]).alias("p_active")
        if "p_active_model" in out.columns
        else pl.col("play_prob").fill_null(1.0).alias("p_active")
    )
    for col, default in (
        ("pred_target_share", 0.08),
        ("pred_carry_share", 0.08),
        ("pred_mean_pass_attempts", 34.0),
        ("pred_mean_rush_attempts", 26.0),
    ):
        if col not in out.columns:
            src = {
                "pred_target_share": "target_share_l5",
                "pred_carry_share": "carry_share_l5",
                "pred_mean_pass_attempts": "attempts_l5",
                "pred_mean_rush_attempts": "carries_l5",
            }[col]
            if src in out.columns and col == "pred_mean_rush_attempts":
                out = out.with_columns((pl.col(src).fill_null(default / 4.0) * 4.0).alias(col))
            elif src in out.columns:
                out = out.with_columns(pl.col(src).fill_null(default).alias(col))
            else:
                out = out.with_columns(pl.lit(default).alias(col))
    return out


def _rolling_event_eval(
    panel: pl.DataFrame,
    train_end: int,
    test_season: int,
    *,
    out_dir: Path,
) -> dict[str, Any]:
    train = panel.filter(pl.col("season") <= train_end)
    test = panel.filter(pl.col("season") == test_season)
    baselines = fit_training_baselines(train, min_positive=30)
    save_baseline_bundle(baselines, out_dir / f"baselines_train_end_{train_end}.json")
    bundle = fit_event_models(train, min_positive=30, class_weight=None)
    save_event_bundle_meta(bundle, out_dir / f"event_models_train_end_{train_end}.json")

    results: dict[str, Any] = {
        "train_end": train_end,
        "test_season": test_season,
        "events": {},
        "denominator_counts": {},
    }
    for event in ("active_label", "participated_label", "positive_usage_label"):
        by_pos: dict[str, Any] = {}
        denom_counts: dict[str, int] = {}
        for pos in ("QB", "RB", "WR", "TE"):
            sub = test.filter(
                (pl.col("position") == pos)
                & pl.col("has_scheduled_game")
                & pl.col(event).is_not_null()
            )
            if event == "participated_label":
                sub = sub.filter(pl.col("active_label") == True)  # noqa: E712
            elif event == "positive_usage_label":
                sub = sub.filter(pl.col("participated_label") == True)  # noqa: E712
            denom_counts[pos] = sub.height
            if sub.is_empty():
                by_pos[pos] = {"skipped": True, "reason": "empty_test"}
                continue
            key = f"{event}:{pos}"
            y = sub[event].cast(pl.Float64).to_numpy()
            if key in bundle.models:
                p = bundle.predict_proba(event, pos, sub)  # type: ignore[arg-type]
                model_kind = "logistic"
            elif key in baselines.cells:
                p = baselines.predict(event, pos, sub)  # type: ignore[arg-type]
                model_kind = f"baseline:{baselines.cells[key].kind}"
            else:
                by_pos[pos] = {"skipped": True, "reason": "no_model"}
                continue
            base_cell = baselines.cells.get(key)
            if base_cell is None:
                by_pos[pos] = {"skipped": True, "reason": "no_baseline"}
                continue
            base_p = baselines.predict(event, pos, sub)  # type: ignore[arg-type]
            metrics = evaluate_event_predictions(
                y,
                p,
                baseline_rate=base_cell.prevalence,
                baseline_probs=base_p,
            )
            metrics["model_kind"] = model_kind
            metrics["baseline_kind"] = base_cell.kind
            metrics["baseline_prevalence_train"] = base_cell.prevalence
            metrics["beats_depth_baseline_brier"] = metrics["brier"] < metrics["brier_baseline"] - 1e-4
            by_pos[pos] = metrics
        results["events"][event] = by_pos
        results["denominator_counts"][event] = denom_counts
    return results


def _event_gate_pass(folds: list[dict[str, Any]]) -> tuple[bool, str]:
    scored = []
    for fold in folds:
        for event, by_pos in fold["events"].items():
            for pos, m in by_pos.items():
                if m.get("skipped"):
                    continue
                if m.get("brier") is not None and m.get("brier_baseline") is not None:
                    scored.append((m["brier"], m["brier_baseline"], f"{fold['test_season']}:{event}:{pos}"))
    if not scored:
        return False, "no scored event cells"
    wins = sum(1 for b, base, _ in scored if b < base - 1e-4)
    ok = wins >= max(1, int(EVENT_GATE_MAJORITY * len(scored)))
    return ok, f"brier_beats_training_baseline={wins}/{len(scored)}"


def _run_joint_oof_games(
    panel: pl.DataFrame,
    *,
    draw_count: int,
    max_games_per_fold: int,
) -> dict[str, Any]:
    all_draws: dict[str, np.ndarray] = {}
    actuals: dict[str, float] = {}
    point_means: dict[str, float] = {}
    game_payloads: list[dict[str, Any]] = []
    corr_pairs: list[float] = []

    for test_season, train_end in OUTER_FOLDS:
        train = panel.filter(pl.col("season") <= train_end)
        test = panel.filter((pl.col("season") == test_season) & pl.col("has_scheduled_game"))
        baselines = fit_training_baselines(train, min_positive=30)
        bundle = fit_event_models(train, min_positive=30, class_weight=None)
        pred_frame = _attach_fold_event_predictions(test, bundle, baselines)
        game_ids = (
            pred_frame.filter(pl.col("game_id").is_not_null())["game_id"].unique().to_list()[:max_games_per_fold]
        )
        for gi, gid in enumerate(game_ids):
            gdf = pred_frame.filter(pl.col("game_id") == gid)
            teams = gdf["team"].unique().to_list()
            if len(teams) < 2:
                continue
            home_t, away_t = teams[0], teams[1]
            rows = projection_rows_only(list(gdf.iter_rows(named=True)))
            outcome_lookup = {str(r["gsis_id"]): r for r in gdf.iter_rows(named=True)}
            game = build_scheduled_game_from_predictions(
                rows,
                game_id=str(gid),
                season=test_season,
                week=int(gdf["week"][0]),
                home_team=home_t,
                away_team=away_t,
            )
            payload = generate_game_draws(game, draw_count=draw_count, seed=1000 + test_season * 100 + gi)
            game_payloads.append(payload)
            means = player_means_from_game_draws(payload)
            for team_block, tname in ((payload["teams"][0], home_t), (payload["teams"][1], away_t)):
                trows = {str(r["gsis_id"]): r for r in rows if str(r.get("team")) == tname}
                players = team_block["players"]
                if len(players) >= 2:
                    a, b = players[0], players[1]
                    xa = np.array([d.get("pass_yards", d.get("rec_yards", 0.0)) for d in a["draws"]])
                    xb = np.array([d.get("rec_yards", d.get("rush_yards", 0.0)) for d in b["draws"]])
                    corr_pairs.append(correlation_diagnostic(xa, xb))
                for p in players:
                    pid = p["player_id"]
                    fps = []
                    for d in p["draws"]:
                        fp = (
                            0.04 * d.get("pass_yards", 0)
                            + 4 * d.get("pass_tds", 0)
                            - 2 * d.get("pass_ints", 0)
                            + 0.1 * d.get("rush_yards", 0)
                            + 6 * d.get("rush_tds", 0)
                            + 0.1 * d.get("rec_yards", 0)
                            + 6 * d.get("rec_tds", 0)
                            + 0.5 * d.get("receptions", 0)
                        )
                        fps.append(fp)
                    all_draws[pid] = np.array(fps, dtype=float)
                    src = outcome_lookup.get(pid, {})
                    if src:
                        actuals[pid] = float(src.get("fantasy_points") or 0.0)
                    if pid in means:
                        m = means[pid]
                        point_means[pid] = (
                            0.04 * m.get("pass_yards", 0)
                            + 4 * m.get("pass_tds", 0)
                            - 2 * m.get("pass_ints", 0)
                            + 0.1 * m.get("rush_yards", 0)
                            + 6 * m.get("rush_tds", 0)
                            + 0.1 * m.get("rec_yards", 0)
                            + 6 * m.get("rec_tds", 0)
                            + 0.5 * m.get("receptions", 0)
                        )

    conservation = validate_partition_draws(game_payloads, tol=3.0)
    dist = summarize_distributional_eval(
        player_draws=all_draws, actuals=actuals, point_means=point_means
    )
    return {
        "n_games": len(game_payloads),
        "n_players": len(all_draws),
        "conservation": conservation.to_dict(),
        "distributional": dist,
        "teammate_corr_mean": float(np.nanmean(corr_pairs)) if corr_pairs else None,
        "partition_content_hash": hashlib.sha256(
            json.dumps([p.get("game_id") for p in game_payloads], sort_keys=True).encode()
        ).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=Path("data/processed/player_week_panel.parquet"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(f"output/weekly_v2/experiments/{EXPERIMENT_ID}"),
    )
    parser.add_argument("--seasons", type=int, nargs="*", default=[2022, 2023, 2024, 2025])
    parser.add_argument("--draw-count", type=int, default=80)
    parser.add_argument("--max-games-per-fold", type=int, default=40)
    parser.add_argument("--fast", action="store_true", help="CI-sized sample (fewer games)")
    args = parser.parse_args()

    if args.fast:
        args.max_games_per_fold = min(args.max_games_per_fold, 4)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    panel_hash = _sha256_file(args.panel) if args.panel.exists() else ""

    raw = pl.read_parquet(args.panel)
    cohort = build_complete_roster_cohort(raw, seasons=list(args.seasons))
    mixture = build_mixture_panel_v2(cohort)
    cohort_art = persist_cohort_panel(
        mixture,
        out / "cohort",
        source_hashes={"player_week_panel": panel_hash},
    )
    features, outcomes, manifest = split_prediction_outcome_frames(mixture)
    features.write_parquet(out / "prediction_features.parquet")
    outcomes.write_parquet(out / "outcome_labels.parquet")
    (out / "feature_outcome_manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )

    event_dir = out / "event_oof"
    event_dir.mkdir(exist_ok=True)
    event_folds = []
    for test_season, train_end in OUTER_FOLDS:
        event_folds.append(
            _rolling_event_eval(mixture, train_end, test_season, out_dir=event_dir)
        )
    (out / "event_calibration_corrected.json").write_text(
        json.dumps(event_folds, indent=2, sort_keys=True), encoding="utf-8"
    )
    ev_ok, ev_detail = _event_gate_pass(event_folds)

    joint = _run_joint_oof_games(
        mixture,
        draw_count=args.draw_count,
        max_games_per_fold=args.max_games_per_fold,
    )
    cons_ok = bool(joint["conservation"].get("ok"))
    zero_gap = (joint["distributional"].get("zero_mass") or {}).get("abs_gap", 1.0)
    zero_ok = zero_gap == zero_gap and zero_gap < 0.25

    tune_path = Path("output/weekly_v2/experiments/volume_tune_20260831_v2/tuning_selection.json")
    tune = json.loads(tune_path.read_text(encoding="utf-8")) if tune_path.exists() else {}
    point_dispersion_passes = False

    report = JointReadinessReport(
        point_model_classification=GateStatus(
            "point_model_classification",
            True,
            "Trained artifact GO with caveats; volume tune promote=false preserved",
            {"tuning_selection": tune.get("selected"), "promote": tune.get("promote")},
            evidence_hash=panel_hash[:16] if panel_hash else "",
        ),
        event_probability_calibration=GateStatus(
            "event_probability_calibration",
            ev_ok,
            ev_detail,
            {"folds": event_folds},
            evidence_hash=_sha256_file(out / "event_calibration_corrected.json")[:16],
            evidence_path=str(out / "event_calibration_corrected.json"),
        ),
        joint_draw_proper_scores=GateStatus(
            "joint_draw_proper_scores",
            bool(zero_ok),
            f"zero_gap={zero_gap}; leakage-free OOF redraw",
            joint["distributional"],
            evidence_hash=joint.get("partition_content_hash", "")[:16],
        ),
        per_draw_conservation=GateStatus(
            "per_draw_conservation",
            cons_ok,
            f"violations={joint['conservation'].get('n_violations')}; no violation-count bypass",
            joint["conservation"],
            evidence_hash=str(joint["conservation"].get("n_violations", "")),
        ),
        ppfd_component_readiness=GateStatus(
            "ppfd_component_readiness",
            False,
            "Historical PPFD evaluation incomplete in this repair pass",
        ),
        kicker_readiness=GateStatus("kicker_readiness", False, "Historical K baseline incomplete"),
        dst_readiness=GateStatus("dst_readiness", False, "Historical DST baseline incomplete"),
        league_scoring_completeness=GateStatus(
            "league_scoring_completeness",
            False,
            "Exact live six-league rule snapshots not validated in this run",
        ),
        decision_lineup_matchup=GateStatus(
            "decision_lineup_matchup",
            False,
            "Full OOF lineup/matchup backtest incomplete",
        ),
        artifact_integrity=GateStatus(
            "artifact_integrity",
            True,
            f"cohort_content_hash={cohort_art.content_hash[:16]}",
            {"cohort_hash": cohort_art.content_hash, "manifest": manifest.to_dict()},
            evidence_hash=cohort_art.content_hash[:16],
        ),
    )
    report.recompute_decisions(point_dispersion_passes=point_dispersion_passes)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": EXPERIMENT_ID,
        "contract_version": CONTRACT_VERSION_V2,
        "legacy_contract_fingerprint": contract_fingerprint(DEFAULT_CONTRACT),
        "panel_hash": panel_hash,
        "cohort": cohort_art.to_dict(),
        "cohort_exclusions": summarize_cohort_exclusions(mixture),
        "feature_outcome_manifest": manifest.to_dict(),
        "event_calibration_corrected": str(out / "event_calibration_corrected.json"),
        "prior_invalidated_metrics": {
            "event_brier_0_of_21": "oracle test-fold frequency baseline + wrong denominators",
            "joint_crps_2_84_vs_4_89": "same-week actuals in draw inputs",
            "zero_mass_gap_0_252": "leaked partition inputs",
            "teammate_corr_0_012": "invalid evaluation path",
        },
        "joint_oof": joint,
        "readiness": report.to_dict(),
        "shared_latent_tuning_authorized": ev_ok,
        "volume_tune_ref": {
            "path": str(tune_path),
            "promote": tune.get("promote"),
            "selected": tune.get("selected"),
        },
    }
    (out / "event_cohort_eval_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(out),
                "event_gate": ev_detail,
                "joint_classification": report.joint_draw_classification,
                "auto_publish_allowed": report.auto_publish_allowed,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
