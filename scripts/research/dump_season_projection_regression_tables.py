#!/usr/bin/env python3
"""Research-only: print the season-projection tables cited in

    docs/research/SEASON_PROJECTION_REGRESSION_REEXPLORE_2026-09-14.md

Does not train, predict, compose, seal, or promote. Reads committed
artifacts under output/ with the stdlib only.

Usage (from repo root):

    python3 scripts/research/dump_season_projection_regression_tables.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL_JSON = ROOT / "output" / "fantasy_evaluation_summary_2025.json"
EVAL_CSV = ROOT / "output" / "fantasy_evaluation_2025.csv"
HOLDOUT = ROOT / "output" / "backtest" / "veteran_holdout_2025.csv"
ACCURACY = ROOT / "output" / "accuracy_first_2026" / "report.json"
WEIGHTS = ROOT / "output" / "accuracy_first_2026" / "ensemble_weights.json"
CONC = ROOT / "models" / "concentration_calibration.json"


def _print_eval_table(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = data["metadata"]
    print(f"\n=== {path.relative_to(ROOT)} ===")
    print(f"source->target: {meta.get('source_season')} -> {meta.get('target_season')}")
    print(f"exposure_blend_alpha: {meta.get('exposure_blend_alpha')}")
    print(f"artifact_provenance: {meta.get('composition_artifact_provenance')}")
    coverage = meta.get("composition_stage_coverage") or {}
    print("stage_coverage:")
    for key, value in coverage.items():
        print(f"  {key}: {value}")
    starters = meta.get("qb_starter_metrics") or {}
    if starters:
        print("qb_starter_metrics:")
        for scope, row in starters.items():
            print(
                f"  {scope}: n={row.get('n')} rate_rho={row.get('rate_spearman')} "
                f"rate_mae={row.get('rate_mae')} pts_mae={row.get('points_mae')} "
                f"tier={row.get('tier_hits')} bias={row.get('mean_bias')}"
            )
    print(
        "pos\tmethod\tn\tspearman\tpoints_mae\ttier\tvorp_mae\tpred_repl\tactual_repl"
    )
    for row in data["metrics"]:
        if row.get("scope") != "all_eligible":
            continue
        print(
            f"{row['position']}\t{row['method']}\t{row['n']}\t"
            f"{row['spearman']:.4f}\t{row['points_mae']:.2f}\t"
            f"{row.get('tier_hits')}/{row.get('tier_rank')}\t"
            f"{row['vorp_mae']:.2f}\t"
            f"{row['predicted_replacement_points']:.1f}\t"
            f"{row['actual_replacement_points']:.1f}"
        )


def _print_rate_holdout(path: Path) -> None:
    print(f"\n=== {path.relative_to(ROOT)} ===")
    print("position,stat,n_test,model_mae,naive_mae,model_wins")
    wins = losses = 0
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            won = str(row.get("model_wins", "")).lower() == "true"
            wins += int(won)
            losses += int(not won)
            print(
                f"{row['position']},{row['stat']},{row['n_test']},"
                f"{float(row['model_mae']):.4f},{float(row['naive_mae']):.4f},"
                f"{row['model_wins']}"
            )
    print(f"model_wins {wins}/{wins + losses} cells")


def _print_accuracy(path: Path, weights_path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    holdout = data["holdout_2025"]
    gate = holdout["overall_gate"]
    print(f"\n=== {path.relative_to(ROOT)} ===")
    print(f"verdict: {data.get('verdict')}")
    print(
        f"incumbent: MAE {gate['incumbent']['points_mae']:.2f}  "
        f"Spearman {gate['incumbent']['spearman']:.3f}"
    )
    print(
        f"selected:  MAE {gate['proposed']['points_mae']:.2f}  "
        f"Spearman {gate['proposed']['spearman']:.3f}"
    )
    boot = holdout["final"]["bootstrap"]
    mae = boot["mae_delta_candidate_minus_incumbent"]
    rho = boot["spearman_delta_candidate_minus_incumbent"]
    print(
        f"bootstrap MAE delta 95% [{mae['p025']:.2f}, {mae['p975']:.2f}]  "
        f"Spearman [{rho['p025']:+.3f}, {rho['p975']:+.3f}]"
    )
    print("position selections / 2025 holdout:")
    for pos, evidence in holdout["position_evidence"].items():
        selected = evidence["selected"]
        arms = evidence["arms"]
        print(
            f"  {pos}: {selected}  "
            f"incumbent MAE {arms['incumbent']['points_mae']:.2f} rho {arms['incumbent']['spearman']:.3f}  "
            f"selected MAE {arms[selected]['points_mae']:.2f} rho {arms[selected]['spearman']:.3f}"
        )
    weights = json.loads(weights_path.read_text(encoding="utf-8"))
    print("refit 2026 weights:")
    for pos, spec in weights["positions"].items():
        print(f"  {pos}: arm={spec['arm']} {spec['weights']}")
    app = data.get("application_2026") or {}
    print(
        f"2026 application: n_players={app.get('n_players')} "
        f"n_accuracy_ensemble_applied={app.get('n_accuracy_ensemble_applied')}"
    )


def _print_concentration(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    print(f"\n=== {path.relative_to(ROOT)} ===")
    print(f"version: {data.get('version')}")
    for cell, spec in (data.get("cells") or {}).items():
        print(
            f"  {cell}: exponent={spec.get('exponent')} "
            f"fitted={spec.get('fitted_exponent')} promoted={spec.get('promoted')}"
        )


def _print_eval_csv_spotchecks(path: Path) -> None:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    print(f"\n=== {path.relative_to(ROOT)} spot-checks ===")
    print(f"n={len(rows)}")
    games = [float(r["projected_games"]) for r in rows if r.get("projected_games")]
    volume = [
        float(r["projected_volume_games"])
        for r in rows
        if r.get("projected_volume_games")
    ]
    print(
        f"projected_games (Gate A column in this CSV): "
        f"min={min(games):.2f} max={max(games):.2f} never_17={sum(abs(g - 17) > 0.01 for g in games)}"
    )
    print(
        f"projected_volume_games: min={min(volume):.1f} max={max(volume):.1f} "
        f"n={len(volume)}"
    )
    print("QB predicted top 12 (name, pred, actual, actual_finish):")
    qbs = [r for r in rows if r["preseason_position"] == "QB"]
    for row in sorted(qbs, key=lambda r: -float(r["model_points_end_to_end"]))[:12]:
        print(
            f"  {row['display_name']:22} pred={float(row['model_points_end_to_end']):7.1f} "
            f"actual={float(row['actual_points']):7.1f} "
            f"finish={row['actual_position_finish']}"
        )


def main() -> None:
    print("RESEARCH-ONLY dump. No production artifacts were written.")
    _print_eval_table(EVAL_JSON)
    _print_rate_holdout(HOLDOUT)
    _print_accuracy(ACCURACY, WEIGHTS)
    _print_concentration(CONC)
    _print_eval_csv_spotchecks(EVAL_CSV)


if __name__ == "__main__":
    main()
