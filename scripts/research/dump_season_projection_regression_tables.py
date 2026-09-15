#!/usr/bin/env python3
"""Research-only: print the season-projection tables cited in

    docs/research/SEASON_PROJECTION_REGRESSION_REEXPLORE_2026-09-14.md
    docs/research/SEASON_PROJECTION_POSITION_MISS_BRIEF_2026-09-15.md

Does not train, predict, compose, seal, or promote. Reads committed
artifacts under ``output/`` and ``models/`` with the stdlib only.

Those artifacts must be present (and, in a fresh clone, committed) or the
script exits with a missing-file list. It does not download or regenerate
them.

Usage (from repo root):

    python3 scripts/research/dump_season_projection_regression_tables.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL_JSON = ROOT / "output" / "fantasy_evaluation_summary_2025.json"
EVAL_JSON_2023 = ROOT / "output" / "fantasy_evaluation_summary_2023.json"
EVAL_JSON_2024 = ROOT / "output" / "fantasy_evaluation_summary_2024.json"
EVAL_CSV = ROOT / "output" / "fantasy_evaluation_2025.csv"
HOLDOUT = ROOT / "output" / "backtest" / "veteran_holdout_2025.csv"
AVAIL = ROOT / "output" / "backtest" / "availability_rolling.csv"
ACCURACY = ROOT / "output" / "accuracy_first_2026" / "report.json"
WEIGHTS = ROOT / "output" / "accuracy_first_2026" / "ensemble_weights.json"
POINTS_2026 = ROOT / "output" / "accuracy_first_2026" / "fantasy_points_2026.csv"
CONC = ROOT / "models" / "concentration_calibration.json"
REQUIRED_ARTIFACTS = (
    EVAL_JSON,
    EVAL_JSON_2023,
    EVAL_JSON_2024,
    EVAL_CSV,
    HOLDOUT,
    AVAIL,
    ACCURACY,
    WEIGHTS,
    POINTS_2026,
    CONC,
)
TIER_K = {"QB": 12, "RB": 24, "WR": 36, "TE": 12}
REPL_K = {"QB": 13, "RB": 25, "WR": 37, "TE": 13}
SCOPES = ("all_eligible", "starter_8plus_games")
METHODS = ("model", "carry_forward", "availability_adjusted")


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _require_artifacts() -> None:
    missing = [path for path in REQUIRED_ARTIFACTS if not path.exists()]
    if missing:
        listing = "\n  ".join(_rel(path) for path in missing)
        raise SystemExit(f"missing artifacts:\n  {listing}")


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


def _f(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _print_fold_scope_table(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    year = data["metadata"].get("target_season")
    print(f"\n=== {path.relative_to(ROOT)} scopes (cited in the 2026-09-15 brief) ===")
    print("year pos scope method n spearman rate_spearman mae tier vorp pred_repl actual_repl")
    for row in data["metrics"]:
        if row.get("scope") not in SCOPES or row.get("method") not in METHODS:
            continue
        rate = row.get("rate_spearman")
        rate_s = f"{rate:.3f}" if isinstance(rate, (int, float)) else "-"
        print(
            f"{year} {row['position']:3} {row['scope']:22} {row['method']:22} "
            f"n={row['n']:3} ρ={row['spearman']:.3f} rateρ={rate_s:>6} "
            f"MAE={row['points_mae']:.1f} tier={row['tier_hits']}/{row['tier_rank']} "
            f"VORP={row['vorp_mae']:.1f} predR={row['predicted_replacement_points']:.1f} "
            f"actR={row['actual_replacement_points']:.1f}"
        )


def _print_availability_2025(path: Path) -> None:
    print(f"\n=== {path.relative_to(ROOT)} 2025 target_roster_eligible ===")
    print("position n_test model_mae naive_mae model_wins")
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("test_season") != "2025":
                continue
            if row.get("scope") != "target_roster_eligible":
                continue
            print(
                f"{row['position']:3} n={row['n_test']} "
                f"model={float(row['model_mae']):.3f} "
                f"naive={float(row['naive_mae']):.3f} "
                f"wins={row['model_wins']}"
            )


def _print_position_misses(path: Path) -> None:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    print(f"\n=== {path.relative_to(ROOT)} per-position misses (v1 native 2025) ===")
    print("Season scoreboard is model_points_end_to_end vs actual_points.")
    for pos in ("QB", "RB", "WR", "TE"):
        k = TIER_K[pos]
        rnk = REPL_K[pos]
        group = [x for x in rows if x.get("preseason_position") == pos]
        for item in group:
            item["_pred"] = _f(item.get("model_points_end_to_end")) or 0.0
            item["_act"] = _f(item.get("actual_points")) or 0.0
            item["_games"] = _f(item.get("actual_games_played")) or 0.0
            item["_pred_rank"] = _f(item.get("model_position_rank")) or 0.0
            item["_act_rank"] = _f(item.get("actual_position_finish")) or 0.0
            item["_rookie"] = item.get("is_rookie") == "True"
            item["_depth"] = item.get("depth_tier") or ""
            item["_gate"] = _f(item.get("projected_games"))
        pred_sorted = sorted(group, key=lambda x: -x["_pred"])
        act_sorted = sorted(group, key=lambda x: -x["_act"])
        pred_top = pred_sorted[:k]
        act_top = act_sorted[:k]
        pred_ids = {x["player_id"] for x in pred_top}
        act_ids = {x["player_id"] for x in act_top}
        hits = pred_ids & act_ids
        false_neg = [x for x in act_top if x["player_id"] not in pred_ids]
        false_pos = [x for x in pred_top if x["player_id"] not in act_ids]
        repl_pred = pred_sorted[rnk - 1] if len(pred_sorted) >= rnk else None
        repl_act = act_sorted[rnk - 1] if len(act_sorted) >= rnk else None
        print(f"\n-- {pos} tier {len(hits)}/{k} --")
        if repl_pred and repl_act:
            print(
                f"replacement {rnk}th: pred {repl_pred['display_name']} "
                f"{repl_pred['_pred']:.1f}  actual {repl_act['display_name']} "
                f"{repl_act['_act']:.1f}"
            )
        print("predicted top K (HIT/MISS name pred actual gp prank arank depth rook)")
        for item in pred_top:
            tag = "HIT" if item["player_id"] in act_ids else "MISS"
            print(
                f"  {tag:4} {item['display_name']:22} "
                f"pred={item['_pred']:6.1f} act={item['_act']:6.1f} "
                f"gp={item['_games']:4.0f} prank={item['_pred_rank']:4.0f} "
                f"arank={item['_act_rank']:4.0f} d={item['_depth'] or '-'} "
                f"rook={item['_rookie']}"
            )
        print("false negatives (actual top K, not in predicted top K):")
        for item in sorted(false_neg, key=lambda x: x["_act_rank"]):
            print(
                f"  {item['display_name']:22} pred={item['_pred']:6.1f} "
                f"act={item['_act']:6.1f} gp={item['_games']:4.0f} "
                f"prank={item['_pred_rank']:4.0f} arank={item['_act_rank']:4.0f} "
                f"d={item['_depth'] or '-'} rook={item['_rookie']}"
            )
        print("false positives (predicted top K, not in actual top K):")
        for item in sorted(false_pos, key=lambda x: x["_pred_rank"]):
            print(
                f"  {item['display_name']:22} pred={item['_pred']:6.1f} "
                f"act={item['_act']:6.1f} gp={item['_games']:4.0f} "
                f"prank={item['_pred_rank']:4.0f} arank={item['_act_rank']:4.0f} "
                f"d={item['_depth'] or '-'} rook={item['_rookie']}"
            )
        low = [x for x in pred_top if x["_games"] <= 10]
        full = [x for x in pred_top if x["_games"] >= 15]
        if low:
            low_err = sum(x["_pred"] - x["_act"] for x in low) / len(low)
            full_err = (
                sum(x["_pred"] - x["_act"] for x in full) / len(full) if full else float("nan")
            )
            print(
                f"pred-top gp<=10 n={len(low)} mean err={low_err:+.1f}  "
                f"gp>=15 n={len(full)} mean err={full_err:+.1f}"
            )


def _print_2026_component_disagreement(path: Path) -> None:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    print(f"\n=== {path.relative_to(ROOT)} 2026 component disagreement ===")
    print("NOT 2026 outcomes. Market vs v1/v2/ensemble on the sealed point board.")
    for pos in ("QB", "RB", "WR", "TE"):
        parsed = []
        for row in rows:
            if row.get("position") != pos:
                continue
            adp = _f(row.get("adp"))
            if adp is None:
                continue
            parsed.append(
                {
                    "name": row["display_name"],
                    "adp": adp,
                    "inc": _f(row.get("incumbent_pred")) or 0.0,
                    "v2": _f(row.get("v2_pred")) or 0.0,
                    "adp_pts": _f(row.get("adp_points")) or 0.0,
                    "ens": _f(row.get("accuracy_ensemble_pred")) or 0.0,
                    "applied": row.get("accuracy_ensemble_applied"),
                    "arm": row.get("accuracy_ensemble_arm"),
                }
            )
        parsed.sort(key=lambda item: item["adp"])
        print(f"\n-- {pos} top-12 ADP --")
        print("ADP name incumbent v2 adp_pts ensemble applied arm ens-inc")
        for item in parsed[:12]:
            print(
                f"  {item['adp']:5.1f} {item['name']:22} "
                f"inc={item['inc']:6.1f} v2={item['v2']:6.1f} "
                f"adp_pts={item['adp_pts']:6.1f} ens={item['ens']:6.1f} "
                f"applied={item['applied']} arm={item['arm']} "
                f"Δ={item['ens'] - item['inc']:+6.1f}"
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
    _require_artifacts()
    print("RESEARCH-ONLY dump. No production artifacts were written.")
    _print_eval_table(EVAL_JSON)
    for fold in (EVAL_JSON_2023, EVAL_JSON_2024, EVAL_JSON):
        _print_fold_scope_table(fold)
    _print_rate_holdout(HOLDOUT)
    _print_availability_2025(AVAIL)
    _print_accuracy(ACCURACY, WEIGHTS)
    _print_concentration(CONC)
    _print_eval_csv_spotchecks(EVAL_CSV)
    _print_position_misses(EVAL_CSV)
    _print_2026_component_disagreement(POINTS_2026)


if __name__ == "__main__":
    main()
