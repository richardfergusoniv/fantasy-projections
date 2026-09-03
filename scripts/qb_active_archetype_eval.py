#!/usr/bin/env python3
"""Active-start × archetype QB experiment (rolling-origin, predeclared gates).

Does not modify the sealed release, active pointer, or production compose defaults.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.projection.qb_active_archetype.active_rates import (
    append_eval_season_active,
    build_active_season_rates,
    load_weekly_qb,
    merge_rush_splits,
    player_decomposition,
)
from src.projection.qb_active_archetype.apply import compose_candidate
from src.projection.qb_active_archetype.archetypes import classify_archetype, hierarchical_rush_priors
from src.projection.qb_active_archetype.evaluate import decide, evaluate_season
from src.projection.qb_active_archetype.thresholds import EVAL_SEASONS, thresholds_dict
from src.projection.qb_repair.apply_board import score_long_to_fantasy

OUT = ROOT / "output" / "qb_active_archetype"
LAMAR = "00-0034796"
BURROW = "00-0036442"
FOCUS = {
    "Josh Allen": "00-0034857",
    "Lamar Jackson": LAMAR,
    "Jayden Daniels": "00-0039910",
    "Jalen Hurts": "00-0036389",
    "Joe Burrow": BURROW,
    "Patrick Mahomes": "00-0033873",
}


def _dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def build_history() -> pd.DataFrame:
    weekly = load_weekly_qb()
    active = build_active_season_rates(weekly)
    active = merge_rush_splits(active)
    # 2025 approx from fantasy evaluation when weekly missing
    active = append_eval_season_active(active, 2025)
    active = merge_rush_splits(active)  # refresh designed splits for 2025
    if "provenance" not in active.columns:
        active["provenance"] = "weekly"
    active.loc[active["provenance"].isna(), "provenance"] = "weekly"
    return active


def stage_table_for_player(raw, baseline_board, candidate_board, fantasy_base, fantasy_cand, pid: str) -> dict:
    def rates(board):
        sub = board[(board.player_id.astype(str) == pid)]
        out = {}
        for _, r in sub.iterrows():
            out[str(r.stat)] = {
                "pred_pg": float(r.pred_pg),
                "team_volume_scale": float(r.team_volume_scale) if "team_volume_scale" in r.index and pd.notna(r.team_volume_scale) else None,
            }
        return out

    raw_r = rates(raw[raw.player_id.astype(str) == pid] if "player_id" in raw.columns else raw)
    # raw may not have team_volume_scale
    raw_rates = {
        str(r.stat): float(r.pred_pg)
        for _, r in raw[raw.player_id.astype(str) == pid].iterrows()
    }
    fb = fantasy_base[fantasy_base.player_id.astype(str) == pid]
    fc = fantasy_cand[fantasy_cand.player_id.astype(str) == pid]
    return {
        "raw": raw_rates,
        "baseline_composed": rates(baseline_board),
        "candidate_composed": rates(candidate_board),
        "baseline_fantasy": None
        if fb.empty
        else {
            "rank": int(fb.iloc[0]["rank"]),
            "ppg": float(fb.iloc[0]["fantasy_pts"]),
            "season": float(fb.iloc[0]["fantasy_pts_season"]),
        },
        "candidate_fantasy": None
        if fc.empty
        else {
            "rank": int(fc.iloc[0]["rank"]),
            "ppg": float(fc.iloc[0]["fantasy_pts"]),
            "season": float(fc.iloc[0]["fantasy_pts_season"]),
        },
    }


def burrow_attribution(history, compose_result) -> dict:
    decomp = player_decomposition(history, player_id=BURROW, seasons=(2022, 2023, 2024, 2025))
    audit = compose_result["rewrite_audit"]["players"].get(BURROW, {})
    cand = compose_result["candidate_fantasy"]
    row = cand[cand.player_id.astype(str) == BURROW]
    base = compose_result["baseline_fantasy"]
    brow = base[base.player_id.astype(str) == BURROW]
    board = compose_result["candidate_board"]
    att = board[(board.player_id == BURROW) & (board.stat == "attempts")]
    return {
        "historical_decomposition": decomp,
        "2026_rewrite": audit,
        "baseline": None
        if brow.empty
        else {
            "rank": int(brow.iloc[0]["rank"]),
            "ppg": float(brow.iloc[0]["fantasy_pts"]),
            "attempts": float(brow.iloc[0]["attempts"]) if "attempts" in brow.columns else None,
        },
        "candidate": None
        if row.empty
        else {
            "rank": int(row.iloc[0]["rank"]),
            "ppg": float(row.iloc[0]["fantasy_pts"]),
            "attempts": float(row.iloc[0]["attempts"]) if "attempts" in row.columns else None,
        },
        "candidate_attempts_pg": float(att.pred_pg.iloc[0]) if not att.empty else None,
        "expected_active_starts": compose_result["rewrite_audit"]["expected_active_starts"].get(BURROW),
        "diagnosis": (
            "Shortened 2023/2025 seasons cut active_starts (availability), while "
            "attempts_per_active stayed near healthy starter levels (~36–38). "
            "Low sealed output is driven by team-volume/backup squeeze on conflated "
            "rates and full-season backup floors — not by a collapsed active-start "
            "attempt rate. Candidate rewrites active rate + expected starts separately."
        ),
    }


def lamar_rush_stages(history, compose_result) -> dict:
    decomp = player_decomposition(history, player_id=LAMAR, seasons=(2022, 2023, 2024, 2025))
    rush = hierarchical_rush_priors(history, player_id=LAMAR, target_season=2026)
    arch = classify_archetype(history, player_id=LAMAR, target_season=2026)
    board_b = compose_result["baseline_board"]
    board_c = compose_result["candidate_board"]
    raw = pd.read_csv(ROOT / "output" / "projections_2026_raw.csv")

    def rush_slice(board, label):
        sub = board[board.player_id.astype(str) == LAMAR]
        return {
            label: {
                str(r.stat): float(r.pred_pg)
                for _, r in sub.iterrows()
                if r.stat in ("carries", "rushing_yards", "rushing_tds", "attempts")
            }
        }

    return {
        "historical_decomposition": decomp,
        "archetype_2026": arch,
        "hierarchical_priors_2026": rush,
        "stages": {
            "raw": {
                str(r.stat): float(r.pred_pg)
                for _, r in raw[raw.player_id == LAMAR].iterrows()
                if r.stat in ("carries", "rushing_yards", "rushing_tds")
            },
            **rush_slice(board_b, "baseline_composed"),
            **rush_slice(board_c, "candidate_composed"),
            "designed_scramble_prior": {
                "designed_carries_per_active": rush["priors"].get("designed_carries_per_active"),
                "scramble_per_dropback": rush["priors"].get("scramble_per_dropback"),
                "carries_per_active": rush["priors"].get("carries_per_active"),
                "rushing_yards_per_active": rush["priors"].get("rushing_yards_per_active"),
                "rushing_tds_per_active": rush["priors"].get("rushing_tds_per_active"),
            },
        },
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    # Freeze thresholds artifact BEFORE fold metrics are written alongside.
    _dump(OUT / "predeclared_thresholds.json", thresholds_dict())

    history = build_history()
    history.to_parquet(OUT / "active_season_rates.parquet", index=False)

    folds = []
    for season in EVAL_SEASONS:
        metrics = evaluate_season(history, season)
        # Drop bulky row dump from summary; keep separately
        rows = metrics.pop("rows", [])
        folds.append(metrics)
        _dump(OUT / f"fold_{season}_rows.json", {"season": season, "rows": rows})
    _dump(OUT / "fold_metrics.json", {"folds": folds})

    decision = decide(folds)
    # 2026 diagnostic (not used for selection)
    raw = pd.read_csv(ROOT / "output" / "projections_2026_raw.csv")
    compose_result = compose_candidate(raw, history, target_season=2026)
    sanity = {}
    for name, pid in FOCUS.items():
        sanity[name] = stage_table_for_player(
            raw,
            compose_result["baseline_board"],
            compose_result["candidate_board"],
            compose_result["baseline_fantasy"],
            compose_result["candidate_fantasy"],
            pid,
        )
    # QB12 on candidate board
    cf = compose_result["candidate_fantasy"].sort_values("rank").reset_index(drop=True)
    if len(cf) >= 12:
        row12 = cf.iloc[11].to_dict()
        sanity["QB12_candidate"] = {
            "player_id": str(row12["player_id"]),
            "display_name": row12.get("display_name"),
            "rank": int(row12["rank"]),
            "ppg": float(row12["fantasy_pts"]),
        }

    _dump(
        OUT / "sanity_2026.json",
        {
            "note": "Diagnostic only; excluded from selection by predeclared gate",
            "non_qb_invariance": compose_result["non_qb_invariance"],
            "players": sanity,
            "burrow": burrow_attribution(history, compose_result),
            "lamar": lamar_rush_stages(history, compose_result),
            "allocation_report_summary": {
                "mode": compose_result["allocation_report"].get("mode"),
                "teams": compose_result["allocation_report"].get("teams"),
                "n_conservation_flags": len(
                    compose_result["allocation_report"].get("conservation_violations") or []
                ),
            },
        },
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "folds": [
            {
                "season": f["season"],
                "n": f["n"],
                "cohorts": f.get("cohorts"),
                "primary_bootstrap_delta_mae": f.get("primary_bootstrap_delta_mae"),
            }
            for f in folds
        ],
        "non_qb_invariance": compose_result["non_qb_invariance"],
    }
    _dump(OUT / "selection_decision.json", payload)
    print("verdict", decision["verdict"])
    print("reasons", decision["reasons"])
    print("gates", decision["gates"])
    if decision["verdict"] == "NO-GO":
        print("next_hypothesis", decision["next_falsifiable_hypothesis"])
    print("non_qb", compose_result["non_qb_invariance"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
