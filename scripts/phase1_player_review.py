#!/usr/bin/env python3
"""Player-facing delta review for Phase 1 release candidate sign-off."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.projection.evaluation.accuracy_first import TOP_ADP, sha256_file
from src.projection.inference.recenter import board_points_series
from src.projection.release_bundle import (
    bundle_root,
    load_sealed_manifest,
    player_id_set_hash,
    public_release_dir,
    selected_points_vector_hash,
)

FINISH_COLS = [
    "p_finish_top6",
    "p_finish_top12",
    "p_finish_top24",
    "p_finish_top36",
    "p_finish_top48",
]
SIM_VORP_COLS = ["sim_vorp_p10", "sim_vorp_p50", "sim_vorp_p90", "p_vorp_positive"]
RANK_COLS = ["expected_pos_rank", "median_pos_rank", "overall_rank", "pos_rank"]


def _idx(players_doc: dict) -> dict[str, dict]:
    return {str(p["player_id"]): p for p in players_doc.get("players") or []}


def verify_hash_alignment(season: int, namespace: str) -> dict:
    root = bundle_root(season, namespace)
    manifest, digest = load_sealed_manifest(root)
    board_hash = manifest["board"]["selected_board_file_hash"]
    points_hash = manifest["board"]["selected_points_vector_hash"]
    overlay_hash = manifest["overlay"]["simulated_player_population_hash"]

    board_path = root / f"fantasy_points_{season}.csv"
    board_df = pd.read_csv(board_path)
    recomputed_board = sha256_file(board_path)
    recomputed_points = selected_points_vector_hash(board_points_series(board_df))
    overlay_ids = [str(row["player_id"]) for row in json.loads((root / f"players_{season}.json").read_text())["players"]]
    recomputed_overlay = player_id_set_hash(overlay_ids)

    sim_manifest = json.loads((root / f"simulation_manifest_{season}.json").read_text())
    release_report = json.loads((root / f"release_report_{season}.json").read_text())
    public_manifest_path = public_release_dir(namespace) / "release_bundle_manifest.json"
    public_digest = sha256_file(public_manifest_path) if public_manifest_path.is_file() else None

    refs = {
        "manifest_sha256": digest,
        "manifest_board_hash": board_hash,
        "manifest_points_hash": points_hash,
        "manifest_overlay_hash": overlay_hash,
        "recomputed_board_hash": recomputed_board,
        "recomputed_points_hash": recomputed_points,
        "recomputed_overlay_hash": recomputed_overlay,
        "simulation_manifest_board_hash": sim_manifest.get("selected_board_hash"),
        "release_report_board_hash": release_report.get("simulation", {}).get("provenance", {}).get("selected_board_hash"),
        "public_manifest_sha256": public_digest,
    }
    checks = [
        ("board_file_hash", board_hash == recomputed_board),
        ("points_vector_hash", points_hash == recomputed_points),
        ("overlay_population_hash", overlay_hash == recomputed_overlay),
        ("simulation_manifest_board", board_hash == sim_manifest.get("selected_board_hash")),
        ("release_report_board", board_hash == release_report.get("simulation", {}).get("provenance", {}).get("selected_board_hash")),
        ("public_manifest", digest == public_digest),
    ]
    return {
        "refs": refs,
        "checks": [{"check": name, "passed": passed} for name, passed in checks],
        "all_passed": all(p for _, p in checks),
    }


def eligible_rb_wr_ids(contract: dict, board_df: pd.DataFrame) -> set[str]:
    ids = set(str(x) for x in contract.get("eligibility_ids") or [])
    board_df = board_df.copy()
    board_df["player_id"] = board_df["player_id"].astype(str)
    rb_wr = board_df[board_df["position"].isin(["RB", "WR"])]
    if ids:
        return set(rb_wr["player_id"]) & ids
    if "adp" in board_df.columns:
        return set(rb_wr.loc[pd.to_numeric(rb_wr["adp"], errors="coerce").le(TOP_ADP), "player_id"])
    return set(rb_wr["player_id"])


def review_player_deltas(
    *,
    cand_players: dict,
    legacy_players: dict,
    board_df: pd.DataFrame,
    contract: dict,
) -> dict:
    eligible = eligible_rb_wr_ids(contract, board_df)
    rows = []
    for pid in sorted(eligible):
        c = cand_players.get(pid)
        l = legacy_players.get(pid)
        if not c or not l:
            continue
        rows.append(
            {
                "player_id": pid,
                "display_name": c.get("display_name"),
                "position": c.get("position"),
                "treatment": board_df.loc[board_df["player_id"].astype(str).eq(pid), "contract_treatment"].iloc[0]
                if "contract_treatment" in board_df.columns and board_df["player_id"].astype(str).eq(pid).any()
                else None,
                "legacy_pts": l.get("fantasy_pts"),
                "candidate_pts": c.get("fantasy_pts"),
                "delta_pts": round(float(c.get("fantasy_pts", 0)) - float(l.get("fantasy_pts", 0)), 3),
                "legacy_vorp": l.get("vorp"),
                "candidate_vorp": c.get("vorp"),
                "delta_vorp": round(float(c.get("vorp", 0)) - float(l.get("vorp", 0)), 2),
                "legacy_rank": l.get("overall_rank"),
                "candidate_rank": c.get("overall_rank"),
                "delta_rank": int(l.get("overall_rank", 0)) - int(c.get("overall_rank", 0)),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return {"eligible_count": 0}
    by_pos = {}
    for pos in ("RB", "WR"):
        sub = df[df["position"] == pos]
        by_pos[pos] = {
            "count": int(len(sub)),
            "mean_delta_pts": round(float(sub["delta_pts"].mean()), 3),
            "median_delta_pts": round(float(sub["delta_pts"].median()), 3),
            "max_abs_delta_pts": round(float(sub["delta_pts"].abs().max()), 3),
            "mean_delta_vorp": round(float(sub["delta_vorp"].mean()), 2),
            "rank_moves_gt_10": int((sub["delta_rank"].abs() > 10).sum()),
        }
    top_moves = df.reindex(df["delta_pts"].abs().sort_values(ascending=False).index).head(15)
    return {
        "eligible_count": int(len(df)),
        "by_position": by_pos,
        "largest_point_moves": top_moves.to_dict(orient="records"),
        "selected_treatment_count": int((df["treatment"] == "selected").sum()) if "treatment" in df.columns else None,
    }


def review_propagated_outputs(cand_players: dict, legacy_players: dict) -> dict:
  common = set(cand_players) & set(legacy_players)
  tier_moves = 0
  replacement_shifts = []
  rank_shifts = []
  for pid in common:
      c, l = cand_players[pid], legacy_players[pid]
      if c.get("overall_tier") != l.get("overall_tier"):
          tier_moves += 1
      if c.get("replacement_pts") is not None and l.get("replacement_pts") is not None:
          replacement_shifts.append(float(c["replacement_pts"]) - float(l["replacement_pts"]))
      if c.get("overall_rank") and l.get("overall_rank"):
          rank_shifts.append(int(l["overall_rank"]) - int(c["overall_rank"]))
  return {
      "players_compared": len(common),
      "tier_changes": tier_moves,
      "mean_replacement_shift": round(float(pd.Series(replacement_shifts).mean()), 3) if replacement_shifts else 0.0,
      "rank_moves_mean": round(float(pd.Series(rank_shifts).mean()), 2) if rank_shifts else 0.0,
      "rank_moves_abs_gt_20": sum(1 for x in rank_shifts if abs(x) > 20),
  }


def spot_check_fields(cand_players: dict, board_df: pd.DataFrame) -> dict:
    players = list(cand_players.values())
    finish = {col: sum(1 for p in players if p.get(col) is not None) for col in FINISH_COLS}
    sim_vorp = {col: sum(1 for p in players if p.get(col) is not None) for col in SIM_VORP_COLS}
    rank = {col: sum(1 for p in players if p.get(col) is not None) for col in RANK_COLS}

    board_df = board_df.copy()
    board_df["player_id"] = board_df["player_id"].astype(str)
    treatments = board_df.set_index("player_id")["contract_treatment"].to_dict() if "contract_treatment" in board_df.columns else {}

    by_treatment: dict[str, dict] = {}
    for treatment in sorted(set(treatments.values()) | {"unknown"}):
        ids = [pid for pid, t in treatments.items() if t == treatment] if treatment != "unknown" else []
        subset = [cand_players[pid] for pid in ids if pid in cand_players]
        if not subset:
            continue
        by_treatment[treatment] = {
            "count": len(subset),
            "finish_populated": {col: sum(1 for p in subset if p.get(col) is not None) for col in FINISH_COLS[:2]},
            "sim_vorp_populated": {col: sum(1 for p in subset if p.get(col) is not None) for col in SIM_VORP_COLS},
            "rank_populated": {col: sum(1 for p in subset if p.get(col) is not None) for col in RANK_COLS},
            "sample": {
                "display_name": subset[0].get("display_name"),
                "position": subset[0].get("position"),
                "p_finish_top6": subset[0].get("p_finish_top6"),
                "sim_vorp_p50": subset[0].get("sim_vorp_p50"),
                "median_pos_rank": subset[0].get("median_pos_rank"),
            },
        }

    monotonic_finish_violations = 0
    for p in players:
        vals = [p.get(c) for c in FINISH_COLS if p.get(c) is not None]
        # Wider cutoffs subsume narrower ones: P(top6) <= P(top12) <= ...
        if len(vals) >= 2 and any(vals[i] > vals[i + 1] for i in range(len(vals) - 1)):
            monotonic_finish_violations += 1

    return {
        "finish_field_population": finish,
        "sim_vorp_field_population": sim_vorp,
        "rank_moment_population": rank,
        "finish_monotonic_violations": monotonic_finish_violations,
        "by_treatment": by_treatment,
    }


def run_review(season: int, namespace: str, legacy_players_path: Path) -> dict:
    root = bundle_root(season, namespace)
    manifest, digest = load_sealed_manifest(root)
    board_df = pd.read_csv(root / f"fantasy_points_{season}.csv")
    contract = json.loads((root / "application_contract.json").read_text())
    cand_doc = json.loads((root / f"players_{season}.json").read_text())
    legacy_doc = json.loads(legacy_players_path.read_text())
    cand_players = _idx(cand_doc)
    legacy_players = _idx(legacy_doc)

    hash_review = verify_hash_alignment(season, namespace)
    delta_review = review_player_deltas(
        cand_players=cand_players,
        legacy_players=legacy_players,
        board_df=board_df,
        contract=contract,
    )
    propagated = review_propagated_outputs(cand_players, legacy_players)
    spot_check = spot_check_fields(cand_players, board_df)

    plausibility_flags = []
    if not hash_review["all_passed"]:
        plausibility_flags.append("hash_alignment_failed")
    if delta_review.get("by_position", {}).get("WR", {}).get("max_abs_delta_pts", 0) > 10:
        plausibility_flags.append("wr_large_point_move")
    if spot_check["finish_monotonic_violations"] > 0:
        plausibility_flags.append("finish_probability_not_monotonic")
    if spot_check["finish_field_population"].get("p_finish_top6", 0) < len(cand_players) * 0.9:
        plausibility_flags.append("finish_probs_underpopulated")
    if spot_check["sim_vorp_field_population"].get("sim_vorp_p50", 0) < len(cand_players) * 0.9:
        plausibility_flags.append("sim_vorp_underpopulated")

    verdict = "approve" if not plausibility_flags else "review_required"
    return {
        "namespace": namespace,
        "manifest_sha256": digest,
        "release_id": manifest["bundle"]["release_id"],
        "hash_alignment": hash_review,
        "eligible_rb_wr_deltas": delta_review,
        "propagated_outputs": propagated,
        "spot_check": spot_check,
        "plausibility_flags": plausibility_flags,
        "sign_off_verdict": verdict,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--artifact-namespace", default="phase1_rehearsal_20260829")
    parser.add_argument(
        "--legacy-players",
        type=Path,
        default=Path("draft_assistant/data/players_2026.json"),
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    report = run_review(args.season, args.artifact_namespace, args.legacy_players)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    return 0 if report["sign_off_verdict"] == "approve" else 1


if __name__ == "__main__":
    raise SystemExit(main())
