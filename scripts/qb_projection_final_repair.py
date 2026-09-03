#!/usr/bin/env python3
"""QB projection final repair: attribution, arms, evaluation, optional candidate.

Does not overwrite v2_baseline_20260830, does not promote the active pointer,
and does not deploy.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.projection.composition import shipped_context
from src.projection.qb_repair.apply_board import (
    apply_arm_to_2026,
    compare_to_sealed,
    non_qb_invariance_check,
    team_passing_conservation,
)
from src.projection.qb_repair.arms import ALL_ARMS, ARM_BASELINE, run_arm
from src.projection.qb_repair.gates import run_selection_pipeline
from src.projection.qb_repair.history import load_qb_season_history
from src.projection.qb_repair.provenance import explain_mobile_rush_provenance
from src.projection.qb_repair.rate_eval import evaluate_rate_prior_components
from src.projection.qb_repair.stage_attribution import build_stage_attribution

OUT = ROOT / "output" / "qb_repair"


def _json_default(obj):
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def run_attribution() -> pd.DataFrame:
    table = build_stage_attribution()
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "qb1_stage_attribution.csv", index=False)
    return table


def run_2026_arms() -> dict:
    raw = pd.read_csv(ROOT / "output" / "projections_2026_raw.csv")
    for col in (
        "pred_season",
        "pred_season_low",
        "pred_season_high",
        "team_volume_scale",
        "td_rate_clip_applied",
    ):
        if col in raw.columns:
            raw = raw.drop(columns=[col])
    ctx = shipped_context(conn=None, target_season=2026)
    baseline = run_arm(raw, ctx, ARM_BASELINE, target_season=2026)
    results = {}
    for arm in ALL_ARMS:
        applied = apply_arm_to_2026(arm)
        inv = non_qb_invariance_check(
            baseline_long=baseline.board,
            candidate_long=applied["board"],
        )
        cons = team_passing_conservation(applied["board"])
        cmp_ = compare_to_sealed(applied["fantasy"])
        results[arm] = {
            "non_qb_invariance": inv,
            "conservation": cons,
            "comparison": cmp_,
            "provenance_keys": sorted(applied["provenance"].keys()),
        }
        applied["fantasy"].to_csv(OUT / f"fantasy_qb_{arm}_2026.csv", index=False)
        write_json(OUT / f"arm_provenance_{arm}.json", applied["provenance"])
    write_json(OUT / "arms_2026_summary.json", results)
    return results


def maybe_publish_candidate(decision: dict, selected_arm: str) -> dict | None:
    """Create an immutable candidate namespace only if gates pass.

    Full simulation/VORP regeneration requires the projections DB and is
    recorded as blocked when unavailable. Component + fantasy boards are still
    written under output/qb_repair/candidates/<namespace>/ for review.
    """
    if decision.get("verdict") != "GO":
        return {
            "published": False,
            "reason": "gates_returned_NO-GO",
            "verdict": decision.get("verdict"),
        }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    namespace = f"qb_repair_candidate_{stamp}"
    cand = OUT / "candidates" / namespace
    cand.mkdir(parents=True, exist_ok=True)
    applied = apply_arm_to_2026(selected_arm)
    # Merge repaired QB rates into sealed long projections for non-QB invariance.
    sealed_proj = pd.read_csv(
        ROOT / "draft_assistant" / "data" / "releases" / "v2_baseline_20260830" / "projections_2026.csv"
    )
    repaired = applied["board"]
    qb_mask = sealed_proj["position"].astype(str).eq("QB")
    non_qb = sealed_proj[~qb_mask].copy()
    qb_new = repaired[repaired["position"].astype(str).eq("QB")].copy()
    # Align columns
    for col in non_qb.columns:
        if col not in qb_new.columns:
            qb_new[col] = pd.NA
    qb_new = qb_new[non_qb.columns]
    merged = pd.concat([non_qb, qb_new], ignore_index=True)
    inv = non_qb_invariance_check(baseline_long=sealed_proj, candidate_long=merged)
    merged.to_csv(cand / "projections_2026.csv", index=False)
    applied["fantasy"].to_csv(cand / "fantasy_points_qb_only_2026.csv", index=False)
    write_json(
        cand / "candidate_manifest.json",
        {
            "namespace": namespace,
            "parent_baseline": "v2_baseline_20260830",
            "selected_arm": selected_arm,
            "decision": decision,
            "non_qb_invariance": inv,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "promotion": "NOT_PROMOTED",
            "rollback": "Leave active_release_2026.json pointing at v2_baseline_20260830; delete this candidate namespace if discarded.",
            "blocked_artifacts": [
                "players_2026.json",
                "simulations",
                "VORP",
                "full release_bundle_manifest",
            ],
            "blocked_reason": "projections.db unavailable in this environment; full seal requires DB-backed prepare/simulate",
        },
    )
    return {
        "published": True,
        "namespace": namespace,
        "path": str(cand),
        "non_qb_invariance": inv,
        "promotion_recommendation": "DO_NOT_PROMOTE until full seal artifacts are regenerated with DB",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-arms", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    load_qb_season_history(refresh=True)

    print("== stage attribution ==")
    attr = run_attribution()
    print(f"wrote qb1_stage_attribution.csv n={len(attr)}")

    print("== Lamar rush provenance ==")
    prov = explain_mobile_rush_provenance()
    write_json(OUT / "lamar_rush_provenance.json", prov)
    print(prov.get("verdict"), prov.get("causes", [])[:2])

    arms_summary = {}
    if not args.skip_arms:
        print("== 2026 experimental arms ==")
        arms_summary = run_2026_arms()
        for arm, payload in arms_summary.items():
            inv = payload["non_qb_invariance"]
            print(f"  {arm}: non_qb_pass={inv.get('pass')} top15={payload['comparison']['top15_new'][:3]}")

    decision_bundle = {}
    if not args.skip_eval:
        print("== rolling-origin selection ==")
        decision_bundle = run_selection_pipeline(list(ALL_ARMS) + ["carry_forward", "simple_multi_season_rate"])
        rate_eval = evaluate_rate_prior_components(holdout_season=2025)
        decision_bundle["component_rate_eval_2025"] = rate_eval
        write_json(OUT / "selection_decision.json", decision_bundle)
        print("selected", decision_bundle["fit"]["selected_arm"])
        print("verdict", decision_bundle["decision"]["verdict"], decision_bundle["decision"]["reasons"])
        print("component_rate_eval n", rate_eval.get("n_player_components"))

        pub = maybe_publish_candidate(
            decision_bundle["decision"],
            decision_bundle["fit"]["selected_arm"],
        )
        write_json(OUT / "candidate_publish.json", pub)
        print("publish", pub)

    # Sanity snapshot for selected or baseline arm
    arm = (decision_bundle.get("fit") or {}).get("selected_arm") or ARM_BASELINE
    sanity = apply_arm_to_2026(arm)
    cmp_ = compare_to_sealed(sanity["fantasy"])
    write_json(OUT / "sanity_2026.json", {"arm": arm, **cmp_})

    write_json(
        OUT / "run_summary.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "attribution_rows": int(len(attr)),
            "lamar_provenance": prov,
            "selection": decision_bundle.get("decision"),
            "selected_arm": arm,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
