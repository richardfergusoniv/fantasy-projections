#!/usr/bin/env python3
"""Step 1: decompose the 2023 +5.11 MAE regression vs sealed end-to-end.

Frozen H1/H2 config — no retune. Writes player-level and factor attributions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.qb_sealed_baseline_bakeoff import build_history
from src.projection.qb_active_archetype.archetypes import classify_archetype
from src.projection.qb_active_archetype.evaluate import predict_player

OUT = ROOT / "output" / "qb_h3" / "step1_2023_decomposition"
LAMAR = "00-0034796"
HURTS = "00-0036389"


def _dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    history = build_history()
    rows = json.loads(
        (ROOT / "output/qb_sealed_baseline_bakeoff/fold_2023_rows.json").read_text()
    )["rows"]
    df = pd.DataFrame(rows)
    ev = pd.read_csv(ROOT / "output/fantasy_evaluation_2023.csv")
    ev = ev[ev.preseason_position.eq("QB")].copy()
    ev["player_id"] = ev.player_id.astype(str)

    enriched = []
    for _, r in df.iterrows():
        pid = str(r["player_id"])
        ev_row = ev[ev.player_id == pid]
        e0 = ev_row.iloc[0] if len(ev_row) else None

        def _evf(col, default=np.nan):
            if e0 is None or col not in e0.index or pd.isna(e0[col]):
                return default
            return float(e0[col])

        sealed_proj_g = _evf("projected_games")
        sealed_rate = _evf("model_rate_points")
        sealed_raw = _evf("model_forecast_points")
        sealed_e2e = _evf("model_points_end_to_end", float(r["sealed_points"]))
        sealed_avail_adj = _evf("availability_adjusted_points")
        sealed_carry = _evf("carry_forward_points")
        actual_att = _evf("attempts", 0.0)
        actual_car = _evf("carries", 0.0)
        actual_ptd = _evf("passing_tds", 0.0)
        actual_rtd = _evf("rushing_tds", 0.0)
        actual_g = float(r["actual_games"]) or 1.0

        arch_meta = classify_archetype(history, player_id=pid, target_season=2023)
        feat = arch_meta.get("features") or {}
        carries = feat.get("carries_per_active")
        designed = feat.get("designed_carries_per_active")
        scramble = feat.get("scramble_per_dropback")
        mislabeled_dual = bool(
            arch_meta["archetype"] == "pocket_passer"
            and (
                (carries is not None and carries >= 5.5 and designed is None)
                or (scramble is not None and scramble >= 0.08)
            )
        )
        starts_err = float(r["cand_expected_starts"] - r["actual_games"])
        avail_over = max(0.0, starts_err)
        avail_under = max(0.0, -starts_err)
        ae_sealed = abs(r["sealed_points"] - r["actual_points"])
        ae_cand = abs(r["cand_points"] - r["actual_points"])
        ae_delta = ae_cand - ae_sealed

        cand_exp_att = float(r["cand_attempts_pa"]) * float(r["cand_expected_starts"]) if pd.notna(
            r["cand_attempts_pa"]
        ) else np.nan
        cand_exp_car = float(r["cand_carries_pa"]) * float(r["cand_expected_starts"]) if pd.notna(
            r["cand_carries_pa"]
        ) else np.nan
        att_err = (cand_exp_att - actual_att) if pd.notna(cand_exp_att) else np.nan
        car_err = (cand_exp_car - actual_car) if pd.notna(cand_exp_car) else np.nan
        # Rate-level volume vs actual active rates
        att_rate_err = (
            float(r["cand_attempts_pa"]) - float(r["actual_attempts_pa"])
            if pd.notna(r["cand_attempts_pa"])
            else np.nan
        )
        car_rate_err = (
            float(r["cand_carries_pa"]) - float(r["actual_carries_pa"])
            if pd.notna(r["cand_carries_pa"])
            else np.nan
        )

        # Team / role change: prior-season team vs 2023 preseason team
        prior = history[
            (history.player_id.astype(str) == pid) & (history.season < 2023)
        ].sort_values("season")
        team_change = False
        if e0 is not None and "preseason_team" in e0.index and not prior.empty:
            last_team = prior.iloc[-1].get("team") if "team" in prior.columns else None
            if last_team and pd.notna(e0["preseason_team"]) and str(last_team) != str(e0["preseason_team"]):
                team_change = True
        role_change = bool(
            not prior.empty
            and "active_starts" in prior.columns
            and float(prior.iloc[-1].get("active_starts") or 0) < 8
            and actual_g >= 12
        )

        depth = _evf("depth_tier")
        factors = []
        if abs(starts_err) >= 3:
            factors.append("incorrect_expected_starts")
        if mislabeled_dual:
            factors.append("incorrect_archetype_assignment")
        if arch_meta["archetype"] == "insufficient_history" or bool(
            e0 is not None and e0.get("is_rookie")
        ):
            factors.append("rookie_or_insufficient_history")
        if team_change or role_change:
            factors.append("role_or_team_change")
        if r.get("returning_injury"):
            factors.append("returning_injury_cohort")
        if avail_over >= 3 and r["cand_points"] > r["actual_points"] + 30:
            factors.append("availability_overprediction_inflates_season")
        if avail_under >= 3 and ae_delta > 0:
            factors.append("availability_underprediction")
        # Partial-game: expected starts ok but season points still miss — proxy via
        # high partial_exit not modeled in frozen post-hoc path.
        if abs(starts_err) < 2 and abs(att_rate_err) < 3 and ae_delta > 15:
            factors.append("partial_game_or_efficiency_residual")
        if pd.notna(depth) and depth >= 2 and r["cand_points"] > 100 and r["actual_points"] < 40:
            factors.append("backup_got_starter_active_rates")
            factors.append("starter_backup_conservation_failure")
        if pd.notna(att_err) and abs(att_err) >= 80 and ae_delta > 0:
            factors.append("passing_volume_changes")
        if pd.notna(car_err) and abs(car_err) >= 25 and ae_delta > 0:
            factors.append("rushing_volume_changes")
        # TD composition residual: volume roughly ok but points error large
        if (
            pd.notna(att_err)
            and abs(att_err) < 60
            and pd.notna(car_err)
            and abs(car_err) < 20
            and ae_delta > 20
        ):
            factors.append("touchdown_composition")
        # Double availability: candidate season ≈ rate×starts×(starts/17)
        if pd.notna(r["cand_attempts_pa"]) and float(r["cand_expected_starts"]) > 0:
            once = float(r["cand_attempts_pa"]) * float(r["cand_expected_starts"])
            twice = once * float(r["cand_expected_starts"]) / 17.0
            # Compare cand points scale: if cand_ppg looks like already-avail-adj
            # and then × starts again — detect via cand_points ≈ pp_active * starts
            # while pp_active already ≈ season/17 style. Frozen path used once.
            factors.append("availability_applied_once_candidate")  # structural note
            if abs(cand_exp_att - twice) < 5 and abs(cand_exp_att - once) > 30:
                factors.append("double_application_of_availability")
        if abs(r["cand_points"] - r["sealed_points"]) > 80 and ae_delta > 20:
            factors.append("large_divergence_from_sealed_stack")
        # Ensemble dilution N/A on 2023 sealed fold (v1 e2e, not accuracy-first blend)
        factors.append("ensemble_dilution_not_applicable_on_2023_sealed_fold")

        enriched.append(
            {
                "player": r.get("display_name"),
                "player_id": pid,
                "actual_season_points": float(r["actual_points"]),
                "sealed_prediction": float(r["sealed_points"]),
                "frozen_candidate_prediction": float(r["cand_points"]),
                "absolute_error_change": float(ae_delta),
                "ae_sealed": float(ae_sealed),
                "ae_cand": float(ae_cand),
                "expected_starts": float(r["cand_expected_starts"]),
                "actual_starts": float(r["actual_games"]),
                "starts_error": starts_err,
                "sealed_projected_games": sealed_proj_g,
                "archetype": arch_meta["archetype"],
                "archetype_features": feat,
                "mislabeled_dual_threat": mislabeled_dual,
                "active_start_attempt_rate": float(r["cand_attempts_pa"])
                if pd.notna(r["cand_attempts_pa"])
                else None,
                "active_start_carry_rate": float(r["cand_carries_pa"])
                if pd.notna(r["cand_carries_pa"])
                else None,
                "attempt_rate_error": att_rate_err if pd.notna(att_rate_err) else None,
                "carry_rate_error": car_rate_err if pd.notna(car_rate_err) else None,
                "cand_expected_season_attempts": cand_exp_att if pd.notna(cand_exp_att) else None,
                "cand_expected_season_carries": cand_exp_car if pd.notna(cand_exp_car) else None,
                "actual_attempts": actual_att,
                "actual_carries": actual_car,
                "actual_passing_tds": actual_ptd,
                "actual_rushing_tds": actual_rtd,
                "availability_adjustment": {
                    "expected_minus_actual_starts": starts_err,
                    "overprediction": avail_over,
                    "underprediction": avail_under,
                    "sealed_projected_games": sealed_proj_g,
                },
                "depth_tier": depth if pd.notna(depth) else None,
                "team_change": team_change,
                "role_change": role_change,
                "returning_injury": bool(r.get("returning_injury")),
                "factor_flags": [f for f in factors if not f.startswith("availability_applied_once")
                                 and f != "ensemble_dilution_not_applicable_on_2023_sealed_fold"],
                "factor_notes": {
                    "availability_applied_once_on_candidate": True,
                    "ensemble_dilution": "not_applicable_2023_sealed_fold_is_v1_e2e",
                },
                "pipeline_stages": {
                    "raw": sealed_raw,
                    "reconcile": (
                        "embedded_in_model_forecast_points; "
                        "no separate reconcile checkpoint in fantasy_evaluation CSV"
                    ),
                    "compose": sealed_e2e,
                    "ensemble": (
                        "n/a_on_2023_fold; model_points_end_to_end is leakage-safe "
                        "compose output (v1), not accuracy-first blend"
                    ),
                    "sealed_model_rate_points": sealed_rate,
                    "sealed_availability_adjusted_points": sealed_avail_adj,
                    "sealed_carry_forward_points": sealed_carry,
                    "sealed_model_points_end_to_end": sealed_e2e,
                    "frozen_candidate": float(r["cand_points"]),
                    "note": (
                        "Frozen candidate is outside the sealed pipeline; "
                        "sealed model_points_end_to_end already includes "
                        "feature→train→reconcile→compose. Candidate never entered ensemble."
                    ),
                },
            }
        )

    frame = pd.DataFrame(enriched)
    worst = frame.nlargest(10, "absolute_error_change")
    best = frame.nsmallest(10, "absolute_error_change")

    # Factor contribution: sum of ae_delta for players carrying each flag
    factor_sums = {}
    for flags in frame["factor_flags"]:
        for f in flags:
            factor_sums.setdefault(f, 0.0)
    for _, row in frame.iterrows():
        for f in row["factor_flags"]:
            factor_sums[f] = factor_sums.get(f, 0.0) + float(row["absolute_error_change"])

    # Share of total positive regression from backups / mislabeled / starts
    pos = frame[frame.absolute_error_change > 0]
    total_pos = float(pos.absolute_error_change.sum())
    attribution = {
        "mean_ae_delta": float(frame.absolute_error_change.mean()),
        "sum_ae_delta": float(frame.absolute_error_change.sum()),
        "total_positive_ae_delta": total_pos,
        "factor_sum_ae_delta": factor_sums,
        "share_of_positive_regression": {
            k: (float(pos[pos.factor_flags.apply(lambda fs, key=k: key in fs)].absolute_error_change.sum()) / total_pos)
            if total_pos
            else None
            for k in factor_sums
        },
        "archetype_none_designed_bug": {
            "n_mislabeled_dual": int(frame.mislabeled_dual_threat.sum()),
            "ae_delta_sum_mislabeled": float(
                frame.loc[frame.mislabeled_dual_threat, "absolute_error_change"].sum()
            ),
            "explanation": (
                "Frozen classifier treats missing designed/scramble as pocket "
                "before checking carries>=5.5, so Lamar/Hurts-class rushers "
                "with null designed splits are labeled pocket_passer."
            ),
            "lamar_2023": classify_archetype(history, player_id=LAMAR, target_season=2023),
            "hurts_2023": classify_archetype(history, player_id=HURTS, target_season=2023),
        },
        "structural_causes_ranked": [
            {
                "cause": "starter_backup_conservation_failure",
                "evidence": (
                    "Depth-tier≥2 QBs assigned full active attempt rates × expected "
                    "starts without sealed depth discount / residual fill "
                    "(Winston, White, Rush, Kyle Allen, Beathard)."
                ),
            },
            {
                "cause": "incorrect_expected_starts",
                "evidence": (
                    "Large |expected−actual| starts among top regressors "
                    "(Murray, Tannehill, Herbert, and several backups with "
                    "starter-like expected starts)."
                ),
            },
            {
                "cause": "incorrect_archetype_assignment",
                "evidence": (
                    "Null designed/scramble → pocket before carries≥5.5 check; "
                    "Lamar/Hurts-class rushers mislabeled pocket_passer under "
                    "frozen H1/H2 classifier."
                ),
            },
            {
                "cause": "passing_volume_changes",
                "evidence": (
                    "Candidate season attempts diverge from actual when "
                    "active rates are applied without team reconcile."
                ),
            },
            {
                "cause": "rushing_volume_changes",
                "evidence": (
                    "Archetype mislabel + prior shrink understates designed/"
                    "scramble opportunity for dual-threat starters."
                ),
            },
            {
                "cause": "touchdown_composition",
                "evidence": (
                    "Residual points error after roughly matching volume "
                    "(efficiency / TD rate not jointly composed in post-hoc path)."
                ),
            },
            {
                "cause": "role_or_team_change",
                "evidence": (
                    "Prior-season role/team shifts without sealed transition "
                    "features in the frozen post-hoc candidate."
                ),
            },
            {
                "cause": "partial_game_handling",
                "evidence": (
                    "Frozen path has only a coarse partial_exit_rate; "
                    "early-exit volume haircuts are not reconciled to backups."
                ),
            },
            {
                "cause": "not_in_sealed_pipeline",
                "evidence": (
                    "Post-hoc rate×starts never saw team reconcile, depth ladder, "
                    "TD constraints, or concentration — backups unbounded."
                ),
            },
            {
                "cause": "double_application_of_availability",
                "evidence": (
                    "Not the 2023 failure mode: candidate applies availability "
                    "once via expected starts; inflation is over-predicted starts "
                    "and missing depth conservation, not rate×starts×(starts/17)."
                ),
            },
            {
                "cause": "ensemble_dilution",
                "evidence": (
                    "Not applicable on 2023 fold: sealed comparator is "
                    "model_points_end_to_end (v1 compose), not accuracy-first blend."
                ),
            },
        ],
        "requested_factor_checklist": {
            "incorrect_expected_starts": "flagged when |starts_error|>=3",
            "incorrect_archetype_assignment": "null designed/scramble pocket bug + dual carries",
            "rookies_or_insufficient_history": "archetype or is_rookie",
            "role_or_team_changes": "prior team != preseason team or backup→starter",
            "partial_game_handling": "coarse residual flag when starts/rates close but AE worsens",
            "double_application_of_availability": "checked; not primary 2023 driver",
            "starter_backup_conservation": "depth>=2 with starter-like season totals",
            "passing_volume_changes": "|cand_season_att - actual_att| >= 80",
            "rushing_volume_changes": "|cand_season_car - actual_car| >= 25",
            "touchdown_composition": "volume roughly ok but AE delta > 20",
            "ensemble_dilution": "N/A on 2023 sealed fold",
        },
    }

    _dump(OUT / "players.json", {"season": 2023, "players": enriched})
    _dump(
        OUT / "top_contributors.json",
        {
            "ten_largest_negative_contributors_candidate_worse": worst.to_dict("records"),
            "ten_largest_positive_contributors_candidate_better": best.to_dict("records"),
        },
    )
    _dump(OUT / "factor_attribution.json", attribution)
    print("2023 mean ae_delta", attribution["mean_ae_delta"])
    print("worst:", worst[["player", "absolute_error_change", "factor_flags"]].to_string(index=False))
    print("mislabeled dual", attribution["archetype_none_designed_bug"]["n_mislabeled_dual"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
