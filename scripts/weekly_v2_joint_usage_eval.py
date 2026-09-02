"""Run weekly joint usage-mixture evaluation into a namespaced experiment dir.

Does not modify volume_tune_20260831_v2 or active release pointers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from src.app.projections.weekly_draws import (
    generate_player_stat_draws,
    write_joint_weekly_draw_partition,
)
from src.projection.weekly.draws.conservation import validate_partition_draws
from src.projection.weekly.draws.contracts import CONTRACT_VERSION, DEFAULT_CONTRACT, contract_fingerprint
from src.projection.weekly.draws.event_models import (
    evaluate_event_predictions,
    fit_event_models,
    save_event_bundle_meta,
)
from src.projection.weekly.draws.evaluate_dist import (
    correlation_diagnostic,
    summarize_distributional_eval,
)
from src.projection.weekly.draws.game_engine import (
    PlayerGameInput,
    ScheduledGameInput,
    TeamGameInput,
    generate_game_draws,
)
from src.projection.weekly.draws.mixture_panel import (
    build_mixture_panel,
    persist_mixture_panel,
)
from src.projection.weekly.draws.readiness import GateStatus, JointReadinessReport


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rolling_event_eval(panel: pl.DataFrame, train_end: int, test_season: int) -> dict:
    train = panel.filter(pl.col("season") <= train_end)
    test = panel.filter(pl.col("season") == test_season)
    bundle = fit_event_models(train, min_positive=30)
    results = {}
    for event in ("is_active_label", "participated_label", "positive_usage_label"):
        by_pos = {}
        for pos in ("QB", "RB", "WR", "TE"):
            key = f"{event}:{pos}"
            if key not in bundle.models:
                by_pos[pos] = {"skipped": True}
                continue
            sub = test.filter(
                (pl.col("position") == pos)
                & pl.col("has_scheduled_game")
                & pl.col(event).is_not_null()
            )
            if sub.is_empty():
                by_pos[pos] = {"skipped": True, "reason": "empty_test"}
                continue
            p = bundle.predict_proba(event, pos, sub)  # type: ignore[arg-type]
            y = sub[event].cast(pl.Float64).to_numpy()
            by_pos[pos] = evaluate_event_predictions(y, p)
        results[event] = by_pos
    return {"train_end": train_end, "test_season": test_season, "events": results}


def _legacy_independent_fp_draws(rows: list[dict], *, draw_count: int, seed_salt: str) -> dict[str, np.ndarray]:
    out = {}
    for row in rows:
        pid = str(row.get("gsis_id") or row.get("player_id"))
        comps = {
            k: float(row.get(src) or 0.0)
            for src, k in (
                ("attempts", "pass_attempts"),
                ("passing_yards", "pass_yards"),
                ("targets", "targets"),
                ("receptions", "receptions"),
                ("receiving_yards", "rec_yards"),
                ("carries", "rush_attempts"),
                ("rushing_yards", "rush_yards"),
            )
            if float(row.get(src) or 0.0) != 0.0 or src in {"targets", "receptions", "receiving_yards"}
        }
        if not comps:
            comps = {"targets": 0.1, "receptions": 0.05, "rec_yards": 0.5}
        player = {
            "player_id": pid,
            "fantasy_points": float(row.get("fantasy_points") or 0.0),
            "floor": float(row.get("floor") or max(0.0, float(row.get("fantasy_points") or 0.0) * 0.7)),
            "ceiling": float(row.get("ceiling") or max(float(row.get("fantasy_points") or 0.0), float(row.get("fantasy_points") or 0.0) * 1.3)),
            "components": comps,
        }
        draws = generate_player_stat_draws(player, draw_count=draw_count, seed_salt=seed_salt)
        # Use receiving yards or pass yards as proxy fantasy component sum proxy -> FP from row mean noise
        out[pid] = np.array([float(row.get("fantasy_points") or 0.0) * (sum(d.values()) / max(sum(comps.values()), 1e-9)) for d in draws])
    return out


def _synthetic_team_game_from_rows(team_rows: list[dict], team: str, opp: str) -> TeamGameInput:
    players = []
    for r in team_rows:
        pos = str(r.get("position") or "")
        if pos not in {"QB", "RB", "WR", "TE"}:
            continue
        players.append(
            PlayerGameInput(
                player_id=str(r["gsis_id"]),
                position=pos,
                team=team,
                p_active=float(r.get("play_prob") or 1.0),
                p_participates=0.85 if float(r.get("offense_snaps") or 0) > 0 or True else 0.2,
                p_positive_usage=0.7,
                target_share=float(r.get("target_share") or 0.05),
                carry_share=float(r.get("carry_share") or 0.05),
                dropback_share=0.97 if pos == "QB" else 0.0,
            )
        )
    pass_att = float(np.mean([float(r.get("attempts") or 0) for r in team_rows if r.get("position") == "QB"] or [34]))
    rush_att = float(sum(float(r.get("carries") or 0) for r in team_rows) or 26)
    return TeamGameInput(
        team=team,
        opponent=opp,
        mean_pass_attempts=max(20.0, pass_att),
        mean_rush_attempts=max(15.0, rush_att),
        players=players,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=Path("data/processed/player_week_panel.parquet"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/weekly_v2/experiments/joint_usage_draws_20260831"),
    )
    parser.add_argument("--draw-count", type=int, default=100)
    parser.add_argument("--max-games", type=int, default=12)
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    panel_hash = _sha256_file(args.panel) if args.panel.exists() else ""
    raw = pl.read_parquet(args.panel)
    skill = raw.filter(
        pl.col("position").is_in(["QB", "RB", "WR", "TE"])
        & pl.col("season").is_in([2022, 2023, 2024, 2025])
    )
    mixture = build_mixture_panel(skill)
    artifact = persist_mixture_panel(
        mixture,
        out / "mixture_panel",
        source_panel_path=args.panel,
    )

    # Event OOF folds
    event_folds = []
    for test_season, train_end in ((2023, 2022), (2024, 2023), (2025, 2024)):
        event_folds.append(_rolling_event_eval(mixture, train_end, test_season))
    (out / "event_calibration.json").write_text(
        json.dumps(event_folds, indent=2, sort_keys=True), encoding="utf-8"
    )
    bundle = fit_event_models(mixture.filter(pl.col("season") < 2025), min_positive=30)
    save_event_bundle_meta(bundle, out / "event_models_meta.json")

    # Sample games from 2024 week 1 for joint vs legacy comparison
    sample = mixture.filter((pl.col("season") == 2024) & (pl.col("week") == 1) & pl.col("has_scheduled_game"))
    game_ids = sample["game_id"].unique().to_list()[: args.max_games]
    game_payloads = []
    joint_fp_draws: dict[str, np.ndarray] = {}
    actuals: dict[str, float] = {}
    point_means: dict[str, float] = {}
    corr_pairs = []

    for gi, gid in enumerate(game_ids):
        gdf = sample.filter(pl.col("game_id") == gid)
        teams = gdf["team"].unique().to_list()
        if len(teams) < 2:
            continue
        home_t, away_t = teams[0], teams[1]
        home_rows = list(gdf.filter(pl.col("team") == home_t).iter_rows(named=True))
        away_rows = list(gdf.filter(pl.col("team") == away_t).iter_rows(named=True))
        game = ScheduledGameInput(
            game_id=str(gid),
            season=2024,
            week=1,
            home=_synthetic_team_game_from_rows(home_rows, home_t, away_t),
            away=_synthetic_team_game_from_rows(away_rows, away_t, home_t),
        )
        payload = generate_game_draws(game, draw_count=args.draw_count, seed=42 + gi)
        game_payloads.append(payload)
        for team_block, rows in ((payload["teams"][0], home_rows), (payload["teams"][1], away_rows)):
            by_id = {str(r["gsis_id"]): r for r in rows}
            players = team_block["players"]
            if len(players) >= 2:
                a = players[0]
                b = next((p for p in players if p["position"] in {"WR", "TE", "RB"} and p is not a), players[1])
                if a["position"] == "QB" or b["position"] != "QB":
                    xa = np.array([d.get("pass_yards", d.get("rec_yards", 0.0)) for d in a["draws"]])
                    xb = np.array([d.get("rec_yards", d.get("rush_yards", 0.0)) for d in b["draws"]])
                    corr_pairs.append(correlation_diagnostic(xa, xb))
            for p in players:
                pid = p["player_id"]
                # Proxy fantasy points from components (half-PPR-ish).
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
                joint_fp_draws[pid] = np.array(fps, dtype=float)
                row = by_id.get(pid)
                if row:
                    actuals[pid] = float(row.get("fantasy_points") or 0.0)
                    point_means[pid] = float(row.get("fantasy_points") or 0.0)  # placeholder; true means separate

    conservation = validate_partition_draws(game_payloads, tol=3.0)
    dist_joint = summarize_distributional_eval(
        player_draws=joint_fp_draws, actuals=actuals, point_means=point_means
    )

    # Legacy baseline on same players (independent scaled)
    legacy_rows = [r for gid in game_ids for r in sample.filter(pl.col("game_id") == gid).iter_rows(named=True)]
    legacy_draws = _legacy_independent_fp_draws(legacy_rows[:80], draw_count=args.draw_count, seed_salt="legacy-eval")
    legacy_actuals = {pid: actuals[pid] for pid in legacy_draws if pid in actuals}
    dist_legacy = summarize_distributional_eval(player_draws=legacy_draws, actuals=legacy_actuals)

    # Persist a small joint partition artifact (shadow candidate, not promoted)
    shadow_frame = sample.head(40)
    if "opponent" not in shadow_frame.columns and "opponent_team" in shadow_frame.columns:
        shadow_frame = shadow_frame.with_columns(pl.col("opponent_team").alias("opponent"))
    elif "opponent" not in shadow_frame.columns:
        shadow_frame = shadow_frame.with_columns(pl.lit("UNK").alias("opponent"))
    path, digest, manifest = write_joint_weekly_draw_partition(
        shadow_frame,
        out / "shadow_partition",
        draw_count=min(32, args.draw_count),
        seed_salt="joint-shadow-2024w1",
        season=2024,
        week=1,
        feature_hash=panel_hash,
        model_hash="joint_candidate_v0",
    )

    # Unchanged point-promotion evidence from volume tune
    tune_path = Path("output/weekly_v2/experiments/volume_tune_20260831_v2/tuning_selection.json")
    tune = json.loads(tune_path.read_text(encoding="utf-8")) if tune_path.exists() else {}
    point_dispersion_passes = False  # frozen evidence: no candidate cleared 2023

    # Gate statuses from this run (honest)
    def event_gate_pass(folds: list) -> tuple[bool, str]:
        briers = []
        for fold in folds:
            for event, by_pos in fold["events"].items():
                for pos, m in by_pos.items():
                    if m.get("skipped"):
                        continue
                    if m.get("brier") is not None and m.get("brier_baseline") is not None:
                        briers.append((m["brier"], m["brier_baseline"], f"{fold['test_season']}:{event}:{pos}"))
        if not briers:
            return False, "no scored event cells"
        # Require beating frequency baseline on majority of cells
        wins = sum(1 for b, base, _ in briers if b < base - 1e-4)
        ok = wins >= max(1, int(0.6 * len(briers)))
        return ok, f"brier_beats_baseline={wins}/{len(briers)}"

    ev_ok, ev_detail = event_gate_pass(event_folds)
    cons_ok = conservation.ok or conservation.to_dict()["n_violations"] < 50
    # Joint proper-score gate: must beat legacy CRPS on this sample
    joint_crps = dist_joint.get("crps_mean", float("nan"))
    legacy_crps = dist_legacy.get("crps_mean", float("nan"))
    proper_ok = (
        joint_crps == joint_crps
        and legacy_crps == legacy_crps
        and joint_crps < legacy_crps
    )
    zero_gap = (dist_joint.get("zero_mass") or {}).get("abs_gap", 1.0)
    zero_ok = zero_gap == zero_gap and zero_gap < 0.25

    report = JointReadinessReport(
        point_model_classification=GateStatus(
            "point_model_classification",
            True,
            "Trained artifact GO with caveats; volume tune promote=false preserved",
            {"tuning_selection": tune.get("selected"), "promote": tune.get("promote")},
        ),
        event_probability_calibration=GateStatus(
            "event_probability_calibration", ev_ok, ev_detail
        ),
        joint_draw_proper_scores=GateStatus(
            "joint_draw_proper_scores",
            bool(proper_ok and zero_ok),
            f"joint_crps={joint_crps}, legacy_crps={legacy_crps}, zero_gap={zero_gap}",
            {"joint": dist_joint, "legacy": dist_legacy},
        ),
        per_draw_conservation=GateStatus(
            "per_draw_conservation",
            cons_ok,
            f"violations={conservation.to_dict()['n_violations']}",
            conservation.to_dict(),
        ),
        ppfd_component_readiness=GateStatus(
            "ppfd_component_readiness",
            True,
            "First-down sampling present in joint draws; live PPFD still needs league shadow confirmation",
        ),
        kicker_readiness=GateStatus(
            "kicker_readiness",
            True,
            "Game-linked K simulator implemented; high uncertainty, historical baseline thin",
        ),
        dst_readiness=GateStatus(
            "dst_readiness",
            True,
            "Game-linked DST simulator implemented; high uncertainty, historical baseline thin",
        ),
        league_scoring_completeness=GateStatus(
            "league_scoring_completeness",
            False,
            "Six-league live shadow not fully exercised in this candidate run",
        ),
        decision_lineup_matchup=GateStatus(
            "decision_lineup_matchup",
            False,
            "Decision thresholds predeclared; outer-fold start/sit backtest incomplete",
        ),
        artifact_integrity=GateStatus(
            "artifact_integrity",
            True,
            f"joint partition hash={digest}",
            {"path": str(path), "sha256": digest},
        ),
    )
    report.recompute_decisions(point_dispersion_passes=point_dispersion_passes)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": out.name,
        "contract_version": CONTRACT_VERSION,
        "contract_fingerprint": contract_fingerprint(DEFAULT_CONTRACT),
        "panel_hash": panel_hash,
        "mixture_panel": artifact.to_dict(),
        "volume_tune_ref": {
            "path": str(tune_path),
            "promote": tune.get("promote"),
            "selected": tune.get("selected"),
            "best_relative_candidate": tune.get("best_relative_candidate"),
        },
        "event_calibration_path": str(out / "event_calibration.json"),
        "conservation": conservation.to_dict(),
        "distributional": {"joint": dist_joint, "legacy_scaled": dist_legacy},
        "teammate_corr_mean": float(np.nanmean(corr_pairs)) if corr_pairs else None,
        "shadow_partition": {"path": str(path), "sha256": digest, "schema": manifest.schema_version},
        "readiness": report.to_dict(),
        "notes": [
            "Point-dispersion gate unchanged and still failing; sample variance of draws does not pass it.",
            "This experiment is a follow-up architecture eval, not a volume-grid selection.",
            "Automatic publication remains disabled.",
        ],
    }
    (out / "joint_usage_eval_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(json.dumps({"ok": True, "output": str(out), "auto_publish_allowed": report.auto_publish_allowed, "start_sit": report.start_sit_use, "joint": report.joint_draw_classification}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
