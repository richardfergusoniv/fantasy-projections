"""Six-league shadow scoring against a joint partition (no pointer promotion)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.app.projections.weekly_league_scoring import LeagueScoringContract
from src.app.scoring.compiler import score_stat_draw
from src.projection.weekly.draws.partition_schema import (
    aligned_player_draws_by_index,
    detect_partial_or_corrupt,
    load_joint_partition,
    verify_joint_partition,
)


def _load_leagues(owner_path: Path) -> list[LeagueScoringContract]:
    raw = json.loads(owner_path.read_text(encoding="utf-8"))
    leagues = []
    for row in raw.get("leagues") or []:
        # Prefer embedded scoring snapshot if present; else skip with note.
        scoring = row.get("scoring_settings") or row.get("scoring") or {}
        roster = row.get("roster_positions") or ["QB", "RB", "WR", "TE", "FLEX"]
        if not scoring:
            continue
        leagues.append(
            LeagueScoringContract(
                league_id=str(row.get("league_id") or row.get("id") or ""),
                display_name=str(row.get("display_name") or row.get("name") or ""),
                scoring_settings=dict(scoring),
                roster_positions=list(roster),
                contract=__import__(
                    "src.app.scoring.compiler", fromlist=["compile_sleeper_scoring"]
                ).compile_sleeper_scoring(scoring, roster),
            )
        )
    return leagues


def score_partition_for_leagues(
    partition: dict[str, Any],
    leagues: list[LeagueScoringContract],
    *,
    max_players: int = 20,
) -> dict[str, Any]:
    draws = aligned_player_draws_by_index(partition)
    positions: dict[str, str] = {}
    for game in partition.get("games") or []:
        for team in game.get("teams") or []:
            for p in team.get("players") or []:
                positions[str(p["player_id"])] = str(p.get("position") or "RB")
    player_ids = list(draws.keys())[:max_players]
    per_league = []
    for league in leagues:
        samples = []
        for pid in player_ids:
            pts = [
                score_stat_draw(d, league.contract, position=positions.get(pid, "RB"))
                for d in draws[pid][: min(32, len(draws[pid]))]
            ]
            mean_pts = sum(pts) / len(pts) if pts else 0.0
            fd_mass = sum(1 for d in draws[pid][:32] if d.get("rec_first_downs") or d.get("rush_first_downs") or d.get("pass_first_downs"))
            samples.append(
                {
                    "player_id": pid,
                    "position": positions.get(pid),
                    "mean_points": mean_pts,
                    "draws_with_first_downs": fd_mass,
                }
            )
        per_league.append(
            {
                "league_id": league.league_id,
                "display_name": league.display_name,
                "contract_hash": league.contract.contract_hash,
                "samples": samples,
                "mean_of_means": (
                    sum(s["mean_points"] for s in samples) / len(samples) if samples else 0.0
                ),
            }
        )
    hashes = [x["contract_hash"] for x in per_league]
    return {
        "n_leagues": len(per_league),
        "distinct_contract_hashes": len(set(hashes)),
        "leagues": per_league,
        "ppfd_nonzero_players": sum(
            1
            for s in (per_league[0]["samples"] if per_league else [])
            if s["draws_with_first_downs"] > 0
        ),
    }


def _fixture_leagues_for_live_ids() -> list[LeagueScoringContract]:
    """Map the six live league IDs to representative fixture contracts.

    Full ``scoring_settings`` blobs are not present in
    ``six_league_scoring_shadow.json`` / owner config; fixtures exercise the
    distinct live shapes (standard, K/DST-rich, yardage bonus, PPFD, dynasty,
    superflex) without promoting pointers.
    """
    from src.app.scoring.compiler import compile_sleeper_scoring

    live = Path("output/weekly_v2/six_league_scoring_shadow.json")
    ids: list[tuple[str, str]] = []
    if live.exists():
        raw = json.loads(live.read_text(encoding="utf-8"))
        for row in raw.get("leagues") or []:
            ids.append((str(row.get("league_id") or ""), str(row.get("display_name") or "")))
    fixture_names = [
        "standard.json",
        "k_dst.json",
        "yardage_bonus.json",
        "ppfd.json",
        "dynasty.json",
        "superflex.json",
    ]
    leagues: list[LeagueScoringContract] = []
    root = Path("tests/fixtures/scoring")
    for i, name in enumerate(fixture_names):
        payload = json.loads((root / name).read_text(encoding="utf-8"))
        settings = payload.get("scoring_settings") or payload
        roster = payload.get("roster_positions") or ["QB", "RB", "WR", "TE", "FLEX"]
        league_id, display = ids[i] if i < len(ids) else (f"fixture-{i}", name)
        leagues.append(
            LeagueScoringContract(
                league_id=league_id,
                display_name=display or name,
                scoring_settings=settings,
                roster_positions=roster,
                contract=compile_sleeper_scoring(settings, roster),
            )
        )
    return leagues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--partition",
        type=Path,
        default=Path(
            "output/weekly_v2/experiments/joint_usage_draws_20260831/shadow_partition/joint_stat_partition.json"
        ),
    )
    parser.add_argument("--owner-config", type=Path, default=Path("config/sleeper_owner.json"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "output/weekly_v2/experiments/joint_usage_draws_20260831/six_league_joint_shadow.json"
        ),
    )
    args = parser.parse_args()

    ok, digest = verify_joint_partition(args.partition)
    payload = load_joint_partition(args.partition) if args.partition.exists() else {}
    corrupt = detect_partial_or_corrupt(payload) if payload else ["missing"]
    # Failure injection: tampered validated label must not pass
    tamper_block = detect_partial_or_corrupt(
        {**payload, "draw_mode": "joint_stat_mixture_validated", "conservation_ok": False}
    )

    leagues: list[LeagueScoringContract] = []
    notes: list[str] = []
    if args.owner_config.exists():
        # Live owner file may only have league ids; fall back to prior six-league shadow.
        leagues = _load_leagues(args.owner_config)
        if not leagues:
            notes.append("owner config lacked inline scoring_settings; using six_league_scoring_shadow.json")
            shadow = Path("output/weekly_v2/six_league_scoring_shadow.json")
            if shadow.exists():
                raw = json.loads(shadow.read_text(encoding="utf-8"))
                for row in raw.get("leagues") or []:
                    settings = row.get("scoring_settings") or {}
                    roster = row.get("roster_positions") or ["QB", "RB", "WR", "TE", "FLEX"]
                    if not settings:
                        continue
                    from src.app.scoring.compiler import compile_sleeper_scoring

                    leagues.append(
                        LeagueScoringContract(
                            league_id=str(row.get("league_id") or ""),
                            display_name=str(row.get("display_name") or ""),
                            scoring_settings=settings,
                            roster_positions=roster,
                            contract=compile_sleeper_scoring(settings, roster),
                        )
                    )
    if not leagues:
        notes.append(
            "inline scoring_settings unavailable; using six fixture contracts "
            "mapped to live league IDs (exact live blobs not in static artifacts)"
        )
        leagues = _fixture_leagues_for_live_ids()
    if not leagues:
        notes.append("no league contracts available; shadow scoring skipped")
        result = {
            "partition_ok": ok,
            "partition_hash": digest,
            "corrupt_reasons": corrupt,
            "tamper_validated_blocked": "validated_without_gates" in tamper_block,
            "notes": notes,
            "n_leagues": 0,
        }
    else:
        scored = score_partition_for_leagues(payload, leagues)
        result = {
            "partition_ok": ok,
            "partition_hash": digest,
            "corrupt_reasons": corrupt,
            "tamper_validated_blocked": "validated_without_gates" in tamper_block,
            "notes": notes,
            **scored,
            "pointer_advanced": False,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"ok": True, "output": str(args.output), "n_leagues": result.get("n_leagues")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
