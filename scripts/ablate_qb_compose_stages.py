"""Stage attribution ablation: QB fantasy PPG at each compose checkpoint.

Usage:
    python scripts/ablate_qb_compose_stages.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.qb_tracks_util import KEY_QB_IDS, KEY_QB_NAMES, write_json
from src.projection.composition import compose_board_stages, shipped_context

STAGES = (
    "raw_model",
    "post_team_volume_reconcile",
    "post_td_clip",
    "final_shipped",
)


def _stage_delta(stages: dict, pid: str) -> dict:
    raw = stages["raw_model"].get(pid, {}).get("fantasy_ppg")
    out = {}
    for stage in STAGES[1:]:
        val = stages[stage].get(pid, {}).get("fantasy_ppg")
        out[stage] = {
            "fantasy_ppg": val,
            "delta_from_raw": round(val - raw, 3) if val is not None and raw is not None else None,
        }
    out["raw_model"] = {"fantasy_ppg": raw, "delta_from_raw": 0.0}
    return out


def main() -> None:
    raw_path = ROOT / "output" / "projections_2026_raw.csv"
    if not raw_path.exists():
        raise SystemExit(f"Missing pre-compose board: {raw_path}")
    long = pd.read_csv(raw_path)
    for col in ("pred_season", "pred_season_low", "pred_season_high", "team_volume_scale", "td_rate_clip_applied"):
        if col in long.columns:
            long = long.drop(columns=[col])

    ctx = shipped_context(conn=None, target_season=2026)
    stages = compose_board_stages(long, ctx)

    qbs = long[long["position"].eq("QB")].drop_duplicates("player_id")
    if "display_name" not in qbs.columns:
        names = pd.read_csv(ROOT / "output" / "fantasy_points_2026.csv")[
            ["player_id", "display_name"]
        ].drop_duplicates("player_id")
        qbs = qbs.merge(names, on="player_id", how="left")

    top = (
        pd.read_csv(ROOT / "output" / "fantasy_points_2026.csv")
        .query("position == 'QB'")
        .sort_values("fantasy_pts", ascending=False)
        .head(20)
    )
    focus_ids = list(dict.fromkeys(
        list(top["player_id"])
        + [KEY_QB_IDS[n] for n in KEY_QB_NAMES if n in KEY_QB_IDS]
    ))

    players = {}
    for pid in focus_ids:
        meta = stages["final_shipped"].get(pid, {})
        players[pid] = {
            "display_name": meta.get("display_name", pid),
            "team": meta.get("team", ""),
            "stages": _stage_delta(stages, pid),
        }

    elite_hurt = []
    for pid, info in players.items():
        raw = info["stages"]["raw_model"]["fantasy_ppg"]
        final = info["stages"]["final_shipped"]["fantasy_ppg"]
        if raw is None or final is None:
            continue
        delta = final - raw
        if raw >= 14.0 and delta <= -1.0:
            elite_hurt.append({
                "player_id": pid,
                "display_name": info["display_name"],
                "raw_ppg": raw,
                "final_ppg": final,
                "total_delta": round(delta, 3),
                "worst_stage": min(
                    ((s, info["stages"][s]["delta_from_raw"]) for s in STAGES[1:]),
                    key=lambda x: x[1] if x[1] is not None else 0,
                )[0],
            })
    elite_hurt.sort(key=lambda x: x["total_delta"])

    payload = {
        "season": 2026,
        "stage_order": list(STAGES),
        "players": players,
        "elite_hurt_by_compose": elite_hurt,
        "stage_summary": {
            stage: {
                "mean_delta_from_raw": round(
                    pd.Series([
                        p["stages"][stage]["delta_from_raw"]
                        for p in players.values()
                        if p["stages"][stage]["delta_from_raw"] is not None
                    ]).mean(),
                    3,
                )
                if players else None
            }
            for stage in STAGES[1:]
        },
    }
    out_path = ROOT / "output" / "ablation_qb_compose_stages_2026.json"
    write_json(out_path, payload)
    print(f"Wrote {out_path}")
    print("\nElite compose drag (raw>=14, final drop>=1 PPG):")
    for row in elite_hurt[:8]:
        print(
            f"  {row['display_name']}: {row['raw_ppg']:.2f} -> {row['final_ppg']:.2f} "
            f"(worst={row['worst_stage']}, d={row['total_delta']:+.2f})"
        )


if __name__ == "__main__":
    main()
