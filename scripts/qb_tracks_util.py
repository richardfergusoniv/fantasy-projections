"""Shared helpers for QB improvement-track ablations."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

KEY_QB_NAMES = (
    "Josh Allen",
    "Lamar Jackson",
    "Trevor Lawrence",
    "Bo Nix",
    "Jacoby Brissett",
    "Joe Burrow",
    "Jalen Hurts",
    "Patrick Mahomes",
)

KEY_QB_IDS = {
    "Josh Allen": "00-0034857",
    "Lamar Jackson": "00-0034796",
    "Trevor Lawrence": "00-0036971",
    "Bo Nix": "00-0039732",
    "Jacoby Brissett": "00-0033119",
    "Joe Burrow": "00-0036442",
    "Jalen Hurts": "00-0036389",
    "Patrick Mahomes": "00-0033873",
}


def holdout_qb_metrics(summary: pd.DataFrame) -> dict:
    """QB starter holdout metrics from fantasy_evaluation summary."""
    model = summary[(summary["method"] == "model") & (summary["position"] == "QB")]
    out = {}
    for scope in ("starter_depth_tier_1", "starter_8plus_games"):
        row = model[model["scope"] == scope]
        if row.empty:
            continue
        r = row.iloc[0]
        out[scope] = {
            "rate_spearman": round(float(r["rate_spearman"]), 4),
            "rate_mae": round(float(r["rate_mae"]), 3),
            "mean_bias": round(float(r["mean_bias"]), 3),
            "tier_hits": f"{int(r['tier_hits'])}/{int(r['tier_rank'])}",
        }
    return out


def board_qb_snapshot(fantasy_points_path: Path) -> dict:
    """2026 board ranks/PPG for key QBs plus elite under-rank count."""
    df = pd.read_csv(fantasy_points_path)
    qbs = df[df["position"].eq("QB")].sort_values("fantasy_pts", ascending=False).reset_index(drop=True)
    qbs["rank"] = qbs.index + 1
    key = {}
    for name in KEY_QB_NAMES:
        row = qbs[qbs["display_name"].eq(name)]
        if row.empty:
            pid = KEY_QB_IDS.get(name)
            if pid:
                row = qbs[qbs["player_id"].eq(pid)]
        if row.empty:
            key[name] = {"rank": None, "fantasy_ppg": None}
        else:
            r = row.iloc[0]
            key[name] = {
                "rank": int(r["rank"]),
                "fantasy_ppg": round(float(r["fantasy_pts"]), 3),
                "team": str(r.get("team", "")),
            }
    above_18 = int((qbs["fantasy_pts"] >= 18.0).sum())
    return {
        "key_qbs": key,
        "qb_count": int(len(qbs)),
        "qbs_above_18_ppg": above_18,
        "top12": qbs.head(12)[["rank", "display_name", "team", "fantasy_pts"]].to_dict("records"),
    }


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
