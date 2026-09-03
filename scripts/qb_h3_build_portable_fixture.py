#!/usr/bin/env python3
"""Build the portable QB reconciliation fixture from public-derived sources.

Does not commit projections.db or player_week_panel.parquet.
Prediction-side columns use only seasons < prediction_season.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.qb_sealed_baseline_bakeoff import build_history
from src.projection.qb_h3.portable_contract import (
    FIXTURE_DIR,
    FIXTURE_PARQUET,
    LABEL_COLUMNS,
    PREDICTION_COLUMNS,
    SCHEMA_VERSION,
    file_sha256,
    write_manifest,
)
from src.projection.qb_h3.role_allocation import role_from_preseason
from src.projection.transitions import SEASON_GAMES

WEEKLY = ROOT / "data" / "raw" / "weekly_qb_repair_cache" / "qb_weekly.parquet"
ACTIVE = ROOT / "output" / "qb_active_archetype" / "active_season_rates.parquet"
EVAL_SEASONS = (2023, 2024, 2025)


def _hash_if_exists(path: Path) -> str | None:
    if path.exists() and path.stat().st_size > 0:
        return file_sha256(path)
    return None


def team_volume_prior(weekly: pd.DataFrame, *, before_season: int) -> pd.DataFrame:
    """Leakage-safe team QB passing/rushing totals from the last completed season."""
    prior = weekly[pd.to_numeric(weekly["season"], errors="coerce") == before_season - 1]
    if prior.empty:
        prior = weekly[pd.to_numeric(weekly["season"], errors="coerce") < before_season]
        if prior.empty:
            return pd.DataFrame(columns=["team", "prior_team_pass_attempts", "prior_team_qb_carries"])
        last = int(prior["season"].max())
        prior = prior[prior["season"] == last]
    team_col = "recent_team" if "recent_team" in prior.columns else "team"
    g = prior.groupby(team_col, as_index=False).agg(
        prior_team_pass_attempts=("attempts", "sum"),
        prior_team_qb_carries=("carries", "sum"),
    )
    g = g.rename(columns={team_col: "team"})
    return g


def prior_player_rates(history: pd.DataFrame, *, player_id: str, target_season: int) -> dict:
    hist = history[
        (history.player_id.astype(str) == str(player_id))
        & (history.season < int(target_season))
    ]
    if hist.empty:
        return {
            "prior_active_starts_sum": 0.0,
            "prior_active_starts_mean": 0.0,
            "prior_partial_exits_sum": 0.0,
            "prior_player_attempts_per_active": None,
            "prior_player_carries_per_active": None,
        }
    w = pd.to_numeric(hist["active_starts"], errors="coerce").fillna(0.0)
    partial = pd.to_numeric(hist.get("partial_exits"), errors="coerce").fillna(0.0)

    def wmean(col):
        if col not in hist.columns:
            return None
        vals = pd.to_numeric(hist[col], errors="coerce")
        mask = vals.notna() & w.gt(0)
        if not mask.any():
            return None
        return float(np.average(vals[mask], weights=w[mask]))

    return {
        "prior_active_starts_sum": float(w.sum()),
        "prior_active_starts_mean": float(w.mean()) if len(w) else 0.0,
        "prior_partial_exits_sum": float(partial.sum()),
        "prior_player_attempts_per_active": wmean("attempts_per_active"),
        "prior_player_carries_per_active": wmean("carries_per_active"),
    }


def main() -> int:
    if not WEEKLY.exists():
        print(f"FATAL: weekly cache missing: {WEEKLY}", file=sys.stderr)
        return 2
    weekly = pd.read_parquet(WEEKLY)
    history = build_history()
    sources = {
        "weekly_qb": _hash_if_exists(WEEKLY),
        "active_season_rates": _hash_if_exists(ACTIVE),
    }
    rows = []
    for season in EVAL_SEASONS:
        ev_path = ROOT / "output" / f"fantasy_evaluation_{season}.csv"
        if not ev_path.exists():
            print(f"FATAL: missing {ev_path}", file=sys.stderr)
            return 2
        sources[f"fantasy_evaluation_{season}"] = file_sha256(ev_path)
        ev = pd.read_csv(ev_path)
        qb = ev[ev.preseason_position.astype(str).eq("QB")].copy()
        team_prior = team_volume_prior(weekly, before_season=season)
        team_map = team_prior.set_index("team") if len(team_prior) else pd.DataFrame()
        for _, r in qb.iterrows():
            pid = str(r["player_id"])
            team = r.get("preseason_team")
            depth = r.get("depth_tier")
            is_rookie = bool(r.get("is_rookie"))
            prior = prior_player_rates(history, player_id=pid, target_season=season)
            if len(team_map) and team in team_map.index:
                t_att = float(team_map.loc[team, "prior_team_pass_attempts"])
                t_car = float(team_map.loc[team, "prior_team_qb_carries"])
            else:
                t_att = t_car = 0.0
            rows.append(
                {
                    "prediction_season": int(season),
                    "prediction_cutoff": f"{season}-08-01",
                    "team": team,
                    "player_id": pid,
                    "display_name": r.get("display_name"),
                    "preseason_depth_tier": float(depth) if pd.notna(depth) else np.nan,
                    "preseason_role": role_from_preseason(
                        depth_tier=depth, is_rookie=is_rookie
                    ),
                    "is_rookie_at_cutoff": is_rookie,
                    **prior,
                    "prior_team_pass_attempts": t_att,
                    "prior_team_qb_carries": t_car,
                    "pred_team_pass_attempts_pg": t_att / SEASON_GAMES if t_att else np.nan,
                    "pred_team_qb_carries_pg": t_car / SEASON_GAMES if t_car else np.nan,
                    "destination_team_at_cutoff": team,
                    "actual_starts": float(r["actual_games_played"])
                    if pd.notna(r.get("actual_games_played"))
                    else np.nan,
                    "actual_attempts": float(r["attempts"]) if pd.notna(r.get("attempts")) else np.nan,
                    "actual_carries": float(r["carries"]) if pd.notna(r.get("carries")) else np.nan,
                    "actual_passing_yards": float(r["passing_yards"])
                    if pd.notna(r.get("passing_yards"))
                    else np.nan,
                    "actual_rushing_yards": float(r["rushing_yards"])
                    if pd.notna(r.get("rushing_yards"))
                    else np.nan,
                    "actual_passing_tds": float(r["passing_tds"])
                    if pd.notna(r.get("passing_tds"))
                    else np.nan,
                    "actual_rushing_tds": float(r["rushing_tds"])
                    if pd.notna(r.get("rushing_tds"))
                    else np.nan,
                    "actual_points": float(r["actual_points"])
                    if pd.notna(r.get("actual_points"))
                    else np.nan,
                    "sealed_model_points_end_to_end": float(r["model_points_end_to_end"])
                    if pd.notna(r.get("model_points_end_to_end"))
                    else np.nan,
                    "sealed_projected_games": float(r["projected_games"])
                    if pd.notna(r.get("projected_games"))
                    else np.nan,
                    "source_weekly": str(WEEKLY.relative_to(ROOT)),
                    "source_eval": str(ev_path.relative_to(ROOT)),
                    "source_active_rates": str(ACTIVE.relative_to(ROOT)) if ACTIVE.exists() else None,
                    "schema_version": SCHEMA_VERSION,
                }
            )
    frame = pd.DataFrame(rows)
    # Sanity: prediction columns must not be computed from actual_* .
    for col in PREDICTION_COLUMNS:
        if col not in frame.columns:
            raise RuntimeError(f"missing prediction column {col}")
    for col in LABEL_COLUMNS:
        if col not in frame.columns:
            raise RuntimeError(f"missing label column {col}")
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(FIXTURE_PARQUET, index=False)
    manifest = write_manifest(
        frame=frame,
        sources={k: v for k, v in sources.items() if v},
        extra={
            "builder": "scripts/qb_h3_build_portable_fixture.py",
            "n_players": int(frame.player_id.nunique()),
            "large_sources_not_used": [
                "data/projections.db",
                "data/processed/player_week_panel.parquet",
            ],
        },
    )
    print("wrote", FIXTURE_PARQUET, "rows", len(frame), "hash", manifest["content_hash"][:12])
    print("leakage_ok", manifest["leakage_audit"]["ok"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
