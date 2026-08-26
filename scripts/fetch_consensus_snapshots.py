"""Fetch and freeze preseason ADP(/ECR) consensus snapshots for market backtests.

Uses Fantasy Football Calculator historical half-PPR 12-team ADP (?year=YYYY).
Current-season FantasyPros ECR is attached when scrape_date year matches; older
seasons are ADP-only (free historical ECR is not available from nflreadpy).

Writes data/consensus/consensus_{season}.json in the same row schema as 2026.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.draft_assistant.compare_prepare import (  # noqa: E402
    FFC_UA,
    _join_keys,
    _norm_name,
    load_fp_ecr_ppr,
    load_id_map,
)
from src.projection.market_metrics import norm_name  # noqa: E402

CONSENSUS_DIR = REPO_ROOT / "data" / "consensus"


def fetch_ffc_adp_year(year: int, *, scoring: str = "half-ppr", teams: int = 12):
    """FFC ADP for a historical preseason year."""
    import urllib.request

    url = f"https://fantasyfootballcalculator.com/api/v1/adp/{scoring}?teams={teams}&year={year}"
    req = urllib.request.Request(url, headers={"User-Agent": FFC_UA})
    with urllib.request.urlopen(req, timeout=45) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if payload.get("status") != "Success":
        raise RuntimeError(f"FFC ADP failed for {year}: {payload.get('status')}")
    meta = payload.get("meta") or {}
    players = payload.get("players") or []
    return players, {"url": url, "scoring": scoring, "teams": teams, "year": year, **meta}


def _gsis_lookups(id_map: pd.DataFrame) -> tuple[dict[str, str], dict[tuple[str, str], str], dict[tuple[str, str, str], str]]:
    by_fp: dict[str, str] = {}
    by_name_pos: dict[tuple[str, str], str] = {}
    by_name_pos_team: dict[tuple[str, str, str], str] = {}
    name_col = "name" if "name" in id_map.columns else "player_name"
    for _, row in id_map.iterrows():
        gsis = row.get("gsis_id")
        if pd.isna(gsis) or not gsis:
            continue
        gsis = str(gsis)
        if "fantasypros_id" in id_map.columns and pd.notna(row.get("fantasypros_id")):
            by_fp[str(int(row["fantasypros_id"]))] = gsis
        pos = str(row.get("position") or row.get("pos") or "").upper()
        team = row.get("team") or row.get("latest_team")
        nm = _norm_name(row.get(name_col))
        if nm and pos in ("QB", "RB", "WR", "TE"):
            by_name_pos[(nm, pos)] = gsis
            n, p, t = _join_keys(row.get(name_col), pos, team)
            if t:
                by_name_pos_team[(n, p, t)] = gsis
    return by_fp, by_name_pos, by_name_pos_team


def build_snapshot(season: int, *, teams: int = 12, include_ecr: bool = True) -> dict[str, Any]:
    ffc_players, ffc_meta = fetch_ffc_adp_year(season, teams=teams)
    id_map = load_id_map()
    by_fp, by_name_pos, by_name_pos_team = _gsis_lookups(id_map)

    ecr_by_gsis: dict[str, dict] = {}
    ecr_by_key: dict[tuple[str, str], dict] = {}
    ecr_by_key_team: dict[tuple[str, str, str], dict] = {}
    ecr_meta: dict[str, Any] = {"source": None, "note": "not attached"}
    if include_ecr:
        try:
            ecr_df, ecr_meta = load_fp_ecr_ppr()
            scrape = str(ecr_meta.get("scrape_date") or "")
            scrape_year = int(scrape[:4]) if scrape[:4].isdigit() else None
            if scrape_year != season:
                ecr_meta = {
                    "source": ecr_meta.get("source"),
                    "scrape_date": scrape,
                    "note": (
                        f"Skipped: live ECR scrape_date year {scrape_year} "
                        f"!= snapshot season {season}; ADP-only snapshot"
                    ),
                }
                ecr_df = ecr_df.iloc[0:0]
            for _, row in ecr_df.iterrows():
                rec = {
                    "ecr": float(row["ecr"]) if pd.notna(row.get("ecr")) else None,
                    "ecr_sd": float(row["sd"]) if pd.notna(row.get("sd")) else None,
                    "fp_id": str(int(row["id"])) if pd.notna(row.get("id")) else None,
                    "player": row.get("player"),
                    "pos": row.get("pos"),
                    "team": row.get("team") or row.get("tm"),
                }
                n, p, t = _join_keys(rec["player"], rec["pos"], rec["team"])
                ecr_by_key[(n, p)] = rec
                if t:
                    ecr_by_key_team[(n, p, t)] = rec
                if rec["fp_id"] and rec["fp_id"] in by_fp:
                    ecr_by_gsis[by_fp[rec["fp_id"]]] = rec
        except Exception as exc:  # noqa: BLE001 — snapshot still useful ADP-only
            ecr_meta = {"source": None, "error": str(exc), "note": "ECR load failed"}

    rows: list[dict] = []
    for row in ffc_players:
        name = row.get("name")
        pos = str(row.get("position") or "").upper()
        team = row.get("team")
        if pos not in ("QB", "RB", "WR", "TE"):
            continue
        n, p, t = _join_keys(name, pos, team)
        gsis = by_name_pos_team.get((n, p, t)) or by_name_pos.get((n, p))
        if not gsis:
            # Placeholder id so consensus_spread name fallback still works
            gsis = f"ffc-{row.get('player_id')}"
        ecr = ecr_by_gsis.get(gsis) or ecr_by_key_team.get((n, p, t)) or ecr_by_key.get((n, p))
        rows.append(
            {
                "player_id": gsis,
                "display_name": name,
                "position": pos,
                "team": team,
                "ecr": round(float(ecr["ecr"]), 2) if ecr and ecr.get("ecr") is not None else None,
                "ecr_sd": ecr.get("ecr_sd") if ecr else None,
                "adp": float(row["adp"]) if row.get("adp") is not None else None,
                "adp_stdev": float(row["stdev"]) if row.get("stdev") is not None else None,
            }
        )

    # Also include ECR-only players when ECR matches season (union with ADP list)
    if ecr_by_gsis or ecr_by_key:
        seen = {(norm_name(r["display_name"]), r["position"]) for r in rows}
        for key, rec in ecr_by_key.items():
            if key in seen:
                continue
            gsis = by_name_pos.get(key) or (
                by_fp.get(rec["fp_id"]) if rec.get("fp_id") else None
            )
            if not gsis:
                continue
            rows.append(
                {
                    "player_id": gsis,
                    "display_name": rec["player"],
                    "position": rec["pos"],
                    "team": rec.get("team"),
                    "ecr": round(float(rec["ecr"]), 2) if rec.get("ecr") is not None else None,
                    "ecr_sd": rec.get("ecr_sd"),
                    "adp": None,
                    "adp_stdev": None,
                }
            )

    matched_ecr = sum(1 for r in rows if r.get("ecr") is not None)
    matched_adp = sum(1 for r in rows if r.get("adp") is not None)
    return {
        "meta": {
            "season": season,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "player_count": len(rows),
            "matched_ecr": matched_ecr,
            "matched_adp": matched_adp,
            "scoring_our": "half-PPR, 4pt pass TD",
            "ecr": ecr_meta,
            "adp": {
                "source": "Fantasy Football Calculator",
                "attribution": "https://fantasyfootballcalculator.com",
                **{
                    k: ffc_meta.get(k)
                    for k in (
                        "url",
                        "scoring",
                        "teams",
                        "year",
                        "total_drafts",
                        "start_date",
                        "end_date",
                    )
                    if ffc_meta.get(k) is not None
                },
            },
            "delta_note": "delta = our_rank − market (negative = we rank them higher)",
            "as_of": ffc_meta.get("end_date") or ffc_meta.get("start_date"),
        },
        "rows": rows,
    }


def write_snapshot(season: int, *, teams: int = 12, force: bool = False) -> Path:
    CONSENSUS_DIR.mkdir(parents=True, exist_ok=True)
    out = CONSENSUS_DIR / f"consensus_{season}.json"
    if out.exists() and season == 2026 and not force:
        # Preserve the hand-curated 2026 board-joined snapshot unless forced.
        print(f"Keeping existing {out} (pass --force to overwrite)")
        return out
    payload = build_snapshot(season, teams=teams)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seasons",
        default="2023,2024,2025,2026",
        help="Comma-separated seasons to fetch",
    )
    parser.add_argument("--teams", type=int, default=12)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing snapshots (including consensus_2026.json)",
    )
    args = parser.parse_args()
    seasons = [int(s.strip()) for s in args.seasons.split(",") if s.strip()]
    for season in seasons:
        path = write_snapshot(season, teams=args.teams, force=args.force)
        with path.open(encoding="utf-8") as fh:
            meta = json.load(fh)["meta"]
        print(
            f"{season}: wrote {path.name} n={meta['player_count']} "
            f"adp={meta['matched_adp']} ecr={meta['matched_ecr']} "
            f"as_of={meta.get('as_of')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
