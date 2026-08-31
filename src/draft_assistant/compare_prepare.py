"""Build comparison JSON: our board + free external ECR/ADP."""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from datetime import datetime, timezone
from typing import Any

import pandas as pd

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DRAFT_DATA_DIR = os.path.join(REPO_ROOT, "draft_assistant", "data")

FFC_UA = "fantasy-projections-draft-assistant/1.0 (personal; +https://fantasyfootballcalculator.com attribution)"


def _norm_name(name: str | None) -> str:
    if not name:
        return ""
    s = str(name).lower().strip()
    s = s.replace(".", "").replace("'", "").replace("’", "")
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _norm_team(team: str | None) -> str:
    if not team:
        return ""
    t = str(team).upper().strip()
    aliases = {"JAC": "JAX", "WSH": "WAS", "WAS": "WAS", "LA": "LAR", "STL": "LAR"}
    return aliases.get(t, t)


def load_our_players(season: int) -> list[dict]:
    path = os.path.join(DRAFT_DATA_DIR, f"players_{season}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Run `python -m src.draft_assistant.prepare --season {season}` first."
        )
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload.get("players") or []


def fetch_ffc_adp(*, scoring: str = "half-ppr", teams: int = 12) -> tuple[list[dict], dict]:
    """Fantasy Football Calculator free ADP API (attribution required)."""
    url = f"https://fantasyfootballcalculator.com/api/v1/adp/{scoring}?teams={teams}"
    req = urllib.request.Request(url, headers={"User-Agent": FFC_UA})
    with urllib.request.urlopen(req, timeout=45) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if payload.get("status") != "Success":
        raise RuntimeError(f"FFC ADP failed: {payload.get('status')}")
    meta = payload.get("meta") or {}
    players = payload.get("players") or []
    return players, {"url": url, "scoring": scoring, "teams": teams, **meta}


def load_fp_ecr_ppr() -> tuple[pd.DataFrame, dict]:
    """FantasyPros PPR ECR via nflverse/DynastyProcess redistribute."""
    try:
        import nflreadpy as nfl
    except ImportError as exc:
        raise ImportError(
            "nflreadpy is required for ECR. pip install nflreadpy"
        ) from exc

    rankings = nfl.load_ff_rankings(type="draft")
    if hasattr(rankings, "to_pandas"):
        df = rankings.to_pandas()
    else:
        df = rankings
    page = "/nfl/rankings/ppr-cheatsheets.php"
    ecr = df[df["fp_page"] == page].copy()
    if ecr.empty:
        # fallback: any overall ppr-like cheat sheet
        mask = df["fp_page"].astype(str).str.contains("ppr-cheatsheets", na=False)
        ecr = df[mask].copy()
    ecr = ecr[ecr["pos"].isin(["QB", "RB", "WR", "TE"])].copy()
    scrape = None
    if "scrape_date" in ecr.columns and len(ecr):
        scrape = str(ecr["scrape_date"].iloc[0])
    meta = {
        "source": "FantasyPros ECR via nflverse/DynastyProcess",
        "fp_page": page,
        "scrape_date": scrape,
        "note": "PPR consensus cheatsheet (closest free ECR to half-PPR board)",
    }
    return ecr, meta


def load_id_map() -> pd.DataFrame:
    import nflreadpy as nfl

    ids = nfl.load_ff_playerids()
    if hasattr(ids, "to_pandas"):
        return ids.to_pandas()
    return ids


def _join_keys(name: str, pos: str, team: str | None) -> tuple[str, str, str]:
    return _norm_name(name), str(pos or "").upper(), _norm_team(team)


def build_comparison(
    *,
    season: int = 2026,
    team_count: int = 12,
    ffc_scoring: str = "half-ppr",
) -> dict[str, Any]:
    ours = load_our_players(season)
    ffc_players, ffc_meta = fetch_ffc_adp(scoring=ffc_scoring, teams=team_count)
    ecr_df, ecr_meta = load_fp_ecr_ppr()
    id_map = load_id_map()

    # fantasypros_id -> gsis_id
    fp_to_gsis: dict[str, str] = {}
    if "fantasypros_id" in id_map.columns and "gsis_id" in id_map.columns:
        for _, row in id_map.dropna(subset=["fantasypros_id"]).iterrows():
            fp_id = str(int(row["fantasypros_id"])) if pd.notna(row["fantasypros_id"]) else None
            gsis = row.get("gsis_id")
            if fp_id and pd.notna(gsis):
                fp_to_gsis[fp_id] = str(gsis)

    # name+pos(+team) -> ecr row
    ecr_by_key: dict[tuple[str, str], dict] = {}
    ecr_by_key_team: dict[tuple[str, str, str], dict] = {}
    ecr_by_gsis: dict[str, dict] = {}
    for _, row in ecr_df.iterrows():
        rec = {
            "ecr": float(row["ecr"]) if pd.notna(row.get("ecr")) else None,
            "ecr_sd": float(row["sd"]) if pd.notna(row.get("sd")) else None,
            "ecr_best": float(row["best"]) if pd.notna(row.get("best")) else None,
            "ecr_worst": float(row["worst"]) if pd.notna(row.get("worst")) else None,
            "fp_id": str(int(row["id"])) if pd.notna(row.get("id")) else None,
            "player": row.get("player"),
            "pos": row.get("pos"),
            "team": row.get("team") or row.get("tm"),
        }
        n, p, t = _join_keys(rec["player"], rec["pos"], rec["team"])
        ecr_by_key[(n, p)] = rec
        if t:
            ecr_by_key_team[(n, p, t)] = rec
        if rec["fp_id"] and rec["fp_id"] in fp_to_gsis:
            ecr_by_gsis[fp_to_gsis[rec["fp_id"]]] = rec

    ffc_by_key: dict[tuple[str, str], dict] = {}
    ffc_by_key_team: dict[tuple[str, str, str], dict] = {}
    for row in ffc_players:
        rec = {
            "adp": float(row["adp"]) if row.get("adp") is not None else None,
            "adp_stdev": float(row["stdev"]) if row.get("stdev") is not None else None,
            "adp_high": row.get("high"),
            "adp_low": row.get("low"),
            "times_drafted": row.get("times_drafted"),
            "bye": row.get("bye"),
        }
        n, p, t = _join_keys(row.get("name"), row.get("position"), row.get("team"))
        ffc_by_key[(n, p)] = rec
        if t:
            ffc_by_key_team[(n, p, t)] = rec

    rows: list[dict] = []
    for p in ours:
        pid = str(p.get("player_id"))
        name = p.get("display_name")
        pos = p.get("position")
        team = p.get("team")
        n, po, t = _join_keys(name, pos, team)

        ecr = ecr_by_gsis.get(pid)
        if ecr is None:
            ecr = ecr_by_key_team.get((n, po, t)) or ecr_by_key.get((n, po))
        adp = ffc_by_key_team.get((n, po, t)) or ffc_by_key.get((n, po))

        our_rank = p.get("overall_rank")
        ecr_rank = ecr.get("ecr") if ecr else None
        adp_val = adp.get("adp") if adp else None

        def _delta(ours_r, theirs):
            if ours_r is None or theirs is None:
                return None
            return round(float(ours_r) - float(theirs), 2)

        rows.append(
            {
                "player_id": pid,
                "display_name": name,
                "position": pos,
                "team": team,
                "our_rank": our_rank,
                "our_pos_rank": p.get("pos_rank"),
                "fantasy_pts": p.get("fantasy_pts"),
                "fantasy_pts_season": p.get("fantasy_pts_season"),
                "vorp": p.get("vorp"),
                "ecr": round(ecr_rank, 2) if ecr_rank is not None else None,
                "ecr_sd": ecr.get("ecr_sd") if ecr else None,
                "adp": round(adp_val, 2) if adp_val is not None else None,
                "adp_stdev": adp.get("adp_stdev") if adp else None,
                "delta_ecr": _delta(our_rank, ecr_rank),
                "delta_adp": _delta(our_rank, adp_val),
                "matched_ecr": ecr is not None,
                "matched_adp": adp is not None,
                "sentiment_score": p.get("sentiment_score"),
                "sentiment_confidence": p.get("sentiment_confidence"),
                "sentiment_coverage": p.get("sentiment_coverage"),
                "sentiment_as_of": p.get("sentiment_as_of"),
                "sentiment_model_active": p.get("sentiment_model_active"),
            }
        )

    rows.sort(key=lambda r: (r["our_rank"] is None, r["our_rank"] or 9999))

    matched_ecr = sum(1 for r in rows if r["matched_ecr"])
    matched_adp = sum(1 for r in rows if r["matched_adp"])

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
                **{k: v for k, v in ffc_meta.items() if k in ("url", "scoring", "teams", "total_drafts", "start_date", "end_date")},
            },
            "delta_note": "delta = our_rank − market (negative = we rank them higher)",
        },
        "players": rows,
    }


def rebase_comparison_payload(board: dict, comparison: dict) -> dict[str, Any]:
    """Replace only our side of a frozen market comparison.

    ECR/ADP snapshots are time-sensitive.  When a new board is selected after
    those snapshots were captured, rebuilding the comparison should not
    silently refetch a different market.  This transform preserves every
    market field and updates our rank, points, VORP, sentiment, and deltas.
    """
    board_players = board.get("players") or []
    old_by_id = {
        str(row.get("player_id")): row
        for row in (comparison.get("players") or [])
        if row.get("player_id") is not None
    }

    def delta(our_rank, market_rank):
        if our_rank is None or market_rank is None:
            return None
        return round(float(our_rank) - float(market_rank), 2)

    rows: list[dict] = []
    for player in board_players:
        player_id = str(player.get("player_id"))
        old = dict(old_by_id.get(player_id) or {})
        old.update(
            {
                "player_id": player_id,
                "display_name": player.get("display_name"),
                "position": player.get("position"),
                "team": player.get("team"),
                "our_rank": player.get("overall_rank"),
                "our_pos_rank": player.get("pos_rank"),
                "fantasy_pts": player.get("fantasy_pts"),
                "fantasy_pts_season": player.get("fantasy_pts_season"),
                "vorp": player.get("vorp"),
                "sentiment_score": player.get("sentiment_score"),
                "sentiment_confidence": player.get("sentiment_confidence"),
                "sentiment_coverage": player.get("sentiment_coverage"),
                "sentiment_as_of": player.get("sentiment_as_of"),
                "sentiment_model_active": player.get("sentiment_model_active"),
            }
        )
        old["matched_ecr"] = old.get("ecr") is not None
        old["matched_adp"] = old.get("adp") is not None
        old["delta_ecr"] = delta(old.get("our_rank"), old.get("ecr"))
        old["delta_adp"] = delta(old.get("our_rank"), old.get("adp"))
        rows.append(old)

    rows.sort(key=lambda row: (row.get("our_rank") is None, row.get("our_rank") or 9999))
    board_meta = board.get("meta") or {}
    meta = dict(comparison.get("meta") or {})
    meta.update(
        {
            "season": board_meta.get("season", meta.get("season")),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "player_count": len(rows),
            "matched_ecr": sum(1 for row in rows if row["matched_ecr"]),
            "matched_adp": sum(1 for row in rows if row["matched_adp"]),
            "board_generated_at": board_meta.get("generated_at"),
            "board_model_id": board_meta.get("model_id"),
            "board_source_file": board_meta.get("source_file"),
            "market_snapshot_preserved": True,
        }
    )
    return {"meta": meta, "players": rows}


def rebase_comparison_file(
    players_path: str,
    comparison_path: str,
    *,
    out_path: str | None = None,
) -> str:
    with open(players_path, encoding="utf-8") as fh:
        board = json.load(fh)
    with open(comparison_path, encoding="utf-8") as fh:
        comparison = json.load(fh)
    payload = rebase_comparison_payload(board, comparison)
    destination = out_path or comparison_path
    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
    with open(destination, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, allow_nan=False)
    return destination


def export_comparison(
    season: int = 2026,
    *,
    team_count: int = 12,
    ffc_scoring: str = "half-ppr",
) -> str:
    payload = build_comparison(
        season=season, team_count=team_count, ffc_scoring=ffc_scoring
    )
    os.makedirs(DRAFT_DATA_DIR, exist_ok=True)
    out_path = os.path.join(DRAFT_DATA_DIR, f"comparison_{season}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, allow_nan=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export rankings comparison data")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--teams", type=int, default=12, help="FFC ADP league size")
    parser.add_argument(
        "--ffc-scoring",
        default="half-ppr",
        choices=["half-ppr", "ppr", "standard"],
    )
    parser.add_argument(
        "--players-path",
        default=None,
        help="Rebase a frozen comparison onto this players JSON without refetching markets",
    )
    parser.add_argument(
        "--comparison-path",
        default=None,
        help="Existing comparison JSON whose ECR/ADP snapshot should be preserved",
    )
    parser.add_argument("--out", default=None, help="Optional comparison JSON output path")
    args = parser.parse_args()
    if args.players_path:
        comparison_path = args.comparison_path or os.path.join(
            DRAFT_DATA_DIR, f"comparison_{args.season}.json"
        )
        path = rebase_comparison_file(
            args.players_path,
            comparison_path,
            out_path=args.out,
        )
        print(f"Wrote {path}")
        return
    path = export_comparison(
        args.season, team_count=args.teams, ffc_scoring=args.ffc_scoring
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
