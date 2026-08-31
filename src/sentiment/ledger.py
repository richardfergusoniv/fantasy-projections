"""Legacy evidence ledger — one row per parsed mention, contradictions preserved."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.sentiment.markdown import (
    RESEARCH_AS_OF,
    TEAM_RESEARCH_FILES,
    iter_scored_team_claims,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESEARCH_DIR = REPO_ROOT / "perplexity research"
DEFAULT_LEDGER_DIR = REPO_ROOT / "data" / "sentiment" / "ledger"
IMPORTER_VERSION = "legacy_import_v1"
EVIDENCE_TIER_LEGACY = "legacy_unverified"


def _claim_id(source_file: str, line_number: int, player_id: str, label: str) -> str:
    payload = f"{source_file}|{line_number}|{player_id}|{label}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _content_hash(context: str) -> str:
    return hashlib.sha256(context.encode("utf-8")).hexdigest()


def build_legacy_claim(
    *,
    season: int,
    claim: dict,
) -> dict:
    """Materialize one ledger record from a scored parser row."""
    return {
        "claim_id": _claim_id(
            claim["source_file"],
            claim["line_number"],
            claim["player_id"],
            claim["parsed_label"],
        ),
        "content_hash": _content_hash(claim["context"]),
        "player_id": claim["player_id"],
        "display_name": claim["display_name"],
        "team": claim["team"],
        "position": claim["position"],
        "season": int(season),
        "source_file": claim["source_file"],
        "line_number": int(claim["line_number"]),
        "parsed_label": claim["parsed_label"],
        "polarity": claim["polarity"],
        "context": claim["context"],
        "extraction_method": claim["extraction_method"],
        "research_cutoff": RESEARCH_AS_OF.isoformat(),
        "importer_version": IMPORTER_VERSION,
        "source_url": None,
        "publication_date": None,
        "evidence_tier": EVIDENCE_TIER_LEGACY,
        "training_eligible": False,
    }


def assert_training_eligible_allowed(record: dict) -> None:
    """Legacy unverified claims are permanently excluded from training."""
    if record.get("evidence_tier") == EVIDENCE_TIER_LEGACY and record.get("training_eligible"):
        raise ValueError("legacy_unverified claims cannot be training-eligible")


def import_legacy_ledger(
    players: pd.DataFrame,
    *,
    season: int,
    research_dir: str | Path = DEFAULT_RESEARCH_DIR,
) -> list[dict]:
    """Import every scored mention from the 32 registered summaries."""
    root = Path(research_dir)
    missing = [name for name in TEAM_RESEARCH_FILES.values() if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing sentiment research files: {missing}")

    records: list[dict] = []
    for team, filename in TEAM_RESEARCH_FILES.items():
        team_players = players[players["team"].eq(team)]
        claims = iter_scored_team_claims(team_players, team, root / filename)
        for claim in claims:
            record = build_legacy_claim(season=season, claim=claim)
            assert_training_eligible_allowed(record)
            records.append(record)
    records.sort(key=lambda row: (row["source_file"], row["line_number"], row["player_id"], row["claim_id"]))
    return records


def audit_ledger(records: list[dict]) -> dict:
    """Summarize ledger contents for human review."""
    frame = pd.DataFrame(records)
    by_player = frame.groupby("player_id").size()
    missing_provenance = {
        "source_url_null": int(frame["source_url"].isna().sum()) if "source_url" in frame else 0,
        "publication_date_null": int(frame["publication_date"].isna().sum())
        if "publication_date" in frame
        else 0,
    }
    return {
        "importer_version": IMPORTER_VERSION,
        "claim_count": int(len(records)),
        "source_file_count": int(frame["source_file"].nunique()) if not frame.empty else 0,
        "player_count": int(frame["player_id"].nunique()) if not frame.empty else 0,
        "players_with_one_claim": int((by_player == 1).sum()) if not by_player.empty else 0,
        "players_with_multiple_claims": int((by_player > 1).sum()) if not by_player.empty else 0,
        "by_team": frame.groupby("team").size().sort_index().astype(int).to_dict() if not frame.empty else {},
        "by_polarity_sign": {
            "positive": int((frame["polarity"] > 0).sum()) if not frame.empty else 0,
            "neutral": int((frame["polarity"] == 0).sum()) if not frame.empty else 0,
            "negative": int((frame["polarity"] < 0).sum()) if not frame.empty else 0,
        },
        "by_extraction_method": frame.groupby("extraction_method").size().astype(int).to_dict()
        if not frame.empty
        else {},
        "missing_provenance": missing_provenance,
        "training_eligible_count": int(frame["training_eligible"].sum()) if not frame.empty else 0,
    }


def write_ledger(records: list[dict], out_path: str | Path, *, audit_path: str | Path | None = None) -> dict:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
    report = audit_ledger(records)
    audit_target = Path(audit_path) if audit_path else out_path.with_name(out_path.stem + "_audit.json")
    audit_target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--players-path", default=None)
    parser.add_argument("--research-dir", default=str(DEFAULT_RESEARCH_DIR))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    players_path = Path(args.players_path) if args.players_path else (
        REPO_ROOT / "output" / f"fantasy_points_{args.season}.csv"
    )
    players = pd.read_csv(players_path).drop_duplicates("player_id")
    records = import_legacy_ledger(players, season=args.season, research_dir=args.research_dir)
    out_path = Path(args.out) if args.out else (
        DEFAULT_LEDGER_DIR / f"legacy_{args.season}.jsonl"
    )
    report = write_ledger(records, out_path)
    print(json.dumps(report, indent=2))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
