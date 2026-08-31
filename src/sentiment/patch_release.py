"""Patch diagnostic sentiment fields onto a sealed release without changing forecasts."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from src.draft_assistant.prepare import build_sentiment_meta
from src.projection.active_release import read_active_pointer, write_active_pointer
from src.projection.release_bundle import (
    MANIFEST_FILENAME,
    bundle_root,
    load_sealed_manifest,
    public_release_dir,
    seal_manifest,
    sha256_file,
)
from src.sentiment.snapshot import build_sentiment_snapshot

DIAGNOSTIC_FIELDS = (
    "sentiment_tone",
    "sentiment_peer_label",
    "sentiment_evidence_tier",
)


def _json_value(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return value


def patch_players_payload(payload: dict, *, season: int, fantasy_points: pd.DataFrame) -> dict:
    """Merge diagnostic sentiment onto an existing players JSON payload."""
    generated_at = payload["meta"]["generated_at"]
    base = fantasy_points.drop_duplicates("player_id").copy()
    as_of_values = (
        base["sentiment_as_of"].dropna().astype(str).unique().tolist()
        if "sentiment_as_of" in base.columns
        else []
    )
    if len(as_of_values) > 1:
        raise ValueError(f"Mixed sentiment_as_of in bundle board: {as_of_values}")
    as_of = as_of_values[0][:10] if as_of_values else None
    snapshot = build_sentiment_snapshot(base, season=season, as_of=as_of)
    diag = snapshot.set_index("player_id")[list(DIAGNOSTIC_FIELDS)]
    for player in payload.get("players") or []:
        player_id = str(player["player_id"])
        if player_id not in diag.index:
            continue
        for field in DIAGNOSTIC_FIELDS:
            player[field] = _json_value(diag.at[player_id, field])
    payload = dict(payload)
    payload["meta"] = dict(payload["meta"])
    payload["meta"]["sentiment"] = build_sentiment_meta(
        season, snapshot, generated_at=generated_at
    )
    return payload


def patch_release_sentiment_diagnostics(*, season: int, namespace: str) -> dict:
    """Update players artifact + manifest hash for one sealed namespace."""
    root = bundle_root(season, namespace)
    manifest, _ = load_sealed_manifest(root)
    players_entry = next(entry for entry in manifest["artifacts"] if entry["role"] == "players")
    players_rel = players_entry["path"]
    players_path = root / players_rel
    fantasy_path = root / "fantasy_points_2026.csv"
    if not fantasy_path.is_file():
        raise FileNotFoundError(f"Missing bundle board for sentiment patch: {fantasy_path}")

    payload = json.loads(players_path.read_text(encoding="utf-8"))
    patched = patch_players_payload(
        payload,
        season=season,
        fantasy_points=pd.read_csv(fantasy_path),
    )
    players_path.write_text(json.dumps(patched, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    players_entry["sha256"] = sha256_file(players_path)
    players_entry["byte_size"] = int(players_path.stat().st_size)
    manifest, digest = seal_manifest(manifest, root=root)

    public = public_release_dir(namespace)
    public.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / MANIFEST_FILENAME, public / MANIFEST_FILENAME)
    shutil.copy2(players_path, public / players_rel)

    pointer_updated = False
    pointer = read_active_pointer(season)
    if pointer and str(pointer["namespace"]) == str(namespace):
        updated = dict(pointer)
        updated["manifest_sha256"] = digest
        write_active_pointer(updated)
        pointer_updated = True

    return {
        "namespace": namespace,
        "manifest_sha256": digest,
        "players_sha256": players_entry["sha256"],
        "pointer_updated": pointer_updated,
        "player_count": len(patched.get("players") or []),
        "sentiment_meta": patched["meta"].get("sentiment"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--namespace", required=True)
    args = parser.parse_args()
    report = patch_release_sentiment_diagnostics(
        season=args.season,
        namespace=args.namespace,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
