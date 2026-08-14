"""CLI: detect injuries, propose depth/status changes, optionally apply."""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import pandas as pd

from src.depth_chart.events import detect_injury_events
from src.depth_chart.live import (
    build_live_depth_chart,
    load_curated_depth_chart,
    live_path,
    write_live_depth_chart,
)
from src.depth_chart.sleeper_status import ingest_sleeper_player_status

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(REPO_ROOT, "output")
DEPTH_DIR = os.path.join(REPO_ROOT, "src", "depth_chart")


def overrides_path(season: int) -> str:
    return os.path.join(DEPTH_DIR, f"status_overrides_{season}.csv")


def proposals_path(season: int) -> str:
    return os.path.join(OUTPUT_DIR, f"depth_refresh_proposals_{season}.csv")


def _merge_overrides(existing: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "season", "gsis_id", "player_name", "as_of_date", "mode",
        "projected_games", "reason",
    ]
    if existing is None or existing.empty:
        base = pd.DataFrame(columns=cols)
    else:
        base = existing.copy()
        for c in cols:
            if c not in base.columns:
                base[c] = pd.NA
        base = base[cols]

    if new_rows is None or new_rows.empty:
        return base

    add = new_rows.copy()
    for c in cols:
        if c not in add.columns:
            add[c] = pd.NA
    add = add[cols]

    # Newer as_of_date for the same gsis_id+mode replaces older.
    combined = pd.concat([base, add], ignore_index=True)
    combined["as_of_date"] = pd.to_datetime(combined["as_of_date"], errors="coerce")
    combined = combined.sort_values(["gsis_id", "mode", "as_of_date"])
    combined = combined.drop_duplicates(["gsis_id", "mode"], keep="last")
    combined["as_of_date"] = combined["as_of_date"].dt.strftime("%Y-%m-%d")
    return combined.reset_index(drop=True)


def events_to_override_rows(events: pd.DataFrame, season: int) -> pd.DataFrame:
    rows = []
    for _, e in events.iterrows():
        mode = e.get("override_mode")
        if mode is None or (isinstance(mode, float) and pd.isna(mode)):
            continue
        if pd.isna(e.get("gsis_id")):
            continue
        rows.append({
            "season": season,
            "gsis_id": e["gsis_id"],
            "player_name": e.get("player_name"),
            "as_of_date": e.get("as_of_date"),
            "mode": mode,
            "projected_games": e.get("override_games"),
            "reason": e.get("reason"),
        })
    return pd.DataFrame(rows)


def _players_id_lookup() -> pd.DataFrame:
    """Best-effort gsis lookup from the local players table + curated names."""
    from src.comparison.sleeper_compare import _normalize_name
    from src.projection.data_prep import get_conn

    frames = []
    try:
        conn = get_conn()
        try:
            players = pd.read_sql(
                "select gsis_id, display_name from players where gsis_id is not null",
                conn,
            )
            players["name_key"] = players["display_name"].map(_normalize_name)
            frames.append(players.rename(columns={"display_name": "player_name"}))
        finally:
            conn.close()
    except Exception:
        pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def refresh_depth_chart(
    season: int,
    apply: bool = False,
    force_ingest: bool = False,
    as_of: str | None = None,
    proposals_file: str | None = None,
) -> dict:
    """Ingest Sleeper status, write proposals, optionally apply live chart + overrides.

    Without ``--apply``, only the proposal CSV is written (curated unchanged).
    With ``--apply``, auto_safe and confirmed events update
    ``status_overrides_{season}.csv`` and ``live_depth_{season}.csv``.
    """
    as_of = as_of or datetime.now(timezone.utc).date().isoformat()
    status = ingest_sleeper_player_status(force=force_ingest)
    curated = load_curated_depth_chart(season)
    lookup = _players_id_lookup()
    events = detect_injury_events(
        status, curated, as_of=as_of, source="sleeper", id_lookup=lookup,
    )

    # If a prior proposals file marks confirmed=true, merge that flag by event_id.
    prop_path = proposals_file or proposals_path(season)
    if os.path.exists(prop_path) and not events.empty:
        prior = pd.read_csv(prop_path)
        if "event_id" in prior.columns and "confirmed" in prior.columns:
            conf = (
                prior.dropna(subset=["event_id"])
                .drop_duplicates("event_id")
                .set_index("event_id")["confirmed"]
            )
            mapped = events["event_id"].map(conf)
            events["confirmed"] = (
                mapped.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))
                | mapped.fillna(False).astype(bool)
            )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    events.to_csv(prop_path, index=False)

    result = {
        "proposals_path": prop_path,
        "n_events": len(events),
        "applied": False,
        "live_path": None,
        "overrides_path": None,
    }
    if not apply:
        return result

    live, applied = build_live_depth_chart(
        curated, events, as_of_date=as_of, apply_auto_safe=True, apply_confirmed=True,
    )
    live_out = write_live_depth_chart(live, season)

    ov_path = overrides_path(season)
    existing = pd.read_csv(ov_path) if os.path.exists(ov_path) else pd.DataFrame()
    new_ov = events_to_override_rows(applied, season)
    merged = _merge_overrides(existing, new_ov)
    merged.to_csv(ov_path, index=False)

    result.update({
        "applied": True,
        "live_path": live_out,
        "overrides_path": ov_path,
        "n_applied": len(applied),
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh depth chart from live injury status. "
            "Default writes proposals only; pass --apply to update "
            "live_depth + status_overrides for auto-safe/confirmed events. "
            "Ops loop after apply: python -m src.projection.predict --season N; "
            "python -m src.projection.fantasy_points --season N; "
            "python -m src.draft_assistant.prepare --season N && "
            "python -m src.team_stats.prepare --season N"
        )
    )
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply auto-safe/confirmed events to live_depth and status_overrides",
    )
    parser.add_argument(
        "--force-ingest",
        action="store_true",
        help="Re-fetch Sleeper players even if today's parquet exists",
    )
    parser.add_argument(
        "--as-of",
        default=None,
        help="ISO date for proposals/overrides (default: today UTC)",
    )
    parser.add_argument(
        "--proposals",
        default=None,
        help="Path to proposals CSV (read confirmed flags; write updated proposals)",
    )
    args = parser.parse_args()
    result = refresh_depth_chart(
        season=args.season,
        apply=args.apply,
        force_ingest=args.force_ingest,
        as_of=args.as_of,
        proposals_file=args.proposals,
    )
    print(
        f"Wrote {result['n_events']} proposal(s) to {result['proposals_path']}"
    )
    if result["applied"]:
        print(
            f"Applied {result['n_applied']} event(s): "
            f"live={result['live_path']} overrides={result['overrides_path']}"
        )
    else:
        print("Dry run only (pass --apply to update live_depth and overrides).")


if __name__ == "__main__":
    main()
