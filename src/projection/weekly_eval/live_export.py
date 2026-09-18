"""Export Role 1 weekly_props snapshots into a leakage-safe Role 2 CSV.

Role 2 consumes ``live_snapshots.csv`` with gsis ``player_id``, ``as_of``,
and schedule ``kickoff_at`` (``as_of <= kickoff_at``). This module only builds
that frame — it does not promote, blend (Role 3), or touch sealed pointers.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from src.ingest.props.contracts import (
    PLAYER_MARKET_ALIASES,
    NormalizedQuote,
    ProviderSnapshot,
)
from src.ingest.props.normalize import canonicalize_market
from src.projection.market_quotes import robust_median
from src.projection.weekly_eval.schema import ALLOWED_MARKETS
from src.projection.weekly_latent.availability import kickoff_at_from_row
from src.projection.weekly_props.identity import resolve_quote_group_key

ExportMode = Literal["consensus", "single"]

LIVE_SNAPSHOT_COLUMNS: tuple[str, ...] = (
    "player_id",
    "player_name",
    "season",
    "week",
    "market",
    "line",
    "implied_p_over",
    "implied_mean",
    "as_of",
    "kickoff_at",
    "source",
)

DEFAULT_SCHEDULE_REL = (
    "src/projection/weekly_latent/fixtures/nfl_schedules_2026_reg.csv"
)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _iso(value: datetime) -> str:
    return _aware_utc(value).isoformat().replace("+00:00", "Z")


def is_gsis_shaped(player_id: str | None) -> bool:
    return bool(player_id) and str(player_id).startswith("00-")


def canonicalize_role2_market(name: str) -> str | None:
    """Map a provider market name onto Role 2 ``ALLOWED_MARKETS``."""
    canon = canonicalize_market(name)
    if canon is None:
        key = str(name or "").strip().lower().replace(" ", "_").replace("-", "_")
        canon = PLAYER_MARKET_ALIASES.get(key)
    if canon is None or canon not in ALLOWED_MARKETS:
        return None
    return canon


def build_identity_map_from_records(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Index identity rows the same way ``build_identity_map`` indexes baselines.

    Each record needs a projection id (gsis preferred) plus optional name/team/pos.
    """
    from src.app.availability.identity import normalize_name
    from src.draft_assistant.market_adp import canonicalize_player_name

    identity_map: dict[str, dict[str, Any]] = {}
    by_name: dict[str, list[str]] = {}
    for raw in records:
        gsis = raw.get("gsis_id") or raw.get("player_id")
        if not gsis:
            continue
        pid = str(gsis)
        # Prefer an explicit gsis column when the PK is sleeper-shaped.
        if raw.get("gsis_id"):
            pid = str(raw["gsis_id"])
        elif not is_gsis_shaped(pid):
            continue
        payload = {
            "player_id": pid,
            "name": raw.get("name") or raw.get("player_name"),
            "team": raw.get("team"),
            "position": raw.get("position") or raw.get("pos"),
            "opponent": raw.get("opponent"),
        }
        identity_map[pid] = payload
        # Also index the raw PK when it differs (sleeper id → gsis).
        raw_pk = raw.get("player_id")
        if raw_pk and str(raw_pk) != pid:
            identity_map[str(raw_pk)] = payload
        sleeper = raw.get("sleeper_id")
        if sleeper:
            identity_map[str(sleeper)] = payload
        canon = canonicalize_player_name(str(payload.get("name") or ""))
        norm = normalize_name(str(payload.get("name") or ""))
        if canon:
            by_name.setdefault(canon, []).append(pid)
        if norm and norm != canon:
            by_name.setdefault(norm, []).append(pid)
        team = str(payload.get("team") or "").upper()
        pos = str(payload.get("position") or "").upper()
        if canon and team:
            identity_map[f"{canon}|{team}"] = payload
            if pos:
                identity_map[f"{canon}|{team}|{pos}"] = payload
        if norm and team:
            identity_map[f"{norm}|{team}"] = payload
    for name, ids in by_name.items():
        unique = sorted(set(ids))
        if len(unique) == 1:
            identity_map[name] = identity_map[unique[0]]
    return identity_map


def build_identity_map_from_player_identity_rows(
    rows: Iterable[Any],
) -> dict[str, dict[str, Any]]:
    """Build an identity map from ORM ``PlayerIdentity`` rows (or duck-typed)."""
    from src.app.availability.gsis_link import projection_id_for_identity

    records: list[dict[str, Any]] = []
    for row in rows:
        # Prefer projection_id_for_identity when available.
        try:
            pid = projection_id_for_identity(row)
        except Exception:  # noqa: BLE001 — duck-typed fixtures
            pid = getattr(row, "gsis_id", None) or getattr(row, "player_id", None)
        if not is_gsis_shaped(str(pid) if pid else None):
            continue
        records.append(
            {
                "player_id": getattr(row, "player_id", None),
                "gsis_id": getattr(row, "gsis_id", None) or pid,
                "sleeper_id": getattr(row, "sleeper_id", None),
                "name": getattr(row, "name", None),
                "position": getattr(row, "position", None),
                "team": getattr(row, "team", None),
            }
        )
    return build_identity_map_from_records(records)


def kickoffs_by_team_from_schedule(
    schedule: pd.DataFrame,
    *,
    season: int,
    week: int,
) -> dict[str, datetime]:
    """Map team abbreviation → kickoff datetime for one slate week."""
    frame = schedule.copy()
    if "season" in frame.columns:
        frame = frame[frame["season"].astype(int) == int(season)]
    if "week" in frame.columns:
        frame = frame[frame["week"].astype(int) == int(week)]
    out: dict[str, datetime] = {}
    for row in frame.to_dict(orient="records"):
        kickoff = kickoff_at_from_row(
            kickoff_at=row.get("kickoff_at"),
            gameday=row.get("gameday"),
            gametime=row.get("gametime"),
        )
        if kickoff is None:
            continue
        kickoff = _aware_utc(kickoff)
        for side in ("home_team", "away_team"):
            team = str(row.get(side) or "").upper().strip()
            if team:
                out[team] = kickoff
    return out


def load_schedule_kickoffs(
    path: Path | str,
    *,
    season: int,
    week: int,
) -> dict[str, datetime]:
    from src.projection.weekly_latent.schedule import load_schedule_csv

    return kickoffs_by_team_from_schedule(
        load_schedule_csv(path), season=season, week=week
    )


def _resolve_kickoff(
    quote: NormalizedQuote,
    *,
    kickoffs_by_team: Mapping[str, datetime],
) -> datetime | None:
    team = str(quote.team or "").upper()
    if team and team in kickoffs_by_team:
        return _aware_utc(kickoffs_by_team[team])
    # Opponent home/away unknown — try opponent only as last schedule miss.
    opponent = str(quote.opponent or "").upper()
    if opponent and opponent in kickoffs_by_team:
        return _aware_utc(kickoffs_by_team[opponent])
    if quote.event_start is not None:
        return _aware_utc(quote.event_start)
    return None


def _quote_as_of(quote: NormalizedQuote, snap_fetched_at: datetime) -> datetime:
    return _aware_utc(quote.fetched_at or snap_fetched_at)


def _mean_implied_p_over(quotes: Sequence[NormalizedQuote]) -> float | None:
    vals = [
        float(q.over_prob_novig)
        for q in quotes
        if q.over_prob_novig is not None
    ]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _row(
    *,
    player_id: str,
    player_name: str | None,
    season: int,
    week: int,
    market: str,
    line: float,
    implied_p_over: float | None,
    as_of: datetime,
    kickoff_at: datetime,
    source: str,
) -> dict[str, Any]:
    return {
        "player_id": player_id,
        "player_name": player_name,
        "season": int(season),
        "week": int(week),
        "market": market,
        "line": float(line),
        "implied_p_over": implied_p_over,
        "implied_mean": float(line),
        "as_of": _iso(as_of),
        "kickoff_at": _iso(kickoff_at),
        "source": source,
    }


def export_role2_snapshot_rows(
    snapshots: Sequence[ProviderSnapshot],
    *,
    identity_map: Mapping[str, dict[str, Any]] | None = None,
    kickoffs_by_team: Mapping[str, datetime] | None = None,
    mode: ExportMode = "consensus",
    require_gsis: bool = True,
) -> list[dict[str, Any]]:
    """Turn provider snapshots into Role 2 CSV rows.

    ``consensus`` merges books with a robust median (source=``consensus``).
    ``single`` keeps one row per book with a ``source`` column.

    Rows with unresolved / non-gsis ids are dropped when ``require_gsis`` is
    True. Post-kickoff quotes (``as_of > kickoff_at``) are dropped, never
    written — fail closed for leakage.
    """
    identity_map = identity_map or {}
    kickoffs_by_team = kickoffs_by_team or {}

    # (player_id, market) → list[(quote, as_of, kickoff, meta)]
    grouped: dict[tuple[str, str], list[tuple[NormalizedQuote, datetime, datetime, dict[str, Any]]]] = (
        defaultdict(list)
    )
    season = None
    week = None

    for snap in snapshots:
        if not snap.success:
            continue
        season = int(snap.season) if season is None else season
        week = int(snap.week) if week is None else week
        for quote in snap.quotes:
            if quote.kind != "book" or quote.reject_reason:
                continue
            market = canonicalize_role2_market(quote.market)
            if market is None:
                continue
            key, ident = resolve_quote_group_key(quote, identity_map)
            player_id = str(ident.get("player_id") or quote.player_id or key)
            if require_gsis and not is_gsis_shaped(player_id):
                continue
            kickoff = _resolve_kickoff(quote, kickoffs_by_team=kickoffs_by_team)
            if kickoff is None:
                continue
            as_of = _quote_as_of(quote, snap.fetched_at)
            if as_of > kickoff:
                continue
            meta = {
                "player_name": ident.get("name") or quote.player_name_raw,
                "source": snap.source or quote.sportsbook or quote.source,
            }
            grouped[(player_id, market)].append((quote, as_of, kickoff, meta))

    if season is None or week is None:
        return []

    rows: list[dict[str, Any]] = []
    if mode == "single":
        for (player_id, market), items in sorted(grouped.items()):
            for quote, as_of, kickoff, meta in items:
                rows.append(
                    _row(
                        player_id=player_id,
                        player_name=meta.get("player_name"),
                        season=season,
                        week=week,
                        market=market,
                        line=float(quote.line),
                        implied_p_over=(
                            None
                            if quote.over_prob_novig is None
                            else float(quote.over_prob_novig)
                        ),
                        as_of=as_of,
                        kickoff_at=kickoff,
                        source=str(meta.get("source") or quote.source),
                    )
                )
        return rows

    # Consensus: one row per (player, market). Do not reuse promotion
    # evaluate_quote wall-clock gates (stale / event_started) — Role 2 only
    # needs pre-kickoff quotes already filtered above.
    for (player_id, market), items in sorted(grouped.items()):
        quotes = [q for q, _, _, _ in items]
        if not quotes:
            continue
        line = float(robust_median([q.line for q in quotes]))
        # as_of = latest contributing quote timestamp (still <= kickoff).
        as_of = max(as_of for _, as_of, _, _ in items)
        kickoff = min(kickoff for _, _, kickoff, _ in items)
        if as_of > kickoff:
            continue
        name = next(
            (m.get("player_name") for _, _, _, m in items if m.get("player_name")),
            None,
        )
        books = sorted(
            {
                str(m.get("source") or q.source).strip().lower()
                for q, _, _, m in items
            }
        )
        source = "consensus" if len(books) > 1 else (books[0] if books else "consensus")
        rows.append(
            _row(
                player_id=player_id,
                player_name=name,
                season=season,
                week=week,
                market=market,
                line=line,
                implied_p_over=_mean_implied_p_over(quotes),
                as_of=as_of,
                kickoff_at=kickoff,
                source=source,
            )
        )
    return rows


def rows_to_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=list(LIVE_SNAPSHOT_COLUMNS))
    frame = pd.DataFrame(list(rows))
    for col in LIVE_SNAPSHOT_COLUMNS:
        if col not in frame.columns:
            frame[col] = None
    return frame[list(LIVE_SNAPSHOT_COLUMNS)]


def write_live_snapshots_csv(
    rows: Sequence[Mapping[str, Any]],
    path: Path | str,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = rows_to_frame(rows)
    frame.to_csv(out, index=False)
    return out


def describe_s3_env_blocker() -> str:
    """Human-readable blocker when production artifact bodies cannot be loaded."""
    return (
        "Live Role 1 weekly_props bodies live at s3://fantasy-app/artifacts/... "
        "and are loaded via get_artifact_store().get_json(artifact_uri). "
        "Set ARTIFACT_BACKEND=s3 plus S3_ENDPOINT_URL, S3_BUCKET, "
        "S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY, S3_REGION, and a Postgres "
        "DATABASE_URL (or JOB_DATABASE_URL) that can read public.source_snapshot "
        "and public.player_identity. Then re-run:\n"
        "  uv run python scripts/export_role2_live_props.py "
        "--season 2026 --week 2\n"
        "Fixture / offline path (CI, no S3):\n"
        "  uv run python scripts/export_role2_live_props.py "
        "--from-fixtures --season 2026 --week 1"
    )


def missing_live_export_env() -> list[str]:
    """Return missing env var names required for the DB+S3 export path."""
    import os

    missing: list[str] = []
    if not (os.environ.get("DATABASE_URL") or os.environ.get("JOB_DATABASE_URL")):
        missing.append("DATABASE_URL|JOB_DATABASE_URL")
    backend = (os.environ.get("ARTIFACT_BACKEND") or "local").strip().lower()
    if backend != "s3":
        missing.append("ARTIFACT_BACKEND=s3")
    for key in (
        "S3_ENDPOINT_URL",
        "S3_ACCESS_KEY_ID",
        "S3_SECRET_ACCESS_KEY",
    ):
        if not os.environ.get(key):
            missing.append(key)
    return missing
