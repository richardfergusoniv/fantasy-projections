"""Resolve Role 2 snapshot CSVs: supplied, M3 fixture, or live path."""
from __future__ import annotations

import os
from pathlib import Path

from src.projection.contracts import REPO_ROOT

LIVE_PROPS_ENV = "WEEKLY_EVAL_PROPS_PATH"
DEFAULT_LIVE_PROPS_REL = "output/shadow_vegas_props_compare/live_snapshots.csv"
M3_SYNTHETIC_PROPS_REL = (
    "src/projection/weekly_latent/fixtures/vegas_props_m3_synthetic.csv"
)

LIVE_DATA_MUST_SUPPLY = (
    (
        "CSV of timestamped Vegas weekly prop snapshots at WEEKLY_EVAL_PROPS_PATH "
        "or output/shadow_vegas_props_compare/live_snapshots.csv"
    ),
    (
        "player_id matching the M3 / sealed board (gsis-style ids such as 00-0034857), "
        "not dry-run names (p-qb / p-wr)"
    ),
    "season, week, market using Role 2 names (pass_yards, rec_yards, receptions, ...)",
    "as_of (ISO-8601 UTC) and kickoff_at with as_of <= kickoff_at",
    "line or implied_mean; implied_p_over when de-vig is available",
    "optional outcomes CSV after the week: player_id, season, week, market, actual",
)


def live_data_blocker(*, n_board: int = 0) -> dict[str, object]:
    return {
        "status": "blocked",
        "n_board": int(n_board),
        "must_supply": list(LIVE_DATA_MUST_SUPPLY),
        "note": (
            "Role 1 weekly_props snapshots live in the DB / artifact store, not "
            "as committed as_of CSVs. Export a leakage-safe Role 2 frame before "
            "kickoff; do not reuse a later closing line as the historical as-of."
        ),
    }


def resolve_snapshots_path(
    *,
    snapshots_path: str | Path | None = None,
    dry_run: bool = False,
    repo_root: str | Path | None = None,
) -> tuple[Path | None, str]:
    """Return ``(path, source)``.

    Source is one of: supplied, m3_synthetic_fixture, live_timestamped,
    missing_live_snapshots.
    """
    root = Path(repo_root or REPO_ROOT)
    if snapshots_path is not None:
        return Path(snapshots_path), "supplied"
    if dry_run:
        return root / M3_SYNTHETIC_PROPS_REL, "m3_synthetic_fixture"
    env = os.environ.get(LIVE_PROPS_ENV, "").strip()
    if env:
        env_path = Path(env)
        if env_path.is_file():
            return env_path, "live_timestamped"
    live = root / DEFAULT_LIVE_PROPS_REL
    if live.is_file():
        return live, "live_timestamped"
    return None, "missing_live_snapshots"
