"""Write Milestone 1 shadow artifacts. Large player-week tables are gitignored."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.projection.weekly_latent.constants import SCHEMA_VERSION

SUMMARY_NAME = "summary.json"
CONSERVATION_NAME = "conservation.json"
SAMPLE_NAME = "sample_player_weeks.csv"
TEAM_WEEKS_NAME = "team_weeks.csv"
PLAYER_WEEKS_NAME = "player_weeks.csv"
SHARES_NAME = "shares.csv"
README_NAME = "README.md"

COMMITTED_NAMES = (SUMMARY_NAME, CONSERVATION_NAME, SAMPLE_NAME, README_NAME)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value).replace("\\", "/")
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_jsonable) + "\n", encoding="utf-8")


def write_shadow_outputs(
    output_dir: str | Path,
    *,
    team_weeks: pd.DataFrame,
    player_weeks: pd.DataFrame,
    shares: pd.DataFrame,
    conservation: Mapping[str, Any],
    summary: Mapping[str, Any],
    sample_rows: int = 40,
    readme_title: str | None = None,
    extra_json: Mapping[str, Mapping[str, Any]] | None = None,
    cli_name: str | None = None,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    team_weeks.to_csv(out / TEAM_WEEKS_NAME, index=False)
    player_weeks.to_csv(out / PLAYER_WEEKS_NAME, index=False)
    shares.to_csv(out / SHARES_NAME, index=False)
    sample_cols = [
        c
        for c in (
            "player_id",
            "display_name",
            "team",
            "position",
            "week",
            "opponent",
            "is_home",
            "is_bye",
            "A_i_w",
            "available_at",
            "attempts",
            "targets",
            "carries",
            "receiving_yards",
            "rushing_yards",
            "fantasy_points",
        )
        if c in player_weeks.columns
    ]
    sample = player_weeks
    if "team" in sample.columns and "BUF" in set(sample["team"].astype(str)):
        sample = sample[sample["team"].astype(str).eq("BUF")]
    sample = sample.sort_values(["week", "fantasy_points"], ascending=[True, False]).head(sample_rows)
    sample[sample_cols].to_csv(out / SAMPLE_NAME, index=False)
    write_json(out / CONSERVATION_NAME, dict(conservation))
    schema = summary.get("schema_version", SCHEMA_VERSION)
    payload = {
        "schema_version": schema,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **{k: _jsonable(v) for k, v in summary.items()},
    }
    write_json(out / SUMMARY_NAME, payload)
    paths = {
        "summary": str(out / SUMMARY_NAME),
        "conservation": str(out / CONSERVATION_NAME),
        "sample": str(out / SAMPLE_NAME),
        "team_weeks": str(out / TEAM_WEEKS_NAME),
        "player_weeks": str(out / PLAYER_WEEKS_NAME),
    }
    if extra_json:
        for name, body in extra_json.items():
            write_json(out / name, dict(body))
            paths[name] = str(out / name)
    (out / README_NAME).write_text(
        _readme_text(
            payload,
            conservation,
            title=readme_title,
            cli_name=cli_name,
        ),
        encoding="utf-8",
    )
    return paths


def _readme_text(
    summary: Mapping[str, Any],
    conservation: Mapping[str, Any],
    *,
    title: str | None = None,
    cli_name: str | None = None,
) -> str:
    failing = conservation.get("failing_checks") or []
    heading = title or "Shadow weekly schedule allocation (Milestone 1)"
    script = cli_name or (
        "scripts/run_weekly_schedule_m2.py"
        if int(summary.get("milestone") or 1) >= 2
        else "scripts/run_weekly_schedule_m1.py"
    )
    return (
        f"# {heading}\n\n"
        "Research/shadow only. This directory is **not** a sealed League Value "
        "board, not a Vegas replacement, and not a PWA input. "
        "Gate verdict: not promoting.\n\n"
        f"- Schema: `{summary.get('schema_version')}`\n"
        f"- Conservation / identity passes: `{conservation.get('passes')}`\n"
        f"- Failing checks: {failing or 'none'}\n"
        f"- Dry run: `{summary.get('dry_run')}`\n"
        f"- Still shadow: `{summary.get('still_shadow', summary.get('milestone') == 1)}`\n\n"
        "Large `player_weeks.csv` / `team_weeks.csv` / `shares.csv` are gitignored. "
        "`summary.json`, `conservation.json`, and `sample_player_weeks.csv` are "
        "the committed evidence files. Milestone 2 also commits "
        "`backtest_synthetic.json` and `backtest_historical.json`.\n\n"
        "Local run (Windows DB):\n\n"
        "```bat\n"
        "set FANTASY_PROJECTIONS_DATA_DIR=D:\\fantasy-projections-data\n"
        "set FANTASY_PROJECTIONS_DB_PATH=D:\\fantasy-projections-data\\projections.db\n"
        f"uv run python {script} --season 2026\n"
        "```\n"
    )
