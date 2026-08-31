"""Fixed replacement ranks and points from the displayed accuracy-first board."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.draft_assistant.vorp import (
    DEFAULT_TEAM_COUNT,
    STARTERS,
    FLEX_SHARE,
    add_vorp_columns,
)
from src.projection.contracts import REPO_ROOT
from src.projection.evaluation.accuracy_first import canonical_json_hash, sha256_file
from src.projection.fantasy_points import SCORING

CONTRACT_VERSION = "phase2_fixed_replacement_v1"
DEFAULT_ROSTER_CONFIG = Path(REPO_ROOT) / "config" / "roster_default.json"
POSITIONS = ("QB", "RB", "WR", "TE")


def scoring_configuration_hash() -> str:
    return hashlib.sha256(
        json.dumps(SCORING, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_roster_configuration(path: Path | None = None) -> dict:
    config_path = Path(path or DEFAULT_ROSTER_CONFIG)
    return json.loads(config_path.read_text(encoding="utf-8"))


def roster_configuration_hash(config: dict) -> str:
    return canonical_json_hash(config)


def default_selected_board_path(season: int) -> Path:
    from src.projection.contracts import OUTPUT_DIR

    return Path(OUTPUT_DIR) / f"accuracy_first_{season}" / f"fantasy_points_{season}.csv"


def load_selected_board(
    season: int,
    *,
    board_path: Path | None = None,
) -> pd.DataFrame:
    path = Path(board_path or default_selected_board_path(season))
    if not path.exists():
        raise FileNotFoundError(f"Selected accuracy-first board missing: {path}")
    frame = pd.read_csv(path)
    frame["player_id"] = frame["player_id"].astype(str)
    frame = frame[frame["position"].isin(POSITIONS)].copy()
    if "selected_fantasy_points" not in frame.columns:
        frame["selected_fantasy_points"] = pd.to_numeric(
            frame["fantasy_pts_season"], errors="coerce"
        )
    return frame


def _replacement_player_at_rank(group: pd.DataFrame, rank: int) -> tuple[str, float]:
    ordered = group.sort_values(
        ["vorp_input_pts", "player_id"],
        ascending=[False, True],
    )
    idx = min(max(int(rank), 1), len(ordered)) - 1
    row = ordered.iloc[idx]
    return str(row["player_id"]), float(row["vorp_input_pts"])


def build_replacement_contract(
    board: pd.DataFrame,
    *,
    season: int,
    selected_board_hash: str,
    selected_board_model_id: str,
    canonical_projection_run_id: str,
    roster_config: dict | None = None,
    team_count: int | None = None,
) -> dict:
    """Derive fixed replacement ranks/points from the displayed board only."""
    roster_config = roster_config or load_roster_configuration()
    team_count = int(team_count or roster_config.get("league_size", DEFAULT_TEAM_COUNT))
    enriched = add_vorp_columns(
        board.copy(),
        team_count=team_count,
        points_col="selected_fantasy_points"
        if "selected_fantasy_points" in board.columns
        else "fantasy_pts_season",
        adjust_replacement_for_availability=bool(
            (roster_config.get("availability_adjustment_configuration") or {}).get(
                "enabled", True
            )
        ),
        curve_weight={"QB": 0.0, "RB": 0.0, "WR": 0.0, "TE": 0.0},
        curves={},
    )
    replacement_by_position: dict[str, dict[str, Any]] = {}
    ranks = enriched.attrs.get("vorp_replacement_ranks") or {}
    for position in POSITIONS:
        group = enriched[enriched["position"].astype(str).eq(position)]
        if group.empty:
            continue
        rank = int(ranks.get(position, 0))
        player_id, points = _replacement_player_at_rank(group, rank)
        replacement_pts = float(
            group.loc[group["player_id"].eq(player_id), "replacement_pts"].iloc[0]
        )
        replacement_by_position[position] = {
            "replacement_rank": rank,
            "replacement_player_id": player_id,
            "replacement_points": replacement_pts,
            "replacement_vorp_input_points": points,
        }
    payload = {
        "season": int(season),
        "selected_board_hash": str(selected_board_hash),
        "selected_board_model_id": str(selected_board_model_id),
        "canonical_projection_run_id": str(canonical_projection_run_id),
        "roster_configuration": roster_config,
        "roster_configuration_hash": roster_configuration_hash(roster_config),
        "scoring_configuration_hash": scoring_configuration_hash(),
        "replacement_by_position": replacement_by_position,
        "contract_version": CONTRACT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    # Hash content identity only — wall-clock generated_at must not affect contract_hash,
    # or overlay compares across rebuilds falsely hold on replacement_contract_hash.
    body = dict(payload)
    body.pop("contract_hash", None)
    body.pop("generated_at", None)
    payload["contract_hash"] = canonical_json_hash(body)
    return payload


def replacement_points_map(contract: dict) -> dict[str, float]:
    return {
        str(pos): float(spec["replacement_points"])
        for pos, spec in (contract.get("replacement_by_position") or {}).items()
    }


def contract_output_dir(season: int, selected_board_hash: str) -> Path:
    from src.projection.contracts import OUTPUT_DIR

    return (
        Path(OUTPUT_DIR)
        / "model_v3"
        / "simulated_vorp"
        / f"season={season}"
        / f"board={selected_board_hash}"
    )


def write_replacement_contract(contract: dict, path: Path | None = None) -> Path:
    out = Path(
        path
        or contract_output_dir(
            int(contract["season"]), str(contract["selected_board_hash"])
        )
        / "replacement_contract.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return out


def read_replacement_contract(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
