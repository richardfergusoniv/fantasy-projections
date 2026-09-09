"""Build immutable Candidate rows from weekly props consensus + baseline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.app.releases.publication import Candidate, CandidateRow
from src.ingest.props.contracts import ConsensusManifest, PlayerConsensus
from src.projection.weekly_props.config import (
    ARTIFACT_MODE,
    DEFAULT_WEEKLY_POLICY,
    GATE_VERSION,
    MODEL_VERSION,
    WeeklyPropsPolicy,
)
from src.projection.weekly_props.scoring import build_mean_json


@dataclass(frozen=True)
class BaselinePlayer:
    player_id: str
    team: str | None
    opponent: str | None
    position: str | None
    name: str | None
    availability_probability: float
    mean_json: dict[str, Any]
    quantiles_json: dict[str, float]
    on_slate: bool = True


def _quantiles_from_points(points: float) -> dict[str, float]:
    return {
        "0.1": max(0.0, points * 0.7),
        "0.5": max(0.0, points),
        "0.9": max(0.0, points * 1.3),
    }


def _resolve_quantiles(
    *,
    mean_json: dict[str, Any],
    baseline: BaselinePlayer | None,
    on_slate: bool,
) -> dict[str, float]:
    """Preserve baseline quantiles for untouched players; scale when markets move points."""
    points = float(mean_json.get("points") or 0.0)
    if not on_slate:
        if baseline is not None and baseline.quantiles_json:
            return {str(k): 0.0 for k in baseline.quantiles_json}
        return _quantiles_from_points(0.0)

    sources = mean_json.get("component_sources") or {}
    market_touched = any(src == "weekly_props_consensus" for src in sources.values())
    if baseline is not None and baseline.quantiles_json and not market_touched:
        return {str(k): float(v) for k, v in baseline.quantiles_json.items()}

    if baseline is not None and baseline.quantiles_json:
        base_points = float((baseline.mean_json or {}).get("points") or 0.0)
        if base_points > 0:
            ratio = points / base_points
            return {str(k): float(v) * ratio for k, v in baseline.quantiles_json.items()}
    return _quantiles_from_points(points)


def build_candidate_row(
    consensus: PlayerConsensus,
    baseline: BaselinePlayer | None,
    *,
    policy: WeeklyPropsPolicy = DEFAULT_WEEKLY_POLICY,
) -> CandidateRow:
    baseline_mean = None if baseline is None else baseline.mean_json
    # Align on_slate with baseline bye/inactive when provided.
    effective = consensus
    if baseline is not None and not baseline.on_slate:
        from dataclasses import replace

        effective = replace(consensus, on_slate=False)
    mean_json = build_mean_json(
        consensus=effective, baseline=baseline_mean, policy=policy
    )
    availability = 0.0 if not effective.on_slate else (
        1.0 if baseline is None else float(baseline.availability_probability)
    )
    player_id = (
        consensus.player_id
        or (baseline.player_id if baseline is not None else None)
        or consensus.player_name
    )
    return CandidateRow(
        player_id=player_id,
        team=consensus.team or (None if baseline is None else baseline.team),
        opponent=consensus.opponent or (None if baseline is None else baseline.opponent),
        availability_probability=availability,
        mean_json=mean_json,
        quantiles_json=_resolve_quantiles(
            mean_json=mean_json,
            baseline=baseline,
            on_slate=effective.on_slate,
        ),
    )


def build_candidate_rows(
    manifest: ConsensusManifest,
    baselines: Mapping[str, BaselinePlayer],
    *,
    policy: WeeklyPropsPolicy = DEFAULT_WEEKLY_POLICY,
) -> tuple[CandidateRow, ...]:
    """Union of consensus players and uncovered baseline players."""
    by_id: dict[str, CandidateRow] = {}
    covered_ids: set[str] = set()

    for player in manifest.players:
        pid = player.player_id
        baseline = baselines.get(pid) if pid else None
        row = build_candidate_row(player, baseline, policy=policy)
        by_id[row.player_id] = row
        if pid:
            covered_ids.add(pid)

    for pid, baseline in baselines.items():
        if pid in covered_ids:
            continue
        # Uncovered: keep baseline components with baseline_only class.
        from src.ingest.props.contracts import PlayerConsensus as PC

        stub = PC(
            player_id=pid,
            player_name=baseline.name or pid,
            team=baseline.team,
            opponent=baseline.opponent,
            position=baseline.position,
            markets={},
            scoring_class="baseline_only",
            on_slate=baseline.on_slate,
        )
        by_id[pid] = build_candidate_row(stub, baseline, policy=policy)

    return tuple(sorted(by_id.values(), key=lambda row: row.player_id))


def build_candidate(
    *,
    manifest: ConsensusManifest,
    rows: tuple[CandidateRow, ...],
    manifest_uri: str,
    baseline_run_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Candidate:
    meta = {
        "derivation": "weekly_props_v1",
        "gate_version": GATE_VERSION,
        "policy_version": manifest.policy_version,
        "semantic_input_hash": manifest.semantic_input_hash,
        "baseline_run_id": baseline_run_id,
        "player_count": len(rows),
        "source": "weekly_props",
        **(metadata or {}),
    }
    return Candidate(
        mode="weekly",
        season=manifest.season,
        week=manifest.week,
        run_id=f"weekly-props-{manifest.season}-w{manifest.week:02d}-{manifest.semantic_input_hash[:12]}",
        model_version=MODEL_VERSION,
        input_hash=manifest.semantic_input_hash,
        manifest_uri=manifest_uri,
        artifact_mode=ARTIFACT_MODE,
        partition_mode="weekly",
        rows=rows,
        metadata=meta,
    )
