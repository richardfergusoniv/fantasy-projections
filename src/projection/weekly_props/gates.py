"""Publication gates for the weekly_props production source."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Iterable

from src.app.releases.gates import GateResult
from src.app.releases.publication import Candidate, CandidateRow
from src.ingest.props.contracts import ConsensusManifest, ProviderSnapshot
from src.projection.weekly_props.config import (
    DEFAULT_WEEKLY_POLICY,
    GATE_VERSION,
    WeeklyPropsPolicy,
)


def validate_snapshots(
    snapshots: Iterable[ProviderSnapshot],
    *,
    season: int,
    week: int,
    policy: WeeklyPropsPolicy = DEFAULT_WEEKLY_POLICY,
    now: datetime | None = None,
) -> GateResult:
    failures: list[str] = []
    warnings: list[str] = []
    clock = now or datetime.now(timezone.utc)
    snaps = list(snapshots)
    if not snaps:
        failures.append("no_provider_snapshots")
        return GateResult(passed=False, failures=failures, warnings=warnings)

    successes = [s for s in snaps if s.success]
    if len(successes) < policy.min_source_success_count:
        failures.append(
            f"source_success_below_minimum:{len(successes)}<{policy.min_source_success_count}"
        )
    usable = 0
    for snap in successes:
        if snap.season != season or snap.week != week:
            failures.append(f"snapshot_season_week_mismatch:{snap.source}")
            continue
        age_h = (clock - snap.fetched_at).total_seconds() / 3600.0
        if age_h > policy.max_snapshot_age_hours:
            warnings.append(f"stale_snapshot:{snap.source}:{age_h:.1f}h")
            continue
        if not snap.quotes:
            warnings.append(f"empty_snapshot:{snap.source}")
            continue
        usable += 1
    if usable < policy.min_source_success_count:
        failures.append(f"usable_source_below_minimum:{usable}")
    return GateResult(passed=not failures, failures=failures, warnings=warnings)


def validate_consensus_coverage(
    manifest: ConsensusManifest,
    *,
    policy: WeeklyPropsPolicy = DEFAULT_WEEKLY_POLICY,
) -> GateResult:
    failures: list[str] = []
    warnings: list[str] = []
    on_slate = [p for p in manifest.players if p.on_slate]
    if len(on_slate) < policy.min_candidate_players:
        failures.append(
            f"on_slate_player_count_below_minimum:{len(on_slate)}<{policy.min_candidate_players}"
        )
    resolved = [p for p in on_slate if p.player_id]
    join_rate = (len(resolved) / len(on_slate)) if on_slate else 0.0
    if join_rate < policy.min_identity_join_rate:
        failures.append(
            f"identity_join_rate_below_minimum:{join_rate:.3f}<{policy.min_identity_join_rate}"
        )
    book_markets = 0
    for player in on_slate:
        for market in player.markets.values():
            if market.kind == "books" and market.line is not None:
                book_markets += 1
                if market.coverage.book_count < policy.min_distinct_books_per_market:
                    warnings.append(
                        f"low_book_count:{player.player_id or player.player_name}:{market.market}"
                    )
    if book_markets == 0:
        failures.append("no_book_backed_markets")
    return GateResult(passed=not failures, failures=failures, warnings=warnings)


def validate_candidate_sanity(
    rows: tuple[CandidateRow, ...],
    *,
    policy: WeeklyPropsPolicy = DEFAULT_WEEKLY_POLICY,
) -> GateResult:
    failures: list[str] = []
    warnings: list[str] = []
    if len(rows) < policy.min_candidate_players:
        failures.append(f"candidate_player_count_below_minimum:{len(rows)}")
    negative = [
        row.player_id
        for row in rows
        if float(row.mean_json.get("points") or 0.0) < 0
    ]
    if negative:
        failures.append(f"negative_mean_points:{len(negative)}")
    return GateResult(passed=not failures, failures=failures, warnings=warnings)


def validate_baseline_movement(
    rows: tuple[CandidateRow, ...],
    baseline_points: dict[str, float],
    *,
    policy: WeeklyPropsPolicy = DEFAULT_WEEKLY_POLICY,
) -> GateResult:
    failures: list[str] = []
    warnings: list[str] = []
    if not baseline_points:
        warnings.append("no_baseline_points_for_movement_gate")
        return GateResult(passed=True, failures=failures, warnings=warnings)

    deltas: list[tuple[str, float]] = []
    for row in rows:
        base = baseline_points.get(row.player_id)
        if base is None:
            continue
        pts = float(row.mean_json.get("points") or 0.0)
        if base == 0 and pts == 0:
            continue
        scale = max(abs(base), 1.0)
        rel = abs(pts - base) / scale
        deltas.append((row.player_id, rel))
    if deltas:
        avg_rel = sum(rel for _, rel in deltas) / len(deltas)
        if avg_rel > policy.max_aggregate_baseline_movement:
            failures.append(
                f"aggregate_baseline_movement:{avg_rel:.3f}>{policy.max_aggregate_baseline_movement}"
            )

    # Top-N displacement vs baseline ranking.
    base_rank = {
        pid: idx
        for idx, (pid, _) in enumerate(
            sorted(baseline_points.items(), key=lambda item: item[1], reverse=True)
        )
    }
    cand_sorted = sorted(
        rows,
        key=lambda row: float(row.mean_json.get("points") or 0.0),
        reverse=True,
    )
    top_n = policy.top_n_for_displacement
    displacements = 0
    for idx, row in enumerate(cand_sorted[:top_n]):
        prev = base_rank.get(row.player_id)
        if prev is None:
            continue
        if abs(prev - idx) > policy.max_top_n_displacement:
            displacements += 1
    if displacements > policy.max_top_n_displacement:
        failures.append(f"top_n_displacement_excess:{displacements}")
    return GateResult(passed=not failures, failures=failures, warnings=warnings)


def validate_props_provenance(candidate: Candidate) -> GateResult:
    failures: list[str] = []
    meta = candidate.metadata or {}
    if candidate.artifact_mode != "market":
        failures.append(f"unexpected_artifact_mode:{candidate.artifact_mode}")
    if not str(candidate.model_version).startswith("weekly_props"):
        failures.append(f"unexpected_model_version:{candidate.model_version}")
    if meta.get("gate_version") != GATE_VERSION:
        failures.append("missing_or_mismatched_gate_version")
    if not meta.get("semantic_input_hash"):
        failures.append("missing_semantic_input_hash")
    if meta.get("derivation") != "weekly_props_v1":
        failures.append("missing_weekly_props_derivation")
    return GateResult(passed=not failures, failures=failures)
