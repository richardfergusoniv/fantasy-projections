"""Component merge and league-rescorable weekly prop means."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.ingest.props.contracts import PlayerConsensus
from src.projection.weekly_props.config import DEFAULT_WEEKLY_POLICY, WeeklyPropsPolicy
from src.projection.weekly_props.consensus import player_scoring_class

# Canonical mean_json keys used by decisions / league rescoring.
COMPONENT_KEYS: tuple[str, ...] = (
    "pass_yards",
    "pass_tds",
    "pass_ints",
    "pass_attempts",
    "pass_completions",
    "rush_yards",
    "rush_tds",
    "rush_attempts",
    "rec_yards",
    "rec_tds",
    "receptions",
    "targets",
    "fumbles",
)

# Half-PPR / 4-pt pass TD parity helper (reference only — not decision scoring).
PASS_TD_POINTS = 4.0
RUSH_REC_TD_POINTS = 6.0
RECEPTION_POINTS = 0.5


def half_ppr_parity_points(components: Mapping[str, float]) -> float:
    """Parity with checklist vegas_fantasy_points; not a league scoring contract."""
    return (
        float(components.get("pass_yards") or 0.0) / 25.0
        + float(components.get("pass_tds") or 0.0) * PASS_TD_POINTS
        + float(components.get("rush_yards") or 0.0) / 10.0
        + float(components.get("rush_tds") or 0.0) * RUSH_REC_TD_POINTS
        + float(components.get("rec_yards") or 0.0) / 10.0
        + float(components.get("rec_tds") or 0.0) * RUSH_REC_TD_POINTS
        + float(components.get("receptions") or 0.0) * RECEPTION_POINTS
        - float(components.get("pass_ints") or 0.0) * 2.0
        - float(components.get("fumbles") or 0.0) * 2.0
    )


def baseline_components(mean_json: Mapping[str, Any] | None) -> dict[str, float]:
    """Extract numeric stat components from a baseline player mean_json."""
    if not mean_json:
        return {}
    out: dict[str, float] = {}
    aliases = {
        "passing_yards": "pass_yards",
        "passing_tds": "pass_tds",
        "interceptions": "pass_ints",
        "pass_ints": "pass_ints",
        "rushing_yards": "rush_yards",
        "rushing_tds": "rush_tds",
        "receiving_yards": "rec_yards",
        "receiving_tds": "rec_tds",
        "carries": "rush_attempts",
    }
    for key, value in mean_json.items():
        dest = aliases.get(key, key)
        if dest not in COMPONENT_KEYS and dest != "points":
            continue
        try:
            out[dest] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def merge_player_components(
    consensus: PlayerConsensus,
    baseline: Mapping[str, Any] | None,
    *,
    policy: WeeklyPropsPolicy = DEFAULT_WEEKLY_POLICY,
) -> tuple[dict[str, float], dict[str, str], str]:
    """Merge book-backed market lines onto baseline components.

    Returns ``(components, component_sources, scoring_class)``.
    Bye / off-slate players return zeroed points-capable components with
    ``bye_or_inactive`` provenance and do not inherit nonzero baseline FP.
    """
    if not consensus.on_slate:
        zeros = {key: 0.0 for key in COMPONENT_KEYS}
        sources = {key: "bye_or_inactive" for key in COMPONENT_KEYS}
        return zeros, sources, "baseline_only"

    base = baseline_components(baseline)
    components: dict[str, float] = dict(base)
    sources: dict[str, str] = {
        key: "status_adjusted_baseline" for key in base
    }

    for market_name, market in consensus.markets.items():
        if market.kind != "books" or market.line is None:
            continue
        if market_name not in policy.scoring_markets and market_name not in COMPONENT_KEYS:
            continue
        # Only book-backed accepted consensus may replace baseline.
        components[market_name] = float(market.line)
        sources[market_name] = "weekly_props_consensus"

    # Ensure baseline-only components remain tagged even if absent from props.
    for key in policy.baseline_only_components:
        if key in components and key not in sources:
            sources[key] = "status_adjusted_baseline"
        elif key in base:
            components[key] = base[key]
            sources[key] = "status_adjusted_baseline"

    scoring_class = player_scoring_class(
        consensus.markets,
        scoring_markets=policy.scoring_markets,
        baseline_components=set(base) & set(policy.scoring_markets),
    )
    return components, sources, scoring_class


def _market_replaced_components(component_sources: Mapping[str, str]) -> bool:
    return any(src == "weekly_props_consensus" for src in component_sources.values())


def resolve_points(
    *,
    components: Mapping[str, float],
    component_sources: Mapping[str, str],
    baseline: Mapping[str, Any] | None,
    on_slate: bool,
) -> tuple[float, str]:
    """Score points on the same basis as the week-level baseline when possible.

    Untouched (baseline-only) players keep sealed weekly points so publication
    and the movement gate do not silently re-base onto half-PPR parity.
    Market-touched players apply a half-PPR component delta on top of that
    sealed baseline so both sides of the gate stay comparable.
    """
    if not on_slate:
        return 0.0, "bye_or_inactive"

    baseline = baseline or {}
    baseline_points = baseline.get("points")
    try:
        sealed_points = float(baseline_points) if baseline_points is not None else None
    except (TypeError, ValueError):
        sealed_points = None

    if not _market_replaced_components(component_sources):
        if sealed_points is not None:
            return sealed_points, str(
                baseline.get("points_source") or "sealed_per_game_baseline"
            )
        return half_ppr_parity_points(components), "half_ppr_parity_reference"

    base_comps = baseline_components(baseline)
    base_half = half_ppr_parity_points(base_comps)
    new_half = half_ppr_parity_points(components)
    if sealed_points is not None:
        return sealed_points + (new_half - base_half), "baseline_plus_market_delta"
    return new_half, "half_ppr_parity_reference"


def build_mean_json(
    *,
    consensus: PlayerConsensus,
    baseline: Mapping[str, Any] | None,
    policy: WeeklyPropsPolicy = DEFAULT_WEEKLY_POLICY,
) -> dict[str, Any]:
    components, component_sources, scoring_class = merge_player_components(
        consensus, baseline, policy=policy
    )
    mean: dict[str, Any] = {
        "position": consensus.position
        or (baseline or {}).get("position"),
        "name": consensus.player_name or (baseline or {}).get("name"),
        "team": consensus.team or (baseline or {}).get("team"),
        "derivation": "weekly_props_v1",
        "scoring_class": scoring_class,
        "component_sources": component_sources,
        "market_coverage": {
            name: market.coverage.to_dict() for name, market in consensus.markets.items()
        },
    }
    for key, value in components.items():
        mean[key] = value
    points, points_source = resolve_points(
        components=components,
        component_sources=component_sources,
        baseline=baseline,
        on_slate=consensus.on_slate,
    )
    mean["points"] = points
    mean["points_source"] = points_source
    return mean
