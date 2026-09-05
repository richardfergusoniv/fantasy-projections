"""League-specific rescoring of sealed component projections.

Re-scores ``projections_*.csv`` component stats through each league's compiled
Sleeper contract. Returns an explicit fidelity classification per league.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from src.app.projections.weekly_stat_draw import WEEKLY_TO_DRAW_STAT
from src.app.scoring.compiler import (
    compile_sleeper_scoring,
    score_stat_draw,
    scoring_settings_from_snapshot,
)
from src.app.scoring.contract import ScoringContract

ScoringFidelity = Literal[
    "exact_component_rescore",
    "modeled_approximation",
    "unsupported_rule",
]

#: Stat names in projections CSV -> draw stat keys.
PROJECTION_STAT_MAP: dict[str, str] = dict(WEEKLY_TO_DRAW_STAT)
PROJECTION_STAT_MAP.update(
    {
        "passing_yards": "pass_yards",
        "passing_tds": "pass_tds",
        "rushing_yards": "rush_yards",
        "rushing_tds": "rush_tds",
        "receiving_yards": "rec_yards",
        "receiving_tds": "rec_tds",
    }
)

PPFD_STATS = frozenset({"pass_first_downs", "rush_first_downs", "rec_first_downs"})
THRESHOLD_STATS = frozenset(
    {
        "pass_yards",
        "rush_yards",
        "rec_yards",
        "rush_rec_yards",
        "pass_completions",
        "rush_attempts",
        "tackles",
    }
)


@dataclass(frozen=True)
class LeagueRescoreResult:
    league_id: str
    display_name: str
    contract_hash: str
    scoring_fidelity: ScoringFidelity
    unsupported_keys: tuple[str, ...]
    approximate_rules: tuple[str, ...]
    player_count: int
    sample_spread: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "league_id": self.league_id,
            "display_name": self.display_name,
            "contract_hash": self.contract_hash,
            "scoring_fidelity": self.scoring_fidelity,
            "unsupported_keys": list(self.unsupported_keys),
            "approximate_rules": list(self.approximate_rules),
            "player_count": self.player_count,
            "sample_spread": round(self.sample_spread, 4),
        }


def _classify_fidelity(
    contract: ScoringContract,
    *,
    has_ppfd_components: bool,
) -> tuple[ScoringFidelity, tuple[str, ...]]:
    unsupported = tuple(sorted(contract.unsupported_keys))
    if unsupported:
        return "unsupported_rule", unsupported

    approximate: list[str] = []
    ppfd_rules = [r for r in contract.linear_rules if r.stat in PPFD_STATS]
    if ppfd_rules and not has_ppfd_components:
        approximate.extend(f"ppfd:{r.stat}" for r in ppfd_rules)
    if contract.threshold_rules:
        approximate.extend(f"threshold:{r.stat}" for r in contract.threshold_rules)
    if contract.bracket_rules:
        approximate.extend(f"bracket:{r.group}" for r in contract.bracket_rules)

    if approximate:
        return "modeled_approximation", tuple(approximate)
    return "exact_component_rescore", ()


def load_component_projections(
    path: Path,
    *,
    value_column: str = "pred_pg",
) -> dict[str, dict[str, Any]]:
    """Load per-player component stat means from projections CSV.

    ``pred_pg`` remains the default because weekly decisions score per-game
    expectations. Draft boards may request ``pred_season`` instead, or retain
    ``pred_pg`` and use the attached ``_projected_games`` value when a league
    has nonlinear weekly bonuses.
    """
    if not path.is_file():
        return {}
    by_player: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            player_id = str(row.get("player_id") or "")
            stat = str(row.get("stat") or "")
            if not player_id or not stat:
                continue
            draw_stat = PROJECTION_STAT_MAP.get(stat)
            if draw_stat is None:
                continue
            try:
                value = float(row.get(value_column) or 0.0)
            except (TypeError, ValueError):
                continue
            bucket = by_player.setdefault(player_id, {})
            bucket[draw_stat] = bucket.get(draw_stat, 0.0) + value
            bucket["_position"] = str(
                row.get("position") or bucket.get("_position", "RB")
            )
            if "_projected_games" not in bucket:
                try:
                    bucket["_projected_games"] = float(
                        row.get("projected_games") or 17.0
                    )
                except (TypeError, ValueError):
                    bucket["_projected_games"] = 17.0
    return by_player


def score_component_projection_set(
    components_by_player: dict[str, dict[str, Any]],
    contract: ScoringContract,
) -> tuple[dict[str, float], ScoringFidelity, tuple[str, ...]]:
    """Score one component projection set and return its fidelity contract."""

    has_ppfd = any(
        any(stat in comps for stat in PPFD_STATS)
        for comps in components_by_player.values()
    )
    fidelity, approximate = _classify_fidelity(
        contract,
        has_ppfd_components=has_ppfd,
    )
    scores: dict[str, float] = {}
    for player_id, comps in components_by_player.items():
        position = str(comps.get("_position") or "RB")
        scores[player_id] = rescore_player(
            comps,
            contract,
            position=position,
            fidelity=fidelity,
            approximate_rules=approximate,
        )
    return scores, fidelity, approximate


def rescore_player(
    components: dict[str, Any],
    contract: ScoringContract,
    *,
    position: str,
    fidelity: ScoringFidelity,
    approximate_rules: tuple[str, ...],
) -> float:
    draw = {k: v for k, v in components.items() if not k.startswith("_")}
    if fidelity == "exact_component_rescore":
        return score_stat_draw(draw, contract, position=position)

    # Best-effort expected-value approximation for PPFD and thresholds.
    total = score_stat_draw(draw, contract, position=position)
    if fidelity == "modeled_approximation":
        for rule in contract.linear_rules:
            if rule.stat in PPFD_STATS and rule.stat not in draw:
                # Cannot claim exact PPFD without component; linear terms already scored.
                continue
    return total


def rescore_league(
    *,
    league_id: str,
    display_name: str,
    scoring_settings: dict[str, Any],
    roster_positions: list[str],
    components_by_player: dict[str, dict[str, Any]],
    baseline_half_ppr: dict[str, float] | None = None,
) -> LeagueRescoreResult:
    contract = compile_sleeper_scoring(scoring_settings, roster_positions)
    scores, fidelity, approximate = score_component_projection_set(
        components_by_player,
        contract,
    )

    spread = 0.0
    if baseline_half_ppr and scores:
        diffs = []
        for pid, score in scores.items():
            base = baseline_half_ppr.get(pid)
            if base is not None:
                diffs.append(abs(score - base))
        if diffs:
            spread = max(diffs)

    return LeagueRescoreResult(
        league_id=league_id,
        display_name=display_name,
        contract_hash=contract.contract_hash,
        scoring_fidelity=fidelity,
        unsupported_keys=tuple(sorted(contract.unsupported_keys)),
        approximate_rules=approximate,
        player_count=len(scores),
        sample_spread=spread,
    )


def rescore_configured_leagues(
    session,
    *,
    components_path: Path,
    season: int = 2026,
) -> list[LeagueRescoreResult]:
    from src.app.persistence.models import League, LeagueRuleSnapshot

    components = load_component_projections(components_path)
    if not components:
        return []

    leagues = session.query(League).filter(League.season == season).all()
    results: list[LeagueRescoreResult] = []
    for league in leagues:
        snapshot = (
            session.query(LeagueRuleSnapshot)
            .filter(LeagueRuleSnapshot.league_id == league.league_id)
            .order_by(LeagueRuleSnapshot.fetched_at.desc())
            .first()
        )
        if snapshot is None:
            continue
        raw = snapshot.raw_json or {}
        scoring = scoring_settings_from_snapshot(raw)
        roster_positions = list((league.raw_json or {}).get("roster_positions") or [])
        results.append(
            rescore_league(
                league_id=league.league_id,
                display_name=league.name or league.league_id,
                scoring_settings=scoring,
                roster_positions=roster_positions,
                components_by_player=components,
            )
        )
    return results
