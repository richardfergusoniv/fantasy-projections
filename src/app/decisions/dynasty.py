"""Dynasty manager state and draft-order inference.

The features below are computed from the league's own data — the same draw set
and scoring contract every other decision uses — rather than supplied as
constants by the caller. A feature the available data cannot support is reported
as unavailable and dropped from the blend (with the remaining weights
renormalized), so a missing input lowers confidence instead of silently becoming
a made-up number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from src.app.persistence.models import LeagueDraftRule, ManagerState, TradedPick

#: Feature -> weight in the contender score. Renormalized over the features
#: that were actually computable for this league.
FEATURE_WEIGHTS: dict[str, float] = {
    "lineup_strength": 0.35,
    "ros_win_prob": 0.35,
    "multi_year_value": 0.20,
    "pick_capital": 0.10,
}

#: Future rookie-draft seasons a dynasty roster's pick capital is measured over.
PICK_CAPITAL_SEASONS = 3
#: Rounds of rookie picks each roster starts with per season.
BASELINE_ROOKIE_ROUNDS = 2


@dataclass
class ManagerStateResult:
    label: str
    probabilities: dict[str, float]
    overridden_label: str | None
    features: dict
    #: Features that could not be computed, and why. Never silently filled in.
    unavailable_features: dict[str, str] = field(default_factory=dict)
    #: Share of the model's weight that was actually backed by data.
    feature_coverage: float = 1.0


class DynastyService:
    def __init__(self, session: Session) -> None:
        self.session = session

    # ------------------------------------------------------------------ model

    def infer_manager_state(
        self,
        *,
        league_id: str,
        roster_id: int,
        lineup_strength: float | None = None,
        ros_win_prob: float | None = None,
        multi_year_value: float | None = None,
        pick_capital: float | None = None,
        unavailable: dict[str, str] | None = None,
    ) -> ManagerStateResult:
        supplied: dict[str, float] = {}
        for name, value in (
            ("lineup_strength", lineup_strength),
            ("ros_win_prob", ros_win_prob),
            ("multi_year_value", multi_year_value),
            ("pick_capital", pick_capital),
        ):
            if value is not None:
                supplied[name] = max(0.0, min(1.0, float(value)))

        missing = dict(unavailable or {})
        for name in FEATURE_WEIGHTS:
            missing.setdefault(name, "not_computed") if name not in supplied else None

        weight_total = sum(FEATURE_WEIGHTS[name] for name in supplied)
        if weight_total <= 0.0:
            # No usable feature: report the least committal state rather than a
            # confident label derived from nothing.
            probs = {"contender": 0.25, "fringe": 0.25, "retooling": 0.25, "rebuilding": 0.25}
            return ManagerStateResult(
                label="fringe",
                probabilities=probs,
                overridden_label=None,
                features={},
                unavailable_features=missing,
                feature_coverage=0.0,
            )

        contender = sum(FEATURE_WEIGHTS[name] * value for name, value in supplied.items())
        contender = max(0.0, min(1.0, contender / weight_total))
        rebuilding = max(0.0, 1.0 - contender - 0.25)
        fringe = max(0.0, 0.25 - abs(contender - 0.5) * 0.2)
        retooling = max(0.0, 1.0 - contender - rebuilding - fringe)
        total = contender + fringe + retooling + rebuilding
        probs = {
            "contender": contender / total,
            "fringe": fringe / total,
            "retooling": retooling / total,
            "rebuilding": rebuilding / total,
        }
        label = max(probs, key=probs.get)
        return ManagerStateResult(
            label=label,
            probabilities=probs,
            overridden_label=None,
            features=supplied,
            unavailable_features=missing,
            feature_coverage=round(weight_total / sum(FEATURE_WEIGHTS.values()), 4),
        )

    def save_state(self, league_id: str, roster_id: int, result: ManagerStateResult) -> ManagerState:
        row = ManagerState(
            league_id=league_id,
            roster_id=roster_id,
            as_of=datetime.now(UTC),
            label=result.label,
            probabilities_json=result.probabilities,
            features_json={
                **result.features,
                "unavailable": result.unavailable_features,
                "feature_coverage": result.feature_coverage,
            },
            overridden_label=result.overridden_label,
        )
        self.session.add(row)
        self.session.flush()
        return row

    # --------------------------------------------------------------- features

    def league_features(self, league_id: str, week: int) -> tuple[dict[int, dict], dict[str, str]]:
        """Per-roster dynasty features derived from the league's own draws.

        Returns ``(features_by_roster_id, unavailable)``. Every value is a 0-1
        share relative to the rest of the league, which is what "contender"
        means: strong *for this league*, not against an absolute scale.
        """
        # Imported here: the decision services import this module's siblings, and
        # a module-level import would close the cycle.
        from src.app.decisions.lineup import optimize_lineup, win_probability
        from src.app.decisions.services import _LeagueContext
        from src.app.persistence.repositories import LeagueRepository

        unavailable: dict[str, str] = {}
        ctx = _LeagueContext(self.session, league_id, week)
        rosters = LeagueRepository(self.session).latest_rosters(league_id, week)
        if len(rosters) < 2:
            return {}, {name: "fewer_than_two_rosters" for name in FEATURE_WEIGHTS}

        all_player_ids: list[str] = []
        for roster in rosters:
            all_player_ids.extend(pid for pid in (roster.players or []) if pid)
        draw_set, _missing = ctx.build_draws(all_player_ids)

        lineups: dict[int, object] = {}
        for roster in rosters:
            candidates = [pid for pid in (roster.players or []) if pid in draw_set.players]
            if not candidates:
                continue
            lineups[roster.roster_id] = optimize_lineup(
                draw_set, ctx.contract, candidate_ids=candidates, objective="points"
            )
        if len(lineups) < 2:
            return {}, {name: "not_enough_projected_rosters" for name in FEATURE_WEIGHTS}

        strengths = {rid: float(result.totals.mean()) for rid, result in lineups.items()}
        best = max(strengths.values()) or 1.0

        win_probs: dict[int, float] = {}
        for rid, result in lineups.items():
            others = [other for other_id, other in lineups.items() if other_id != rid]
            win_probs[rid] = sum(
                win_probability(result.totals, other.totals)["win"] for other in others
            ) / len(others)

        # Multi-year value: the roster's whole player pool over the dynasty
        # horizon, not just its starters. Ages are not in the identity registry,
        # so this is explicitly *not* age-adjusted and says so.
        unavailable["age_adjustment"] = "no_player_birthdate_in_identity_registry"
        pool_values: dict[int, float] = {}
        for roster in rosters:
            pool_values[roster.roster_id] = sum(
                float(draw_set.players[pid].points.mean())
                for pid in (roster.players or [])
                if pid in draw_set.players
            )
        best_pool = max(pool_values.values()) or 1.0

        pick_capital = self.pick_capital(league_id, ctx.season, [r.roster_id for r in rosters])

        features: dict[int, dict] = {}
        for roster in rosters:
            rid = roster.roster_id
            if rid not in lineups:
                continue
            features[rid] = {
                "lineup_strength": round(strengths[rid] / best, 4),
                "ros_win_prob": round(win_probs[rid], 4),
                "multi_year_value": round(pool_values.get(rid, 0.0) / best_pool, 4),
                "pick_capital": pick_capital.get(rid),
            }
        return features, unavailable

    def pick_capital(
        self, league_id: str, season: int, roster_ids: list[int]
    ) -> dict[int, float | None]:
        """Owned future rookie picks per roster, as a share of the league best.

        Each roster starts with its own picks for the next
        ``PICK_CAPITAL_SEASONS`` seasons; ``traded_pick`` rows move ownership.
        A league with no traded picks recorded still yields a valid (equal)
        distribution, which is the correct answer, not a missing feature.
        """
        owned: dict[int, float] = dict.fromkeys(roster_ids, 0.0)
        future_seasons = range(season + 1, season + 1 + PICK_CAPITAL_SEASONS)
        moved: dict[tuple[int, int, int], int] = {}
        for row in (
            self.session.query(TradedPick)
            .filter(TradedPick.league_id == league_id)
            .all()
        ):
            moved[(row.season, row.round, row.original_roster_id)] = row.owner_roster_id

        for pick_season in future_seasons:
            # Picks further out are worth less and are less certain.
            year_weight = 0.82 ** (pick_season - season - 1)
            for rnd in range(1, BASELINE_ROOKIE_ROUNDS + 1):
                round_weight = 1.0 / rnd
                for original in roster_ids:
                    owner = moved.get((pick_season, rnd, original), original)
                    if owner in owned:
                        owned[owner] += year_weight * round_weight
        best = max(owned.values()) if owned else 0.0
        if best <= 0.0:
            return dict.fromkeys(roster_ids, None)
        return {rid: round(value / best, 4) for rid, value in owned.items()}

    # ------------------------------------------------------------ orchestration

    def evaluate_roster(self, league_id: str, roster_id: int, *, week: int = 1) -> ManagerStateResult:
        """Infer and persist one roster's dynasty state from league data.

        Persisting matters beyond this endpoint: the trade engine reads
        ``manager_state`` to attach contender/rebuilder context to an
        evaluation, and without a stored row that context is always null.
        """
        try:
            features_by_roster, unavailable = self.league_features(league_id, week)
        except Exception as exc:  # noqa: BLE001 - a missing league is not a crash
            features_by_roster, unavailable = {}, {
                name: f"league_context_unavailable:{type(exc).__name__}"
                for name in FEATURE_WEIGHTS
            }
        features = features_by_roster.get(roster_id, {})
        result = self.infer_manager_state(
            league_id=league_id,
            roster_id=roster_id,
            lineup_strength=features.get("lineup_strength"),
            ros_win_prob=features.get("ros_win_prob"),
            multi_year_value=features.get("multi_year_value"),
            pick_capital=features.get("pick_capital"),
            unavailable=unavailable,
        )
        existing = (
            self.session.query(ManagerState)
            .filter(
                ManagerState.league_id == league_id,
                ManagerState.roster_id == roster_id,
            )
            .order_by(ManagerState.as_of.desc())
            .first()
        )
        # A user override is a decision, not a stale value: keep it and report it.
        if existing is not None and existing.overridden_label:
            result.overridden_label = existing.overridden_label
        self.save_state(league_id, roster_id, result)
        return result

    # ------------------------------------------------------------------ drafts

    def draft_order_rule(self, league_id: str) -> str:
        row = (
            self.session.query(LeagueDraftRule)
            .filter(LeagueDraftRule.league_id == league_id)
            .order_by(LeagueDraftRule.confirmed_at.desc())
            .first()
        )
        return row.rule if row else "max_pf"

    def project_rookie_pick_slot(
        self,
        *,
        league_id: str,
        roster_id: int,
        optimal_points: float,
        potential_points: float,
        projected_record: float,
        league_size: int = 12,
    ) -> dict:
        """Project a rookie-pick slot under the league's confirmed rule.

        The two rules are genuinely different orderings and are kept separate:
        ``max_pf`` ranks non-playoff teams by simulated points, ``reverse_standings``
        by projected record. Neither is inferred from the other.
        """
        rule = self.draft_order_rule(league_id)
        if rule == "max_pf":
            score = 0.6 * optimal_points + 0.4 * potential_points
            return {
                "rule": rule,
                "basis": "simulated_optimal_and_potential_points",
                "projected_pick": max(1, min(league_size, int(14 - score / 10))),
                "uncertainty": 1.2,
            }
        return {
            "rule": rule,
            "basis": "projected_final_standings",
            "projected_pick": max(1, min(league_size, int(projected_record))),
            "uncertainty": 1.5,
        }

    def project_rookie_pick_for_roster(
        self, league_id: str, roster_id: int, *, week: int = 1, league_size: int | None = None
    ) -> dict:
        """Rookie-pick projection driven by the roster's own simulated season.

        Both rules need a league-relative ordering, so both are derived from the
        same per-roster features rather than from constants.
        """
        try:
            features_by_roster, _unavailable = self.league_features(league_id, week)
        except Exception:  # noqa: BLE001
            features_by_roster = {}
        size = league_size or max(len(features_by_roster), 2)
        features = features_by_roster.get(roster_id)
        if not features:
            return {
                "rule": self.draft_order_rule(league_id),
                "basis": "unavailable",
                "projected_pick": None,
                "uncertainty": None,
                "reason": "no_projected_roster_for_this_league_and_week",
            }

        # Worse rosters pick earlier under both rules; the rules differ in what
        # "worse" is measured on.
        by_points = sorted(
            features_by_roster.items(), key=lambda item: item[1]["multi_year_value"]
        )
        by_record = sorted(
            features_by_roster.items(), key=lambda item: item[1]["ros_win_prob"]
        )
        rule = self.draft_order_rule(league_id)
        order = by_points if rule == "max_pf" else by_record
        slot = next(i for i, (rid, _f) in enumerate(order, start=1) if rid == roster_id)
        return {
            "rule": rule,
            "basis": (
                "simulated_optimal_and_potential_points"
                if rule == "max_pf"
                else "projected_final_standings"
            ),
            "projected_pick": slot,
            "league_size": size,
            # Uncertainty widens with league size; reverse standings is noisier
            # because a single win swing moves a team several slots.
            "uncertainty": round((1.2 if rule == "max_pf" else 1.5) * size / 12.0, 3),
        }
