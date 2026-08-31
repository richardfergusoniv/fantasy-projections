"""Manager tendency learning from completed trades and logged proposals."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from src.app.persistence.models import LeagueTransaction, ManagerTendency, TradeProposal


LEAGUE_PRIORS = {
    "youth_preference": 0.5,
    "pick_preference": 0.5,
    "consolidation_bias": 0.5,
    "avg_package_size": 2.0,
    "accept_rate": 0.45,
}


@dataclass
class TendencyFeatures:
    youth_preference: float
    pick_preference: float
    consolidation_bias: float
    avg_package_size: float
    accept_rate: float
    sample_size: int


class ManagerTendencyService:
    MODEL_VERSION = "tendency_v1"

    def __init__(self, session: Session) -> None:
        self.session = session

    def rebuild(self, league_id: str) -> dict[int, TendencyFeatures]:
        roster_stats: dict[int, dict[str, float]] = {}
        roster_counts: dict[int, int] = {}

        for txn in self.session.query(LeagueTransaction).filter(LeagueTransaction.league_id == league_id).all():
            if txn.txn_type != "trade":
                continue
            payload = txn.payload or {}
            self._accumulate_trade_payload(roster_stats, roster_counts, payload)

        for proposal in self.session.query(TradeProposal).filter(TradeProposal.league_id == league_id).all():
            self._accumulate_proposal(roster_stats, roster_counts, proposal)

        results: dict[int, TendencyFeatures] = {}
        for roster_id, stats in roster_stats.items():
            n = max(1, roster_counts.get(roster_id, 0))
            shrunk = self._shrink(stats, n)
            row = ManagerTendency(
                league_id=league_id,
                roster_id=roster_id,
                as_of=datetime.now(UTC),
                sample_size=n,
                features_json={
                    "youth_preference": shrunk.youth_preference,
                    "pick_preference": shrunk.pick_preference,
                    "consolidation_bias": shrunk.consolidation_bias,
                    "avg_package_size": shrunk.avg_package_size,
                    "accept_rate": shrunk.accept_rate,
                },
                model_version=self.MODEL_VERSION,
            )
            self.session.add(row)
            results[roster_id] = shrunk
        self.session.flush()
        return results

    def get(self, league_id: str, roster_id: int) -> TendencyFeatures:
        row = (
            self.session.query(ManagerTendency)
            .filter(ManagerTendency.league_id == league_id, ManagerTendency.roster_id == roster_id)
            .order_by(ManagerTendency.as_of.desc())
            .first()
        )
        if row is None:
            return TendencyFeatures(
                youth_preference=LEAGUE_PRIORS["youth_preference"],
                pick_preference=LEAGUE_PRIORS["pick_preference"],
                consolidation_bias=LEAGUE_PRIORS["consolidation_bias"],
                avg_package_size=LEAGUE_PRIORS["avg_package_size"],
                accept_rate=LEAGUE_PRIORS["accept_rate"],
                sample_size=0,
            )
        return TendencyFeatures(sample_size=row.sample_size, **row.features_json)

    def tendency_adjustment(self, league_id: str, roster_id: int, *, package_size: int) -> float:
        features = self.get(league_id, roster_id)
        if features.sample_size < 2:
            return 0.0
        consolidation_signal = features.consolidation_bias - 0.5
        package_signal = (package_size - features.avg_package_size) / max(features.avg_package_size, 1.0)
        return max(-0.25, min(0.25, 0.6 * consolidation_signal + 0.4 * package_signal * 0.1))

    def _accumulate_trade_payload(self, stats: dict, counts: dict, payload: dict) -> None:
        roster_ids = payload.get("roster_ids", [])
        adds = payload.get("adds") or {}
        drops = payload.get("drops") or {}
        picks = payload.get("draft_picks") or []
        for roster_id in roster_ids:
            rid = int(roster_id)
            bucket = stats.setdefault(rid, self._empty_stats())
            counts[rid] = counts.get(rid, 0) + 1
            bucket["trade_count"] += 1
            bucket["package_size_sum"] += len(adds) + len(drops) + len(picks)
            bucket["pick_assets"] += len(picks)
            if len(adds) + len(drops) <= 2:
                bucket["consolidation_hits"] += 1

    def _accumulate_proposal(self, stats: dict, counts: dict, proposal: TradeProposal) -> None:
        rid = int(proposal.created_by_roster_id)
        bucket = stats.setdefault(rid, self._empty_stats())
        counts[rid] = counts.get(rid, 0) + 1
        sides = proposal.sides_json or {}
        assets = len(sides.get("offered", [])) + len(sides.get("received", []))
        bucket["proposal_count"] += 1
        bucket["package_size_sum"] += max(assets, 1)
        if proposal.status == "accepted":
            bucket["accepted"] += 1
        elif proposal.status in {"rejected", "expired"}:
            bucket["rejected"] += 1

    def _empty_stats(self) -> dict[str, float]:
        return {
            "trade_count": 0.0,
            "proposal_count": 0.0,
            "accepted": 0.0,
            "rejected": 0.0,
            "package_size_sum": 0.0,
            "pick_assets": 0.0,
            "consolidation_hits": 0.0,
        }

    def _shrink(self, stats: dict[str, float], sample_size: int) -> TendencyFeatures:
        weight = sample_size / (sample_size + 5)
        events = stats["trade_count"] + stats["proposal_count"]
        accept_rate = (stats["accepted"] / max(stats["accepted"] + stats["rejected"], 1.0)) if events else LEAGUE_PRIORS["accept_rate"]
        avg_package = stats["package_size_sum"] / max(events, 1.0)
        consolidation = stats["consolidation_hits"] / max(stats["trade_count"], 1.0) if stats["trade_count"] else LEAGUE_PRIORS["consolidation_bias"]
        pick_pref = stats["pick_assets"] / max(stats["trade_count"] + stats["proposal_count"], 1.0)
        return TendencyFeatures(
            youth_preference=LEAGUE_PRIORS["youth_preference"],
            pick_preference=LEAGUE_PRIORS["pick_preference"] * (1 - weight) + pick_pref * weight,
            consolidation_bias=LEAGUE_PRIORS["consolidation_bias"] * (1 - weight) + consolidation * weight,
            avg_package_size=LEAGUE_PRIORS["avg_package_size"] * (1 - weight) + avg_package * weight,
            accept_rate=LEAGUE_PRIORS["accept_rate"] * (1 - weight) + accept_rate * weight,
            sample_size=sample_size,
        )
