"""Temporary manual exercise of the decision engines across all six leagues."""

from __future__ import annotations

import os
import time

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("APP_ENABLE_DEV_AUTH", "true")
os.environ.setdefault("TEST_DATABASE_URL", "sqlite+pysqlite:///:memory:?cache=shared")


def main() -> int:
    from src.app.config import get_settings
    from src.app.decisions.services import LineupService, TradeService, WaiverService
    from src.app.decisions.trades import RedraftPickNotTradeable, TradeSide
    from src.app.persistence.database import get_session, init_db
    from src.app.seed import seed_development_data

    get_settings.cache_clear()
    init_db()
    with get_session() as session:
        seed = seed_development_data(session, email="owner@example.com")

    with get_session() as session:
        for league_id in seed["leagues"]:
            started = time.time()
            current = LineupService(session).recommend(league_id, 1, opponent_mode="current")
            optimized = LineupService(session).recommend(
                league_id, 1, opponent_mode="optimized"
            )
            elapsed = time.time() - started
            print()
            print(f"=== {league_id} ({elapsed:.1f}s) fidelity={current['scoring_fidelity']}")
            print(f"  unapplied: {current['unapplied_scoring_rules']}")
            print(f"  current-mode   : {current['matchup_probabilities']} src={current['opponent_lineup_source']}")
            print(f"  optimized-mode : {optimized['matchup_probabilities']} src={optimized['opponent_lineup_source']}")
            print(f"  recommended win {current['win_probability']} vs submitted {current['current_lineup_probabilities']}")
            print(
                "  swaps: "
                + str(
                    [
                        (
                            w["in_player_id"],
                            w["out_player_id"],
                            w["win_probability_delta"],
                            w["significant"],
                        )
                        for w in current["swaps"]
                    ]
                )
            )

            waivers = WaiverService(session).recommend(league_id, 1, remaining_faab=100)
            top = waivers["adds"][:2]
            print(
                "  waivers: "
                + str(
                    [
                        (
                            a["player_id"],
                            a["position"],
                            a["faab_min"],
                            a["faab_max"],
                            a["confidence"],
                        )
                        for a in top
                    ]
                )
            )

            roster_a = current["recommended_starters"][:1]
            roster_b = current["opponent_starters"][:1]
            trade_service = TradeService(session)
            for horizon in ("ros", "dynasty"):
                try:
                    result = trade_service.evaluate(
                        league_id,
                        TradeSide(roster_id=1, player_ids=list(roster_a)),
                        TradeSide(roster_id=2, player_ids=list(roster_b)),
                        horizon=horizon,
                    )
                    print(
                        f"  trade[{horizon}]: gain_a={result.objective['side_a_gain']} "
                        f"fair={result.fairness['fair']} acc_a={result.acceptance['side_a_probability']}"
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"  trade[{horizon}] FAILED: {type(exc).__name__}: {exc}")

            # Future picks: legal only in dynasty.
            picks = [{"season": 2027, "round": 1}]
            try:
                trade_service.evaluate(
                    league_id,
                    TradeSide(roster_id=1, player_ids=list(roster_a), pick_assets=picks),
                    TradeSide(roster_id=2, player_ids=list(roster_b)),
                    horizon="dynasty",
                )
                print("  pick trade: accepted")
            except RedraftPickNotTradeable as exc:
                print(f"  pick trade: correctly rejected ({exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
