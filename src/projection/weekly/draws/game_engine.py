"""Team/game-level joint weekly draw generator.

Fundamental unit is a scheduled game (two teams), not an independent player.
Each simulation index shares a game environment, allocates opportunities jointly
within position rooms, reconciles player components to team totals, and feeds
the same game state into kicker/DST models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from src.projection.weekly.draws.contracts import np_clip_prob
from src.projection.weekly.draws.first_downs import (
    DEFAULT_RATES_BY_POS,
    reconcile_team_pass_rec_first_downs,
    sample_first_downs,
)
from src.projection.weekly.draws.special_teams_game import (
    DstGameContext,
    GameOffenseState,
    simulate_dst_from_game,
    simulate_kicker_from_game,
)


SKILL_POSITIONS = frozenset({"QB", "RB", "WR", "TE"})


@dataclass
class PlayerGameInput:
    player_id: str
    position: str
    team: str
    p_active: float
    p_participates: float
    p_positive_usage: float
    # Mean opportunity intensities conditional on positive usage (shares of team).
    target_share: float = 0.0
    carry_share: float = 0.0
    dropback_share: float = 0.0
    # Efficiency means conditional on opportunities.
    catch_rate: float = 0.65
    ypa: float = 7.2
    ypc: float = 4.2
    ypr: float = 11.0
    pass_td_rate: float = 0.045
    rush_td_rate: float = 0.03
    rec_td_rate: float = 0.06
    int_rate: float = 0.025
    locked: bool = False
    locked_stats: dict[str, float] | None = None


@dataclass
class TeamGameInput:
    team: str
    opponent: str
    home: bool = True
    mean_pass_attempts: float = 34.0
    mean_rush_attempts: float = 27.0
    mean_pass_tds: float = 1.5
    mean_rush_tds: float = 0.9
    mean_points: float = 22.0
    weather_factor: float = 1.0
    venue_factor: float = 1.0
    pressure_rate: float = 0.08
    turnover_prior: float = 0.10
    players: list[PlayerGameInput] = field(default_factory=list)
    other_receiver_reserve_share: float = 0.04
    other_carry_reserve_share: float = 0.03
    throwaway_reserve_share: float = 0.04


@dataclass
class ScheduledGameInput:
    game_id: str
    season: int
    week: int
    kickoff: str | None = None
    home: TeamGameInput | None = None
    away: TeamGameInput | None = None


def _dirichlet_from_means(rng: np.random.Generator, means: np.ndarray, *, concentration: float) -> np.ndarray:
    means = np.asarray(means, dtype=float)
    means = np.clip(means, 1e-9, None)
    means = means / means.sum()
    alpha = means * max(concentration, 1.0)
    return rng.dirichlet(alpha)


def _sample_count(rng: np.random.Generator, mean: float) -> int:
    mean = max(0.0, float(mean))
    if mean <= 0:
        return 0
    return int(rng.poisson(mean))


def _zero_stats() -> dict[str, float]:
    return {
        "pass_attempts": 0.0,
        "pass_completions": 0.0,
        "pass_yards": 0.0,
        "pass_tds": 0.0,
        "pass_ints": 0.0,
        "rush_attempts": 0.0,
        "rush_yards": 0.0,
        "rush_tds": 0.0,
        "targets": 0.0,
        "receptions": 0.0,
        "rec_yards": 0.0,
        "rec_tds": 0.0,
        "pass_first_downs": 0.0,
        "rush_first_downs": 0.0,
        "rec_first_downs": 0.0,
        "offense_snaps": 0.0,
    }


def _sample_team_draw(
    rng: np.random.Generator,
    team: TeamGameInput,
    *,
    env_factor: float,
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, float], GameOffenseState]:
    """Sample one team draw: totals, player lines, reserves, offense state."""
    pass_att = max(0, int(round(_sample_count(rng, team.mean_pass_attempts * env_factor))))
    rush_att = max(0, int(round(_sample_count(rng, team.mean_rush_attempts * env_factor))))
    pass_tds = max(0, int(rng.poisson(max(0.05, team.mean_pass_tds * env_factor))))
    rush_tds = max(0, int(rng.poisson(max(0.05, team.mean_rush_tds * env_factor))))
    # Completions / yards will be rebuilt from player lines + reserve.
    players_out: list[dict[str, Any]] = []

    # Event sampling
    eligible: list[PlayerGameInput] = []
    for p in team.players:
        if p.locked and p.locked_stats is not None:
            stats = dict(_zero_stats())
            stats.update({k: float(v) for k, v in p.locked_stats.items()})
            players_out.append(
                {
                    "player_id": p.player_id,
                    "position": p.position,
                    "stats": stats,
                    "active": True,
                    "participated": True,
                    "positive_usage": False,
                    "locked": True,
                }
            )
            continue
        p_active = np_clip_prob(p.p_active)
        active = bool(rng.random() < p_active)
        participated = False
        positive = False
        if active:
            participated = bool(rng.random() < np_clip_prob(p.p_participates))
        if participated:
            positive = bool(rng.random() < np_clip_prob(p.p_positive_usage))
        players_out.append(
            {
                "player_id": p.player_id,
                "position": p.position,
                "stats": _zero_stats(),
                "active": active,
                "participated": participated,
                "positive_usage": positive,
                "locked": False,
                "_input": p,
            }
        )
        if positive:
            eligible.append(p)

    # Allocate targets among positive-usage pass catchers + reserve.
    receivers = [
        row
        for row in players_out
        if (not row.get("locked"))
        and row.get("positive_usage")
        and row["position"] in {"RB", "WR", "TE"}
    ]
    qbs = [
        row
        for row in players_out
        if (not row.get("locked")) and row.get("positive_usage") and row["position"] == "QB"
    ]
    rushers = [
        row
        for row in players_out
        if (not row.get("locked"))
        and row.get("positive_usage")
        and row["position"] in {"RB", "QB", "WR", "TE"}
    ]

    target_means = []
    for row in receivers:
        inp: PlayerGameInput = row["_input"]
        target_means.append(max(1e-6, inp.target_share))
    reserve_t = max(1e-6, team.other_receiver_reserve_share)
    if receivers:
        weights = _dirichlet_from_means(
            rng, np.array(target_means + [reserve_t]), concentration=40.0
        )
        named_w, reserve_w = weights[:-1], float(weights[-1])
        named_targets = np.floor(pass_att * named_w).astype(int)
        # Assign residual to largest weight to conserve attempts approx.
        assigned = int(named_targets.sum())
        reserve_targets = max(0, pass_att - assigned)
        # Prefer reserve share: shrink named if needed.
        if reserve_targets < int(round(pass_att * reserve_w)):
            # leave as-is; reserve absorbs remainder
            pass
        for row, tgt in zip(receivers, named_targets, strict=True):
            row["stats"]["targets"] = float(tgt)
    else:
        reserve_targets = pass_att

    # QB dropbacks / attempts — concentrate on primary QB.
    if qbs:
        qb_means = np.array([max(1e-6, row["_input"].dropback_share or 0.97) for row in qbs])
        qb_w = _dirichlet_from_means(rng, qb_means, concentration=80.0)
        qb_att = np.floor(pass_att * (1.0 - team.throwaway_reserve_share) * qb_w).astype(int)
        for row, att in zip(qbs, qb_att, strict=True):
            row["stats"]["pass_attempts"] = float(att)
        throwaway = max(0, pass_att - int(sum(r["stats"]["pass_attempts"] for r in qbs)))
    else:
        throwaway = pass_att

    # Carries
    carry_means = []
    for row in rushers:
        carry_means.append(max(1e-6, row["_input"].carry_share))
    reserve_c = max(1e-6, team.other_carry_reserve_share)
    if rushers:
        cw = _dirichlet_from_means(rng, np.array(carry_means + [reserve_c]), concentration=35.0)
        named_c = np.floor(rush_att * cw[:-1]).astype(int)
        for row, c in zip(rushers, named_c, strict=True):
            row["stats"]["rush_attempts"] = float(c)
        reserve_carries = max(0, rush_att - int(named_c.sum()))
    else:
        reserve_carries = rush_att

    # Efficiency conditional on opportunities.
    for row in players_out:
        if row.get("locked"):
            continue
        if not row["positive_usage"]:
            if row["participated"]:
                row["stats"]["offense_snaps"] = float(rng.integers(1, 15))
            continue
        inp: PlayerGameInput = row["_input"]
        stats = row["stats"]
        stats["offense_snaps"] = float(rng.integers(20, 70))
        # Passing
        att = int(stats["pass_attempts"])
        if att > 0:
            comps = int(rng.binomial(att, float(np.clip(inp.catch_rate if False else 0.65, 0.4, 0.8))))
            # Use a dedicated completion rate proxy: ypa applied on attempts.
            comp_rate = float(np.clip(0.62 + 0.02 * (inp.ypa - 7.0), 0.45, 0.78))
            comps = int(rng.binomial(att, comp_rate))
            stats["pass_completions"] = float(comps)
            ypa_draw = float(np.exp(rng.normal(np.log(max(inp.ypa, 1.0)), 0.12)))
            stats["pass_yards"] = float(max(0, int(round(att * ypa_draw))))
            stats["pass_tds"] = float(min(att, int(rng.binomial(att, float(np.clip(inp.pass_td_rate, 0.0, 0.15))))))
            stats["pass_ints"] = float(min(att, int(rng.binomial(att, float(np.clip(inp.int_rate, 0.0, 0.1))))))
        # Rushing
        carries = int(stats["rush_attempts"])
        if carries > 0:
            ypc_draw = float(np.exp(rng.normal(np.log(max(inp.ypc, 1.0)), 0.15)))
            stats["rush_yards"] = float(max(0, int(round(carries * ypc_draw))))
            stats["rush_tds"] = float(min(carries, int(rng.binomial(carries, float(np.clip(inp.rush_td_rate, 0.0, 0.2))))))
        # Receiving
        targets = int(stats["targets"])
        if targets > 0:
            catch = float(np.clip(inp.catch_rate, 0.3, 0.9))
            rec = int(rng.binomial(targets, catch))
            stats["receptions"] = float(rec)
            if rec > 0:
                ypr_draw = float(np.exp(rng.normal(np.log(max(inp.ypr, 1.0)), 0.18)))
                stats["rec_yards"] = float(max(0, int(round(rec * ypr_draw))))
                stats["rec_tds"] = float(min(rec, int(rng.binomial(rec, float(np.clip(inp.rec_td_rate, 0.0, 0.25))))))
            else:
                stats["rec_yards"] = 0.0
                stats["rec_tds"] = 0.0
        # First downs
        fd = sample_first_downs(
            rng,
            position=row["position"],
            completions=stats["pass_completions"],
            carries=stats["rush_attempts"],
            receptions=stats["receptions"],
            rates=DEFAULT_RATES_BY_POS.get(row["position"]),
        )
        stats.update(fd)

    # Reconcile receiving TDs / yards to passing (structural, with reserve).
    qb_pass_yds = sum(r["stats"]["pass_yards"] for r in qbs) if qbs else 0.0
    qb_pass_tds = sum(r["stats"]["pass_tds"] for r in qbs) if qbs else 0.0
    qb_comps = sum(r["stats"]["pass_completions"] for r in qbs) if qbs else 0.0
    rec_yds = sum(r["stats"]["rec_yards"] for r in receivers)
    rec_tds = sum(r["stats"]["rec_tds"] for r in receivers)
    receptions = sum(r["stats"]["receptions"] for r in receivers)

    if not qbs and pass_att > 0:
        # No modeled QB in the room: place the full team passing line in reserve so
        # receiver totals still conserve against an explicit team pass identity.
        qb_pass_yds = float(rec_yds)
        qb_pass_tds = float(rec_tds)
        qb_comps = float(receptions)
        throwaway = float(pass_att)
        reserves_pass = {
            "pass_attempts": float(pass_att),
            "pass_yards": float(rec_yds),
            "pass_tds": float(rec_tds),
            "pass_completions": float(receptions),
        }
    else:
        reserves_pass = {
            "pass_attempts": float(throwaway),
            "pass_yards": 0.0,
            "pass_tds": 0.0,
            "pass_completions": 0.0,
        }
        # Reserves absorb differences rather than proportional rescaling of mixture events.
        if rec_yds > qb_pass_yds and qbs:
            qbs[0]["stats"]["pass_yards"] += rec_yds - qb_pass_yds
            qb_pass_yds = rec_yds
        if rec_tds > qb_pass_tds and qbs:
            qbs[0]["stats"]["pass_tds"] += rec_tds - qb_pass_tds
            qb_pass_tds = rec_tds
        if receptions > qb_comps and qbs:
            qbs[0]["stats"]["pass_completions"] += receptions - qb_comps
            qb_comps = receptions

    reserve_rec_yards = max(0.0, qb_pass_yds - rec_yds)
    reserve_rec_tds = max(0.0, qb_pass_tds - rec_tds)
    reserve_receptions = max(0.0, qb_comps - receptions)

    # Reconcile first downs: pass FD := sum rec FD
    rec_fd_sum = sum(r["stats"]["rec_first_downs"] for r in receivers)
    pass_fd, rec_fd = reconcile_team_pass_rec_first_downs(
        sum(r["stats"]["pass_first_downs"] for r in qbs) if qbs else 0.0,
        rec_fd_sum,
    )
    if qbs:
        for r in qbs:
            r["stats"]["pass_first_downs"] = 0.0
        qbs[0]["stats"]["pass_first_downs"] = float(pass_fd)
    else:
        reserves_pass["pass_first_downs"] = float(pass_fd)

    team_totals = {
        "pass_attempts": float(pass_att),
        "rush_attempts": float(rush_att),
        "pass_tds": float(qb_pass_tds),
        "rush_tds": float(sum(r["stats"]["rush_tds"] for r in players_out)),
        "completions": float(qb_comps),
        "pass_yards": float(qb_pass_yds),
        "targets": float(pass_att),
        "offensive_tds": float(qb_pass_tds + sum(r["stats"]["rush_tds"] for r in players_out)),
        "rec_first_downs": float(rec_fd),
        "pass_first_downs": float(pass_fd),
    }
    reserves = {
        "targets": float(reserve_targets if receivers else pass_att),
        "receptions": float(reserve_receptions),
        "rec_yards": float(reserve_rec_yards),
        "rec_tds": float(reserve_rec_tds),
        "pass_attempts": float(reserves_pass.get("pass_attempts", throwaway)),
        "pass_yards": float(reserves_pass.get("pass_yards", 0.0)),
        "pass_tds": float(reserves_pass.get("pass_tds", 0.0)),
        "rush_attempts": float(reserve_carries),
        "rec_first_downs": 0.0,
        "pass_first_downs": float(reserves_pass.get("pass_first_downs", 0.0)),
    }

    points = float(max(0.0, rng.normal(team.mean_points * env_factor, 6.0)))
    # Prefer TD-linked scoring floor.
    td_points = 6.0 * (qb_pass_tds + sum(r["stats"]["rush_tds"] for r in players_out))
    points = max(points, td_points)
    scoring_drives = max(1.0, qb_pass_tds + sum(r["stats"]["rush_tds"] for r in players_out) + rng.poisson(2.0))
    offense = GameOffenseState(
        team_touchdowns=float(qb_pass_tds + sum(r["stats"]["rush_tds"] for r in players_out)),
        scoring_drives=float(scoring_drives),
        points_scored=points,
        pass_attempts=float(pass_att),
        rush_attempts=float(rush_att),
        turnovers=float(sum(r["stats"]["pass_ints"] for r in qbs) if qbs else 0.0),
        home=team.home,
        weather_factor=team.weather_factor,
        venue_factor=team.venue_factor,
    )

    # Strip private inputs
    cleaned = []
    for row in players_out:
        row = dict(row)
        row.pop("_input", None)
        cleaned.append(row)
    return team_totals, cleaned, reserves, offense


def generate_game_draws(
    game: ScheduledGameInput,
    *,
    draw_count: int,
    seed: int,
) -> dict[str, Any]:
    """Generate correlated joint draws for one scheduled game."""
    rng = np.random.default_rng(seed)
    if game.home is None or game.away is None:
        raise ValueError("game requires home and away TeamGameInput")

    home_blocks: dict[str, Any] = {
        "team": game.home.team,
        "team_totals_by_draw": [],
        "reserves_by_draw": [],
        "players": {},
        "kicker_by_draw": [],
        "dst_by_draw": [],
    }
    away_blocks: dict[str, Any] = {
        "team": game.away.team,
        "team_totals_by_draw": [],
        "reserves_by_draw": [],
        "players": {},
        "kicker_by_draw": [],
        "dst_by_draw": [],
    }
    # Initialize player slots
    for block, team in ((home_blocks, game.home), (away_blocks, game.away)):
        for p in team.players:
            block["players"][p.player_id] = {
                "player_id": p.player_id,
                "position": p.position,
                "team": team.team,
                "draws": [],
                "active_by_draw": [],
                "participated_by_draw": [],
                "positive_usage_by_draw": [],
            }

    for i in range(draw_count):
        # Shared game environment latent (pace / script).
        env = float(np.exp(rng.normal(0.0, 0.08)))
        home_totals, home_players, home_res, home_off = _sample_team_draw(rng, game.home, env_factor=env)
        # Opponent env mildly negatively correlated on points.
        env_away = float(np.exp(rng.normal(-0.3 * np.log(env), 0.08)))
        away_totals, away_players, away_res, away_off = _sample_team_draw(rng, game.away, env_factor=env_away)

        for block, totals, players, reserves, offense, opponent in (
            (home_blocks, home_totals, home_players, home_res, home_off, away_off),
            (away_blocks, away_totals, away_players, away_res, away_off, home_off),
        ):
            block["team_totals_by_draw"].append(totals)
            block["reserves_by_draw"].append(reserves)
            for prow in players:
                pid = prow["player_id"]
                slot = block["players"][pid]
                slot["draws"].append(prow["stats"])
                slot["active_by_draw"].append(bool(prow["active"]))
                slot["participated_by_draw"].append(bool(prow["participated"]))
                slot["positive_usage_by_draw"].append(bool(prow["positive_usage"]))
            kick = simulate_kicker_from_game(rng, offense)
            dst = simulate_dst_from_game(
                rng,
                DstGameContext(
                    opponent_points_scored=opponent.points_scored,
                    opponent_yards=opponent.pass_attempts * 7.0 + opponent.rush_attempts * 4.2,
                    pressure_rate=game.home.pressure_rate if block is home_blocks else game.away.pressure_rate,
                    turnover_prior=game.home.turnover_prior if block is home_blocks else game.away.turnover_prior,
                    home=offense.home,
                    weather_factor=offense.weather_factor,
                ),
            )
            block["kicker_by_draw"].append(kick)
            block["dst_by_draw"].append(dst)

    def finalize(block: dict[str, Any]) -> dict[str, Any]:
        return {
            "team": block["team"],
            "team_totals_by_draw": block["team_totals_by_draw"],
            "reserves_by_draw": block["reserves_by_draw"],
            "players": list(block["players"].values()),
            "kicker_by_draw": block["kicker_by_draw"],
            "dst_by_draw": block["dst_by_draw"],
        }

    return {
        "game_id": game.game_id,
        "season": game.season,
        "week": game.week,
        "kickoff": game.kickoff,
        "draw_count": draw_count,
        "seed": seed,
        "teams": [finalize(home_blocks), finalize(away_blocks)],
    }


def player_means_from_game_draws(game_draws: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """Monte Carlo means of component stats by player_id."""
    out: dict[str, dict[str, float]] = {}
    for team in game_draws.get("teams") or []:
        for player in team.get("players") or []:
            draws = player.get("draws") or []
            if not draws:
                continue
            keys = draws[0].keys()
            means = {k: float(np.mean([d.get(k, 0.0) for d in draws])) for k in keys}
            out[str(player["player_id"])] = means
    return out


def build_game_input_from_projection_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    game_id: str,
    season: int,
    week: int,
    home_team: str,
    away_team: str,
) -> ScheduledGameInput:
    """Construct engine inputs from weekly projection / mixture rows."""

    def team_input(team: str, opponent: str, home: bool) -> TeamGameInput:
        players: list[PlayerGameInput] = []
        team_rows = [r for r in rows if str(r.get("team")) == team]
        pass_att = float(np.mean([float(r.get("attempts") or 0.0) for r in team_rows if r.get("position") == "QB"]) or 34.0)
        rush_att = float(sum(float(r.get("carries") or 0.0) for r in team_rows) or 27.0)
        for r in team_rows:
            pos = str(r.get("position") or "")
            if pos not in SKILL_POSITIONS:
                continue
            pid = str(r.get("gsis_id") or r.get("player_id") or "")
            if not pid:
                continue
            play_prob = float(r.get("play_prob") if r.get("play_prob") is not None else r.get("p_active") or 1.0)
            p_part = float(r.get("p_participates") or r.get("participation_prob") or 0.85)
            p_pos = float(r.get("p_positive_usage") or 0.75)
            players.append(
                PlayerGameInput(
                    player_id=pid,
                    position=pos,
                    team=team,
                    p_active=play_prob,
                    p_participates=p_part,
                    p_positive_usage=p_pos,
                    target_share=float(r.get("pred_target_share") or r.get("target_share") or 0.05),
                    carry_share=float(r.get("pred_carry_share") or r.get("carry_share") or 0.05),
                    dropback_share=float(r.get("pred_dropback_share") or (0.97 if pos == "QB" else 0.0)),
                    catch_rate=float(r.get("pred_catch_rate") or 0.65),
                    ypa=float(r.get("pred_ypa") or 7.2),
                    ypc=float(r.get("pred_ypc") or 4.2),
                    ypr=float(r.get("pred_ypr") or 11.0),
                )
            )
        return TeamGameInput(
            team=team,
            opponent=opponent,
            home=home,
            mean_pass_attempts=pass_att if pass_att > 0 else 34.0,
            mean_rush_attempts=rush_att if rush_att > 0 else 27.0,
            players=players,
        )

    return ScheduledGameInput(
        game_id=game_id,
        season=season,
        week=week,
        home=team_input(home_team, away_team, True),
        away=team_input(away_team, home_team, False),
    )
