"""Per-draw football accounting invariants and diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence


@dataclass
class ConservationViolation:
    game_id: str
    team: str
    draw_index: int
    rule: str
    detail: str
    left: float
    right: float
    tol: float


@dataclass
class ConservationReport:
    violations: list[ConservationViolation] = field(default_factory=list)
    checked_draws: int = 0
    checked_teams: int = 0

    @property
    def ok(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checked_draws": self.checked_draws,
            "checked_teams": self.checked_teams,
            "n_violations": len(self.violations),
            "violations": [asdict(v) for v in self.violations[:200]],
        }


def _f(mapping: Mapping[str, float], key: str, default: float = 0.0) -> float:
    try:
        return float(mapping.get(key, default) or 0.0)
    except (TypeError, ValueError):
        return default


def validate_player_draw(
    stats: Mapping[str, float],
    *,
    active: bool,
    participated: bool,
) -> list[str]:
    """Return rule names violated by a single player draw."""
    bad: list[str] = []
    for key, value in stats.items():
        if key.startswith("_") or key in {"player_id", "position", "team"}:
            continue
        try:
            num = float(value)
        except (TypeError, ValueError):
            continue
        if num != num or num == float("inf") or num == float("-inf"):
            bad.append(f"non_finite:{key}")
        if num < -1e-9:
            bad.append(f"negative:{key}")
    if not active:
        if any(_f(stats, k) > 1e-9 for k in stats):
            bad.append("inactive_nonzero")
    receptions = _f(stats, "receptions")
    targets = _f(stats, "targets")
    if receptions > targets + 1e-6:
        bad.append("receptions_gt_targets")
    rush_fd = _f(stats, "rush_first_downs")
    carries = _f(stats, "rush_attempts")
    if rush_fd > carries + 1e-6:
        bad.append("rush_fd_gt_carries")
    rec_fd = _f(stats, "rec_first_downs")
    if rec_fd > receptions + 1e-6:
        bad.append("rec_fd_gt_receptions")
    if not participated:
        snaps = _f(stats, "offense_snaps")
        if snaps > 1e-9 and (_f(stats, "targets") + _f(stats, "rush_attempts") + _f(stats, "pass_attempts")) > 1e-9:
            # participation false with positive usage is inconsistent
            bad.append("nonparticipant_usage")
    return bad


def validate_team_draw(
    *,
    game_id: str,
    team: str,
    draw_index: int,
    team_totals: Mapping[str, float],
    player_stats: Sequence[Mapping[str, float]],
    reserves: Mapping[str, float] | None = None,
    tol: float = 1e-3,
) -> list[ConservationViolation]:
    """Validate team/player identities for one draw."""
    reserves = dict(reserves or {})
    viol: list[ConservationViolation] = []

    def check(rule: str, left: float, right: float, detail: str) -> None:
        if abs(left - right) > tol:
            viol.append(
                ConservationViolation(
                    game_id=game_id,
                    team=team,
                    draw_index=draw_index,
                    rule=rule,
                    detail=detail,
                    left=left,
                    right=right,
                    tol=tol,
                )
            )

    player_targets = sum(_f(p, "targets") for p in player_stats) + _f(reserves, "targets")
    player_receptions = sum(_f(p, "receptions") for p in player_stats) + _f(reserves, "receptions")
    player_rec_yards = sum(_f(p, "rec_yards") for p in player_stats) + _f(reserves, "rec_yards")
    player_rec_tds = sum(_f(p, "rec_tds") for p in player_stats) + _f(reserves, "rec_tds")
    player_pass_att = sum(_f(p, "pass_attempts") for p in player_stats) + _f(reserves, "pass_attempts")
    player_pass_yds = sum(_f(p, "pass_yards") for p in player_stats) + _f(reserves, "pass_yards")
    player_pass_tds = sum(_f(p, "pass_tds") for p in player_stats) + _f(reserves, "pass_tds")
    player_carries = sum(_f(p, "rush_attempts") for p in player_stats) + _f(reserves, "rush_attempts")
    player_rush_tds = sum(_f(p, "rush_tds") for p in player_stats) + _f(reserves, "rush_tds")
    player_rec_fd = sum(_f(p, "rec_first_downs") for p in player_stats) + _f(reserves, "rec_first_downs")
    player_pass_fd = sum(_f(p, "pass_first_downs") for p in player_stats) + _f(reserves, "pass_first_downs")

    team_pass_att = _f(team_totals, "pass_attempts")
    team_targets = _f(team_totals, "targets", team_pass_att)
    team_completions = _f(team_totals, "completions")
    team_pass_yds = _f(team_totals, "pass_yards")
    team_pass_tds = _f(team_totals, "pass_tds")
    team_carries = _f(team_totals, "rush_attempts")
    team_off_tds = _f(team_totals, "offensive_tds", team_pass_tds + _f(team_totals, "rush_tds"))
    team_rec_fd = _f(team_totals, "rec_first_downs", player_rec_fd)
    team_pass_fd = _f(team_totals, "pass_first_downs", player_pass_fd)

    check("targets_vs_attempts", player_targets, team_targets, "named+reserve targets vs team targets")
    check("qb_pass_attempts", player_pass_att, team_pass_att, "QB+reserve attempts vs team")
    check("receptions_vs_completions", player_receptions, team_completions, "receptions vs completions")
    check("pass_yards_vs_rec_yards", player_pass_yds, player_rec_yards, "pass yards vs receiving yards")
    check("pass_tds_vs_rec_tds", player_pass_tds, player_rec_tds, "pass TDs vs receiving TDs")
    check("carries", player_carries, team_carries, "carries+reserve vs team rush attempts")
    if player_rush_tds + player_rec_tds > team_off_tds + tol:
        viol.append(
            ConservationViolation(
                game_id=game_id,
                team=team,
                draw_index=draw_index,
                rule="offensive_tds_cap",
                detail="rush+rec TDs exceed team offensive TDs",
                left=player_rush_tds + player_rec_tds,
                right=team_off_tds,
                tol=tol,
            )
        )
    check("pass_fd_vs_rec_fd", player_pass_fd, player_rec_fd, "passing first downs vs receiving first downs")
    check("team_rec_fd", player_rec_fd, team_rec_fd, "player rec FD vs team")
    check("team_pass_fd", player_pass_fd, team_pass_fd, "player pass FD vs team")

    for p in player_stats:
        for rule in validate_player_draw(
            p,
            active=bool(p.get("_active", True)),
            participated=bool(p.get("_participated", True)),
        ):
            viol.append(
                ConservationViolation(
                    game_id=game_id,
                    team=team,
                    draw_index=draw_index,
                    rule=rule,
                    detail=str(p.get("player_id", "")),
                    left=_f(p, "targets"),
                    right=_f(p, "receptions"),
                    tol=tol,
                )
            )
    return viol


def validate_partition_draws(
    games: Iterable[Mapping[str, Any]],
    *,
    tol: float = 1e-3,
    max_violations: int = 500,
) -> ConservationReport:
    """Validate a collection of game draw payloads."""
    report = ConservationReport()
    for game in games:
        game_id = str(game.get("game_id") or "")
        draw_count = int(game.get("draw_count") or 0)
        report.checked_draws += draw_count
        for team_block in game.get("teams") or []:
            report.checked_teams += 1
            team = str(team_block.get("team") or "")
            totals_by_draw = team_block.get("team_totals_by_draw") or []
            players = team_block.get("players") or []
            reserves_by_draw = team_block.get("reserves_by_draw") or []
            for i in range(draw_count):
                totals = totals_by_draw[i] if i < len(totals_by_draw) else {}
                reserves = reserves_by_draw[i] if i < len(reserves_by_draw) else {}
                player_stats = []
                for player in players:
                    draws = player.get("draws") or []
                    if i < len(draws):
                        row = dict(draws[i])
                        row["player_id"] = player.get("player_id")
                        row["_active"] = (player.get("active_by_draw") or [True] * draw_count)[i]
                        row["_participated"] = (
                            player.get("participated_by_draw") or [True] * draw_count
                        )[i]
                        player_stats.append(row)
                viol = validate_team_draw(
                    game_id=game_id,
                    team=team,
                    draw_index=i,
                    team_totals=totals,
                    player_stats=player_stats,
                    reserves=reserves,
                    tol=tol,
                )
                report.violations.extend(viol)
                if len(report.violations) >= max_violations:
                    return report
    return report
