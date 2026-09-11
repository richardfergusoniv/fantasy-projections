"""Team QB-room and volume conservation invariants for evaluation."""
from __future__ import annotations

import pandas as pd

from src.projection.transitions import SEASON_GAMES


DEFAULT_VOLUME_TOL = 1e-6
DEFAULT_REL_TOL = 1e-3


def assert_team_qb_room_conserved(
    allocated: pd.DataFrame,
    *,
    team_col: str = "team",
    starts_col: str = "allocated_expected_starts",
    scheduled_games: float = float(SEASON_GAMES),
    tol: float = 1e-6,
) -> None:
    """Each team's QB-room expected starts must equal scheduled games."""
    if allocated is None or allocated.empty:
        raise AssertionError("allocated room frame is empty")
    if starts_col not in allocated.columns:
        raise AssertionError(f"missing {starts_col}")
    if team_col not in allocated.columns:
        total = float(pd.to_numeric(allocated[starts_col], errors="coerce").fillna(0).sum())
        if abs(total - scheduled_games) > tol:
            raise AssertionError(
                f"QB-room starts {total} != scheduled {scheduled_games}"
            )
        return
    for team, room in allocated.groupby(team_col, dropna=False):
        total = float(pd.to_numeric(room[starts_col], errors="coerce").fillna(0).sum())
        if abs(total - scheduled_games) > tol:
            raise AssertionError(
                f"team {team}: QB-room starts {total} != scheduled {scheduled_games}"
            )


def check_team_volume_conservation(
    *,
    team: str,
    realized_pass_attempts: float,
    target_pass_attempts: float,
    realized_qb_carries: float,
    target_qb_carries: float,
    abs_tol: float = DEFAULT_VOLUME_TOL,
    rel_tol: float = DEFAULT_REL_TOL,
) -> dict:
    """Report whether team passing and QB-rushing totals conserve."""

    def _ok(realized: float, target: float) -> bool:
        if not (target == target) or not (realized == realized):  # NaN
            return False
        lim = max(abs_tol, rel_tol * abs(target))
        return abs(float(realized) - float(target)) <= lim

    pass_ok = _ok(realized_pass_attempts, target_pass_attempts)
    rush_ok = _ok(realized_qb_carries, target_qb_carries)
    report = {
        "team": team,
        "pass_ok": pass_ok,
        "rush_ok": rush_ok,
        "ok": pass_ok and rush_ok,
        "realized_pass_attempts": float(realized_pass_attempts),
        "target_pass_attempts": float(target_pass_attempts),
        "realized_qb_carries": float(realized_qb_carries),
        "target_qb_carries": float(target_qb_carries),
        "abs_tol": abs_tol,
        "rel_tol": rel_tol,
    }
    return report


def assert_team_volume_conserved(**kwargs) -> dict:
    report = check_team_volume_conservation(**kwargs)
    if not report["ok"]:
        raise AssertionError(f"team volume conservation failed: {report}")
    return report
