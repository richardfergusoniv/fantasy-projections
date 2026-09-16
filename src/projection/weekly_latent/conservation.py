"""Conservation invariants for Milestone 1 weekly schedule allocation."""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.projection.weekly_latent.constants import (
    CONSERVATION_ATOL,
    CONSERVATION_RTOL,
    GAMES_PER_SEASON,
    PLAYER_SHARE_POOLS,
    TEAM_VOLUME_PG_COLUMNS,
)


def _close(left: float, right: float) -> bool:
    return abs(left - right) <= max(CONSERVATION_ATOL, CONSERVATION_RTOL * max(abs(left), abs(right)))


def _check(name: str, passed: bool, detail: str, *, skipped: bool = False) -> dict[str, Any]:
    return {
        "name": name,
        "passed": None if skipped else bool(passed),
        "skipped": skipped,
        "detail": detail,
    }


def team_volume_conservation(
    team_weeks: pd.DataFrame,
    team_volume: pd.DataFrame,
) -> list[dict[str, Any]]:
    checks = []
    summed = team_weeks.groupby("team", as_index=False)[list(TEAM_VOLUME_PG_COLUMNS)].sum()
    merged = summed.merge(team_volume, on="team", suffixes=("_weekly_sum", "_season"))
    for name in TEAM_VOLUME_PG_COLUMNS:
        left = merged[f"{name}_weekly_sum"].astype(float)
        right = merged[f"{name}_season"].astype(float)
        delta = (left - right).abs()
        worst = float(delta.max()) if len(delta) else 0.0
        team = merged.loc[delta.idxmax(), "team"] if len(delta) else None
        tol = CONSERVATION_ATOL + CONSERVATION_RTOL * right.abs()
        checks.append(
            _check(
                f"{name}_sum_weeks_eq_season",
                bool((delta <= tol).all()),
                f"max_abs_delta={worst:.8f} team={team} n_teams={len(merged)}",
            )
        )
    bye = team_weeks[team_weeks["is_bye"].eq(1)]
    bye_vol = float(bye[list(TEAM_VOLUME_PG_COLUMNS)].abs().to_numpy().sum()) if len(bye) else 0.0
    checks.append(
        _check(
            "bye_weeks_zero_team_volume",
            bye_vol <= CONSERVATION_ATOL,
            f"bye_abs_volume_sum={bye_vol} n_bye_rows={len(bye)}",
        )
    )
    games = team_weeks.groupby("team")["A_team_w"].sum()
    full_slate = set(team_weeks["week"].dropna().astype(int)) >= set(range(1, 19))
    if full_slate:
        checks.append(
            _check(
                "each_team_has_17_active_weeks",
                bool((games - GAMES_PER_SEASON).abs().max() <= 1e-9),
                f"active_weeks_by_team_minmax=({float(games.min())},{float(games.max())})",
            )
        )
    else:
        checks.append(
            _check(
                "each_team_has_17_active_weeks",
                True,
                f"skipped; slate weeks={sorted(int(w) for w in team_weeks['week'].unique())} "
                f"active_minmax=({float(games.min())},{float(games.max())})",
                skipped=True,
            )
        )
    return checks


def share_conservation(shares: pd.DataFrame, player_weeks: pd.DataFrame) -> list[dict[str, Any]]:
    checks = []
    for stat in PLAYER_SHARE_POOLS:
        sub = shares[shares["stat"].eq(stat)]
        named = sub.groupby("team")["share"].sum()
        other = sub.groupby("team")["other_share"].first()
        total = named.add(other, fill_value=0.0)
        delta = (total - 1.0).abs()
        checks.append(
            _check(
                f"{stat}_share_plus_other_eq_1",
                bool((delta <= 1e-9).all()),
                f"max_abs_delta={float(delta.max()) if len(delta) else 0.0} "
                f"overflow_teams={int((sub.groupby('team')['share_mode'].first() == 'rescaled_overflow').sum())}",
            )
        )
        share_col = f"{stat}_role_share"
        other_col = f"{stat}_other_share"
        if share_col in player_weeks.columns and other_col in player_weeks.columns:
            active = player_weeks[player_weeks["is_bye"].eq(0)]
            named_w = active.groupby(["team", "week"])[share_col].sum()
            other_w = active.groupby(["team", "week"])[other_col].first()
            tot_w = named_w.add(other_w, fill_value=0.0)
            d_w = (tot_w - 1.0).abs()
            checks.append(
                _check(
                    f"{stat}_within_week_share_simplex",
                    bool((d_w <= 1e-9).all()),
                    f"max_abs_delta={float(d_w.max()) if len(d_w) else 0.0} n_team_weeks={int(d_w.size)}",
                )
            )
    return checks


def player_inactive_on_bye(player_weeks: pd.DataFrame) -> list[dict[str, Any]]:
    bye = player_weeks[player_weeks["is_bye"].eq(1)]
    vol_cols = [c for c in list(PLAYER_SHARE_POOLS) + ["fantasy_points"] if c in player_weeks.columns]
    total = float(bye[vol_cols].abs().to_numpy().sum()) if len(bye) and vol_cols else 0.0
    return [
        _check(
            "player_bye_A_and_volume_zero",
            total <= CONSERVATION_ATOL
            and (bye["A_i_w"].eq(0).all() if "A_i_w" in bye.columns and len(bye) else True),
            f"bye_player_abs_sum={total} n_bye_player_weeks={len(bye)}",
        )
    ]


def player_season_reconcile(
    player_weeks: pd.DataFrame,
    players: pd.DataFrame,
    shares: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Player weekly sums equal share × team pool (always) and sealed pred_season
    when the team-stat was ``exact_plus_other`` (no overflow rescale).
    """
    checks = []
    weekly_sum = player_weeks.groupby("player_id", as_index=False)[
        [c for c in list(PLAYER_SHARE_POOLS) + ["fantasy_points"] if c in player_weeks.columns]
    ].sum()
    merged = weekly_sum.merge(
        players[["player_id"] + [c for c in PLAYER_SHARE_POOLS if c in players.columns]],
        on="player_id",
        suffixes=("_weekly", "_board"),
    )
    modes = shares.groupby(["player_id", "stat"])["share_mode"].first().unstack("stat")
    exact_ok = True
    exact_notes = []
    for stat in PLAYER_SHARE_POOLS:
        left_col = f"{stat}_weekly" if f"{stat}_weekly" in merged.columns else stat
        right_col = f"{stat}_board" if f"{stat}_board" in merged.columns else None
        if right_col is None or left_col not in merged.columns:
            continue
        if stat not in modes.columns:
            continue
        exact_ids = set(modes.index[modes[stat].eq("exact_plus_other")])
        if not exact_ids:
            continue
        sub = merged[merged["player_id"].isin(exact_ids)]
        delta = (sub[left_col] - sub[right_col]).abs()
        scale = sub[right_col].abs().clip(lower=1.0)
        bad = delta > (CONSERVATION_ATOL + CONSERVATION_RTOL * scale)
        if bad.any():
            exact_ok = False
            row = sub.loc[delta.idxmax()]
            exact_notes.append(
                f"{stat} max_delta={float(delta.max()):.6f} player={row['player_id']}"
            )
    checks.append(
        _check(
            "player_weekly_sum_matches_board_when_shares_unscaled",
            exact_ok,
            "; ".join(exact_notes) or "exact_plus_other player-stats reconcile to sealed pred_season",
        )
    )
    return checks


def no_forbidden_training_features(team_weeks: pd.DataFrame, player_weeks: pd.DataFrame) -> list[dict[str, Any]]:
    from src.projection.weekly_latent.constants import FORBIDDEN_SAME_WEEK_TRAINING_FEATURES

    present = sorted(
        (set(team_weeks.columns) | set(player_weeks.columns))
        & FORBIDDEN_SAME_WEEK_TRAINING_FEATURES
    )
    return [
        _check(
            "no_same_week_team_attempts_training_columns",
            present == [],
            "forbidden columns absent"
            if not present
            else f"present={present}; these are the PR #70 add_team_pass_rate attachments",
        )
    ]


def evaluate_conservation(
    team_weeks: pd.DataFrame,
    team_volume: pd.DataFrame,
    shares: pd.DataFrame,
    player_weeks: pd.DataFrame,
    players: pd.DataFrame,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    checks.extend(team_volume_conservation(team_weeks, team_volume))
    checks.extend(share_conservation(shares, player_weeks))
    checks.extend(player_inactive_on_bye(player_weeks))
    checks.extend(player_season_reconcile(player_weeks, players, shares))
    checks.extend(no_forbidden_training_features(team_weeks, player_weeks))
    failing = [c["name"] for c in checks if c["passed"] is False]
    return {
        "schema_version": "weekly_latent_m1_conservation_v1",
        "passes": len(failing) == 0,
        "failing_checks": failing,
        "n_checks": len(checks),
        "checks": checks,
        "invariant": (
            "M1 locks team-volume conservation (sum_w V_t,k,w = V_t,k_season) "
            "and within-week share conservation (named + other = 1) before any "
            "matchup effect is allowed to change season totals. Player-season "
            "points reconcile to the sealed board when named volume did not "
            "overflow the team pool."
        ),
    }


def m2_multiplicative_identity(team_weeks: pd.DataFrame) -> list[dict[str, Any]]:
    """V_w = A * (V_sealed / n_active) * m_HA * m_opp * m_env on each row."""
    checks = []
    pass_stats = ("team_pass_attempts", "team_passing_yards")
    rush_stats = ("team_rush_attempts", "team_rushing_yards")
    for name, matchup_col in (
        *((s, "pass_matchup_mult") for s in pass_stats),
        *((s, "rush_matchup_mult") for s in rush_stats),
    ):
        need = {name, f"season_{name}", "A_team_w", "n_active_weeks", matchup_col}
        if not need.issubset(team_weeks.columns):
            checks.append(
                _check(
                    f"{name}_m2_identity",
                    False,
                    f"missing columns {sorted(need - set(team_weeks.columns))}",
                )
            )
            continue
        n_active = team_weeks["n_active_weeks"].astype(float)
        sealed = team_weeks[f"season_{name}"].astype(float)
        baseline = pd.Series(0.0, index=team_weeks.index)
        ok = n_active > 1e-12
        baseline.loc[ok] = sealed.loc[ok] / n_active.loc[ok]
        intended = (
            team_weeks["A_team_w"].astype(float)
            * baseline
            * team_weeks[matchup_col].astype(float)
        )
        bye = team_weeks["is_bye"].eq(1)
        intended.loc[bye] = 0.0
        delta = (team_weeks[name].astype(float) - intended).abs()
        worst = float(delta.max()) if len(delta) else 0.0
        checks.append(
            _check(
                f"{name}_m2_identity",
                bool((delta <= CONSERVATION_ATOL + CONSERVATION_RTOL * intended.abs()).all()),
                f"max_abs_delta={worst:.8f}",
            )
        )
    return checks


def season_mass_vs_sealed(
    team_weeks: pd.DataFrame,
    team_volume: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Measure (do not require) season-total movement vs the sealed prior."""
    checks = []
    summed = team_weeks.groupby("team", as_index=False)[list(TEAM_VOLUME_PG_COLUMNS)].sum()
    merged = summed.merge(team_volume, on="team", suffixes=("_weekly_sum", "_season"))
    moved = False
    details = []
    for name in TEAM_VOLUME_PG_COLUMNS:
        left = merged[f"{name}_weekly_sum"].astype(float)
        right = merged[f"{name}_season"].astype(float)
        ratio = left / right.replace(0.0, pd.NA)
        delta = (left - right).abs()
        worst = float(delta.max()) if len(delta) else 0.0
        if (delta > (CONSERVATION_ATOL + CONSERVATION_RTOL * right.abs())).any():
            moved = True
        details.append(
            f"{name} max_abs_delta={worst:.6f} "
            f"ratio_minmax=({float(ratio.min()):.4f},{float(ratio.max()):.4f})"
        )
    checks.append(
        _check(
            "season_mass_may_differ_from_sealed",
            True,
            ("moved_vs_sealed=yes; " if moved else "moved_vs_sealed=no (all m≈1); ")
            + "; ".join(details),
        )
    )
    return checks


def m1_still_conserves_with_same_priors(
    m1_team_weeks: pd.DataFrame | None,
    team_volume: pd.DataFrame,
) -> list[dict[str, Any]]:
    if m1_team_weeks is None:
        return [
            _check(
                "m1_comparison_still_conserves",
                True,
                "skipped; no M1 comparison tables",
                skipped=True,
            )
        ]
    nested = team_volume_conservation(m1_team_weeks, team_volume)
    all_pass = all(c["passed"] is not False for c in nested)
    failing = [c["name"] for c in nested if c["passed"] is False]
    return [
        _check(
            "m1_comparison_still_conserves",
            all_pass,
            "M1 with the same priors still sums to sealed"
            if all_pass
            else f"M1 comparison failed: {failing}",
        )
    ]


def evaluate_m2(
    team_weeks: pd.DataFrame,
    team_volume: pd.DataFrame,
    shares: pd.DataFrame,
    player_weeks: pd.DataFrame,
    players: pd.DataFrame,
    *,
    m1_team_weeks: pd.DataFrame | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    checks.extend(m2_multiplicative_identity(team_weeks))
    checks.extend(season_mass_vs_sealed(team_weeks, team_volume))
    checks.extend(m1_still_conserves_with_same_priors(m1_team_weeks, team_volume))
    checks.extend(share_conservation(shares, player_weeks))
    checks.extend(player_inactive_on_bye(player_weeks))
    checks.extend(no_forbidden_training_features(team_weeks, player_weeks))
    from src.projection.weekly_latent.constants import FORBIDDEN_MARKET_DRIVERS

    market = sorted(
        (set(team_weeks.columns) | set(player_weeks.columns)) & FORBIDDEN_MARKET_DRIVERS
    )
    checks.append(
        _check(
            "no_adp_or_vegas_driver_columns",
            market == [],
            "market driver columns absent" if not market else f"present={market}",
        )
    )
    failing = [c["name"] for c in checks if c["passed"] is False]
    return {
        "schema_version": "weekly_latent_m2_identities_v1",
        "passes": len(failing) == 0,
        "failing_checks": failing,
        "n_checks": len(checks),
        "checks": checks,
        "invariant": (
            "M2 does NOT conserve sealed season team volume. Identity: "
            "V_w = A_w * (V_sealed / n_active) * m_HA * m_opp * m_env "
            "(no renormalize). Named shares + other = 1 still holds. "
            "Bye volume is 0. Same-week team_attempts/carries/targets/"
            "air_yards and ADP/season Vegas are forbidden drivers. "
            "Season totals may differ from the sealed prior; that delta "
            "is measured, not a failure."
        ),
    }
