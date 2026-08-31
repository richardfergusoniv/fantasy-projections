"""V3 generative composition: team draw -> shares -> conversions.

Units are SEASON TOTALS throughout. The board this consumes is a long frame
of per-game rates (``pred_pg``), one row per (player, stat); the team volumes
handed in are season totals, shares are drawn on the simplex, and each
conversion draw produces a season stat line. Callers sum the emitted rows
straight into ``fantasy_pts_season``, so anything emitted at a per-game scale
silently becomes a one-game season.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.projection.models.opportunity_shares import allocate_opportunities
from src.projection.models.receiving import draw_receiving_line
from src.projection.models.rushing import draw_rushing_line
from src.projection.models.passing import draw_passing_line
from src.projection.contracts import TEAM_VOLUME_SHARES
from src.projection.team_reconcile import TARGETS_PER_ATTEMPT
from src.projection.transitions import SEASON_GAMES

# The volume stat each position group's allocation is keyed on. Selecting one
# stat per player is what keeps a player to a single line: the board carries
# one row per (player, stat), so an unfiltered position mask yields a row per
# stat and emits that many stat lines for the same player.
QB_VOLUME_STAT = "attempts"
RECEIVING_VOLUME_STAT = "targets"
RUSHING_VOLUME_STAT = "carries"

DEFAULT_TEAM_PASS_ATTEMPTS = 600.0
DEFAULT_TEAM_CARRIES = 400.0

# Plausible ranges for rates derived from the board's own predictions. A room
# with a near-zero denominator can imply an absurd ratio; clipping keeps one
# malformed row from producing a 40-yards-per-carry back.
RATE_BOUNDS = {
    "comp_rate": (0.40, 0.80),
    "yards_per_comp": (7.0, 16.0),
    "pass_td_rate": (0.005, 0.10),
    "int_rate": (0.002, 0.08),
    "catch_rate": (0.30, 0.90),
    "yards_per_rec": (4.0, 20.0),
    "rec_td_rate": (0.0, 0.20),
    "ypc": (2.0, 7.0),
    "rush_td_rate": (0.0, 0.15),
}


# Season-scale efficiency dispersion, measured as the SD of
# log(actual season efficiency / predicted) on the rolling residuals
# (scripts/fit_conversion_sigmas.py). These replace constants that were
# picked when the path drew PER-GAME lines, and they do not all move the same
# way: receiving tightens (0.35 -> ~0.28) while passing (0.20 -> 0.27) and
# especially rushing (0.25 -> 0.37/0.64) were far too tight.
SEASON_SIGMA = {
    ("WR", "receiving"): 0.273,
    ("TE", "receiving"): 0.242,
    ("RB", "receiving"): 0.315,
    ("QB", "passing"): 0.267,
    ("RB", "rushing"): 0.367,
    ("QB", "rushing"): 0.636,
}
SEASON_SIGMA_DEFAULT = {"receiving": 0.279, "passing": 0.267, "rushing": 0.468}


def _sigma(position: str, kind: str, manifest: dict | None = None) -> float:
    conversion = (manifest or {}).get("conversion_sigmas", {})
    fitted = (
        conversion.get("cells", {}).get(f"{position}:{kind}", {}).get("sigma")
        or conversion.get("defaults", {}).get(kind, {}).get("sigma")
    )
    if fitted is not None and np.isfinite(float(fitted)):
        return float(fitted)
    return SEASON_SIGMA.get(
        (position, kind), SEASON_SIGMA_DEFAULT.get(kind, 0.30))


def _ratio(numerator, denominator, bound_key: str, default: float) -> float:
    """Rate implied by two of the board's own predictions, or the default."""
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    if pd.isna(num) or pd.isna(den) or float(den) <= 0:
        return default
    lo, hi = RATE_BOUNDS[bound_key]
    return float(np.clip(float(num) / float(den), lo, hi))


def _player_stat_table(room: pd.DataFrame) -> pd.DataFrame:
    """One row per player, per-stat pred_pg as columns."""
    if room.empty:
        return pd.DataFrame()
    return room.pivot_table(
        index="player_id", columns="stat", values="pred_pg", aggfunc="first"
    )


def _player_season_table(room: pd.DataFrame) -> pd.DataFrame:
    """One row per player, per-stat SEASON volume as columns.

    Prefers ``pred_season``; falls back to ``pred_pg`` scaled by the player's
    own exposure so a board without season totals still draws in season units.
    """
    if room.empty:
        return pd.DataFrame()
    frame = room.copy()
    if "pred_season" in frame.columns:
        value = pd.to_numeric(frame["pred_season"], errors="coerce")
    else:
        value = pd.Series(np.nan, index=frame.index)
    if "projected_games" in frame.columns:
        games = pd.to_numeric(
            frame["projected_games"], errors="coerce").fillna(SEASON_GAMES)
    else:
        games = pd.Series(float(SEASON_GAMES), index=frame.index)
    fallback = pd.to_numeric(frame["pred_pg"], errors="coerce") * games
    frame["_season_value"] = value.fillna(fallback)
    return frame.pivot_table(
        index="player_id", columns="stat", values="_season_value", aggfunc="first"
    )


def _season_volume(season_table: pd.DataFrame, player_id, stat: str) -> float:
    if season_table.empty or player_id not in season_table.index:
        return 0.0
    value = pd.to_numeric(season_table.loc[player_id].get(stat), errors="coerce")
    return 0.0 if pd.isna(value) or value <= 0 else float(value)


def _allocation_room(
    room: pd.DataFrame,
    availability_games: pd.Series | None,
) -> pd.DataFrame:
    """Attach the exposure-weighted season claim used by the simplex draw."""
    out = room.copy()
    if out.empty:
        out["_allocation_prior"] = pd.Series(dtype=float)
        return out
    if availability_games is None:
        if "pred_season" in out.columns:
            prior = pd.to_numeric(out["pred_season"], errors="coerce")
        else:
            prior = pd.Series(np.nan, index=out.index)
        games = pd.to_numeric(
            out.get("projected_games", pd.Series(SEASON_GAMES, index=out.index)),
            errors="coerce",
        ).fillna(SEASON_GAMES)
    else:
        games = out["player_id"].astype(str).map(availability_games).fillna(0.0)
        prior = pd.Series(np.nan, index=out.index)
    fallback = pd.to_numeric(out["pred_pg"], errors="coerce").fillna(0.0) * games
    out["_allocation_prior"] = prior.fillna(fallback).clip(lower=0.0)
    return out


def _position_share(position: str, stat: str) -> float:
    """Fraction of the team anchor this position room actually owns.

    A room does not get the whole team total. The measured contracts put QB at
    0.941 of team pass attempts and RB at 0.810 of team carries -- the rest is
    QB scrambles, receiver sweeps and the like. Handing each room 100% put RB
    23% over and left the simulated p50 disagreeing with the board it is meant
    to describe by +8.6 points at RB and -16.9 at QB.

    Reuses TEAM_VOLUME_SHARES so the generative path and compose_board cannot
    drift apart on what a room owns.
    """
    entry = TEAM_VOLUME_SHARES.get((position, stat))
    return float(entry[1]) if entry else 1.0


def _team_volume(team_row, key: str, default: float) -> float:
    if team_row is None:
        return default
    value = pd.to_numeric(team_row.get(key, default), errors="coerce")
    if pd.isna(value) or float(value) <= 0:
        return default
    return float(value)


def reconcile_v3_generative(
    players: pd.DataFrame,
    team_environment: pd.DataFrame,
    *,
    rng: np.random.Generator,
    share_manifest: dict | None = None,
    availability_games: pd.Series | None = None,
) -> pd.DataFrame:
    """One simulation draw through the v3 opportunity + conversion graph.

    ``team_environment`` supplies SEASON team volumes per team
    (``team_pass_attempts_mean``, ``team_carries_mean``). Each position room
    splits its team's volume on the simplex, and every player's conversion
    rates are the ones implied by that player's own board predictions rather
    than a league constant, so two QBs on the same attempt volume no longer
    produce identical lines.
    """
    rows = []
    replacement_sinks: dict[str, float] = {}
    env = (
        team_environment.set_index("team")
        if "team" in team_environment.columns
        else team_environment
    )
    for team, room in players.groupby("team", observed=True):
        team_row = env.loc[team] if team in env.index else None
        pass_attempts = _team_volume(
            team_row, "team_pass_attempts_mean", DEFAULT_TEAM_PASS_ATTEMPTS)
        rush_attempts = _team_volume(
            team_row, "team_carries_mean", DEFAULT_TEAM_CARRIES)
        stats = _player_stat_table(room)
        season = _player_season_table(room)

        def rate(player_id, num_stat, den_stat, bound_key, default):
            if stats.empty or player_id not in stats.index:
                return default
            row = stats.loc[player_id]
            return _ratio(row.get(num_stat), row.get(den_stat), bound_key, default)

        # --- passing -------------------------------------------------------
        qb_room = _allocation_room(room[
            room["position"].eq("QB") & room["stat"].eq(QB_VOLUME_STAT)
        ], availability_games)
        qb_alloc = allocate_opportunities(
            qb_room,
            pass_attempts * _position_share("QB", "attempts"),
            rng=rng,
            manifest=share_manifest,
            pool_key="qb_attempts",
            prior_col="_allocation_prior",
            allow_replacement_sink=True,
        )
        replacement_sinks[f"{team}:qb_attempts"] = qb_alloc.attrs.get(
            "replacement_sink_volume", 0.0)
        for _, pl in qb_alloc.iterrows():
            pid = pl["player_id"]
            line = draw_passing_line(
                pl["allocated_volume"],
                comp_rate=rate(pid, "completions", "attempts", "comp_rate", 0.64),
                yards_per_comp=rate(
                    pid, "passing_yards", "completions", "yards_per_comp", 11.0),
                td_rate=rate(pid, "passing_tds", "attempts", "pass_td_rate", 0.045),
                int_rate=rate(pid, "interceptions", "attempts", "int_rate", 0.025),
                sigma=_sigma("QB", "passing", share_manifest),
                rng=rng,
            )
            # QB rushing. Not part of the RB carry pool -- RB owns 0.810 of
            # team carries and scrambles are in the remainder -- so it is drawn
            # from the QB's own projected carries rather than a team share.
            # Omitting it cost 18.4 fantasy points per QB, which was the whole
            # of the -19.6 gap between the simulated QB p50 and the board.
            if availability_games is None:
                qb_carries = _season_volume(season, pid, "carries")
            else:
                games = float(availability_games.get(str(pid), 0.0))
                qb_carries = (
                    float(stats.loc[pid].get("carries", 0.0)) * games
                    if not stats.empty and pid in stats.index else 0.0
                )
            if qb_carries > 0:
                rush_line = draw_rushing_line(
                    qb_carries,
                    ypc=rate(pid, "rushing_yards", "carries", "ypc", 4.3),
                    td_rate=rate(pid, "rushing_tds", "carries", "rush_td_rate", 0.02),
                    sigma=_sigma("QB", "rushing", share_manifest),
                    rng=rng,
                )
                line.update(rush_line)
            line.update({"player_id": pid, "position": "QB", "team": team})
            rows.append(line)

        # --- receiving -----------------------------------------------------
        recv_room = _allocation_room(room[
            room["position"].isin(["WR", "TE", "RB"])
            & room["stat"].eq(RECEIVING_VOLUME_STAT)
        ], availability_games)
        # WR, TE and RB compete for ONE pool of team targets, so the room is
        # keyed without position; splitting by position would hand each group
        # a full team's worth and allocate the team three times over.
        recv = allocate_opportunities(
            recv_room,
            pass_attempts * TARGETS_PER_ATTEMPT,
            rng=rng,
            manifest=share_manifest,
            group_cols=["team", "stat"],
            pool_key="receiving_targets",
            prior_col="_allocation_prior",
            allow_replacement_sink=True,
        )
        replacement_sinks[f"{team}:receiving_targets"] = recv.attrs.get(
            "replacement_sink_volume", 0.0)
        for _, pl in recv.iterrows():
            pid = pl["player_id"]
            line = draw_receiving_line(
                pl["allocated_volume"],
                catch_rate=rate(pid, "receptions", "targets", "catch_rate", 0.65),
                yards_per_rec=rate(
                    pid, "receiving_yards", "receptions", "yards_per_rec", 12.0),
                td_rate=rate(pid, "receiving_tds", "targets", "rec_td_rate", 0.04),
                sigma=_sigma(str(pl["position"]), "receiving", share_manifest),
                rng=rng,
            )
            line.update({"player_id": pid, "position": pl["position"], "team": team})
            rows.append(line)

        # --- rushing -------------------------------------------------------
        rush_room = _allocation_room(room[
            room["position"].eq("RB") & room["stat"].eq(RUSHING_VOLUME_STAT)
        ], availability_games)
        rush = allocate_opportunities(
            rush_room,
            rush_attempts * _position_share("RB", "carries"),
            rng=rng,
            manifest=share_manifest,
            pool_key="rb_carries",
            prior_col="_allocation_prior",
            allow_replacement_sink=True,
        )
        replacement_sinks[f"{team}:rb_carries"] = rush.attrs.get(
            "replacement_sink_volume", 0.0)
        for _, pl in rush.iterrows():
            pid = pl["player_id"]
            line = draw_rushing_line(
                pl["allocated_volume"],
                ypc=rate(pid, "rushing_yards", "carries", "ypc", 4.3),
                td_rate=rate(pid, "rushing_tds", "carries", "rush_td_rate", 0.02),
                sigma=_sigma("RB", "rushing", share_manifest),
                rng=rng,
            )
            line.update({"player_id": pid, "position": "RB", "team": team})
            rows.append(line)
    out = pd.DataFrame(rows)
    out.attrs["replacement_sinks"] = replacement_sinks
    return out


def team_environment_from_board(long_board: pd.DataFrame) -> pd.DataFrame:
    """Season team volumes from the board's own fitted team anchors.

    ``propagate_team_anchors`` already attaches the RidgeCV team-total
    predictions to every row, so the fitted team environment is on the board
    and does not need refitting or a hardcoded constant. Anchors are per-game;
    this returns season totals, the unit the generative path draws in.
    """
    if long_board.empty or "team" not in long_board.columns:
        return pd.DataFrame(
            columns=["team", "team_pass_attempts_mean", "team_carries_mean"])
    out = long_board.groupby("team", as_index=False).first()
    frame = pd.DataFrame({"team": out["team"]})
    for src, dest, default in (
        ("team_pass_attempts_pg_pred", "team_pass_attempts_mean", DEFAULT_TEAM_PASS_ATTEMPTS),
        ("team_carries_pg_pred", "team_carries_mean", DEFAULT_TEAM_CARRIES),
    ):
        if src in out.columns:
            per_game = pd.to_numeric(out[src], errors="coerce")
            frame[dest] = (per_game * SEASON_GAMES).fillna(default)
        else:
            frame[dest] = default
    return frame
