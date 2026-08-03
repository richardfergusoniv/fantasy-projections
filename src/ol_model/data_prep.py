"""Build play-level datasets for the OL attribution model, 2021-2025.

OL identification: `participation.offense_positions` gives a per-play position
label aligned by index with `offense_players`, but it is entirely NULL for
2021-2022 (only populated 2023-2025 in this nflverse release) and uses code
'T' for tackles. For 2021-2022 we fall back to `players.position` (codes
'OT'/'G'/'C'/'OL') keyed on gsis_id - this is a career/latest-known position,
not a point-in-time one, but linemen essentially never change position group
so the approximation is safe. This split is a real data-quality gap, not a
bug - it's reported in PHASE2_REPORT.md.
"""
import os
import sqlite3

import pandas as pd

from src.db.load import DB_PATH

OL_CODES_PARTICIPATION = {"T", "G", "C"}
OL_CODES_PLAYERS = {"OT", "G", "C", "OL"}

PLAY_QUERY = """
select
    p.nflverse_game_id, p.play_id, p.offense_players, p.offense_positions,
    p.defenders_in_box, p.was_pressure, p.time_to_throw,
    b.season, b.week, b.posteam, b.defteam, b.down, b.ydstogo,
    b.score_differential, b.game_seconds_remaining, b.play_type,
    b.sack, b.rushing_yards, b.epa
from participation p
join pbp b
    on p.nflverse_game_id = b.game_id and p.play_id = b.play_id
where b.season = ?
    and b.play_type in ('run', 'pass')
    and b.aborted_play = 0
    and b.down is not null
"""


def _players_position_map(conn):
    df = pd.read_sql("select gsis_id, position from players", conn)
    return dict(zip(df.gsis_id, df.position))


def _resolve_ol(row, pos_map):
    """Return list of OL gsis_ids for a play, or None if not exactly 5."""
    ids = row.offense_players.split(";")
    if row.offense_positions is not None:
        positions = row.offense_positions.split(";")
        if len(positions) != len(ids):
            return None
        ol = [i for i, pos in zip(ids, positions) if pos in OL_CODES_PARTICIPATION]
    else:
        ol = [i for i in ids if pos_map.get(i) in OL_CODES_PLAYERS]
    if len(ol) != 5 or len(set(ol)) != 5:
        return None
    return ol


def load_season(conn, season):
    """Returns (df, drop_report) where df has one row per play with an
    `ol_ids` column (list of 5 gsis_ids) and drop_report is a dict of
    reason -> count for plays excluded."""
    raw = pd.read_sql(PLAY_QUERY, conn, params=(season,))
    pos_map = _players_position_map(conn)

    ol_ids = raw.apply(lambda r: _resolve_ol(r, pos_map), axis=1)
    keep = ol_ids.notna()

    drop_report = {
        "total_plays": len(raw),
        "dropped_not_exactly_5_ol": int((~keep).sum()),
        "kept_plays": int(keep.sum()),
    }

    df = raw.loc[keep].copy()
    df["ol_ids"] = ol_ids.loc[keep]
    return df, drop_report


def _leave_one_out_pressure_rate(pass_df):
    """Opponent pass-rush quality: defteam's pressure rate on pass plays
    against every OTHER offense this season (excludes the current
    posteam-defteam matchup entirely, not just this play, to avoid the
    O-line's own performance leaking into its own opponent-quality control)."""
    grp = pass_df.groupby(["defteam", "posteam"])["pressure_outcome"].agg(["sum", "count"])
    team_totals = pass_df.groupby("defteam")["pressure_outcome"].agg(["sum", "count"])

    def lookup(defteam, posteam):
        pair = grp.loc[(defteam, posteam)]
        total = team_totals.loc[defteam]
        num = total["sum"] - pair["sum"]
        den = total["count"] - pair["count"]
        return num / den if den > 0 else float("nan")

    return pass_df.apply(lambda r: lookup(r.defteam, r.posteam), axis=1)


def build_pass_pro_dataset(df):
    """Pass protection sub-model dataset: one row per dropback play.

    Outcome (`pressure_outcome`): was_pressure where non-null; for 2021-2022
    plays where was_pressure is null (~61-62% of those seasons, NGS tracking
    gap, see PHASE2_REPORT.md), falls back to `sack` as a coarser proxy.
    This means 2021-2022 pressure_outcome undercounts hurries/hits that
    didn't end in a sack - documented limitation, not a fix.

    time_to_throw is deliberately NOT used as a control: pressure causally
    shortens time_to_throw (and sacks/scrambles truncate it), so controlling
    for it would introduce post-treatment bias into the lineman coefficients.
    """
    pp = df[df.play_type == "pass"].copy()
    pp["pressure_outcome"] = pp["was_pressure"]
    fallback = pp["pressure_outcome"].isna()
    pp.loc[fallback, "pressure_outcome"] = pp.loc[fallback, "sack"]
    pp["pressure_fallback_used"] = fallback

    pp["opp_pass_rush_quality"] = _leave_one_out_pressure_rate(pp)
    pp = pp.dropna(subset=["pressure_outcome", "opp_pass_rush_quality", "score_differential",
                            "game_seconds_remaining", "down", "ydstogo"])
    return pp


def build_run_block_dataset(df):
    """Run blocking sub-model dataset: one row per run play.

    Outcome is raw rushing_yards. nflverse pbp in this DB has no literal
    "rushing yards over expected" field (checked PRAGMA table_info(pbp) -
    only xyac_* fields exist, and those are expected-yards-AFTER-CATCH for
    receptions, not rushing). Controlling for down/ydstogo/defenders_in_box/
    score_differential in the ridge regression partials out the situational
    "expected" component, so the lineman coefficients approximate yards over
    expected conditional on those controls - documented substitute, not the
    literal nflverse field.
    """
    rb = df[df.play_type == "run"].copy()
    rb = rb.dropna(subset=["rushing_yards", "defenders_in_box", "score_differential", "down", "ydstogo"])
    return rb


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    for season in range(2021, 2026):
        df, rep = load_season(conn, season)
        print(season, rep)
