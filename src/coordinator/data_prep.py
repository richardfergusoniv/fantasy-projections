"""Load and join the raw tables needed for coordinator tendency profiles.

Join keys validated against the actual DB before use (see PHASE3_REPORT.md):
- ftn -> pbp: ftn.nflverse_game_id == pbp.game_id AND ftn.nflverse_play_id ==
  pbp.play_id (NOT ftn.ftn_play_id, which is FTN's own internal id and does
  not match pbp at all - 0% match). Coverage 2022-2025 on pass/run plays:
  99.7%-100%.
- participation -> pbp: participation.nflverse_game_id == pbp.game_id AND
  participation.play_id == pbp.play_id. Both play_id columns are float64.
"""
import os
import re
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from src.db.load import DB_PATH

OFFENSE_PLAY_TYPES = ("pass", "run")


def get_conn():
    return sqlite3.connect(DB_PATH)


def load_offense_plays(conn):
    """One row per rush/pass play with the columns needed for pace/PROE.
    Excludes kneels/spikes (not representative of play-calling tendency)."""
    q = """
    SELECT game_id, play_id, season, week, posteam, defteam, drive, qtr,
           down, ydstogo, play_type, pass_attempt, rush_attempt, shotgun,
           no_huddle, qb_kneel, qb_spike, game_seconds_remaining,
           score_differential, wp, xpass, pass_oe
    FROM pbp
    WHERE play_type IN ('pass', 'run')
      AND posteam IS NOT NULL AND posteam != ''
      AND qb_kneel = 0 AND qb_spike = 0
    """
    df = pd.read_sql(q, conn)
    df["play_id"] = df["play_id"].astype("Int64")
    return df


def load_playaction(conn):
    """FTN play-action flags joined to pbp play keys. 2022+ only - ftn has
    no rows before 2022 (confirmed gap, not silently filled)."""
    q = """
    SELECT nflverse_game_id AS game_id, nflverse_play_id AS play_id,
           is_play_action, is_no_huddle, is_rpo, is_screen_pass,
           n_offense_backfield
    FROM ftn
    """
    df = pd.read_sql(q, conn)
    df["play_id"] = df["play_id"].astype("Int64")
    return df


PERSONNEL_RE = re.compile(r"(\d+)\s*(RB|TE|WR)")


def _personnel_group(text):
    """'1 RB, 1 TE, 3 WR' or the fuller OL-inclusive string -> '11'/'12'/'21' etc.
    (standard NFL shorthand: first digit = # RB, second digit = # TE)."""
    if not isinstance(text, str):
        return None
    counts = {m.group(2): int(m.group(1)) for m in PERSONNEL_RE.finditer(text)}
    if "RB" not in counts or "TE" not in counts:
        return None
    return f"{counts['RB']}{counts['TE']}"


def load_personnel(conn):
    """participation.offense_personnel joined to pbp play keys, parsed into
    standard personnel-group labels (11, 12, 21, ...). ~20% of participation
    rows have a null offense_personnel (confirmed in DB, not silently
    dropped from the caveat list) - those rows are excluded from the rate
    denominator rather than counted as a phantom group."""
    q = """
    SELECT nflverse_game_id AS game_id, play_id, offense_personnel
    FROM participation
    """
    df = pd.read_sql(q, conn)
    df["play_id"] = df["play_id"].astype("Int64")
    df["personnel_group"] = df["offense_personnel"].map(_personnel_group)
    return df
