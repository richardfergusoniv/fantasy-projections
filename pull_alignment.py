"""RotoWire receiver-alignment puller.

Pulls the undocumented public JSON table at
    https://www.rotowire.com/football/tables/player-alignment.php
into tidy season and weekly tables, with aggressive validation.

Known endpoint behaviour (from the task spec):
  * `year` required, 2018 floor. `pos` required.
  * `week=N` is SILENTLY IGNORED and yields season totals. Only
    `startweek`/`endweek` actually filter, so we never send `week`.
  * every value is a JSON string; cast to int.
  * errors arrive as a JSON *object* with an `error` key under HTTP 200.
  * robots.txt blanket-blocks some UAs (e.g. wget); send a browser UA.
    `/football/tables/` itself is not disallowed.

CLI:
    python pull_alignment.py discover
    python pull_alignment.py pull [--seasons 2018-2025] [--positions WR,TE,RB]
    python pull_alignment.py build
"""

from __future__ import annotations

import argparse
import json
import hashlib
import random
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import requests

BASE_URL = "https://www.rotowire.com/football/tables/player-alignment.php"

# A normal desktop browser UA. robots.txt blanket-blocks tool UAs like wget;
# this is the identity a human hitting the same public table would present.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
)

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw" / "alignment"
OUT_DIR = ROOT / "data"

# The seven mutually exclusive alignment buckets. These must sum to totalplays.
BUCKETS = [
    "backfield",
    "leftoutside",
    "leftslot",
    "lefttight",
    "righttight",
    "rightslot",
    "rightoutside",
]

# RotoWire's own precomputed aggregates, and how they should decompose.
AGGREGATE_RULES = {
    "slot": ["leftslot", "rightslot"],
    "outside": ["leftoutside", "rightoutside"],
    "tight": ["lefttight", "righttight"],
    "leftside": ["leftoutside", "leftslot", "lefttight"],
    "rightside": ["rightoutside", "rightslot", "righttight"],
}
AGGREGATES = list(AGGREGATE_RULES)

MIN_DELAY = 1.5  # seconds. Hard floor - do not lower.


class BlockedError(RuntimeError):
    """Raised on 403/429 or anything that smells like a block. We stop, hard."""


class EndpointError(RuntimeError):
    """The endpoint returned a JSON object with an `error` key (under HTTP 200)."""


@dataclass
class Fetch:
    """One raw response plus the provenance needed to reason about it."""

    params: dict
    payload: object
    from_cache: bool
    url: str


# --------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------

def _cache_path(params: dict) -> Path:
    key = "&".join(f"{k}={params[k]}" for k in sorted(params))
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    label = f"{params.get('year','NA')}_{params.get('pos','NA')}"
    if "startweek" in params:
        label += f"_wk{params['startweek']}-{params['endweek']}"
    else:
        label += "_season"
    return RAW_DIR / f"{label}__{digest}.json"


def fetch(
    year,
    pos,
    startweek=None,
    endweek=None,
    session=None,
    delay=MIN_DELAY,
    use_cache=True,
    max_retries=4,
) -> Fetch:
    """Fetch one (year, pos, week-window) slice, caching the raw payload.

    Deliberately never sends `week` - that param is silently ignored upstream
    and would hand back season totals wearing a weekly label.
    """
    params = {"year": str(year), "pos": str(pos)}
    if startweek is not None:
        if endweek is None:
            raise ValueError("startweek requires endweek")
        params["startweek"] = str(startweek)
        params["endweek"] = str(endweek)

    path = _cache_path(params)
    if use_cache and path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        return Fetch(params, cached["payload"], True, cached["url"])

    session = session or requests.Session()
    delay = max(float(delay), MIN_DELAY)

    last_exc = None
    for attempt in range(max_retries):
        if attempt:
            # exponential backoff with jitter on transient failures
            time.sleep(min(2 ** attempt, 30) + random.random())
        try:
            resp = session.get(
                BASE_URL,
                params=params,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/json, text/plain, */*",
                },
                timeout=30,
            )
        except requests.RequestException as exc:  # transient network fault
            last_exc = exc
            continue

        if resp.status_code in (403, 429):
            raise BlockedError(
                f"HTTP {resp.status_code} for {resp.url}. Stopping as instructed - "
                "not rotating UA, not proxying."
            )
        if resp.status_code >= 500:
            last_exc = RuntimeError(f"HTTP {resp.status_code}")
            continue
        resp.raise_for_status()

        try:
            payload = resp.json()
        except ValueError as exc:
            last_exc = exc
            continue

        RAW_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"url": resp.url, "params": params, "payload": payload},
                indent=1,
            ),
            encoding="utf-8",
        )
        time.sleep(delay)
        return Fetch(params, payload, False, resp.url)

    raise RuntimeError(f"exhausted retries for {params}: {last_exc}")


def payload_error(payload) -> str | None:
    """Return the endpoint's error string, or None if the payload is a data array.

    Errors come back as an object under HTTP 200, so type is the real signal.
    """
    if isinstance(payload, dict):
        return str(payload.get("error", payload))
    if not isinstance(payload, list):
        return f"unexpected payload type {type(payload).__name__}"
    return None


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

def _to_int(value):
    """Cast RotoWire's stringly-typed numerics. Blank/None -> 0."""
    if value is None:
        return 0
    if isinstance(value, bool):
        raise TypeError("refusing to coerce bool")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != int(value):
            raise ValueError(f"non-integral numeric {value!r}")
        return int(value)
    text = str(value).strip().replace(",", "")
    if text in ("", "-", "--"):
        return 0
    return int(text)


def _first_key(record, candidates):
    for key in candidates:
        if key in record:
            return key
    return None


def parse_records(payload, season, position, week=None) -> list[dict]:
    """Turn one raw payload into tidy rows with int counts and identity fields.

    Raises EndpointError if the payload is an error object, so a failed slice
    can never be mistaken for an empty one.
    """
    err = payload_error(payload)
    if err is not None:
        raise EndpointError(err)

    id_keys = ["id", "rotowire_id", "playerID", "playerId", "player_id"]
    name_keys = ["player", "name", "playerName", "player_name", "fullName"]
    team_keys = ["team", "teamAbbrev", "team_abbrev", "tm"]
    pos_keys = ["pos", "position"]

    rows = []
    for record in payload:
        if not isinstance(record, dict):
            raise EndpointError(f"non-object row: {record!r}")

        row = {
            "rotowire_id": str(
                record.get(_first_key(record, id_keys) or "", "")
            ).strip(),
            "player": str(record.get(_first_key(record, name_keys) or "", "")).strip(),
            "team": str(record.get(_first_key(record, team_keys) or "", "")).strip(),
            # trust the queried position over any per-row field, since `pos` is
            # what actually scoped the request
            "position": position,
            "row_position": str(
                record.get(_first_key(record, pos_keys) or "", "")
            ).strip(),
            "season": int(season),
            "week": week,
        }
        for field in BUCKETS + AGGREGATES + ["totalplays"]:
            if field not in record:
                raise EndpointError(f"row missing field {field!r}: {sorted(record)}")
            row[field] = _to_int(record[field])
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

def check_bucket_sum(row) -> str | None:
    """Gate 1: the seven buckets must equal totalplays exactly."""
    total = sum(row[b] for b in BUCKETS)
    if total != row["totalplays"]:
        return f"buckets sum to {total}, totalplays={row['totalplays']} (diff {total - row['totalplays']})"
    return None


def check_aggregates(row) -> list[str]:
    """Gate 2: RotoWire's own aggregates must decompose into the raw buckets."""
    problems = []
    for agg, parts in AGGREGATE_RULES.items():
        expected = sum(row[p] for p in parts)
        if row[agg] != expected:
            problems.append(
                f"{agg}={row[agg]} but {'+'.join(parts)}={expected}"
            )
    return problems


def validate_rows(rows) -> list[dict]:
    """Run gates 1 and 2 over every row; return one record per failure."""
    failures = []
    for row in rows:
        problem = check_bucket_sum(row)
        if problem:
            failures.append({**_ident(row), "gate": "bucket_sum", "detail": problem})
        for problem in check_aggregates(row):
            failures.append({**_ident(row), "gate": "aggregates", "detail": problem})
    return failures


def _ident(row):
    return {
        "rotowire_id": row.get("rotowire_id"),
        "player": row.get("player"),
        "season": row.get("season"),
        "week": row.get("week"),
        "position": row.get("position"),
    }


# --------------------------------------------------------------------------
# derived rates
# --------------------------------------------------------------------------

def add_rates(row) -> dict:
    """Attach share-of-snap rates. side_balance is right-minus-left, normalised
    over aligned (non-backfield) snaps: +1 all right, -1 all left, 0 even."""
    total = row["totalplays"]
    out = dict(row)

    def share(numer):
        return numer / total if total else None

    out["slot_rate"] = share(row["slot"])
    out["wide_rate"] = share(row["outside"])
    out["inline_rate"] = share(row["tight"])
    out["backfield_rate"] = share(row["backfield"])
    out["left_rate"] = share(row["leftside"])
    out["right_rate"] = share(row["rightside"])
    sided = row["leftside"] + row["rightside"]
    out["side_balance"] = (
        (row["rightside"] - row["leftside"]) / sided if sided else None
    )
    return out


# --------------------------------------------------------------------------
# name normalisation (crosswalk fallback)
# --------------------------------------------------------------------------

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_name(name: str) -> str:
    """Fold to ascii, strip punctuation and generational suffixes, lowercase."""
    if not name:
        return ""
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    # periods/apostrophes elide so "D.J." -> "dj" and "Ja'Marr" -> "jamarr";
    # other punctuation (hyphens, slashes) becomes a space
    text = "".join("" if c in ".'’`" else c for c in text)
    text = "".join(c if c.isalnum() or c.isspace() else " " for c in text)
    parts = [p for p in text.split() if p and p not in SUFFIXES]
    return " ".join(parts)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _parse_seasons(text):
    if "-" in text:
        lo, hi = text.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in text.split(",")]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pull = sub.add_parser("pull", help="bulk pull seasons x positions")
    p_pull.add_argument("--seasons", default="2018-2025")
    p_pull.add_argument("--positions", default="WR,TE,RB")
    p_pull.add_argument("--delay", type=float, default=MIN_DELAY)
    p_pull.add_argument("--max-week", type=int, default=18)

    args = parser.parse_args(argv)

    if args.cmd == "pull":
        seasons = _parse_seasons(args.seasons)
        positions = [p.strip().upper() for p in args.positions.split(",")]
        session = requests.Session()
        for season in seasons:
            for pos in positions:
                fetch(season, pos, session=session, delay=args.delay)
                for week in range(1, args.max_week + 1):
                    fetch(
                        season, pos, startweek=week, endweek=week,
                        session=session, delay=args.delay,
                    )
                print(f"pulled {season} {pos}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
