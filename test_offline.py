"""Offline tests for the RotoWire alignment puller.

No network. Runs the parser and every validation gate against fixture records
shaped like real responses, plus deliberately corrupted variants that each gate
must catch.

    python test_offline.py

NOTE ON PROVENANCE: the task described captured real records shipped alongside
this file; they were not present in the repo. The DJ Moore 2024 fixture below is
reconstructed from the one verified example given in the spec
(17+571+114+3+1+91+267 = 1064); its aggregates are derived from those buckets.
The other fixtures are synthetic. So these tests prove the parser and gates
behave, not that the fixtures match a byte-for-byte real response.
"""

from __future__ import annotations

import copy
import sys

import pull_alignment as pa

# --------------------------------------------------------------------------
# fixtures - values are strings, as the endpoint really returns them
# --------------------------------------------------------------------------

DJ_MOORE_2024 = {
    "id": "13051",
    "player": "D.J. Moore",
    "team": "CHI",
    "pos": "WR",
    "backfield": "17",
    "leftoutside": "571",
    "leftslot": "114",
    "lefttight": "3",
    "righttight": "1",
    "rightslot": "91",
    "rightoutside": "267",
    "slot": "205",        # 114 + 91
    "outside": "838",     # 571 + 267
    "tight": "4",         # 3 + 1
    "leftside": "688",    # 571 + 114 + 3
    "rightside": "359",   # 267 + 91 + 1
    "totalplays": "1064",
}

# a low-volume player, to exercise zero buckets and small denominators
BENCH_WR = {
    "id": "99999",
    "player": "Amon-Ra St. Brown Jr.",
    "team": "DET",
    "pos": "WR",
    "backfield": "0",
    "leftoutside": "5",
    "leftslot": "0",
    "lefttight": "0",
    "righttight": "0",
    "rightslot": "2",
    "rightoutside": "1",
    "slot": "2",
    "outside": "6",
    "tight": "0",
    "leftside": "5",
    "rightside": "3",
    "totalplays": "8",
}

ZERO_SNAP = dict(
    BENCH_WR,
    id="88888",
    player="Zero Snaps",
    **{k: "0" for k in pa.BUCKETS + pa.AGGREGATES + ["totalplays"]},
)

GOOD_PAYLOAD = [DJ_MOORE_2024, BENCH_WR, ZERO_SNAP]

ERROR_PAYLOAD = {"error": "The value of the YEAR parameter is too small."}

FAILURES = []


def check(condition, label):
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}")
        FAILURES.append(label)


def expect_raises(exc_type, fn, label):
    try:
        fn()
    except exc_type:
        print(f"  PASS  {label}")
        return
    except Exception as exc:  # wrong exception type is still a failure
        print(f"  FAIL  {label} (raised {type(exc).__name__}: {exc})")
        FAILURES.append(label)
        return
    print(f"  FAIL  {label} (no exception)")
    FAILURES.append(label)


# --------------------------------------------------------------------------

def test_parsing():
    print("parsing")
    rows = pa.parse_records(GOOD_PAYLOAD, season=2024, position="WR")
    check(len(rows) == 3, "parses every record")

    moore = rows[0]
    check(moore["rotowire_id"] == "13051", "id -> rotowire_id")
    check(moore["player"] == "D.J. Moore", "player name preserved")
    check(moore["team"] == "CHI", "team preserved")
    check(moore["season"] == 2024 and moore["week"] is None, "season/week stamped")
    check(moore["position"] == "WR", "position from the query, not the row")

    check(
        all(isinstance(moore[f], int) for f in pa.BUCKETS + pa.AGGREGATES + ["totalplays"]),
        "all counts cast to int",
    )
    check(moore["totalplays"] == 1064, "totalplays value")
    check(moore["leftoutside"] == 571, "bucket value")
    check(pa._to_int("1,064") == 1064, "comma-separated ints cast")
    check(pa._to_int("") == 0 and pa._to_int(None) == 0, "blank -> 0")

    expect_raises(
        ValueError, lambda: pa._to_int("12.5"), "non-integral numeric rejected"
    )


def test_error_payloads():
    print("error payloads (HTTP 200 + object)")
    check(
        pa.payload_error(ERROR_PAYLOAD) == "The value of the YEAR parameter is too small.",
        "error object detected by type, not status code",
    )
    check(pa.payload_error(GOOD_PAYLOAD) is None, "data array is not an error")
    check(pa.payload_error("nope") is not None, "unexpected type flagged")

    expect_raises(
        pa.EndpointError,
        lambda: pa.parse_records(ERROR_PAYLOAD, 2017, "WR"),
        "parse_records raises on error object (never a silent empty)",
    )
    check(
        pa.parse_records([], 2024, "WR") == [],
        "genuinely empty array parses as zero rows",
    )
    # a row missing a field must not silently default to 0
    truncated = {k: v for k, v in DJ_MOORE_2024.items() if k != "rightslot"}
    expect_raises(
        pa.EndpointError,
        lambda: pa.parse_records([truncated], 2024, "WR"),
        "missing bucket field rejected",
    )


def test_gate1_bucket_sum():
    print("gate 1: buckets sum to totalplays")
    rows = pa.parse_records(GOOD_PAYLOAD, 2024, "WR")
    check(
        sum(rows[0][b] for b in pa.BUCKETS) == 1064,
        "DJ Moore 17+571+114+3+1+91+267 == 1064",
    )
    check(all(pa.check_bucket_sum(r) is None for r in rows), "clean rows pass gate 1")

    bad = copy.deepcopy(rows[0])
    bad["rightslot"] += 3
    problem = pa.check_bucket_sum(bad)
    check(problem is not None and "diff 3" in problem, "inflated bucket caught with diff")

    check(
        any(f["gate"] == "bucket_sum" for f in pa.validate_rows([bad])),
        "validate_rows reports gate 1 failure",
    )


def test_gate2_aggregates():
    print("gate 2: internal aggregates")
    rows = pa.parse_records(GOOD_PAYLOAD, 2024, "WR")
    check(all(pa.check_aggregates(r) == [] for r in rows), "clean rows pass gate 2")

    for agg in pa.AGGREGATES:
        bad = copy.deepcopy(rows[0])
        bad[agg] += 1
        problems = pa.check_aggregates(bad)
        check(
            len(problems) == 1 and problems[0].startswith(f"{agg}="),
            f"corrupted {agg} caught",
        )

    # a bucket edit should break the aggregates that contain it
    bad = copy.deepcopy(rows[0])
    bad["leftslot"] += 2
    broken = {p.split("=")[0] for p in pa.check_aggregates(bad)}
    check(broken == {"slot", "leftside"}, "bucket edit breaks exactly its aggregates")


def test_rates():
    print("derived rates")
    rows = [pa.add_rates(r) for r in pa.parse_records(GOOD_PAYLOAD, 2024, "WR")]
    moore = rows[0]
    check(abs(moore["slot_rate"] - 205 / 1064) < 1e-12, "slot_rate")
    check(abs(moore["wide_rate"] - 838 / 1064) < 1e-12, "wide_rate")
    check(abs(moore["inline_rate"] - 4 / 1064) < 1e-12, "inline_rate")
    check(abs(moore["backfield_rate"] - 17 / 1064) < 1e-12, "backfield_rate")
    check(
        abs(moore["slot_rate"] + moore["wide_rate"] + moore["inline_rate"]
            + moore["backfield_rate"] - 1.0) < 1e-12,
        "slot+wide+inline+backfield == 1",
    )
    check(
        abs(moore["left_rate"] + moore["right_rate"] + moore["backfield_rate"] - 1.0) < 1e-12,
        "left+right+backfield == 1",
    )
    check(
        abs(moore["side_balance"] - (359 - 688) / 1047) < 1e-12,
        "side_balance is right-minus-left over aligned snaps",
    )
    check(moore["side_balance"] < 0, "left-heavy player has negative side_balance")

    zero = rows[2]
    check(
        zero["slot_rate"] is None and zero["side_balance"] is None,
        "zero-snap row yields None rates, not a divide-by-zero",
    )


def test_name_normalization():
    print("name normalisation")
    check(pa.normalize_name("D.J. Moore") == "dj moore", "punctuation stripped")
    check(
        pa.normalize_name("Amon-Ra St. Brown") == "amon ra st brown",
        "hyphen split",
    )
    check(pa.normalize_name("Marvin Harrison Jr.") == "marvin harrison", "suffix dropped")
    check(pa.normalize_name("Michael Pittman II") == "michael pittman", "roman suffix dropped")
    check(pa.normalize_name("Equanimeous St. Brown") == pa.normalize_name("EQUANIMEOUS ST BROWN"), "case-fold")
    check(pa.normalize_name("") == "" and pa.normalize_name(None) == "", "empty safe")


def test_cache_keys_and_week_trap():
    print("request shaping (the week= trap)")
    season = pa._cache_path({"year": "2024", "pos": "WR"})
    weekly = pa._cache_path({"year": "2024", "pos": "WR", "startweek": "3", "endweek": "3"})
    check(season != weekly, "season and weekly slices cache separately")
    check("season" in season.name and "wk3-3" in weekly.name, "cache names are legible")

    other = pa._cache_path({"year": "2024", "pos": "WR", "startweek": "4", "endweek": "4"})
    check(weekly != other, "distinct weeks cache separately")

    # the puller must have no way to send `week`, since it is silently ignored
    import inspect
    src = inspect.getsource(pa.fetch)
    check('"week"' not in src and "'week'" not in src, "fetch never sends a bare week param")
    check(pa.MIN_DELAY >= 1.5, "delay floor is 1.5s")


def main():
    for test in (
        test_parsing,
        test_error_payloads,
        test_gate1_bucket_sum,
        test_gate2_aggregates,
        test_rates,
        test_name_normalization,
        test_cache_keys_and_week_trap,
    ):
        test()
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + "; ".join(FAILURES))
        return 1
    print("all offline tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
