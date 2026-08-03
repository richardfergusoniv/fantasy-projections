"""
Phase 0 validation gate for NFL projection system.
Checks data coverage for:
1. load_participation() - play-level personnel data (2023-2025)
2. FTN charting - season range and populated fields
3. was_pressure / time_to_throw availability by season
"""
import sys
import pandas as pd
import nfl_data_py as nfl

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

results = {}

# ---------------------------------------------------------------------------
# 1. load_participation coverage
# ---------------------------------------------------------------------------
section("1. load_participation() coverage 2023-2025")
try:
    part = nfl.import_pbp_participation([2023, 2024, 2025])
    print("SHAPE:", part.shape)
    print("COLUMNS:", list(part.columns))
    if "season" in part.columns:
        print(part["season"].value_counts().sort_index())
    else:
        print("No 'season' column - trying to infer from nflverse_game_id / other cols")
        print(part.head(3))
    results["participation_import"] = "ok"
except Exception as e:
    print("ERROR calling import_pbp_participation with season list:", repr(e))
    results["participation_import"] = f"error: {e}"
    part = None

# Try per-season individually in case the batch call fails for some years
section("1b. load_participation() per-season individually")
per_season_part = {}
for yr in [2021, 2022, 2023, 2024, 2025]:
    try:
        p = nfl.import_pbp_participation([yr])
        n = len(p)
        has_offense_personnel = "offense_players" in p.columns or "offense_personnel" in p.columns
        print(f"season={yr}: rows={n}, columns_sample={list(p.columns)[:15]}")
        per_season_part[yr] = {"rows": n, "columns": list(p.columns)}
    except Exception as e:
        print(f"season={yr}: ERROR: {repr(e)}")
        per_season_part[yr] = {"error": str(e)}

results["participation_per_season"] = per_season_part

# ---------------------------------------------------------------------------
# 2. FTN charting
# ---------------------------------------------------------------------------
section("2. FTN charting coverage")
ftn_results = {}
for yr in [2021, 2022, 2023, 2024, 2025]:
    try:
        ftn = nfl.import_ftn_data([yr])
        n = len(ftn)
        cols = list(ftn.columns)
        # check null rate of key fields if present
        null_report = {}
        for c in cols:
            try:
                null_report[c] = float(ftn[c].isna().mean())
            except Exception:
                pass
        print(f"FTN season={yr}: rows={n}")
        ftn_results[yr] = {"rows": n, "columns": cols, "null_rates": null_report}
    except Exception as e:
        print(f"FTN season={yr}: ERROR: {repr(e)}")
        ftn_results[yr] = {"error": str(e)}

results["ftn"] = ftn_results

# print FTN columns once fully (from most recent successful year)
section("2b. FTN column list + null rates (most recent available season)")
for yr in [2025, 2024, 2023, 2022, 2021]:
    r = ftn_results.get(yr, {})
    if "columns" in r:
        print(f"Using season {yr} for column dump. rows={r['rows']}")
        for c in r["columns"]:
            nr = r["null_rates"].get(c)
            print(f"  {c}: null_rate={nr}")
        break

# ---------------------------------------------------------------------------
# 3. was_pressure / time_to_throw availability (in PBP + NGS)
# ---------------------------------------------------------------------------
section("3. was_pressure / time_to_throw in play-by-play")
pbp_results = {}
for yr in [2021, 2022, 2023, 2024, 2025]:
    try:
        pbp = nfl.import_pbp_data([yr], downcast=True, cache=False)
        cols = pbp.columns
        info = {}
        for field in ["was_pressure", "time_to_throw", "qb_hit", "sack"]:
            if field in cols:
                info[field] = {
                    "present": True,
                    "null_rate": float(pbp[field].isna().mean()),
                    "non_null_count": int(pbp[field].notna().sum()),
                }
            else:
                info[field] = {"present": False}
        print(f"PBP season={yr}: rows={len(pbp)}")
        for field, v in info.items():
            print(f"   {field}: {v}")
        pbp_results[yr] = info
    except Exception as e:
        print(f"PBP season={yr}: ERROR: {repr(e)}")
        pbp_results[yr] = {"error": str(e)}

results["pbp_pressure_fields"] = pbp_results

# ---------------------------------------------------------------------------
# 3b. Next Gen Stats - time_to_throw is more reliably here
# ---------------------------------------------------------------------------
section("3b. Next Gen Stats passing (time_to_throw) by season")
ngs_results = {}
try:
    ngs = nfl.import_ngs_data("passing", [2021, 2022, 2023, 2024, 2025])
    print("NGS passing shape:", ngs.shape)
    print("NGS passing columns:", list(ngs.columns))
    if "season" in ngs.columns and "avg_time_to_throw" in ngs.columns:
        g = ngs.groupby("season")["avg_time_to_throw"].agg(["count", "mean", lambda s: s.isna().mean()])
        print(g)
    ngs_results["ok"] = True
except Exception as e:
    print("ERROR:", repr(e))
    ngs_results["error"] = str(e)

results["ngs_passing"] = ngs_results

# ---------------------------------------------------------------------------
# Dump raw results for reference
# ---------------------------------------------------------------------------
section("DONE - raw results dict keys")
print(list(results.keys()))

import json
with open("phase0_results.json", "w") as f:
    def default(o):
        return str(o)
    json.dump(results, f, indent=2, default=default)
print("\nWrote phase0_results.json")
