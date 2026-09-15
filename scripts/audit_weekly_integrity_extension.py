"""Broader week-level integrity checks beyond the static roll3 contract audit.

Research-only. Does not train, promote, reseal, or change production defaults.
Writes output/weekly_audit/integrity_extension.json.

If data/projections.db (or FANTASY_PROJECTIONS_DB_PATH) exists, also runs
live uniqueness / share / leakage / week-number / box-vs-pbp coverage checks.
Otherwise those live checks are recorded as skipped.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingest.pbp_stats_fallback import aggregate_weekly_stats_from_pbp
from src.paths import DB_PATH
from src.projection.contracts import OUTPUT_DIR
from src.projection.data.features_weekly import build_player_week_features
from src.projection.data_prep import STAT_COLS, _canonicalize_player_weeks, load_weekly_usage
from src.projection.weekly.draws.feature_outcome_split import is_allowed_prediction_column
from src.projection.weekly.features.leakage import filter_as_of
from src.projection.weekly.features.rolling import add_rolling_means
from src.projection.weekly.features.team_context import add_team_pass_rate
from src.projection.weekly.models.volume import VOLUME_FEATURE_CANDIDATES


def _check(name: str, passed: bool, detail: str, *, skipped: bool = False) -> dict:
    return {
        "name": name,
        "passed": None if skipped else bool(passed),
        "skipped": skipped,
        "detail": detail,
    }


def _weekly_row(player_id, season, week, team, position, **stats):
    row = {c: 0.0 for c in STAT_COLS}
    row.update(
        player_id=player_id,
        season=season,
        week=week,
        season_type="REG",
        recent_team=team,
        position=position,
        **stats,
    )
    return row


def synthetic_v3_usage_and_shares() -> list[dict]:
    conn = sqlite3.connect(":memory:")
    rows = [
        _weekly_row("p1", 2024, 16, "LA", "WR", targets=2),
        _weekly_row("p1", 2024, 17, "LA", "WR", targets=4),
        _weekly_row("p1", 2025, 1, "LA", "WR", targets=10),
        _weekly_row("p1", 2025, 2, "LA", "WR", targets=6),
        _weekly_row("p1", 2025, 3, "LA", "WR", targets=90),
        _weekly_row("p2", 2025, 3, "LA", "WR", targets=10),
        _weekly_row("p1", 2025, 19, "LA", "WR", targets=50),  # should drop as POST
        # Alias duplicate for same GSIS in week 4
        _weekly_row("p3", 2025, 4, "LA", "WR", targets=3),
        _weekly_row("p3", 2025, 4, "LA", "WR", targets=1),
    ]
    pd.DataFrame(rows).to_sql("weekly", conn, index=False)
    pd.DataFrame(
        [
            {"gsis_id": "p1", "position": "WR", "pfr_id": "pfr-p1"},
            {"gsis_id": "p2", "position": "WR", "pfr_id": "pfr-p2"},
            {"gsis_id": "p3", "position": "WR", "pfr_id": "pfr-p3"},
        ]
    ).to_sql("players", conn, index=False)

    usage = load_weekly_usage(conn)
    feats = build_player_week_features(conn)
    conn.close()

    checks = []
    # Week numbering: 2025 week 19 inferred POST and dropped
    weeks_2025 = sorted(usage.loc[usage["season"].eq(2025), "week"].unique().tolist())
    checks.append(
        _check(
            "week_numbering_2025_reg_le_18",
            19 not in weeks_2025 and set(weeks_2025) <= set(range(1, 19)),
            f"2025 weeks kept after REG filter: {weeks_2025}",
        )
    )

    key = ["player_id", "season", "week"]
    dupes = int(usage.duplicated(key).sum())
    checks.append(
        _check(
            "unique_player_season_week_after_canonicalize",
            dupes == 0,
            f"duplicate player-week rows after load_weekly_usage: {dupes}",
        )
    )
    p3 = usage[(usage["player_id"] == "p3") & (usage["week"] == 4)]
    checks.append(
        _check(
            "alias_rows_summed_not_double_counted",
            len(p3) == 1 and float(p3["targets"].iloc[0]) == 4.0,
            f"p3 week-4 rows={len(p3)} targets={None if p3.empty else float(p3['targets'].iloc[0])}",
        )
    )

    # Shares: non-negative, <= 1, team-week sums to 1 when volume > 0
    share_ok = True
    share_notes = []
    for col in ("targets_share", "carries_share"):
        mx = float(feats[col].max())
        mn = float(feats[col].min())
        if mx > 1.0 + 1e-9 or mn < -1e-9:
            share_ok = False
            share_notes.append(f"{col} range [{mn}, {mx}]")
    grp = feats.groupby(["season", "week", "team"], as_index=False)["targets_share"].sum()
    bad_sum = grp[(grp["targets_share"] > 0) & ((grp["targets_share"] - 1.0).abs() > 1e-9)]
    if not bad_sum.empty:
        share_ok = False
        share_notes.append(f"team-week target share sums != 1: {bad_sum.to_dict(orient='records')}")
    checks.append(
        _check(
            "shares_in_unit_interval_and_sum_to_one",
            share_ok,
            "; ".join(share_notes) or "targets/carries shares in [0,1] and team-week target shares sum to 1",
        )
    )

    # Leakage: week-3 roll3 must ignore the poisoned current-week 90-target share
    w3 = feats[(feats["player_id"] == "p1") & (feats["season"] == 2025) & (feats["week"] == 3)].iloc[0]
    # Prior rows: 2024 w16 share=1, 2024 w17 share=1, 2025 w1 share=1, 2025 w2 share=1
    # roll3 at 2025w3 = mean of last 3 prior shares = 1.0, not current 0.9
    current_share = float(w3["targets_share"])
    roll3 = float(w3["targets_share_roll3"])
    checks.append(
        _check(
            "roll3_excludes_current_week_outcome",
            abs(roll3 - 1.0) < 1e-9 and abs(current_share - 0.9) < 1e-9,
            f"week3 targets_share={current_share} roll3={roll3} (expect share=0.9, roll3=1.0 from prior weeks)",
        )
    )

    # Week 1 of 2025 uses prior-season rows because groupby is player_id only
    w1 = feats[(feats["player_id"] == "p1") & (feats["season"] == 2025) & (feats["week"] == 1)].iloc[0]
    checks.append(
        _check(
            "v3_week1_roll3_uses_prior_season_appearances",
            pd.notna(w1["targets_share_roll3"]) and float(w1["targets_share_roll3"]) == 1.0,
            f"2025 week1 roll3={w1['targets_share_roll3']} (v3 groups by player_id, so prior-season weeks flow in)",
        )
    )
    return checks


def synthetic_v2_rolling_and_as_of() -> list[dict]:
    checks = []
    panel = pl.DataFrame(
        {
            "gsis_id": ["p1"] * 6,
            "season": [2024] * 6,
            "week": [1, 2, 3, 4, 5, 6],
            "fantasy_points": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    clean = add_rolling_means(panel, ["fantasy_points"], windows=(3,))
    poisoned = panel.with_columns(
        pl.when(pl.col("week") == 4)
        .then(pl.lit(999.0))
        .otherwise(pl.col("fantasy_points"))
        .alias("fantasy_points")
    )
    dirty = add_rolling_means(poisoned, ["fantasy_points"], windows=(3,))
    same_week4 = (
        clean.filter(pl.col("week") == 4)["fantasy_points_l3"].item()
        == dirty.filter(pl.col("week") == 4)["fantasy_points_l3"].item()
    )
    later_changes = (
        clean.filter(pl.col("week") == 5)["fantasy_points_l3"].item()
        != dirty.filter(pl.col("week") == 5)["fantasy_points_l3"].item()
    )
    checks.append(
        _check(
            "v2_rolling_shift1_isolates_current_week",
            same_week4 and later_changes,
            "mutating week-4 labels left week-4 _l3 unchanged and changed week-5 _l3",
        )
    )

    as_of = filter_as_of(panel, season=2024, week=4)
    checks.append(
        _check(
            "filter_as_of_excludes_target_week",
            as_of.filter((pl.col("season") == 2024) & (pl.col("week") >= 4)).is_empty(),
            f"as-of 2024w4 retained weeks {as_of['week'].to_list()}",
        )
    )

    # v2 rolling is within season: week 1 _l3 is null
    two_season = pl.DataFrame(
        {
            "gsis_id": ["p1"] * 3,
            "season": [2024, 2024, 2025],
            "week": [17, 18, 1],
            "fantasy_points": [10.0, 20.0, 30.0],
        }
    )
    rolled = add_rolling_means(two_season, ["fantasy_points"], windows=(3,))
    w1 = rolled.filter((pl.col("season") == 2025) & (pl.col("week") == 1))["fantasy_points_l3"].item()
    checks.append(
        _check(
            "v2_rolling_does_not_cross_seasons",
            w1 is None,
            f"2025 week1 fantasy_points_l3={w1} (expect null; prior-season mean is a separate column)",
        )
    )
    return checks


def synthetic_pbp_fallback_aliases() -> list[dict]:
    pbp = pd.DataFrame(
        {
            "season": [2025, 2025],
            "week": [1, 1],
            "pass_attempt": [1, 1],
            "complete_pass": [1, 0],
            "passing_yards": [12.0, 8.0],
            "pass_touchdown": [0, 0],
            "interception": [0, 0],
            "rush_attempt": [0, 0],
            "rushing_yards": [0.0, 0.0],
            "rush_touchdown": [0, 0],
            "receiving_yards": [0.0, 0.0],
            "passer_player_id": ["00-1", "00-1"],
            "passer_player_name": ["Patrick Mahomes", "P.Mahomes"],
            "rusher_player_id": [None, None],
            "rusher_player_name": [None, None],
            "receiver_player_id": [None, None],
            "receiver_player_name": [None, None],
        }
    )
    raw = aggregate_weekly_stats_from_pbp(pbp)
    raw_dupes = int(raw.duplicated(["player_id", "season", "week"]).sum())
    prepared = raw.rename(columns={"player_name": "ignored"}).assign(
        season_type="REG", team="KC", position="QB"
    )
    for col in STAT_COLS:
        if col not in prepared.columns:
            prepared[col] = 0.0
    canon = _canonicalize_player_weeks(prepared)
    canon_dupes = int(canon.duplicated(["player_id", "season", "week"]).sum())
    return [
        _check(
            "pbp_fallback_raw_unique_player_week",
            raw_dupes == 0,
            f"raw fallback duplicate extra rows={raw_dupes}, n_rows={len(raw)}. "
            "aggregate_weekly_stats_from_pbp groups by player_id+player_name, so name aliases split one GSIS week.",
        ),
        _check(
            "canonicalize_collapses_pbp_fallback_aliases",
            canon_dupes == 0 and float(canon["attempts"].iloc[0]) == 2.0,
            f"canonical rows={len(canon)} attempts={float(canon['attempts'].iloc[0]) if len(canon) else None}",
        ),
    ]


def synthetic_team_pass_rate_same_week_join() -> list[dict]:
    player_stats = pl.DataFrame(
        {
            "season": [2024, 2024],
            "week": [1, 2],
            "team": ["LA", "LA"],
            "attempts": [30.0, 40.0],
            "carries": [20.0, 25.0],
        }
    )
    team_weeks = pl.DataFrame(
        {
            "season": [2024, 2024],
            "week": [1, 2],
            "team": ["LA", "LA"],
        }
    )
    out = add_team_pass_rate(team_weeks, player_stats)
    w2 = out.filter(pl.col("week") == 2)
    same_week_attempts = float(w2["team_attempts"].item())
    lagged_rate = w2["team_pass_rate_l5"].item()
    expected_lag = 30.0 / (30.0 + 20.0 + 1e-6)
    return [
        _check(
            "team_pass_rate_l5_is_lagged",
            abs(float(lagged_rate) - expected_lag) < 1e-6,
            f"week2 team_pass_rate_l5={lagged_rate} expected prior-week {expected_lag}",
        ),
        _check(
            "team_pass_rate_does_not_attach_same_week_team_volume",
            "team_attempts" not in out.columns and "team_carries" not in out.columns,
            f"week2 team_attempts={same_week_attempts}; add_team_pass_rate left-joins current-week "
            "team_attempts/team_carries onto team-weeks (lagged team_pass_rate_l5 is separate and safe). "
            "VOLUME_FEATURE_CANDIDATES does not include those columns.",
        ),
    ]


def model_feature_denylist_check() -> list[dict]:
    raw_outcomes = {
        "targets",
        "carries",
        "attempts",
        "target_share",
        "carry_share",
        "snap_share",
        "fantasy_points",
        "team_attempts",
        "team_carries",
    }
    leaked = sorted(raw_outcomes.intersection(VOLUME_FEATURE_CANDIDATES))
    team_aggregates = ("team_attempts", "team_carries", "team_targets", "team_air_yards")
    allowed = [c for c in team_aggregates if is_allowed_prediction_column(c)]
    lagged_ok = all(is_allowed_prediction_column(f"{c}_l3") for c in team_aggregates)
    return [
        _check(
            "volume_model_features_are_lagged_or_pregame",
            leaked == [],
            "no raw same-week box/share columns in VOLUME_FEATURE_CANDIDATES"
            if not leaked
            else f"same-week columns in volume features: {leaked}",
        ),
        _check(
            "inference_denylist_blocks_same_week_team_aggregates",
            allowed == [] and lagged_ok,
            "is_allowed_prediction_column blocks team_attempts/team_carries/team_targets/"
            "team_air_yards and still allows lagged _l3 forms"
            if allowed == [] and lagged_ok
            else f"same-week team aggregates still allowed as prediction columns: {allowed}",
        ),
    ]


def live_db_checks() -> tuple[list[dict], dict]:
    db_path = Path(DB_PATH)
    meta = {"db_path": str(db_path), "exists": db_path.exists()}
    if not db_path.exists():
        return (
            [
                _check(
                    "live_projections_db",
                    False,
                    f"projections.db not found at {db_path}; live uniqueness/share/coverage skipped",
                    skipped=True,
                )
            ],
            meta,
        )

    conn = sqlite3.connect(str(db_path))
    checks = []
    try:
        tables = {
            row[0]
            for row in conn.execute("select name from sqlite_master where type='table'").fetchall()
        }
        meta["tables"] = sorted(tables)
        if "weekly" not in tables:
            checks.append(_check("live_weekly_table", False, "weekly table missing from projections.db"))
            return checks, meta

        weekly = pd.read_sql(
            "select player_id, season, week, season_type, recent_team, position, targets, carries "
            "from weekly",
            conn,
        )
        meta["weekly_rows"] = int(len(weekly))
        meta["weekly_seasons"] = sorted(int(s) for s in weekly["season"].dropna().unique())
        meta["weekly_week_min"] = int(weekly["week"].min()) if len(weekly) else None
        meta["weekly_week_max"] = int(weekly["week"].max()) if len(weekly) else None

        raw_dupes = int(weekly.duplicated(["player_id", "season", "week"]).sum())
        checks.append(
            _check(
                "live_raw_weekly_unique_player_week",
                raw_dupes == 0,
                f"raw weekly duplicate extra rows={raw_dupes}",
            )
        )

        usage = load_weekly_usage(conn)
        usage_dupes = int(usage.duplicated(["player_id", "season", "week"]).sum())
        checks.append(
            _check(
                "live_canonical_usage_unique_player_week",
                usage_dupes == 0,
                f"canonical usage duplicate extra rows={usage_dupes}, n={len(usage)}",
            )
        )

        weeks_by_season = (
            usage.groupby("season")["week"].agg(["min", "max", "nunique"]).reset_index()
        )
        bad_weeks = usage[(usage["week"] < 1) | (usage["week"] > 18)]
        checks.append(
            _check(
                "live_canonical_weeks_in_1_18",
                bad_weeks.empty,
                f"out-of-range weeks={len(bad_weeks)}; per-season {weeks_by_season.to_dict(orient='records')}",
            )
        )

        feats = build_player_week_features(conn)
        share_max = float(feats[["targets_share", "carries_share"]].max().max())
        share_min = float(feats[["targets_share", "carries_share"]].min().min())
        roll_vs_now = feats.dropna(subset=["targets_share_roll3"])
        identical = int((roll_vs_now["targets_share_roll3"] == roll_vs_now["targets_share"]).sum())
        checks.append(
            _check(
                "live_shares_unit_interval",
                share_min >= -1e-9 and share_max <= 1.0 + 1e-9,
                f"share range [{share_min}, {share_max}]",
            )
        )
        # Identical roll3==current share can happen by coincidence; fail only if almost all match
        coincidence_rate = identical / max(len(roll_vs_now), 1)
        checks.append(
            _check(
                "live_roll3_not_mostly_equal_to_current_share",
                coincidence_rate < 0.5,
                f"{identical}/{len(roll_vs_now)} rows have targets_share_roll3 == current targets_share "
                f"(rate={coincidence_rate:.3f})",
            )
        )

        if "pbp" in tables:
            pbp_weeks = pd.read_sql(
                "select distinct season, week, posteam as team from pbp "
                "where season_type = 'REG' and posteam is not null "
                "and (pass_attempt = 1 or rush_attempt = 1)",
                conn,
            )
            box_weeks = (
                usage.dropna(subset=["team"])[["season", "week", "team"]].drop_duplicates()
            )
            merged = box_weeks.merge(
                pbp_weeks, on=["season", "week", "team"], how="outer", indicator=True
            )
            only_box = int((merged["_merge"] == "left_only").sum())
            only_pbp = int((merged["_merge"] == "right_only").sum())
            both = int((merged["_merge"] == "both").sum())
            coverage = both / max(both + only_box, 1)
            checks.append(
                _check(
                    "live_box_vs_pbp_team_week_coverage",
                    coverage >= 0.95,
                    f"both={both} box_only={only_box} pbp_only={only_pbp} coverage={coverage:.4f}",
                )
            )
        else:
            checks.append(
                _check("live_box_vs_pbp_team_week_coverage", False, "pbp table missing", skipped=True)
            )
    finally:
        conn.close()
    return checks, meta


def main() -> int:
    checks: list[dict] = []
    checks.extend(synthetic_v3_usage_and_shares())
    checks.extend(synthetic_v2_rolling_and_as_of())
    checks.extend(synthetic_pbp_fallback_aliases())
    checks.extend(synthetic_team_pass_rate_same_week_join())
    checks.extend(model_feature_denylist_check())
    live_checks, live_meta = live_db_checks()
    checks.extend(live_checks)

    failing = [c["name"] for c in checks if c["passed"] is False]
    skipped = [c["name"] for c in checks if c["skipped"]]
    report = {
        "schema_version": "weekly_integrity_extension_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "db_path": live_meta.get("db_path"),
        "db_present": bool(live_meta.get("exists")),
        "live_meta": live_meta,
        "checks": checks,
        "passes": len(failing) == 0,
        "failing_checks": failing,
        "skipped_checks": skipped,
        "notes": [
            "Official scripts/audit_weekly_features.py is a static notes/contract registry, not a live-data scan.",
            "v3 features_weekly.py groups roll3 by player_id (cross-season). v2 rolling.py groups by gsis_id+season.",
            "Player-week panels keep same-week box scores as labels; volume model features are lagged/pregame.",
            "SAME_WEEK_OUTCOME_DENYLIST must block team_attempts/team_carries/team_targets/team_air_yards (not advisory).",
            "output/weekly_audit/ is gitignored; this file is force-added so the research PR has evidence.",
        ],
    }
    out_dir = Path(OUTPUT_DIR) / "weekly_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "integrity_extension.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"passes": report["passes"], "path": str(path), "failing_checks": failing, "skipped_checks": skipped}, indent=2))
    return 0 if report["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
