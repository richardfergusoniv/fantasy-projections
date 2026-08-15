"""Depth-chart load, availability overrides, and Gate B rate gating.

Does not import predict.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from src.projection.contracts import (
    CURATED_RESEARCH_DEPTH,
    DEEP_BENCH_GAMES_CAP,
    DEPTH_CHART_PATH,
    LIVE_DEPTH_CHART_PATH,
    STATUS_OVERRIDES_PATH,
)
from src.projection.depth_history import (
    PRESEASON_CHART_DEPTH,
    load_preseason_depth_chart,
)
from src.projection.depth_rates import depth_rate_factor


def load_depth_chart(target_season):
    """Membership/role chart for ``target_season``.

    Prefers ``live_depth_{season}.csv`` (injury-refreshed derived chart) when
    present; otherwise the hand-curated ``starters_{season}.csv``. Empty for
    seasons without a file so gating is a no-op on historical backtests.
    """
    if target_season != 2026:
        return pd.DataFrame(columns=["team", "position", "depth_rank", "gsis_id", "role", "confidence"])
    path = LIVE_DEPTH_CHART_PATH if os.path.exists(LIVE_DEPTH_CHART_PATH) else DEPTH_CHART_PATH
    if not os.path.exists(path):
        return pd.DataFrame(columns=["team", "position", "depth_rank", "gsis_id", "role", "confidence"])
    dc = pd.read_csv(path)
    return dc[dc["season"] == target_season]


def load_status_overrides(target_season, as_of=None):
    """Dated human / refresh availability overrides (IR → zero, PUP → cap).

    When ``as_of`` is set, only rows with ``as_of_date <= as_of`` apply (latest
    row per gsis_id+mode wins after the filter). Never auto-derived from
    roster_status — RES/PUP on seasonal_rosters stays metadata only (Phase 6).
    """
    cols = ["season", "gsis_id", "player_name", "as_of_date", "mode",
            "projected_games", "reason"]
    if target_season != 2026 or not os.path.exists(STATUS_OVERRIDES_PATH):
        return pd.DataFrame(columns=cols)
    ov = pd.read_csv(STATUS_OVERRIDES_PATH)
    ov = ov[ov["season"] == target_season].copy()
    if ov.empty:
        return ov
    ov["_as_of"] = pd.to_datetime(ov["as_of_date"], errors="coerce")
    if as_of is not None:
        cutoff = pd.to_datetime(as_of)
        ov = ov[ov["_as_of"].isna() | (ov["_as_of"] <= cutoff)]
    # Latest dated row wins per player+mode.
    ov = ov.sort_values(["gsis_id", "mode", "_as_of"])
    ov = ov.drop_duplicates(["gsis_id", "mode"], keep="last")
    return ov.drop(columns=["_as_of"]).reset_index(drop=True)


def apply_curated_availability_override(base, depth_chart):
    """Predict-time: curated membership wins for Gate A ``target_depth_rank``.

    Training still uses only the nflverse feature. When a curated chart exists
    for the target season, players off it are forced off-chart (NaN); players
    on it but missing from nflverse get their curated depth_rank clipped to
    PRESEASON_CHART_DEPTH so the availability model sees the curator’s call.
    """
    out = base.copy()
    if depth_chart.empty or "target_depth_rank" not in out.columns:
        return out
    dc = depth_chart.dropna(subset=["gsis_id"]).drop_duplicates(["gsis_id", "position"])
    curated = set(zip(dc["gsis_id"], dc["position"]))
    rank_of = dc.set_index(["gsis_id", "position"])["depth_rank"]
    keys = list(zip(out["player_id"], out["position"]))
    on_curated = pd.Series([(p, pos) in curated for p, pos in keys], index=out.index)
    out.loc[~on_curated, "target_depth_rank"] = np.nan
    need = on_curated & out["target_depth_rank"].isna()
    if need.any():
        for idx in out.index[need]:
            pid, pos = out.at[idx, "player_id"], out.at[idx, "position"]
            raw = float(rank_of.loc[(pid, pos)])
            cap = float(PRESEASON_CHART_DEPTH.get(pos, raw))
            out.at[idx, "target_depth_rank"] = min(raw, cap)
    return out


def apply_status_overrides(df, overrides):
    """Write status overrides into ``projected_games``; keep ``projected_games_raw``.

    mode=zero → 0 games. mode=cap → min(projected_games, projected_games column).
    """
    out = df.copy()
    if "projected_games_raw" not in out.columns:
        out["projected_games_raw"] = pd.to_numeric(out.get("projected_games"), errors="coerce")
    if overrides is None or overrides.empty:
        out["status_override_applied"] = False
        return out
    out["status_override_applied"] = False
    by_id = overrides.drop_duplicates("gsis_id").set_index("gsis_id")
    for pid, row in by_id.iterrows():
        mask = out["player_id"].eq(pid)
        if not mask.any():
            continue
        mode = str(row["mode"]).strip().lower()
        if mode == "zero":
            out.loc[mask, "projected_games"] = 0.0
            out.loc[mask, "status_override_applied"] = True
        elif mode == "cap":
            cap = pd.to_numeric(row.get("projected_games"), errors="coerce")
            if pd.isna(cap):
                raise ValueError(
                    f"status override for {pid} mode=cap requires projected_games"
                )
            cur = pd.to_numeric(out.loc[mask, "projected_games"], errors="coerce")
            out.loc[mask, "projected_games"] = np.minimum(cur.fillna(cap), float(cap))
            out.loc[mask, "status_override_applied"] = True
        else:
            raise ValueError(f"Unknown status override mode {mode!r} for {pid}")
    return out


def apply_deep_bench_games_cap(df):
    """Hard games cap for curated-excluded players; does not touch rates.

    Skips rows already written by a status override (zero must stay zero;
    an explicit status cap is the human’s number)."""
    out = df.copy()
    if "projected_games_raw" not in out.columns:
        out["projected_games_raw"] = pd.to_numeric(out.get("projected_games"), errors="coerce")
    if "depth_chart_status" not in out.columns:
        return out
    if "status_override_applied" in out.columns:
        status_done = out["status_override_applied"].fillna(False).astype(bool)
    else:
        status_done = False
    mask = out["depth_chart_status"].eq("deep_bench_discounted") & ~status_done
    if not mask.any():
        return out
    cur = pd.to_numeric(out.loc[mask, "projected_games"], errors="coerce")
    out.loc[mask, "projected_games"] = np.minimum(cur, DEEP_BENCH_GAMES_CAP)
    return out


def enforce_availability_chart_review(base, depth_chart, overrides, target_season, conn=None):
    """Bidirectional curated↔nflverse conflicts at researched depth.

    Forward (curated-on / nflverse-off): stderr warning only — curated already
    won via ``apply_curated_availability_override``.

    Reverse (curated-off / nflverse-on within CURATED_RESEARCH_DEPTH, and the
    player is in this projection frame): hard review failure unless a status
    override row acknowledges the exclusion (Pearsall zero, or cap ack).
    """
    if depth_chart.empty:
        return
    nfl = load_preseason_depth_chart(target_season, conn=conn)
    if nfl.empty:
        return
    keep = nfl[
        nfl["availability_rank"] <= nfl["position"].map(PRESEASON_CHART_DEPTH)
    ]
    nfl_on_avail = set(zip(keep["player_id"], keep["position"]))
    dc = depth_chart.dropna(subset=["gsis_id"])
    curated_on = set(zip(dc["gsis_id"], dc["position"]))
    name_of = dc.drop_duplicates("gsis_id").set_index("gsis_id")["player_name"]
    role_of = dc.drop_duplicates(["gsis_id", "position"]).set_index(
        ["gsis_id", "position"]
    )["role"]

    # Forward: curated listed, nflverse off at availability depth.
    forward = []
    for pid, pos in curated_on:
        if (pid, pos) not in nfl_on_avail:
            forward.append(
                (name_of.get(pid, pid), pos, role_of.get((pid, pos), "?"))
            )
    if forward:
        names = ", ".join(f"{n} ({p}, curated {r})" for n, p, r in sorted(forward))
        print(
            f"AVAILABILITY CHART DISAGREEMENT (curated-on/nflverse-off): {len(forward)} "
            f"player(s) — curated availability override applied; review chart drift: {names}",
            file=sys.stderr,
        )

    overridden = set()
    if overrides is not None and not overrides.empty:
        overridden = set(overrides["gsis_id"].dropna().astype(str))

    frame_keys = set(zip(base["player_id"], base["position"]))
    nfl_rank = nfl.set_index(["player_id", "position"])["depth_rank"]
    nfl_name = nfl.set_index(["player_id", "position"])["full_name"]
    reverse = []
    for pid, pos in frame_keys:
        if (pid, pos) in curated_on:
            continue
        max_d = CURATED_RESEARCH_DEPTH.get(pos)
        if max_d is None:
            continue
        key = (pid, pos)
        if key not in nfl_rank.index:
            continue
        rank = float(nfl_rank.loc[key])
        if rank <= max_d and str(pid) not in overridden:
            reverse.append(
                (nfl_name.get(key, pid), pos, int(rank), pid)
            )
    if reverse:
        detail = ", ".join(
            f"{n} ({p}{r}, {pid})" for n, p, r, pid in sorted(reverse)
        )
        raise ValueError(
            f"AVAILABILITY CHART REVIEW FAILURE: {len(reverse)} player(s) are on the "
            f"nflverse chart within curated research depth but absent from the curated "
            f"chart, with no status override acknowledging the exclusion. Resolve by "
            f"editing starters_{target_season}.csv or status_overrides_{target_season}.csv: "
            f"{detail}"
        )


def apply_depth_chart_gating(df, depth_chart):
    """Apply the historically calibrated veteran depth-rate factor.

    The curated chart supplies team, membership, and displayed role. The
    separate nflverse preseason rank selects ``depth_rate_factor`` because
    that is the signal calibrated in historical folds. Rows are retained and
    expose the exact multiplier in ``role_discount_factor``; rows outside the
    curated table remain low-confidence rather than silently disappearing.

    Coverage note: src/depth_chart/starters_2026.csv covers QB/RB/WR/TE at
    every one of the 32 teams down to the depths specified in
    PHASE6_REPORT.md (QB1[+2 if competitive], RB1-2, WR1-3, TE1-2) - so for
    2026 there is no "we ran out of time to research this team/position"
    gap; every non-curated veteran at a covered position is a deliberate
    scope decision (below the researched depth), not a data gap. The
    `depth_chart_status` values distinguish this explicitly:
      - 'curated': matched a row in the table (gets depth_rank/role).
      - 'deep_bench_discounted': target_season IS covered by the table
         (i.e. we researched this team's position group) but this specific
         player wasn't in the curated top-N - confirmed outside the
         relevant depth, not merely unresearched.
      - 'not_curated_no_table': target_season has no curated table at all
         (any season other than 2026) - gating is a no-op, not a claim
         about this player's role.

    Rookie rows do not call this function: the veteran-only ladder was
    harmful in the dedicated rookie test, so rookie conditional rates remain
    neutral and depth affects their availability only."""
    df = df.copy()
    if depth_chart.empty:
        df["depth_rank"] = np.nan
        df["role"] = None
        df["formation_role"] = None
        df["depth_chart_status"] = "not_curated_no_table"
        df["role_discount_applied"] = False
        df["role_discount_factor"] = 1.0
        return df
    if "nfl_depth_rank" not in df.columns:
        raise ValueError(
            "apply_depth_chart_gating needs nfl_depth_rank (Gate B) - call "
            "depth_history.attach_depth_rank(df, target_season) first. Defaulting "
            "it to NaN would silently apply the off-chart factor to every player.")

    keep = ["position", "gsis_id", "depth_rank", "role"]
    if "formation_role" in depth_chart.columns:
        keep.append("formation_role")
    dc = depth_chart[keep].rename(columns={"gsis_id": "player_id"})
    dc = dc.dropna(subset=["player_id"]).drop_duplicates(subset=["player_id", "position"])
    df = df.merge(dc, on=["player_id", "position"], how="left")
    if "formation_role" not in df.columns:
        df["formation_role"] = None

    matched = df["depth_rank"].notna()
    df["depth_chart_status"] = np.where(matched, "curated", "deep_bench_discounted")

    # The net multiplier this row received, recorded as it is applied.
    # Consumed by _compose_reframed_receiving_predictions so the Phase-7
    # elite-shrinkage correction (an ADDITIVE yards/game term, fit on
    # undiscounted out-of-sample residuals) can be scaled by the same
    # factor the rest of the row was: without it, a discounted player with
    # an elite season-N rate would have a full-size bonus added on top of
    # a depth-discounted prediction, quietly undoing the discount.
    #
    # Gate B: the factor comes from the nflverse preseason rank via
    # DEPTH_RATE_LADDER, not from the curated `role`. One lookup now covers
    # the calibrated ladder is authoritative; curated role does not select a
    # second, asserted multiplier.
    df["role_discount_factor"] = [
        depth_rate_factor(p, r) for p, r in zip(df["position"], df["nfl_depth_rank"])
    ]

    discounted = df["role_discount_factor"] < 1.0
    for col in ["pred_pg", "pred_pg_low", "pred_pg_high"]:
        df[col] = df[col] * df["role_discount_factor"]
    # low_confidence tracks the CURATED table, not the new factor: a WR4 now
    # keeps a full-size rate (fit 1.11), but "the hand-verified table does
    # not carry him" is still the honest confidence statement about him, and
    # weakening it would quietly drop ~240 players out of the flag that
    # tells a reader to check them.
    df.loc[~matched, "low_confidence"] = True
    df.loc[~matched, "role"] = "deep_bench"
    df.loc[discounted, "low_confidence"] = True
    # Now means exactly "this row's numbers were scaled down", for every
    # path. role_discount_factor beside it says by how much.
    df["role_discount_applied"] = discounted

    return df
