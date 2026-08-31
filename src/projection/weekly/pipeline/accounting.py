"""Accounting constraints: force player stats to sum to team totals."""
# Ported from fantasy-projections-2 (team-first weekly v2). See docs/WEEKLY_V2_PORT_PROVENANCE.md.

from __future__ import annotations

import polars as pl


THROWAY_RESERVE = 0.04  # ~4% of pass attempts as throwaways / unassigned
OTHER_RECEIVER_RESERVE = 0.04  # residual targets for non-rostered / depth players
DEFAULT_COMPLETION_RATE = 0.65
STARTER_DROPBACK_SHARE = 0.97
COMMITTEE_SHARE_GAP = 0.12  # if #1 and #2 within this and both material, keep split
COMMITTEE_SECOND_MIN = 0.28

# Multiply model shares by this before normalize (depth_rank → weight).
# Missing depth chart → treated as deep bench so they don't vacuum volume.
# Carry vs target tables are separate — one schedule cannot fit both RB carries
# and WR/TE targets. DEPTH_PRIOR_STRENGTH blends toward identity to avoid
# double-crushing with volume models that already use depth_rank as a feature.
# Kept below 0.75 so listed WR2/TE1s retain enough raw volume signal (Flowers
# vacuum / Bowers compression) when the two-stage model already encodes depth.
DEPTH_PRIOR_STRENGTH = 0.5500

_CARRY_DEPTH_WEIGHTS = {
    1: 1.0000,
    2: 0.6544,
    3: 0.3599,
    4: 0.2500,
    5: 0.2000,
}
_TARGET_DEPTH_WEIGHTS = {
    1: 1.0000,
    2: 0.7000,  # keep listed WR2/TE2 from collapsing into WR1 vacuum
    3: 0.4000,
    4: 0.2500,
    5: 0.2000,
}
_CARRY_DEPTH_DEFAULT = 0.4380  # unlisted / null depth_rank
_TARGET_DEPTH_DEFAULT = 0.4008
_CARRY_DEPTH_FLOOR = 0.2190  # rank 6+
_TARGET_DEPTH_FLOOR = 0.2004

# Back-compat alias (rookie/stub paths and older imports)
_DEPTH_SHARE_WEIGHTS = _TARGET_DEPTH_WEIGHTS
_DEPTH_SHARE_DEFAULT = _TARGET_DEPTH_DEFAULT
_DEPTH_SHARE_FLOOR = _TARGET_DEPTH_FLOOR

# Expected chart slots; missing ones become ghost reserve (Diggs/Deebo vacuum fix)
_EXPECTED_TARGET_SLOTS: list[tuple[str, int]] = [
    ("WR", 1),
    ("WR", 2),
    ("WR", 3),
    ("TE", 1),
]
_EXPECTED_CARRY_SLOTS: list[tuple[str, int]] = [
    ("RB", 1),
    ("RB", 2),
]

# Hard caps after normalize — leftover stays in reserve (~95th historical by pos)
# TE raised toward historical p95 so TE1s (Bowers) are not clipped below WR1s.
# WR slightly lowered so vacated volume cannot pin a single WR at 32%.
# TE cap closer to WR so mid-TE1s cannot vacuum WR targets after the elite floor.
MAX_TARGET_SHARE = {"WR": 0.28, "TE": 0.28, "RB": 0.22}
MAX_CARRY_SHARE = 0.70
MAX_QB_CARRY_SHARE = 0.24  # designed + scramble; dual-threat QBs run ~20-23%

# Efficiency clips (position-aware; widened so medians are not clipped)
MAX_CATCH_RATE = 0.88
MAX_CATCH_RATE_BY_POS = {"RB": 0.88, "WR": 0.85, "TE": 0.85}
MAX_YPR = 16.0
MAX_YPR_BY_POS = {"RB": 14.0, "WR": 16.0, "TE": 15.0}

# Box-score columns that must come from projections, never panel actuals
_BOX_SCORE_COLS = (
    "attempts",
    "completions",
    "passing_yards",
    "passing_tds",
    "interceptions",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "fumbles_lost",
)


def normalize_shares(
    df: pl.DataFrame,
    share_col: str,
    *,
    group_keys: list[str],
    target_sum: float = 1.0,
    min_share: float = 0.0,
) -> pl.DataFrame:
    """Rescale predicted shares within a group so they sum to target_sum."""
    clipped = pl.col(share_col).fill_null(0.0).clip(min_share, 1.0)
    totals = df.group_by(group_keys).agg(clipped.sum().alias("_share_sum"))
    out = df.join(totals, on=group_keys, how="left")
    return out.with_columns(
        (
            clipped
            / pl.when(pl.col("_share_sum") <= 1e-9)
            .then(pl.lit(1.0))
            .otherwise(pl.col("_share_sum"))
            * target_sum
        ).alias(share_col)
    ).drop("_share_sum")


def depth_share_weight(
    rank: float | None,
    *,
    table: str = "target",
) -> float:
    """Return depth prior weight. ``table`` is 'target' (default) or 'carry'.

    Rookie/season stubs use target weights (receiving-side volume).
    """
    weights, default, floor = _weight_table(table)
    if rank is None:
        return default
    r = int(round(float(rank)))
    if r >= 6:
        return floor
    return float(weights.get(r, default))


def rookie_role_confidence(
    rank: float | None,
    *,
    position: str | None = None,
) -> float:
    """How much of a rookie's projection the draft-capital prior should own.

    ``pred_rookie_fp_pg`` is trained on realized points *per game played*, so it
    already describes a rookie who has a role.  It cannot be rescaled by
    ``depth_share_weight`` -- that is an absolute share multiplier, only safe on
    quantities renormalized to a team total afterwards, and it would cut every
    rookie without a published depth chart by ~60% for no reason.

    Instead this returns a weight for interpolating toward the veteran track,
    which already routes through depth-aware volume models and accounting.  A
    buried rookie therefore inherits a real low projection rather than a
    scaled-down starter one, and the answer stays position-appropriate for
    free: a QB2's veteran projection is near zero, a WR4's is small but real.

    QB role is squared because dropbacks are winner-take-all -- accounting
    already hands the listed starter ``STARTER_DROPBACK_SHARE`` -- so the share
    curve understates how sharply a backup's playing time falls off.
    """
    pos = (position or "").upper()
    table = "carry" if pos in ("QB", "RB") else "target"
    role = depth_share_weight(rank, table=table)
    if pos == "QB":
        role = role * role
    return min(1.0, max(0.0, role))


def _weight_table(table: str) -> tuple[dict[int, float], float, float]:
    if table == "carry":
        return _CARRY_DEPTH_WEIGHTS, _CARRY_DEPTH_DEFAULT, _CARRY_DEPTH_FLOOR
    return _TARGET_DEPTH_WEIGHTS, _TARGET_DEPTH_DEFAULT, _TARGET_DEPTH_FLOOR


def depth_share_weight_expr(
    depth_col: str = "depth_rank",
    *,
    table: str = "target",
) -> pl.Expr:
    """Depth prior expression for carry or target shares."""
    weights, default, floor = _weight_table(table)
    rank = pl.col(depth_col).cast(pl.Float64)
    weight = pl.lit(default)
    for r, w in sorted(weights.items()):
        weight = pl.when(rank == float(r)).then(pl.lit(w)).otherwise(weight)
    weight = pl.when(rank.is_null()).then(pl.lit(default)).otherwise(weight)
    weight = pl.when(rank >= 6.0).then(pl.lit(floor)).otherwise(weight)
    return weight


def apply_depth_share_priors(
    df: pl.DataFrame,
    *,
    strength: float | None = None,
) -> pl.DataFrame:
    """Damp predicted volume shares using depth_rank before team normalize.

    ``strength`` (α) blends toward identity: share *= (1−α) + α·w so volume
    models that already use depth_rank are not double-crushed when α < 1.
    """
    out = df
    alpha = float(DEPTH_PRIOR_STRENGTH if strength is None else strength)
    alpha = min(1.0, max(0.0, alpha))
    if "available" in out.columns:
        for col in ("pred_carry_share", "pred_target_share", "pred_dropback_share"):
            if col in out.columns:
                out = out.with_columns(
                    pl.when(pl.col("available").fill_null(True))
                    .then(pl.col(col).fill_null(0.0))
                    .otherwise(pl.lit(0.0))
                    .alias(col)
                )
    if "depth_rank" not in out.columns or alpha <= 1e-12:
        return out

    carry_w = depth_share_weight_expr(table="carry")
    target_w = depth_share_weight_expr(table="target")
    out = out.with_columns(
        [
            ((1.0 - alpha) + alpha * carry_w).alias("_carry_depth_w"),
            ((1.0 - alpha) + alpha * target_w).alias("_target_depth_w"),
        ]
    )
    if "pred_carry_share" in out.columns:
        out = out.with_columns(
            (pl.col("pred_carry_share").fill_null(0.0) * pl.col("_carry_depth_w")).alias(
                "pred_carry_share"
            )
        )
    if "pred_dropback_share" in out.columns:
        # Dropbacks use carry table (QB depth chart)
        out = out.with_columns(
            (pl.col("pred_dropback_share").fill_null(0.0) * pl.col("_carry_depth_w")).alias(
                "pred_dropback_share"
            )
        )
    if "pred_target_share" in out.columns:
        out = out.with_columns(
            (pl.col("pred_target_share").fill_null(0.0) * pl.col("_target_depth_w")).alias(
                "pred_target_share"
            )
        )
    return out.drop([c for c in ("_carry_depth_w", "_target_depth_w") if c in out.columns])


# Floors applied after depth priors, before team normalize. Stops WR1 vacuum
# (Flowers) and TE1 compression (Bowers) when the volume model collapses the room.
WR2_MIN_FRAC_OF_WR1 = 0.50
# Elite TE1 floor only for true high-usage anchors. At 0.18 nearly every
# starting TE qualified, which pinned Schultz/Goedert/Okonkwo near Bowers and
# erased TE hierarchy on the draft board.
TE1_MIN_TARGET_SHARE = 0.20
TE1_ELITE_PRIOR_SHARE = 0.22


def apply_receiver_share_floors(
    df: pl.DataFrame,
    *,
    group_keys: list[str] | None = None,
    wr2_frac_of_wr1: float = WR2_MIN_FRAC_OF_WR1,
    te1_min_share: float = TE1_MIN_TARGET_SHARE,
    te1_elite_prior: float = TE1_ELITE_PRIOR_SHARE,
) -> pl.DataFrame:
    """Raise listed WR2 / elite TE1 target floors by taking share from WR1/WR3+.

    Operates on raw predicted shares before normalize so team totals still sum
    correctly afterward. TE1 floors require prior-season (or l5) target share
    above ``te1_elite_prior`` so replacement TE1s are not mass-inflated.
    """
    keys = group_keys or [c for c in ("season", "week", "team") if c in df.columns]
    if (
        df.is_empty()
        or "pred_target_share" not in df.columns
        or "position" not in df.columns
        or "depth_rank" not in df.columns
        or not keys
    ):
        return df

    frames: list[pl.DataFrame] = []
    for key_vals in df.select(keys).unique().iter_rows(named=True):
        mask = pl.lit(True)
        for k, v in key_vals.items():
            mask = mask & (pl.col(k) == v)
        chunk = df.filter(mask)

        n = chunk.height
        shares = [float(s or 0.0) for s in chunk["pred_target_share"].to_list()]
        positions = chunk["position"].to_list()
        depths = chunk["depth_rank"].to_list()
        prior = (
            chunk["target_share_prior_season"].to_list()
            if "target_share_prior_season" in chunk.columns
            else [None] * n
        )
        l5 = (
            chunk["target_share_l5"].to_list()
            if "target_share_l5" in chunk.columns
            else [None] * n
        )

        available = [True] * n
        if "available" in chunk.columns:
            available = [bool(a if a is not None else True) for a in chunk["available"].to_list()]
        if "play_prob" in chunk.columns:
            probs = chunk["play_prob"].to_list()
            available = [
                available[i] and (probs[i] is None or float(probs[i]) > 1e-6)
                for i in range(n)
            ]

        def _rank(i: int) -> int | None:
            if depths[i] is None:
                return None
            return int(round(float(depths[i])))

        def _anchor_share(i: int) -> float:
            vals = []
            if prior[i] is not None:
                vals.append(float(prior[i]))
            if l5[i] is not None:
                vals.append(float(l5[i]))
            return max(vals) if vals else 0.0

        wr1_i = wr2_i = te1_i = None
        for i in range(n):
            if not available[i]:
                continue
            r = _rank(i)
            if r is None:
                continue
            if positions[i] == "WR" and r == 1 and wr1_i is None:
                wr1_i = i
            elif positions[i] == "WR" and r == 2 and wr2_i is None:
                wr2_i = i
            elif positions[i] == "TE" and r == 1 and te1_i is None:
                te1_i = i

        if wr1_i is not None and wr2_i is not None:
            floor = max(0.0, wr2_frac_of_wr1) * shares[wr1_i]
            if shares[wr2_i] < floor and shares[wr1_i] > floor:
                take = min(floor - shares[wr2_i], shares[wr1_i] - floor)
                shares[wr1_i] -= take
                shares[wr2_i] += take

        elite_te = te1_i is not None and _anchor_share(te1_i) >= te1_elite_prior
        if elite_te:
            # Prefer the player's own prior usage when it exceeds the flat floor.
            te_floor = max(te1_min_share, 0.85 * _anchor_share(te1_i))
            te_floor = min(te_floor, MAX_TARGET_SHARE.get("TE", 0.28))
            if shares[te1_i] < te_floor:
                # Raise in place — do not require WR donors. Low-volume rooms
                # (LV) have no WR1 surplus to steal; normalize then preserves
                # the TE1's relative weight when shares are rescaled to the cap.
                shares[te1_i] = te_floor

        frames.append(chunk.with_columns(pl.Series("pred_target_share", shares)))

    return pl.concat(frames, how="diagonal_relaxed") if frames else df


def _slot_weight(rank: int, *, table: str = "target") -> float:
    weights, default, floor = _weight_table(table)
    r = int(rank)
    if r >= 6:
        return floor
    return float(weights.get(r, default))


def ghost_slot_weights(
    team_players: pl.DataFrame,
    expected_slots: list[tuple[str, int]],
    *,
    table: str = "target",
) -> tuple[float, float]:
    """Return (present_weight, ghost_weight) for expected depth slots on one team-week.

    If depth_rank is unavailable, treat the room as fully present (no ghost shrink).
    Players with ``available=False`` or ``play_prob≈0`` do not fill slots.
    ``table`` selects carry vs target weight schedule for slot magnitudes.
    """
    if team_players.is_empty() or "position" not in team_players.columns:
        return 1.0, 0.0
    if "depth_rank" not in team_players.columns:
        return 1.0, 0.0
    if team_players["depth_rank"].null_count() == team_players.height:
        return 1.0, 0.0

    eligible = team_players
    if "available" in eligible.columns:
        eligible = eligible.filter(pl.col("available").fill_null(True))
    if "play_prob" in eligible.columns:
        eligible = eligible.filter(pl.col("play_prob").fill_null(1.0) > 1e-6)
    elif "is_out" in eligible.columns:
        eligible = eligible.filter(~pl.col("is_out").fill_null(False))

    filled: set[tuple[str, int]] = set()
    for pos, rank in zip(
        eligible["position"].to_list(),
        eligible["depth_rank"].to_list(),
        strict=False,
    ):
        if pos is None or rank is None:
            continue
        filled.add((str(pos), int(round(float(rank)))))

    present = 0.0
    ghost = 0.0
    for pos, rank in expected_slots:
        w = _slot_weight(rank, table=table)
        if (pos, rank) in filled:
            present += w
        else:
            ghost += w
    # Avoid zeroing the room if chart ranks don't line up with expected slots
    if present <= 1e-9 and ghost > 0:
        return 1.0, 0.0
    return present, ghost


def team_volume_caps_from_ghosts(
    df: pl.DataFrame,
    *,
    group_keys: list[str],
    base_target_cap: float,
    base_carry_cap: float = 0.95,
) -> pl.DataFrame:
    """Per team-week target/carry caps shrunk by missing depth-chart slots."""
    if df.is_empty():
        return pl.DataFrame(
            schema={
                **{k: df.schema.get(k, pl.Int64) for k in group_keys},
                "target_cap": pl.Float64,
                "carry_cap": pl.Float64,
            }
        )

    rows: list[dict] = []
    for key_vals in df.select(group_keys).unique().iter_rows(named=True):
        mask = pl.lit(True)
        for k, v in key_vals.items():
            mask = mask & (pl.col(k) == v)
        team_df = df.filter(mask)
        recv = team_df.filter(pl.col("position").is_in(["WR", "TE", "RB"]))
        rbs = team_df.filter(pl.col("position") == "RB")
        t_present, t_ghost = ghost_slot_weights(
            recv, _EXPECTED_TARGET_SLOTS, table="target"
        )
        c_present, c_ghost = ghost_slot_weights(
            rbs, _EXPECTED_CARRY_SLOTS, table="carry"
        )
        t_denom = t_present + t_ghost
        c_denom = c_present + c_ghost
        t_cap = base_target_cap if t_denom <= 1e-9 else base_target_cap * (t_present / t_denom)
        c_cap = base_carry_cap if c_denom <= 1e-9 else base_carry_cap * (c_present / c_denom)
        rows.append({**key_vals, "target_cap": float(t_cap), "carry_cap": float(c_cap)})

    return pl.DataFrame(rows)


def cap_position_shares(
    df: pl.DataFrame,
    share_col: str,
    *,
    max_by_position: dict[str, float] | None = None,
    max_share: float | None = None,
) -> pl.DataFrame:
    """Clip individual shares; excess is left unallocated (reserve)."""
    if share_col not in df.columns or df.is_empty():
        return df
    if max_by_position and "position" in df.columns:
        capped = pl.col(share_col)
        for pos, mx in max_by_position.items():
            capped = (
                pl.when(pl.col("position") == pos)
                .then(pl.col(share_col).clip(0.0, mx))
                .otherwise(capped)
            )
        return df.with_columns(capped.alias(share_col))
    if max_share is not None:
        return df.with_columns(pl.col(share_col).clip(0.0, max_share).alias(share_col))
    return df


def concentrate_qb_dropbacks(
    qb: pl.DataFrame,
    *,
    group_keys: list[str],
    starter_share: float = STARTER_DROPBACK_SHARE,
) -> pl.DataFrame:
    """Assign nearly all dropbacks to the top QB unless a true committee is predicted.

    Ranking prefers real (non-null) depth_rank within a group. When depth is
    missing for everyone, rank by pred_dropback_share then a stable gsis_id
    tiebreak — never by input row order (which can leak same-week fantasy points).
    Unavailable QBs (Out / play_prob≈0) are excluded from starter selection.
    """
    if qb.is_empty():
        return qb

    ranked = qb.with_columns(
        pl.col("pred_dropback_share")
        .fill_null(0.0)
        .clip(0.0, 1.0)
        .alias("pred_dropback_share")
    )

    if "play_prob" in ranked.columns:
        avail = pl.col("play_prob").cast(pl.Float64).fill_null(1.0)
    else:
        avail = pl.lit(1.0)
    if "is_out" in ranked.columns:
        avail = pl.when(pl.col("is_out").fill_null(False)).then(pl.lit(0.0)).otherwise(avail)
    ranked = ranked.with_columns(avail.alias("_qb_available"))

    has_depth = "depth_rank" in ranked.columns
    # Per-group: use depth only when at least one available QB has a real depth_rank
    if has_depth:
        depth_cov = ranked.group_by(group_keys).agg(
            (
                (pl.col("depth_rank").is_not_null() & (pl.col("_qb_available") > 1e-6)).any()
            ).alias("_use_depth")
        )
        ranked = ranked.join(depth_cov, on=group_keys, how="left")
    else:
        ranked = ranked.with_columns(pl.lit(False).alias("_use_depth"))

    # Sort key: unavailable QBs sink; then depth (asc) or share (desc); then gsis_id
    if "gsis_id" in ranked.columns:
        id_key = pl.col("gsis_id").cast(pl.Utf8).fill_null("")
    else:
        id_key = pl.lit("")

    unavail = (
        pl.when(pl.col("_qb_available") <= 1e-6)
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("_unavail")
    )
    if has_depth:
        depth_key = (
            pl.when(pl.col("_use_depth"))
            .then(pl.col("depth_rank").fill_null(99.0))
            .otherwise(pl.lit(0.0))
            .alias("_depth_key")
        )
    else:
        depth_key = pl.lit(0.0).alias("_depth_key")
    share_key = (
        pl.when(pl.col("_use_depth"))
        .then(pl.lit(0.0))
        .otherwise(-pl.col("pred_dropback_share"))
        .alias("_share_key")
    )
    ranked = ranked.with_columns(
        [unavail, depth_key, share_key, id_key.alias("_id_key")]
    )
    # ordinal rank within group after stable sort
    ranked = ranked.sort(
        group_keys + ["_unavail", "_depth_key", "_share_key", "_id_key"]
    ).with_columns(
        pl.int_range(pl.len()).over(group_keys).add(1).alias("_db_rank")
    )

    top2 = ranked.group_by(group_keys).agg(
        [
            pl.col("pred_dropback_share").filter(pl.col("_db_rank") == 1).first().alias("_top"),
            pl.col("pred_dropback_share").filter(pl.col("_db_rank") == 2).first().alias("_second"),
            (pl.col("_qb_available") > 1e-6).sum().alias("_n_qb"),
        ]
    )
    ranked = ranked.join(top2, on=group_keys, how="left")

    is_committee = (
        (pl.col("_n_qb") >= 2)
        & (pl.col("_second").fill_null(0.0) >= COMMITTEE_SECOND_MIN)
        & ((pl.col("_top") - pl.col("_second").fill_null(0.0)) < COMMITTEE_SHARE_GAP)
    )

    ranked = ranked.with_columns(
        pl.when(pl.col("_qb_available") <= 1e-6)
        .then(pl.lit(0.0))
        .when(is_committee)
        .then(pl.col("pred_dropback_share"))
        .when(pl.col("_db_rank") == 1)
        .then(pl.lit(starter_share))
        .otherwise(
            pl.when(pl.col("_n_qb") <= 1)
            .then(pl.lit(0.0))
            .otherwise((1.0 - starter_share) / (pl.col("_n_qb") - 1).cast(pl.Float64))
        )
        .alias("pred_dropback_share")
    )
    drop_cols = [
        c
        for c in (
            "_db_rank",
            "_top",
            "_second",
            "_n_qb",
            "_qb_available",
            "_use_depth",
            "_unavail",
            "_depth_key",
            "_share_key",
            "_id_key",
        )
        if c in ranked.columns
    ]
    ranked = ranked.drop(drop_cols)

    return normalize_shares(ranked, "pred_dropback_share", group_keys=group_keys, target_sum=1.0)


def apply_accounting(
    players: pl.DataFrame,
    team_totals: pl.DataFrame,
    *,
    throwaway_reserve: float = THROWAY_RESERVE,
    other_reserve: float = OTHER_RECEIVER_RESERVE,
) -> pl.DataFrame:
    """Convert predicted shares + efficiency + team totals into box-score stats.

    Expected columns on players:
      - season, week, team, position, gsis_id
      - pred_target_share / pred_carry_share / pred_dropback_share / ...
      - pred_ypa, pred_ypc, pred_ypr, pred_catch_rate, pred_*_td_rate

    Expected columns on team_totals:
      - season, week, team
      - pred_team_pass_attempts, pred_team_rush_attempts, pred_team_pass_tds, pred_team_rush_tds
    """
    keys = ["season", "week", "team"]
    df = players.join(team_totals, on=keys, how="left")

    # Strip panel actuals so projected box scores cannot leak current-week stats
    drop_actuals = [c for c in _BOX_SCORE_COLS if c in df.columns]
    if drop_actuals:
        df = df.drop(drop_actuals)

    # Fill missing share predictions with zeros
    for col in (
        "pred_dropback_share",
        "pred_carry_share",
        "pred_target_share",
        "pred_air_yards_share",
        "pred_snap_share",
        "pred_redzone_target_share",
    ):
        if col not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias(col))
        else:
            df = df.with_columns(pl.col(col).fill_null(0.0).alias(col))

    # Depth chart prior before normalize — stops RB3/WR5 from vacuuming volume
    # when true starters are missing or model shares are flat.
    df = apply_depth_share_priors(df)
    # Listed WR2 / TE1 floors — stops Flowers-style WR1 vacuum and Bowers compression
    df = apply_receiver_share_floors(df, group_keys=keys)

    base_target_cap = max(0.0, 1.0 - throwaway_reserve - other_reserve)
    caps = team_volume_caps_from_ghosts(
        df, group_keys=keys, base_target_cap=base_target_cap, base_carry_cap=0.95
    )
    df = df.join(caps, on=keys, how="left")
    df = df.with_columns(
        [
            pl.col("target_cap").fill_null(base_target_cap),
            pl.col("carry_cap").fill_null(0.95),
        ]
    )

    # Normalize QB dropbacks — concentrate on starter unless committee
    qb = df.filter(pl.col("position") == "QB")
    non_qb = df.filter(pl.col("position") != "QB")
    if not qb.is_empty():
        qb = concentrate_qb_dropbacks(qb, group_keys=keys)
        # Cap QB rush share before carving RB room (no double-count of team rushes)
        qb = qb.with_columns(
            pl.col("pred_carry_share").clip(0.0, MAX_QB_CARRY_SHARE).alias("pred_carry_share")
        )
        # Zero residual backup QB carries — they were diluting dual-threat
        # starters without a corresponding dropback role.
        if "pred_dropback_share" in qb.columns:
            qb = qb.with_columns(
                pl.when(pl.col("pred_dropback_share").fill_null(0.0) >= 0.50)
                .then(pl.col("pred_carry_share"))
                .otherwise(pl.lit(0.0))
                .alias("pred_carry_share")
            )

    # Normalize RB carry shares into remaining rush budget after QB shares
    rb = non_qb.filter(pl.col("position") == "RB")
    wrte = non_qb.filter(pl.col("position").is_in(["WR", "TE"]))
    if not rb.is_empty():
        qb_carry = (
            qb.group_by(keys).agg(pl.col("pred_carry_share").sum().alias("_qb_carry_sum"))
            if not qb.is_empty()
            else None
        )
        if qb_carry is not None:
            rb = rb.join(qb_carry, on=keys, how="left").with_columns(
                pl.col("_qb_carry_sum").fill_null(0.0)
            )
        else:
            rb = rb.with_columns(pl.lit(0.0).alias("_qb_carry_sum"))

        rb_frames: list[pl.DataFrame] = []
        for key_vals in rb.select(keys).unique().iter_rows(named=True):
            mask = pl.lit(True)
            for k, v in key_vals.items():
                mask = mask & (pl.col(k) == v)
            chunk = rb.filter(mask)
            team_cap = float(chunk["carry_cap"][0]) if "carry_cap" in chunk.columns else 0.95
            qb_sum = float(chunk["_qb_carry_sum"][0]) if chunk.height else 0.0
            rb_cap = max(0.0, team_cap - qb_sum)
            chunk = normalize_shares(chunk, "pred_carry_share", group_keys=keys, target_sum=rb_cap)
            rb_frames.append(chunk)
        rb = pl.concat(rb_frames, how="diagonal_relaxed") if rb_frames else rb
        rb = cap_position_shares(rb, "pred_carry_share", max_share=MAX_CARRY_SHARE)
        if "_qb_carry_sum" in rb.columns:
            rb = rb.drop("_qb_carry_sum")

    # Targets: leave reserve for throwaways + other + ghost depth slots
    recv = pl.concat([rb, wrte], how="diagonal_relaxed") if not rb.is_empty() or not wrte.is_empty() else wrte
    if not recv.is_empty() and "pred_target_share" in recv.columns:
        recv_frames: list[pl.DataFrame] = []
        for key_vals in recv.select(keys).unique().iter_rows(named=True):
            mask = pl.lit(True)
            for k, v in key_vals.items():
                mask = mask & (pl.col(k) == v)
            chunk = recv.filter(mask)
            cap = float(chunk["target_cap"][0]) if "target_cap" in chunk.columns else base_target_cap
            chunk = normalize_shares(chunk, "pred_target_share", group_keys=keys, target_sum=cap)
            recv_frames.append(chunk)
        recv = pl.concat(recv_frames, how="diagonal_relaxed") if recv_frames else recv
        recv = cap_position_shares(recv, "pred_target_share", max_by_position=MAX_TARGET_SHARE)

    df = pl.concat([qb, recv], how="diagonal_relaxed") if not qb.is_empty() else recv

    # Default efficiency if missing
    eff_defaults = {
        "pred_ypa": 7.0,
        "pred_ypc": 4.2,
        "pred_ypr": 11.0,
        "pred_catch_rate": 0.65,
        "pred_completion_rate": DEFAULT_COMPLETION_RATE,
        "pred_pass_td_rate": 0.04,
        "pred_int_rate": 0.025,
        "pred_rush_td_rate": 0.03,
        "pred_rec_td_rate": 0.05,
    }
    for col, default in eff_defaults.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(default).alias(col))
        else:
            df = df.with_columns(pl.col(col).fill_null(default).alias(col))

    # Team totals defaults
    for col, default in (
        ("pred_team_pass_attempts", 35.0),
        ("pred_team_rush_attempts", 26.0),
        ("pred_team_pass_tds", 1.5),
        ("pred_team_rush_tds", 0.9),
    ):
        if col not in df.columns:
            df = df.with_columns(pl.lit(default).alias(col))
        else:
            df = df.with_columns(pl.col(col).fill_null(default).alias(col))

    # Volume from shares
    df = df.with_columns(
        [
            pl.when(pl.col("position") == "QB")
            .then(pl.col("pred_dropback_share") * pl.col("pred_team_pass_attempts"))
            .otherwise(0.0)
            .alias("proj_attempts"),
            pl.when(pl.col("position").is_in(["RB", "QB"]))
            .then(pl.col("pred_carry_share").fill_null(0.0) * pl.col("pred_team_rush_attempts"))
            .otherwise(0.0)
            .alias("proj_carries"),
            pl.when(pl.col("position").is_in(["RB", "WR", "TE"]))
            .then(pl.col("pred_target_share") * pl.col("pred_team_pass_attempts"))
            .otherwise(0.0)
            .alias("proj_targets"),
        ]
    )

    # Clip efficiency before converting to counting stats
    ypr_clip = pl.col("pred_ypr")
    catch_clip = pl.col("pred_catch_rate")
    if "position" in df.columns:
        for pos, mx in MAX_YPR_BY_POS.items():
            ypr_clip = (
                pl.when(pl.col("position") == pos)
                .then(pl.col("pred_ypr").clip(3.0, mx))
                .otherwise(ypr_clip)
            )
        for pos, mx in MAX_CATCH_RATE_BY_POS.items():
            catch_clip = (
                pl.when(pl.col("position") == pos)
                .then(pl.col("pred_catch_rate").clip(0.3, mx))
                .otherwise(catch_clip)
            )
        ypr_clip = ypr_clip.clip(3.0, MAX_YPR)
        catch_clip = catch_clip.clip(0.3, MAX_CATCH_RATE)
    else:
        ypr_clip = pl.col("pred_ypr").clip(3.0, MAX_YPR)
        catch_clip = pl.col("pred_catch_rate").clip(0.3, MAX_CATCH_RATE)

    df = df.with_columns(
        [
            pl.col("pred_ypa").clip(5.0, 10.0),
            pl.col("pred_ypc").clip(1.5, 7.0),
            ypr_clip.alias("pred_ypr"),
            catch_clip.alias("pred_catch_rate"),
            pl.col("pred_completion_rate").clip(0.45, 0.78),
            pl.col("pred_pass_td_rate").clip(0.0, 0.12),
            pl.col("pred_int_rate").clip(0.01, 0.06),
            # Allow dual-threat QB scramble TD rates up to ~0.20 before team rescale
            pl.col("pred_rush_td_rate").clip(0.0, 0.20),
            pl.col("pred_rec_td_rate").clip(0.0, 0.20),
            pl.col("proj_attempts").clip(0.0, 60.0),
            pl.col("proj_carries").clip(0.0, 35.0),
            pl.col("proj_targets").clip(0.0, 20.0),
        ]
    )

    # Efficiency -> yards / TDs / receptions / completions
    df = df.with_columns(
        [
            (pl.col("proj_attempts") * pl.col("pred_ypa")).clip(0, 500).alias("passing_yards"),
            (pl.col("proj_attempts") * pl.col("pred_pass_td_rate")).clip(0, 6).alias("passing_tds"),
            (pl.col("proj_attempts") * pl.col("pred_int_rate")).clip(0, 4).alias("interceptions"),
            pl.min_horizontal(
                pl.col("proj_attempts"),
                (pl.col("proj_attempts") * pl.col("pred_completion_rate")).clip(0, 60),
            ).alias("completions"),
            (pl.col("proj_carries") * pl.col("pred_ypc")).clip(0, 200).alias("rushing_yards"),
            (pl.col("proj_carries") * pl.col("pred_rush_td_rate")).clip(0, 4).alias("rushing_tds"),
            (pl.col("proj_targets") * pl.col("pred_catch_rate")).clip(0, 15).alias("receptions"),
            (
                pl.col("proj_targets") * pl.col("pred_catch_rate") * pl.col("pred_ypr")
            ).clip(0, 250).alias("receiving_yards"),
            (pl.col("proj_targets") * pl.col("pred_rec_td_rate")).clip(0, 4).alias("receiving_tds"),
            pl.lit(0.0).alias("fumbles_lost"),
            pl.col("proj_attempts").alias("attempts"),
            pl.col("proj_carries").alias("carries"),
            pl.col("proj_targets").alias("targets"),
        ]
    )

    # Enforce completions <= attempts (belt-and-suspenders)
    df = df.with_columns(
        pl.min_horizontal(pl.col("completions"), pl.col("attempts")).alias("completions")
    )

    # Rescale receiving yards to match team pass yards (approx = sum of receiver yards)
    # Team pass yards ≈ pass attempts * average YPA from QB predictions
    qb_ypa = (
        df.filter(pl.col("position") == "QB")
        .group_by(keys)
        .agg(
            (
                (pl.col("pred_ypa") * pl.col("pred_dropback_share")).sum()
                / pl.col("pred_dropback_share").sum().clip(1e-6, None)
            ).alias("_team_ypa")
        )
    )
    df = df.join(qb_ypa, on=keys, how="left")
    # Shrink team pass yards by the same ghost factor used for target_cap so
    # missing WR2s (Diggs/Deebo) don't get their yards reassigned to WR1.
    df = df.with_columns(
        (
            pl.col("pred_team_pass_attempts")
            * pl.col("_team_ypa").fill_null(7.0).clip(5.0, 10.0)
            * (
                pl.col("target_cap").fill_null(base_target_cap)
                / max(base_target_cap, 1e-9)
            ).clip(0.0, 1.0)
        ).alias("_team_pass_yards")
    )

    recv_yards = df.group_by(keys).agg(
        pl.col("receiving_yards").sum().alias("_sum_rec_yards")
    )
    df = df.join(recv_yards, on=keys, how="left")
    df = df.with_columns(
        pl.when(pl.col("position").is_in(["RB", "WR", "TE"]) & (pl.col("_sum_rec_yards") > 1.0))
        .then(
            pl.col("receiving_yards")
            * pl.col("_team_pass_yards").clip(50, 500)
            / pl.col("_sum_rec_yards")
        )
        .otherwise(pl.col("receiving_yards"))
        .clip(0, 250)
        .alias("receiving_yards")
    )

    # Rescale receptions toward team completions ≈ QB attempts * completion%
    # Shrink by ghost factor so missing WR2 targets aren't reassigned via catches.
    team_comp = df.group_by(keys).agg(
        [
            pl.col("completions").filter(pl.col("position") == "QB").sum().alias("_qb_comp"),
            pl.col("receptions").sum().alias("_sum_rec"),
        ]
    )
    df = df.join(team_comp, on=keys, how="left")
    ghost_recv_scale = (
        pl.col("target_cap").fill_null(base_target_cap) / max(base_target_cap, 1e-9)
    ).clip(0.0, 1.0)
    df = df.with_columns(
        pl.when(
            pl.col("position").is_in(["RB", "WR", "TE"])
            & (pl.col("_sum_rec") > 0.5)
            & (pl.col("_qb_comp") > 0)
        )
        .then(
            pl.col("receptions")
            * pl.col("_qb_comp").clip(0, 45)
            * ghost_recv_scale
            / pl.col("_sum_rec")
        )
        .otherwise(pl.col("receptions"))
        .clip(0, 15)
        .alias("receptions")
    )

    # Rescale TDs to team totals
    pass_td_sum = df.group_by(keys).agg(pl.col("passing_tds").sum().alias("_sum_pass_tds"))
    df = df.join(pass_td_sum, on=keys, how="left")
    df = df.with_columns(
        pl.when((pl.col("position") == "QB") & (pl.col("_sum_pass_tds") > 1e-6))
        .then(pl.col("passing_tds") * pl.col("pred_team_pass_tds") / pl.col("_sum_pass_tds"))
        .otherwise(pl.col("passing_tds"))
        .alias("passing_tds")
    )

    rush_td_sum = df.group_by(keys).agg(pl.col("rushing_tds").sum().alias("_sum_rush_tds"))
    df = df.join(rush_td_sum, on=keys, how="left")
    df = df.with_columns(
        pl.when((pl.col("position").is_in(["QB", "RB"])) & (pl.col("_sum_rush_tds") > 1e-6))
        .then(pl.col("rushing_tds") * pl.col("pred_team_rush_tds") / pl.col("_sum_rush_tds"))
        .otherwise(pl.col("rushing_tds"))
        .alias("rushing_tds")
    )

    # Receiving TDs are the same events as team pass TDs — rescale (ghost-shrunk)
    rec_td_sum = df.group_by(keys).agg(pl.col("receiving_tds").sum().alias("_sum_rec_tds"))
    df = df.join(rec_td_sum, on=keys, how="left")
    ghost_td_scale = (
        pl.col("target_cap").fill_null(base_target_cap) / max(base_target_cap, 1e-9)
    ).clip(0.0, 1.0)
    df = df.with_columns(
        pl.when(
            pl.col("position").is_in(["RB", "WR", "TE"])
            & (pl.col("_sum_rec_tds") > 1e-6)
        )
        .then(
            pl.col("receiving_tds")
            * pl.col("pred_team_pass_tds")
            * ghost_td_scale
            / pl.col("_sum_rec_tds")
        )
        .otherwise(pl.col("receiving_tds"))
        .alias("receiving_tds")
    )

    drop_caps = [c for c in ("target_cap", "carry_cap") if c in df.columns]
    if drop_caps:
        df = df.drop(drop_caps)

    # Drop helper columns
    helpers = [c for c in df.columns if c.startswith("_")]
    if helpers:
        df = df.drop(helpers)

    return df


def assert_shares_sum(
    df: pl.DataFrame,
    share_col: str,
    *,
    group_keys: list[str] | None = None,
    expected: float = 1.0,
    tol: float = 0.05,
) -> None:
    group_keys = group_keys or ["season", "week", "team"]
    totals = df.group_by(group_keys).agg(pl.col(share_col).sum().alias("s"))
    bad = totals.filter((pl.col("s") - expected).abs() > tol)
    if bad.height:
        raise AssertionError(
            f"{share_col} sums deviate from {expected} by >{tol} in {bad.height} groups"
        )
