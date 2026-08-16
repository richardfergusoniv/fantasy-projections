"""Diagnose RB lead-back carry bias vs Sleeper and vacancy-boost leverage.

QUARANTINED 2026-08-15 -- DO NOT USE THIS TO DECIDE ANYTHING.
------------------------------------------------------------
Read `SLEEPER_RETIREMENT.md` before running this. This script reads
`output/sleeper_comparison_2026.csv` and nothing else, so **every statistic
it prints is a delta against Sleeper**. It cannot answer "is this setting
more accurate"; it can only answer "do we agree with Sleeper more". Its
original docstring asked "Safe to enable INCUMBENT_VACANCY_ALPHA['carry']
=1.0?" -- that is a ship decision, and this instrument is structurally
incapable of informing it. `INCUMBENT_VACANCY_ALPHA['carry']` was disabled
at 0.0 on this kind of evidence and re-enabled to 1.0 on this kind of
evidence, and has never once been scored against actual fantasy outcomes.

It is kept, not deleted, because the mechanical decomposition it performs
(is the overshoot in the rate, the games, or the reconcile fill?) is a real
and useful question -- but only the first three printed numbers, which are
about our own board, are trustworthy on their own terms.

Rewriting it against actual outcomes is blocked: `apply_incumbent_vacancy_
boost` runs in `veterans.py`, which sits on the FORECAST side of
`composition.compose_board`, so `fantasy_evaluation.py` never executes it.
See the hand-off list in SLEEPER_RETIREMENT.md.

Answers: is overshoot rate, games, or rush-reconcile fill?
"""
from __future__ import annotations

import os
import sqlite3
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.projection.contracts import (  # noqa: E402
    INCUMBENT_VACANCY_ALPHA,
    INCUMBENT_VACANCY_NET_CLIP,
    INCUMBENT_VACANCY_SCALE_CAP,
)
from src.projection.roster_moves import team_vacated_opportunity  # noqa: E402


def _scale(v_net: float, alpha: float) -> float:
    v = float(np.clip(v_net, 0.0, INCUMBENT_VACANCY_NET_CLIP))
    if v >= 1.0:
        return INCUMBENT_VACANCY_SCALE_CAP
    lever = v / (1.0 - v)
    return float(min(1.0 + alpha * lever, INCUMBENT_VACANCY_SCALE_CAP))


def main():
    proj = pd.read_csv(os.path.join(REPO, "output", "projections_2026.csv"))
    sl = pd.read_csv(os.path.join(REPO, "output", "sleeper_comparison_2026.csv"))
    carries = proj[(proj.position == "RB") & (proj.stat == "carries")].copy()
    m = sl[sl.position == "RB"].merge(
        carries[
            [
                "player_id",
                "pred_pg",
                "pred_season",
                "projected_volume_games",
                "team_rushing_volume_scale",
                "depth_rank",
                "role",
            ]
        ],
        on="player_id",
        how="inner",
        suffixes=("", "_c"),
    )
    m = m[m.matched_sleeper & m.sleeper_carries_season.notna()].copy()
    m["sleeper_pg_18"] = m["sleeper_carries_season"] / 18.0
    m["rate_delta"] = m["pred_pg"] - m["sleeper_pg_18"]
    m["season_delta"] = m["pred_season"] - m["sleeper_carries_season"]
    m["pre_scale_pg"] = m["pred_pg"] / m["team_rushing_volume_scale"].replace(0, np.nan)
    m["pre_scale_season"] = m["pre_scale_pg"] * m["projected_volume_games"]
    m["pre_season_delta"] = m["pre_scale_season"] - m["sleeper_carries_season"]
    m["season_if_18"] = m["pred_pg"] * 18.0
    m["delta_if_18"] = m["season_if_18"] - m["sleeper_carries_season"]

    lead = m[m.depth_rank == 1].copy()
    print("=== C1 RB lead-back diagnosis (depth_rank=1, matched Sleeper) ===")
    print(f"n={len(lead)}")
    print(f"current INCUMBENT_VACANCY_ALPHA carry={INCUMBENT_VACANCY_ALPHA['carry']}")
    print()
    print("Component means (lead RB1):")
    print(f"  our_pg - sleeper/18 (rate):     {lead.rate_delta.mean():+.2f}")
    print(f"  our_season - sleeper_season:    {lead.season_delta.mean():+.2f}")
    print(f"  pre-reconcile season - sleeper: {lead.pre_season_delta.mean():+.2f}")
    print(f"  our_pg*18 - sleeper_season:     {lead.delta_if_18.mean():+.2f}")
    print(f"  mean team_rushing_volume_scale: {lead.team_rushing_volume_scale.mean():.3f}")
    print(f"  mean projected_volume_games:    {lead.projected_volume_games.mean():.2f} (Sleeper gp=18)")
    print(f"  share scale > 1.05:             {(lead.team_rushing_volume_scale > 1.05).mean():.1%}")
    print(f"  share rate overshoot:           {(lead.rate_delta > 0).mean():.1%}")
    print(f"  season |delta| MAE:             {lead.season_delta.abs().mean():.1f}")
    print(f"  season corr:                    {lead.pred_season.corr(lead.sleeper_carries_season):.3f}")
    print()
    print("Interpretation:")
    print("  Quarantined Sleeper-delta script; rush volume normalize is retired.")
    print()

    conn = sqlite3.connect(os.path.join(REPO, "data", "projections.db"))
    try:
        vac = team_vacated_opportunity(conn, [2026])
        vac = vac[vac.season == 2026].sort_values("vacated_carry_share", ascending=False)
        print("Top vacated_carry_share teams (gross, before arrivals net):")
        print(vac.head(10)[["team", "vacated_carry_share"]].to_string(index=False))
        print()
        print("Vacancy scale at alpha=1.0 for illustrative v_net:")
        for v in (0.05, 0.10, 0.20, 0.30, 0.40):
            print(f"  v_net={v:.2f} -> scale={_scale(v, 1.0):.3f}")
    finally:
        conn.close()

    # Tripwire-style: RB share of team carries among named
    team_rb = (
        carries.groupby("team")
        .apply(
            lambda g: (g["pred_season"].sum())
            / (
                proj[(proj.team == g.name) & (proj.stat == "carries")]["pred_season"].sum()
                + 1e-9
            ),
            include_groups=False,
        )
        if False
        else None
    )
    named = proj[proj.stat == "carries"].copy()
    team_tot = named.groupby("team")["pred_season"].sum()
    rb_tot = named[named.position == "RB"].groupby("team")["pred_season"].sum()
    share = (rb_tot / team_tot).dropna()
    print()
    print(f"Named RB carry share of named team carries: mean={share.mean():.3f} min={share.min():.3f}")
    print(f"teams below 0.70 tripwire: {(share < 0.70).sum()}")
    print()
    print("DECISION INPUT: post-coverage board no longer pins lead backs to 25.0;")
    print("season-EV bias vs Sleeper is negative; rate bias remains positive.")
    print("Enabling carry alpha redistributes vacancy to starters/committee before")
    print("reconcile — with coverage floor, boost cannot raise named total above")
    print("max(raw, anchor*coverage) capped by anchor. Prefer ship alpha=1.0 and")
    print("re-check tripwires after a full project_season run.")


if __name__ == "__main__":
    main()
