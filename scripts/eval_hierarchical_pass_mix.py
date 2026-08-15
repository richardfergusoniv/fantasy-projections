"""Phase-4 spot check for hierarchical pass mix on the 2026 board."""
import os
import sys
import sqlite3

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.projection.data_prep import get_conn
from src.projection.predict import project_season
from src.projection.team_pass_mix import validate_mix_model


def main():
    conn = get_conn()
    try:
        summary = validate_mix_model(conn)
        print("MIX LOSO", {k: summary[k] for k in summary if k != "folds"})
        out = project_season(conn, 2026)
    finally:
        conn.close()

    names = pd.read_sql(
        "select gsis_id as player_id, display_name from players",
        sqlite3.connect("data/projections.db"),
    ).drop_duplicates("player_id")
    out = out.merge(names, on="player_id", how="left")

    jax = out[out["team"] == "JAX"]
    wrs = jax[(jax["position"] == "WR") & (jax["stat"] == "targets")].sort_values(
        "pred_pg", ascending=False
    )
    print("--- JAX WR targets ---")
    cols = [
        "display_name", "pred_pg", "depth_rank", "role", "source",
        "within_group_target_share", "hierarchical_pass_scale",
    ]
    print(wrs[cols].head(10).to_string(index=False))
    mix = {c: wrs[c].iloc[0] for c in [
        "wr_target_share", "te_target_share", "rb_target_share", "mix_source",
    ]}
    print("mix", mix)
    att = float(wrs["team_pass_attempts_pg_pred"].iloc[0])
    wr_s = float(wrs["wr_target_share"].iloc[0])
    print(
        "WR season", float((wrs["pred_pg"] * wrs["projected_volume_games"]).sum()),
        "budget", att * 17 * wr_s,
    )
    for name in [
        "Brian Thomas Jr.", "Travis Hunter", "Jakobi Meyers", "Parker Washington",
    ]:
        r = wrs[wrs["display_name"] == name]
        if r.empty:
            print(name, "MISSING")
        else:
            row = r.iloc[0]
            print(
                name,
                f"tgt={row.pred_pg:.2f}",
                f"season={row.pred_pg * row.projected_volume_games:.1f}",
                f"role={row.role}",
            )

    dal = out[(out["team"] == "DAL") & (out["position"] == "WR") & (out["stat"] == "targets")]
    dal = dal.sort_values("pred_pg", ascending=False)
    print("--- DAL WR ---")
    print(dal[["display_name", "pred_pg", "depth_rank", "role"]].head(4).to_string(index=False))

    out.to_csv("output/projections_2026_hierarchical.csv", index=False)
    print("wrote output/projections_2026_hierarchical.csv", len(out))


if __name__ == "__main__":
    main()
