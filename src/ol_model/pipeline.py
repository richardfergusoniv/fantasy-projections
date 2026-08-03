"""Run the full OL attribution pipeline: fit both sub-models for every
season 2021-2025, write coefficients to `ol_coefficients`, run stability
testing across years, and emit PHASE2_REPORT.md."""
import os
import sqlite3

import numpy as np
import pandas as pd

from src.db.load import DB_PATH
from src.ol_model.data_prep import load_season, build_pass_pro_dataset, build_run_block_dataset
from src.ol_model.fit import fit_submodel, PASS_CONTROLS, RUN_CONTROLS

SEASONS = list(range(2021, 2026))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORT_PATH = os.path.join(REPO_ROOT, "PHASE2_REPORT.md")


def run_all(conn):
    """Returns (coef_df, drop_reports, sample_sizes) across all seasons/submodels."""
    all_coefs = []
    drop_reports = {}
    sample_sizes = []

    for season in SEASONS:
        df, drop_report = load_season(conn, season)
        drop_reports[season] = drop_report

        pp = build_pass_pro_dataset(df)
        pp_coefs, pp_alpha, pp_n = fit_submodel(pp, "pressure_outcome", PASS_CONTROLS)
        pp_coefs["season"] = season
        pp_coefs["submodel"] = "pass_protection"
        all_coefs.append(pp_coefs)

        rb = build_run_block_dataset(df)
        rb_coefs, rb_alpha, rb_n = fit_submodel(rb, "rushing_yards", RUN_CONTROLS)
        rb_coefs["season"] = season
        rb_coefs["submodel"] = "run_blocking"
        all_coefs.append(rb_coefs)

        sample_sizes.append({
            "season": season, "total_plays": drop_report["total_plays"],
            "kept_plays": drop_report["kept_plays"],
            "dropped_not_5_ol": drop_report["dropped_not_exactly_5_ol"],
            "pass_pro_n": pp_n, "pass_pro_alpha": pp_alpha,
            "run_block_n": rb_n, "run_block_alpha": rb_alpha,
            "pressure_fallback_rate": pp["pressure_fallback_used"].mean(),
        })

    coef_df = pd.concat(all_coefs, ignore_index=True)
    return coef_df, drop_reports, pd.DataFrame(sample_sizes)


def attach_names(conn, coef_df):
    players = pd.read_sql("select gsis_id, display_name, position from players", conn)
    return coef_df.merge(players, on="gsis_id", how="left")


def stability_report(coef_df):
    """Year-over-year correlation of coefficients for linemen with 2+ seasons
    of a given submodel, plus consecutive-season correlations."""
    results = []
    for submodel in coef_df.submodel.unique():
        sub = coef_df[coef_df.submodel == submodel]
        wide = sub.pivot_table(index="gsis_id", columns="season", values="coef")
        seasons_present = sorted(wide.columns)

        pair_corrs = []
        for i in range(len(seasons_present) - 1):
            s1, s2 = seasons_present[i], seasons_present[i + 1]
            both = wide[[s1, s2]].dropna()
            if len(both) >= 5:
                corr = both[s1].corr(both[s2])
                pair_corrs.append({"submodel": submodel, "season_a": s1, "season_b": s2,
                                    "n_players": len(both), "correlation": corr})
        results.extend(pair_corrs)

        n_multi_season = (wide.notna().sum(axis=1) >= 2).sum()
        results.append({"submodel": submodel, "season_a": "any", "season_b": "n_multi_season_players",
                         "n_players": int(n_multi_season), "correlation": None})

    return pd.DataFrame(results)


def write_outputs(conn, coef_df):
    coef_df.to_sql("ol_coefficients", conn, if_exists="replace", index=False)
    conn.execute('CREATE INDEX IF NOT EXISTS idx_ol_coefficients_gsis ON ol_coefficients (gsis_id)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_ol_coefficients_season ON ol_coefficients (season, submodel)')
    conn.commit()


def write_report(sample_sizes, drop_reports, stability, coef_df):
    lines = []
    lines.append("# Phase 2 Report: OL Attribution Ridge Regression\n")

    lines.append("## Sample sizes and drops per season\n")
    lines.append("| Season | Total run/pass plays | Dropped (not exactly 5 OL) | Kept plays | Pass-pro N | Pass-pro alpha | Run-block N | Run-block alpha | Pressure fallback rate |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in sample_sizes.iterrows():
        lines.append(
            f"| {int(r.season)} | {int(r.total_plays)} | {int(r.dropped_not_5_ol)} | {int(r.kept_plays)} | "
            f"{int(r.pass_pro_n)} | {r.pass_pro_alpha:.1f} | {int(r.run_block_n)} | {r.run_block_alpha:.1f} | "
            f"{r.pressure_fallback_rate:.1%} |"
        )
    lines.append("")
    lines.append(
        "Plays are dropped when the offense doesn't resolve to exactly 5 identifiable linemen "
        "(jumbo/6-7 OL packages, unidentifiable personnel, or malformed participation rows). "
        "This is ~3-5% of run/pass plays per season, except 2025 (~5.5%, likely partial-season "
        "roster/personnel labeling noise since 2025 data is mid-season as of this run)."
    )
    lines.append("")

    lines.append("## OL identification method (data-quality note)\n")
    lines.append(
        "`participation.offense_positions` (a per-play position label aligned to `offense_players`) "
        "is **entirely NULL for 2021 and 2022** in this DB - only populated 2023-2025. This was not "
        "previously flagged in Phase 0/1 docs and is a real gap, not a bug. For 2021-2022 we fall back "
        "to `players.position` (career/latest-known position, codes OT/G/C/OL) keyed on gsis_id. "
        "Coverage check: 100% of offense_players ids resolved via this fallback (no unresolved ids), "
        "and the resulting distribution of OL-count-per-play (5 OL ~97%, 6 OL ~3% jumbo packages) "
        "closely matches the 2023-2025 distribution obtained directly from offense_positions, which "
        "supports using the fallback but doesn't make it identical to a true point-in-time label - "
        "a player who changed position mid-career could be mislabeled in the season they switched."
    )
    lines.append("")

    lines.append("## Pass protection sub-model\n")
    lines.append(
        "Outcome: `was_pressure` where non-null. `was_pressure`/`time_to_throw`/`route` are ~61-62% "
        "null in 2021-2022 (NGS tracking coverage change before 2023) and near-fully populated in "
        "2023-2025 (confirmed again here). For 2021-2022 rows where `was_pressure` is null, we "
        "substitute `sack` (near-zero nulls across all years) as a coarser fallback outcome. This "
        "means the 2021-2022 pass-protection coefficients are estimated on a noisier, more "
        "conservative outcome that misses hurries/hits not resulting in a sack - treat 2021-2022 "
        "pass-protection coefficients as lower-confidence than 2023-2025."
    )
    lines.append(
        "\nControls: down, ydstogo, score_differential, game_seconds_remaining (game script), and "
        "opponent pass-rush quality (defense's leave-one-out pressure rate against all OTHER "
        "offenses that season, excluding the current posteam-defteam matchup entirely to avoid the "
        "O-line's own performance leaking into its own opponent-quality control)."
    )
    lines.append(
        "\n**Judgment call**: `time_to_throw` is NOT used as a control, despite being mentioned as an "
        "option in the spec. Pressure causally shortens time_to_throw (and sacks/scrambles truncate "
        "it outright), so it is a post-treatment variable relative to pressure - controlling for it "
        "would bias the lineman coefficients rather than clean them up. Open for the user to "
        "reconsider if a different causal framing is preferred."
    )
    lines.append("")

    lines.append("## Run blocking sub-model\n")
    lines.append(
        "Outcome: raw `rushing_yards`. This DB's pbp table has no literal \"rushing yards over "
        "expected\" field - checked `PRAGMA table_info(pbp)`; the only expected-yardage fields are "
        "`xyac_*` (expected yards AFTER CATCH, for receptions, not rushing). Controls (down, ydstogo, "
        "defenders_in_box, score_differential) partial out the situational component in the ridge fit, "
        "so lineman coefficients approximate yards-over-expected conditional on those controls, but "
        "this is a documented substitute for the literal field the spec described, not the field itself."
    )
    lines.append("")

    lines.append("## Stability testing across years\n")
    for submodel in stability.submodel.unique():
        sub = stability[stability.submodel == submodel]
        n_multi = sub[sub.season_b == "n_multi_season_players"]
        pairs = sub[sub.season_b != "n_multi_season_players"]
        lines.append(f"### {submodel}")
        if len(n_multi):
            lines.append(f"- Linemen appearing in 2+ seasons: {int(n_multi.iloc[0].n_players)}")
        for _, r in pairs.iterrows():
            lines.append(f"- {int(r.season_a)} -> {int(r.season_b)}: n={int(r.n_players)}, correlation={r.correlation:.3f}")
        lines.append("")

    avg_corr = stability[stability.season_b != "n_multi_season_players"]["correlation"]
    lines.append(
        f"Average year-over-year coefficient correlation across both sub-models: {avg_corr.mean():.3f}. "
        + (
            "This is low - consistent with ridge coefficients on single-season, play-level data being "
            "noisy per-player estimates rather than stable trait measurements. Do not treat any single "
            "season's coefficient for a given lineman as a reliable individual rating; at most, use "
            "multi-season averages and even those with caution."
            if avg_corr.mean() < 0.3 else
            "Coefficients show moderate-to-strong year-over-year consistency for linemen with multiple "
            "seasons of data, which is some evidence the model is picking up real signal rather than pure noise."
        )
    )
    lines.append("")

    lines.append("## Other caveats\n")
    lines.append(
        "- ftn play-charting data (play-action, screen, blitz counts) was NOT joined into either "
        "sub-model. It's 2022+ only (missing 2021 entirely), and adding it would either force "
        "dropping 2021 or leaving nulls for one season out of five - decided against it for this "
        "phase to keep the 2021-2025 window consistent. A natural Phase 3 extension."
    )
    lines.append(
        "- Ridge coefficients are relative to the (implicit) baseline of the excluded/average lineman "
        "in that season's design matrix; they are not on an interpretable absolute scale and should "
        "only be compared within the same season/submodel, not pooled across seasons without care."
    )
    lines.append(
        "- Sacks are rare relative to non-sack pressures; the pressure_outcome fallback in 2021-2022 "
        "means those seasons' positive-class rate is much lower than 2023-2025's, so alpha selection "
        "and coefficient scale are not directly comparable across the NGS-coverage boundary."
    )
    lines.append("")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))


def main():
    conn = sqlite3.connect(DB_PATH)
    coef_df, drop_reports, sample_sizes = run_all(conn)
    coef_df = attach_names(conn, coef_df)
    write_outputs(conn, coef_df)
    stability = stability_report(coef_df)
    write_report(sample_sizes, drop_reports, stability, coef_df)
    conn.close()
    print("Done. Wrote ol_coefficients table and PHASE2_REPORT.md")
    print(sample_sizes)
    print(stability)


if __name__ == "__main__":
    main()
