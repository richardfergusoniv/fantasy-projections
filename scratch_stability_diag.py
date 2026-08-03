"""Ad hoc diagnostics for Phase 2 OL coefficient stability investigation.
Read-only: does not touch src/ol_model or the ol_coefficients table.
"""
import sqlite3
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.preprocessing import StandardScaler

from src.db.load import DB_PATH
from src.ol_model.data_prep import load_season, build_pass_pro_dataset, build_run_block_dataset
from src.ol_model.fit import PASS_CONTROLS, RUN_CONTROLS, ALPHAS

RNG = np.random.default_rng(42)


def design_matrix(df, controls):
    all_ol = sorted({pid for ids in df.ol_ids for pid in ids})
    col_idx = {pid: i for i, pid in enumerate(all_ol)}
    rows, cols = [], []
    for r, ids in enumerate(df.ol_ids):
        for pid in ids:
            rows.append(r)
            cols.append(col_idx[pid])
    indicator = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(df), len(all_ol)))
    control_vals = StandardScaler().fit_transform(df[controls].to_numpy(dtype=float))
    X = sparse.hstack([indicator, sparse.csr_matrix(control_vals)]).tocsr()
    return X, all_ol


def fit_ridgecv(df, outcome_col, controls):
    X, all_ol = design_matrix(df, controls)
    y = df[outcome_col].to_numpy(dtype=float)
    model = RidgeCV(alphas=ALPHAS, cv=5)
    model.fit(X, y)
    n_ol = len(all_ol)
    coefs = pd.Series(model.coef_[:n_ol], index=all_ol)
    return coefs, model.alpha_


def fit_fixed_alpha(df, outcome_col, controls, alpha):
    X, all_ol = design_matrix(df, controls)
    y = df[outcome_col].to_numpy(dtype=float)
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    n_ol = len(all_ol)
    coefs = pd.Series(model.coef_[:n_ol], index=all_ol)
    return coefs


print("=" * 80)
print("Loading 2023 data...")
conn = sqlite3.connect(DB_PATH)
df23, drop_report = load_season(conn, 2023)
pp23 = build_pass_pro_dataset(df23)
rb23 = build_run_block_dataset(df23)
print("pass_pro rows:", len(pp23), "run_block rows:", len(rb23))

# ---------------------------------------------------------------------------
# 1. Split-half reliability (split by GAME, not by play)
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("1. SPLIT-HALF RELIABILITY (2023)")
print("=" * 80)


def split_half_corr(df, outcome_col, controls, n_splits=5):
    games = df["nflverse_game_id"].unique()
    corrs = []
    ns = []
    for i in range(n_splits):
        rng = np.random.default_rng(100 + i)
        shuffled = rng.permutation(games)
        half = len(shuffled) // 2
        games_a = set(shuffled[:half])
        games_b = set(shuffled[half:])
        df_a = df[df["nflverse_game_id"].isin(games_a)]
        df_b = df[df["nflverse_game_id"].isin(games_b)]

        coefs_a, alpha_a = fit_ridgecv(df_a, outcome_col, controls)
        coefs_b, alpha_b = fit_ridgecv(df_b, outcome_col, controls)

        both = pd.DataFrame({"a": coefs_a, "b": coefs_b}).dropna()
        corr = both["a"].corr(both["b"])
        corrs.append(corr)
        ns.append(len(both))
        print(f"  split {i}: n_players_common={len(both)}, n_plays_a={len(df_a)}, n_plays_b={len(df_b)}, "
              f"alpha_a={alpha_a:.1f}, alpha_b={alpha_b:.1f}, corr={corr:.3f}")
    return corrs, ns


print("\n-- pass_protection split-half --")
pp_corrs, pp_ns = split_half_corr(pp23, "pressure_outcome", PASS_CONTROLS)
print(f"  mean split-half corr: {np.mean(pp_corrs):.3f}")

print("\n-- run_blocking split-half --")
rb_corrs, rb_ns = split_half_corr(rb23, "rushing_yards", RUN_CONTROLS)
print(f"  mean split-half corr: {np.mean(rb_corrs):.3f}")

print(f"\nSUMMARY: pass_pro split-half mean={np.mean(pp_corrs):.3f}, run_block split-half mean={np.mean(rb_corrs):.3f}")
print(f"Overall split-half mean: {np.mean(pp_corrs + rb_corrs):.3f}")
print("Compare to year-over-year avg from PHASE2_REPORT.md: 0.144")

# ---------------------------------------------------------------------------
# 2. Sample size per lineman
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("2. SAMPLE SIZE PER LINEMAN (play counts) - 2023")
print("=" * 80)


def play_counts(df):
    counts = {}
    for ids in df.ol_ids:
        for pid in ids:
            counts[pid] = counts.get(pid, 0) + 1
    return pd.Series(counts)


pp_counts_23 = play_counts(pp23)
rb_counts_23 = play_counts(rb23)
print("pass_pro play-count distribution (2023):")
print(pp_counts_23.describe())
print("\nrun_block play-count distribution (2023):")
print(rb_counts_23.describe())

# Now check: does year-over-year correlation improve for high-snap linemen?
# Need coefficients across all seasons - refit RidgeCV per season (reuse pipeline logic)
print("\nFitting RidgeCV per season 2021-2025 to check YoY corr by snap-count bucket...")

season_coefs = {"pass_protection": {}, "run_blocking": {}}
season_counts = {"pass_protection": {}, "run_blocking": {}}
season_alphas = {"pass_protection": {}, "run_blocking": {}}

for season in range(2021, 2026):
    dfS, _ = load_season(conn, season)
    ppS = build_pass_pro_dataset(dfS)
    rbS = build_run_block_dataset(dfS)

    coefs, alpha = fit_ridgecv(ppS, "pressure_outcome", PASS_CONTROLS)
    season_coefs["pass_protection"][season] = coefs
    season_counts["pass_protection"][season] = play_counts(ppS)
    season_alphas["pass_protection"][season] = alpha

    coefs, alpha = fit_ridgecv(rbS, "rushing_yards", RUN_CONTROLS)
    season_coefs["run_blocking"][season] = coefs
    season_counts["run_blocking"][season] = play_counts(rbS)
    season_alphas["run_blocking"][season] = alpha
    print(f"  season {season} done (pp_alpha={season_alphas['pass_protection'][season]:.1f}, "
          f"rb_alpha={season_alphas['run_blocking'][season]:.1f})")


def yoy_corr_by_threshold(submodel, threshold):
    coefs = season_coefs[submodel]
    counts = season_counts[submodel]
    seasons = sorted(coefs.keys())
    results = []
    for i in range(len(seasons) - 1):
        s1, s2 = seasons[i], seasons[i + 1]
        both = pd.DataFrame({"c1": coefs[s1], "c2": coefs[s2]}).dropna()
        # require min play count in BOTH seasons >= threshold
        cnt1 = counts[s1].reindex(both.index).fillna(0)
        cnt2 = counts[s2].reindex(both.index).fillna(0)
        mask = (cnt1 >= threshold) & (cnt2 >= threshold)
        sub = both[mask]
        if len(sub) >= 5:
            corr = sub["c1"].corr(sub["c2"])
            results.append((s1, s2, len(sub), corr))
    return results


print("\n-- YoY correlation by snap-count threshold --")
for submodel in ["pass_protection", "run_blocking"]:
    print(f"\n{submodel}:")
    for threshold in [0, 300, 500, 700]:
        res = yoy_corr_by_threshold(submodel, threshold)
        if res:
            avg = np.mean([r[3] for r in res])
            total_n = sum(r[2] for r in res)
            print(f"  threshold>={threshold}: avg_corr={avg:.3f} across {len(res)} pairs, "
                  f"total_player_season_pairs={total_n}")
            for s1, s2, n, c in res:
                print(f"      {s1}->{s2}: n={n}, corr={c:.3f}")

# ---------------------------------------------------------------------------
# 3. Multicollinearity / identifiability for stable starting fives
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("3. MULTICOLLINEARITY / IDENTIFIABILITY (2023, pass_pro dataset)")
print("=" * 80)

X23, all_ol_23 = design_matrix(pp23, PASS_CONTROLS)
indicator_23 = X23[:, :len(all_ol_23)]
# condition number / rank of indicator matrix (dense is too big potentially; check size)
print(f"Indicator matrix shape: {indicator_23.shape}, nnz={indicator_23.nnz}")

# Compute singular values via sparse SVD (top and bottom via dense if feasible)
dense_ind = indicator_23.toarray()
print(f"Dense indicator matrix size: {dense_ind.shape}, memory ~{dense_ind.nbytes/1e6:.1f}MB")
# rank via SVD
svals = np.linalg.svd(dense_ind, compute_uv=False)
print(f"Number of singular values: {len(svals)}")
print(f"Largest 5 singular values: {svals[:5]}")
print(f"Smallest 5 singular values: {svals[-5:]}")
tol = svals[0] * max(dense_ind.shape) * np.finfo(float).eps
rank = np.sum(svals > tol)
print(f"Numerical rank: {rank} out of {dense_ind.shape[1]} columns (tol={tol:.2e})")
print(f"Condition number (max/min sval): {svals[0]/svals[-1]:.2e}")
n_near_zero = np.sum(svals < 1e-6)
print(f"Number of near-zero (<1e-6) singular values: {n_near_zero}")

# Identify teams with a highly stable starting 5 (same 5 on 90%+ of snaps)
print("\n-- Team lineup stability check (2023, using pass_pro plays) --")
pp23_team = pp23.copy()
pp23_team["ol_tuple"] = pp23_team.ol_ids.apply(lambda x: tuple(sorted(x)))
team_stability = []
for team, grp in pp23_team.groupby("posteam"):
    top_lineup = grp["ol_tuple"].value_counts()
    if len(top_lineup) == 0:
        continue
    top_frac = top_lineup.iloc[0] / len(grp)
    team_stability.append({"team": team, "n_plays": len(grp), "top_lineup_frac": top_frac,
                            "n_distinct_lineups": len(top_lineup)})
team_stability_df = pd.DataFrame(team_stability).sort_values("top_lineup_frac", ascending=False)
print(team_stability_df.to_string(index=False))

stable_teams = team_stability_df[team_stability_df.top_lineup_frac >= 0.90]
print(f"\nTeams with same-5 lineup on >=90% of pass plays: {len(stable_teams)} / {len(team_stability_df)}")

# For one very stable team, check correlation structure among their 5 indicator columns
if len(stable_teams):
    example_team = stable_teams.iloc[0]["team"]
    ex_grp = pp23_team[pp23_team.posteam == example_team]
    top_lineup = ex_grp["ol_tuple"].value_counts().index[0]
    print(f"\nExample: {example_team}, most common lineup (n=5): {top_lineup}")
    # rows where this exact lineup played
    mask = ex_grp["ol_tuple"] == top_lineup
    print(f"  plays with this exact lineup: {mask.sum()} / {len(ex_grp)} team plays")
    # Since these 5 always appear together, their indicator columns are identical
    # over these rows -> within this subset the columns for these 5 players are
    # perfectly collinear (rank deficient by 4 out of 5 for this block).
    cols_idx = [all_ol_23.index(p) for p in top_lineup if p in all_ol_23]
    sub_block = dense_ind[np.array(pp23_team.index.get_indexer(ex_grp[mask].index)), :][:, cols_idx]
    print(f"  sub-block shape (plays x 5 players): {sub_block.shape}")
    print(f"  are all rows identical (all-ones across these 5 cols)? {np.all(sub_block == 1)}")

# ---------------------------------------------------------------------------
# 4. Alpha sensitivity
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("4. ALPHA SENSITIVITY (2023)")
print("=" * 80)


def alpha_sensitivity(df, outcome_col, controls, name):
    coefs_cv, alpha_cv = fit_ridgecv(df, outcome_col, controls)
    print(f"\n{name}: RidgeCV chose alpha={alpha_cv:.1f}")
    results = {"cv": (alpha_cv, coefs_cv)}
    for mult, label in [(0.1, "10x_lower"), (10, "10x_higher")]:
        alpha_fixed = alpha_cv * mult
        coefs_fixed = fit_fixed_alpha(df, outcome_col, controls, alpha_fixed)
        corr_to_cv = coefs_cv.corr(coefs_fixed)
        print(f"  {label} (alpha={alpha_fixed:.1f}): corr to CV-chosen coefs = {corr_to_cv:.3f}, "
              f"coef std={coefs_fixed.std():.4f} (CV coef std={coefs_cv.std():.4f})")
        results[label] = (alpha_fixed, coefs_fixed)
    return results


pp_alpha_results = alpha_sensitivity(pp23, "pressure_outcome", PASS_CONTROLS, "pass_protection 2023")
rb_alpha_results = alpha_sensitivity(rb23, "rushing_yards", RUN_CONTROLS, "run_blocking 2023")

# Now check: does changing alpha materially change split-half STABILITY (not just
# similarity to the CV coefs)? Refit split halves at each alpha level and compute
# split-half correlation.
print("\n-- Does alpha choice change split-half stability? --")


def split_half_corr_fixed_alpha(df, outcome_col, controls, alpha, n_splits=3):
    games = df["nflverse_game_id"].unique()
    corrs = []
    for i in range(n_splits):
        rng = np.random.default_rng(100 + i)
        shuffled = rng.permutation(games)
        half = len(shuffled) // 2
        games_a = set(shuffled[:half])
        games_b = set(shuffled[half:])
        df_a = df[df["nflverse_game_id"].isin(games_a)]
        df_b = df[df["nflverse_game_id"].isin(games_b)]
        coefs_a = fit_fixed_alpha(df_a, outcome_col, controls, alpha)
        coefs_b = fit_fixed_alpha(df_b, outcome_col, controls, alpha)
        both = pd.DataFrame({"a": coefs_a, "b": coefs_b}).dropna()
        corrs.append(both["a"].corr(both["b"]))
    return np.mean(corrs)


for name, df_, outcome_col, controls, alpha_cv in [
    ("pass_protection", pp23, "pressure_outcome", PASS_CONTROLS, pp_alpha_results["cv"][0]),
    ("run_blocking", rb23, "rushing_yards", RUN_CONTROLS, rb_alpha_results["cv"][0]),
]:
    print(f"\n{name} (RidgeCV alpha={alpha_cv:.1f}):")
    for mult, label in [(0.1, "10x_lower"), (1.0, "cv_alpha"), (10, "10x_higher"), (100, "100x_higher")]:
        alpha_fixed = alpha_cv * mult
        sh_corr = split_half_corr_fixed_alpha(df_, outcome_col, controls, alpha_fixed)
        print(f"  alpha={alpha_fixed:.1f} ({label}): split-half corr = {sh_corr:.3f}")

conn.close()
print("\nDONE")
