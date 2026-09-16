# Academic Review: Weekly NFL Player Projection Stack

## Executive answer

The strategic change is mathematically correct: replace the frozen season-value board with a **weekly, game-conditioned, probabilistic projection system**, retain weekly player props as the production benchmark, and keep the independent model in shadow until it demonstrates genuine out-of-sample improvement. The key correction is to model the football-generating process—availability, team plays, player opportunity, efficiency, and scoring events—rather than regress fantasy points directly on a small set of season-level inputs.

The current code and data platform already support a shadow-model workflow better than they support a reliable research model. The repository contains weekly projection, weekly-props ingestion, comparison, coverage, scoring, job orchestration, and UI modules; Supabase already stores projection runs, per-player quantiles, prop snapshots, player-week benchmark rows, coverage rows, and scoring outcomes. However, the current weekly model is a static weighted baseline built mainly from seasonal attempts/carries/targets, a team-total multiplier, an opponent rank adjustment, and Gaussian uncertainty; it is not yet a learned game-level model.[^1][^2][^3]

## Existing stack

### Architecture

The deployed system is a Python/FastAPI service on Vercel backed by Supabase/Postgres, with GitHub source containing the API, model pipelines, feature code, operational jobs, tests, and UI. The public endpoints include health, API readiness, full readiness, leagues, players, valuation, model/scoring, coverage, weekly props, and admin refresh routes.[^4][^5]

The database has a strong run-oriented foundation: `projection_run`, `player_projection`, `prop_market_snapshot`, `player_week_model_input`, `player_week_projection_benchmark`, `player_week_projection_coverage`, `projection_scoring_result`, `source_snapshot`, and `job_run` are already present. This means the correct next step is not a rewrite of Vercel or Supabase; it is a versioned modeling layer plus richer game/player-week features.[^1]

### What is already good

- **Immutable/versioned outputs:** model runs have IDs, versions, configuration JSON, code SHA fields, timestamps, and projection quantiles.[^1]
- **Benchmark separation:** weekly prop snapshots are stored separately from internal projections, and benchmark/coverage/scoring tables exist.[^1]
- **Operational observability:** source snapshots and job runs capture freshness, health, status, errors, and metadata.[^1]
- **Consumer-ready uncertainty shape:** player projections already expose P10/P50/P90 quantiles, even though the current method of producing them should change.[^2]
- **Reasonable service boundary:** computation lives in pipelines and features; the FastAPI app primarily serves results and schedules work.[^5][^4]

### Current weaknesses

The `run_weekly_projection` path uses hard-coded per-position rates and multipliers such as pass yards per attempt, receiving yards per target, touchdown rates, and interception rates. It estimates volume from season attempts, carries, or targets divided by 17, then applies team-total and opponent-rank modifiers; uncertainty is generated with a fixed coefficient of variation and normal quantiles. This creates five mathematical problems:[^2]

1. **No fitted conditional distribution.** Parameters are constants rather than estimates learned from historical player-games.
2. **Season-average opportunity is substituted for next-game opportunity.** That suppresses changes in role, injuries, depth-chart movement, and game scripts.
3. **Sequential event dependencies are lost.** Receptions cannot exceed targets and passing completions cannot exceed attempts, yet independently projected box-score components can violate those constraints.
4. **Tail behavior is misspecified.** Fantasy outcomes are skewed, zero-inflated for marginal players, and driven by discrete touchdowns; a normal interval with fixed CV is generally not a credible generative distribution.
5. **Opponent adjustment is too coarse.** A single defense rank applied uniformly by position does not represent expected plays, pass rate, coverage tendencies, pressure, box counts, or game state.

The current Supabase data confirms that the weekly projection payload contains box-score means and only three quantiles, while the props runs do not use the same mean/quantile schema. Current source snapshots are dominated by Sleeper league/player/trending endpoints plus one week of DraftKings and FanDuel props; they do not yet show a historical play-by-play, snap, route, injury, depth-chart, weather, or schedule-feature pipeline.

Operationally, recent failures include duplicate prepared-statement errors in `sync-leagues` and a stale/invalid Sleeper-user 404. These do not invalidate the architecture, but reliable weekly retraining and grading requires fixing connection-pool/prepared-statement behavior and treating deleted or renamed league users as an expected state.

## Correct mathematical target

### Unit of observation

Use one row per **player-game opportunity**, keyed by `(season, week, game_id, player_id, position)`. Keep all features timestamped with an `available_at` or snapshot timestamp, and construct multiple forecast vintages if useful—for example, Tuesday, Friday, and 90 minutes before kickoff. The training row may only contain information that was observable at that forecast cutoff.

Do not train primarily on fantasy points. Train a coherent multistage generative model and calculate fantasy points from simulated box-score outcomes:

\[
P(Y_{i,g} \mid X_{i,g}) = \sum_{z} P(Y_{i,g} \mid z, X_{i,g})P(z \mid X_{i,g}),
\]

where \(z\) represents latent game script, team volume, player availability, role, and efficiency. This produces internally consistent statistics and an empirical distribution of fantasy points rather than only a point estimate.

### Recommended hierarchy

| Layer | Quantity | Recommended model |
|---|---|---|
| Availability | active, inactive, limited, snap ceiling | Logistic/ordinal model plus explicit injury scenarios |
| Team environment | offensive plays, dropbacks, rushes, TD opportunities | Hierarchical negative binomial or boosted count model with team/opponent random effects |
| Player role | snap share, route share, target share, carry share | Logistic-normal or beta regression; jointly allocate shares so totals are coherent |
| Conversions | completions/attempts, receptions/targets | Binomial or beta-binomial |
| Yardage | yards per attempt/carry/target or conditional yards | Gamma/lognormal/Tweedie or gradient boosting with residual distribution |
| Touchdowns | pass/rush/receive TD counts | Bernoulli/hurdle/negative-binomial model conditioned on red-zone opportunities |
| Fantasy output | scoring-rule transformation | Monte Carlo from component distributions |

The beta-binomial is a standard extension when binomial outcomes are overdispersed. Flexible count families such as negative binomial or Conway–Maxwell–Poisson are preferable to naive Poisson when variance differs materially from the mean; sports applications show the latter can handle either over- or under-dispersion.[^6][^7][^8]

### Partial pooling

Use hierarchical effects for player, team, opponent, coach/play-caller, position, and season. Partial pooling is especially important early in a season and for rookies, backups, traded players, coaching changes, and small role samples. Public NFL analytics work has used multilevel models to isolate player contributions from positional and play context, while contemporary NFL Bayesian examples emphasize that shrinkage should vary with sample size, between-player variance, and play-level noise.[^9][^10]

A practical specification for a count component is:

\[
Y_{i,g} \sim \operatorname{NegBin}(\mu_{i,g}, \phi),
\]

\[
\log \mu_{i,g} = \alpha + u_i^{\text{player}} + u_t^{\text{team}} + u_d^{\text{defense}} + u_s^{\text{season}} + X_{i,g}\beta.
\]

For the first production-grade version, a regularized GLM/GAM or CatBoost/LightGBM model with carefully encoded lagged features is preferable to a neural network. The sample size is modest, the data-generating regime shifts each season, interpretability matters, and a published NFL DFS neural-network pipeline achieved only roughly the 31st percentile against real DraftKings lineups, illustrating that architectural complexity alone does not create forecast quality.[^11]

### Bayesian updating

ADP and season-long markets are useful **preseason priors**, not contemporaneous weekly covariates that should dominate after evidence arrives. Let prior player-role or efficiency parameters be informed by previous seasons, draft capital, age, team change, and preseason market expectations; update them with 2026 usage and performance using exponential time decay or a state-space model.

A simple dynamic parameterization is:

\[
\theta_{i,t} = \rho\theta_{i,t-1} + \eta_{i,t}, \qquad \eta_{i,t} \sim N(0,\sigma_\eta^2),
\]

with observations generated conditionally on \(\theta_{i,t}\). This formalizes the desired behavior: early forecasts shrink toward priors; new weeks increasingly control the posterior; genuine role changes can move faster when process variance is larger.

## Data to add

The strongest free source is nflverse. Its Python loader exposes play-by-play, schedules, players, weekly rosters, snap counts, Next Gen Stats, FTN charting, participation, depth charts, trades, and expected-fantasy-points data. The schedules dataset includes team rest, moneylines, spread, total, roof, surface, temperature, wind, and stadium fields.[^12][^13][^14][^15]

### Priority matrix

| Priority | Free data | Features to derive | Why it matters |
|---|---|---|---|
| P0 | nflverse play-by-play | team plays, neutral pass rate, situation-neutral pace, EPA, success rate, early-down tendencies, red-zone opportunities, air yards, YAC, pressure proxies | Establishes opportunity and efficiency at the correct play/game grain |
| P0 | Weekly player stats and rosters | lagged attempts, targets, carries, route/snap proxies, active roster, team/position continuity | Core player-game panel and identity reconciliation |
| P0 | Schedules and game lines | opponent, venue, rest, spread, total, implied team total, roof, surface, temperature, wind | Conditions volume and scoring environment; free schedule data already includes these fields[^13][^15] |
| P0 | Snap counts | offensive snap share and recent role trend | Separates real role change from noisy box-score production; nflverse exposes historical snap counts[^12][^16] |
| P0 | Official weekly injury/practice reports | DNP/limited/full sequence, final designation, active/inactive, teammate absences | Availability and redistributed opportunity are first-order drivers; public injury reports have been used as research data[^17] |
| P1 | nflverse depth charts/weekly rosters | starter rank, backup promotion, teammate vacancy flags | Improves cold starts and sudden role changes[^12][^16] |
| P1 | FTN charting via nflverse | motion, play action, RPO, defenders in box, blitzers, catchable/contested/drop flags | Adds schematic and opportunity-quality context from 2022 onward[^18] |
| P1 | nflverse expected opportunity | expected fantasy points, expected yards/TD value by opportunity, FPOE | Strong baseline and feature set; ffopportunity builds XFP from play-by-play with XGBoost and offers precomputed weekly/player-play data[^19][^20][^21] |
| P1 | Coaching/play-caller history | pace/pass-rate priors, personnel changes, coordinator continuity | Team behavior changes materially across staffs |
| P2 | Next Gen Stats | efficiency and separation-related features where coverage is reliable | Potential lift, but missingness and changing definitions require care[^12] |
| P2 | Transactions and Sleeper trends | role/news proxy, roster churn | Useful as weak nowcast features, not primary football inputs |

The missing feature family with the highest likely lift is **opportunity quality**, not additional season-long market data. Expected fantasy points quantify what an average player would score from the location and type of opportunities received; the free ffopportunity package trains on public nflverse play-by-play and can return weekly and play-level expected-point data.[^19][^21]

### Features by position

**Quarterback:** expected dropbacks, designed rush share, scramble rate, pressure/sack rate, neutral pass rate, play-action/RPO rate, air yards per attempt, completion probability over expected, red-zone dropbacks, opponent pressure and coverage proxies, offensive-line injuries, weather.

**Running back:** active probability, snap share, route share, carry share by game state, goal-line carry share, two-minute role, target share, box count, run direction, team rushing expected points, offensive-line continuity, spread and implied team total.

**Wide receiver/tight end:** routes per dropback, target share, first-read/designed-read share when available, air-yard share, average depth of target, end-zone/red-zone targets, slot/wide alignment proxies, catchable-ball rate, opponent pass-defense/coverage tendencies, teammate vacancy, quarterback status. FTN's free charting fields include primary-read coding, catchable and contested-ball flags, motion, play action, RPO, box count, and blitzers.[^18]

## Market data policy

Weekly props should be handled in three distinct roles:

1. **Production benchmark:** continue serving the market-derived estimate in the app while the independent model is in shadow.
2. **Evaluation comparator:** freeze a timestamped market snapshot for every player-stat line, de-vig over/under prices, and compare model distributions to both realized outcomes and market-implied distributions.
3. **Optional ensemble after independence testing:** only blend the market into the model after reporting a market-free challenger. Otherwise the system cannot determine whether its football features add information or merely reproduce the benchmark.

ADP and season-long Vegas should be limited to preseason/early-season priors, missing-player imputation, and sanity-check alarms. They must be recorded with snapshot dates; using final or later-season ADP/market information in earlier historical rows creates leakage.

Prop lines alone are medians or near-median thresholds, not complete expected-value distributions. Store book, market, line, both prices, timestamp, limits if available, and player/game identifiers; de-vig probabilities and construct a distribution using alternate lines when available. The existing database's prop snapshot and odds fields are a useful start.[^1]

## Validation design

### Backtest

Use rolling-origin evaluation, never random train/test splits:

- Train through season/week \(t-1\), predict week \(t\).
- Rebuild every historical feature exactly as it would have appeared at the chosen cutoff.
- Keep 2025 as a final pre-2026 holdout if enough earlier seasons are available.
- In 2026, append outcomes weekly but retain immutable forecast snapshots.
- Report by position, stat market, starter/backup tier, injury status, and forecast horizon.

The literature commonly uses recent moving windows in weekly fantasy prediction, but a pure four-week window throws away useful priors. A hierarchical/dynamic model should instead retain older data with decay and partial pooling.[^22]

### Baselines

Every candidate must beat simple baselines before promotion:

| Baseline | Purpose |
|---|---|
| Last game / trailing 3-game mean | Detect needless complexity |
| Season-to-date per-game rate | Stable naive baseline |
| Exponentially weighted mean | Recency baseline |
| Position/team hierarchical mean | Cold-start and shrinkage baseline |
| nflverse expected opportunity | Public football-data baseline[^19] |
| Weekly prop consensus | Product and market benchmark |
| Market plus simple residual model | Tests whether football features add orthogonal signal |

### Metrics

Do not select by fantasy-point RMSE alone. Evaluate component outcomes and the full predictive distribution:

- MAE and RMSE for means/medians.
- Pinball loss for P10/P50/P90.
- CRPS for the full simulated distribution.
- Prediction-interval coverage and width.
- PIT/reliability plots for calibration.
- Brier score and log loss for prop over/under probabilities.
- Rank correlation and top-N decision regret for lineup/use cases.
- Incremental loss versus closing props, with paired bootstrap confidence intervals clustered by game/week.

Probabilistic forecast evaluation should maximize sharpness subject to calibration; PIT histograms, marginal calibration, interval width, and proper scoring rules are standard tools. CRPS is a proper scoring rule for full continuous predictive distributions.[^23][^24][^25]

### Promotion gate

A defensible promotion policy is:

- Minimum 6–8 live shadow weeks, preferably a full season for broad claims.
- No material leakage or missingness regressions.
- Better than naive and expected-opportunity baselines across most positions.
- Statistically credible improvement or parity versus the weekly-prop benchmark on primary losses.
- Calibrated 50%, 80%, and 90% intervals within prespecified tolerances.
- No severe subgroup failure for backups, questionable players, or low-volume positions.
- Stable performance across forecast vintages and books.

With only one or two 2026 weeks, the model should update but not be declared superior. Use historical seasons for estimation and 2026 as sequential external validation.

## Recommended implementation

### Data layer

Add normalized tables rather than embedding all features in JSON:

- `game`: schedule, venue, spread/total snapshots, weather.
- `player_game_observation`: realized box score and fantasy points.
- `player_game_feature_snapshot`: cutoff timestamp, feature version, all pregame covariates.
- `player_availability_snapshot`: practice and game designation history.
- `player_role_snapshot`: snaps, routes/proxies, target/carry shares, depth rank.
- `model_artifact`: training window, hyperparameters, code SHA, data hash, metrics, artifact URI.
- `forecast_sample` or compressed distribution parameters: enough to reproduce quantiles and over probabilities.

Retain the existing run and benchmark tables, but add uniqueness constraints on `(model_run_id, player_id, game_id, forecast_vintage)` and `(source, book, market, player_id, game_id, snapshot_at)`. Keep raw source snapshots for auditability.

### Model path

1. Build a historical player-game feature mart from nflverse.
2. Implement leakage-safe rolling features and identity checks.
3. Fit transparent position-specific regularized baselines.
4. Add hierarchical or CatBoost components for team volume, shares, and efficiency.
5. Generate 5,000–20,000 coherent Monte Carlo samples per player-game.
6. Save mean, median, P10/P25/P75/P90, component means, active probability, and diagnostic metadata.
7. Score all forecasts after stat corrections, including the frozen market snapshot.
8. Publish a weekly shadow report and only then consider ensembling or promotion.

### Stack decision

Keep Supabase for the research mart, immutable forecasts, benchmark snapshots, and scoring. Keep Vercel for the read API/UI and short orchestration triggers, but avoid fitting substantial models inside request handlers. Train in GitHub Actions or another batch runner, write artifacts/results to Supabase, and let Vercel serve completed runs. Current Vercel account context did not expose a linked team or Git deployment, so the live deployment linkage could not be independently verified.

Before trusting weekly automation, fix the duplicate prepared-statement failure, add idempotent job keys, use transaction-safe upserts, and separate ingestion success from model success. Add data contracts for row counts, game coverage, player identity resolution, null rates, feature cutoff compliance, and prop-market coverage.

## Conclusion

The new direction is the right one, but the mathematically correct object is not a weekly fantasy-points regressor. It is a leakage-safe, hierarchical, game-level probabilistic model of availability, team volume, player opportunity, conversion, yardage, and touchdowns, converted to fantasy points through simulation.

The existing stack should be evolved rather than replaced. Its run versioning, quantiles, props benchmark, coverage, and scoring schemas are valuable; the substantive gap is free football data and learned distributions. Add nflverse play-by-play, schedules/lines/weather, weekly rosters, snap counts, depth charts, expected opportunity, FTN charting, and official injury/practice data first, then prove incremental value against frozen weekly-prop snapshots with rolling-origin calibration and proper scoring rules.

---

## References

1. [Simulation-Based Decision Making in the NFL using NFLSimulatoR](https://arxiv.org/pdf/2102.01846.pdf) - In this paper, we introduce an R software package for simulating plays and
drives using play-by-play...

2. [Deep Artificial Intelligence for Fantasy Football Language Understanding](https://arxiv.org/ftp/arxiv/papers/2111/2111.02874.pdf) - ...million articles, videos and
podcasts each day enables the system to comprehend natural language ...

3. [Automated player identification and indexing using two-stage deep learning network](https://pmc.ncbi.nlm.nih.gov/articles/PMC10282031/) - American football games attract significant worldwide attention every year. Identifying players from...

4. [next-gen-scraPy: Extracting NFL Tracking Data from Images to Evaluate
  Quarterbacks and Pass Defenses](https://arxiv.org/abs/1906.03339v2) - The NFL collects detailed tracking data capturing the location of all players
and the ball during ea...

5. [Automated player identification and indexing using two-stage deep
  learning network](https://arxiv.org/ftp/arxiv/papers/2204/2204.13809.pdf) - American football games attract significant worldwide attention every year.
Identifying players from...

6. [Bayesian Bivariate Conway-Maxwell-Poisson Regression Model for
  Correlated Count Data in Sports](https://arxiv.org/html/2409.17129v1) - ... performance
of our proposed CMP model matches or outperforms standard Poisson and Negative
Binom...

7. [A new regression model for overdispersed binomial data accounting for outliers and an excess of zeros](https://pmc.ncbi.nlm.nih.gov/articles/PMC8360060/) - Binary outcomes are extremely common in biomedical research. Despite its popularity, binomial regres...

8. [Comparison of Hierarchical Bayesian Models for Overdispersed Count Data Using DIC and Bayes' Factors](https://academic.oup.com/biometrics/article/65/3/962/7331879) - Summary. When replicate count data are overdispersed, it is common practice to incorporate this extr...

9. [Chapter 32: Bayesian Methods in Football | NFL Analytics Textbook](https://nflanalytic.com/tutorials-chapter-32.html) - Bayesian methods for football analytics. Probabilistic models for NFL betting and uncertainty quanti...

10. [a reproducible method for offensive player evaluation in ...](https://scispace.com/pdf/nflwar-a-reproducible-method-for-offensive-player-evaluation-1zlic1pkrc.pdf)

11. [Method and Validation for Optimal Lineup Creation for Daily Fantasy ...](https://arxiv.org/abs/2309.15253) - Daily fantasy sports (DFS) are weekly or daily online contests where real-game performances of indiv...

12. [nflverse/nflreadpy: python port of nflreadr package ...](https://github.com/nflverse/nflreadpy) - python port of nflreadr package for loading nflverse data - nflverse/nflreadpy

13. [nflfastR/NEWS.md at master · nflverse/nflfastR](https://github.com/nflverse/nflfastR/blob/master/NEWS.md) - A Set of Functions to Efficiently Scrape NFL Play by Play Data - nflverse/nflfastR

14. [Load Game/Schedule Data — load_schedules - nflreadr - nflverse](https://nflreadr.nflverse.com/reference/load_schedules.html) - This returns game/schedule information as maintained by Lee Sharpe.

15. [Data Dictionary - Schedules - CRAN](https://cran.r-project.org/web/packages/nflreadr/vignettes/dictionary_schedules.html) - nflverse dataframes. The spread line for the game. The total line for the game. The temperature at t...

16. [NFL dataset loaders | sdv-py](https://py.sportsdataverse.org/docs/0.0.74/nfl/reference/loaders) - Automation status

17. [Injury Rates Remained Elevated in the Second National Football League Season After the Onset of the COVID-19 Pandemic](https://pmc.ncbi.nlm.nih.gov/articles/PMC9742207/) - Purpose
The purpose of this study was to compare the injury incidence of the 2018-2019 and 2020 Nati...

18. [Data Dictionary - FTN Charting - nflreadr - nflverse](https://nflreadr.nflverse.com/articles/dictionary_ftn_charting.html)

19. [Expected Points Models for Fantasy Football • ffopportunity](https://ffopportunity.ffverse.com/) - ffopportunity builds a dataframe of Expected Fantasy Points by preprocessing and applying an xgboost...

20. [Load Expected Fantasy Points — load_ff_opportunity - nflreadr](https://nflreadr.nflverse.com/reference/load_ff_opportunity.html) - This function downloads precomputed expected points data from ffopportunity automated releases.

21. [Build EP — ep_build - ffopportunity](https://ffopportunity.ffverse.com/reference/ep_build.html) - This function builds Expected Fantasy Points predictions by downloading the xgboost models and play-...

22. [Abstract](https://ar5iv.labs.arxiv.org/html/2309.15253)

23. [HAL Id: hal-00363242](https://hal.science/hal-00363242v1/document)

24. [Evaluating Probabilistic Predictions via Proper Scoring Rules](https://arxiv.org/html/2603.08206v5)

25. [Probabilistic Forecasts, Calibration and Sharpness](https://academic.oup.com/jrsssb/article/69/2/243/7109375) - by T Gneiting · 2007 · Cited by 2665 — A scoring rule is proper if the expected value of the penalty...

