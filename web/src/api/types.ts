/** Shared response metadata from /api/v1 read endpoints. */
export interface ApiMeta {
  data_as_of: string;
  projection_run_id: string;
  snapshot_ids?: Record<string, string>;
}

export interface ApiError {
  detail:
    | string
    | { msg: string; type: string }[]
    | { code?: string; message?: string };
}

export interface User {
  id: string;
  email: string;
  csrf_token?: string;
  session_expires_at?: string;
}

export interface MagicLinkResponse {
  status: "sent" | "dev_link";
  link?: string;
}

export interface VerifyResponse {
  status: "ok";
  csrf_token: string;
}

export interface LeagueSummary {
  id: string;
  name: string;
  season: number;
  scoring_type: string;
  roster_positions: string[];
  is_dynasty: boolean;
  /** False when the league is synced but not in the owner config file. */
  is_configured?: boolean;
}

export interface LeagueDetail extends LeagueSummary {
  total_rosters: number;
  playoff_week_start?: number;
  settings: Record<string, unknown>;
}

export interface LeagueRules {
  league_id: string;
  scoring: Record<string, number>;
  roster_slots: Record<string, number>;
  unsupported_keys: string[];
}

export interface RosterPlayer {
  player_id: string;
  name: string;
  position: string;
  team?: string;
  slot?: string;
}

export interface Roster {
  roster_id: number;
  week: number;
  players: string[];
  starters: string[];
  reserve: string[];
  manager_name?: string;
  player_details?: RosterPlayer[];
}

export interface Matchup {
  week: number;
  roster_id: number;
  opponent_roster_id: number;
  projected_points: number;
  win_probability: number;
  starters: RosterPlayer[];
}

export interface PlayerProjection {
  player_id: string;
  name: string;
  position: string;
  team?: string;
  points_mean: number;
  points_p10: number;
  points_p90: number;
  meta: ApiMeta;
}

export interface RankingEntry {
  player_id: string;
  name: string;
  position: string;
  team?: string;
  rank: number;
  points_mean: number;
  points_p10: number;
  points_p90: number;
}

export type RankingMode = "weekly" | "ros" | "dynasty";
export type OpponentMode = "current" | "optimized";

/**
 * A points range straight from the release bundle quantiles.
 *
 * `null` fields mean the API did not publish that quantile. Callers must render
 * "not available" rather than substituting the mean, so the UI never implies a
 * confidence interval the model did not produce.
 */
export interface PointsRange {
  p10: number | null;
  p50: number | null;
  p90: number | null;
  mean: number | null;
}

export interface LineupSwap {
  out_player_id: string;
  in_player_id: string;
  win_probability_delta: number;
  reason: string;
}

export interface LineupRecommendation {
  week: number;
  opponent_mode: OpponentMode;
  starters: RosterPlayer[];
  swaps: LineupSwap[];
  win_probability: number | null;
  /** Full matchup probability vector (win/loss/tie) when the API supplies it. */
  matchup_probabilities: Record<string, number | null>;
  points: PointsRange;
  contract_hash?: string;
  meta: ApiMeta;
}

export interface WaiverAdd {
  player_id: string;
  name: string;
  position: string;
  faab_min: number;
  faab_max: number;
  drop_player_id?: string;
  reason: string;
  /** Individual rationale lines as published by the waiver engine. */
  rationale: string[];
  confidence: number | null;
  start_probability: number | null;
  incremental_utility: number | null;
}

export interface WaiverRecommendation {
  week: number;
  adds: WaiverAdd[];
  meta: ApiMeta;
}

export interface TradeSide {
  roster_id: number;
  player_ids: string[];
  pick_ids?: string[];
}

/**
 * Trade evaluation exactly as published by `POST /trades/evaluate`.
 *
 * Every numeric field is nullable on purpose: the UI renders "not available"
 * for anything the API did not return. There is deliberately no letter grade —
 * the API does not publish one and the client must not invent one.
 */
export interface TradeEvaluation {
  league_id: string;
  horizon: "weekly" | "ros" | "dynasty";
  objective: {
    side_a_value: number | null;
    side_b_value: number | null;
    side_a_gain: number | null;
    side_b_gain: number | null;
  };
  fairness: {
    gap: number | null;
    uncertainty: number | null;
    fair: boolean | null;
  };
  acceptance: {
    side_a_probability: number | null;
    side_b_probability: number | null;
    tendency_adjustment: number | null;
  };
  meta: ApiMeta;
}

export interface TradeProposal {
  proposal_id: string;
  status: "pending" | "accepted" | "rejected" | "countered";
  sides: TradeSide[];
  evaluation?: TradeEvaluation;
}

export interface ManagerTendencies {
  roster_id: number;
  trade_frequency: number;
  win_now_bias: number;
  pick_premium: number;
  position_preferences: Record<string, number>;
}

export interface DraftBoardEntry {
  player_id: string;
  name: string;
  position: string;
  team?: string;
  rank: number;
  tier?: number;
  vorp?: number;
  points_mean?: number;
  replacement_points?: number;
  replacement_rank?: number;
}

export interface DraftBoardProfile {
  league_specific: boolean;
  ranking_basis?: "league_vorp" | "sealed_vorp";
  points_unit?: "season_total";
  team_count?: number;
  roster_positions: string[];
  contract_hash?: string;
  scoring_fidelity?: string;
  replacement_ranks: Record<string, number>;
  caveats: string[];
}

export interface DraftContext {
  draft_status: "live" | "preseason";
  draft_id?: string | null;
  season?: number;
  nfl_week?: number;
  current_pick?: number | null;
  on_clock_roster_id?: number | null;
}

export interface DraftBoard {
  league_id: string;
  entries: DraftBoardEntry[];
  context?: DraftContext;
  profile?: DraftBoardProfile;
  meta: ApiMeta;
}

export type DraftRankTier = "adp" | "ecr" | "prior_pts" | "none";

export interface DraftChecklistEntry {
  player_id: string;
  sleeper_id?: string;
  name: string;
  position: string;
  team?: string;
  adp?: number | null;
  ecr?: number | null;
  prior_pts?: number | null;
  rank_tier: DraftRankTier;
  pos_market_rank?: number;
  /** League-scoring VORP overall rank when meta.rank_source is league_vorp. */
  overall_rank?: number;
  league_pts?: number | null;
  unranked_break?: boolean;
  checks: Record<string, boolean>;
}

export interface DraftChecklistTeam {
  abbr: string;
  name: string;
  offense_rank?: number | null;
  ol_pass_rank?: number | null;
  ol_run_rank?: number | null;
  ol_unit_rank?: number | null;
  sos_pass_rank?: number | null;
  sos_rush_rank?: number | null;
  sos_unit_rank?: number | null;
}

export interface DraftChecklistMarketAsOf {
  adp_start?: string | null;
  adp_end?: string | null;
  ecr_scrape?: string | null;
  scoring?: string;
  teams?: number;
  comparison_generated_at?: string | null;
  matched_adp?: number | null;
  matched_ecr?: number | null;
}

export interface DraftChecklist {
  league_id: string;
  season: number;
  available: boolean;
  entries: DraftChecklistEntry[];
  teams: DraftChecklistTeam[];
  criteria_by_position: Record<string, string[]>;
  criteria_labels: Record<string, string>;
  checklist_meta: {
    market_as_of?: DraftChecklistMarketAsOf;
    sos_included?: boolean;
    ol_included?: boolean;
    volume_caveat?: string;
    schedule_2026_reg_games?: number;
    scoring_flavor?: string;
    team_count?: number;
    rank_source?: string;
    board_order?: {
      scoring?: string;
      as_of?: string | null;
      source?: string;
      replacement?: Record<string, number>;
    };
  };
  meta: ApiMeta;
}

export interface Citation {
  title: string;
  url: string;
  publisher: string;
  published_at?: string;
  confidence?: number | null;
}

export interface InjuryEvidence {
  player_id: string;
  status: string;
  summary: string;
  sources: Citation[];
  meta: ApiMeta;
}

export interface ProjectionChange {
  player_id: string;
  from_run_id: string;
  to_run_id: string;
  delta_points: number;
  drivers: string[];
  meta: ApiMeta;
}

export interface AssistantMessage {
  role: "user" | "assistant";
  content: string;
}

export interface AssistantResponse {
  response_id: string;
  messages: AssistantMessage[];
  tool_calls?: Array<{ name: string; result_id: string }>;
  meta: ApiMeta;
}

export interface JobStatus {
  job_id: string;
  status: "queued" | "running" | "succeeded" | "failed";
  correlation_id: string;
  started_at?: string;
  finished_at?: string;
  error?: string;
}

export interface SyncResponse {
  job_id: string;
}

/**
 * How the data behind a recommendation was produced.
 *
 * The owner has to be able to tell fixture data from live data and a trained
 * model from a fallback, so these come straight from the API rather than being
 * inferred in the UI.
 */
export interface OperationsModes {
  /** `fixture` = recorded Sleeper payloads; `live` = the real read-only API. */
  sleeper_source?: "fixture" | "live";
  projection_source?: string;
  weekly_rnd_enabled?: boolean;
  /** `trained` | `fallback` | `fixture` — the weekly v2 artifact state. */
  weekly_v2_state?: string;
  weekly_v2_model_version?: string | null;
  weekly_v2_reasons?: string[];
  /** False when required trained artifacts are missing or incompatible. */
  auto_publish_allowed?: boolean;
  by_horizon?: Record<string, string | null>;
}

export interface CapabilityStatus {
  capability: string;
  verdict: string;
  source: string;
  detail: string;
  caveats?: string[];
}

export interface OperationsProductionPanel {
  healthy?: boolean;
  degraded_capabilities?: string[];
  sealed_release?: CapabilityStatus | null;
  status_overlay?: CapabilityStatus | null;
}

export interface OperationsWeeklyRndPanel {
  healthy?: boolean;
  state?: string;
  auto_publish_allowed?: boolean;
  failed_gates?: string[];
}

export interface OperationsStatus {
  data_as_of?: string;
  // The API sends `null` when nothing has been recorded yet, and "never synced"
  // is a state the UI has to distinguish from "not in the payload".
  last_sync_at?: string | null;
  active_projection_run_id?: string | null;
  modes?: OperationsModes;
  production?: OperationsProductionPanel;
  weekly_rnd?: OperationsWeeklyRndPanel;
  capabilities?: {
    capabilities: CapabilityStatus[];
    production_healthy: boolean;
    weekly_rnd_healthy: boolean;
  };
  failed_gates: string[];
  estimated_month_cost_usd?: number;
  latest_job?: {
    id: string;
    name: string;
    status: string;
    finished_at?: string;
  } | null;
  latest_promotion?: {
    mode: string;
    promoted: boolean;
    created_at: string;
  } | null;
  openai_configured?: boolean;
}

export type ReadonlyRecommendationKey = "lineup" | "waivers";

export interface CachedRecommendation<T> {
  key: ReadonlyRecommendationKey;
  leagueId: string;
  week: number;
  /**
   * Request-shaping parameters that change the payload (e.g. `opponent_mode`).
   * Two variants of the same screen must never share a cache slot.
   */
  variant: string;
  fetchedAt: string;
  payload: T;
}
