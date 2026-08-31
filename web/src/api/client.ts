import {
  adaptAssistant,
  adaptInjuryEvidence,
  adaptLeagues,
  adaptLineup,
  adaptMagicLink,
  adaptManagerTendencies,
  adaptProjectionChanges,
  adaptRankings,
  adaptRosters,
  adaptSync,
  adaptTradeEvaluation,
  adaptWaivers,
} from "./adapters";
import type {
  ApiError,
  AssistantResponse,
  DraftBoard,
  InjuryEvidence,
  LeagueDetail,
  LeagueRules,
  LeagueSummary,
  LineupRecommendation,
  MagicLinkResponse,
  ManagerTendencies,
  Matchup,
  OperationsStatus,
  OpponentMode,
  PlayerProjection,
  ProjectionChange,
  RankingEntry,
  RankingMode,
  SyncResponse,
  TradeEvaluation,
  TradeProposal,
  TradeSide,
  User,
  VerifyResponse,
  WaiverRecommendation,
} from "./types";

const API_PREFIX = "/api/v1";

export class ApiClientError extends Error {
  constructor(
    message: string,
    public readonly status: number,
    public readonly body?: ApiError,
  ) {
    super(message);
    this.name = "ApiClientError";
  }
}

export interface ApiClientOptions {
  baseUrl?: string;
  csrfToken?: string;
}

type RawRecord = Record<string, unknown>;

/** Called whenever the API rejects a request because the session is gone. */
export type UnauthorizedListener = () => void;

interface RequestInitExtras {
  idempotencyKey?: string;
  /**
   * Suppress the global 401 broadcast. Only the auth endpoints set this: a 401
   * from `/auth/verify` means "that token is bad", not "your session expired".
   */
  suppressUnauthorized?: boolean;
}

export class ApiClient {
  private readonly baseUrl: string;
  private csrfToken?: string;
  private readonly unauthorizedListeners = new Set<UnauthorizedListener>();

  constructor(options: ApiClientOptions = {}) {
    this.baseUrl = (options.baseUrl ?? import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");
    this.csrfToken = options.csrfToken;
  }

  setCsrfToken(token: string | undefined): void {
    this.csrfToken = token;
  }

  /**
   * Register a central handler for authentication loss. Returns an unsubscribe
   * function. Without this every screen would have to translate its own 401
   * into "you were signed out", which is how expiry used to surface as a
   * random per-screen error.
   */
  onUnauthorized(listener: UnauthorizedListener): () => void {
    this.unauthorizedListeners.add(listener);
    return () => {
      this.unauthorizedListeners.delete(listener);
    };
  }

  private url(path: string): string {
    const normalized = path.startsWith("/") ? path : `/${path}`;
    return `${this.baseUrl}${API_PREFIX}${normalized}`;
  }

  private async request<T>(path: string, init: RequestInit & RequestInitExtras = {}): Promise<T> {
    const headers = new Headers(init.headers);
    if (!headers.has("Content-Type") && init.body) {
      headers.set("Content-Type", "application/json");
    }
    if (init.idempotencyKey) {
      headers.set("Idempotency-Key", init.idempotencyKey);
    }
    if (this.csrfToken && init.method && init.method !== "GET") {
      headers.set("X-CSRF-Token", this.csrfToken);
    }

    const {
      idempotencyKey: _idempotencyKey,
      suppressUnauthorized = false,
      ...fetchInit
    } = init;
    const response = await fetch(this.url(path), {
      ...fetchInit,
      headers,
      credentials: "include",
    });

    if (!response.ok) {
      let body: ApiError | undefined;
      try {
        body = (await response.json()) as ApiError;
      } catch {
        // ignore non-json errors
      }
      if (response.status === 401 && !suppressUnauthorized) {
        for (const listener of [...this.unauthorizedListeners]) {
          listener();
        }
      }
      throw new ApiClientError(
        body?.detail
          ? typeof body.detail === "string"
            ? body.detail
            : body.detail.map((d) => d.msg).join(", ")
          : response.statusText,
        response.status,
        body,
      );
    }

    if (response.status === 204) {
      return undefined as T;
    }

    return (await response.json()) as T;
  }

  requestMagicLink(email: string): Promise<MagicLinkResponse> {
    return this.request<RawRecord>("/auth/magic-link", {
      method: "POST",
      body: JSON.stringify({ email }),
      suppressUnauthorized: true,
    }).then(adaptMagicLink);
  }

  verifyMagicLink(token: string): Promise<VerifyResponse> {
    return this.request<VerifyResponse>("/auth/verify", {
      method: "POST",
      body: JSON.stringify({ token }),
      suppressUnauthorized: true,
    });
  }

  logout(): Promise<{ status: string }> {
    return this.request("/auth/logout", { method: "POST", suppressUnauthorized: true });
  }

  getMe(): Promise<User> {
    return this.request<User>("/me");
  }

  connectSleeper(username: string, idempotencyKey?: string): Promise<{ status: string }> {
    return this.request("/sleeper/connect", {
      method: "POST",
      body: JSON.stringify({ username }),
      idempotencyKey,
    });
  }

  triggerSync(idempotencyKey?: string): Promise<SyncResponse> {
    return this.request<RawRecord>("/sync", {
      method: "POST",
      idempotencyKey: idempotencyKey ?? `sync-${Date.now()}`,
    }).then(adaptSync);
  }

  getJob(jobId: string): Promise<import("./types").JobStatus> {
    return this.request(`/jobs/${jobId}`);
  }

  getLeagues(): Promise<LeagueSummary[]> {
    return this.request<RawRecord>("/leagues").then(adaptLeagues);
  }

  getLeague(leagueId: string): Promise<LeagueDetail> {
    return this.request<RawRecord>(`/leagues/${leagueId}`).then((raw) => {
      const league = (raw.league as RawRecord) ?? raw;
      return {
        id: String(league.league_id ?? leagueId),
        name: String(league.name),
        season: Number(league.season),
        scoring_type: String(league.type ?? "standard"),
        roster_positions: (league.raw as RawRecord)?.roster_positions as string[] ?? [],
        is_dynasty: league.type === "dynasty",
        total_rosters: 0,
        settings: (league.raw as RawRecord) ?? {},
      };
    });
  }

  getLeagueRules(leagueId: string): Promise<LeagueRules> {
    return this.request<RawRecord>(`/leagues/${leagueId}/rules`).then((raw) => ({
      league_id: leagueId,
      scoring: (raw.rules as Record<string, number>) ?? {},
      roster_slots: {},
      unsupported_keys: [],
    }));
  }

  updateDraftOrderRule(leagueId: string, rule: string, idempotencyKey?: string): Promise<{ status: string }> {
    return this.request(`/leagues/${leagueId}/draft-order-rule`, {
      method: "PUT",
      body: JSON.stringify({ rule }),
      idempotencyKey: idempotencyKey ?? `draft-rule-${Date.now()}`,
    });
  }

  getRosters(leagueId: string): Promise<import("./types").Roster[]> {
    return this.request<RawRecord>(`/leagues/${leagueId}/rosters`).then(adaptRosters);
  }

  getMatchup(leagueId: string, week: number): Promise<Matchup> {
    return this.request<RawRecord>(`/leagues/${leagueId}/matchups/${week}`).then((raw) => ({
      week,
      roster_id: Number((raw.matchups as RawRecord[])?.[0]?.roster_id ?? 0),
      opponent_roster_id: 0,
      projected_points: 0,
      win_probability: 0.5,
      starters: [],
    }));
  }

  getPlayerProjection(playerId: string): Promise<PlayerProjection> {
    return this.request<RawRecord>(`/projections/players/${playerId}`).then((raw) => ({
      player_id: playerId,
      name: String((raw.mean as RawRecord)?.name ?? playerId),
      position: String((raw.mean as RawRecord)?.position ?? "FLEX"),
      team: (raw.mean as RawRecord)?.team as string | undefined,
      points_mean: Number((raw.mean as RawRecord)?.points ?? 0),
      points_p10: Number((raw.quantiles as RawRecord)?.p10 ?? 0),
      points_p90: Number((raw.quantiles as RawRecord)?.p90 ?? 0),
      meta: {
        data_as_of: String(raw.data_as_of ?? new Date().toISOString()),
        projection_run_id: String(raw.projection_run_id ?? "fixture"),
      },
    }));
  }

  getRankings(leagueId: string, mode: RankingMode, week = 1): Promise<RankingEntry[]> {
    const query = mode === "weekly" ? `?mode=${mode}&week=${week}` : `?mode=${mode}`;
    return this.request<RawRecord>(`/leagues/${leagueId}/rankings${query}`).then((raw) =>
      adaptRankings(raw, mode),
    );
  }

  getLineup(leagueId: string, week: number, opponentMode: OpponentMode = "current"): Promise<LineupRecommendation> {
    return this.request<RawRecord>(
      `/leagues/${leagueId}/lineup/${week}?opponent_mode=${opponentMode}`,
    ).then(adaptLineup);
  }

  getWaivers(leagueId: string, week: number): Promise<WaiverRecommendation> {
    return this.request<RawRecord>(`/leagues/${leagueId}/waivers/${week}`).then(adaptWaivers);
  }

  getDraftBoard(leagueId: string): Promise<DraftBoard> {
    return this.request<RawRecord>(`/leagues/${leagueId}/draft/board`).then((raw) => {
      const board = (raw.board as RawRecord) ?? raw;
      const entries = ((board.entries as RawRecord[]) ?? (raw.entries as RawRecord[]) ?? []).map(
        (row, index) => ({
          player_id: String(row.player_id),
          name: String(row.name ?? row.player_id),
          position: String(row.position ?? "FLEX"),
          team: row.team ? String(row.team) : undefined,
          rank: Number(row.rank ?? index + 1),
          tier: row.tier != null ? Number(row.tier) : undefined,
          vorp: row.vorp != null ? Number(row.vorp) : undefined,
        }),
      );
      return {
        league_id: leagueId,
        entries,
        meta: {
          data_as_of: String(board.data_as_of ?? raw.data_as_of ?? new Date().toISOString()),
          projection_run_id: String(board.projection_run_id ?? raw.projection_run_id ?? "fixture"),
        },
      };
    });
  }

  evaluateTrade(
    leagueId: string,
    sideA: TradeSide,
    sideB: TradeSide,
    horizon: "weekly" | "ros" | "dynasty" = "ros",
    idempotencyKey?: string,
  ): Promise<TradeEvaluation> {
    return this.request<RawRecord>(`/leagues/${leagueId}/trades/evaluate`, {
      method: "POST",
      body: JSON.stringify({
        side_a: { roster_id: sideA.roster_id, player_ids: sideA.player_ids },
        side_b: { roster_id: sideB.roster_id, player_ids: sideB.player_ids },
        horizon,
      }),
      idempotencyKey: idempotencyKey ?? `trade-eval-${Date.now()}`,
    }).then(adaptTradeEvaluation);
  }

  createTradeProposal(
    leagueId: string,
    createdByRosterId: number,
    sides: RawRecord,
    idempotencyKey?: string,
  ): Promise<TradeProposal> {
    return this.request<RawRecord>(`/leagues/${leagueId}/trades/proposals`, {
      method: "POST",
      body: JSON.stringify({ created_by_roster_id: createdByRosterId, sides_json: sides }),
      idempotencyKey: idempotencyKey ?? `trade-proposal-${Date.now()}`,
    }).then((raw) => ({
      proposal_id: String(raw.proposal_id),
      status: String(raw.status) as TradeProposal["status"],
      sides: [],
    }));
  }

  getManagerTendencies(leagueId: string, rosterId: number): Promise<ManagerTendencies> {
    return this.request<RawRecord>(`/leagues/${leagueId}/managers/${rosterId}/tendencies`).then(
      adaptManagerTendencies,
    );
  }

  getDynastyState(leagueId: string, rosterId: number): Promise<RawRecord> {
    return this.request<RawRecord>(`/leagues/${leagueId}/dynasty/${rosterId}`);
  }

  postAssistant(
    leagueId: string,
    message: string,
    week = 1,
    idempotencyKey?: string,
  ): Promise<AssistantResponse> {
    return this.request<RawRecord>("/assistant/responses", {
      method: "POST",
      body: JSON.stringify({ league_id: leagueId, message, week }),
      idempotencyKey: idempotencyKey ?? `assistant-${Date.now()}`,
    }).then(adaptAssistant);
  }

  getInjuryEvidence(playerId: string): Promise<InjuryEvidence> {
    return this.request<RawRecord>(`/players/${playerId}/injury-evidence`).then(adaptInjuryEvidence);
  }

  getProjectionChanges(playerId: string): Promise<ProjectionChange[]> {
    return this.request<RawRecord>(`/players/${playerId}/projection-changes`).then(adaptProjectionChanges);
  }

  getOperationsStatus(): Promise<OperationsStatus> {
    return this.request<OperationsStatus>("/operations/status");
  }

  runOperationsJob(jobName: string, idempotencyKey?: string): Promise<{ job_id: string; status: string }> {
    return this.request(`/operations/jobs/${jobName}/run`, {
      method: "POST",
      idempotencyKey: idempotencyKey ?? `ops-job-${jobName}-${Date.now()}`,
    });
  }

  /**
   * Roll the active projection pointer back one generation.
   *
   * `season` and `week` are required: a rollback targeted at a hardcoded season
   * would silently restore the wrong pointer once the calendar moves on.
   */
  rollbackProjection(
    params: { mode: string; season: number; week: number | null },
    idempotencyKey?: string,
  ): Promise<{ status: string; active_projection_run_id?: string }> {
    const query = new URLSearchParams({
      mode: params.mode,
      season: String(params.season),
    });
    if (params.week != null) {
      query.set("week", String(params.week));
    }
    return this.request(`/operations/projections/rollback?${query}`, {
      method: "POST",
      idempotencyKey: idempotencyKey ?? `ops-rollback-${Date.now()}`,
    });
  }
}

export const api = new ApiClient();
