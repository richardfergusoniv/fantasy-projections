/** Normalize raw /api/v1 payloads into typed frontend models. */

import type {
  AssistantResponse,
  Citation,
  InjuryEvidence,
  LeagueSummary,
  LineupBoardSource,
  LineupRecommendation,
  LineupStarter,
  MagicLinkResponse,
  ManagerTendencies,
  PointsRange,
  ProjectionChange,
  RankingEntry,
  RankingMode,
  Roster,
  SyncResponse,
  TradeEvaluation,
  WaiverRecommendation,
} from "./types";

type RawRecord = Record<string, unknown>;

function metaFrom(raw: RawRecord): { data_as_of: string; projection_run_id: string } {
  const nested = raw.meta as RawRecord | undefined;
  return {
    data_as_of: String(nested?.data_as_of ?? raw.data_as_of ?? new Date().toISOString()),
    projection_run_id: String(nested?.projection_run_id ?? raw.projection_run_id ?? "fixture"),
  };
}

/**
 * Coerce to a finite number, or `null`.
 *
 * `Number(undefined)` is `NaN` and `Number({})` is `NaN`, and both used to slip
 * through as displayed values. Anything the API did not publish must reach the
 * UI as `null` so it can be rendered as "not available".
 */
function numberOrNull(value: unknown): number | null {
  if (value == null || typeof value === "boolean") {
    return null;
  }
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

/** Read a quantile map keyed either `"0.1"` (release bundle) or `"p10"`. */
function quantile(raw: RawRecord | undefined, decimal: string, alias: string): number | null {
  if (!raw) return null;
  return numberOrNull(raw[decimal] ?? raw[alias]);
}

function pointsRangeFrom(raw: RawRecord): PointsRange {
  const quantiles = raw.quantiles as RawRecord | undefined;
  return {
    p10: quantile(quantiles, "0.1", "p10"),
    p50: quantile(quantiles, "0.5", "p50"),
    p90: quantile(quantiles, "0.9", "p90"),
    mean: numberOrNull(raw.expected_points ?? raw.points),
  };
}

/** Derive a human publisher label from a source URL hostname. */
function publisherFrom(url: string, fallback: string): string {
  try {
    const host = new URL(url).hostname.replace(/^www\./, "");
    return host || fallback;
  } catch {
    return fallback;
  }
}

export function adaptLeagues(raw: RawRecord): LeagueSummary[] {
  const leagues = (raw.leagues as RawRecord[] | undefined) ?? (Array.isArray(raw) ? raw : []);
  return leagues.map((league) => ({
    id: String(league.id ?? league.league_id),
    name: String(league.name),
    season: Number(league.season),
    scoring_type: String(league.scoring_type ?? league.type ?? "standard"),
    roster_positions: (league.roster_positions as string[]) ?? [],
    is_dynasty: Boolean(league.is_dynasty ?? league.type === "dynasty"),
    is_configured: league.is_configured !== false,
    owner_roster_id:
      league.owner_roster_id == null || league.owner_roster_id === ""
        ? null
        : Number(league.owner_roster_id),
    decision_ready:
      league.decision_ready == null
        ? league.owner_roster_id != null && league.owner_roster_id !== ""
        : Boolean(league.decision_ready),
    member_count:
      league.member_count == null || league.member_count === ""
        ? undefined
        : Number(league.member_count),
  }));
}

export function adaptMagicLink(raw: RawRecord): MagicLinkResponse {
  const devLink = raw.development_link as string | undefined;
  return {
    status: devLink ? "dev_link" : "sent",
    link: devLink,
  };
}

export function adaptRosters(raw: RawRecord): Roster[] {
  const rosters = (raw.rosters as RawRecord[] | undefined) ?? [];
  return rosters.map((roster) => ({
    roster_id: Number(roster.roster_id),
    week: Number(roster.week ?? 0),
    players: ((roster.players as unknown[]) ?? []).map(String).filter(Boolean),
    starters: ((roster.starters as unknown[]) ?? []).map(String).filter(Boolean),
    reserve: ((roster.reserve as unknown[]) ?? []).map(String).filter(Boolean),
    manager_name: roster.manager_name ? String(roster.manager_name) : undefined,
    player_details: ((roster.player_details as RawRecord[] | undefined) ?? []).map((player) => ({
      player_id: String(player.player_id),
      name: String(player.name ?? player.player_id),
      position: String(player.position ?? "FLEX"),
      team: player.team ? String(player.team) : undefined,
    })),
  }));
}

function adaptLineupStarter(raw: RawRecord): LineupStarter {
  const points = pointsRangeFrom({
    expected_points: raw.expected_points ?? raw.points_mean ?? raw.points,
    quantiles: {
      "0.1": raw.points_p10,
      "0.5": raw.points_p50,
      "0.9": raw.points_p90,
      ...(typeof raw.quantiles === "object" && raw.quantiles ? (raw.quantiles as RawRecord) : {}),
    },
  });
  return {
    player_id: String(raw.player_id ?? ""),
    name: String(raw.name ?? raw.player_id ?? ""),
    position: String(raw.position ?? "FLEX"),
    team: raw.team ? String(raw.team) : undefined,
    slot: raw.slot ? String(raw.slot) : undefined,
    expected_points: points.mean,
    points_p10: points.p10,
    points_p50: points.p50,
    points_p90: points.p90,
    opponent: raw.opponent != null ? String(raw.opponent) : raw.nfl_opponent != null ? String(raw.nfl_opponent) : null,
    locked: Boolean(raw.locked),
  };
}

function adaptBoardSource(raw: RawRecord): LineupBoardSource | undefined {
  const direct = raw.board_source ?? (raw.meta as RawRecord | undefined)?.board_source;
  if (direct === "vegas_props" || direct === "league_value") {
    return direct;
  }
  const effective = String(
    raw.effective_source ?? (raw.meta as RawRecord | undefined)?.effective_source ?? "",
  );
  if (effective === "weekly_props") return "vegas_props";
  if (effective === "sealed_release" || effective === "status_adjusted_release") {
    return "league_value";
  }
  return undefined;
}

export function adaptLineup(raw: RawRecord): LineupRecommendation {
  const meta = metaFrom(raw);
  const swaps = (raw.swaps as RawRecord[] | undefined) ?? [];
  const probabilities = (raw.matchup_probabilities as Record<string, unknown> | undefined) ?? {};
  const matchupAllowed = raw.matchup_win_probability_available !== false;
  const startersRaw = (raw.starters as RawRecord[] | undefined) ?? [];
  const benchRaw = (raw.bench as RawRecord[] | undefined) ?? [];
  // Prefer enriched detail rows; fall back to empty when API only published IDs.
  const oppStarterRaw =
    (raw.opponent_starter_details as RawRecord[] | undefined) ??
    (Array.isArray(raw.opponent_starters) &&
    raw.opponent_starters.length > 0 &&
    typeof raw.opponent_starters[0] === "object"
      ? (raw.opponent_starters as RawRecord[])
      : []);
  const oppBenchRaw = (raw.opponent_bench as RawRecord[] | undefined) ?? [];
  return {
    week: Number(raw.week),
    opponent_mode: (raw.opponent_mode as LineupRecommendation["opponent_mode"]) ?? "current",
    starters: startersRaw.map(adaptLineupStarter),
    bench: benchRaw.map(adaptLineupStarter),
    opponent_starters: oppStarterRaw.map(adaptLineupStarter),
    opponent_bench: oppBenchRaw.map(adaptLineupStarter),
    swaps: swaps.map((swap) => ({
      out_player_id: String(swap.out_player_id ?? swap.drop ?? ""),
      in_player_id: String(swap.in_player_id ?? swap.add ?? ""),
      win_probability_delta: Number(swap.win_probability_delta ?? swap.win_probability_gain ?? 0),
      reason: String(swap.reason ?? `Start ${swap.add} over ${swap.drop}`),
    })),
    win_probability: matchupAllowed
      ? numberOrNull(raw.win_probability ?? probabilities.win)
      : null,
    matchup_probabilities: {
      win: numberOrNull(probabilities.win),
      tie: numberOrNull(probabilities.tie),
      loss: numberOrNull(probabilities.loss),
    },
    points: pointsRangeFrom(raw),
    opponent_expected_points: numberOrNull(raw.opponent_expected_points),
    opponent_roster_id:
      raw.opponent_roster_id == null || raw.opponent_roster_id === ""
        ? null
        : Number(raw.opponent_roster_id),
    opponent_lineup_source: raw.opponent_lineup_source
      ? String(raw.opponent_lineup_source)
      : undefined,
    contract_hash: raw.contract_hash ? String(raw.contract_hash) : undefined,
    board_source: adaptBoardSource(raw),
    effective_source: raw.effective_source
      ? String(raw.effective_source)
      : (raw.meta as RawRecord | undefined)?.effective_source
        ? String((raw.meta as RawRecord).effective_source)
        : undefined,
    meta,
  };
}

export function adaptWaivers(raw: RawRecord): WaiverRecommendation {
  const meta = metaFrom(raw);
  const adds = (raw.adds as RawRecord[] | undefined) ?? [];
  // The engine's own rationale lines and confidence live on `recommendations`;
  // `adds` is the flattened view. Join them so the UI can show why, not just what.
  const detailById = new Map<string, RawRecord>(
    ((raw.recommendations as RawRecord[] | undefined) ?? []).map((row) => [
      String(row.player_id),
      row,
    ]),
  );
  return {
    week: Number(raw.week),
    adds: adds.map((add) => {
      const playerId = String(add.player_id);
      const detail = detailById.get(playerId);
      const rationale = (detail?.rationale as unknown[] | undefined)?.map(String) ?? [];
      const reason = String(add.reason ?? rationale.join("; ") ?? "");
      return {
        player_id: playerId,
        name: String(add.name),
        position: String(add.position),
        faab_min: Number(add.faab_min ?? add.faab_low ?? 0),
        faab_max: Number(add.faab_max ?? add.faab_high ?? 0),
        reason,
        rationale: rationale.length ? rationale : reason ? reason.split("; ") : [],
        confidence: numberOrNull(detail?.confidence),
        start_probability: numberOrNull(detail?.start_probability),
        incremental_utility: numberOrNull(detail?.incremental_utility),
      };
    }),
    meta,
  };
}

export function adaptRankings(raw: RawRecord, mode: RankingMode): RankingEntry[] {
  const rankings = (raw.rankings as RawRecord[] | undefined) ?? [];
  return rankings.map((row, index) => ({
    player_id: String(row.player_id),
    name: String(row.name ?? row.player_id),
    position: String(row.position ?? "FLEX"),
    team: row.team ? String(row.team) : undefined,
    rank: index + 1,
    points_mean: Number(row.points ?? (row.mean as RawRecord)?.points ?? 0),
    points_p10: Number((row.quantiles as RawRecord)?.p10 ?? row.points ?? 0),
    points_p90: Number((row.quantiles as RawRecord)?.p90 ?? row.points ?? 0),
    mode,
  }));
}

/**
 * Pass through the API's objective / fairness / acceptance values verbatim.
 *
 * No letter grade is produced here. The API does not publish one, and a grade
 * derived from a fairness threshold in the browser is manufactured analysis.
 */
export function adaptTradeEvaluation(raw: RawRecord): TradeEvaluation {
  const objective = (raw.objective as RawRecord | undefined) ?? {};
  const fairness = (raw.fairness as RawRecord | undefined) ?? {};
  const acceptance = (raw.acceptance as RawRecord | undefined) ?? {};
  const horizon = String(raw.horizon ?? "ros");
  return {
    league_id: String(raw.league_id ?? ""),
    horizon: (["weekly", "ros", "dynasty"].includes(horizon)
      ? horizon
      : "ros") as TradeEvaluation["horizon"],
    objective: {
      side_a_value: numberOrNull(objective.side_a_value),
      side_b_value: numberOrNull(objective.side_b_value),
      side_a_gain: numberOrNull(objective.side_a_gain),
      side_b_gain: numberOrNull(objective.side_b_gain),
    },
    fairness: {
      gap: numberOrNull(fairness.gap),
      uncertainty: numberOrNull(fairness.uncertainty),
      fair: typeof fairness.fair === "boolean" ? fairness.fair : null,
    },
    acceptance: {
      side_a_probability: numberOrNull(acceptance.side_a_probability),
      side_b_probability: numberOrNull(acceptance.side_b_probability),
      tendency_adjustment: numberOrNull(acceptance.tendency_adjustment),
    },
    meta: metaFrom(raw),
  };
}

export function adaptAssistant(raw: RawRecord): AssistantResponse {
  const meta = metaFrom(raw);
  const tools = (raw.tools_called as string[] | undefined) ?? [];
  return {
    response_id: String(raw.response_id ?? crypto.randomUUID()),
    messages: [
      {
        role: "assistant",
        content: String(raw.message ?? "No response"),
      },
    ],
    tool_calls: tools.map((name) => ({ name, result_id: name })),
    meta,
  };
}

export function adaptSync(raw: RawRecord): SyncResponse {
  return { job_id: String(raw.job_id ?? "") };
}

export function adaptManagerTendencies(raw: RawRecord): ManagerTendencies {
  const features = (raw.features as RawRecord) ?? {};
  return {
    roster_id: Number(raw.roster_id),
    trade_frequency: Number(features.avg_package_size ?? 0),
    win_now_bias: Number(features.youth_preference ?? 0.5),
    pick_premium: Number(features.pick_preference ?? 0.5),
    position_preferences: {
      consolidation: Number(features.consolidation_bias ?? 0.5),
      accept_rate: Number(features.accept_rate ?? 0.5),
    },
  };
}

export function adaptInjuryEvidence(raw: RawRecord): InjuryEvidence {
  const meta = metaFrom(raw);
  const evidence = (raw.evidence as RawRecord[] | undefined) ?? [];
  const first = evidence[0];
  const claim = (first?.claim as RawRecord) ?? {};
  const sources: Citation[] = evidence
    .map((row) => {
      const url = String(row.source_url ?? "");
      const title = String(row.source_title ?? "Source");
      return {
        title,
        url,
        publisher: publisherFrom(url, title),
        published_at: row.published_at ? String(row.published_at) : undefined,
        confidence: numberOrNull(row.confidence),
      };
    })
    .filter((source) => Boolean(source.url));
  return {
    player_id: String(raw.player_id),
    status: String(claim.status ?? "unknown"),
    summary: String(claim.reported_injury ?? first?.source_title ?? "No evidence"),
    sources,
    meta,
  };
}

export function adaptProjectionChanges(raw: RawRecord): ProjectionChange[] {
  const changes = (raw.changes as RawRecord[] | undefined) ?? [];
  return changes.map((change) => ({
    player_id: String(change.player_id),
    from_run_id: String(change.from_run_id),
    to_run_id: String(change.to_run_id),
    delta_points: Number(change.delta_points),
    drivers: (change.drivers as string[]) ?? [],
    meta: metaFrom(change),
  }));
}
