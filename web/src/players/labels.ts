import type { Roster, RosterPlayer } from "../api/types";

type PlayerLabelSource = Pick<RosterPlayer, "player_id" | "name" | "position"> & {
  sleeper_id?: string;
  gsis_id?: string;
};

function trim(value: string | undefined): string {
  return (value ?? "").trim();
}

/** True when `name` is just echoing an identifier instead of a display name. */
export function isUnresolvedPlayerName(name: string, ...ids: Array<string | undefined>): boolean {
  const resolved = trim(name);
  if (!resolved) return true;
  return ids.some((id) => id != null && trim(id) === resolved);
}

export function formatRosterPlayerLabel(
  rosterPlayerId: string,
  detail?: PlayerLabelSource,
): string {
  const name = trim(detail?.name);
  if (!isUnresolvedPlayerName(name, rosterPlayerId, detail?.player_id)) {
    const position = trim(detail?.position);
    return position ? `${name} (${position})` : name;
  }
  return rosterPlayerId;
}

function indexDetails(details: PlayerLabelSource[]): Map<string, PlayerLabelSource> {
  const byId = new Map<string, PlayerLabelSource>();
  for (const detail of details) {
    byId.set(detail.player_id, detail);
    if (detail.sleeper_id) byId.set(detail.sleeper_id, detail);
    if (detail.gsis_id) byId.set(detail.gsis_id, detail);
  }
  return byId;
}

/**
 * Old `GET /leagues/{id}/rosters` remapped `player_details.player_id` onto GSIS
 * and never published `sleeper_id` / `gsis_id`. The fixed server always emits
 * those aliases. Index-joining is only for that legacy payload.
 */
function isLegacyRemappedDetails(details: PlayerLabelSource[]): boolean {
  return details.length > 0 && details.every((detail) => !trim(detail.sleeper_id) && !trim(detail.gsis_id));
}

/**
 * Server builds `player_details` as
 * `[player_label(pid) for pid in (r.players or []) if pid]`, so order matches
 * `players` when lengths match. If a detail's `player_id` is a roster slot at
 * another index, that contract is already broken — do not guess by position.
 * A wrong name in Trade Lab is worse than a raw id.
 */
function detailsFollowPlayersOrder(players: string[], details: PlayerLabelSource[]): boolean {
  const indexByPlayer = new Map(players.map((id, index) => [id, index]));
  for (const [index, detail] of details.entries()) {
    const rosterIndex = indexByPlayer.get(detail.player_id);
    if (rosterIndex != null && rosterIndex !== index) {
      return false;
    }
  }
  return true;
}

function shouldJoinByIndex(players: string[], details: PlayerLabelSource[]): boolean {
  return (
    isLegacyRemappedDetails(details) &&
    details.length === players.length &&
    detailsFollowPlayersOrder(players, details)
  );
}

/**
 * Map every rostered player id onto "Name (POS)".
 *
 * Prefer identifier joins (`player_id`, `sleeper_id`, `gsis_id`). Fall back to
 * index only for the legacy remapped payload, and only when lengths match and
 * no detail's `player_id` sits on a different roster slot.
 */
export function buildRosterPlayerLabels(rosters: Roster[]): Map<string, string> {
  const labels = new Map<string, string>();
  for (const roster of rosters) {
    const details = roster.player_details ?? [];
    const byId = indexDetails(details);
    const joinByIndex = shouldJoinByIndex(roster.players, details);

    for (const [index, playerId] of roster.players.entries()) {
      const detail = byId.get(playerId) ?? (joinByIndex ? details[index] : undefined);
      labels.set(playerId, formatRosterPlayerLabel(playerId, detail));
    }

    for (const detail of details) {
      const label = formatRosterPlayerLabel(detail.player_id, detail);
      for (const id of [detail.player_id, detail.sleeper_id, detail.gsis_id]) {
        if (id && !labels.has(id)) {
          labels.set(id, label);
        }
      }
    }
  }
  return labels;
}
