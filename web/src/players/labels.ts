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
 * Map every rostered player id onto "Name (POS)".
 *
 * `GET /leagues/{id}/rosters` historically remapped `player_details.player_id`
 * onto GSIS while `players` kept the original Sleeper id. Join by every known
 * identifier and, when the arrays are aligned, by index so Trade Lab still
 * resolves names against that payload.
 */
export function buildRosterPlayerLabels(rosters: Roster[]): Map<string, string> {
  const labels = new Map<string, string>();
  for (const roster of rosters) {
    const details = roster.player_details ?? [];
    const byId = indexDetails(details);
    const aligned = details.length === roster.players.length;

    for (const [index, playerId] of roster.players.entries()) {
      const detail = byId.get(playerId) ?? (aligned ? details[index] : undefined);
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
