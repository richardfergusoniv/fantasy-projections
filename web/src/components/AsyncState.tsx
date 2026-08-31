/**
 * One shared vocabulary for the six async conditions every read-only screen can
 * be in. They are deliberately distinct: "offline", "stale", and "failed" are
 * different problems with different user actions, so collapsing them into one
 * generic message hides the only information that would help.
 */

export type AsyncStateKind =
  | "loading"
  | "offline"
  | "stale"
  | "error"
  | "empty"
  | "partial";

export interface AsyncNotice {
  kind: AsyncStateKind;
  /** Short label used as the visible tag, e.g. "Offline". */
  title: string;
  /** Full sentence explaining the condition and what the user is looking at. */
  detail: string;
  /** True when the condition warrants an assertive announcement. */
  assertive: boolean;
}

export interface AsyncStateInput {
  /** A request is in flight right now. */
  loading: boolean;
  /** The browser reports no network. */
  offline: boolean;
  /** Last request failed; message is shown verbatim. */
  error: string | null;
  /** True when the rendered payload came from the offline cache. */
  fromCache: boolean;
  /** ISO timestamp of the cached copy, if any. */
  cachedAt?: string | null;
  /** ISO `meta.data_as_of` of the rendered payload. */
  dataAsOf?: string | null;
  /** True when there is a payload to render. */
  hasData: boolean;
  /** True when a payload exists but contains no rows. */
  isEmpty: boolean;
  /** Human description of what is missing when the payload is incomplete. */
  missing?: string[];
  /** Overrides the default empty message with something screen-specific. */
  emptyMessage?: string;
}

/** Data older than this is called out as stale even on a successful fetch. */
export const STALE_AFTER_MS = 12 * 60 * 60 * 1000;

function ageMs(iso: string | null | undefined): number | null {
  if (!iso) return null;
  const parsed = Date.parse(iso);
  if (Number.isNaN(parsed)) return null;
  return Date.now() - parsed;
}

function humanizeAge(ms: number): string {
  const minutes = Math.round(ms / 60_000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes} minute${minutes === 1 ? "" : "s"} ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 48) return `${hours} hour${hours === 1 ? "" : "s"} ago`;
  const days = Math.round(hours / 24);
  return `${days} day${days === 1 ? "" : "s"} ago`;
}

function formatWhen(iso: string | null | undefined): string {
  if (!iso) return "an unknown time";
  const age = ageMs(iso);
  return age == null ? "an unknown time" : humanizeAge(age);
}

/**
 * Translate raw hook state into the notices to render, most urgent first.
 *
 * Multiple notices can apply at once (offline *and* stale, for example) and all
 * of them are returned; the caller renders them as separate rows.
 */
export function describeAsyncState(input: AsyncStateInput): AsyncNotice[] {
  const notices: AsyncNotice[] = [];

  if (input.offline) {
    notices.push({
      kind: "offline",
      title: "Offline",
      detail: input.hasData
        ? `No network. Showing the copy saved on this device from ${formatWhen(input.cachedAt)}. Values will not update until you reconnect.`
        : "No network, and nothing is saved on this device for this view yet. Reconnect to load it.",
      assertive: false,
    });
  } else if (input.error && input.hasData) {
    notices.push({
      kind: "stale",
      title: "Refresh failed",
      detail: `Could not reach the server (${input.error}). Showing the copy saved from ${formatWhen(input.cachedAt)} instead of live data.`,
      assertive: true,
    });
  } else if (input.error) {
    notices.push({
      kind: "error",
      title: "Could not load",
      detail: input.error,
      assertive: true,
    });
  }

  if (input.loading) {
    notices.push({
      kind: "loading",
      title: "Loading",
      detail: input.hasData
        ? "Refreshing from the server…"
        : "Loading from the server…",
      assertive: false,
    });
  }

  if (!input.offline && !input.error && input.hasData) {
    const age = ageMs(input.dataAsOf);
    if (input.fromCache) {
      notices.push({
        kind: "stale",
        title: "Saved copy",
        detail: `Showing the copy saved on this device from ${formatWhen(input.cachedAt)}. Not yet confirmed against the server.`,
        assertive: false,
      });
    } else if (age != null && age > STALE_AFTER_MS) {
      notices.push({
        kind: "stale",
        title: "Stale data",
        detail: `The projection release backing this view is from ${humanizeAge(age)}. Run a refresh from Operations to publish newer numbers.`,
        assertive: false,
      });
    }
  }

  if (input.missing?.length) {
    notices.push({
      kind: "partial",
      title: "Partial data",
      detail: `Loaded, but the API did not publish: ${input.missing.join(", ")}. Those fields show "not available" below.`,
      assertive: false,
    });
  }

  if (!input.loading && !input.error && (!input.hasData || input.isEmpty)) {
    notices.push({
      kind: "empty",
      title: "Nothing to show",
      detail:
        input.emptyMessage ??
        "The API returned no rows for this selection. Try another week or run a sync.",
      assertive: false,
    });
  }

  return notices;
}

const GLYPHS: Record<AsyncStateKind, string> = {
  loading: "◌",
  offline: "⚡",
  stale: "◔",
  error: "✕",
  empty: "∅",
  partial: "◑",
};

export function AsyncStateBanner({
  label,
  onRetry,
  ...input
}: AsyncStateInput & { label: string; onRetry?: () => void }) {
  const notices = describeAsyncState(input);
  if (notices.length === 0) {
    return null;
  }

  const assertive = notices.some((notice) => notice.assertive);
  return (
    <div
      className="async-state"
      role="status"
      aria-live={assertive ? "assertive" : "polite"}
      aria-busy={input.loading}
      aria-label={`${label} status`}
    >
      {notices.map((notice) => (
        <p key={notice.kind + notice.title} className={`state-notice state-${notice.kind}`}>
          <span className="state-glyph" aria-hidden="true">
            {GLYPHS[notice.kind]}
          </span>
          <span>
            <strong className="state-title">{notice.title}.</strong> {notice.detail}
          </span>
          {onRetry && (notice.kind === "error" || notice.kind === "stale") ? (
            <button className="btn btn-inline" type="button" onClick={onRetry}>
              Retry
            </button>
          ) : null}
        </p>
      ))}
    </div>
  );
}
