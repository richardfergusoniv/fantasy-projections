import type { Citation } from "../api/types";

function formatPublished(iso?: string): string | null {
  if (!iso) return null;
  const parsed = Date.parse(iso);
  if (Number.isNaN(parsed)) return null;
  return new Date(parsed).toLocaleDateString();
}

/**
 * Render source citations for a claim.
 *
 * Every link opens in a new tab and carries `rel="noopener noreferrer"`:
 * `noopener` keeps the opened page from reaching back through `window.opener`,
 * and `noreferrer` keeps our URL out of the third party's referrer log.
 */
export function CitationList({
  citations,
  label,
  emptyMessage = "No sources published for this claim.",
}: {
  citations: Citation[];
  label: string;
  emptyMessage?: string;
}) {
  if (citations.length === 0) {
    return <p className="citation-empty">{emptyMessage}</p>;
  }

  return (
    <ul className="citation-list" aria-label={label}>
      {citations.map((citation) => {
        const published = formatPublished(citation.published_at);
        return (
          <li key={`${citation.url}-${citation.title}`} className="citation">
            <a href={citation.url} target="_blank" rel="noopener noreferrer">
              {citation.title}
            </a>
            <span className="citation-meta">
              <span className="citation-publisher">{citation.publisher}</span>
              {published ? <span className="citation-date"> · {published}</span> : null}
              {citation.confidence != null ? (
                <span className="citation-confidence">
                  {" "}
                  · confidence {(citation.confidence * 100).toFixed(0)}%
                </span>
              ) : null}
              <span className="visually-hidden"> (opens in a new tab)</span>
            </span>
          </li>
        );
      })}
    </ul>
  );
}
