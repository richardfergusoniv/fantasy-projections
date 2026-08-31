import { useState } from "react";
import { Panel } from "../components/Panel";
import { useAppState } from "../hooks/useAppState";
import { api } from "../api/client";

export function AssistantScreen() {
  const { selectedLeagueId, week, availableWeeks } = useAppState();
  const [message, setMessage] = useState("");
  const [response, setResponse] = useState<string | null>(null);
  const [tools, setTools] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canAsk = Boolean(selectedLeagueId) && week != null && Boolean(message.trim());

  async function onSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (!selectedLeagueId || week == null || !message.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const result = await api.postAssistant(selectedLeagueId, message, week);
      setResponse(result.messages[0]?.content ?? "No response");
      setTools(result.tool_calls?.map((call) => call.name) ?? []);
    } catch (err) {
      setResponse(null);
      setError(err instanceof Error ? err.message : "Assistant request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="screen">
      <Panel title="Assistant">
        <p className="muted">
          League-aware chat backed by deterministic tools. Projections and injury claims always come
          from app data, never invented stats.
        </p>
        <p className="muted">
          Asking about week {week ?? "— no week available"}
          {availableWeeks.length ? ` (synced weeks: ${availableWeeks.join(", ")})` : ""}. Change the
          week in the header.
        </p>
        <form className="stack" onSubmit={onSubmit}>
          <div className="field">
            <label htmlFor="assistant-message">Message</label>
            <textarea
              id="assistant-message"
              name="message"
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              placeholder="Who should I start at RB this week?"
              rows={3}
            />
          </div>
          <button className="btn btn-primary" type="submit" disabled={loading || !canAsk}>
            {loading ? "Thinking…" : "Ask assistant"}
          </button>
        </form>
        <div role="status" aria-live="polite">
          {error ? (
            <p className="state-notice state-error">
              <span className="state-glyph" aria-hidden="true">
                ✕
              </span>
              <span>
                <strong className="state-title">Assistant unavailable.</strong> {error}
              </span>
            </p>
          ) : null}
          {loading ? <p className="muted">Waiting for the assistant…</p> : null}
        </div>
        {response ? (
          <div className="assistant-thread">
            <p>{response}</p>
            {tools.length ? <p className="muted">Tools: {tools.join(", ")}</p> : null}
          </div>
        ) : (
          <div className="assistant-thread empty-state">
            Ask about lineup swaps, waiver targets, or trade evaluations.
          </div>
        )}
      </Panel>
    </div>
  );
}
