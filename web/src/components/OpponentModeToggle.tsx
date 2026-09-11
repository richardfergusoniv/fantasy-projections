import type { OpponentMode } from "../api/types";

interface ModeOption {
  value: OpponentMode;
  label: string;
  shortLabel: string;
  description: string;
}

export const OPPONENT_MODES: ModeOption[] = [
  {
    value: "current",
    label: "Opponent's current lineup",
    shortLabel: "Their lineup",
    description:
      "Score the matchup against the starters your opponent has set right now. Use this to decide today.",
  },
  {
    value: "optimized",
    label: "Opponent's best possible lineup",
    shortLabel: "Best possible",
    description:
      "Assume your opponent starts their optimal lineup. A worst-case check, usually a lower win probability.",
  },
];

/**
 * Matchup assumption control.
 *
 * `compact` renders a Sleeper-style segmented control for the Lineup screen;
 * the default keeps the descriptive radio group for denser surfaces.
 */
export function OpponentModeToggle({
  value,
  onChange,
  disabled = false,
  compact = false,
}: {
  value: OpponentMode;
  onChange: (mode: OpponentMode) => void;
  disabled?: boolean;
  compact?: boolean;
}) {
  if (compact) {
    return (
      <div className="lineup-mode-seg" role="group" aria-label="Matchup assumption">
        {OPPONENT_MODES.map((option) => {
          const active = value === option.value;
          return (
            <button
              key={option.value}
              type="button"
              className={`lineup-mode-seg-btn${active ? " is-active" : ""}`}
              aria-pressed={active}
              title={option.description}
              disabled={disabled}
              onClick={() => onChange(option.value)}
            >
              {option.shortLabel}
            </button>
          );
        })}
      </div>
    );
  }

  return (
    <fieldset className="mode-toggle" disabled={disabled}>
      <legend id="opponent-mode-legend">Matchup assumption</legend>
      <div className="mode-options" role="radiogroup" aria-labelledby="opponent-mode-legend">
        {OPPONENT_MODES.map((option) => {
          const id = `opponent-mode-${option.value}`;
          return (
            <div key={option.value} className="mode-option">
              <input
                type="radio"
                id={id}
                name="opponent-mode"
                value={option.value}
                checked={value === option.value}
                onChange={() => onChange(option.value)}
                aria-describedby={`${id}-description`}
              />
              <label htmlFor={id}>{option.label}</label>
              <p id={`${id}-description`} className="mode-description">
                {option.description}
              </p>
            </div>
          );
        })}
      </div>
    </fieldset>
  );
}
