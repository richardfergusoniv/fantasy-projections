import type { OpponentMode } from "../api/types";

interface ModeOption {
  value: OpponentMode;
  label: string;
  description: string;
}

export const OPPONENT_MODES: ModeOption[] = [
  {
    value: "current",
    label: "Opponent's current lineup",
    description:
      "Score the matchup against the starters your opponent has set right now. Use this to decide today.",
  },
  {
    value: "optimized",
    label: "Opponent's best possible lineup",
    description:
      "Assume your opponent starts their optimal lineup. A worst-case check, usually a lower win probability.",
  },
];

/**
 * Radio group for the matchup assumption.
 *
 * A radio group rather than a `<select>`: both options and the difference
 * between them stay visible, arrow keys move between them, and each option
 * carries its own description instead of hiding the semantics behind a label.
 */
export function OpponentModeToggle({
  value,
  onChange,
  disabled = false,
}: {
  value: OpponentMode;
  onChange: (mode: OpponentMode) => void;
  disabled?: boolean;
}) {
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
