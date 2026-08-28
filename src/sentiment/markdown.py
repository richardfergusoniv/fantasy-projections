"""Deterministic parser for the 32 local Perplexity sentiment summaries."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import re
import unicodedata

import pandas as pd


RESEARCH_AS_OF = date(2026, 8, 24)

# Explicit mapping is safer than deriving team identity from heterogeneous
# titles (49ers, Washington Commanders, both Los Angeles clubs, etc.).
TEAM_RESEARCH_FILES = {
    "ARI": "arizona_cardinals_fantasy_sentiment_summary_2026-08-24.md",
    "ATL": "falcons_player_sentiment_2026.md",
    "BAL": "ravens-player-sentiment-feb-17-2026.md",
    "BUF": "bills_player_sentiment_since_2026-02-17.md",
    "CAR": "carolina_panthers_fantasy_sentiment_summary.md",
    "CHI": "chicago_bears_player_sentiment_since_2026-02-17.md",
    "CIN": "bengals_player_sentiment_since_2026-02-17.md",
    "CLE": "cleveland_browns_fantasy_sentiment_2026.md",
    "DAL": "dallas_cowboys_fantasy_sentiment_summary.md",
    "DEN": "Fantasy_Research_Denver_Broncos_Sentiment_2026-08-24.md",
    "DET": "detroit_lions_fantasy_sentiment_summary.md",
    "GB": "packers_player_sentiment_summary.md",
    "HOU": "texans_fantasy_sentiment_summary_2026-08-24.md",
    "IND": "colts_player_sentiment_summary.md",
    "JAX": "jacksonville_jaguars_player_sentiment_summary.md",
    "KC": "chiefs_player_sentiment_summary_2026-08-24.md",
    "LA": "los-angeles-rams-player-sentiment-feb-17-2026.md",
    "LAC": "Chargers_Player_Sentiment_Analysis.md",
    "LV": "raiders_player_sentiment_summary_2026.md",
    "MIA": "miami_dolphins_player_sentiment_feb17_2026.md",
    "MIN": "minnesota_vikings_player_sentiment_2026.md",
    "NE": "Fantasy_Research_Patriots_Sentiment_Summary.md",
    "NO": "saints_fantasy_sentiment_update_2026-08-24.md",
    "NYG": "Giants_player_sentiment_update_2026-08-24.md",
    "NYJ": "jets_fantasy_sentiment_summary.md",
    "PHI": "philadelphia-eagles-player-sentiment-2026-08-24.md",
    "PIT": "steelers-player-sentiment-2026.md",
    "SEA": "Seattle_Seahawks_Coachspeak_Sentiment_Feb17-Aug23_2026.md",
    "SF": "49ers_fantasy_sentiment_summary_2026-08-24.md",
    "TB": "tampa_bay_buccaneers_fantasy_sentiment_2026.md",
    "TEN": "tennessee-titans-fantasy-sentiment-feb-aug-2026.md",
    "WAS": "Washington_Commanders_Player_Sentiment_2026-02-17_to_2026-08-24.md",
}


def norm_name(value: str | None) -> str:
    """Normalize display names while preserving enough text for exact matching."""
    if not value:
        return ""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    text = text.lower().replace(".", "").replace("'", "")
    text = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class Candidate:
    player_text: str
    label: str
    context: str
    heading: str
    method: str
    line_number: int


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _table_candidates(lines: list[str]) -> list[Candidate]:
    out: list[Candidate] = []
    i = 0
    while i + 1 < len(lines):
        if not lines[i].lstrip().startswith("|") or not lines[i + 1].lstrip().startswith("|"):
            i += 1
            continue
        header = _cells(lines[i])
        sep = _cells(lines[i + 1])
        if not sep or not all(re.fullmatch(r":?-{3,}:?", c.replace(" ", "")) for c in sep):
            i += 1
            continue
        lowered = [c.lower() for c in header]
        if "player" not in lowered:
            i += 2
            continue
        player_idx = lowered.index("player")
        sentiment_idx = next(
            (idx for idx, col in enumerate(lowered) if "sentiment" in col), None
        )
        # Availability/issue-only tables are controls, not sentiment.
        if sentiment_idx is None:
            i += 2
            continue
        heading = ""
        for prior in reversed(lines[:i]):
            if prior.startswith("#"):
                heading = prior.lstrip("# ").strip()
                break
        j = i + 2
        while j < len(lines) and lines[j].lstrip().startswith("|"):
            row = _cells(lines[j])
            if len(row) > max(player_idx, sentiment_idx):
                context = " | ".join(row)
                out.append(
                    Candidate(
                        player_text=row[player_idx],
                        label=row[sentiment_idx],
                        context=context,
                        heading=heading,
                        method="table",
                        line_number=j + 1,
                    )
                )
            j += 1
        i = j
    return out


def _bullet_candidates(lines: list[str]) -> list[Candidate]:
    out: list[Candidate] = []
    heading = ""
    for idx, line in enumerate(lines):
        if line.startswith("#"):
            heading = line.lstrip("# ").strip()
            continue
        if not re.match(r"^\s*(?:[-*]|\d+\.)\s+", line):
            continue
        bold = re.findall(r"\*\*(.+?)\*\*", line)
        if bold:
            bold_text = bold[0].strip().strip(":")
            # Several summaries use **Player — sentiment:** inside one bold
            # span. Split that into the identity and the explicit label.
            parts = re.split(r"\s+[—–]\s+", bold_text, maxsplit=1)
            player_text = parts[0].strip()
            inline_label = parts[1].strip() if len(parts) == 2 else ""
            remainder = line.split("**", 2)[-1].lstrip(" :—-")
        else:
            # Other summaries use ordinary bullets: "- Player: assessment".
            plain = re.sub(r"^\s*(?:[-*]|\d+\.)\s+", "", line).strip()
            if ":" not in plain:
                continue
            player_text, remainder = plain.split(":", 1)
            inline_label = ""
            if len(player_text.split()) > 6:
                continue
        first_clause = re.split(r"[.;]", remainder, maxsplit=1)[0].strip()
        label = inline_label or first_clause or heading
        out.append(
            Candidate(
                player_text=player_text,
                label=label,
                context=line.strip(),
                heading=heading,
                method="bullet",
                line_number=idx + 1,
            )
        )
    return out


NO_SIGNAL = (
    "no recent material signal",
    "no material fresh sentiment",
    "no direct player-specific commentary",
    "no current direct usage",
    "no meaningful role-specific sentiment",
    "no substantive post",
    "neutral / absent",
)

OBJECTIVE_TERMS = (
    "acl",
    "injury",
    "injured",
    "hamstring",
    "knee",
    "thumb",
    "medical",
    "health",
    "recovery",
    "rehab",
    "absence",
    "pup",
    "ir",
    "waived",
    "released",
    "season-ending",
    "out for the season",
    "no longer on",
    "ruled out",
)


def score_sentiment(label: str, heading: str = "", context: str = "") -> float | None:
    """Map an explicit qualitative label to [-1, 1].

    Positive ability plus an injury caveat remains positive; injury-only
    negative labels are excluded because availability is already modeled.

    The section heading plays two distinct roles and only one of them was wrong.
    It legitimately *supplies* polarity for a row carrying no label of its own,
    and it legitimately *qualifies* a row that has one -- "## Positive, but
    conditional" really does temper every row beneath it.  What it must not do
    is *override* an explicit label: scoring ``label + heading`` as one string
    let a heading such as "## Top positive signals" match the positive tier and
    return before the negative branch was reached, scoring an explicit
    "Strongly bearish" row at +0.55.

    So polarity is read from the label alone, falling back to the heading only
    when the label carries no signal, while the dampening qualifiers are still
    read from label and heading together.  Bullets already resolve the heading
    as a fallback when they build their label; see ``_bullet_candidates``.
    """
    score = _score_label(label, context)
    if score is None and heading:
        score = _score_label(f"{label} {heading}", context)
    if score is not None and score > 0 and heading:
        score = _damp_positive(score, f"{label} {heading}")
    return score


def _normalize(text: str) -> str:
    return text.lower().replace("–", "-").replace("—", "-")


def _damp_positive(score: float, text: str) -> float:
    """Temper a positive reading with the caveats attached to it."""
    primary = _normalize(text)
    if any(term in primary for term in ("mild", "cautious", "mixed", "conditional", "developmental", "role-dependent", "role-capped", "volatile")):
        return min(score, 0.3)
    if any(term in primary for term in ("neutral-to-positive", "neutral to positive", "neutral-positive")):
        return 0.15
    return score


def _score_label(text: str, context: str) -> float | None:
    primary = _normalize(text)
    all_text = f"{primary} {context.lower()}"
    if any(term in all_text for term in NO_SIGNAL):
        return None

    positive_strengths = (
        (1.0, ("elite positive", "elite sentiment", "extremely positive", "exceptional positive", "very bullish", "very positive", "strongly bullish", "strongly positive")),
        (0.8, ("strong positive", "strong bullish", "strongly rising", "very strong positive", "very strong")),
        (0.55, ("bullish", "positive", "rising", "improving", "upward", "secure")),
    )
    positive_score = next(
        (score for score, terms in positive_strengths if any(term in primary for term in terms)),
        None,
    )
    if positive_score is not None:
        return _damp_positive(positive_score, primary)

    negative_score = None
    if any(term in primary for term in ("strongly bearish", "strongly negative", "extremely negative")):
        negative_score = -0.9
    elif any(term in primary for term in ("bearish", "negative", "discount")):
        negative_score = -0.55
    if negative_score is not None:
        # Do not relabel an objective transaction/medical event as opinion.
        if any(term in all_text for term in OBJECTIVE_TERMS):
            return None
        return negative_score

    if any(term in primary for term in ("neutral", "mixed", "unclear", "unresolved")):
        return 0.0
    return None


def _candidate_matches(candidate: Candidate, player_name: str) -> bool:
    candidate_name = norm_name(candidate.player_text)
    player = norm_name(player_name)
    if not player or not candidate_name:
        return False
    if candidate_name == player:
        return True
    # Multi-player bullets/tables use "A / B" or "A and B".
    return bool(re.search(rf"(?:^|\s){re.escape(player)}(?:$|\s)", candidate_name))


def parse_team_research(
    players: pd.DataFrame,
    team: str,
    path: Path,
) -> list[dict]:
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    candidates = _table_candidates(lines) + _bullet_candidates(lines)
    rows: list[dict] = []
    for player in players.itertuples(index=False):
        matches = [c for c in candidates if _candidate_matches(c, player.display_name)]
        scored: list[tuple[Candidate, float]] = []
        for candidate in matches:
            score = score_sentiment(candidate.label, candidate.heading, candidate.context)
            if score is not None:
                scored.append((candidate, score))
        if not scored:
            continue
        # Prefer an explicit table label, then the strongest-confidence bullet.
        scored.sort(key=lambda item: (item[0].method == "table", abs(item[1])), reverse=True)
        selected, score = scored[0]
        confidence = 0.75 if selected.method == "table" else 0.62
        rows.append(
            {
                "player_id": str(player.player_id),
                "display_name": player.display_name,
                "team": team,
                "position": player.position,
                "text_sentiment_raw": float(score),
                "text_confidence": confidence,
                "sentiment_label": selected.label,
                "sentiment_claim_count": len(scored),
                "sentiment_source_count": 1,
                "sentiment_source_ref": f"{path.name}:{selected.line_number}",
                "sentiment_parse_method": selected.method,
            }
        )
    return rows


def parse_research_directory(
    players: pd.DataFrame,
    research_dir: str | Path,
    *,
    as_of: date,
) -> pd.DataFrame:
    """Parse all team files at or after their shared research cutoff."""
    columns = [
        "player_id", "display_name", "team", "position", "text_sentiment_raw",
        "text_confidence", "sentiment_label", "sentiment_claim_count",
        "sentiment_source_count", "sentiment_source_ref", "sentiment_parse_method",
    ]
    if as_of < RESEARCH_AS_OF:
        return pd.DataFrame(columns=columns)
    root = Path(research_dir)
    missing = [name for name in TEAM_RESEARCH_FILES.values() if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing sentiment research files: {missing}")
    rows: list[dict] = []
    for team, filename in TEAM_RESEARCH_FILES.items():
        team_players = players[players["team"].eq(team)]
        rows.extend(parse_team_research(team_players, team, root / filename))
    return pd.DataFrame(rows, columns=columns)
