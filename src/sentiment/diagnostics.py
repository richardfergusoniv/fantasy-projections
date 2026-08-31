"""Diagnostic sentiment labels for release artifacts and UI."""
from __future__ import annotations

import pandas as pd


TONE_BULLISH = "bullish"
TONE_MIXED_NEUTRAL = "mixed_neutral"
TONE_BEARISH = "bearish"
TONE_UNAVAILABLE = "unavailable"

PEER_ABOVE = "above_peers"
PEER_TYPICAL = "typical"
PEER_BELOW = "below_peers"

EVIDENCE_LEGACY_PLUS_MARKET = "legacy_plus_market"
EVIDENCE_LEGACY_ONLY = "legacy_only"
EVIDENCE_MARKET_ONLY = "market_only"
EVIDENCE_NONE = "none"

PEER_SCORE_ABOVE = 15
PEER_SCORE_BELOW = -15


def sentiment_tone_from_raw(raw: float | None) -> str:
    """Absolute interpretation of the strongest legacy text mention."""
    if raw is None or pd.isna(raw):
        return TONE_UNAVAILABLE
    value = float(raw)
    if value > 0:
        return TONE_BULLISH
    if value < 0:
        return TONE_BEARISH
    return TONE_MIXED_NEUTRAL


def sentiment_peer_label_from_score(score: float | None) -> str | None:
    """Position-relative residual expressed as peer comparison."""
    if score is None or pd.isna(score):
        return None
    value = float(score)
    if value > PEER_SCORE_ABOVE:
        return PEER_ABOVE
    if value < PEER_SCORE_BELOW:
        return PEER_BELOW
    return PEER_TYPICAL


def sentiment_evidence_tier(
    *,
    has_text: bool,
    has_market: bool,
) -> str:
    """Honest evidence tier replacing numeric confidence language."""
    if has_text and has_market:
        return EVIDENCE_LEGACY_PLUS_MARKET
    if has_text:
        return EVIDENCE_LEGACY_ONLY
    if has_market:
        return EVIDENCE_MARKET_ONLY
    return EVIDENCE_NONE


def attach_diagnostic_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Add tone, peer label, and evidence tier columns to a sentiment snapshot."""
    out = frame.copy()
    raw = pd.to_numeric(out.get("text_sentiment_raw"), errors="coerce")
    out["sentiment_tone"] = raw.map(sentiment_tone_from_raw)

    score = pd.to_numeric(out.get("sentiment_score"), errors="coerce")
    out["sentiment_peer_label"] = score.map(sentiment_peer_label_from_score)

    has_text = out.get("text_sentiment_z", pd.Series(index=out.index)).notna()
    has_market = out.get("market_gap_z", pd.Series(index=out.index)).notna()
    out["sentiment_evidence_tier"] = [
        sentiment_evidence_tier(has_text=bool(t), has_market=bool(m))
        for t, m in zip(has_text, has_market)
    ]
    return out
