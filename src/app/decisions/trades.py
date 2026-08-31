"""Trade evaluation: objective utility, fairness, and bounded acceptance.

Three outputs are kept strictly separate so that manager psychology can never
masquerade as roster value:

1. ``objective`` — the change in roster utility for each side, in projected
   fantasy points over the requested horizon, plus roster-construction effects.
2. ``fairness`` — the size of the objective gap and its uncertainty.
3. ``acceptance`` — a bounded probability in which objective benefit supplies
   75–90% of the signal and manager tendency at most 10–25%. Tendency can never
   flip a materially harmful trade into a recommended one.

Future draft picks are only tradeable in dynasty leagues. Redraft leagues reject
them with a precise error rather than silently valuing them at zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Acceptance weighting bounds mandated by the blueprint.
MIN_OBJECTIVE_WEIGHT = 0.75
MAX_OBJECTIVE_WEIGHT = 0.90

#: A trade this far below neutral (in horizon points) is materially harmful.
MATERIAL_HARM_POINTS = 8.0


class RedraftPickNotTradeable(ValueError):
    """Raised when a redraft trade includes a future draft pick."""


@dataclass
class PickAsset:
    season: int
    round: int
    original_roster_id: int | None = None

    @classmethod
    def parse(cls, payload: dict) -> PickAsset:
        try:
            season = int(payload["season"])
            round_ = int(payload["round"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "pick assets require integer 'season' and 'round' fields"
            ) from exc
        original = payload.get("original_roster_id")
        return cls(
            season=season,
            round=round_,
            original_roster_id=int(original) if original is not None else None,
        )

    @property
    def label(self) -> str:
        return f"{self.season}-R{self.round}"


@dataclass
class TradeSide:
    roster_id: int
    player_ids: list[str] = field(default_factory=list)
    pick_assets: list[dict] = field(default_factory=list)

    def picks(self) -> list[PickAsset]:
        return [PickAsset.parse(p) for p in self.pick_assets]


@dataclass
class TradeEvaluationResult:
    objective: dict
    fairness: dict
    acceptance: dict
    assets: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def value_future_pick(
    pick: PickAsset,
    *,
    current_season: int,
    league_size: int = 12,
    expected_slot: float | None = None,
    slot_uncertainty: float | None = None,
    rookie_points_at_slot: dict[int, float] | None = None,
) -> tuple[float, float]:
    """Value a future rookie pick in horizon points, with its uncertainty.

    The value is anchored on the expected production of the pick's likely slot
    and then discounted for how far away the draft is. Uncertainty widens with
    each additional season because both the owner's finish and the class
    strength are unknown. Nothing here consults the trade being evaluated, so a
    pick's value cannot justify itself circularly.
    """
    years_out = max(0, pick.season - current_season)
    if expected_slot is None:
        # Without roster-trajectory information, assume the middle of the round.
        expected_slot = (pick.round - 1) * league_size + league_size / 2.0
    if slot_uncertainty is None:
        slot_uncertainty = league_size / 4.0

    # Widen the slot distribution one quarter-round per extra season.
    slot_uncertainty = slot_uncertainty * (1.0 + 0.35 * years_out)

    curve = rookie_points_at_slot or {}
    if curve:
        nearest = min(curve, key=lambda slot: abs(slot - expected_slot))
        base = float(curve[nearest])
    else:
        # Monotone decreasing anchor: pick 1 is most valuable, decaying by slot.
        base = 120.0 / (1.0 + 0.16 * max(expected_slot - 1.0, 0.0))

    # Future picks are discounted for time-to-contribution and class risk.
    discount = 0.82**years_out
    value = base * discount

    # Uncertainty as a fraction of value: slot spread plus per-year class risk.
    relative = min(0.85, 0.18 + 0.08 * years_out + slot_uncertainty / (4.0 * league_size))
    return round(value, 4), round(value * relative, 4)


def _side_totals(
    side: TradeSide,
    player_values: dict[str, float],
    pick_values: dict[str, tuple[float, float]],
) -> tuple[float, float, list[str]]:
    total = 0.0
    variance = 0.0
    missing: list[str] = []
    for pid in side.player_ids:
        if pid not in player_values:
            missing.append(pid)
            continue
        total += float(player_values[pid])
    for pick in side.picks():
        value, uncertainty = pick_values.get(pick.label, (0.0, 0.0))
        total += value
        variance += uncertainty**2
    return total, variance**0.5, missing


def evaluate_trade(
    side_a: TradeSide,
    side_b: TradeSide,
    *,
    player_values: dict[str, float],
    league_type: str,
    current_season: int,
    pick_values: dict[str, tuple[float, float]] | None = None,
    tendency_adjustment: float = 0.0,
    tendency_sample_size: int = 0,
    roster_context: dict | None = None,
    horizon: str = "ros",
) -> TradeEvaluationResult:
    """Evaluate a two-sided trade.

    ``player_values`` and ``pick_values`` must be computed by the caller from a
    projection release; this function never invents an asset value. Assets with
    no available valuation are reported in ``warnings`` and excluded, so a
    missing projection cannot silently read as a worthless player.
    """
    is_dynasty = str(league_type).lower() == "dynasty"
    all_picks = list(side_a.pick_assets) + list(side_b.pick_assets)
    if all_picks and not is_dynasty:
        labels = [PickAsset.parse(p).label for p in all_picks]
        raise RedraftPickNotTradeable(
            "redraft leagues do not trade future draft picks; "
            f"remove {labels} or evaluate this trade in a dynasty league"
        )

    pick_values = dict(pick_values or {})
    if is_dynasty:
        for payload in all_picks:
            pick = PickAsset.parse(payload)
            if pick.label not in pick_values:
                pick_values[pick.label] = value_future_pick(
                    pick, current_season=current_season
                )

    a_value, a_sigma, a_missing = _side_totals(side_a, player_values, pick_values)
    b_value, b_sigma, b_missing = _side_totals(side_b, player_values, pick_values)

    warnings: list[str] = []
    for pid in a_missing + b_missing:
        warnings.append(f"no_projection_for_asset:{pid}")

    # Each side gains what it receives and loses what it sends.
    gain_a = b_value - a_value
    gain_b = a_value - b_value

    scale = max(a_value + b_value, 1.0)
    combined_sigma = (a_sigma**2 + b_sigma**2) ** 0.5
    gap = abs(gain_a)
    relative_gap = gap / scale

    # Tendency influence shrinks toward zero when the behavioural sample is thin,
    # so a manager with two logged proposals cannot move the estimate much.
    shrink = tendency_sample_size / (tendency_sample_size + 8.0)
    objective_weight = MAX_OBJECTIVE_WEIGHT - (
        MAX_OBJECTIVE_WEIGHT - MIN_OBJECTIVE_WEIGHT
    ) * shrink
    tendency_weight = 1.0 - objective_weight
    bounded_tendency = max(-1.0, min(1.0, tendency_adjustment)) * shrink

    def acceptance_for(gain: float, tendency_sign: float) -> float:
        # Normalised objective signal in [-1, 1].
        signal = max(-1.0, min(1.0, gain / scale * 3.0))
        raw = 0.5 + objective_weight * signal * 0.5 + tendency_weight * (
            bounded_tendency * tendency_sign
        )
        probability = max(0.02, min(0.95, raw))
        if gain <= -MATERIAL_HARM_POINTS:
            # Behaviour may never recommend a materially harmful trade.
            probability = min(probability, 0.15)
        return round(probability, 4)

    acceptance_a = acceptance_for(gain_a, +1.0)
    acceptance_b = acceptance_for(gain_b, -1.0)

    context = roster_context or {}
    return TradeEvaluationResult(
        objective={
            "horizon": horizon,
            "side_a_value_sent": round(a_value, 4),
            "side_b_value_sent": round(b_value, 4),
            "side_a_gain": round(gain_a, 4),
            "side_b_gain": round(gain_b, 4),
            "side_a_gain_range": [
                round(gain_a - combined_sigma, 4),
                round(gain_a + combined_sigma, 4),
            ],
            "side_b_gain_range": [
                round(gain_b - combined_sigma, 4),
                round(gain_b + combined_sigma, 4),
            ],
            "roster_construction": {
                "side_a_assets_out": len(side_a.player_ids) + len(side_a.pick_assets),
                "side_b_assets_out": len(side_b.player_ids) + len(side_b.pick_assets),
                "consolidating_side": (
                    "side_a"
                    if len(side_a.player_ids) > len(side_b.player_ids)
                    else "side_b"
                    if len(side_b.player_ids) > len(side_a.player_ids)
                    else "even"
                ),
            },
            "context": {
                "side_a_state": context.get("side_a_state"),
                "side_b_state": context.get("side_b_state"),
                "league_type": "dynasty" if is_dynasty else "redraft",
                "picks_permitted": is_dynasty,
            },
        },
        fairness={
            "gap": round(gap, 4),
            "relative_gap": round(relative_gap, 4),
            "uncertainty": round(combined_sigma, 4),
            "favors": "side_a" if gain_a > 0 else "side_b" if gain_b > 0 else "even",
            # "Fair" requires both a small relative gap AND that neither side is
            # materially harmed in absolute horizon points. A 10% gap on a large
            # dynasty package is still tens of points and is not fair.
            "fair": bool(
                relative_gap <= 0.10
                and gain_a > -MATERIAL_HARM_POINTS
                and gain_b > -MATERIAL_HARM_POINTS
            ),
            "within_uncertainty": bool(gap <= combined_sigma),
        },
        acceptance={
            "side_a_probability": acceptance_a,
            "side_b_probability": acceptance_b,
            "objective_weight": round(objective_weight, 4),
            "tendency_weight": round(tendency_weight, 4),
            "tendency_adjustment": round(bounded_tendency, 4),
            "tendency_sample_size": tendency_sample_size,
            "tendency_shrinkage": round(shrink, 4),
            "materially_harmful_for": [
                side
                for side, gain in (("side_a", gain_a), ("side_b", gain_b))
                if gain <= -MATERIAL_HARM_POINTS
            ],
        },
        assets={
            "player_values": {
                pid: round(float(v), 4)
                for pid, v in player_values.items()
                if pid in set(side_a.player_ids) | set(side_b.player_ids)
            },
            "pick_values": {
                label: {"value": round(v, 4), "uncertainty": round(u, 4)}
                for label, (v, u) in pick_values.items()
            },
        },
        warnings=warnings,
    )
