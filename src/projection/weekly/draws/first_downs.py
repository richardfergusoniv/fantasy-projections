"""First-down component models for PPFD leagues."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class FirstDownRates:
    """Conditional first-down rates by opportunity type."""

    pass_fd_per_completion: float = 0.55
    rush_fd_per_carry: float = 0.22
    rec_fd_per_reception: float = 0.55


DEFAULT_RATES_BY_POS: dict[str, FirstDownRates] = {
    "QB": FirstDownRates(pass_fd_per_completion=0.58, rush_fd_per_carry=0.35, rec_fd_per_reception=0.0),
    "RB": FirstDownRates(pass_fd_per_completion=0.0, rush_fd_per_carry=0.24, rec_fd_per_reception=0.35),
    "WR": FirstDownRates(pass_fd_per_completion=0.0, rush_fd_per_carry=0.15, rec_fd_per_reception=0.58),
    "TE": FirstDownRates(pass_fd_per_completion=0.0, rush_fd_per_carry=0.12, rec_fd_per_reception=0.52),
}


def estimate_rates_from_history(
    rows: list[Mapping[str, float]],
    *,
    position: str,
) -> FirstDownRates:
    """Empirical beta-style rates with weak priors from defaults."""
    prior = DEFAULT_RATES_BY_POS.get(position, FirstDownRates())
    comp = sum(float(r.get("completions") or r.get("pass_completions") or 0.0) for r in rows)
    pass_fd = sum(float(r.get("passing_first_downs") or r.get("pass_first_downs") or 0.0) for r in rows)
    carries = sum(float(r.get("carries") or r.get("rush_attempts") or 0.0) for r in rows)
    rush_fd = sum(float(r.get("rushing_first_downs") or r.get("rush_first_downs") or 0.0) for r in rows)
    rec = sum(float(r.get("receptions") or 0.0) for r in rows)
    rec_fd = sum(float(r.get("receiving_first_downs") or r.get("rec_first_downs") or 0.0) for r in rows)

    def shrink(success: float, trials: float, prior_rate: float, prior_n: float = 50.0) -> float:
        return float((success + prior_rate * prior_n) / (trials + prior_n)) if trials + prior_n > 0 else prior_rate

    return FirstDownRates(
        pass_fd_per_completion=shrink(pass_fd, comp, prior.pass_fd_per_completion),
        rush_fd_per_carry=shrink(rush_fd, carries, prior.rush_fd_per_carry),
        rec_fd_per_reception=shrink(rec_fd, rec, prior.rec_fd_per_reception),
    )


def sample_first_downs(
    rng: np.random.Generator,
    *,
    position: str,
    completions: float,
    carries: float,
    receptions: float,
    rates: FirstDownRates | None = None,
) -> dict[str, float]:
    """Binomial first-down counts co-varying with realized opportunities."""
    rates = rates or DEFAULT_RATES_BY_POS.get(position, FirstDownRates())
    c = max(0, int(round(completions)))
    carries_i = max(0, int(round(carries)))
    rec_i = max(0, int(round(receptions)))
    pass_fd = int(rng.binomial(c, float(np.clip(rates.pass_fd_per_completion, 0.0, 1.0)))) if c else 0
    rush_fd = (
        int(rng.binomial(carries_i, float(np.clip(rates.rush_fd_per_carry, 0.0, 1.0)))) if carries_i else 0
    )
    rec_fd = (
        int(rng.binomial(rec_i, float(np.clip(rates.rec_fd_per_reception, 0.0, 1.0)))) if rec_i else 0
    )
    # Bounds are structural.
    rush_fd = min(rush_fd, carries_i)
    rec_fd = min(rec_fd, rec_i)
    pass_fd = min(pass_fd, c)
    return {
        "pass_first_downs": float(pass_fd),
        "rush_first_downs": float(rush_fd),
        "rec_first_downs": float(rec_fd),
    }


def reconcile_team_pass_rec_first_downs(
    qb_pass_fd: float,
    receiver_rec_fd: float,
) -> tuple[float, float]:
    """Under nflverse weekly definitions, team pass FD ≈ team receiving FD.

    Prefer the receiver-sum identity for fantasy PPFD and mirror onto QB.
    """
    value = float(receiver_rec_fd)
    return value, value
