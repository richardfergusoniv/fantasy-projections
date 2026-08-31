# QB team-volume share: revert 1.000 to the measured 0.941/0.942

**Date:** 2026-08-26
**Verdict:** `revert_to_measured_share`
**Scope:** `TEAM_VOLUME_SHARES[("QB", "attempts")]`, `TEAM_VOLUME_SHARES[("QB", "passing_yards")]`

## What shipped, and why it was suspect

An uncommitted batch moved the QB anchor share from the measured starter
figures (0.941 attempts / 0.942 passing yards) to a structural 1.000, on this
reasoning:

> QB reconciliation covers the entire room, so it owns 100% of the team
> passing anchor. The measured 0.941/0.942 figures describe the starter, not
> the room.

The argument is coherent. It was also never measured, and it shipped alongside
a second QB-facing change in the same batch — pass and rush TDs were detached
from volume scaling by emptying their `TEAM_VOLUME_SIBLINGS` entries. Two
changes on one position in one batch is the shape where a pair of errors
cancels and the net reads as an improvement, so neither could be judged from
the board.

## Method

`scripts/ablate_qb_volume_share.py` scores held-out seasons through
`build_leakage_safe_forecasts`, attaching outcomes only afterwards via the
shipped path's own `attach_actual_outcomes`. Each arm varies **one** factor
from shipped; `pre_change_both` reverts the pair together.

Every arm forecasts the same frozen population, so the comparison is paired
per player. That matters more than usual here: player-to-player spread in
season fantasy points is an order of magnitude larger than the effect, and an
unpaired summary cannot see past it. Single seasons (~106 QBs) were
underpowered — pooling 2023–2025 to n=312 is what resolved the question.

## Result (pooled, paired, vs shipped; negative = better)

| Arm | QB mean Δ | QB 95% CI | overall mean Δ | overall 95% CI | QBs improved |
|---|---|---|---|---|---|
| `qb_share_starter` | **−0.478** | **[−0.86, −0.10]** | **−0.151** | **[−0.23, −0.08]** | 52% |
| `tds_ride_volume` | +0.078 | [−0.26, +0.42] | −0.047 | [−0.13, +0.04] | 49% |
| `alpha_050` | −0.010 | [−0.20, +0.18] | +0.038 | [−0.03, +0.11] | 18% |
| `pre_change_both` | −0.364 | [−0.98, +0.25] | −0.204 | [−0.34, −0.07] | 53% |

`qb_share_starter` is the only arm whose interval excludes zero on both QB and
overall.

## Decision

Revert the QB anchor share to 0.941/0.942. **Keep TDs detached from volume
scaling.**

Reverting the share alone beats reverting both (QB −0.478 vs −0.364, and
`pre_change_both`'s QB interval does not exclude zero). The two changes partly
cancel: reattaching TDs is +0.078 on QB on its own. Measuring the middle,
rather than only shipped-vs-full-revert, is what separated them.

## Why the structural argument loses

The projected QB room does not in fact cover every team pass attempt —
population gaps, scrambles charted elsewhere, occasional non-QB passers. So
scaling the room to a full team anchor hands it volume the room never sees.
0.941/0.942 is what that coverage measures out at; it was never purely a
"starter vs room" distinction.

## Caveats

- Only ~52% of QBs individually improve. This moves the mean more than it
  moves the typical player, and the gain is concentrated in a minority of
  large corrections rather than spread across the room.
- `alpha_050` is the cautionary reading in this table: its mean delta is
  ~0 and on 2024 it was *negative* (−0.025, which reads as "slightly
  better"), while only **8%** of QBs improved that season and 18% pooled.
  A mean over a heavy-tailed error distribution hid a large majority getting
  worse. The fitted 0.75 stands.
- `tds_ride_volume` shows a Wilcoxon p=0.000 on `overall` while its
  t-interval spans zero, with 47% improved — mean and median disagree. Not
  treated as evidence either way.

## Follow-up

The published 2026 board still carries 1.000. It needs a republish to pick
this up; until then `output/projections_2026.csv` and the draft board reflect
the reverted-from configuration.
