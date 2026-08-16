# Hierarchical rush mix (2026-08-14)

> **RETIRED.** Hierarchical rush mix and team rushing volume normalization
> were deleted from the shipped pipeline with the rest of volume composition.
> Kept as historical design notes.

Thin L2 rush layer that mirrored pass mix: RB / QB / OTHER carry shares.
No TE/FB package splits.

## LOSO gate (historical)

| | MAE |
|---|---|
| scheme+lag | **0.0343** |
| prior-season | 0.0347 |
| league-mean | 0.0415 |

`beats_prior=True` at the time; the module is gone nonetheless because the
composition path as a whole caused more harm than good on the draft board.
