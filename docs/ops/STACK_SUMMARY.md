# Fantasy Decisions — stack summary

Short briefing for teammates (and X Connector searches). Last updated 2026-09-17.

## Product

**Fantasy Decisions** is a weekly game-level projection product.

| Surface | Status |
| --- | --- |
| Vegas weekly props board | **Live** until promote |
| Weekly latent / model board | **Shadow only** |
| League Value | Frozen (draft-with-learnings) |
| ADP / season Vegas | Sanity bands only — not model drivers |

**Promote gate:** ≥6–8 credited live Role-2 weeks (leakage-safe). No promote until then.

**Out of scope (for now):** Monte Carlo, Coach chat, DFS salary boards, expanding Operations, Role-3 blend.

## Team lanes

| Lane | Owns | Does not |
| --- | --- | --- |
| **Orchestrator** | Mission briefs, who-does-what, merge/promote gates, UX simplicity | Ship code or deep research alone |
| **Research** | Prior art, calm UI/PWA patterns, skills/plugins (recommend-only), M3 / Role-2 model-math recommendations | PRs / merges |
| **Engineering** | Cursor Cloud Agents, CI, GitHub PRs, Claude→Cursor apply, Eng CE review before merge-ready | Product direction |
| **X Connector** | On-demand X bookmark summaries and searches | Routines (avoids API credit burn) |

Prefer the **App Development** group for shared Fantasy Decisions work. Keep 1:1 for secrets, noisy mid-debug, or Orchestrator-only decisions.

## Engineering stack

- **Repo:** `richardfergusoniv/fantasy-projections`
- **Backend / models:** Python projection pipeline (`uv`, pytest, GitHub Actions)
- **App:** React + Vite PWA on Vercel
- **Ship loop:** Cursor Cloud Agent → draft PR → Claude Code Review (fail-closed) → Cursor apply → Eng CE review → merge-ready ping in App Development
- **Cloud Agent discipline:** one agent per workstream; reply instead of launching a second; no create retries on `ResourceExhausted`; model **Auto** unless Richard names one; prefer serial over parallel

## Modeling locks (M3 / Role 2)

Already landed / locked:

- Week-varying availability \(A_{i,w}\): bye + short-rest + lagged sit CSV with `available_at ≤ kickoff` (fail-closed)
- Conversions: \(\kappa_{conv} = 0.50\), clip \([0.90, 1.10]\), frozen (not fit to Vegas)
- Conservation + leakage CI
- Injury = manual sit CSV only
- Sanity bands Ops-only
- No probabilistic-sim M3; no Role-3 blend; no ADP/season Vegas as drivers

Role 2 is **measurement only** (shadow vs Vegas props). Fixture / dry-run weeks do **not** credit toward the 6–8 week sample.

## Open engineering work (as of 2026-09-17)

| PR | Focus | Notes |
| --- | --- | --- |
| [#86](https://github.com/richardfergusoniv/fantasy-projections/pull/86) | Role 2 M3 shadow compare | Draft; live weeks still need Richard’s leakage-safe props CSV (`player_id,season,week,market,as_of,kickoff_at` + `line` or `implied_mean`; gsis ids; `as_of ≤ kickoff`) |
| [#88](https://github.com/richardfergusoniv/fantasy-projections/pull/88) | PWA load-feel + as-of chrome | Draft; `test-windows` green after Waivers fix |
| [#89](https://github.com/richardfergusoniv/fantasy-projections/pull/89) | Claude review workflow harden | Ready; concurrency cancel-in-progress + fail-closed gate on action `is_error` |

Master CI is green. Model stays shadow-only.

## Useful X search angles

Prefer:

- Calm fantasy projection / matchup board UX (dense rows, as-of freshness, PWA load feel)
- Weekly latent / opportunity-share / availability + conversion discussion
- How other fantasy apps handle **props vs model boards**
- Leakage-safe props snapshots and player-id joins (gsis)

Skip: DFS salary spam, noise unrelated to weekly projections.

## Interrupt discipline

Interrupt Richard only for merges, spend/secrets, promote-to-production, or true product forks. Otherwise: board work → Cursor → babysit PR/CI/Claude → ping when merge-ready.
