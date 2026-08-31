# Promotion provenance — 2026-08-30

## Lifecycle trap

Publication writes `release_bundle_validation.json` next to the sealed
bundle. Routine validation rewrites that sidecar. Treating it as restore
authority would let any freshly published, never-promoted namespace claim
prior approval after a HEAD move — weakening first-promotion provenance.

Restore authorization therefore comes only from:

1. the active pointer's current or previous release identity, or
2. a git-tracked `release_promotion_receipt_v1` written after a successful
   activation.

## Derived modes

| Mode | When | Git rule |
|---|---|---|
| `initial` | No matching pointer/previous/receipt identity | `HEAD == source_commit` |
| `restore` | Exact namespace + release ID + manifest hash match pointer current/previous or a tracked receipt | `git merge-base --is-ancestor <source_commit> HEAD` |

Both modes still require `source_dirty == false` and a clean current worktree.
Public `promote_release()` / `rollback_release()` APIs expose no mode,
allow, or provenance bypass parameters; mode is derived internally and
recorded on the invariant result and receipt.

## Receipt trust boundary

Receipts live under `draft_assistant/data/promotion_receipts/<season>/` and
are small tracked files. They record season, namespace, release ID, manifest
hash, source commit, provenance mode, activation timestamp, and the
promotion-invariant verdict.

- Idempotent for an identical release identity.
- Fail on conflicting reuse of the same namespace path.
- Untracked, missing, malformed, or hash-conflicting receipts do not
  authorize restore.
- Schema-v1 phase1 bundles remain permanently non-promotable; no receipts
  were created for them.

## Ancestry rule

Restore does not require checking out the historical commit. It requires
that the bundle's `source_commit` already be in the history of the current
`HEAD`, so later documentation/control-plane commits can re-promote a prior
release without pretending the tree is still at seal time.

## Local-artifact dependency

Full release bundles under `output/model_v3/release_bundles/` stay gitignored
(including manifests and mutable validation sidecars). A fresh clone can serve
and smoke-test the checked-in public board, but cannot regenerate, fully
validate, promote, or roll back without the local full bundle. Promotion
receipts and the committed pointer recover *authorization*; they do not
replace sealed artifact bytes.
