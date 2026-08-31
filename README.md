# Fantasy Projections

NFL fantasy projections, Monte Carlo simulation, and sealed release promotion for the draft assistant.

## Requirements

- Python 3.14
- [uv](https://docs.astral.sh/uv/) for locked dependency installs

## Environment

| Variable | Purpose |
|---|---|
| `FANTASY_PROJECTIONS_DATA_DIR` | Root for `projections.db` / raw cache (see `src/paths.py`) |
| `FANTASY_PROJECTIONS_V2` | Path to the sibling v2 repo (default `../fantasy-projections-2`) |

A fresh clone can serve and pytest-smoke the checked-in public board under
`draft_assistant/data/`. Regenerating boards, fully validating a sealed
bundle, promoting, or rolling back still requires the local ignored full
bundle under `output/model_v3/release_bundles/`.

## Setup

```bash
uv sync --frozen --all-extras --dev
```

## Common commands

```bash
# Full test suite
uv run pytest -q

# Build a promotion-eligible release bundle (clean git tree required)
uv run python -m src.projection.publish --season 2026 --simulation-profile publish --artifact-namespace <namespace>

# Validate and promote
uv run python scripts/validate_release_bundle.py --season 2026 --artifact-namespace <namespace>
uv run python -m src.projection.promote_release --season 2026 --artifact-namespace <namespace>

# Serve draft assistant locally
uv run python -m src.draft_assistant.serve --port 8766

# Browser verification (pointer-driven when --namespace is omitted)
uv run python scripts/verify_browser_surfaces.py --base-url http://127.0.0.1:8766 --season 2026
```

## Release model

New bundles use `release_bundle_manifest_v2` with six mandatory promotion invariants. Schema-v1 bundles remain readable but cannot be promoted. Rollback/restore uses tracked promotion receipts plus git ancestry — not the mutable validation sidecar. See `docs/PIPELINE_MAP.md` §8a/§10 and `docs/decisions/PROMOTION_PROVENANCE_2026-08-30.md`.
