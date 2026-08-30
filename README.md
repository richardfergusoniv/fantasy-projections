# Fantasy Projections

NFL fantasy projections, Monte Carlo simulation, and sealed release promotion for the draft assistant.

## Requirements

- Python 3.14
- [uv](https://docs.astral.sh/uv/) for locked dependency installs

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
```

## Release model

New bundles use `release_bundle_manifest_v2` with six mandatory promotion invariants. Schema-v1 bundles remain readable but cannot be promoted. See `docs/PIPELINE_MAP.md` section 8a.
