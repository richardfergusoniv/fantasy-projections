"""Fixture provider ingest smoke tests."""

from __future__ import annotations

from pathlib import Path

from src.ingest.props.service import default_fixture_providers, run_ingest
from src.ingest.props.snapshot import SnapshotStore

FIXTURES = Path(__file__).resolve().parents[3] / "data" / "props" / "fixtures" / "providers"


def test_fixture_ingest_succeeds(tmp_path):
    providers = default_fixture_providers(FIXTURES)
    assert providers
    result = run_ingest(
        season=2026,
        week=1,
        providers=providers,
        store=SnapshotStore(tmp_path),
    )
    assert result.success_count >= 1
    assert any(len(s.quotes) > 0 for s in result.snapshots if s.success)
