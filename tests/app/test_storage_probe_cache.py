"""Storage readiness probes should not hammer S3 on every overlapping request."""

from __future__ import annotations

from src.app.storage import release_bundle as rb


def test_probe_storage_round_trip_caches_within_ttl(monkeypatch):
    rb.clear_storage_probe_cache()
    calls = {"n": 0}

    class FakeStore:
        def put_json(self, payload):
            calls["n"] += 1
            return "local://probe"

        def get_json(self, uri):
            return {"probe": "readiness"}

    monkeypatch.setattr(rb, "get_artifact_store", lambda: FakeStore())
    first = rb.probe_storage_round_trip()
    second = rb.probe_storage_round_trip()
    forced = rb.probe_storage_round_trip(force=True)

    assert first["status"] == "healthy"
    assert second == first
    assert forced["status"] == "healthy"
    assert calls["n"] == 2
