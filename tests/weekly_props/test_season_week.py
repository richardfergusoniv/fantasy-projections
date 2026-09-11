"""Season/week advancement for the weekly_props scheduled job."""

from __future__ import annotations

from src.app.jobs.handlers import SeasonWeek, resolve_season_week


class _FakeQuery:
    def __init__(self, result=None):
        self._result = result

    def filter(self, *args, **kwargs):
        return self

    def order_by(self, *args, **kwargs):
        return self

    def first(self):
        return self._result


class _FakeSession:
    def query(self, model):
        return _FakeQuery(None)


def test_resolve_season_week_prefers_live_nfl_state_over_pointer():
    session = _FakeSession()
    # Even if a DB pointer would say week 1, live state advances the job.
    resolved = resolve_season_week(
        session,
        nfl_state={"season": 2026, "week": 1, "display_week": 2},
    )
    assert resolved == SeasonWeek(2026, 2, "nfl_state")


def test_run_weekly_props_uses_live_nfl_state(monkeypatch, db_session):
    from src.app.jobs import handlers

    calls: dict[str, object] = {}

    class _Client:
        def get_nfl_state(self):
            return {"season": 2026, "week": 2, "display_week": 2}

    class _Sync:
        def __init__(self, session, use_fixtures=False):
            self.client = _Client()

    def _fake_resolve(session, *, nfl_state=None):
        calls["nfl_state"] = nfl_state
        return SeasonWeek(2026, 2, "nfl_state")

    class _Ingest:
        success_count = 0

        def __init__(self) -> None:
            self.snapshots: list = []

        def to_dict(self):
            return {"success_count": 0}

    monkeypatch.setattr(handlers, "SleeperSyncService", _Sync)
    monkeypatch.setattr(handlers, "resolve_season_week", _fake_resolve)
    monkeypatch.setattr(
        "src.ingest.props.service.build_providers",
        lambda **_k: [],
    )
    monkeypatch.setattr(
        "src.ingest.props.service.run_ingest",
        lambda **_k: _Ingest(),
    )
    monkeypatch.setattr(
        "src.app.ops.alerts.send_ops_alert",
        lambda *_a, **_k: None,
    )
    class _SeasonRefresh:
        def to_dict(self):
            return {"success_count": 0}

    monkeypatch.setattr(
        "src.ingest.props.season_refresh.refresh_season_ou",
        lambda **_k: _SeasonRefresh(),
    )

    result = handlers.run_weekly_props(db_session, automatic=False)
    assert calls["nfl_state"] == {"season": 2026, "week": 2, "display_week": 2}
    assert result["season"] == 2026
    assert result["week"] == 2
    assert result["season_week_source"] == "nfl_state"
    assert result["status"] == "ingest_failed"
