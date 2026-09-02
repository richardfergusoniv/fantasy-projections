"""Fixture-based vertical smoke: login through assistant.

This script deliberately does two things that an earlier version did not, both
of which were needed to catch real defects:

1. It runs the data refresh **before** the read checks. Reading first and
   syncing last hid a defect where a refresh replaced every roster with
   unprojectable ids: the reads had already passed.
2. It asserts on response *content*, not just status codes. A 200 that carries
   an empty recommendation, a tool error, or no projection run is a failure
   here, because that is exactly what the user would experience as broken.
"""

from __future__ import annotations

import os
import sys
import uuid

from fastapi.testclient import TestClient

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("APP_ENABLE_DEV_AUTH", "true")
os.environ.setdefault("TEST_DATABASE_URL", "sqlite+pysqlite:///:memory:?cache=shared")
os.environ.setdefault("TRUSTED_HOSTS", "testserver,localhost,127.0.0.1,*")
os.environ.setdefault("APP_ALLOWED_EMAIL", "owner@example.com")
os.environ.setdefault("EMAIL_PROVIDER", "development")
os.environ.setdefault("SLEEPER_USE_FIXTURES", "true")
os.environ.setdefault("INJURY_RESEARCH_MODE", "fixture")

Failure = tuple[str, str]


def main() -> int:
    from src.app.config import get_settings
    from src.app.factory import create_app
    from src.app.persistence.database import get_session, init_db
    from src.app.seed import seed_development_data

    get_settings.cache_clear()
    init_db()
    with get_session() as session:
        seed = seed_development_data(session, email="owner@example.com")
    client = TestClient(create_app())

    link = client.post("/api/v1/auth/magic-link", json={"email": "owner@example.com"}).json()[
        "development_link"
    ]
    token = link.split("token=")[-1]
    verify = client.post("/api/v1/auth/verify", json={"token": token})
    csrf = verify.json()["csrf_token"]
    league_ids = seed["leagues"]
    league_id = league_ids[0]
    headers = {"X-CSRF-Token": csrf}

    failures: list[Failure] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        if ok:
            print(f"OK   {name}")
        else:
            print(f"FAIL {name}: {detail}")
            failures.append((name, detail))

    def get(path: str):
        response = client.get(path)
        return response, (response.json() if response.status_code < 400 else {})

    # 1. Refresh first: everything after this reads post-refresh state.
    sync = client.post(
        "/api/v1/sync",
        headers={**headers, "Idempotency-Key": f"smoke-sync-{uuid.uuid4().hex}"},
    )
    sync_body = sync.json() if sync.status_code < 400 else {}
    check(
        "sync",
        sync.status_code == 200 and sync_body.get("status") == "succeeded",
        f"{sync.status_code} {sync.text[:200]}",
    )
    unresolved = (sync_body.get("metadata") or {}).get("unresolved_player_ids") or []
    check("sync resolves every rostered player", not unresolved, str(unresolved[:10]))

    response, body = get("/api/v1/me")
    check("me", response.status_code == 200 and bool(body.get("email")), response.text[:200])

    response, body = get("/api/v1/leagues")
    listed = {row["league_id"] for row in body.get("leagues", [])}
    check(
        "leagues",
        response.status_code == 200 and set(league_ids) <= listed,
        f"missing {sorted(set(league_ids) - listed)}",
    )

    # 2. Every league must still be recommendable after the refresh, with a
    #    projection run behind it and no rostered player left unprojectable.
    contract_hashes: set[str] = set()
    for lid in league_ids:
        response, body = get(f"/api/v1/leagues/{lid}/rules")
        rules = body.get("rules") or {}
        check(
            f"rules[{lid}]",
            response.status_code == 200
            and not rules.get("unsupported_keys")
            and not rules.get("unsupported_slots"),
            response.text[:200],
        )
        if body.get("contract_hash"):
            contract_hashes.add(body["contract_hash"])

        response, body = get(f"/api/v1/leagues/{lid}/lineup/1?opponent_mode=current")
        check(
            f"lineup[{lid}]",
            response.status_code == 200
            and bool(body.get("recommended_starters"))
            and body.get("projection_available") is True
            and not body.get("players_without_projection"),
            response.text[:250],
        )

    check(
        "six distinct scoring contracts",
        len(contract_hashes) == len(league_ids),
        f"{len(contract_hashes)} distinct for {len(league_ids)} leagues",
    )

    response, body = get(f"/api/v1/leagues/{league_id}/waivers/1")
    check(
        "waivers",
        response.status_code == 200 and bool(body.get("recommendations")),
        response.text[:200],
    )

    response, body = get(f"/api/v1/leagues/{league_id}/rankings?mode=weekly&week=1")
    check("rankings", response.status_code == 200 and bool(body.get("rankings")), response.text[:200])

    response, body = get(f"/api/v1/leagues/{league_id}/matchups/1")
    check("matchups", response.status_code == 200, response.text[:200])

    response, body = get("/api/v1/operations/status")
    check(
        "operations",
        response.status_code == 200
        and body.get("modes", {}).get("sleeper_source") == "fixture",
        response.text[:200],
    )

    trade = client.post(
        f"/api/v1/leagues/{league_id}/trades/evaluate",
        json={
            "side_a": {"roster_id": 1, "player_ids": ["00-0034857"]},
            "side_b": {"roster_id": 2, "player_ids": ["00-0033280"]},
        },
        headers={**headers, "Idempotency-Key": f"smoke-trade-{uuid.uuid4().hex}"},
    )
    trade_body = trade.json() if trade.status_code < 400 else {}
    check(
        "trade",
        trade.status_code == 200 and bool(trade_body.get("objective")),
        trade.text[:200],
    )

    # 3. The assistant must actually answer from a tool, not merely return 200
    #    with a swallowed tool error.
    assistant = client.post(
        "/api/v1/assistant/responses",
        json={"message": "lineup starters for week 1", "league_id": league_id, "week": 1},
        headers={**headers, "Idempotency-Key": f"smoke-assistant-{uuid.uuid4().hex}"},
    )
    assistant_body = assistant.json() if assistant.status_code < 400 else {}
    tool_result = assistant_body.get("tool_result") or {}
    check(
        "assistant",
        assistant.status_code == 200
        and assistant_body.get("tools_called") == ["recommend_lineup"]
        and "error" not in tool_result
        and bool(tool_result.get("recommended_starters")),
        f"{assistant.status_code} {str(assistant_body)[:250]}",
    )

    if failures:
        print(f"\nvertical smoke failed: {len(failures)} check(s)")
        return 1
    print("\nvertical smoke passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
