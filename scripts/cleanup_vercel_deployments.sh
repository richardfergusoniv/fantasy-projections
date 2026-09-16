#!/usr/bin/env bash
# Delete old Vercel deployments to free Functions/Deployment Storage.
#
# Keeps:
#   - Current production deployment (latest with target=production)
#   - KEEP_PRODUCTION additional recent production deployments (default: 3)
#   - KEEP_PREVIEW recent non-production deployments (default: 5)
# Optionally shortens the project retention policy so storage does not refill.
#
# Required env:
#   VERCEL_TOKEN
# Optional env:
#   VERCEL_ORG_ID / VERCEL_TEAM_ID (default: team_2wadxBpdAExHEXF0iyvy6t1F)
#   VERCEL_PROJECT_ID (default: prj_TrOVfWAUKG2VHfvV7PHiROTkFWH6)
#   KEEP_PRODUCTION (default: 3)
#   KEEP_PREVIEW (default: 5)
#   DRY_RUN=1 (list only; no deletes / no policy change)
#   SET_RETENTION=1 (also PATCH short retention defaults; default: 1)
#   PREVIEW_RETENTION_DAYS (default: 7)
#   PRODUCTION_RETENTION_DAYS (default: 30)
#   CANCELED_RETENTION_DAYS (default: 1)
#   ERRORED_RETENTION_DAYS (default: 1)
#   DEPLOYMENTS_TO_KEEP (default: KEEP_PRODUCTION + 1)

set -euo pipefail

TOKEN="${VERCEL_TOKEN:?VERCEL_TOKEN is required}"
TEAM_ID="${VERCEL_ORG_ID:-${VERCEL_TEAM_ID:-team_2wadxBpdAExHEXF0iyvy6t1F}}"
PROJECT_ID="${VERCEL_PROJECT_ID:-prj_TrOVfWAUKG2VHfvV7PHiROTkFWH6}"
KEEP_PRODUCTION="${KEEP_PRODUCTION:-3}"
KEEP_PREVIEW="${KEEP_PREVIEW:-5}"
DRY_RUN="${DRY_RUN:-0}"
SET_RETENTION="${SET_RETENTION:-1}"
PREVIEW_RETENTION_DAYS="${PREVIEW_RETENTION_DAYS:-7}"
PRODUCTION_RETENTION_DAYS="${PRODUCTION_RETENTION_DAYS:-30}"
CANCELED_RETENTION_DAYS="${CANCELED_RETENTION_DAYS:-1}"
ERRORED_RETENTION_DAYS="${ERRORED_RETENTION_DAYS:-1}"
DEPLOYMENTS_TO_KEEP="${DEPLOYMENTS_TO_KEEP:-$((KEEP_PRODUCTION + 1))}"

API="https://api.vercel.com"

echo "Team: ${TEAM_ID}"
echo "Project: ${PROJECT_ID}"
echo "Keep production (extra): ${KEEP_PRODUCTION}"
echo "Keep preview: ${KEEP_PREVIEW}"
echo "Dry run: ${DRY_RUN}"

export TOKEN TEAM_ID PROJECT_ID KEEP_PRODUCTION KEEP_PREVIEW DRY_RUN SET_RETENTION
export PREVIEW_RETENTION_DAYS PRODUCTION_RETENTION_DAYS CANCELED_RETENTION_DAYS
export ERRORED_RETENTION_DAYS DEPLOYMENTS_TO_KEEP API

python3 <<'PY'
import json
import os
import sys
import urllib.error
import urllib.request

API = os.environ["API"]
TOKEN = os.environ["TOKEN"]
TEAM_ID = os.environ["TEAM_ID"]
PROJECT_ID = os.environ["PROJECT_ID"]
KEEP_PRODUCTION = int(os.environ["KEEP_PRODUCTION"])
KEEP_PREVIEW = int(os.environ["KEEP_PREVIEW"])
DRY_RUN = os.environ["DRY_RUN"] == "1"
SET_RETENTION = os.environ["SET_RETENTION"] == "1"


def request(method, path, body=None):
    data = None
    headers = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(f"{API}{path}", data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read()
            if not raw:
                return {}
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise SystemExit(f"{method} {path} failed: {e.code} {detail}") from e


deployments: list[dict] = []
until = None
page = 0
while True:
    page += 1
    qs = f"projectId={PROJECT_ID}&teamId={TEAM_ID}&limit=100"
    if until is not None:
        qs += f"&until={until}"
    payload = request("GET", f"/v6/deployments?{qs}")
    batch = payload.get("deployments") or []
    print(f"Fetched page {page}: {len(batch)} deployments")
    if not batch:
        break
    deployments.extend(batch)
    until = (payload.get("pagination") or {}).get("next")
    if until is None:
        break

print(f"Total deployments: {len(deployments)}")


def created(d: dict) -> int:
    return int(d.get("createdAt") or d.get("created") or 0)


deployments.sort(key=created, reverse=True)
production = [d for d in deployments if d.get("target") == "production"]
preview = [d for d in deployments if d.get("target") != "production"]

keep: set[str] = set()
for d in production[: max(1, KEEP_PRODUCTION + 1)]:
    keep.add(d["uid"])
for d in preview[:KEEP_PREVIEW]:
    keep.add(d["uid"])

protected_states = {"BUILDING", "INITIALIZING", "QUEUED"}
for d in deployments:
    state = d.get("readyState") or d.get("state")
    if state in protected_states:
        keep.add(d["uid"])

to_delete = [d for d in deployments if d["uid"] not in keep]
print(f"Keeping {len(keep)} deployments")
print(f"Deleting {len(to_delete)} deployments")
print(f"  production kept: {sum(1 for d in production if d['uid'] in keep)}")
print(f"  preview/other kept: {sum(1 for d in preview if d['uid'] in keep)}")
if production:
    newest = next(d for d in production if d["uid"] in keep)
    print(f"  current production: {newest.get('url')} ({newest['uid']})")

deleted = 0
failed = 0
if DRY_RUN:
    print("DRY_RUN=1 — skipping deletes. Sample delete candidates:")
    for d in to_delete[:20]:
        print(f"  {d['uid']} target={d.get('target')} state={d.get('readyState') or d.get('state')} url={d.get('url')}")
else:
    for d in to_delete:
        uid = d["uid"]
        try:
            request("DELETE", f"/v13/deployments/{uid}?teamId={TEAM_ID}")
            deleted += 1
            if deleted % 25 == 0:
                print(f"Deleted {deleted}...")
        except SystemExit as e:
            print(f"Failed to delete {uid}: {e}", file=sys.stderr)
            failed += 1
    print(f"Deleted {deleted}; failed {failed}")

if SET_RETENTION:
    # Hobby/team APIs reject some update shapes; deletions above are the
    # storage win. Treat retention PATCH as best-effort.
    body = {
        "deploymentExpiration": {
            "expirationDays": int(os.environ["PREVIEW_RETENTION_DAYS"]),
            "expirationDaysProduction": int(os.environ["PRODUCTION_RETENTION_DAYS"]),
            "expirationDaysCanceled": int(os.environ["CANCELED_RETENTION_DAYS"]),
            "expirationDaysErrored": int(os.environ["ERRORED_RETENTION_DAYS"]),
            "deploymentsToKeep": int(os.environ["DEPLOYMENTS_TO_KEEP"]),
        }
    }
    print(f"Setting retention policy: {json.dumps(body)}")
    if DRY_RUN:
        print("DRY_RUN=1 — skipping retention PATCH")
    else:
        try:
            request("PATCH", f"/v9/projects/{PROJECT_ID}?teamId={TEAM_ID}", body)
            print("Retention policy updated")
        except SystemExit as e:
            print(f"Retention PATCH skipped (non-fatal): {e}", file=sys.stderr)
            current = request("GET", f"/v9/projects/{PROJECT_ID}?teamId={TEAM_ID}")
            print(f"Current deploymentExpiration: {current.get('deploymentExpiration')}")

print("Done.")
PY
