#!/usr/bin/env bash
# Predeploy secret gate.
#
# Fails closed. The previous version exited 0 with a warning when no scanner
# was installed, which is how a live Resend API key sat in
# config/phone_access.secrets.example.json through every predeploy check: the
# step reported success while scanning nothing.
#
# Scans the working tree rather than git history. The working tree is what gets
# deployed, and it is an order of magnitude faster than a full history walk.
# Auditing history for previously committed secrets is a separate job.
#
# Findings are summarised as rule + file + line only. The repository is public,
# so its CI logs are public; printing a matched secret to fix a leak would
# publish the very thing being fixed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"   # scan relative to the repo root so report paths stay readable
CONFIG="$ROOT/.gitleaks.toml"
REPORT="${TMPDIR:-/tmp}/gitleaks-report.json"

summarise() {
  python3 - "$REPORT" <<'PY' 2>/dev/null || echo "  (report at $REPORT)"
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
if not p.is_file() or not p.read_text().strip():
    raise SystemExit
rows = json.loads(p.read_text())
print(f"\n{len(rows)} finding(s) -- rule / file:line (values withheld: CI logs are public)\n")
seen = {}
for r in rows:
    seen.setdefault((r["RuleID"], r["File"]), []).append(r["StartLine"])
for (rule, f), lines in sorted(seen.items()):
    shown = ",".join(str(n) for n in sorted(lines)[:5])
    more = f" (+{len(lines)-5} more)" if len(lines) > 5 else ""
    print(f"  {rule:<24} {f}:{shown}{more}")
PY
}

if command -v gitleaks >/dev/null 2>&1; then
  if gitleaks detect --no-git --no-banner --redact --source . \
       --config "$CONFIG" --report-format json --report-path "$REPORT"; then
    exit 0
  fi
  summarise
  exit 1
fi

if command -v trufflehog >/dev/null 2>&1; then
  # --fail makes trufflehog exit non-zero on a verified finding; without it the
  # command returns 0 no matter what it prints.
  exec trufflehog filesystem --directory . --fail
fi

echo "predeploy_secret_scan: no secret scanner found (need gitleaks or trufflehog)." >&2
echo "Refusing to report a clean scan that never ran. Install gitleaks:" >&2
echo "  https://github.com/gitleaks/gitleaks/releases" >&2
exit 1
