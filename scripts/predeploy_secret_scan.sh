#!/usr/bin/env bash
# Predeploy secret gate.
#
# Fails closed. The previous version exited 0 with a warning when no scanner
# was installed, which is how a live Resend API key sat in
# config/phone_access.secrets.example.json through every predeploy check: the
# step reported success while scanning nothing.
#
# Scans the working tree rather than git history. The working tree is what gets
# deployed, and it takes ~2s against ~17s for a full history walk. Auditing
# history for previously committed secrets is a separate, occasional job.
set -euo pipefail

CONFIG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.gitleaks.toml"

if command -v gitleaks >/dev/null 2>&1; then
  exec gitleaks detect --no-git --no-banner --redact --source . --config "$CONFIG"
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
