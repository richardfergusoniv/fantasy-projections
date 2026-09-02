#!/usr/bin/env bash
set -euo pipefail

if command -v gitleaks >/dev/null 2>&1; then
  gitleaks detect --no-banner --redact --source .
elif command -v trufflehog >/dev/null 2>&1; then
  trufflehog filesystem --directory .
else
  echo "No secret scanner installed; skipping automated scan" >&2
  exit 0
fi
