#!/usr/bin/env bash
# Smoke test a running API (local or container). Usage: scripts/smoke_test_api.sh [base_url]
set -euo pipefail
BASE="${1:-http://localhost:8000}"
KEY_HEADER=()
[ -n "${API_KEY:-}" ] && KEY_HEADER=(-H "X-API-Key: ${API_KEY}")
echo "== health";  curl -fsS "$BASE/health"; echo
echo "== ask";     curl -fsS -X POST "$BASE/ask" "${KEY_HEADER[@]}" -H 'Content-Type: application/json' \
  -d '{"question":"How should an intermediate biomarker result be handled?"}' | head -c 600; echo
echo "== agent";   curl -fsS -X POST "$BASE/agent" "${KEY_HEADER[@]}" -H 'Content-Type: application/json' \
  -d '{"task":"The assay ratio was 0.45. How should an intermediate result be interpreted?"}' | head -c 600; echo
echo "== metrics"; curl -fsS "$BASE/metrics" | head -5
