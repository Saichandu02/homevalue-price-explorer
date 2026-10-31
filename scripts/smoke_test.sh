#!/usr/bin/env bash
# Smoke test against a running server. Usage: bash scripts/smoke_test.sh [base]
set -e
BASE="${1:-http://127.0.0.1:8101}"
echo "== healthz"; curl -sf "$BASE/healthz"; echo
echo "== readyz";  curl -sf "$BASE/readyz"; echo
echo "== model";   curl -sf "$BASE/api/model" | head -c 400; echo
echo "== predict"
curl -sf -X POST "$BASE/api/predict" -H 'Content-Type: application/json' \
  -d '{"MedInc":5.0,"HouseAge":20,"AveRooms":6.0,"AveBedrms":1.0,"Population":1000,"AveOccup":3.0,"Latitude":37.88,"Longitude":-122.23}'
echo; echo "SMOKE OK"
