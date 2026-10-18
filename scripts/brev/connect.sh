#!/usr/bin/env bash
# Laptop: tunnel the Brev gateway to localhost:8000, where the web UI (localhost:8800) proxies /api and /ws.
# Stop any local gateway first; only one thing can hold port 8000. Usage: connect.sh <instance>
set -euo pipefail
INST=${1:?usage: connect.sh <instance>}
brev port-forward "$INST" --port 8000:8000
echo "Gateway on Brev → http://localhost:8000 · open the UI at http://localhost:8800 (cd web && npm run dev)"
