#!/usr/bin/env bash
# Laptop: tunnel the Brev gateway to localhost:8000, where the web UI (localhost:8800) proxies /api and /ws.
# Stop any local gateway first; only one thing can hold port 8000. Usage: connect.sh <instance>
# Runs in the foreground and holds the tunnel; Ctrl-C closes it.
# A dedicated SSH connection (no multiplexing): `brev port-forward` rides brev's long-lived control master,
# where the forward can take a while to appear and outlives the command. This one is up at once and
# closes with Ctrl-C. If 8000 is "already in use", an earlier `brev port-forward` still holds it: use that.
set -euo pipefail
INST=${1:?usage: connect.sh <instance>}
echo "Gateway on Brev → http://localhost:8000 · UI at http://localhost:8800 (cd web && npm run dev)"
exec ssh -o ControlMaster=no -o ControlPath=none -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -N -L 8000:127.0.0.1:8000 "$INST"
