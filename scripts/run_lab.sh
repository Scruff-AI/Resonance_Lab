#!/usr/bin/env bash
# Start the Resonance Lab web UI.
#
# Run this on the SAME machine as the daemon — the daemon binds its ZMQ ports
# on 127.0.0.1 and nothing outside that host can reach them. To use the UI from
# a laptop, tunnel the HTTP port:
#
#     ssh -N -L 8800:127.0.0.1:8800 you@the-gpu-node
#
# Then open http://127.0.0.1:8800
#
# Environment (all optional):
#   HTTP_PORT=8800            port the UI listens on
#   MODEL_BACKEND=ollama      ollama | openai | none
#   OLLAMA_URL=http://127.0.0.1:11434
#   OPENAI_BASE_URL=https://api.openai.com/v1
#   RESONANCE_LAB_API_KEY=... key for the openai backend
#   DAEMON_HOST=127.0.0.1

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! python3 -c "import fastapi, zmq, numpy, PIL" 2>/dev/null; then
    echo "Missing Python dependencies. Install them with:" >&2
    echo "    pip install -r lab/requirements.txt" >&2
    exit 1
fi

exec python3 -m lab.server
