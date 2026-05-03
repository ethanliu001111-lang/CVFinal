#!/usr/bin/env bash
# Start a Jupyter server on the remote Linux machine, then open an SSH tunnel.
#
# Usage (run on YOUR LAPTOP):
#   bash demo/scripts/start_remote_jupyter.sh user@host
#
# Then open http://localhost:8888/?token=... (token is printed below) in a browser.

set -euo pipefail

REMOTE="${1:?Usage: start_remote_jupyter.sh <user@host> [port]}"
PORT="${2:-8888}"
VENV="${VENV:-~/.venv/cv-final}"

echo "─── Starting Jupyter on $REMOTE (port $PORT) ───"

# Ensure jupyter is installed in the venv (idempotent)
ssh "$REMOTE" "source $VENV/bin/activate && pip install -q jupyter ipykernel"

# Start jupyter in a screen/tmux session if available; else background it
ssh "$REMOTE" "source $VENV/bin/activate && \
    nohup jupyter notebook --no-browser --port=$PORT \
        --notebook-dir=~/cv-final \
        > ~/jupyter.log 2>&1 & \
    sleep 3 && grep token ~/jupyter.log | head -1" || true

echo
echo "─── Opening SSH tunnel localhost:$PORT → $REMOTE:$PORT ───"
echo "    (Ctrl-C to close the tunnel; jupyter keeps running on the server)"
echo
echo "Open: http://localhost:$PORT/?token=<paste-from-above>"
echo

ssh -N -L "$PORT:localhost:$PORT" "$REMOTE"
