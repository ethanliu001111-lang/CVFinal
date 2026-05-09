#!/usr/bin/env bash
# Push code (and optionally models) to a Linux GPU server via rsync.
#
# Usage:
#   bash demo/scripts/sync_to_server.sh user@host                            # code only
#   bash demo/scripts/sync_to_server.sh user@host /scratch/cv-final          # custom remote path
#   bash demo/scripts/sync_to_server.sh user@host ~/cv-final --with-models   # also push 4.5 GB models

set -euo pipefail

REMOTE="${1:?Usage: sync_to_server.sh <user@host> [remote_dir] [--with-models]}"
REMOTE_DIR="${2:-~/cv-final}"
WITH_MODELS=0
for a in "$@"; do [[ "$a" == "--with-models" ]] && WITH_MODELS=1; done

LOCAL_REPO="$(cd "$(dirname "$0")/../.." && pwd)"
LOCAL_MODELS="${CV_MODEL_ROOT:-$HOME/Documents/Intro to CV/CVFinal/model}"

echo "Repo  → $REMOTE:$REMOTE_DIR"
rsync -avzh --partial --info=progress2 \
    --exclude='.venv' --exclude='.git' --exclude='.claude' \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='.ipynb_checkpoints' \
    --exclude='*.aux' --exclude='*.bbl' --exclude='*.blg' \
    --exclude='*.fdb_latexmk' --exclude='*.fls' --exclude='*.log' --exclude='*.out' \
    --include='demo/checkpoints/.gitignore' --include='demo/checkpoints/DOWNLOAD.md' \
    --exclude='demo/checkpoints/*' \
    --exclude='*.pkl' --exclude='*.npz' --exclude='*.ckpt' \
    --exclude='*.pth' --exclude='*.pt' --exclude='*.bin' \
    --exclude='*.DS_Store' --exclude='._*' \
    "$LOCAL_REPO/" "$REMOTE:$REMOTE_DIR/"

if [[ $WITH_MODELS -eq 1 ]]; then
    echo
    echo "Models → $REMOTE:$REMOTE_DIR/../model/  (≈ 4.5 GB, one-time)"
    rsync -avzh --partial --info=progress2 \
        "$LOCAL_MODELS/" \
        "$REMOTE:$REMOTE_DIR/../model/"
fi

cat <<EOF

✓ sync done.

On the server:
  ssh $REMOTE
  cd $REMOTE_DIR
  bash demo/scripts/setup_linux_server.sh
  source ~/.venv/cv-final/bin/activate
  CV_MODEL_ROOT=\$(realpath ../model) bash demo/scripts/setup_smpl_paths.sh
  python demo/scripts/run_pipeline.py demo/test_images/*.jpg --out-dir demo/results

Pull results back:
  rsync -avz $REMOTE:$REMOTE_DIR/demo/results/ ./demo/results/
EOF
