#!/usr/bin/env bash
# Clear all Jupyter notebook outputs in-place before committing.
# Keeps diffs clean and notebook cells lightweight on git.
#
# Usage:  bash demo/scripts/clear_notebook_outputs.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

if ! command -v jupyter >/dev/null 2>&1; then
    if [[ -d ".venv" ]]; then
        # shellcheck disable=SC1091
        source .venv/bin/activate
    fi
fi
command -v jupyter >/dev/null 2>&1 || { echo "❌ jupyter not in PATH; run: pip install jupyter"; exit 1; }

count=0
while IFS= read -r nb; do
    jupyter nbconvert --clear-output --inplace "$nb"
    count=$((count + 1))
done < <(find demo/notebooks -name "*.ipynb" -not -path "*/.ipynb_checkpoints/*")

echo "✅ Cleared outputs on $count notebooks"
