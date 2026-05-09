#!/usr/bin/env bash
# Local sanity check: load SMPL-X, run T-pose forward, render a 3D PNG.
# Should finish in < 5 seconds on Mac M-series CPU.

set -euo pipefail
cd "$(dirname "$0")/../.."     # repo root

# Find a usable venv: prefer repo-root .venv, fall back to ~/.venv/cv-final
if [[ -d ".venv" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
elif [[ -d "$HOME/.venv/cv-final" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/.venv/cv-final/bin/activate"
else
    echo "❌ No venv found. Create one at ./.venv or ~/.venv/cv-final first."
    echo "   python3 -m venv .venv && source .venv/bin/activate && \\"
    echo "   pip install -r demo/envs/requirements_local_mac.txt"
    exit 1
fi

# Locate $CV_MODEL_ROOT — env var wins; otherwise probe a few sensible defaults.
if [[ -z "${CV_MODEL_ROOT:-}" ]]; then
    GIT_TOP="$(git rev-parse --show-toplevel 2>/dev/null || true)"
    for candidate in \
        "$(pwd)/../model" \
        "$(pwd)/../../model" \
        "$(pwd)/../../../model" \
        "${GIT_TOP:+$GIT_TOP/../model}" \
        "$HOME/cv-models"
    do
        [[ -z "$candidate" ]] && continue
        if [[ -d "$candidate" && -d "$candidate/smplx" ]]; then
            CV_MODEL_ROOT="$(cd "$candidate" && pwd)"
            break
        fi
    done
fi

if [[ -z "${CV_MODEL_ROOT:-}" ]]; then
    echo "❌ Could not auto-detect CV_MODEL_ROOT."
    echo "   Set it manually: export CV_MODEL_ROOT=/path/to/your/models"
    echo "   See demo/checkpoints/DOWNLOAD.md for the expected layout."
    exit 1
fi
export CV_MODEL_ROOT
echo "Using CV_MODEL_ROOT=$CV_MODEL_ROOT"

python -c "
import sys, os
sys.path.insert(0, os.getcwd())
from demo.src.smpl_forward import export_tpose_obj, smpl_forward_eq1
from demo.src.visualize import render_mesh_matplotlib
import matplotlib.pyplot as plt

obj = export_tpose_obj('demo/results/tpose_smplx.obj', model_type='smplx')
print(f'  ✓ {obj}')

verts, _, model = smpl_forward_eq1(model_type='smplx')
img = render_mesh_matplotlib(verts.squeeze().detach().numpy(), model.faces, azim=45, elev=10)
plt.imsave('demo/results/tpose_smplx_demo.png', img)
print('  ✓ demo/results/tpose_smplx_demo.png')
print()
print('Mac smoke test PASSED — SMPL-X library + matplotlib rendering OK')
"
