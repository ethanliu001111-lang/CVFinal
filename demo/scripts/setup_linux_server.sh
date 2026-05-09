#!/usr/bin/env bash
# Linux GPU server setup for the CV final project.
#
# Tested on:  Ubuntu 22.04 / CUDA 11.8 driver / Python 3.10
# Requires:   sudo (for apt) OR a sysadmin who can install osmesa/freeglut once
# GPU:        ≥ 8 GB VRAM (4D-Humans ViT-L peak ≈ 3 GB)
#
# Usage:
#   bash demo/scripts/setup_linux_server.sh           # full install (~10 min)
#   bash demo/scripts/setup_linux_server.sh --nosudo  # skip apt steps

set -euo pipefail

# ───────────────────────── Args ─────────────────────────
SKIP_APT=0
for arg in "$@"; do
    case "$arg" in
        --nosudo|--no-apt) SKIP_APT=1 ;;
    esac
done

# ───────────────────────── 1. Prereqs ─────────────────────────
echo "─── 1/6 Checking prereqs ───"
command -v nvidia-smi >/dev/null || { echo "❌ no GPU / no nvidia driver"; exit 1; }
nvidia-smi -L | head -2
PY=python3.10
command -v $PY >/dev/null || { echo "⚠️  python3.10 not found, falling back to python3"; PY=python3; }
$PY --version

# ───────────────────────── 2. System libs (osmesa for headless render) ─────────────────────────
if [[ $SKIP_APT -eq 0 ]]; then
    if command -v apt-get >/dev/null; then
        echo "─── 2/6 Installing system libraries via apt (Debian/Ubuntu) ───"
        sudo apt-get update -qq
        sudo apt-get install -y \
            libosmesa6-dev freeglut3-dev libglfw3-dev \
            libgl1 libegl1 libgles2 \
            ffmpeg build-essential git wget
    elif command -v dnf >/dev/null; then
        echo "─── 2/6 Installing system libraries via dnf (RHEL/CentOS/Fedora) ───"
        sudo dnf install -y \
            mesa-libOSMesa-devel freeglut-devel glfw-devel \
            mesa-libGL mesa-libEGL \
            ffmpeg gcc-c++ make git wget
    elif command -v yum >/dev/null; then
        echo "─── 2/6 Installing system libraries via yum (older RHEL/CentOS) ───"
        sudo yum install -y \
            mesa-libOSMesa-devel freeglut-devel glfw-devel \
            mesa-libGL mesa-libEGL \
            ffmpeg gcc-c++ make git wget
    else
        echo "⚠️  Unknown package manager. Install manually:"
        echo "    osmesa, freeglut, glfw, GL/EGL runtime, ffmpeg, build tools"
        exit 1
    fi
else
    echo "─── 2/6 (skipped — --nosudo) ───"
    echo "    Required system libs: osmesa freeglut glfw libGL ffmpeg + build tools"
    echo "    Ask sysadmin to install if rendering fails."
fi

# ───────────────────────── 3. Python virtualenv ─────────────────────────
echo "─── 3/6 Creating venv ───"
VENV_DIR="${VENV_DIR:-$HOME/.venv/cv-final}"
$PY -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install -q --upgrade pip wheel

# ───────────────────────── 4. PyTorch + MMPose stack ─────────────────────────
echo "─── 4/6 Installing PyTorch (CUDA 11.8) + MMPose ───"
pip install --quiet torch==2.1.0 torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu118

pip install --quiet -U openmim
mim install --quiet "mmengine>=0.7" "mmcv>=2.0,<2.2" "mmdet>=3.1" "mmpose>=1.1"

# ───────────────────────── 5. 4D-Humans (HMR 2.0) ─────────────────────────
echo "─── 5/6 Cloning + installing 4D-Humans ───"
REPO_DIR="${REPO_DIR:-$HOME/4D-Humans}"
if [[ ! -d "$REPO_DIR" ]]; then
    git clone --depth 1 https://github.com/shubham-goel/4D-Humans.git "$REPO_DIR"
fi
cd "$REPO_DIR"
pip install --quiet -e '.[all]'
cd - >/dev/null

# Pre-fetch HMR 2.0 weights (≈ 1.2 GB, one-time)
python - <<'PY'
# 4D-Humans exposes download_models from hmr2.models, not from a separate download_util module
from hmr2.models import download_models
try:
    from hmr2.configs import CACHE_DIR_4DHUMANS
except ImportError:
    import os
    CACHE_DIR_4DHUMANS = os.path.join(os.path.expanduser("~"), ".cache", "4DHumans")
download_models()
print(f"  HMR2 weights → {CACHE_DIR_4DHUMANS}")
PY

# ───────────────────────── 6. SMPL family + viz ─────────────────────────
echo "─── 6/6 SMPL libraries + viz ───"
pip install --quiet smplx trimesh pyrender imageio matplotlib pandas opencv-python scipy

# chumpy is required to load classic SMPL .pkl in Python 3
pip install --quiet chumpy || echo "⚠️  chumpy install failed; SMPL .pkl loading may need pickle-latin1 fallback"

# ───────────────────────── Verify ─────────────────────────
echo "─── Verify ───"
python - <<'PY'
import torch
from importlib import import_module
mods = ['torch', 'torchvision', 'mmpose', 'mmcv', 'mmdet',
        'smplx', 'trimesh', 'pyrender', 'imageio',
        'matplotlib', 'pandas', 'cv2', 'detectron2']
for m in mods:
    try:
        v = import_module(m).__version__ if hasattr(import_module(m), '__version__') else 'ok'
        print(f"  ✓ {m:<12} {v}")
    except Exception as e:
        print(f"  ✗ {m:<12} {type(e).__name__}: {e}")

print(f"\n  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA device:     {torch.cuda.get_device_name(0)}")
    print(f"  CUDA capability: {torch.cuda.get_device_capability(0)}")

try:
    from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
    print(f"  ✓ hmr2 ready, default ckpt: {DEFAULT_CHECKPOINT}")
except Exception as e:
    print(f"  ✗ hmr2 import failed: {e}")
PY

cat <<EOF

─────────────────────────────────────────────────────────────
✅ Linux server setup complete.

Activate:        source $VENV_DIR/bin/activate
4D-Humans repo:  $REPO_DIR
HMR2 cache:      \$HOME/.cache/4DHumans/

Next steps:
  1.  bash demo/scripts/setup_smpl_paths.sh         # symlink SMPL files
  2.  python demo/scripts/run_pipeline.py --help    # CLI demo
  3.  jupyter notebook --no-browser --port 8888     # remote notebook (see SSH tunnel guide)
─────────────────────────────────────────────────────────────
EOF
