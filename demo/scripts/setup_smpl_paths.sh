#!/usr/bin/env bash
# Wire SMPL/SMPL-X model files into the locations smplx, 4D-Humans, and our cache expect.
#
# Pre-conditions (you registered + downloaded the models manually):
#   $CV_MODEL_ROOT/SMPL_python_v.1.1.0/smpl/models/basicmodel_*_lbs_10_207_0_v1.1.0.pkl
#   $CV_MODEL_ROOT/smplx/SMPLX_*.npz
#   $CV_MODEL_ROOT/vposer_v1_0/  (optional, only for stretch path)
#
# Usage:
#   export CV_MODEL_ROOT=/path/to/registered/models
#   bash demo/scripts/setup_smpl_paths.sh
#   bash demo/scripts/setup_smpl_paths.sh --4dhumans-repo ~/4D-Humans

set -euo pipefail

MODEL_ROOT="${CV_MODEL_ROOT:?Set CV_MODEL_ROOT to your registered-models directory}"
FOURD_REPO="${FOURD_REPO:-$HOME/4D-Humans}"
HMR2_CACHE="$HOME/.cache/4DHumans"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --4dhumans-repo) FOURD_REPO="$2"; shift 2 ;;
        --cache)         HMR2_CACHE="$2"; shift 2 ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

[[ -d "$MODEL_ROOT" ]] || { echo "❌ MODEL_ROOT not found: $MODEL_ROOT"; exit 1; }

# ─── 1. smplx library layout: $MODEL_ROOT/smpl/SMPL_<GENDER>.pkl ───
mkdir -p "$MODEL_ROOT/smpl"
cd "$MODEL_ROOT/smpl"
ln -sf ../SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl SMPL_NEUTRAL.pkl
ln -sf ../SMPL_python_v.1.1.0/smpl/models/basicmodel_m_lbs_10_207_0_v1.1.0.pkl       SMPL_MALE.pkl
ln -sf ../SMPL_python_v.1.1.0/smpl/models/basicmodel_f_lbs_10_207_0_v1.1.0.pkl       SMPL_FEMALE.pkl

# ─── 2. 4D-Humans v1.0.0 alias (its README expects this exact filename) ───
cd "$MODEL_ROOT/SMPL_python_v.1.1.0/smpl/models"
ln -sf basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl basicModel_neutral_lbs_10_207_0_v1.0.0.pkl

# ─── 3. 4D-Humans repo data/ dir ───
if [[ -d "$FOURD_REPO" ]]; then
    mkdir -p "$FOURD_REPO/data"
    ln -sf "$MODEL_ROOT/SMPL_python_v.1.1.0/smpl/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl" \
           "$FOURD_REPO/data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
fi

# ─── 4. HMR2 runtime cache (~/.cache/4DHumans/data/smpl/) ───
mkdir -p "$HMR2_CACHE/data/smpl"
ln -sf "$MODEL_ROOT/smpl/SMPL_NEUTRAL.pkl" "$HMR2_CACHE/data/smpl/SMPL_NEUTRAL.pkl"

cat <<EOF
✅ Symlinks placed:
   smplx library: $MODEL_ROOT/smpl/SMPL_<GENDER>.pkl
   4D-Humans v1.0.0 alias: $MODEL_ROOT/SMPL_python_v.1.1.0/smpl/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
   4D-Humans repo: $FOURD_REPO/data/  (skipped if repo absent)
   HMR2 cache: $HMR2_CACHE/data/smpl/SMPL_NEUTRAL.pkl
EOF

ls -la "$MODEL_ROOT/smpl/" 2>&1 | sed 's/^/  /'
