#!/usr/bin/env bash
# Generate every "presentation extra" asset in one go.
#
# Outputs (under demo/results/extras/):
#   pose_similarity.png             — C: 4×4 SMPL-pose distance matrix
#   reproj_<stem>.png   (×4)        — B: HMR 3D joints reprojected vs HRNet 2D
#   tennis_wrist_3d.png             — D: right-wrist 3D trajectory (time-coloured)
#   tennis_joint_angles.png         — D: right-elbow / right-knee angles over time
#   mini_smplify.png                — A: SMPLify-style fit on the 4 images
#   mini_smplify_convergence.png    — A: per-image loss curves
#   mini_smplify_runtime.csv        — A: timing comparison vs HMR
#   tennis_clip.mp4                 — 5-second loop for slide 5
#   hmr_extras_cache.npz            — bbox+cam cache used by B and A
#
# Prerequisites: HRNet + HMR pipeline already ran (hrnet_kpts.npz +
# hmr2_meshes.npz exist under demo/results/).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export CV_MODEL_ROOT="${CV_MODEL_ROOT:-$REPO_ROOT/model}"
PY="$REPO_ROOT/.venv/bin/python"
EXTRAS_DIR="demo/results/extras"
mkdir -p "$EXTRAS_DIR"

echo "▶ C — pose similarity heatmap"
"$PY" demo/scripts/extras/pose_similarity.py

echo
echo "▶ B — reprojection consistency overlay (re-runs ViTDet + HMR2, ~3 min)"
"$PY" demo/scripts/extras/reprojection_overlay.py

echo
echo "▶ D — tennis 4D trajectory + joint angles"
"$PY" demo/scripts/extras/tennis_trajectory.py

echo
echo "▶ A — mini-SMPLify (CPU + GPU pass, ~30 s)"
"$PY" demo/scripts/extras/mini_smplify.py

echo
echo "▶ ffmpeg — 5-second tennis clip"
ffmpeg -y -ss 2.0 -i demo/results/tennis_phalp/PHALP_tennis.mp4 -t 5.0 \
    -an -vf "scale=1280:720:flags=lanczos" \
    -c:v libx264 -preset fast -crf 23 \
    "$EXTRAS_DIR/tennis_clip.mp4" 2>&1 | tail -1

echo
echo "✅ All extras written to $EXTRAS_DIR/"
ls -lh "$EXTRAS_DIR/"
