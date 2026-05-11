#!/usr/bin/env bash
# Regenerate all demo extras.
# Prerequisites:
#   run_pipeline.py has produced demo/results/{hrnet_kpts,hmr2_meshes}.npz
#   PHALP has produced demo/results/tennis_phalp/{PHALP_tennis.mp4, results/demo_tennis.pkl}
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-$REPO_ROOT/.venv/bin/python}"
SHOWCASE="demo/showcase"
EXTRAS="$SHOWCASE/extras"
VIDEO="$SHOWCASE/video"
PHALP_OUT="demo/results/tennis_phalp"

mkdir -p "$EXTRAS" "$VIDEO"

echo "[1/5] pose similarity heatmap"
"$PY" demo/scripts/extras/pose_similarity.py

echo "[2/5] reprojection consistency overlay (re-runs ViTDet + HMR2)"
"$PY" demo/scripts/extras/reprojection_overlay.py

echo "[3/5] tennis 4D trajectory + joint angles"
"$PY" demo/scripts/extras/tennis_trajectory.py

echo "[4/5] mini-SMPLify (CPU + GPU)"
"$PY" demo/scripts/extras/mini_smplify.py

echo "[5/5] PHALP highlights (frame stills + 5 s clip)"
if [[ -f "$PHALP_OUT/PHALP_tennis.mp4" ]]; then
    cp "$PHALP_OUT/PHALP_tennis.mp4" "$VIDEO/PHALP_tennis.mp4"
    for t in 0.5 2.0 4.0 6.0; do
        ffmpeg -y -ss "$t" -i "$PHALP_OUT/PHALP_tennis.mp4" -frames:v 1 \
            "$VIDEO/frame_${t}s.png" 2>/dev/null
    done
    ffmpeg -y -ss 2.0 -i "$PHALP_OUT/PHALP_tennis.mp4" -t 5.0 \
        -an -vf "scale=1280:720:flags=lanczos" \
        -c:v libx264 -preset fast -crf 23 \
        "$EXTRAS/tennis_clip.mp4" 2>&1 | tail -1
else
    echo "  (skip — $PHALP_OUT/PHALP_tennis.mp4 not found, run PHALP first)"
fi

echo
echo "✓ all outputs in $SHOWCASE/"
ls -lh "$EXTRAS" "$VIDEO"
