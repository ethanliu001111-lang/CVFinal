#!/usr/bin/env bash
# Build a self-contained release tarball ready to ship to a server or attach to
# the Brightspace submission.
#
# Output:  ../<GroupName>_Code.tar.gz   (or .zip if --zip)
# Excludes:
#   - .venv, .git, __pycache__, *.pyc
#   - LaTeX intermediates (*.aux, *.bbl, *.blg, *.fls, *.fdb_latexmk, *.log, *.out)
#   - All model checkpoints (*.pkl, *.npz, *.ckpt, *.pth, *.pt)
#   - macOS Finder metadata (.DS_Store, ._*)
#
# Includes:
#   - All source code (demo/src, demo/scripts, demo/notebooks)
#   - The compiled report PDF (LiteratureReview/main.pdf)
#   - LaTeX source (.tex / .bib)
#   - README.md + LICENSE + .gitignore
#   - test_images/ (if any committed)
#   - demo/results/*.png (visual baselines, NOT *.gif/*.mp4)
#
# Usage:
#   bash demo/scripts/make_release_tarball.sh                          # default name
#   bash demo/scripts/make_release_tarball.sh --name Liu-Liang-PoseRecovery
#   bash demo/scripts/make_release_tarball.sh --zip                    # produce .zip instead

set -euo pipefail
cd "$(dirname "$0")/../.."

NAME="Liu-Liang-PoseRecovery_Code"
FORMAT="tar.gz"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name) NAME="$2"; shift 2 ;;
        --zip)  FORMAT="zip"; shift ;;
        *)      echo "unknown arg: $1"; exit 1 ;;
    esac
done

OUT_DIR="$(pwd)/../"
STAMP="$(date +%Y%m%d-%H%M)"
ARCHIVE="${OUT_DIR%/}/${NAME}_${STAMP}.${FORMAT}"

# Build a clean staging copy via rsync (avoids modifying the working tree)
STAGE="$(mktemp -d)/${NAME}"
mkdir -p "$STAGE"

rsync -a \
    --exclude='.venv' --exclude='.git' --exclude='.claude' \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='*.pyo' \
    --exclude='.ipynb_checkpoints' \
    --exclude='*.aux' --exclude='*.bbl' --exclude='*.blg' \
    --exclude='*.fdb_latexmk' --exclude='*.fls' --exclude='*.log' --exclude='*.out' --exclude='*.toc' --exclude='*.synctex.gz' \
    --exclude='*.pkl' --exclude='*.npz' --exclude='*.ckpt' --exclude='*.pth' --exclude='*.pt' --exclude='*.bin' \
    --exclude='*.gif' --exclude='*.mp4' \
    --exclude='.DS_Store' --exclude='._*' \
    --exclude='hrnet_kpts.npz' --exclude='hmr2_meshes.npz' --exclude='2d_vis' \
    --exclude='papers' \
    --exclude='CV_Project_Instructions.pdf' \
    ./ "$STAGE/"

# Sanity print
echo "─── Staged contents ───"
find "$STAGE" -type f | sort | sed "s|$STAGE/||" | head -50
echo "..."
echo
echo "Total files: $(find "$STAGE" -type f | wc -l | tr -d ' ')"
echo "Total size:  $(du -sh "$STAGE" | cut -f1)"

# Pack
case "$FORMAT" in
    tar.gz)
        ( cd "$(dirname "$STAGE")" && tar -czf "$ARCHIVE" "$(basename "$STAGE")" )
        ;;
    zip)
        ( cd "$(dirname "$STAGE")" && zip -qr "$ARCHIVE" "$(basename "$STAGE")" )
        ;;
esac

rm -rf "$(dirname "$STAGE")"

echo
echo "✅ Release archive: $ARCHIVE"
ls -lh "$ARCHIVE"

cat <<EOF

─── Next steps ───
Submission (Brightspace):
    Use this exact archive as the GroupName_Code submission.

Server deploy:
    scp "$ARCHIVE" user@server:~/
    ssh user@server "tar -xzf $(basename "$ARCHIVE") && cd ${NAME} && bash demo/scripts/setup_linux_server.sh"

Verify integrity:
    Re-run pipeline from a fresh extract:
        cd /tmp && tar -xzf "$ARCHIVE" && cd "${NAME}" && bash demo/scripts/run_smoke_test.sh
EOF
