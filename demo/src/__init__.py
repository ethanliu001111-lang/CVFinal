"""Pipeline source code for CV final project — From 2D Pose to 3D Human Recovery."""
import os
from pathlib import Path

# Local model registry — points at the team's pre-downloaded SMPL/SMPL-X assets.
# Contains symlinks created by `scripts/setup_smpl_paths.sh`.
# Set CV_MODEL_ROOT env var before running anything that needs SMPL/SMPL-X.
MODEL_ROOT = Path(os.environ.get(
    "CV_MODEL_ROOT",
    str(Path(__file__).resolve().parents[2].parent / "model")  # fallback: ../model relative to repo
))

SMPL_NEUTRAL_PKL  = MODEL_ROOT / "smpl"  / "SMPL_NEUTRAL.pkl"
SMPLX_NEUTRAL_NPZ = MODEL_ROOT / "smplx" / "SMPLX_NEUTRAL.npz"
VPOSER_V1_DIR     = MODEL_ROOT / "vposer_v1_0"

# Repo-relative directories
REPO_ROOT      = Path(__file__).resolve().parent.parent
TEST_IMAGES    = REPO_ROOT / "test_images"
RESULTS        = REPO_ROOT / "results"
CHECKPOINTS    = REPO_ROOT / "checkpoints"

__all__ = [
    "MODEL_ROOT", "SMPL_NEUTRAL_PKL", "SMPLX_NEUTRAL_NPZ", "VPOSER_V1_DIR",
    "REPO_ROOT", "TEST_IMAGES", "RESULTS", "CHECKPOINTS",
]
