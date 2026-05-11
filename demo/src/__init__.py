"""Pipeline source — paths to SMPL/SMPL-X assets (set CV_MODEL_ROOT or fall back to <repo>/model)."""
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = Path(os.environ.get("CV_MODEL_ROOT", str(_REPO_ROOT / "model")))

SMPL_NEUTRAL_PKL  = MODEL_ROOT / "smpl"  / "SMPL_NEUTRAL.pkl"
SMPLX_NEUTRAL_NPZ = MODEL_ROOT / "smplx" / "SMPLX_NEUTRAL.npz"
VPOSER_V1_DIR     = MODEL_ROOT / "vposer_v1_0"

__all__ = ["MODEL_ROOT", "SMPL_NEUTRAL_PKL", "SMPLX_NEUTRAL_NPZ", "VPOSER_V1_DIR"]
