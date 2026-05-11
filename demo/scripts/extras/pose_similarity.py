#!/usr/bin/env python3
"""4×4 SMPL-pose distance heatmap (root-centered, torso-normalized)."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from demo.src import SMPL_NEUTRAL_PKL  # noqa: E402

IMAGES = ["img1_standing", "img2_complex_pose", "img3_occluded", "img4_multi_person"]
LABELS = ["Standing", "Yoga", "Occluded", "Multi-person"]


def load_j_regressor() -> np.ndarray:
    with open(SMPL_NEUTRAL_PKL, "rb") as f:
        smpl = pickle.load(f, encoding="latin1")
    J = smpl["J_regressor"]
    return np.asarray(J.todense() if hasattr(J, "todense") else J)


def pick_center_person(verts_all: np.ndarray) -> np.ndarray:
    if verts_all.shape[0] == 1:
        return verts_all[0]
    centroids = verts_all.mean(axis=1)
    return verts_all[np.argmin(np.linalg.norm(centroids[:, :2], axis=1))]


def normalize_pose(joints_24: np.ndarray) -> np.ndarray:
    rooted = joints_24 - joints_24[0]
    torso = np.linalg.norm(joints_24[12] - joints_24[0])
    return rooted / (torso + 1e-9)


def main():
    npz_path = REPO_ROOT / "demo" / "results" / "hmr2_meshes.npz"
    out_path = REPO_ROOT / "demo" / "showcase" / "extras" / "pose_similarity.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    z = np.load(npz_path)
    Jreg = load_j_regressor()

    poses = []
    for stem in IMAGES:
        verts_all = z[f"{stem}.jpg_verts"]
        joints = Jreg @ pick_center_person(verts_all)
        poses.append(normalize_pose(joints))

    P = np.stack(poses)
    n = len(IMAGES)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = np.sqrt(((P[i] - P[j]) ** 2).sum(axis=-1)).mean()

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    im = ax.imshow(D, cmap="viridis", vmin=0)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(LABELS, rotation=30, ha="right")
    ax.set_yticklabels(LABELS)
    for i in range(n):
        for j in range(n):
            txt_color = "white" if D[i, j] < D.max() * 0.5 else "black"
            ax.text(j, i, f"{D[i, j]:.2f}", ha="center", va="center",
                    color=txt_color, fontsize=11)
    ax.set_title("Pose-feature distance matrix\n(SMPL joints, root-centered, torso-scaled)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean per-joint L2 (torso units)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
