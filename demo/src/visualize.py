"""Mesh + keypoint visualization helpers.

Three rendering backends:
  - matplotlib (always works, no GPU, slow)  — used on Mac / CI
  - pyrender (requires osmesa headless on Colab/Linux) — used in main pipeline
  - trimesh.scene.show — interactive only, not used here

The quad-plot layout follows v3 plan §F: 2x2 grid (Input / 2D-overlay /
3D-front / 3D-rotating).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _viewbox(verts: np.ndarray, pad: float = 0.1) -> tuple[float, float, float, float, float, float]:
    """Return (x0, y0, z0, x1, y1, z1) for a cubic axis bounding box."""
    lo, hi = verts.min(0), verts.max(0)
    c = (lo + hi) / 2
    r = (hi - lo).max() / 2 * (1 + pad)
    return float(c[0] - r), float(c[1] - r), float(c[2] - r), float(c[0] + r), float(c[1] + r), float(c[2] + r)


def render_mesh_matplotlib(
    verts: np.ndarray, faces: np.ndarray, *,
    azim: float = 45.0, elev: float = 10.0,
    color: str = "steelblue", figsize: tuple[int, int] = (6, 6),
    face_stride: int = 8,   # display every Nth face for speed
) -> np.ndarray:
    """Software 3D mesh render → returns RGBA numpy array (H, W, 4)."""
    fig = plt.figure(figsize=figsize, dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    f = faces[::face_stride]
    poly = Poly3DCollection(verts[f], alpha=0.4, facecolor=color, edgecolor="gray", linewidth=0.1)
    ax.add_collection3d(poly)

    x0, y0, z0, x1, y1, z1 = _viewbox(verts)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_zlim(z0, z1)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return img


def quad_plot(
    input_rgb: np.ndarray,           # (H, W, 3) original image
    overlay_2d: np.ndarray,          # (H, W, 3) HRNet 2D keypoints overlay
    verts: np.ndarray,               # (V, 3) SMPL/SMPL-X vertices
    faces: np.ndarray,               # (F, 3)
    out_png: str | Path,
    title_prefix: str = "",
) -> Path:
    """2×2 grid: Input | 2D overlay // 3D front | 3D side."""
    front = render_mesh_matplotlib(verts, faces, azim=0,  elev=10)
    side  = render_mesh_matplotlib(verts, faces, azim=90, elev=10)

    fig, ax = plt.subplots(2, 2, figsize=(12, 12), dpi=120)
    ax[0, 0].imshow(input_rgb);  ax[0, 0].set_title(f"{title_prefix}Input");          ax[0, 0].axis("off")
    ax[0, 1].imshow(overlay_2d); ax[0, 1].set_title(f"{title_prefix}HRNet 2D");       ax[0, 1].axis("off")
    ax[1, 0].imshow(front);      ax[1, 0].set_title(f"{title_prefix}HMR 2.0 (Front)");ax[1, 0].axis("off")
    ax[1, 1].imshow(side);       ax[1, 1].set_title(f"{title_prefix}HMR 2.0 (Side)"); ax[1, 1].axis("off")

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return out_png


def rotation_gif(
    verts: np.ndarray,
    faces: np.ndarray,
    out_gif: str | Path,
    *,
    n_frames: int = 15,
    fps: int = 10,
) -> Path:
    """Render a 360° rotating mesh as an animated GIF."""
    import imageio.v3 as iio

    angles = np.linspace(0, 360, n_frames, endpoint=False)
    frames = [
        render_mesh_matplotlib(verts, faces, azim=float(a), elev=10, figsize=(4, 4))[:, :, :3]
        for a in angles
    ]

    out_gif = Path(out_gif)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_gif, frames, duration=1000 // fps, loop=0)
    return out_gif


def pick_center_person(verts_all: np.ndarray, cam_t_all: np.ndarray, img_shape: tuple[int, int, int]) -> int | None:
    """Multi-person guard: pick the detection whose camera projection is closest to image center.

    Avoids matplotlib silently rendering the wrong person on multi-person images
    (gemini round-2 finding).
    """
    if len(verts_all) == 0:
        return None
    if len(verts_all) == 1:
        return 0
    H, W = img_shape[:2]
    norm = max(W, H)
    center = np.array([W / 2 / norm, H / 2 / norm])
    dists = [np.linalg.norm(np.asarray(t)[:2] - center) for t in cam_t_all]
    return int(np.argmin(dists))
