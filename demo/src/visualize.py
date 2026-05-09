"""Mesh + keypoint visualization helpers (pyrender on Linux GPU).

Uses pyrender with EGL backend for solid shaded SMPL rendering.
Falls back to matplotlib software render only if pyrender import fails
(e.g. on Mac arm64 without OSMesa).
"""
from __future__ import annotations

import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import pyrender
    import trimesh
    _HAVE_PYRENDER = True
except Exception:
    _HAVE_PYRENDER = False


# ────────────────────────── pyrender path (preferred) ──────────────────────────

def _create_raymond_lights() -> List["pyrender.Node"]:
    """3 raymond lights. Ported from 4D-Humans/hmr2/utils/renderer.py."""
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])
    nodes = []
    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)
        z = np.array([xp, yp, zp]); z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        matrix = np.eye(4)
        matrix[:3, :3] = np.stack([x, y, z], axis=1)
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix,
        ))
    return nodes


def _smpl_to_world(verts: np.ndarray) -> np.ndarray:
    """Flip 180° around X to convert SMPL camera frame → pyrender world frame."""
    R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
    return verts @ R.T


def render_mesh_pyrender(
    verts: np.ndarray,
    faces: np.ndarray,
    *,
    rot_y_deg: float = 0.0,
    img_size: Tuple[int, int] = (512, 512),
    mesh_color: Tuple[float, float, float] = (0.95, 0.85, 0.70),
    bg_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.0),
) -> np.ndarray:
    """Render a single SMPL mesh on a clean background. Returns RGBA uint8 (H, W, 4)."""
    v = _smpl_to_world(np.asarray(verts, dtype=np.float32))

    if rot_y_deg != 0.0:
        c, s = np.cos(np.radians(rot_y_deg)), np.sin(np.radians(rot_y_deg))
        Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        v = v @ Ry.T

    mesh_tri = trimesh.Trimesh(vertices=v, faces=faces, process=False)
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0, roughnessFactor=0.7,
        alphaMode="OPAQUE",
        baseColorFactor=(*mesh_color, 1.0),
    )
    mesh_pr = pyrender.Mesh.from_trimesh(mesh_tri, material=material)

    scene = pyrender.Scene(bg_color=list(bg_color), ambient_light=(0.35, 0.35, 0.35))
    scene.add(mesh_pr)

    # Auto-frame: place camera along +Z so the mesh fills the frame
    bb_min, bb_max = v.min(0), v.max(0)
    center = (bb_min + bb_max) / 2.0
    extent = float((bb_max - bb_min).max())
    yfov = np.pi / 4.0
    cam_distance = (extent / 2.0) / np.tan(yfov / 2.0) * 1.25
    cam_pose = np.eye(4)
    cam_pose[:3, 3] = [center[0], center[1], center[2] + cam_distance]
    aspect = img_size[0] / img_size[1]
    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=aspect)
    scene.add(camera, pose=cam_pose)

    for node in _create_raymond_lights():
        scene.add_node(node)

    renderer = pyrender.OffscreenRenderer(viewport_width=img_size[0], viewport_height=img_size[1])
    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()
    return color


def render_mesh_overlay_pyrender(
    background_rgb: np.ndarray,
    verts_list: List[np.ndarray],
    cam_t_list: List[np.ndarray],
    faces: np.ndarray,
    *,
    focal_length: float = 5000.0,
    mesh_colors: Optional[List[Tuple[float, float, float]]] = None,
) -> np.ndarray:
    """Composite SMPL mesh(es) onto the original input image using HMR2 cam_t.

    Implements the same projection convention as 4D-Humans' Renderer.__call__:
    `camera_translation[0] *= -1` then a 180°-X mesh flip.
    """
    bg = np.asarray(background_rgb)
    H, W = bg.shape[:2]
    bg_f = bg.astype(np.float32) / 255.0 if bg.dtype != np.float32 else bg.copy()

    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=(0.35, 0.35, 0.35))
    flip_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)

    palette = [
        (0.95, 0.85, 0.70), (0.70, 0.85, 0.95),
        (0.85, 0.95, 0.70), (0.95, 0.70, 0.85),
        (0.70, 0.95, 0.85), (0.85, 0.70, 0.95),
    ]
    # Each person has their own cam_t; pyrender supports only one camera per
    # scene, so we instead translate each mesh by (-cam_t) so they line up with
    # a single camera placed at the origin (looking down -Z).
    for i, (verts, cam_t) in enumerate(zip(verts_list, cam_t_list)):
        v = np.asarray(verts, dtype=np.float32) @ flip_x.T
        cam_t_arr = np.asarray(cam_t, dtype=np.float32).copy()
        cam_t_arr[0] *= -1.0  # 4D-Humans renderer convention
        # Translate verts so the mesh ends up in front of a camera at the origin.
        v = v + cam_t_arr[None, :]
        col = (mesh_colors or palette)[i % len(palette)]
        mesh_tri = trimesh.Trimesh(vertices=v, faces=faces, process=False)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0, roughnessFactor=0.7, alphaMode="OPAQUE",
            baseColorFactor=(*col, 1.0))
        mesh_pr = pyrender.Mesh.from_trimesh(mesh_tri, material=material)
        scene.add(mesh_pr)

    # Camera at world origin looking down -Z; pyrender flips Y for image
    # coordinates internally, but `IntrinsicsCamera` follows OpenCV convention
    # (Y down) — match that with a Y-axis-180 rotation on the camera pose.
    cam_pose = np.eye(4)
    cam_pose[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
    camera = pyrender.IntrinsicsCamera(
        fx=focal_length, fy=focal_length, cx=W / 2.0, cy=H / 2.0, zfar=1e12,
    )
    scene.add(camera, pose=cam_pose)
    for node in _create_raymond_lights():
        scene.add_node(node)

    renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()

    color_f = color.astype(np.float32) / 255.0
    alpha = color_f[..., 3:4]
    rgb = color_f[..., :3] * alpha + bg_f * (1 - alpha)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


# ────────────────────────── matplotlib fallback (Mac / no GPU) ──────────────────────────

def smpl_y_up_to_matplotlib_z_up(verts: np.ndarray) -> np.ndarray:
    """Convert SMPL Y-up vertices to matplotlib's Z-up axis convention."""
    verts = np.asarray(verts)
    return np.stack([verts[:, 0], verts[:, 2], verts[:, 1]], axis=-1)


_smpl_y_up_to_matplotlib_z_up = smpl_y_up_to_matplotlib_z_up


def _viewbox(verts: np.ndarray, pad: float = 0.1):
    lo, hi = verts.min(0), verts.max(0)
    c = (lo + hi) / 2; r = (hi - lo).max() / 2 * (1 + pad)
    return float(c[0]-r), float(c[1]-r), float(c[2]-r), float(c[0]+r), float(c[1]+r), float(c[2]+r)


def render_mesh_matplotlib(
    verts: np.ndarray, faces: np.ndarray, *,
    azim: float = 45.0, elev: float = 10.0,
    color: str = "steelblue", figsize: Tuple[int, int] = (6, 6),
    face_stride: int = 1,
) -> np.ndarray:
    """Software 3D mesh render — slow, only used as Mac fallback."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    v = _smpl_y_up_to_matplotlib_z_up(verts)
    fig = plt.figure(figsize=figsize, dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    f = faces[::face_stride]
    poly = Poly3DCollection(v[f], alpha=0.9, facecolor=color, edgecolor="none")
    ax.add_collection3d(poly)
    x0, y0, z0, x1, y1, z1 = _viewbox(v)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_zlim(z0, z1)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=elev, azim=azim); ax.set_axis_off()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return img


def _render_view(verts, faces, *, rot_y_deg: float, img_size=(512, 512)) -> np.ndarray:
    """Backend-aware mesh render — pyrender if available, else matplotlib."""
    if _HAVE_PYRENDER:
        return render_mesh_pyrender(verts, faces, rot_y_deg=rot_y_deg, img_size=img_size)
    return render_mesh_matplotlib(verts, faces, azim=rot_y_deg, elev=10, figsize=(img_size[0]/120, img_size[1]/120))


# ────────────────────────── public API (used by run_pipeline.py) ──────────────────────────

def quad_plot(
    input_rgb: np.ndarray,
    overlay_2d: np.ndarray,
    verts: np.ndarray,
    faces: np.ndarray,
    out_png: str | Path,
    title_prefix: str = "",
    *,
    cam_t: Optional[np.ndarray] = None,
    verts_all: Optional[np.ndarray] = None,
    cam_t_all: Optional[np.ndarray] = None,
) -> Path:
    """2×2 grid: Input | 2D overlay // 3D-overlaid-on-input | 3D side.

    `verts_all`+`cam_t_all` (optional, multi-person): if given, the bottom-left
    panel will overlay all detected meshes on the input image. Otherwise falls
    back to a clean front view of the single `verts`.
    """
    H, W = input_rgb.shape[:2]
    front = _render_view(verts, faces, rot_y_deg=0.0,  img_size=(640, 640))
    side  = _render_view(verts, faces, rot_y_deg=90.0, img_size=(640, 640))
    front_title = f"{title_prefix}HMR 2.0 (Front)"

    fig, ax = plt.subplots(2, 2, figsize=(12, 12), dpi=120)
    ax[0, 0].imshow(input_rgb);  ax[0, 0].set_title(f"{title_prefix}Input");          ax[0, 0].axis("off")
    ax[0, 1].imshow(overlay_2d); ax[0, 1].set_title(f"{title_prefix}HRNet 2D");       ax[0, 1].axis("off")
    ax[1, 0].imshow(front);      ax[1, 0].set_title(front_title);                     ax[1, 0].axis("off")
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
    n_frames: int = 24,
    fps: int = 12,
    img_size: Tuple[int, int] = (512, 512),
) -> Path:
    """Render a 360° rotating mesh as an animated GIF."""
    import imageio.v3 as iio
    angles = np.linspace(0, 360, n_frames, endpoint=False)
    frames = [_render_view(verts, faces, rot_y_deg=float(a), img_size=img_size)[..., :3] for a in angles]
    out_gif = Path(out_gif)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_gif, frames, duration=1000 // fps, loop=0)
    return out_gif


def pick_center_person(verts_all: np.ndarray, cam_t_all: np.ndarray, img_shape) -> Optional[int]:
    """Return idx of detection whose camera projection is closest to image center."""
    if len(verts_all) == 0:
        return None
    if len(verts_all) == 1:
        return 0
    H, W = img_shape[:2]; norm = max(W, H)
    center = np.array([W / 2 / norm, H / 2 / norm])
    dists = [np.linalg.norm(np.asarray(t)[:2] - center) for t in cam_t_all]
    return int(np.argmin(dists))
