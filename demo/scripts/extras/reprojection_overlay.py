#!/usr/bin/env python3
"""B — Reprojection consistency overlay.

Take HMR 2.0's predicted 3D joints, project them back to original-image
pixel coords using HMR2's `cam_crop_to_full`, and overlay them on the
input image alongside HRNet's direct 2D detections. If the dots line up,
the bridge from S2 ("2D supervises 3D") is empirically self-consistent.

We re-run ViTDet detection here (instead of relying on the cached
hmr2_meshes.npz) because the cache doesn't store bboxes — they're needed
to convert HMR's box-frame cam_t back to full-image coords.

Output: demo/results/extras/reproj_<stem>.png  (one per test image).

Indices map: HRNet uses COCO-17, HMR2 emits 45 joints in OpenPose-25 +
extras layout. We overlay only the 12 reliable correspondences (limbs).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

# COCO-17 → HMR2 OpenPose-25 mapping for the 12 limb joints
COCO_TO_HMR_OP25 = {
    5: 5, 6: 2,           # L/R shoulder
    7: 6, 8: 3,           # L/R elbow
    9: 7, 10: 4,          # L/R wrist
    11: 12, 12: 9,        # L/R hip
    13: 13, 14: 10,       # L/R knee
    15: 14, 16: 11,       # L/R ankle
}

IMAGES = ["img1_standing", "img2_complex_pose", "img3_occluded", "img4_multi_person"]


def cam_crop_to_full_np(cam_bbox: np.ndarray, box_center: np.ndarray,
                        box_size: np.ndarray, img_w: int, img_h: int,
                        focal_length: float = 5000.0) -> np.ndarray:
    """numpy port of hmr2.utils.renderer.cam_crop_to_full (per-detection)."""
    s, tx, ty = cam_bbox
    cx, cy = box_center
    b = box_size
    bs = b * s + 1e-9
    tz = 2.0 * focal_length / bs
    tx_full = (2.0 * (cx - img_w / 2.0) / bs) + tx
    ty_full = (2.0 * (cy - img_h / 2.0) / bs) + ty
    return np.array([tx_full, ty_full, tz], dtype=np.float32)


def project_pinhole(joints_3d: np.ndarray, full_cam_t: np.ndarray,
                    img_w: int, img_h: int, focal: float = 5000.0) -> np.ndarray:
    """Standard pinhole: u = f * (J + t).x / (J+t).z + W/2."""
    Jc = joints_3d + full_cam_t[None, :]
    u = focal * Jc[:, 0] / Jc[:, 2] + img_w / 2.0
    v = focal * Jc[:, 1] / Jc[:, 2] + img_h / 2.0
    return np.stack([u, v], axis=-1)


def run_hmr_with_bbox(images: list[Path], device: str = "cuda"):
    """Re-run detection + HMR2, returning per-image bbox + cam_bbox + joints + verts."""
    from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
    from hmr2.utils import recursive_to
    from hmr2.datasets.vitdet_dataset import ViTDetDataset
    from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
    from detectron2.config import LazyConfig
    import hmr2 as _h

    print("Loading HMR 2.0 + ViTDet …")
    model, cfg = load_hmr2(DEFAULT_CHECKPOINT)
    model = model.to(device).eval()

    cfg_path = Path(_h.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
    det_cfg = LazyConfig.load(str(cfg_path))
    det_cfg.train.init_checkpoint = (
        "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
        "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    )
    for i in range(3):
        det_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.5
    detector = DefaultPredictor_Lazy(det_cfg)

    out = {}
    for img_path in images:
        img_cv2 = cv2.imread(str(img_path))
        H, W = img_cv2.shape[:2]
        det = detector(img_cv2)["instances"]
        cls = det.pred_classes.cpu().numpy()
        boxes = det.pred_boxes.tensor.cpu().numpy()[cls == 0]
        if len(boxes) == 0:
            print(f"  {img_path.name}: no person detected — skipping")
            continue

        ds = ViTDetDataset(cfg, img_cv2, boxes)
        dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)

        joints_all, verts_all, cam_bbox_all, box_centers, box_sizes = [], [], [], [], []
        for i, batch in enumerate(dl):
            batch = recursive_to(batch, device)
            with torch.no_grad():
                pred = model(batch)
            joints_all.append(pred["pred_keypoints_3d"][0].cpu().numpy())  # (45, 3)
            verts_all.append(pred["pred_vertices"][0].cpu().numpy())
            cam_bbox_all.append(pred["pred_cam"][0].cpu().numpy())          # (s, tx, ty)
            box_centers.append(batch["box_center"][0].cpu().numpy())
            box_sizes.append(float(batch["box_size"][0].cpu().numpy()))

        out[img_path.name] = {
            "boxes": boxes, "img_w": W, "img_h": H,
            "joints": np.stack(joints_all),  # (P, 45, 3)
            "verts": np.stack(verts_all),
            "cam_bbox": np.stack(cam_bbox_all),  # (P, 3)
            "box_centers": np.stack(box_centers),  # (P, 2)
            "box_sizes": np.asarray(box_sizes),  # (P,)
        }
        print(f"  {img_path.name}: {len(boxes)} persons")
    return out


def pick_center_idx(boxes: np.ndarray, W: int, H: int) -> int:
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
    dists = np.linalg.norm(centers - np.array([W / 2, H / 2]), axis=-1)
    return int(np.argmin(dists))


def overlay_image(img_rgb: np.ndarray, hrnet_kpts: np.ndarray,
                  hrnet_scores: np.ndarray, hmr_proj_2d: np.ndarray,
                  out_path: Path, title: str):
    """Side-by-side: HRNet kpts (cyan) and HMR-reprojected joints (red)."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_rgb)

    # HRNet COCO-17 (filter low confidence)
    hr_pts = []
    for i in range(17):
        if hrnet_scores[i] > 0.3 and i in COCO_TO_HMR_OP25:
            hr_pts.append((hrnet_kpts[i, 0], hrnet_kpts[i, 1], COCO_TO_HMR_OP25[i]))
    if hr_pts:
        hr_pts = np.asarray(hr_pts)
        ax.scatter(hr_pts[:, 0], hr_pts[:, 1], s=110, marker="o",
                   facecolors="none", edgecolors="cyan", linewidths=2.5,
                   label="HRNet 2D detection")

    # HMR projected (matched joints)
    matched = list(set(int(c) for _, _, c in hr_pts)) if len(hr_pts) else list(set(COCO_TO_HMR_OP25.values()))
    proj_pts = hmr_proj_2d[matched]
    ax.scatter(proj_pts[:, 0], proj_pts[:, 1], s=70, marker="x",
               c="red", linewidths=2.5,
               label="HMR 3D → reprojected 2D")

    # Lines connecting matched pairs (consistency error vector)
    if len(hr_pts):
        for x, y, hmr_idx in hr_pts:
            px, py = hmr_proj_2d[int(hmr_idx)]
            ax.plot([x, px], [y, py], color="yellow", lw=1, alpha=0.6)

    # Mean error annotation
    if len(hr_pts):
        errs = []
        for x, y, hmr_idx in hr_pts:
            px, py = hmr_proj_2d[int(hmr_idx)]
            errs.append(np.hypot(x - px, y - py))
        mean_err = np.mean(errs)
        ax.text(0.02, 0.98, f"mean reproj error: {mean_err:.0f} px",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=11, color="white",
                bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"))

    ax.set_title(title)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ wrote {out_path.relative_to(REPO_ROOT)}")


def main():
    out_dir = REPO_ROOT / "demo" / "results" / "extras"
    out_dir.mkdir(parents=True, exist_ok=True)

    images = [REPO_ROOT / "demo" / "test_images" / f"{s}.jpg" for s in IMAGES]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    hmr_data = run_hmr_with_bbox(images, device=device)

    # Persist a richer cache for downstream extras (mini-SMPLify also reads boxes)
    cache_path = out_dir / "hmr_extras_cache.npz"
    payload = {}
    for name, d in hmr_data.items():
        for k, v in d.items():
            payload[f"{name}_{k}"] = np.asarray(v)
    np.savez_compressed(cache_path, **payload)
    print(f"✓ wrote {cache_path.relative_to(REPO_ROOT)}")

    hrnet = np.load(REPO_ROOT / "demo" / "results" / "hrnet_kpts.npz", allow_pickle=True)

    for img_path in images:
        if img_path.name not in hmr_data:
            continue
        d = hmr_data[img_path.name]
        idx = pick_center_idx(d["boxes"], d["img_w"], d["img_h"])
        joints_3d = d["joints"][idx]
        full_cam_t = cam_crop_to_full_np(
            d["cam_bbox"][idx], d["box_centers"][idx], d["box_sizes"][idx],
            d["img_w"], d["img_h"])
        proj_2d = project_pinhole(joints_3d, full_cam_t, d["img_w"], d["img_h"])

        img_rgb = cv2.imread(str(img_path))[:, :, ::-1]
        hk = hrnet[f"{img_path.name}_kpts"][0]
        hs = hrnet[f"{img_path.name}_scores"][0]

        overlay_image(img_rgb, hk, hs, proj_2d,
                      out_dir / f"reproj_{img_path.stem}.png",
                      title=f"{img_path.stem} — HMR 3D joints reprojected vs HRNet 2D")


if __name__ == "__main__":
    main()
