#!/usr/bin/env python3
"""HMR 3D joints reprojected to image space, overlaid against HRNet 2D detections.

Re-runs ViTDet + HMR (instead of using the cached hmr2_meshes.npz) because the
cache doesn't store bboxes — needed to project box-frame cam_t back to full image.
Writes an extras cache (with bbox info) that mini_smplify.py also consumes.
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


# COCO-17 → HMR2 OpenPose-25 mapping for the 12 limb correspondences
COCO_TO_HMR_OP25 = {
    5: 5, 6: 2, 7: 6, 8: 3, 9: 7, 10: 4,
    11: 12, 12: 9, 13: 13, 14: 10, 15: 14, 16: 11,
}
IMAGES = ["img1_standing", "img2_complex_pose", "img3_occluded", "img4_multi_person"]


def cam_crop_to_full_np(cam_bbox, box_center, box_size, img_w, img_h, focal_length=5000.0):
    s, tx, ty = cam_bbox
    cx, cy = box_center
    bs = box_size * s + 1e-9
    return np.array([
        (2.0 * (cx - img_w / 2.0) / bs) + tx,
        (2.0 * (cy - img_h / 2.0) / bs) + ty,
        2.0 * focal_length / bs,
    ], dtype=np.float32)


def project_pinhole(joints_3d, full_cam_t, img_w, img_h, focal=5000.0):
    Jc = joints_3d + full_cam_t[None, :]
    u = focal * Jc[:, 0] / Jc[:, 2] + img_w / 2.0
    v = focal * Jc[:, 1] / Jc[:, 2] + img_h / 2.0
    return np.stack([u, v], axis=-1)


def run_hmr_with_bbox(images, device="cuda"):
    from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
    from hmr2.utils import recursive_to
    from hmr2.datasets.vitdet_dataset import ViTDetDataset
    from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
    from detectron2.config import LazyConfig
    import hmr2 as _h

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
            print(f"  {img_path.name}: no person")
            continue

        ds = ViTDetDataset(cfg, img_cv2, boxes)
        dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
        joints_all, verts_all, cam_bbox_all, box_centers, box_sizes = [], [], [], [], []
        for batch in dl:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                pred = model(batch)
            joints_all.append(pred["pred_keypoints_3d"][0].cpu().numpy())
            verts_all.append(pred["pred_vertices"][0].cpu().numpy())
            cam_bbox_all.append(pred["pred_cam"][0].cpu().numpy())
            box_centers.append(batch["box_center"][0].cpu().numpy())
            box_sizes.append(float(batch["box_size"][0].cpu().numpy()))

        out[img_path.name] = {
            "boxes": boxes, "img_w": W, "img_h": H,
            "joints": np.stack(joints_all),
            "verts": np.stack(verts_all),
            "cam_bbox": np.stack(cam_bbox_all),
            "box_centers": np.stack(box_centers),
            "box_sizes": np.asarray(box_sizes),
        }
        print(f"  {img_path.name}: {len(boxes)} persons")
    return out


def pick_center_idx(boxes, W, H):
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
    return int(np.argmin(np.linalg.norm(centers - np.array([W / 2, H / 2]), axis=-1)))


def overlay_image(img_rgb, hrnet_kpts, hrnet_scores, hmr_proj_2d, out_path, title):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_rgb)

    hr_pts = []
    for i in range(17):
        if hrnet_scores[i] > 0.3 and i in COCO_TO_HMR_OP25:
            hr_pts.append((hrnet_kpts[i, 0], hrnet_kpts[i, 1], COCO_TO_HMR_OP25[i]))
    if hr_pts:
        hr_arr = np.asarray(hr_pts)
        ax.scatter(hr_arr[:, 0], hr_arr[:, 1], s=110, marker="o",
                   facecolors="none", edgecolors="cyan", linewidths=2.5,
                   label="HRNet 2D detection")

    matched = list(set(int(c) for _, _, c in hr_pts)) if hr_pts else list(set(COCO_TO_HMR_OP25.values()))
    proj_pts = hmr_proj_2d[matched]
    ax.scatter(proj_pts[:, 0], proj_pts[:, 1], s=70, marker="x",
               c="red", linewidths=2.5, label="HMR 3D → reprojected 2D")

    if hr_pts:
        errs = []
        for x, y, hmr_idx in hr_pts:
            px, py = hmr_proj_2d[int(hmr_idx)]
            ax.plot([x, px], [y, py], color="yellow", lw=1, alpha=0.6)
            errs.append(np.hypot(x - px, y - py))
        ax.text(0.02, 0.98, f"mean reproj error: {np.mean(errs):.0f} px",
                transform=ax.transAxes, va="top", ha="left", fontsize=11, color="white",
                bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"))

    ax.set_title(title)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")


def main():
    cache_dir = REPO_ROOT / "demo" / "results"
    out_dir = REPO_ROOT / "demo" / "showcase" / "extras"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = [REPO_ROOT / "demo" / "test_images" / f"{s}.jpg" for s in IMAGES]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hmr_data = run_hmr_with_bbox(images, device=device)

    cache_path = cache_dir / "hmr_extras_cache.npz"
    payload = {}
    for name, d in hmr_data.items():
        for k, v in d.items():
            payload[f"{name}_{k}"] = np.asarray(v)
    np.savez_compressed(cache_path, **payload)
    print(f"wrote {cache_path.relative_to(REPO_ROOT)}")

    hrnet = np.load(cache_dir / "hrnet_kpts.npz", allow_pickle=True)
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
