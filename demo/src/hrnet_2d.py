"""HRNet-W48 2D keypoint inference with externally-provided bboxes."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np


_COCO_SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
]


def _draw_overlay(img_bgr: np.ndarray, kpts_all: np.ndarray, scores_all: np.ndarray,
                  out_path: Path, kpt_thr: float = 0.3) -> Path:
    import cv2
    img = img_bgr.copy()
    palette = [
        (0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0),
        (255, 255, 0), (255, 0, 0), (255, 0, 255), (180, 105, 255),
    ]
    for pi, (kpts, scores) in enumerate(zip(kpts_all, scores_all)):
        color = palette[pi % len(palette)]
        for a, b in _COCO_SKELETON:
            if scores[a] > kpt_thr and scores[b] > kpt_thr:
                pa = tuple(np.round(kpts[a]).astype(int))
                pb = tuple(np.round(kpts[b]).astype(int))
                cv2.line(img, pa, pb, color, 2, cv2.LINE_AA)
        for j, (x, y) in enumerate(kpts):
            if scores[j] > kpt_thr:
                cv2.circle(img, (int(round(x)), int(round(y))), 4, color, -1, cv2.LINE_AA)
                cv2.circle(img, (int(round(x)), int(round(y))), 5, (255, 255, 255), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return out_path


def init_hrnet(device: str = "cuda"):
    """Load HRNet-W48 via MMPose's model registry. `det_model='whole_image'` disables
    the bundled RTMDet so we can feed external ViTDet boxes via inference_topdown."""
    from mmpose.apis import MMPoseInferencer
    inf = MMPoseInferencer(
        pose2d="td-hm_hrnet-w48_8xb32-210e_coco-256x192",
        det_model="whole_image",
        device=device,
    )
    return inf.inferencer.model


def run_hrnet_with_boxes(
    img_paths: list[str | Path],
    boxes_by_name: dict[str, np.ndarray],
    out_dir: str | Path,
    device: str = "cuda",
) -> dict[str, dict]:
    """Run HRNet-W48 with externally-provided xyxy boxes (shape (N, 4) per image).

    Returns {name: {kpts, scores, vis_path, runtime_s}}.
    """
    import cv2
    from mmpose.apis import inference_topdown

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model = init_hrnet(device)

    results: dict[str, dict] = {}
    for img_path in img_paths:
        img_path = Path(img_path)
        boxes = boxes_by_name.get(img_path.name)
        if boxes is None or len(boxes) == 0:
            results[img_path.name] = {
                "kpts": np.zeros((0, 17, 2)), "scores": np.zeros((0, 17)),
                "vis_path": "", "runtime_s": 0.0,
            }
            continue

        img_bgr = cv2.imread(str(img_path))
        t0 = time.time()
        ds_list = inference_topdown(model, str(img_path), bboxes=boxes[:, :4], bbox_format="xyxy")
        rt = time.time() - t0

        kpts = np.stack([ds.pred_instances.keypoints[0] for ds in ds_list])
        sc   = np.stack([ds.pred_instances.keypoint_scores[0] for ds in ds_list])

        vis_path = out_dir / img_path.name
        _draw_overlay(img_bgr, kpts, sc, vis_path)

        results[img_path.name] = {
            "kpts": kpts, "scores": sc,
            "vis_path": str(vis_path), "runtime_s": rt,
        }
    return results


def save_kpts_npz(results: dict[str, dict], out_npz: str | Path) -> Path:
    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        **{f"{name}_kpts":   v["kpts"]   for name, v in results.items()},
        **{f"{name}_scores": v["scores"] for name, v in results.items()},
    )
    return out_npz
