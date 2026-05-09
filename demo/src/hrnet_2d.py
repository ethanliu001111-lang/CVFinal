"""HRNet (CVPR 2019) 2D keypoint inference via MMPose.

Returns COCO-17 keypoints + a visualization overlay image.
For Colab: install via openmim (see envs/env_hrnet_hmr2.yml).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np


def run_hrnet_w48(
    img_paths: list[str | Path],
    out_dir: str | Path = "results/hrnet_vis",
    device: str = "cuda",
) -> dict[str, dict]:
    """Run HRNet-W48 top-down on a list of images.

    Returns a dict keyed by image stem:
        {
          'kpts': np.ndarray (N_persons, 17, 2),
          'scores': np.ndarray (N_persons, 17),
          'vis_path': str,        # path to overlay image
          'runtime_s': float,     # end-to-end inference time
        }
    """
    from mmpose.apis import MMPoseInferencer

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    inferencer = MMPoseInferencer(
        pose2d="td-hm_hrnet-w48_8xb32-210e_coco-256x192",
        det_model="rtmdet-m",
        device=device,
    )

    results: dict[str, dict] = {}
    for img_path in img_paths:
        img_path = Path(img_path)
        t0 = time.time()
        gen = inferencer(str(img_path), return_vis=True, vis_out_dir=str(out_dir))
        result = next(gen)
        rt = time.time() - t0

        preds = result["predictions"][0]   # list of detected people
        if not preds:
            results[img_path.name] = {"kpts": np.zeros((0, 17, 2)),
                                       "scores": np.zeros((0, 17)),
                                       "vis_path": "",
                                       "runtime_s": rt}
            continue

        kpts = np.array([p["keypoints"]       for p in preds])
        sc   = np.array([p["keypoint_scores"] for p in preds])
        vis_path = next(out_dir.glob(f"{img_path.stem}*"), None)
        results[img_path.name] = {
            "kpts": kpts, "scores": sc,
            "vis_path": str(vis_path) if vis_path else "",
            "runtime_s": rt,
        }

    return results


def save_kpts_npz(results: dict[str, dict], out_npz: str | Path) -> Path:
    """Persist 2D keypoints for downstream stages (avoids re-running HRNet)."""
    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        **{f"{name}_kpts":   v["kpts"]   for name, v in results.items()},
        **{f"{name}_scores": v["scores"] for name, v in results.items()},
    )
    return out_npz
