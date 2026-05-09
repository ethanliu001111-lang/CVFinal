"""SMPLify-X (Pavlakos et al. CVPR 2019) — STRETCH GOAL only.

Hard cutoff per v3 plan: 5/5 EOD.  If installation/runtime fails, this entire
file is skipped and the demo proceeds with HRNet + 4D-Humans only.

SMPLify-X is *not* a pip package; it ships as a script (`smplifyx/main.py`).
This wrapper:
  1. Converts COCO-17 keypoints → BODY_25 OpenPose JSON format
  2. Lays out the directory structure SMPLify-X expects
  3. Calls the CLI entrypoint via subprocess
  4. Loads the per-image .pkl results
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

# COCO-17 → BODY_25 mapping. BODY_25 indices documented at:
# https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md
COCO17_TO_BODY25 = {
    # body_25_idx : coco17_idx
    0: 0,   # nose
    1: 6,   # neck (use right shoulder as proxy if no neck in COCO)
    2: 6, 3: 8, 4: 10,                        # right shoulder, elbow, wrist
    5: 5, 6: 7, 7: 9,                          # left  shoulder, elbow, wrist
    8: 11,                                     # mid-hip = left hip (proxy)
    9: 12, 10: 14, 11: 16,                     # right hip, knee, ankle
    12: 11, 13: 13, 14: 15,                    # left  hip, knee, ankle
    15: 2, 16: 1, 17: 4, 18: 3,                # eyes, ears
    # 19..24: feet — leave zero (COCO-17 doesn't have them)
}


def coco17_to_body25(kpts_17: np.ndarray, conf_17: np.ndarray | None = None) -> np.ndarray:
    """Args: kpts_17 (17, 2) pixel coords, conf_17 (17,) optional.
    Returns: (25, 3) where last col is confidence.
    """
    if conf_17 is None:
        conf_17 = np.ones(17)
    out = np.zeros((25, 3))
    for b25, c17 in COCO17_TO_BODY25.items():
        out[b25, :2] = kpts_17[c17]
        out[b25, 2]  = conf_17[c17]
    return out


def write_smplifyx_input(
    img_paths: list[Path],
    hrnet_results: dict,         # output of run_hrnet_w48
    workdir: str | Path,
) -> Path:
    """Lay out the directory tree SMPLify-X expects:
        workdir/
        ├── images/      ← copies of input images
        └── keypoints/   ← OpenPose JSON
    """
    workdir = Path(workdir)
    (workdir / "images").mkdir(parents=True, exist_ok=True)
    (workdir / "keypoints").mkdir(parents=True, exist_ok=True)

    for ip in img_paths:
        ip = Path(ip)
        if ip.name not in hrnet_results:
            continue
        # 1. copy image
        shutil.copy2(ip, workdir / "images" / ip.name)

        # 2. write keypoints JSON  (one person; SMPLify-X picks the first)
        kp = hrnet_results[ip.name]["kpts"]
        sc = hrnet_results[ip.name]["scores"]
        if len(kp) == 0:
            continue
        body25 = coco17_to_body25(kp[0], sc[0])
        out_json = {"version": 1.3, "people": [{
            "person_id": [-1],
            "pose_keypoints_2d": body25.flatten().tolist(),
            "face_keypoints_2d": [],
            "hand_left_keypoints_2d": [],
            "hand_right_keypoints_2d": [],
            "pose_keypoints_3d": [],
            "face_keypoints_3d": [],
            "hand_left_keypoints_3d": [],
            "hand_right_keypoints_3d": [],
        }]}
        with open(workdir / "keypoints" / f"{ip.stem}_keypoints.json", "w") as f:
            json.dump(out_json, f)

    return workdir


def run_smplifyx_cli(
    workdir: str | Path,
    output_dir: str | Path,
    *,
    smplx_dir: str | Path,        # path to /smplx folder containing SMPLX_NEUTRAL.npz
    vposer_ckpt: str | Path,      # path to vposer_v1_0
    smplifyx_repo: str | Path,    # path to cloned smplify-x repo
    config: str = "fit_smplx.yaml",
) -> dict[str, dict]:
    """Invoke `python smplifyx/main.py ...` via subprocess.  Stretch goal only.

    Returns dict keyed by image stem: {'params': {...}, 'runtime_s': float}.
    """
    workdir, output_dir = Path(workdir), Path(output_dir)
    smplifyx_repo = Path(smplifyx_repo)

    cfg_path = smplifyx_repo / "cfg_files" / config

    t0 = time.time()
    cmd = [
        "python", str(smplifyx_repo / "smplifyx" / "main.py"),
        "--config",        str(cfg_path),
        "--data_folder",   str(workdir),
        "--output_folder", str(output_dir),
        "--visualize",     "False",
        "--model_folder",  str(Path(smplx_dir).parent),     # parent of /smplx
        "--vposer_ckpt",   str(vposer_ckpt),
        "--part_segm_fn",  str(smplifyx_repo / "smplx_parts_segm.pkl"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    total = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"SMPLify-X failed:\n{proc.stderr[-2000:]}")

    # Parse per-image results. Per-image runtime is a coarse average — SMPLify-X
    # itself does not log per-image timings.
    import pickle
    pkls = list((output_dir / "results").rglob("*.pkl"))
    avg_per_image = total / max(len(pkls), 1)
    out: dict[str, dict] = {}
    for pkl in pkls:
        with open(pkl, "rb") as f:
            params = pickle.load(f)
        # SMPLify-X writes ./results/<img_stem>/000.pkl
        out[pkl.parent.name] = {"params": params, "runtime_s": avg_per_image}
    return out
