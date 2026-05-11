#!/usr/bin/env python3
"""Right-wrist 3D trajectory + per-frame joint angles from PHALP's track pickle."""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
PKL_PATH = REPO_ROOT / "demo" / "results" / "tennis_phalp" / "results" / "demo_tennis.pkl"
OUT_DIR = REPO_ROOT / "demo" / "showcase" / "extras"
FPS = 25.0

# HMR2 outputs 45 joints in OpenPose-25 + 19 extras layout
J_R_SHOULDER, J_R_ELBOW, J_R_WRIST = 2, 3, 4
J_R_HIP, J_R_KNEE, J_R_ANKLE = 9, 10, 11


def angle_deg(a, b, c):
    ba = a - b
    bc = c - b
    cosv = (ba * bc).sum(axis=-1) / (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cosv, -1, 1)))


def load_track(pkl_path: Path, target_tid: int = 1):
    d = joblib.load(pkl_path)
    keys = sorted(d.keys())
    frames, joints3d = [], []
    for k in keys:
        frame = d[k]
        if target_tid in frame["tid"]:
            idx = frame["tid"].index(target_tid)
            frames.append(frame["time"])
            joints3d.append(np.asarray(frame["3d_joints"][idx]))
    return np.asarray(frames), np.stack(joints3d)


def plot_wrist_3d(times_s, J, out_path):
    wrist = J[:, J_R_WRIST]
    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(wrist[:, 0], wrist[:, 2], -wrist[:, 1],
                    c=times_s, cmap="plasma", s=10, alpha=0.85)
    ax.plot(wrist[:, 0], wrist[:, 2], -wrist[:, 1], color="gray", alpha=0.3, lw=1)
    ax.set_xlabel("x  (m, lateral)")
    ax.set_ylabel("z  (m, depth)")
    ax.set_zlabel("y  (m, vertical)")
    ax.set_title("Right-wrist 3D trajectory through the serve")
    cb = fig.colorbar(sc, ax=ax, pad=0.1, fraction=0.04)
    cb.set_label("time (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")


def plot_joint_angles(times_s, J, out_path):
    elbow = angle_deg(J[:, J_R_SHOULDER], J[:, J_R_ELBOW], J[:, J_R_WRIST])
    knee = angle_deg(J[:, J_R_HIP], J[:, J_R_KNEE], J[:, J_R_ANKLE])
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(times_s, elbow, label="right elbow", color="#d62728", lw=2)
    ax.plot(times_s, knee, label="right knee", color="#1f77b4", lw=2, alpha=0.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("joint angle (°)")
    ax.set_title("Joint angles extracted per-frame from HMR 2.0 + PHALP")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    ax.set_ylim(0, 200)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames, J = load_track(PKL_PATH, target_tid=1)
    times_s = (frames - frames.min()) / FPS
    print(f"loaded {len(frames)} frames, span {times_s.max():.1f} s, joints shape {J.shape}")
    plot_wrist_3d(times_s, J, OUT_DIR / "tennis_wrist_3d.png")
    plot_joint_angles(times_s, J, OUT_DIR / "tennis_joint_angles.png")


if __name__ == "__main__":
    main()
