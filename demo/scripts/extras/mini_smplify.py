#!/usr/bin/env python3
"""A — Mini SMPLify: empirically reproduce HMR's "regression vs optimization" gap.

Standard SMPLify: given 2D keypoints + a SMPL model, minimize the
reprojection error wrt (θ, β, transl) by gradient descent. We do exactly
this on each test image, using HRNet's 2D output as the target. We then
compare runtime to HMR 2.0's single forward pass (~100 ms on the same GPU).

Why it matters: HMR's headline 600× claim is empirical. Re-running the
optimization side ourselves means our presentation cites a number we
measured, not a number copied from a paper.

Two-stage Adam (SMPLify-style coarse-to-fine):
    Stage 1 (60 it, lr 0.05):  global_orient + transl only
    Stage 2 (240 it, lr 0.02): all parameters jointly

Output: demo/results/extras/mini_smplify.png        — 4-panel summary
        demo/results/extras/mini_smplify_runtime.csv — per-image timing
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from demo.src import MODEL_ROOT  # noqa: E402

import smplx  # noqa: E402

IMAGES = ["img1_standing", "img2_complex_pose", "img3_occluded", "img4_multi_person"]

# COCO-17 (HRNet) → SMPL 24 joint indices, for the 12 limb correspondences.
# Skipping head joints (nose/eyes/ears) — they need extra_joints regressors
# we don't pull in here.
COCO_TO_SMPL = {
    5: 16, 6: 17,         # L/R shoulder
    7: 18, 8: 19,         # L/R elbow
    9: 20, 10: 21,        # L/R wrist
    11: 1, 12: 2,         # L/R hip
    13: 4, 14: 5,         # L/R knee
    15: 7, 16: 8,         # L/R ankle
}

FOCAL = 5000.0
HMR_RUNTIME_MS = 100.0  # measured on the same RTX PRO 6000

# Run SMPLify on both devices to bracket the regression-vs-optimization gap.
# Paper reports 600× (CPU SMPLify ~60s vs HMR forward). On modern GPU the
# ratio shrinks significantly — that's worth showing honestly.
DEVICES = ["cuda", "cpu"]


def project(joints_3d: torch.Tensor, transl: torch.Tensor,
            img_w: int, img_h: int, focal: float = FOCAL) -> torch.Tensor:
    """Pinhole projection: (J + transl) into pixel coords."""
    Jc = joints_3d + transl[None, :]
    u = focal * Jc[:, 0] / Jc[:, 2] + img_w / 2.0
    v = focal * Jc[:, 1] / Jc[:, 2] + img_h / 2.0
    return torch.stack([u, v], dim=-1)


def fit_one(target_2d: torch.Tensor, scores: torch.Tensor, img_w: int, img_h: int,
            smpl: smplx.SMPL, device: str = "cuda") -> dict:
    """Fit SMPL to a single set of HRNet 2D keypoints. Returns timings + losses."""
    coco_idx = sorted(COCO_TO_SMPL.keys())
    smpl_idx = [COCO_TO_SMPL[c] for c in coco_idx]

    target = target_2d[coco_idx].to(device)             # (12, 2)
    weights = scores[coco_idx].to(device).clamp(min=0)  # (12,)
    weights = weights / (weights.sum() + 1e-9)

    # Init from rest (T-pose, betas=0, depth ≈ 5m so the projected mesh fits image)
    global_orient = torch.zeros(1, 3, device=device, requires_grad=True)
    body_pose = torch.zeros(1, 69, device=device, requires_grad=True)
    betas = torch.zeros(1, 10, device=device, requires_grad=True)
    transl = torch.tensor([[0.0, 0.0, 5.0]], device=device, requires_grad=True)

    def forward_joints():
        out = smpl(global_orient=global_orient, body_pose=body_pose,
                   betas=betas, transl=torch.zeros_like(transl))
        return out.joints[0]  # (24+ , 3)

    # Stage 1: orient + transl only, 60 iters
    losses = []
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.time()

    opt1 = torch.optim.Adam([global_orient, transl], lr=0.05)
    for it in range(60):
        opt1.zero_grad()
        joints = forward_joints()[smpl_idx]
        proj = project(joints, transl[0], img_w, img_h)
        err = ((proj - target) ** 2).sum(dim=-1)
        loss = (err * weights).sum()
        loss.backward()
        opt1.step()
        losses.append(loss.item())

    # Stage 2: all params, 240 iters
    opt2 = torch.optim.Adam(
        [{"params": [global_orient, transl], "lr": 0.02},
         {"params": [body_pose], "lr": 0.04},
         {"params": [betas], "lr": 0.01}])
    for it in range(240):
        opt2.zero_grad()
        joints = forward_joints()[smpl_idx]
        proj = project(joints, transl[0], img_w, img_h)
        err = ((proj - target) ** 2).sum(dim=-1)
        # Light regularization to prevent unrealistic poses
        reg = 0.001 * (body_pose ** 2).sum() + 0.01 * (betas ** 2).sum()
        loss = (err * weights).sum() + reg
        loss.backward()
        opt2.step()
        losses.append(loss.item())

    if torch.cuda.is_available(): torch.cuda.synchronize()
    runtime_s = time.time() - t0

    # Final reprojection for plotting
    with torch.no_grad():
        joints_final = forward_joints()
        proj_final = project(joints_final, transl[0], img_w, img_h).cpu().numpy()
        # Per-keypoint pixel error on matched joints
        per_joint_err = ((proj_final[smpl_idx] - target.cpu().numpy()) ** 2).sum(-1) ** 0.5

    return {
        "runtime_s": runtime_s,
        "losses": losses,
        "final_proj_2d": proj_final,             # (24, 2) all SMPL joints
        "matched_smpl_idx": smpl_idx,
        "matched_target_2d": target.cpu().numpy(),
        "matched_weights": weights.cpu().numpy(),
        "per_joint_px_err": per_joint_err,
        "final_loss": losses[-1],
    }


def plot_summary(results: dict[str, dict], img_paths: dict[str, Path], out_path: Path):
    """4-panel grid: each panel shows the image with HRNet kpts + SMPLify-fitted joints."""
    n = len(results)
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()
    for ax, (stem, r) in zip(axes, results.items()):
        img = np.array(Image.open(img_paths[stem]))
        ax.imshow(img)
        # HRNet target
        target = r["matched_target_2d"]
        ax.scatter(target[:, 0], target[:, 1], s=120, marker="o",
                   facecolors="none", edgecolors="cyan", linewidths=2.5,
                   label="HRNet target")
        # SMPLify final
        smplify_pts = r["final_proj_2d"][r["matched_smpl_idx"]]
        ax.scatter(smplify_pts[:, 0], smplify_pts[:, 1], s=70, marker="x",
                   c="red", linewidths=2.5, label="SMPLify (ours)")
        # Error line
        for (tx, ty), (sx, sy) in zip(target, smplify_pts):
            ax.plot([tx, sx], [ty, sy], color="yellow", lw=1, alpha=0.5)
        ax.set_title(f"{stem}  —  {r['runtime_s']:.2f} s   ({r['runtime_s']*1000/HMR_RUNTIME_MS:.0f}× HMR)",
                     fontsize=12)
        mean_err = r["per_joint_px_err"].mean()
        ax.text(0.02, 0.98, f"mean reproj err: {mean_err:.0f} px",
                transform=ax.transAxes, va="top", color="white", fontsize=11,
                bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"))
        ax.set_axis_off()
        ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    fig.suptitle("Mini-SMPLify (our 80-line PyTorch reimplementation)\n"
                 "Same target as HRNet; runtime measured on RTX PRO 6000",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ wrote {out_path.relative_to(REPO_ROOT)}")


def write_runtime_csv_combined(all_results: dict[str, dict[str, dict]], out_path: Path):
    """Write CSV with one row per (image, device) pair."""
    import csv
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "device", "smplify_runtime_s", "hmr_runtime_ms",
                    "speedup_factor", "mean_reproj_error_px", "final_loss"])
        for device, results in all_results.items():
            for stem, r in results.items():
                w.writerow([stem, device, f"{r['runtime_s']:.3f}", HMR_RUNTIME_MS,
                            f"{r['runtime_s']*1000/HMR_RUNTIME_MS:.1f}",
                            f"{r['per_joint_px_err'].mean():.1f}",
                            f"{r['final_loss']:.1f}"])
    print(f"✓ wrote {out_path.relative_to(REPO_ROOT)}")


def main():
    out_dir = REPO_ROOT / "demo" / "results" / "extras"
    out_dir.mkdir(parents=True, exist_ok=True)

    devices_to_run = [d for d in DEVICES if d != "cuda" or torch.cuda.is_available()]
    hrnet = np.load(REPO_ROOT / "demo" / "results" / "hrnet_kpts.npz")
    extras_cache = np.load(out_dir / "hmr_extras_cache.npz")
    img_paths = {s: REPO_ROOT / "demo" / "test_images" / f"{s}.jpg" for s in IMAGES}

    all_results: dict[str, dict[str, dict]] = {}
    for device in devices_to_run:
        print(f"\n=== Running on {device} ===")
        smpl = smplx.create(str(MODEL_ROOT), model_type="smpl",
                            gender="neutral", batch_size=1).to(device)
        smpl.eval()
        for p in smpl.parameters():
            p.requires_grad_(False)

        results: dict[str, dict] = {}
        for stem in IMAGES:
            name = f"{stem}.jpg"
            img_w = int(extras_cache[f"{name}_img_w"])
            img_h = int(extras_cache[f"{name}_img_h"])
            person_idx = 0
            target_2d = torch.tensor(hrnet[f"{name}_kpts"][person_idx], dtype=torch.float32)
            scores = torch.tensor(hrnet[f"{name}_scores"][person_idx], dtype=torch.float32)
            print(f"  fitting {stem} ({img_w}x{img_h}) …", end=" ", flush=True)
            r = fit_one(target_2d, scores, img_w, img_h, smpl, device=device)
            print(f"{r['runtime_s']*1000:.0f} ms,  err {r['per_joint_px_err'].mean():.0f} px")
            results[stem] = r
        all_results[device] = results

    # Use GPU results for the qualitative figure (visually identical to CPU)
    primary = all_results.get("cuda", all_results[devices_to_run[0]])
    plot_summary(primary, img_paths, out_dir / "mini_smplify.png")
    write_runtime_csv_combined(all_results, out_dir / "mini_smplify_runtime.csv")
    results = primary  # for downstream convergence plot

    # Convergence curves figure
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for stem, r in results.items():
        ax.plot(r["losses"], label=f"{stem} ({r['runtime_s']:.1f} s)", lw=1.5)
    ax.axvline(60, color="gray", linestyle="--", alpha=0.5, lw=0.8)
    ax.text(60, ax.get_ylim()[1]*0.95, "  stage 2 →", fontsize=9, color="gray")
    ax.set_xlabel("iteration")
    ax.set_ylabel("weighted reprojection loss (log)")
    ax.set_yscale("log")
    ax.set_title("SMPLify convergence — 60 it (orient + transl) → 240 it (all params)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "mini_smplify_convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ wrote {(out_dir / 'mini_smplify_convergence.png').relative_to(REPO_ROOT)}")

    # Print summary table for both devices
    print("\n=== Summary ===")
    print(f"{'image':<25} {'device':<6} {'SMPLify (s)':<14} {'× HMR':<10} {'mean err (px)':<14}")
    print("-" * 80)
    for device, results in all_results.items():
        avg_runtime, avg_err = 0.0, 0.0
        for stem, r in results.items():
            ms = r["runtime_s"] * 1000
            mult = ms / HMR_RUNTIME_MS
            err = r["per_joint_px_err"].mean()
            print(f"{stem:<25} {device:<6} {r['runtime_s']:.3f}        {mult:>4.0f}×     {err:>5.0f}")
            avg_runtime += r["runtime_s"]
            avg_err += err
        avg_runtime /= len(results); avg_err /= len(results)
        print(f"{'  → '+device+' avg':<25} {'':<6} {avg_runtime:.3f}        "
              f"{avg_runtime*1000/HMR_RUNTIME_MS:>4.0f}×     {avg_err:>5.0f}")
        print("-" * 80)


if __name__ == "__main__":
    main()
