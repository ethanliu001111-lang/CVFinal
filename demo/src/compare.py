"""Comparison & reporting — runtime + Procrustes-aligned joint MPJPE.

We deliberately call the cross-method metric "agreement" (not "accuracy"), per
codex round-1 finding: using one model's output as pseudo-GT for another is
circular reasoning.  True accuracy is reported by *citing* paper numbers
(HMR2 ~44 mm on 3DPW, SMPLify ~82 mm) in the LaTeX report §4.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def procrustes_align(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Similarity-transform `pred` onto `target` (translation + rotation + uniform scale).

    Both inputs: (J, 3).  Returns: (J, 3) aligned to target.
    """
    mu_p, mu_t = pred.mean(0, keepdims=True), target.mean(0, keepdims=True)
    P, T = pred - mu_p, target - mu_t
    # Optimal rotation via SVD on cross-covariance
    H = P.T @ T
    U, _S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    # Optimal isotropic scale
    var_p = (P ** 2).sum()
    s = (np.diag([1, 1, d]) * np.linalg.svd(H, compute_uv=False)).sum() / max(var_p, 1e-12)
    return s * (P @ R.T) + mu_t


def joint_mpjpe(pred_joints: np.ndarray, target_joints: np.ndarray) -> float:
    """Procrustes-aligned mean per-joint position error in millimeters.

    Inputs assumed in meters (SMPL/SMPL-X default scale).
    """
    aligned = procrustes_align(pred_joints, target_joints)
    return float(np.linalg.norm(aligned - target_joints, axis=-1).mean() * 1000)


def runtime_table(
    hrnet_results: dict, hmr2_results: dict,
    smplifyx_results: dict | None = None,
) -> pd.DataFrame:
    """Build the headline table that goes into report §4 + slide 11."""
    rows = []
    for name in hrnet_results:
        row = {
            "image":          name,
            "HRNet_2D_ms":    hrnet_results[name]["runtime_s"]   * 1000,
            "HMR2_3D_ms":     hmr2_results [name]["runtime_s"]   * 1000,
        }
        if smplifyx_results and name in smplifyx_results:
            row["SMPLifyX_3D_s"] = smplifyx_results[name]["runtime_s"]
        rows.append(row)

    df = pd.DataFrame(rows)
    df.loc["mean"] = df.mean(numeric_only=True)
    return df


def agreement_table(
    hmr2_joints: dict[str, np.ndarray],
    smplifyx_joints: dict[str, np.ndarray],
    common_joint_idx_smpl: list[int],
    common_joint_idx_smplx: list[int],
) -> pd.DataFrame:
    """Per-image joint agreement (mm) between HMR 2.0 and SMPLify-X.

    *Not* an accuracy metric — see module docstring.
    """
    rows = []
    for name in smplifyx_joints:
        if name not in hmr2_joints:
            continue
        ja = hmr2_joints[name]    [common_joint_idx_smpl]
        jb = smplifyx_joints[name][common_joint_idx_smplx]
        rows.append({"image": name, "agreement_mm": joint_mpjpe(jb, ja)})
    return pd.DataFrame(rows)


def save_tables(df_runtime: pd.DataFrame, df_agreement: pd.DataFrame | None,
                out_dir: str | Path) -> None:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df_runtime.to_csv(out_dir / "runtime_table.csv")
    # index=True so the 'mean' row label is preserved in the LaTeX table
    df_runtime.to_latex(out_dir / "runtime_table.tex", float_format="%.1f", index=True)
    if df_agreement is not None and not df_agreement.empty:
        df_agreement.to_csv(out_dir / "agreement_table.csv", index=False)
        df_agreement.to_latex(out_dir / "agreement_table.tex", float_format="%.1f", index=False)


# Common joint subsets (14 J — used for cross-model comparison).
# SMPL has 24 joints, SMPL-X has 54; both share the body-25 ordering for the
# first 22 joints.  We restrict to a 14-joint body-only subset for stability.
SMPL_BODY14_IDX  = [16, 17, 18, 19, 20, 21,  1, 2, 4, 5, 7, 8, 12, 15]   # ↪ rough COCO body-only mapping
SMPLX_BODY14_IDX = SMPL_BODY14_IDX                                       # SMPL-X body joints share SMPL ordering
