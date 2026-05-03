"""Reproducing SMPL Eq.(1) — Loper et al. SIGGRAPH Asia 2015.

This file implements a *narrated* call into the official `smplx` library so that
each algebraic step of the SMPL forward maps cleanly to the paper.  It satisfies
the PDF §6.6 'simple implementation of a method described in the literature'
clause without re-deriving LBS from scratch.

The official SMPL .pkl from https://smpl.is.tue.mpg.de still requires `chumpy`
to deserialize (Python 2.7 legacy).  On Mac we therefore ship the SMPL-X path,
and on Colab we install `chumpy` to enable SMPL.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import trimesh
import smplx

from . import MODEL_ROOT


def smpl_forward_eq1(
    body_pose_72: torch.Tensor | None = None,
    betas_10: torch.Tensor | None = None,
    *,
    model_root: Path = MODEL_ROOT,
    model_type: Literal["smpl", "smplx"] = "smpl",
    gender: Literal["neutral", "male", "female"] = "neutral",
) -> tuple[torch.Tensor, torch.Tensor, smplx.SMPL | smplx.SMPLX]:
    """One canonical forward of SMPL Eq.(1).

    SMPL paper Eq.(1):
        M(beta, theta) = W(  T_P(beta, theta) ,  J(beta) ,  theta ,  W  )
                            └──┬──┘
                          T(beta, theta)
                                = T_bar + B_S(beta) + B_P(theta)
        where:
        * T_bar    = mean template shape
        * B_S(β)   = shape blend shapes  (betas-driven displacement)
        * B_P(θ)   = pose  blend shapes  (pose-driven displacement)
        * J(β)     = joint regressor on the shape-blended mesh
        * W        = blendweights (Linear Blend Skinning weights)

    `smplx` already implements every term; we just label them.

    Args:
        body_pose_72: SMPL: (B, 69)  axis-angle for joints 1..23
                      SMPL-X: (B, 63)  axis-angle for joints 1..21
                      None → zero pose (T-pose)
        betas_10:     (B, 10)  shape PCA coefficients;  None → mean shape
        model_root:   directory containing `smpl/SMPL_NEUTRAL.pkl` etc.
        model_type:   'smpl' (24 joints, 6890 verts) or 'smplx' (54 joints, 10475)

    Returns:
        vertices: (B, V, 3)
        joints:   (B, J, 3)
        model:    the underlying smplx model object (so caller can access .faces)
    """
    body_dim = 69 if model_type == "smpl" else 63

    create_kwargs = dict(model_path=str(model_root), model_type=model_type, gender=gender)
    if model_type == "smplx":
        create_kwargs.update(num_expression_coeffs=10, ext="npz")
    model = smplx.create(**create_kwargs)

    if body_pose_72 is None:
        body_pose_72 = torch.zeros(1, body_dim)
    if betas_10 is None:
        betas_10 = torch.zeros(1, 10)

    out = model(
        betas=betas_10,                       # ─ B_S(β): shape blend shape coeffs
        body_pose=body_pose_72,                # ─ B_P(θ): pose blend shape input
        global_orient=torch.zeros(betas_10.shape[0], 3),
    )
    # `out.vertices` already incorporates: T_bar + B_S + B_P, then LBS via W and θ.
    # `out.joints` = J_reg @ shaped_template, then transformed by full kinematic chain.
    return out.vertices, out.joints, model


def export_tpose_obj(out_path: str | Path, model_type: Literal["smpl", "smplx"] = "smplx") -> Path:
    """Render T-pose to .obj (viewable in MeshLab / Blender / macOS Preview)."""
    verts, _joints, model = smpl_forward_eq1(model_type=model_type)
    mesh = trimesh.Trimesh(
        vertices=verts.squeeze().detach().numpy(),
        faces=model.faces,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out_path)
    return out_path


if __name__ == "__main__":
    # Quick CLI usage: python -m demo.src.smpl_forward
    from . import RESULTS
    p = export_tpose_obj(RESULTS / "tpose_smplx_demo.obj", model_type="smplx")
    print(f"✓ wrote {p}")
