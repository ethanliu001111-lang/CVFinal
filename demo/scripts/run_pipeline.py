#!/usr/bin/env python3
"""CLI version of `notebooks/01_main_pipeline.ipynb`.

Designed for SSH-only Linux GPU servers — no Jupyter required.

    python demo/scripts/run_pipeline.py demo/test_images/*.jpg \\
        --out-dir demo/results \\
        --device cuda \\
        --batch-size 1

Produces (per input image):
    out_dir/2d_vis/<stem>.jpg            — HRNet 2D overlay
    out_dir/quadplot_<stem>.png           — 2x2 grid (input | 2D | mesh-front | mesh-side)
    out_dir/rotation_<stem>.gif           — 360° rotating SMPL mesh
And global:
    out_dir/runtime_table.csv / .tex
    out_dir/hrnet_kpts.npz
    out_dir/hmr2_meshes.npz
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# ─── Headless rendering must be set BEFORE importing pyrender / matplotlib ───
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", nargs="+", help="paths to input images")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "demo" / "results")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--batch-size", type=int, default=1, help="HMR2 batch size (T4: 1, A100: 8)")
    p.add_argument("--skip-2d", action="store_true",
                   help="reuse cached HRNet kpts npz from a previous run")
    p.add_argument("--skip-3d", action="store_true",
                   help="reuse cached HMR2 meshes npz from a previous run")
    return p.parse_args()


def stage_hrnet(images: list[Path], out_dir: Path, device: str) -> dict:
    """Stage 1 — HRNet 2D inference."""
    from demo.src.hrnet_2d import run_hrnet_w48, save_kpts_npz

    print(f"\n[Stage 1] HRNet-W48 on {len(images)} images …")
    res = run_hrnet_w48(images, out_dir=out_dir / "2d_vis", device=device)
    for name, r in res.items():
        print(f"  {name}: {len(r['kpts'])} persons, {r['runtime_s']*1000:.0f} ms")
    save_kpts_npz(res, out_dir / "hrnet_kpts.npz")
    return res


def stage_hmr2(images: list[Path], device: str, batch_size: int) -> dict:
    """Stage 2 — 4D-Humans (HMR 2.0)."""
    import torch, cv2, numpy as np
    from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
    from hmr2.utils import recursive_to
    from hmr2.datasets.vitdet_dataset import ViTDetDataset
    from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
    from detectron2.config import LazyConfig
    import hmr2  as _h

    print(f"\n[Stage 2] 4D-Humans HMR 2.0 on {len(images)} images …")
    model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
    model = model.to(device).eval()

    cfg_path = Path(_h.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
    if not cfg_path.exists():
        for candidate in [
            REPO_ROOT.parent / "4D-Humans" / "hmr2" / "configs" / cfg_path.name,
            Path.home() / "4D-Humans" / "hmr2" / "configs" / cfg_path.name,
        ]:
            if candidate.exists():
                cfg_path = candidate; break
    det_cfg = LazyConfig.load(str(cfg_path))
    # Override the COCO-trained ViTDet checkpoint URL (config defaults to MAE pretrain only)
    det_cfg.train.init_checkpoint = (
        "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
        "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    )
    for i in range(3):
        det_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.5
    detector = DefaultPredictor_Lazy(det_cfg)

    out: dict = {}
    for img_path in images:
        img_path = Path(img_path)
        img_cv2 = cv2.imread(str(img_path))
        if img_cv2 is None:
            print(f"  ⚠️ skip unreadable {img_path}"); continue

        det = detector(img_cv2)["instances"]
        cls = det.pred_classes.cpu().numpy()
        boxes = det.pred_boxes.tensor.cpu().numpy()[cls == 0]   # person class only
        if len(boxes) == 0:
            out[img_path.name] = {"verts": np.zeros((0,6890,3)), "cam_t": np.zeros((0,3)),
                                    "boxes": np.zeros((0,4)), "runtime_s": 0.0}
            print(f"  {img_path.name}: 0 persons (skip)"); continue

        ds = ViTDetDataset(model_cfg, img_cv2, boxes)
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)

        t0 = time.time(); vs, ts = [], []
        for batch in dl:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                pred = model(batch)
            vs.append(pred["pred_vertices"].cpu().numpy())
            ts.append(pred["pred_cam_t"].cpu().numpy())
        rt = time.time() - t0
        out[img_path.name] = {
            "verts": np.concatenate(vs, 0),
            "cam_t": np.concatenate(ts, 0),
            "boxes": boxes,
            "runtime_s": rt,
        }
        print(f"  {img_path.name}: {len(boxes)} persons, {rt*1000:.0f} ms")

    return out, model.smpl.faces  # (mesh, faces) for downstream rendering


def stage_render(images: list[Path], hrnet_res: dict, hmr2_res: dict, faces, out_dir: Path):
    """Stage 3 — quad-plot + rotation GIF per image."""
    import cv2
    from demo.src.visualize import quad_plot, rotation_gif, pick_center_person

    print(f"\n[Stage 3] Rendering {len(images)} quad-plots + GIFs …")
    for img_path in images:
        img_path = Path(img_path)
        img = cv2.imread(str(img_path))[:, :, ::-1]
        vis_path = next((out_dir / "2d_vis").glob(f"{img_path.stem}*"), None)
        if vis_path is None: continue
        overlay = cv2.imread(str(vis_path))[:, :, ::-1]

        verts_all = hmr2_res[img_path.name]["verts"]
        cam_t_all = hmr2_res[img_path.name]["cam_t"]
        idx = pick_center_person(verts_all, cam_t_all, img.shape)
        if idx is None: continue
        verts = verts_all[idx]

        quad_plot(img, overlay, verts, faces, out_dir / f"quadplot_{img_path.stem}.png")
        rotation_gif(verts, faces, out_dir / f"rotation_{img_path.stem}.gif", n_frames=15)
        print(f"  ✓ {img_path.name}")


def stage_report(hrnet_res: dict, hmr2_res: dict, out_dir: Path):
    """Stage 4 — runtime table."""
    from demo.src.compare import runtime_table, save_tables

    df = runtime_table(hrnet_res, hmr2_res)
    print("\n[Stage 4] Runtime table:")
    print(df.to_string())
    save_tables(df, df_agreement=None, out_dir=out_dir)


def _load_hrnet_cache(out_dir: Path, images: list) -> dict | None:
    cache = out_dir / "hrnet_kpts.npz"
    if not cache.exists(): return None
    import numpy as np
    z = np.load(cache, allow_pickle=True)
    res: dict = {}
    for img_path in images:
        name = Path(img_path).name
        kpts_key = f"{name}_kpts"; sc_key = f"{name}_scores"
        if kpts_key not in z.files: return None
        res[name] = {"kpts": z[kpts_key], "scores": z[sc_key],
                      "vis_path": "", "runtime_s": 0.0}
    return res


def _save_hmr2_cache(hmr2_res: dict, out_dir: Path) -> None:
    import numpy as np
    np.savez_compressed(
        out_dir / "hmr2_meshes.npz",
        **{f"{name}_verts": v["verts"] for name, v in hmr2_res.items()},
        **{f"{name}_camt":  v["cam_t"] for name, v in hmr2_res.items()},
    )


def _load_hmr2_cache(out_dir: Path, images: list):
    cache = out_dir / "hmr2_meshes.npz"
    if not cache.exists(): return None, None
    import numpy as np
    z = np.load(cache)
    res: dict = {}
    for img_path in images:
        name = Path(img_path).name
        if f"{name}_verts" not in z.files: return None, None
        res[name] = {"verts": z[f"{name}_verts"], "cam_t": z[f"{name}_camt"],
                      "boxes": np.zeros((0, 4)), "runtime_s": 0.0}
    # Faces are not cacheable as ndarray scalar; reload via smplx
    import smplx
    from demo.src import MODEL_ROOT
    smpl = smplx.create(str(MODEL_ROOT), model_type="smpl", gender="neutral")
    return res, smpl.faces


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    images = [Path(p) for p in args.images]

    hrnet_res = _load_hrnet_cache(args.out_dir, images) if args.skip_2d else None
    if hrnet_res is None:
        hrnet_res = stage_hrnet(images, args.out_dir, args.device)
        import torch, gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    else:
        print("[Stage 1] Reusing cached hrnet_kpts.npz")

    hmr2_res, faces = (_load_hmr2_cache(args.out_dir, images) if args.skip_3d else (None, None))
    if hmr2_res is None:
        hmr2_res, faces = stage_hmr2(images, args.device, args.batch_size)
        _save_hmr2_cache(hmr2_res, args.out_dir)
    else:
        print("[Stage 2] Reusing cached hmr2_meshes.npz")

    stage_render(images, hrnet_res, hmr2_res, faces, args.out_dir)
    stage_report(hrnet_res, hmr2_res, args.out_dir)

    print(f"\n✅ Done. Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
