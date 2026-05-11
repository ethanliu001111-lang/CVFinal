#!/usr/bin/env python3
"""Image pipeline: ViTDet → HRNet-W48 → HMR 2.0 → quadplot + rotation GIF."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("images", nargs="+", help="input image paths")
    p.add_argument("--out-dir", type=Path,
                   default=REPO_ROOT / "demo" / "showcase",
                   help="committed artifacts (PNG/GIF/CSV)")
    p.add_argument("--cache-dir", type=Path,
                   default=REPO_ROOT / "demo" / "results",
                   help="intermediate caches (NPZ)")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--skip-2d", action="store_true", help="reuse cached HRNet kpts")
    p.add_argument("--skip-3d", action="store_true", help="reuse cached HMR2 meshes")
    p.add_argument("--score-thresh", type=float, default=0.5,
                   help="ViTDet person score threshold")
    return p.parse_args()


def _load_vitdet(device: str, score_thresh: float):
    from detectron2.config import LazyConfig
    from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
    import hmr2 as _h

    cfg_path = Path(_h.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
    cfg = LazyConfig.load(str(cfg_path))
    cfg.train.init_checkpoint = (
        "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
        "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    )
    for i in range(3):
        cfg.model.roi_heads.box_predictors[i].test_score_thresh = score_thresh
    return DefaultPredictor_Lazy(cfg)


def stage_detect(images: list[Path], device: str, score_thresh: float) -> dict[str, dict]:
    """ViTDet-H person detection (shared by HRNet + HMR)."""
    import cv2

    print(f"\n[Stage 0] ViTDet-H on {len(images)} images (score>{score_thresh}) …")
    detector = _load_vitdet(device, score_thresh)

    out: dict[str, dict] = {}
    for img_path in images:
        img_path = Path(img_path)
        img_cv2 = cv2.imread(str(img_path))
        if img_cv2 is None:
            print(f"  skip unreadable {img_path}"); continue
        t0 = time.time()
        det = detector(img_cv2)["instances"]
        rt = time.time() - t0
        cls = det.pred_classes.cpu().numpy()
        boxes = det.pred_boxes.tensor.cpu().numpy()[cls == 0]
        out[img_path.name] = {"boxes": boxes, "runtime_s": rt}
        print(f"  {img_path.name}: {len(boxes)} persons, {rt*1000:.0f} ms")
    return out


def stage_hrnet(images: list[Path], boxes_by_name: dict, out_dir: Path, device: str) -> dict:
    """HRNet-W48 2D keypoints, fed externally-provided ViTDet boxes."""
    from demo.src.hrnet_2d import run_hrnet_with_boxes, save_kpts_npz

    print(f"\n[Stage 1] HRNet-W48 on {len(images)} images …")
    boxes_dict = {name: d["boxes"] for name, d in boxes_by_name.items()}
    res = run_hrnet_with_boxes(images, boxes_dict, out_dir=out_dir / "2d_vis", device=device)
    for name, r in res.items():
        print(f"  {name}: {len(r['kpts'])} persons, {r['runtime_s']*1000:.0f} ms")
    return res


def _cam_crop_to_full(cam_bbox, box_center, box_size, img_w, img_h, focal_length=5000.0):
    """Convert HMR's box-frame (s,tx,ty) to full-image cam_t (matches hmr2.utils.renderer)."""
    import numpy as np
    s, tx, ty = cam_bbox
    cx, cy = box_center
    bs = float(box_size) * s + 1e-9
    return np.array([
        (2.0 * (cx - img_w / 2.0) / bs) + tx,
        (2.0 * (cy - img_h / 2.0) / bs) + ty,
        2.0 * focal_length / bs,
    ], dtype=np.float32)


def stage_hmr2(images: list[Path], boxes_by_name: dict, device: str, batch_size: int) -> dict:
    """HMR 2.0 SMPL mesh. Caches cam_t in full-image frame."""
    import torch, cv2, numpy as np
    from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
    from hmr2.utils import recursive_to
    from hmr2.datasets.vitdet_dataset import ViTDetDataset

    print(f"\n[Stage 2] HMR 2.0 on {len(images)} images …")
    model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
    model = model.to(device).eval()

    out: dict = {}
    for img_path in images:
        img_path = Path(img_path)
        img_cv2 = cv2.imread(str(img_path))
        if img_cv2 is None:
            print(f"  skip unreadable {img_path}"); continue
        H, W = img_cv2.shape[:2]

        boxes = boxes_by_name[img_path.name]["boxes"]
        if len(boxes) == 0:
            out[img_path.name] = {"verts": np.zeros((0, 6890, 3)), "cam_t": np.zeros((0, 3)),
                                  "boxes": np.zeros((0, 4)), "runtime_s": 0.0}
            print(f"  {img_path.name}: 0 persons"); continue

        ds = ViTDetDataset(model_cfg, img_cv2, boxes)
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)

        t0 = time.time(); vs, ts_full = [], []
        for batch in dl:
            batch_gpu = recursive_to(batch, device)
            with torch.no_grad():
                pred = model(batch_gpu)
            vs.append(pred["pred_vertices"].cpu().numpy())
            pred_cam = pred["pred_cam"].cpu().numpy()
            box_center = batch["box_center"].cpu().numpy()
            box_size = batch["box_size"].cpu().numpy()
            for i in range(len(pred_cam)):
                ts_full.append(_cam_crop_to_full(pred_cam[i], box_center[i], box_size[i], W, H))
        rt = time.time() - t0
        out[img_path.name] = {
            "verts": np.concatenate(vs, 0),
            "cam_t": np.stack(ts_full),
            "boxes": boxes,
            "runtime_s": rt,
        }
        print(f"  {img_path.name}: {len(boxes)} persons, {rt*1000:.0f} ms")

    return out, model.smpl.faces


def stage_render(images: list[Path], hmr2_res: dict, faces, out_dir: Path):
    """Quad-plot (input | 2D | mesh front | mesh side) + 360° rotation GIF per image."""
    import cv2
    from demo.src.visualize import quad_plot, rotation_gif, pick_center_person

    print(f"\n[Stage 3] Rendering {len(images)} quadplots + GIFs …")
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

        quad_plot(img, overlay, verts, faces, out_dir / f"quadplot_{img_path.stem}.png",
                  verts_all=verts_all, cam_t_all=cam_t_all)
        rotation_gif(verts, faces, out_dir / f"rotation_{img_path.stem}.gif", n_frames=24)
        print(f"  ✓ {img_path.name}")


def stage_report(hrnet_res: dict, hmr2_res: dict, det_res: dict, out_dir: Path):
    """Save per-image runtime table (CSV + LaTeX)."""
    from demo.src.compare import runtime_table, save_tables

    df = runtime_table(hrnet_res, hmr2_res)
    df.insert(1, "ViTDet_ms", [det_res[n]["runtime_s"] * 1000 for n in hrnet_res] + [
        sum(det_res[n]["runtime_s"] for n in hrnet_res) * 1000 / max(len(hrnet_res), 1)
    ])
    df.insert(2, "n_persons", [len(det_res[n]["boxes"]) for n in hrnet_res] + [
        sum(len(det_res[n]["boxes"]) for n in hrnet_res) / max(len(hrnet_res), 1)
    ])
    print("\n[Stage 4] Runtime table:")
    print(df.to_string())
    save_tables(df, df_agreement=None, out_dir=out_dir)


def _save_boxes(det_res: dict, cache_dir: Path) -> None:
    import numpy as np
    np.savez_compressed(
        cache_dir / "vitdet_boxes.npz",
        **{f"{name}_boxes": v["boxes"] for name, v in det_res.items()},
        **{f"{name}_rt": np.array([v["runtime_s"]]) for name, v in det_res.items()},
    )


def _load_boxes(cache_dir: Path, images: list) -> dict | None:
    cache = cache_dir / "vitdet_boxes.npz"
    if not cache.exists(): return None
    import numpy as np
    z = np.load(cache)
    res = {}
    for img_path in images:
        name = Path(img_path).name
        if f"{name}_boxes" not in z.files: return None
        res[name] = {"boxes": z[f"{name}_boxes"], "runtime_s": float(z[f"{name}_rt"][0])}
    return res


def _save_hrnet(hrnet_res: dict, cache_dir: Path) -> None:
    from demo.src.hrnet_2d import save_kpts_npz
    save_kpts_npz(hrnet_res, cache_dir / "hrnet_kpts.npz")


def _load_hrnet(cache_dir: Path, images: list) -> dict | None:
    cache = cache_dir / "hrnet_kpts.npz"
    if not cache.exists(): return None
    import numpy as np
    z = np.load(cache, allow_pickle=True)
    res = {}
    for img_path in images:
        name = Path(img_path).name
        if f"{name}_kpts" not in z.files: return None
        res[name] = {"kpts": z[f"{name}_kpts"], "scores": z[f"{name}_scores"],
                     "vis_path": "", "runtime_s": 0.0}
    return res


def _save_hmr2(hmr2_res: dict, cache_dir: Path) -> None:
    import numpy as np
    np.savez_compressed(
        cache_dir / "hmr2_meshes.npz",
        **{f"{name}_verts": v["verts"] for name, v in hmr2_res.items()},
        **{f"{name}_camt": v["cam_t"] for name, v in hmr2_res.items()},
    )


def _load_hmr2(cache_dir: Path, images: list):
    cache = cache_dir / "hmr2_meshes.npz"
    if not cache.exists(): return None, None
    import numpy as np
    z = np.load(cache)
    res = {}
    for img_path in images:
        name = Path(img_path).name
        if f"{name}_verts" not in z.files: return None, None
        res[name] = {"verts": z[f"{name}_verts"], "cam_t": z[f"{name}_camt"],
                     "boxes": np.zeros((0, 4)), "runtime_s": 0.0}
    import smplx
    from demo.src import MODEL_ROOT
    smpl = smplx.create(str(MODEL_ROOT), model_type="smpl", gender="neutral")
    return res, smpl.faces


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    images = [Path(p) for p in args.images]

    det_res = _load_boxes(args.cache_dir, images) if (args.skip_2d and args.skip_3d) else None
    if det_res is None:
        det_res = stage_detect(images, args.device, args.score_thresh)
        _save_boxes(det_res, args.cache_dir)
    else:
        print("[Stage 0] reusing cached vitdet_boxes.npz")

    hrnet_res = _load_hrnet(args.cache_dir, images) if args.skip_2d else None
    if hrnet_res is None:
        hrnet_res = stage_hrnet(images, det_res, args.out_dir, args.device)
        _save_hrnet(hrnet_res, args.cache_dir)
        import torch, gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    else:
        print("[Stage 1] reusing cached hrnet_kpts.npz")

    hmr2_res, faces = (_load_hmr2(args.cache_dir, images) if args.skip_3d else (None, None))
    if hmr2_res is None:
        hmr2_res, faces = stage_hmr2(images, det_res, args.device, args.batch_size)
        _save_hmr2(hmr2_res, args.cache_dir)
    else:
        print("[Stage 2] reusing cached hmr2_meshes.npz")

    stage_render(images, hmr2_res, faces, args.out_dir)
    stage_report(hrnet_res, hmr2_res, det_res, args.out_dir)

    print(f"\n✓ Done. Showcase: {args.out_dir}  ·  Cache: {args.cache_dir}")


if __name__ == "__main__":
    main()
