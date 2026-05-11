# Demo — From 2D Keypoints to 4D Humans

Working pipeline for the *Intro to Computer Vision* (Spring 2026) final project.

```
single image → ViTDet-H bbox → HRNet-W48 (2D kpts) → HMR 2.0 (SMPL mesh)        [image path]
video        → PHALP (HMR 2.0 + tracking)            → per-track 4D meshes       [video path]
```

## Layout

```
demo/
├── scripts/
│   ├── run_pipeline.py         # main CLI: ViTDet → HRNet → HMR 2.0 → quadplot/GIF
│   ├── setup_smpl_paths.sh     # one-time: wire SMPL_NEUTRAL.pkl symlinks
│   ├── make_extras.sh          # regenerate all extras + PHALP showcase
│   └── extras/                 # pose similarity / reprojection / mini-SMPLify / tennis 4D
├── src/                        # MODEL_ROOT + hrnet_2d + visualize + compare
├── checkpoints/DOWNLOAD.md     # SMPL/SMPL-X registration + download instructions
├── test_images/                # 4 CC0 photos covering single/multi-person + occlusion
├── test_videos/tennis.mp4      # 8.6 s clip for the video path
├── results/                    # intermediate caches (gitignored)
└── showcase/                   # rendered outputs (committed for review)
    ├── 2d_vis/                 # HRNet keypoint overlays
    ├── quadplot_*.png          # 2×2 grids (input | 2D | mesh front | mesh side)
    ├── rotation_*.gif          # 24-frame 360° SMPL turntable
    ├── runtime_table.{csv,tex}
    ├── extras/                 # mini-SMPLify, reprojection, pose-similarity, tennis joints
    └── video/                  # PHALP overlay frames + 5 s clip
```

## One-time setup

```bash
git clone git@github.com:ethanliu001111-lang/CVFinal.git && cd CVFinal
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt  # or install per the report appendix
pip install -e third_party/4D-Humans -e third_party/PHALP

# Register at smpl.is.tue.mpg.de, download the body models, then:
export CV_MODEL_ROOT=/abs/path/to/your/registered/models
bash demo/scripts/setup_smpl_paths.sh
```

Tested on **Python 3.12, CUDA 13, PyTorch 2.11, RTX PRO 6000 (sm_120)**. Older
CUDA 11.8+ with ≥ 8 GB VRAM works after a wheel swap.

## Image path

```bash
python demo/scripts/run_pipeline.py demo/test_images/*.jpg --device cuda
```

Writes to:
- `demo/showcase/2d_vis/<stem>.jpg` — HRNet 2D overlay (all detected people)
- `demo/showcase/quadplot_<stem>.png` — input | 2D | mesh front | mesh side
- `demo/showcase/rotation_<stem>.gif` — 360° rotating SMPL mesh
- `demo/showcase/runtime_table.{csv,tex}` — per-image ViTDet/HRNet/HMR2 timing
- `demo/results/{vitdet_boxes,hrnet_kpts,hmr2_meshes}.npz` — caches (gitignored)

Flags:
- `--skip-2d` / `--skip-3d` reuse cached HRNet / HMR outputs (re-render only)
- `--score-thresh` lowers ViTDet's person score gate (default 0.5)
- `--device cpu` runs on CPU (~10× slower)

## Video path (PHALP)

```bash
python -m phalp.track \
    video.source=demo/test_videos/tennis.mp4 \
    video.output_dir=demo/results/tennis_phalp \
    video.end_frame=1300
```

Writes per-frame tracking pickle to `demo/results/tennis_phalp/results/demo_tennis.pkl`
and a per-track-overlay video to `demo/results/tennis_phalp/PHALP_tennis.mp4`.
`make_extras.sh` (below) copies the highlights into `demo/showcase/video/`.

## Extras + showcase regeneration

```bash
bash demo/scripts/make_extras.sh
```

Runs all four extras (pose-similarity heatmap, reprojection overlay, tennis
4D trajectory, mini-SMPLify), pulls four sample frames out of `PHALP_tennis.mp4`,
and produces a 5 s clip — everything lands under `demo/showcase/`.

## Reproducibility checklist

- Set `CV_MODEL_ROOT` before any script — `demo/src/__init__.py` falls back to
  `<repo>/model` only if the env var is unset.
- Caches in `demo/results/` are version-pinned to the source images. Wipe them
  (`rm -rf demo/results/*.npz`) after editing `test_images/` or rotating SMPL assets.
- All scripts resolve paths relative to the repo root via
  `Path(__file__).resolve().parents[...]`; no absolute paths are baked in.

## License

Code: MIT. SMPL / SMPL-X / 4D-Humans / PHALP weights are non-commercial academic;
each user must register and download independently. See
[`checkpoints/DOWNLOAD.md`](checkpoints/DOWNLOAD.md). Never commit
`*.pkl` / `*.npz` / `*.ckpt`.
