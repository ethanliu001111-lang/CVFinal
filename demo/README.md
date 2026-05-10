# Demo — From 2D Keypoints to 4D Humans

Working model for the *Intro to Computer Vision* (Spring 2026) final project.

```
single image → ViTDet bbox → HRNet-W48 (2D kpts) → HMR 2.0 (SMPL mesh)        ← image path
video        → PHALP (HMR 2.0 + tracking)         → per-track 4D meshes        ← video path
```

## Layout

```
demo/
├── scripts/
│   ├── run_pipeline.py            # main CLI: HRNet → HMR 2.0 → quadplot/gif/runtime
│   └── setup_smpl_paths.sh        # one-time: wire SMPL_NEUTRAL.pkl symlinks
├── src/
│   ├── __init__.py                # MODEL_ROOT + path constants
│   ├── hrnet_2d.py                # MMPose HRNet-W48 inference
│   ├── visualize.py               # pyrender quad-plot + 360° rotation GIF
│   └── compare.py                 # runtime table (HMR vs reported SMPLify)
├── checkpoints/DOWNLOAD.md        # SMPL/SMPL-X registration + download guide
├── test_images/*.jpg              # 4 CC0 in-the-wild photos
├── test_videos/tennis.mp4         # 8.6 s clip for the video path
├── results/                       # generated artifacts (mostly gitignored)
└── docs/pipeline_diagram.png      # used in the report
```

The HMR 2.0 (4D-Humans) and PHALP source trees are vendored at the repo
root under `third_party/` (code + LICENSE only, no upstream demo assets)
so our patches travel with the code. Install them editable into `.venv`:

```bash
pip install -e third_party/4D-Humans -e third_party/PHALP
```

## Image path — single image → 3D mesh

```bash
# one-time
export CV_MODEL_ROOT=/path/to/registered/models
bash demo/scripts/setup_smpl_paths.sh

# run
python demo/scripts/run_pipeline.py demo/test_images/*.jpg \
    --out-dir demo/results --device cuda
```

Outputs per image:

- `results/quadplot_<stem>.png` — front view + 90° side view
- `results/rotation_<stem>.gif` — 24-frame 360° turntable (12 fps)
- `results/2d_vis/<stem>.jpg` — HRNet 2D keypoint overlay
- `results/runtime_table.{csv,tex}` — wall-clock table for the report

## Video path — video → 4D (PHALP)

```bash
cd third_party/4D-Humans
python track.py video.source=../../demo/test_videos/tennis.mp4 \
    video.output_dir=../../demo/results/tennis_phalp
```

Produces `PHALP_tennis.mp4` (per-track overlay) and a per-frame pickle.

## Hardware tested

RTX PRO 6000 Blackwell (sm_120), CUDA 13.0, PyTorch 2.11, Python 3.12.
Anything CUDA 11.8+ with ≥ 8 GB VRAM should work after a wheel-version swap.

## License

Code: MIT. SMPL / SMPL-X / 4D-Humans / PHALP weights are non-commercial
academic; each user must register and download independently. See
[`checkpoints/DOWNLOAD.md`](checkpoints/DOWNLOAD.md). Never commit
`*.pkl` / `*.npz` / `*.ckpt`.
