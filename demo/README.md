# Demo — From 2D Pose to 3D Human Recovery

Working-model accompaniment to the literature review report (Intro to CV, Spring 2026).

## TL;DR

```
Input image → ViTDet (person bbox) → HRNet-W48 (2D keypoints) → 4D-Humans HMR 2.0 (3D SMPL mesh)
                                                                        ↓
                                                       quad-plot + 360° GIF + runtime table
```

Empirically validates HMR (CVPR 2018)'s central claim: **single-pass regression
delivers ~600× speedup over iterative SMPLify optimization** at comparable
visual quality.

## Layout

```
demo/
├── src/                              # Reusable pipeline modules
│   ├── smpl_forward.py               #   SMPL Eq.(1) narrated wrapper
│   ├── hrnet_2d.py                   #   MMPose HRNet inference
│   ├── hmr2_demo_wrapper.py          #   4D-Humans inference (Colab only)
│   ├── smplifyx_cli.py               #   STRETCH: SMPLify-X CLI wrapper
│   ├── compare.py                    #   Procrustes-aligned joint MPJPE + tables
│   └── visualize.py                  #   matplotlib quad-plot + GIF + multi-person guard
├── notebooks/
│   ├── 00_mac_smoke_test.ipynb       # Run locally to verify SMPL-X loads (~10 s)
│   ├── 01_main_pipeline.ipynb        # ★ MAIN demo — run on Colab (~7 s for 5 images)
│   ├── 02_smplifyx_stretch.ipynb     # Stretch goal, hard cutoff 5/5 EOD
│   └── 03_smpl_explain.ipynb         # Pedagogical SMPL Eq.(1) walkthrough
├── envs/
│   ├── requirements_local_mac.txt    # Mac dev: torch / smplx / trimesh / matplotlib
│   ├── env_hrnet_hmr2.yml            # Colab main: + mmpose / 4D-Humans / chumpy
│   └── env_smplifyx.yml              # Stretch: PyTorch 1.x for SMPLify-X compat
├── scripts/
│   ├── setup_smpl_paths.sh           # Create model symlinks (one-time)
│   └── run_smoke_test.sh             # Local smoke test
├── checkpoints/
│   ├── .gitignore                    # Never commit binaries (license)
│   └── DOWNLOAD.md                   # Registration + download guide
├── test_images/README.md             # 5 CC0 test images TODO
├── results/                          # Generated artifacts (some kept in repo)
└── docs/pipeline_diagram.png         # Goes into report §3
```

## Quick start (Mac local)

```bash
# 1. Clone, enter the worktree
cd <repo>

# 2. Create a venv at repo root (already exists if you've followed plan v3)
python3 -m venv .venv
source .venv/bin/activate
pip install -r demo/envs/requirements_local_mac.txt

# 3. Set up SMPL/SMPL-X symlinks (after registering at smpl.is.tue.mpg.de)
bash demo/scripts/setup_smpl_paths.sh

# 4. Smoke test (< 10 s)
jupyter notebook demo/notebooks/00_mac_smoke_test.ipynb
# or, programmatically:
python -c "from demo.src.smpl_forward import export_tpose_obj; export_tpose_obj('demo/results/tpose_smplx.obj', model_type='smplx')"
```

## Full pipeline (Linux GPU server — preferred)

Run on any Linux machine with an NVIDIA GPU (≥ 8 GB VRAM, CUDA 11.8 driver).
This is the recommended path: more reliable than Colab, no quota limits,
checkpoints persist between runs.

```bash
# ─── On your laptop ───
bash demo/scripts/sync_to_server.sh user@gpu-server.example.edu ~/cv-final --with-models

# ─── On the server ───
ssh user@gpu-server.example.edu
cd ~/cv-final
bash demo/scripts/setup_linux_server.sh        # ~10 min, one-time
source ~/.venv/cv-final/bin/activate
CV_MODEL_ROOT=$(realpath ../model) bash demo/scripts/setup_smpl_paths.sh

# Run the pipeline (CLI, no Jupyter needed)
python demo/scripts/run_pipeline.py demo/test_images/*.jpg \
    --out-dir demo/results --device cuda --batch-size 1

# ─── Pull results back to your laptop ───
rsync -avz user@gpu-server:~/cv-final/demo/results/ ./demo/results/
```

Optional — interactive Jupyter on the server (handy for debugging cell by cell):

```bash
# On YOUR laptop:
bash demo/scripts/start_remote_jupyter.sh user@gpu-server.example.edu
# → opens an SSH tunnel and prints the token URL.
# Browse to http://localhost:8888 with that token.
```

### Server hardware sanity-check

| Resource | Minimum | Recommended | What it gates |
|---|---|---|---|
| GPU VRAM | 8 GB | 16 GB+ | 4D-Humans ViT-L peak ~3 GB; HRNet ~0.5 GB |
| CUDA driver | 11.8 | 12.x | PyTorch 2.1 wheel |
| CPU RAM | 16 GB | 32 GB | mesh tensors + numpy buffers |
| Disk | 20 GB | 30 GB | models (4.5 GB) + venv (3 GB) + results (~50 MB) + 4D-Humans repo |
| Python | 3.10 | 3.10 / 3.11 | matched against PyTorch wheel |

If you only have CPU on the server, the pipeline still runs (~20× slower).
Pass `--device cpu` to `run_pipeline.py`; expect ~3 minutes for 5 images.

---

## Full pipeline (Colab — fallback)

1. Upload **5 test images** to `/content/test_images/` on Colab (or sync from
   Drive).
2. Make sure SMPL `.pkl` and SMPL-X `.npz` are in your Drive at
   `MyDrive/smpl/` and `MyDrive/smplx/` respectively.
3. Open `notebooks/01_main_pipeline.ipynb` in Colab.
4. Runtime → Change runtime type → **GPU (T4)**.
5. **Run all** (~3 min for setup, ~7 s for 5-image inference).
6. Download:
   - `results/quadplot_*.png`
   - `results/rotation_*.gif`
   - `results/runtime_table.csv` and `.tex`
7. Final 30 s `full_demo.mp4` is assembled from these — see slide 11 of the
   presentation deck.

## Validation hierarchy

| Status     | Layer                           | Where verified |
|------------|---------------------------------|----------------|
| ✅ Verified | Mac: SMPL-X load + forward     | `notebooks/00_mac_smoke_test.ipynb` |
| ✅ Verified | matplotlib mesh rendering     | `demo/results/smplx_4poses.png`     |
| ⏳ Pending  | Linux: HRNet-W48 inference    | `python demo/scripts/run_pipeline.py` Stage 1 |
| ⏳ Pending  | Linux: 4D-Humans HMR 2.0      | `python demo/scripts/run_pipeline.py` Stage 2 |
| ⏳ Pending  | Linux: 5-image full run       | 5/4–5/5 P1 task |
| 🌟 Stretch  | SMPLify-X agreement metric    | `notebooks/02_smplifyx_stretch.ipynb` (5/5 EOD cutoff) |

## License & redistribution

- **Code in this directory**: MIT (your choice; matches 4D-Humans).
- **SMPL / SMPL-X / VPoser**: non-commercial academic. Each user must
  register independently.
- **Never** commit `.pkl` / `.npz` / `.ckpt` to git.
- **Never** include them in the submission `.zip`.

## Report integration

Generated artifacts that flow into the LaTeX report (`LiteratureReview/main.tex`):

- `docs/pipeline_diagram.png` → §3 (architecture overview)
- `results/runtime_table.tex` → §4 (HMR 600× speedup claim)
- `results/quadplot_img1.png` → §4 (qualitative example)
- `results/agreement_table.tex` → §4 (only if stretch succeeds)
