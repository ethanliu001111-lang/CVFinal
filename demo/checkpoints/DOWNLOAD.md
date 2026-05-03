# Model Checkpoint Download Guide

This directory holds **only path/symlink references**, never the binaries themselves.
All `.pkl`, `.npz`, `.ckpt` are listed in `.gitignore`.

---

## Required (registration walls)

### 1. SMPL — for 4D-Humans main path
- Register: https://smpl.is.tue.mpg.de/ (academic email recommended)
- Download: `SMPL_python_v.1.1.0.zip`
- Place: any local directory (called `<MODEL_ROOT>` below)
- Required file: `<MODEL_ROOT>/smpl/SMPL_NEUTRAL.pkl`
  (or alias to `basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl`)

### 2. SMPL-X — for SMPLify-X stretch path
- Register: https://smpl-x.is.tue.mpg.de/
- Download: `models_smplx_v1_1.zip`
- Required file: `<MODEL_ROOT>/smplx/SMPLX_NEUTRAL.npz`

### 3. VPoser v1 — for SMPLify-X
- Bundled with SMPL-X registration
- Required directory: `<MODEL_ROOT>/vposer_v1_0/`

---

## Auto-downloaded (no registration)

These are pulled at runtime by the demo notebooks; do NOT pre-download manually.

- **HRNet-W48 COCO**: `MMPoseInferencer('td-hm_hrnet-w48_8xb32-210e_coco-256x192')` triggers auto-download.
- **ViTDet detector**: 4D-Humans bundles a detectron2 config that downloads on first use.
- **HMR 2.0 (4D-Humans) checkpoint**:
  ```python
  from hmr2.utils.download_util import download_models, CACHE_DIR_4DHUMANS
  download_models(CACHE_DIR_4DHUMANS)
  ```

---

## License compliance (CRITICAL)

- SMPL / SMPL-X / VPoser are **non-commercial academic license** — DO NOT redistribute.
- Never commit `.pkl` / `.npz` / `.ckpt` files to git.
- Never include them in the submission `.zip` either; reviewers must register independently.
- 4D-Humans **code** is MIT, but the trained **weights** inherit MPI license.

## Local layout (recommended, outside the repo)

Place all registered models under any directory; export its path as
`CV_MODEL_ROOT` so the demo can find it:

```
$CV_MODEL_ROOT/
├── smpl/SMPL_NEUTRAL.pkl                        # symlink (created by setup_smpl_paths.sh)
├── SMPL_python_v.1.1.0/smpl/models/
│   ├── basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl   # the file as downloaded (v1.1.0)
│   └── basicModel_neutral_lbs_10_207_0_v1.0.0.pkl   # symlink alias for 4D-Humans (created by setup script)
├── smplx/SMPLX_NEUTRAL.npz
└── vposer_v1_0/                                  # only needed for SMPLify-X stretch path
```

> **About the v1.0.0 ↔ v1.1.0 alias.** Today's SMPL download from MPI is
> `v1.1.0`. Older 4D-Humans code expects the original `v1.0.0` filename
> (capital `M`); they are **mathematically equivalent** apart from minor
> numerical refinements. `setup_smpl_paths.sh` creates the alias automatically.

```bash
export CV_MODEL_ROOT=/path/to/your/models
bash demo/scripts/setup_smpl_paths.sh
```

The notebooks and `run_pipeline.py` read `CV_MODEL_ROOT` from the environment
(see `demo/src/__init__.py`); never hardcode a personal path.
