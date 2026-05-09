# From 2D Pose to 3D Human Recovery

**Research question:** *How has the field progressed from estimating 2D keypoints in pixel space to recovering a full 3D parametric body shape from a single RGB image, and what design choices in representation, supervision, and architecture enabled this leap?*

This repository contains the literature review, slides, and a runnable working
model for the **Intro to Computer Vision (Spring 2026) Final Project**.

We trace the 2D-to-3D progression through three landmark papers and empirically
validate HMR's central claim — *single-pass regression delivers ~600× speedup
over iterative SMPLify optimization at comparable quality* — on five
in-the-wild images.

| Paper | Role in our story |
|---|---|
| **OpenPose** — Cao et al., TPAMI 2019 | Bottom-up multi-person 2D via Part Affinity Fields |
| **HRNet** — Sun et al., CVPR 2019 | Top-down high-resolution 2D backbone (used as our 2D stage) |
| **HMR** — Kanazawa et al., CVPR 2018 | Single-image 3D SMPL recovery with an adversarial body prior |

## Repository layout

| Path | What lives there |
|---|---|
| [`demo/`](demo/README.md) | Working model — Mac smoke test + Linux/Colab full pipeline |
| [`LiteratureReview/`](LiteratureReview/) | 2-page LaTeX report (`main.tex`, `refs.bib`, `main.pdf`) |
| [`papers/`](papers/) | The three core papers (PDFs) |
| [`.claude/plan/cv-final-2d-to-3d-v3.md`](.claude/plan/cv-final-2d-to-3d-v3.md) | Latest plan (two rounds of multi-model review applied) |

## Quick start

```bash
# 1. Clone and create a Python env
git clone git@github.com:ethanliu001111-lang/CVFinal.git && cd CVFinal
python3 -m venv .venv && source .venv/bin/activate
pip install -r demo/envs/requirements_local_mac.txt

# 2. Register and download SMPL/SMPL-X (one-time, see demo/checkpoints/DOWNLOAD.md)
export CV_MODEL_ROOT=/path/to/registered/models
bash demo/scripts/setup_smpl_paths.sh

# 3. Local Mac smoke test (~10 s, verifies SMPL-X loading + matplotlib rendering)
bash demo/scripts/run_smoke_test.sh

# 4. Full pipeline (Linux GPU server preferred — see demo/README.md)
python demo/scripts/run_pipeline.py demo/test_images/*.jpg --device cuda
```

## Course information

- **Submission**: Sunday, May 10, 2026 — 11:59 PM EDT
- **Presentation**: Monday, May 11, 2026 — 5:00–9:00 PM, **60FA 110**
- **Deliverables**: presentation slides + 2-page LaTeX report + working model (this repo)

## Plan and responsibility

### Yiqiao Liu — OpenPose / HMR / Demo / Future
- §3.1 OpenPose section (drafted, finalize 5/7)
- §3.3 HMR section (write 5/6, finalize 5/7)
- §5 Future Directions incl. Sapiens / VIBE / 3DGS-Avatar (write 5/6)
- Working model: 4D-Humans (HMR 2.0) demo on 5 test images, deliver 5/8
- Slides 8–13 (HMR + Comparison + Demo + Future)
- Live presentation: HMR + Demo walkthrough (≥5 min, second half)

### Taijia Liang — Intro / Background / HRNet / Comparison
- §1 Introduction (write 5/7)
- §2 Background incl. SMPL / SMPLify / CPM / Hourglass (write 5/7)
- §3.2 HRNet section (write 5/6, finalize 5/7)
- §4 Connections, Differences, Limitations + comparison table (write 5/6)
- `refs.bib` maintenance — 11 entries already populated, top up if needed
- HRNet 2D-keypoint demo via MMPose for the comparison figures (5/5, 5/8)
- Slides 1–7 (Intro + Background + OpenPose + HRNet)
- LaTeX 2-page compile verification (5/8 dry-run, 5/10 final)
- Live presentation: Intro + HRNet + Comparison (≥5 min, first half)

### Joint
- 5/9: full 10-min slide rehearsal (verify pacing and per-person time)
- 5/10: cross-review and final submission to Brightspace
- 5/11 17:00–21:00: live presentation at 60FA 110

## Daily checkpoints (5/3 → 5/11)

| Day | Date | Focus | Hard checkpoint | Status |
|---|---|---|---|---|
| 0 | 5/3 (Sun) | Plan locked + Mac smoke test + project skeleton + GitHub repo | SMPL-X loads on Mac; `demo/` scaffolded; double multi-agent review passed | ✅ done |
| 1 | 5/4 (Mon) | Linux/Colab env setup, 5 CC0 test images chosen | `run_pipeline.py` runs end-to-end on a single image | ⏳ |
| 2 | 5/5 (Tue) | First 5-image outputs + stretch decision | 5 quad-plots saved; SMPLify-X go/no-go decided | ⏳ |
| 3 | 5/6 (Wed) | Report main body | §3.2 / §3.3 / §4 / §5 drafts in `main.tex` | ⏳ |
| 4 | 5/7 (Thu) | Report integration | §1 / §2 done; refs.bib finalized | ⏳ |
| 5 | 5/8 (Fri) | Demo polish + 2-page compile dry-run | `full_demo.mp4` + GitHub link ready; `latexmk` ≤ 2 pages | ⏳ |
| 6 | 5/9 (Sat) | PPT v1 + full rehearsal | 13 slides, single handoff (Taijia 5min → Yiqiao 5min) | ⏳ |
| 7 | 5/10 (Sun) | **Submission** | 4 deliverables on Brightspace by 23:59 EDT | ⏳ |
| 8 | 5/11 (Mon) | **Live presentation** | 60FA 110, 17:00–21:00 | ⏳ |

> Day numbering follows the v3 plan (5/3 = Day 0).

## Today's progress (5/3)

- ✅ SMPL/SMPL-X/VPoser model files already downloaded (placed under `$CV_MODEL_ROOT`)
- ✅ Symlinks wired up via `bash demo/scripts/setup_smpl_paths.sh`
- ✅ Mac venv + smoke test passed (SMPL-X T-pose forward + matplotlib render)
- ✅ Project skeleton scaffolded under `demo/` (6 src modules + 4 notebooks + envs + Linux deploy scripts)
- ✅ Generated `pipeline_diagram.png`, `tpose_smplx.obj`, `smplx_4poses.png`, `tpose_rotation.gif`
- ✅ Two rounds of codex + gemini multi-agent review applied; all critical findings inline-patched

## License & third-party models

This repository is MIT-licensed (see [`LICENSE`](LICENSE)). The SMPL, SMPL-X,
VPoser, and 4D-Humans model weights are **not** distributed here; each user
must register and download them independently. Full third-party license matrix
in [`demo/checkpoints/DOWNLOAD.md`](demo/checkpoints/DOWNLOAD.md).
