# Presentation Script — From 2D Keypoints to 4D Humans

**Speakers:** Yiqiao Liu, Taijia Liang
**Slot:** 5/11 17:00–21:00 @ 60FA 110
**Total budget:** 8 min  ·  Yiqiao 3 min (paper deep-dives) → Taijia 3 min (HRNet + bridge + 4D) → Joint live demo 2 min (Taijia voices)
**No alternation.** One continuous block per speaker.

> Deck file: `presentation/Intro to CV.pptx`  ·  rebuilt by `presentation/build_deck.py`.

---

## Time budget

| Block | Speaker | Slides | Length |
|---|---|---|---|
| 0 — Title + section open | Yiqiao | 1, 2 | 0:25 |
| 1 — OpenPose + HMR (Yiqiao's papers) | Yiqiao | 3, 4 | 2:35 |
| 2 — HRNet + 2D→3D bridge + 4D | Taijia | 5, 6, 7 | 2:30 |
| 3 — Section open + image demo + video demo | Taijia | 8, 9, 10 | 2:00 |
| 4 — Analysis & close | Taijia | 11, 12 | 0:30 |
| **Total** | | **12 slides** | **8:00** |

Each slide ≤ 90 s. **Bold** passages = say verbatim. *Italics* = visual / note to self.

---

## Block 1 — Yiqiao  (3:00, slides 1–4)

### Slide 1 — Title  (15 s)

*Visual:* template title slide — `From 2D Keypoints to 4D Humans` in purple Frank Ruhl Libre, NYU mark, two names below.

> **"Hi, we're Yiqiao and Taijia. Our project traces a single thread — how computer vision moved from 2D pixel keypoints to 3D parametric meshes, and now to 4D human tracks. I'll deep-dive two of the three papers on that thread; Taijia will cover the third, connect them, and run our live demo."**

*Advance to section divider.*

### Slide 2 — Section: Related Work & Methodology  (10 s)

*Visual:* template's section divider — purple title only.

> *"First the papers."*

### Slide 3 — OpenPose  (1:15)

*Visual:* 4 bullet points; speaker chip "Yiqiao" bottom-left; bridge note in purple at bottom.

- *Core contribution (25 s):*
  > "OpenPose's key idea is **Part Affinity Fields** — for every pixel on a limb, the network predicts a 2D vector pointing along that limb's direction. PAFs are produced alongside the usual keypoint heatmaps."
- *Why it matters (25 s):*
  > "This unlocks a **bottom-up** pipeline: detect every keypoint in the image first, then use the PAFs to bipartite-match keypoints to people. **Inference time doesn't grow with the number of people** — that's the headline result, real-time multi-person at 22 fps."
- *Limitation + bridge (25 s):*
  > "The trade-off is that keypoint precision is bounded by the network's heatmap resolution. **The next paper, HRNet, attacks exactly that limitation. And the third paper, HMR, consumes whatever 2D output we produce as supervision for 3D.**"

### Slide 4 — HMR  (1:30)

*Visual:* bullets on the left; soft-purple panel on the right with our measured runtime; chip "Yiqiao".

- *Core contribution (30 s):*
  > "HMR's headline move is **single-pass regression**. A CNN takes one RGB image and directly outputs SMPL parameters — 72-d pose, 10-d shape, 3-d camera. **No iterative optimization. This replaces SMPLify's roughly 60 seconds per image with one forward pass.**"
- *Two tricks (35 s):*
  > "Two design choices make it work. **First, an adversarial body prior** — a discriminator trained on 4 million Mocap poses keeps the predicted poses anatomically plausible. That's critical because a single 2D image has many valid 3D interpretations. **Second, a reprojection loss** — predicted 3D joints are projected back to 2D and supervised against 2D keypoints, **the same kind of output OpenPose and HRNet produce.** So the 2D papers aren't competitors — they're HMR's training signal."
- *Result + honest disclosure + handoff (25 s):*
  > "The paper claims roughly 600× speedup over SMPLify at comparable quality. We didn't run SMPLify ourselves end-to-end, but we ran HMR 2.0 — a 2023 ViT-based reimplementation that keeps the same single-pass + body prior + reprojection design — and measured **about 100 ms per image on our GPU**. **Now Taijia takes over to walk through HRNet, the bridge, and the 4D extension."*

*[Hand mic / advance.]*

---

## Block 2 — Taijia  (3:00, slides 5–7)

> Take over without re-introducing the project. Land on the bridge slide — that's the centerpiece.

### Slide 5 — HRNet  (45 s)

*Visual:* 3 bullets on the left, **our HRNet output on the yoga photo** on the right with caption; chip "Taijia".

- *Core contribution (15 s):*
  > "Thanks Yiqiao. HRNet takes the **opposite** path from OpenPose. Instead of down-sampling and back up, it maintains **parallel multi-resolution branches end-to-end** with cross-scale fusion. **The high-resolution stream is never lost.**"
- *Empirical evidence on our demo (20 s):*
  > "**This image on the right is HRNet's output on a yoga photo from our pipeline.** Look at the wrist, the ankle, the toe positions — that crisp localization is the empirical consequence of high-resolution preservation. A network that down-samples would smear those joints."
- *Trade-off + bridge (10 s):*
  > "The trade-off: HRNet is top-down — it needs a person detector first, so cost grows linearly with the number of people. **OpenPose and HRNet are two complementary endpoints of the 2D paradigm.**"

### Slide 6 — From 2D to 3D — SMPL & Reprojection  (1:00)  ★ **the pivot**

*Visual:* SMPL formula at top; box-diagram OpenPose+HRNet → 2D kpts → HMR → reproject → loss; right side shows our reprojection-overlay PNG.

- *Set up (15 s):*
  > "Now the question is — how does 2D become 3D? Two enablers. **First, SMPL. A parametric body model that maps a 72-dimensional pose plus a 10-dimensional shape vector to a 6890-vertex mesh. Without SMPL we'd be regressing 6890 vertices directly — not learnable. With it, we're regressing 82 numbers.**"
- *The bridge mechanism (25 s):*
  > "**Second, the reprojection loss — and this is the punchline.** Look at this diagram. OpenPose and HRNet produce 2D keypoints. HMR predicts 3D joints. **Project those 3D joints back to 2D. The L2 distance against the HRNet-style 2D output is the reprojection loss that trains HMR. The 2D papers literally are HMR's training signal — not its competitors.**"
- *Empirical confirmation (20 s):*
  > "**This figure on the right is our own evidence.** Cyan circles are HRNet's 2D detections on the standing image. Red Xs are HMR's 3D joints projected back through HMR's predicted camera. They're visibly close — the bridge isn't just conceptual, it's quantitatively self-consistent on our pipeline."

### Slide 7 — From 3D to 4D — HMR 2.0 + PHALP  (45 s)

*Visual:* 4 bullets on the left, a PHALP frame on the right; chip "Taijia".

- *Mechanism (25 s):*
  > "Once you have single-image 3D, the next axis is time. **HMR 2.0**, from 2023, replaces ResNet with ViT but keeps the same single-pass + reprojection + body prior design. **PHALP**, from 2022, runs HMR 2.0 on every video frame, then stitches identities across frames using appearance and pose. **No new mechanism — the regression in a temporal stack.**"
- *Output framing (20 s):*
  > "**The output is no longer a rendered video. It's a per-frame SMPL time-series.** Pose vectors, joint trajectories, body shape, all timestamped. That's the actual product of '4D human understanding' — structured 3D data, ready for biomechanics, animation, sports analytics. **Let me show you what it looks like running on our hardware.**"

*[Advance to section divider.]*

---

## Block 3 — Live demo  (2:00, slides 8–10)

### Slide 8 — Section: Preliminary Results  (5 s)

*Visual:* template's section divider.

> *"Two demos — image then video."*

### Slide 9 — Image pipeline  (1:00)

*Visual:* 4 quad-plot thumbnails in a row, captions in purple, subtitle explaining each panel; chip "Taijia"; bottom note "Same forward pass — no per-image tuning, no iteration."

> *Open (10 s):*
> "These are our four test photos. Each panel shows the input, HRNet's 2D, HMR 2.0's mesh from the front, and the same mesh from the side. Roughly **100 ms per image**, end-to-end."

> *Walk through (40 s):*
> "Standing — clean baseline, the mesh hugs the body. Yoga — **complex pose, no explosion. That's the adversarial prior in action — a network without that prior would output something anatomically broken.** Occluded — most of the body is hidden behind the leaves; the prior is filling in joints we never observed. Multi-person — top-down means the detector crops each person, then HMR runs per crop independently."

> *Land (10 s):*
> "**Same forward pass for all four. No per-image tuning, no iteration. That's the regression paradigm at work.**"

### Slide 10 — Video pipeline  (1:00)

*Visual:* embedded `tennis_clip.mp4` (or static frame) on the left, 4 bullets on the right, caption underneath the video.

> *Press play / point at the visual (40 s):*
> "Eight-and-a-half seconds of tennis, 216 frames. **Every frame independently fits SMPL via HMR 2.0 — the mesh you see is regenerated each frame. PHALP keeps the player's identity stable across the motion blur of the serve.** Notice the mesh holds through the full extension and follow-through — body prior again."

> *The point (20 s):*
> "**The valuable output isn't this video — it's the structured time-series behind it. Per-frame pose vectors and 3D joint positions, ready to feed Blender for animation, or biomechanics tools for serve mechanics, or coaching analytics. That's what '4D' delivers that an image classifier saying 'tennis' never could.**"

*[Advance.]*

---

## Block 4 — Close  (30 s, slides 11–12)

### Slide 11 — Analysis, Limitations & Next Steps  (30 s)

*Visual:* three columns — Take-aways (purple), Limitations (red), What comes next (blue).

> "Three take-aways. **One — pipeline thinking:** OpenPose and HRNet aren't replaced by HMR; they supervise it. **Two — single-pass paradigm wins** on every metric except absolute precision. **Three — adding the time axis isn't a new field; it's the natural next step.**"
>
> "Three honest limitations: per-frame jitter, narrow training domains, and meshes that lack clothing or hair. Three forward directions: **VIBE/SLAHMR** for temporal smoothing, **Sapiens** as a human foundation model, and **3DGS-Avatar** for neural surfaces."
>
> "**The story from 2018 to today is one chain — every step adds a dimension on top of what was already there. Thank you.**"

### Slide 12 — Thank you  (5 s)

*Visual:* template thanks slide.

---

## Pre-talk checklist

- [x] Deck rebuilt:  `python presentation/build_deck.py`  →  `Intro to CV.pptx`
- [x] Image asset:  `demo/results/2d_vis/img2_complex_pose.jpg`        (slide 5)
- [x] Image asset:  `demo/results/extras/reproj_img1_standing.png`     (slide 6)
- [x] Image asset:  `demo/results/tennis_phalp/frame_2.0s.png`         (slide 7)
- [x] Image assets: `demo/results/quadplot_img{1..4}_*.png`            (slide 9)
- [x] Video asset:  `demo/results/extras/tennis_clip.mp4`              (slide 10)
- [ ] Open the pptx in PowerPoint (Mac/Windows) once before rehearsal — Frank Ruhl Libre may need to be installed; otherwise it falls back to the OS default serif which still reads cleanly.
- [ ] If `tennis_clip.mp4` did not embed (build script logs `embedded=False`), insert it manually in PowerPoint over the static frame.
- [ ] Backup: have the full `PHALP_tennis.mp4` queued in a window for slide 10 fallback.

## Q&A prep — likely questions

| Likely question | Short answer |
|---|---|
| Why HMR 2.0 instead of HMR 1.0? | "HMR 1.0's open code is hard to run on modern hardware. HMR 2.0 keeps the core insight — single-pass + adversarial prior + reprojection — and is the contemporary instantiation. Our demo evidence is for the design choices, not the specific weights." |
| Did you reproduce the 600× speedup? | "We measured HMR 2.0 at ~100 ms; SMPLify's ~60 s comes from the paper. We also wrote an 80-line PyTorch SMPLify and got around 1 s per image on our GPU — about 10× HMR. The 600× gap shrinks on modern hardware; the *scaling* story holds." |
| Why is the tennis demo's track ID stable? | "The player is alone in frame, so PHALP's tracking has an easy job. We did disable PHALP's neural-renderer texture module due to a CUDA conflict; pose tracking is unaffected." |
| What's missing from your demo for it to be a research result? | "Quantitative MPJPE on a held-out dataset, multi-person tracking with the texture module enabled, and an ablation on the body prior." |
| Why didn't you also run OpenPose? | "Two reasons: (1) it would duplicate the 2D output category HRNet already covers; (2) for our literature review OpenPose is a conceptual milestone — bottom-up paradigm — not the demo target." |

## Speaker handoff cues

- End of Yiqiao slide 4: **"Now Taijia takes over to walk through HRNet, the bridge, and the 4D extension."**
- Start of Taijia slide 5: **"Thanks Yiqiao. HRNet takes the *opposite* path from OpenPose…"**

## Extras we generated but did NOT put on slides (Q&A backup only)

These artifacts exist under `demo/results/extras/` and can be opened from
the file system if a question pulls in that direction:

- `pose_similarity.png`  — 4×4 SMPL-pose distance matrix across the test images.
- `tennis_wrist_3d.png` + `tennis_joint_angles.png` — per-frame trajectories.
- `mini_smplify.png` + `mini_smplify_runtime.csv` — our own SMPLify reimplementation.
- `reproj_img{2..4}_*.png` — the reprojection overlay applied to the other test images.

Pulling any of these in mid-talk is risky — they need their own framing.
Keep them out of the deck unless asked.
