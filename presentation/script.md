# Presentation Script — From 2D Keypoints to 4D Humans

**Speakers:** Yiqiao Liu, Taijia Liang
**Slot:** 5/11 17:00–21:00 @ 60FA 110
**Total budget:** 8 min — Yiqiao 3 min (paper deep-dives) → Taijia 3 min (frame + connect + HRNet) → Joint live demo 2 min (Taijia voices)
**Speaking order:** No alternation. Yiqiao does one continuous block, then Taijia does one continuous block (which includes the demo).

---

## Time budget

| Block | Speaker | Slides | Length |
|---|---|---|---|
| 1 — Paper deep-dives | Yiqiao | 1–3 | 3:00 |
| 2 — Frame + HRNet + Connect | Taijia | 4–7 | 3:00 |
| 3 — Live demo + close | Taijia | 8–9 | 2:00 |
| **Total** | | **9** | **8:00** |

Each slide ≤ 60s. Bold passages = say verbatim. Italics = visual / note to self.

---

## Block 1 — Yiqiao (3:00)

> Yiqiao opens cold. He has to set up the project topic in one sentence so the paper deep-dives don't feel context-less, then dive into OpenPose and HMR. Hand off to Taijia at the end.

### Slide 1 — Opener + paper menu (20s)

*Visual:* project title centered. Below it, three paper cards in a row: **OpenPose · HRNet · HMR**. Each card has a small icon (skeleton / heatmap / mesh). Two cards (OpenPose, HMR) are highlighted = "I'll cover these"; the third (HRNet) is dimmed = "Taijia covers".

**Speaker notes:**

> "We picked three landmark papers tracing how computer vision moved from 2D pixel-space pose estimation to 3D human mesh recovery and beyond. **I'll deep-dive two of them — OpenPose and HMR — and then Taijia will cover HRNet, tie all three together, and show our demo.** Let's start."

### Slide 2 — OpenPose (Cao et al., TPAMI 2019) — 80s

*Visual:* Left half = paper Fig 1 (the two-branch network: confidence maps + Part Affinity Fields). Right half = a simple bottom-up vs top-down comparison diagram. Bottom strip = `MPII multi-person mAP 75.6 · 22 fps · O(1) wrt #people`.

**Speaker notes:**

> *Core contribution (25s):* "The key idea is **Part Affinity Fields** — a 2D vector field that, for each pixel on a limb, points along that limb's direction. The network produces these PAFs alongside the usual keypoint heatmaps."

> *Why it matters (25s):* "This unlocks a **bottom-up** pipeline: detect every keypoint in the image, then use PAFs to score which keypoints belong to the same person via bipartite matching. Crucially, **inference time doesn't grow with the number of people** — that's the headline result, real-time multi-person pose at 22 fps."

> *Limitation + bridge (30s):* "The trade-off is that keypoint precision is bounded by the network's heatmap resolution. The next paper, HRNet, attacks exactly that limitation, and Taijia will cover it. **Now to the third paper, HMR — the jump from 2D to 3D.**"

### Slide 3 — HMR (Kanazawa et al., CVPR 2018) — 80s

*Visual:* Top: paper Fig 3 (CNN encoder → SMPL params + camera, with discriminator on the right). **Two red boxes** highlight: (1) "Adversarial body prior — discriminator trained on 4M Mocap poses"; (2) "Reprojection loss — uses 2D keypoints as supervision". Bottom: runtime table.

| Method | Year | Runtime | Source |
|---|---|---|---|
| SMPLify (iterative) | 2016 | ~60 s/img | paper |
| HMR 1.0 (ResNet-50) | 2018 | ~150 ms/img | paper |
| HMR 2.0 (ViT-H, what we run) | 2023 | **~100 ms/img** | **our measurement, RTX 6000** |

**Speaker notes:**

> *Core contribution (25s):* "HMR's headline move is **single-pass regression**: a CNN takes one RGB image and directly outputs SMPL parameters — 72-d pose, 10-d shape, 3-d camera. No iterative optimization. **This replaces SMPLify's ~60 seconds per image with a single forward pass.**"

> *Two tricks make it work (30s):* "First, **an adversarial body prior** — a discriminator trained on 4 million Mocap poses keeps the predicted poses anatomically plausible, which is critical because a single 2D image has many valid 3D interpretations. Second, **a reprojection loss** — predicted 3D joints are projected back to 2D and supervised against 2D keypoints, **the same kind of output OpenPose and HRNet produce.** So the 2D papers aren't competitors — they're HMR's training signal."

> *Result + honest disclosure + bridge (25s):* "The paper reports a roughly 600× speedup over SMPLify at comparable quality. We didn't reproduce SMPLify, but we ran HMR 2.0 — a 2023 ViT-based reimplementation that keeps the same single-pass + body prior + reprojection design — and measured ~100 ms per image on our GPU, confirming the single-pass paradigm's speed claim. **I'll hand off to Taijia now to set up the bigger picture and walk through HRNet, the connections between all three papers, and our live demo.**"

*[Hand mic / advance to Taijia.]*

---

## Block 2 — Taijia (3:00)

> Taijia takes over. He frames the research question retroactively (audience has now heard 2 papers, so framing lands), introduces SMPL as the parametric body model that HMR outputs, deep-dives HRNet (his paper) tied to our demo's 2D output, and then explicitly connects all three papers via a diagram. Ends with a 15s transition into the live demo.

### Slide 4 — Research question + SMPL background (60s)

*Visual:* Top half = research question in big text. Bottom half: left = simplified SMPL formula `M(θ, β) → 6890 vertices`; right = our smoke-test rendered SMPL T-pose mesh (rotating GIF of `tpose_smplx.obj`).

**Speaker notes:**

> *Frame retroactively (25s):* "Thanks Yiqiao. **Now that you've heard OpenPose and HMR, here's the question we're really asking:** *How did human understanding climb from 2D pixels, to 3D parametric mesh, and now toward 4D temporal humans — and what design choices made each leap possible?* The two papers Yiqiao covered sit at two rungs of that ladder. HRNet sits on the 2D rung with OpenPose; everything else builds toward 3D and 4D."

> *SMPL as the bridge representation (35s):* "One concept ties this all together: **SMPL** — a parametric body model that maps a 72-dimensional pose vector and a 10-dimensional shape vector to a 6890-vertex 3D mesh. SMPL is what HMR outputs. **It's also what makes 'recovering 3D from a 2D image' tractable — instead of regressing 6890 vertices directly, HMR regresses 82 parameters.** This rotating mesh on the right is from our own SMPL-X smoke test, not a paper figure. Now to HRNet."

### Slide 5 — HRNet (Sun et al., CVPR 2019) + our 2D demo (75s)

*Visual:* Left third = paper Fig 1 (the parallel multi-resolution branches). Middle third = **our HRNet output on the yoga image** (clean keypoint+skeleton overlay, hi-res). Right third = a small comparison table:

| | OpenPose | HRNet |
|---|---|---|
| Paradigm | bottom-up | top-down |
| Cost vs N | O(1) | O(N) |
| Strength | speed / scalability | per-keypoint precision |

**Speaker notes:**

> *Core contribution (25s):* "HRNet takes the **opposite** path from OpenPose. Instead of down-sampling and back up like an hourglass, it maintains **parallel multi-resolution branches end-to-end** with cross-resolution feature exchange. The high-resolution stream is never lost."

> *Empirical evidence on our demo (30s):* "**This middle image is HRNet's output on a yoga photo we ran through our pipeline.** Look at the wrist, the ankle, the toes — **that crisp localization is the empirical consequence of high-resolution preservation.** A network that down-sampled would smear those joints. The trade-off, shown in the right table, is that HRNet is top-down — it needs a person detector first, so inference cost grows linearly with the number of people. OpenPose and HRNet are two complementary endpoints of the 2D paradigm space."

> *Bridge (20s):* "But 2D is just pixels. To get 3D, you need a body prior — which brings us back to HMR. **Let me show you how the three papers fit together.**"

### Slide 6 — Connections: three papers as one pipeline (60s)

*Visual:* Centered diagram:

```
   ┌──────────────┐    ┌──────────────┐
   │   OpenPose   │    │    HRNet     │
   │  (bottom-up) │    │  (top-down)  │
   └──────┬───────┘    └──────┬───────┘
          │   2D keypoints    │
          └─────────┬─────────┘
                    │ supervision via
                    │ reprojection loss
                    ▼
              ┌──────────────┐
              │     HMR      │
              │  (3D SMPL)   │
              └──────────────┘
                    │
                    │ shared limitation:
                    │ frame-independent
                    ▼
                  4D ?  → demo
```

Three numbered take-aways below the diagram:
1. OpenPose ↔ HRNet — complementary 2D paradigms
2. 2D papers → HMR — supervision via reprojection
3. Shared limitation — no temporal coherence

**Speaker notes:**

> "Three things tie these papers together. **First**, OpenPose and HRNet aren't replacements — they're complementary 2D paradigms, one optimized for scalability, the other for precision. **Second, and this is the key insight:** OpenPose and HRNet are not just predecessors of HMR — **they're HMR's supervision source.** HMR's reprojection loss takes its predicted 3D joints, projects them back to 2D, and compares against the 2D keypoints from exactly the kind of detectors OpenPose and HRNet pioneered. **The 2D papers literally train the 3D paper.** **Third**, all three share one limitation: they operate on a single frame. No temporal coherence. So the natural next question is: *what happens when we add the time dimension?* Let me show you."

### Slide 7 — From 3D to 4D (15s, transition)

*Visual:* One-line timeline `HMR (2018, single image) → PHALP (2022, video) → Sapiens (2024, foundation)`. Big arrow points to a play-button graphic.

**Speaker notes:**

> "We extended HMR-style methods to a real video — a tennis serve — using PHALP, a 2022 paper that adds cross-frame appearance tracking on top of per-frame HMR. **Live demo.**"

---

## Block 3 — Live Demo (Taijia voices, 2:00)

> Single speaker continuity preserved. Taijia drives both the video play and the data overlay. Two slides.

### Slide 8 — LIVE: tennis PHALP video + pickle data (90s)

*Visual:* Slide is split:
- Top half: video player embedded — `presentation/assets/tennis_phalp_clip.mp4` (5–6 s loop of high-action segment, looped 2-3× to fill 50s)
- Bottom half: two static plots from `demo_tennis.pkl`
  - Left: right-elbow joint angle (degrees) over frames
  - Right: 3D joint scatter for one frame (SMPL skeleton in 3D)

**Speaker notes:**

> *Phase 1 — video plays (50s):*
>
> *0–10s of video, first loop:* "Original video on the left of every frame, SMPL mesh overlay on the right. **Notice how the mesh hugs the body in real time — not a generic skeleton, an actual parametric body shape.**"
>
> *10–25s, serving motion:* "Mid-serve — the player's arm is fully extended, body arched. **The mesh holds — that's HMR's adversarial body prior in action, frame by frame.** Without that prior, single-image 3D from this kind of motion blur would explode into nonsense."
>
> *25–40s, walking afterward:* "Stand-up posture, walking — natural transition. The track ID flickers occasionally because we disabled PHALP's appearance-feature module due to an incompatible CUDA dependency. Pose estimation is unaffected."
>
> *40–50s, end:* "Eight seconds of video, 216 frames, every frame independently fits SMPL via HMR-2 — 4D, in the original sense of '3D + time'."
>
> *Phase 2 — pickle data (40s, switch focus to bottom plots):*
>
> "But this is the part I want you to remember. **The output isn't a pretty video — it's structured 3D data.** This left plot is the right-elbow angle extracted from the pickle file PHALP saved, frame by frame. You can see the serving motion — extension, peak, follow-through. **This is biomechanical data ready to feed sport analytics, motion-capture pipelines, or Blender for animation.** That's the difference between an image classifier saying 'tennis' and a structured 3D understanding of *how* the human moved. **This is what '4D human understanding' actually delivers.**"

### Slide 9 — Take-aways + open question (30s)

*Visual:* Three numbered take-aways, then an open question, then thanks/contact:

1. **Pipeline thinking** — OpenPose & HRNet are HMR's supervision, not its competitors.
2. **Single-pass paradigm wins** — ~600× speedup over optimization at comparable quality.
3. **Adding time** — HMR → PHALP → 4D Humans is a natural extension, not a new field.

> *Open question:* When 4D-video and multi-person are solved, is the next leap action understanding, physics-aware interaction, or unified human foundation models like Sapiens?

Bottom: GitHub URL, names, dates.

**Speaker notes:**

> "Three take-aways. **One — pipeline thinking:** OpenPose and HRNet aren't replaced by HMR; they supervise it. **Two — single-pass wins:** ~600× speedup at comparable quality is what let image-to-3D leave the lab. **Three — the time dimension is the natural next step**, and the field is already there with PHALP and 4D-Humans. **Open question for you:** when 4D video plus multi-person is solved, what's the next leap — action understanding, physics-aware interaction, or human-centric foundation models like Sapiens? **Thank you.**"

---

## Pre-talk checklist

- [ ] `presentation/assets/tennis_phalp_clip.mp4` — 5–6 s clip, looping segment cut from `demo/results/tennis_phalp/PHALP_tennis.mp4`
- [ ] `presentation/assets/elbow_angle_plot.png` — extracted from `demo/results/tennis_phalp/results/demo_tennis.pkl`
- [ ] `presentation/assets/joint_3d_scatter.png` — single-frame 3D joint plot
- [ ] Slide 5: HRNet 2D output on yoga image — ready at `demo/results/2d_vis/img2_complex_pose.jpg`
- [ ] Slide 4: T-pose rotating GIF — ready at `demo/results/tpose_rotation.gif`
- [ ] Slide 3: HMR runtime table populated with our measured ~100 ms
- [ ] Slide 6: connection diagram — to draw or set as ASCII text
- [ ] Backup: have the full 8-second `PHALP_tennis.mp4` queued in a window in case of network/embed issues

## Q&A prep — likely questions

| Likely question | Short answer |
|---|---|
| Why did you use HMR 2.0 instead of HMR 1.0? | "HMR 1.0's open code is hard to run on modern hardware. HMR 2.0 keeps the core insight — single-pass + adversarial prior + reprojection — and is the contemporary instantiation. We're treating it as evidence the design choices held up." |
| Did you reproduce the 600× speedup? | "We measured HMR 2.0 at ~100 ms; we cited SMPLify's ~60 s from the paper. We didn't run SMPLify ourselves — that requires an old PyTorch stack we judged not worth a separate environment for. The paper's 600× claim is consistent with our 100 ms measurement." |
| Why is your tennis demo's track ID flickering? | "We disabled PHALP's neural-renderer-based texture module due to a CUDA-version conflict on our hardware. Pose-only tracking is what's running. Pose estimation per-frame is unaffected." |
| What's missing from your demo to make it a research result? | "Quantitative MPJPE on a held-out dataset, multi-person tracking with the texture module enabled, and an ablation on the body prior." |
| Why didn't you also run OpenPose? | "Two reasons: (1) it would duplicate the 2D output category HRNet already covers in the demo; (2) the literature review's role for OpenPose is conceptual — bottom-up paradigm — not the hands-on demo." |

## Speaker handoff cues

- End of Yiqiao's slide 3: "**I'll hand off to Taijia now to set up the bigger picture and walk through HRNet, the connections between all three papers, and our live demo.**"
- Start of Taijia's slide 4: "**Thanks Yiqiao. Now that you've heard OpenPose and HMR, here's the question we're really asking…**"

## Open follow-ups

- Need a 5th test image (selfie) per the original plan — *not* required for slides above (we cite "4 in-the-wild images") but nice to have for the report.
- Optional: extract elbow angle from pickle and produce the static plot for slide 8 — this is the "extending the demo" deliverable.
- Optional: if time permits in rehearsal, compress slide 5 to 60s to give 15s of buffer.
