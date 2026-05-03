# 📋 CV Final Project 实施计划：From 2D Pose to 3D Human Recovery

**项目方向**: Human Shape and Pose Recognition and Estimation (PDF §A.15)
**团队**: Yiqiao Liu + Taijia Liang (2 人)
**演讲时长**: 5 min × 2 = 10 min（每人至少 5 分钟）
**截止时间**: 2026-05-10 23:59 EDT 提交 / 2026-05-11 17:00–21:00 现场展示（60FA 110）
**今天日期**: 2026-05-02 (剩余 8 天 → 提交，9 天 → 展示)

---

## (A) 整体故事线 & Research Question

### 🎯 Research Question（一句话锁定）

> **"How has the field progressed from estimating 2D human keypoints in pixel space to recovering full 3D parametric body shape and pose from a single RGB image — and what design choices in representation, supervision, and architecture enabled this leap?"**

中文版（自检用）："**单张 RGB 图像下，人体姿态估计如何从 2D 关键点跨越到 3D 参数化人体重建？三种代表性方法（OpenPose / HRNet / HMR）在表示、监督、架构上的关键设计差异是什么？**"

### 📐 故事线（pipeline 三段式）

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ Stage 1: 多人 2D  │ →  │ Stage 2: 高精度   │ →  │ Stage 3: 单图 3D  │
│ OpenPose (PAFs)  │    │ HRNet (HR repr.) │    │ HMR (SMPL+adv.)  │
│ Bottom-up 关联    │    │ Top-down heatmap │    │ Param. regression│
└──────────────────┘    └──────────────────┘    └──────────────────┘
   解决 "Who is who"        解决 "Where exactly"     解决 "What in 3D"
```

**串联逻辑（PPT/Report 共用的核心叙事）**：
1. **维度跃迁** — 从 2D 像素坐标 → 3D 网格表面（一个 vs 一组关键点 → SMPL 6890 顶点 + 24 关节角）
2. **监督升级** — 2D ground-truth keypoints（COCO/MPII） → 弱监督 + 3D 对抗先验（突破 3D 标注稀缺）
3. **架构演进** — Multi-stage refinement (OpenPose) → Multi-resolution parallel (HRNet) → Iterative regression + GAN prior (HMR)
4. **应用闭环** — 视频动捕 / VR avatar / AR 试衣 / 体育分析 / 医疗康复 — 每段都对应实际产业落地

### 🔑 三个"贯穿主题"（连接三篇论文的 thread）

| Thread | OpenPose | HRNet | HMR |
|--------|----------|-------|-----|
| **空间精度 vs 速度** | PAFs 实时多人，精度中等 | 全程高分辨率，精度 SOTA | 单次前向，3D 速度优先 |
| **监督信号** | 全监督 2D keypoints | 全监督 2D heatmap | 弱监督（reprojection + adv.）|
| **结构先验** | 骨架二部图 + 贪心解析 | 多分辨率并行融合 | SMPL 参数 + 对抗判别器 |

---

## (B) 论文分工 & 衔接 + 补充论文

### 📄 LaTeX Report 结构（2 页严格上限）

```
§1 Introduction (~0.25 页)         — Taijia
   - Motivation: 动捕/VR/医疗应用
   - Research question: 2D → 3D 跨越
   - Roadmap: 三篇论文+连接

§2 Related Work / Background (~0.2 页) — 共写
   - 经典史: Shotton 2011 (Kinect), DPM-based pose
   - SMPL (Loper et al. 2015) 必须提及（HMR 的基础）

§3 Three Core Methods (~1.0 页, 三 subsection)
   §3.1 OpenPose (Cao 2019)         — Yiqiao（已有初稿）
   §3.2 HRNet (Sun 2019)            — Taijia
   §3.3 HMR (Kanazawa 2018)         — Yiqiao

§4 Connections, Differences & Limitations (~0.35 页) — Taijia
   - 表格: 输入/输出/监督/速度/精度对比
   - 衔接: HRNet keypoints → HMR 输入 (实际 PyMAF 已这样做)
   - 共同短板: 遮挡、多人交互、动态服装

§5 Future Directions & Conclusion (~0.2 页) — Yiqiao
   - Foundation models (Sapiens 2024)
   - Video temporal coherence (VIBE/MEVA)
   - Implicit 3D (PIFu/PIFuHD/ICON)
   - Working model 1 行链接

References (BibTeX) — 共维护
```

### 📚 必须补充的引用论文（7 篇，refs.bib 需要新增）

| # | 论文 | 用途 | 引用位置 |
|---|------|------|----------|
| 1 | **SMPL** — Loper et al., TOG 2015 | HMR 的参数化人体模型基础 | §2 + §3.3 |
| 2 | **SMPLify** — Bogo et al., ECCV 2016 | "2D keypoints → SMPL fitting" 的优化路线，与 HMR 的回归路线对比 | §3.3 + §4 |
| 3 | **CPM (Convolutional Pose Machines)** — Wei et al., CVPR 2016 | OpenPose 的多阶段 refine 思路源头 | §3.1 |
| 4 | **Stacked Hourglass** — Newell et al., ECCV 2016 | HRNet 之前的 SOTA，对比锚点 | §3.2 |
| 5 | **Sapiens** — Khirodkar et al., ECCV 2024 | 2024 foundation model（PDF 标注 BONUS）| §5 future |
| 6 | **VIBE** — Kocabas et al., CVPR 2020 | 把 HMR 扩展到视频 | §5 future |
| 7 | **PyMAF / HMR 2.0** — Zhang et al., ICCV 2021 / Goel 2023 | HMR 后续改进（feature pyramid alignment） | §4 + §5 |

可选（presentation 提及但不必引用）：
- **DensePose** (Güler 2018) — pixel-to-surface 的 dense alternative
- **PIFu / PIFuHD** (Saito 2019/2020) — 隐式 3D，有衣服细节
- **3DGS Avatar** (2024 多篇) — 把 3DGS 用在人体

### 🎤 Presentation (PPT) 分工

**总长 10 min ≈ 12-15 张 slide**

| Slide | 时长 | 内容 | 主讲 |
|-------|------|------|------|
| 1 | 30s | Title + Motivation hook（一段 demo 视频 / 吸睛图） | Taijia |
| 2 | 1m | Problem formulation: 2D vs 3D, why hard | Taijia |
| 3 | 30s | 三篇论文路线图（一张 pipeline 图） | Taijia |
| 4-5 | 2m | OpenPose: PAFs 直觉 + 关联算法 + 实时 demo gif | Yiqiao |
| 6-7 | 2m | HRNet: 多分辨率融合架构 + COCO SOTA 表 | Taijia |
| 8-9 | 2m | HMR: SMPL 模型 + 对抗训练 + 单图 3D demo | Yiqiao |
| 10 | 1m | 三方对比表 + 共同 limitations | Taijia |
| 11 | 1m | Working model live demo（见 C 节） | Yiqiao |
| 12 | 30s | Future: Sapiens, VIBE, 3DGS-Avatar | Yiqiao |
| 13 | 0 | Q&A | 共答 |

---

## (C) Working Model / Demo 选择 + 复现路径

### 🏆 推荐：HMR 单图 3D 重建 demo（命中 bonus 概率最高）

**理由（综合 4 维度评分，满分 5）**：

| 候选 | 可达性 | 视觉冲击 | bonus 命中 | 解释新概念 | 总分 |
|------|--------|----------|-----------|-----------|------|
| OpenPose 实时多人 2D | 4 (官方 release，但配 OpenCV/CMake 麻烦) | 4 | 3 | 3 | 14 |
| HRNet 单人 2D | 5 (MMPose 一行命令) | 3 | 3 | 3 | 14 |
| **HMR / HMR 2.0 单图 3D** | **4 (4D-Humans 仓库 PyTorch 易跑)** | **5 (3D 旋转网格非常震撼)** | **5 (跨越 2D→3D 直击 research question)** | **5 (展示 SMPL 参数化的力量)** | **19 ⭐** |
| Sapiens 多任务 | 3 (模型大，依赖多) | 5 | 4 | 3 | 15 |

### 🛠 HMR 复现路径（最小化阻力版本）

**优先选项 A: 4D-Humans (Goel 2023, HMR 2.0) — Colab 可跑**

```bash
# 仓库: https://github.com/shubham-goel/4D-Humans
# 论文: "Humans in 4D: Reconstructing and Tracking Humans with Transformers" ICCV 2023
# 这是 HMR 的现代继任者，Demo 比原版稳定，效果好得多

# 路径:
1. fork 到自己 GitHub（report 里要交代码 link）
2. 在 Colab 上 pip install + 下载 checkpoint（一键脚本）
3. 准备 5-10 张测试图片（YouTube 截图、Unsplash 人物）
4. 运行 demo.py 得到：
   - 原图 + 2D keypoints overlay
   - 旋转 3D mesh (.gif/.mp4)
   - 多视角 mesh 渲染
5. （加分）对比同一张图：
   - HRNet 输出的 2D keypoints
   - HMR 2.0 输出的 3D mesh
   - 验证 "2D→3D" pipeline 叙事
```

**备选 B: 经典 HMR (Kanazawa 2018, Pytorch 复刻)**

```bash
# 仓库: https://github.com/MandyMo/pytorch_HMR
# 优势: 直接对应论文，代码量小（讲解时容易讲清楚）
# 劣势: TF1 原版已废弃，PyTorch 版精度略低
```

**备选 C (保险方案): MMPose pre-trained HRNet + SMPLify 后处理**

```bash
# 如果 HMR 跑不通: MMPose 跑 HRNet 拿 2D → SMPLify 优化拿 3D
pip install mmpose mmdet
python demo/topdown_demo.py ... --output-2d
# 然后用 https://github.com/vchoutas/smplify-x 或 EasyMocap 优化 3D
# 这就成了一个真实的 "HRNet → SMPL" pipeline，正好讲两篇论文的连接
```

### ✅ Demo 验收标准（report 里要写）

- [ ] 至少 3 张测试图，每张展示：原图 / 2D keypoints / 3D mesh 三视图
- [ ] 一段 10-15s 视频（多张图 → 旋转 mesh 拼接）
- [ ] 一个 README 含运行命令、依赖、checkpoint 下载链接
- [ ] **关键**：在 report 里写明 demo 验证了哪个 paper claim
  （比如 "HMR claims weak supervision suffices — our demo confirms reasonable mesh from in-the-wild images"）

### ⚠️ 风险预案

| 风险 | 缓解 |
|------|------|
| GPU 不够 / Colab 断连 | 优先用 4D-Humans 的 quick-demo，CPU 也能跑 inference |
| SMPL 模型注册流程慢（需要邮箱注册）| 5/3 立即注册，4D-Humans 已自带兼容 checkpoint |
| Demo 跑出来效果差 | 故意挑选简单姿势（站立、单人、清晰背景）作为 fallback |

---

## (D) 三篇论文结合的创新点 / 可探索方向

> ⚠️ 注意：项目要求是**文献综述**，不是发新 paper。这些"创新点"是**写在 §5 Future Directions** 和**口头答辩**时的 talking points，不要求实施。

### 💡 4 个递进式 idea（按可行性排序）

#### Idea 1 ⭐⭐⭐⭐⭐ "HRNet → HMR" 模块替换（最可行，可作为 demo 加分）

- **逻辑**: HMR 原始用 ResNet-50 backbone 提特征。把 HRNet 高分辨率特征（已知比 ResNet 在 keypoint 上更准）替换进去，看 3D mesh 精度是否提升。
- **现实背景**: 这个思路其实**已经被 PyMAF (ICCV 2021) 实现** — 用 feature pyramid alignment 改进 HMR。可以引用 PyMAF 论证我们的想法在工业界已经被验证。
- **demo 加分**: 跑两遍 HMR（一次 ResNet backbone，一次 HRNet backbone），对比 PA-MPJPE。

#### Idea 2 ⭐⭐⭐⭐ "OpenPose 多人 → HMR 多人 3D" pipeline

- **逻辑**: 原版 HMR 假设"图里只有一个人"。用 OpenPose 的 multi-person bottom-up 检测每个人 → bbox crop → 各自跑 HMR → 在世界坐标系拼回去。
- **现实背景**: 已有 BEV (Sun 2022) / ROMP (Sun 2021) 做这件事。
- **可探索**: 处理人物之间的遮挡（A 挡住 B 时 HMR 会预测错误）— 用 OpenPose 的 occlusion-aware keypoints 给 HMR 加约束。

#### Idea 3 ⭐⭐⭐ "Sapiens 蒸馏" — 用基础模型当 teacher

- **逻辑**: Sapiens (2024 ECCV) 在 0.3B-2B 参数上预训练，是当前 human vision 的 SOTA。把 Sapiens 的 keypoint head 输出当 soft label，蒸馏到一个轻量 HRNet（适合手机部署）。
- **延伸**: Sapiens 同时输出 pose + segmentation + depth + normal — 可以多任务联合监督一个学生网络。
- **限制**: 需要算力 + 数据，超出 1 周项目范围，只作为 future work。

#### Idea 4 ⭐⭐⭐ "3D Gaussian Splatting + HMR" — 数字化身

- **逻辑**: HMR 给出 SMPL mesh（裸模型），3DGS 可以从多视角图像学到带衣服/头发的高保真渲染。把 HMR 的 SMPL 作为 3DGS 的几何 prior，训一个 animatable Gaussian avatar。
- **现实背景**: 已有 GaussianAvatar (2024)、Animatable 3D Gaussians (2024)。
- **价值**: 把课程话题（人体姿态）和当下最热的 3DGS 联系起来，老师听了会眼前一亮。

### 🎯 在 §5 怎么写（推荐措辞）

> "While the three covered papers establish the 2D→3D pipeline, several open challenges remain. **First**, end-to-end fusion of multi-person bottom-up detection (à la OpenPose) with single-person 3D regression (à la HMR) is still fragile under heavy occlusion (cf. ROMP, Sun 2021). **Second**, recent foundation models such as Sapiens (Khirodkar et al., 2024) suggest that scale + diverse 2D supervision can subsume both keypoint and 3D tasks, hinting at unified human representations. **Third**, integrating parametric body models with neural rendering primitives (3D Gaussian Splatting) opens a path to photorealistic, animatable avatars, an avenue we believe is the natural successor to the HMR line."

---

## (E) 时间表（5/2 → 5/11，按天分配）

### 📅 总览（9 天 daily breakdown）

| 日期 | 阶段 | 关键产物 | Yiqiao Liu | Taijia Liang |
|------|------|---------|------------|--------------|
| **5/2 (六)** | Day 0: 立项 | 本计划文件确认 | 阅读 OpenPose + HMR 论文（精读） | 阅读 HRNet 论文（精读）+ 项目要求复核 |
| **5/3 (日)** | Day 1: 论文消化 | 个人笔记 | 完成 OpenPose §3.1 ✓ + 起草 HMR §3.3 草稿 | 起草 HRNet §3.2 草稿 + SMPL/SMPLify 背景调研 |
| **5/4 (一)** | Day 2: 环境搭建 | Demo 环境就绪 | clone 4D-Humans，注册 SMPL 邮箱，跑通 quick-demo | clone MMPose，跑通 HRNet pretrained，准备 5 张测试图 |
| **5/5 (二)** | Day 3: Demo 跑通 | 第一版 demo 输出 | HMR 2.0 跑出 3 张测试图的 3D mesh | HRNet 跑出对应 2D keypoints 叠加图 |
| **5/6 (三)** | Day 4: Report 主体 | report 草稿 v1 | 写 §3.3 HMR 终稿 + §5 Future（含 4 个 idea）| 写 §3.2 HRNet 终稿 + §4 Connection 表格 |
| **5/7 (四)** | Day 5: Report 整合 | report 草稿 v2 | 整合所有 §3，校对引用 | 写 §1 Intro + §2 Background，refs.bib 补全 7 篇 |
| **5/8 (五)** | Day 6: Demo 加分 | demo 视频 + GitHub | 录制 HMR 3D 旋转 mp4，传 GitHub | 跑 "HRNet→HMR" 对比实验（idea 1），出对比图 |
| **5/9 (六)** | Day 7: PPT 制作 | slides v1 (12-15 张) | 做 slide 4-5 (OpenPose) + 8-9 (HMR) + 11-12 (demo+future) | 做 slide 1-3 (intro+roadmap) + 6-7 (HRNet) + 10 (compare) |
| **5/10 (日) ⏰** | Day 8: 提交 | **22:00 前所有交付物 final** | 录 demo 旁白草稿 + 最后排练 | LaTeX 编译验证 2 页 + 上传 brightspace + 提交 GitHub link |
| **5/11 (一) 17:00–21:00** | Day 9: 现场 | 演讲完成 | 主讲 OpenPose + HMR + Demo (5+ min) | 主讲 Intro + HRNet + Compare (5+ min) |

### ⚠️ 硬约束 checkpoint

- **5/3 EOD**: SMPL 邮箱注册必须发出（审核要 1-2 天）
- **5/5 EOD**: Demo 必须跑出至少 1 张可用结果（否则切到 plan B = MMPose+SMPLify）
- **5/7 EOD**: 2 页限制必须验证（多则砍 §2 背景或 §5 future）
- **5/9 EOD**: 至少做一次完整排练（10 min 时长 + 验证语速）
- **5/10 18:00**: 双方 cross-review 对方 slides，修拼写/格式

### 📋 README.md 要补的"Plan and Responsibility"

```markdown
## Plan and Responsibility

### Yiqiao Liu (Primary: OpenPose §3.1 ✓ + HMR §3.3 + §5 Future + Demo)
- §3.1 OpenPose section (✓ drafted)
- §3.3 HMR section (write 5/6, finalize 5/7)
- §5 Future Directions (write 5/6)
- Working model: 4D-Humans demo on 5 test images (deliver 5/8)
- Slides: 4-5 (OpenPose), 8-9 (HMR), 11-12 (demo + future)
- Live presentation: OpenPose + HMR + Demo walkthrough (~5 min)

### Taijia Liang (Primary: §1 Intro + §2 Background + HRNet §3.2 + §4 Compare)
- §1 Introduction (write 5/7)
- §2 Background incl. SMPL/SMPLify (write 5/7)
- §3.2 HRNet section (write 5/6, finalize 5/7)
- §4 Connections, Differences & Limitations (write 5/6)
- refs.bib maintenance (7 new entries by 5/7)
- HRNet 2D keypoint demo + comparison figures (deliver 5/5, 5/8)
- Slides: 1-3 (intro+roadmap), 6-7 (HRNet), 10 (compare table)
- LaTeX 2-page compile verification (5/8 and 5/10)
- Live presentation: Intro + HRNet + Comparison (~5 min)

### Joint
- 5/9: Joint slide rehearsal (full 10 min run-through)
- 5/10: Cross-review and final submission to brightspace
- 5/11: Live presentation at 60FA 110, 17:00-21:00
```

---

## 🚀 立即可执行的 5 个 first action（今天 5/2 晚 / 明天 5/3 上午）

1. **[Yiqiao]** 注册 SMPL 模型下载（https://smpl.is.tue.mpg.de/）— 提前 2 天审批
2. **[Taijia]** 把本计划末尾的 "Plan and Responsibility" 写进 README.md
3. **[共同]** 确认 LaTeX 模板：当前 main.tex 用的是 `IEEEtran` conference 双栏。**核对 PDF §6.5 要求**："Use standard LaTeX document class (`article`)"。**冲突！需要决定**：保留 IEEEtran（专业但与要求不一致）vs 切到 article（合规）。
4. **[Yiqiao]** 把 4D-Humans 的 README 通读一遍，确认依赖 + 数据准备步骤
5. **[Taijia]** 把 refs.bib 里 7 篇补充论文的 BibTeX 准备好（从 Google Scholar 一键复制）

### 🚨 Critical 待确认事项

> **LaTeX 模板冲突**: PDF §6.5 明确要求 `\documentclass{article}`，但 LiteratureReview/main.tex 用的是 `\documentclass[conference]{IEEEtran}`。
>
> **建议**: 切换到 `article` class，使用 `geometry` 包控制 margin（PDF §6.5 推荐），单栏 11-12pt，更接近 PDF 模板要求。如果坚持双栏（视觉更紧凑能塞更多内容），可在邮件里和老师确认是否接受 IEEEtran。

---

## 📊 风险矩阵

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| HMR demo 跑不通 | 中 | 高 | Plan B: MMPose HRNet + SMPLify 拼装 (D 节方案 C) |
| SMPL 注册延迟 | 中 | 中 | 4D-Humans checkpoint 已绑定 SMPL，不强制单独下载 |
| LaTeX 超过 2 页 | 高 | 高 | 5/8 早测，砍 §2 背景，把对比放图表 |
| 演讲超时 / 不足 5 min/人 | 中 | 高 | 5/9 必排练 + 计时；准备 30s 可砍内容 |
| 团队冲突 / 进度卡住 | 低 | 高 | 每 2 天一个 sync checkpoint（见时间表） |
| GitHub Code 提交 README 不全 | 中 | 低 | 5/8 同步写 README，含运行命令 |

---

## ✅ 交付物 Checklist（PDF §7 严格对照）

| 类型 | 文件名 | 截止 | 状态 |
|------|--------|------|------|
| Slides | `Group<Name>_Presentation.pdf` 或 `.pptx` | 5/10 23:59 (展示前一天) | ⏳ |
| LaTeX Source | `Group<Name>_Report.tex` | 5/10 23:59 | ⏳ |
| LaTeX PDF | `Group<Name>_Report.pdf` | 5/10 23:59 (≤2 页) | ⏳ |
| Code (bonus) | `Group<Name>_Code.zip` 或 GitHub URL（写在 report 里）| 5/10 23:59 | ⏳ |

> ⚠️ Group 名要在 5/3 之前定，并在 Excel 协调表登记（PDF §3.5）。

---

## 🔗 后续步骤

1. **审查本计划** — 修改/澄清任何不合理之处
2. **执行启动** — 满意后用 `/ccg:execute .claude/plan/cv-final-2d-to-3d.md` 进入实施阶段
3. **可选深度技术验证** — 如果对 HMR 复现路径不放心，可单独触发 `/ccg:analyze "评估 4D-Humans vs 经典 HMR 在 Mac M-series + RTX 6000 环境的可行性"` 调多模型分析

---

## 📝 SESSION_ID（供 /ccg:execute 使用）

- **CODEX_SESSION**: (本次未启动外部模型 — 学术规划任务直接基于已有上下文综合)
- **GEMINI_SESSION**: (本次未启动外部模型)
- **建议**: `/ccg:execute` 阶段如需写论文/排版/做 PPT 时再调用 gemini，跑代码 demo 时调用 codex
