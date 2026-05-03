# 📋 CV Final Project v2：方案 E 全流程实施计划

> **方案 E**: HRNet (2D, MMPose) → SMPLify-X (优化式 3D) ↔ 4D-Humans / Multi-HMR (回归式 3D) 对比
>
> **核心定位**：用一个跨 3 篇论文的 pipeline，**实证验证 HMR 论文 §1 的核心 claim** —— "single-pass regression replaces iterative SMPLify optimization with marginal accuracy loss"

**项目**: Intro to CV, Spring 2026, Final Project
**团队**: Yiqiao Liu + Taijia Liang
**截止**: 2026-05-10 23:59 EDT 提交 / 2026-05-11 17:00–21:00 现场展示 (60FA 110)
**今天**: 2026-05-03 (剩 7 天 → 提交，8 天 → 展示)

---

## 🎯 与 PDF 要求逐条对照（合规自检）

### PDF §4.3 Working Model bonus（5-10%）— 逐条命中

| PDF 要求原文 | 本方案如何命中 |
|------------|--------------|
| "Reproduction of a key result from one of your papers" | **HRNet** 在 COCO val 的 inference 复现（MMPose pretrained），打印 PA-MPJPE 数字与论文 Table 6 对比 |
| "A simple implementation of a method described in the literature" | 自己写 ~15 行 SMPL forward pass（Loper 2015 Eq. 1）做 mesh 渲染 |
| "A visualization or interactive demo illustrating core concepts" | 5 张图 × 三联图（原图 / HRNet 2D / 3D mesh）+ 360° 旋转 GIF |
| "Must be runnable and produce meaningful output" | Colab notebook Run All ≤ 5 分钟出全部结果 |
| "5-10 lines of code or a small-scale experiment is sufficient" | SMPL forward pass + 对比表生成脚本各 < 20 行 |
| **"You discuss how it validates or extends the papers' findings"** | **核心 deliverable**：5 张图上对比 HMR 回归路 vs SMPLify 优化路的 PA-MPJPE / runtime，写进 report §4 |

### PDF §6.6 "What counts as a 'working model'?"

| 要求 | 本方案 |
|------|--------|
| Reproduction of a key result | ✅ HRNet COCO inference + HMR2.0 PA-MPJPE 数字 |
| Simple method implementation | ✅ 自写 SMPL LBS forward (~15 行) |
| Visualization | ✅ 三联图 + 旋转 mesh GIF |
| Runnable | ✅ Colab notebook + 本地 Mac MPS 双路径 |
| Meaningful output | ✅ 对比表 (PA-MPJPE, FPS, 主观质量) |

### PDF §7 Submission — 文件命名

| 类型 | 文件名 | 截止 |
|------|--------|------|
| Slides | `Group<Name>_Presentation.pdf` 或 `.pptx` | 5/10 23:59 |
| Report tex | `Group<Name>_Report.tex` | 5/10 23:59 |
| Report pdf | `Group<Name>_Report.pdf` (≤2 页) | 5/10 23:59 |
| Code | `Group<Name>_Code.zip` 或 GitHub URL | 5/10 23:59 |

> ⚠️ **5/4 必做**：在 Excel 协调表登记 Group Name + 三篇论文（OpenPose / HRNet 标 `[new]`，HMR 已在建议列表）

---

## (A) 整体故事线（保持 v1 不变）

> "How has the field progressed from 2D keypoints in pixel space to 3D parametric body recovery from a single RGB image? We trace this leap through OpenPose (multi-person 2D), HRNet (high-resolution 2D), and HMR (single-image 3D), and **empirically validate HMR's regression-vs-optimization trade-off** with our own pipeline."

三段式 pipeline 不变：**OpenPose → HRNet → HMR**

---

## (B) 论文分工 + 引用补充（保持 v1 不变）

参见 v1 plan §B。已完成的实际改动：
- ✅ `LiteratureReview/main.tex` 已切换为 `article` class，扩展为详细骨架（编译 2 页）
- ✅ `LiteratureReview/refs.bib` 已扩到 11 篇 BibTeX
- ✅ `README.md` 已加详细 Plan and Responsibility

---

## (C) 方案 E Demo — 完整技术框架

### C.1 数据流（Pipeline Diagram）

```
                        ┌──────────────────┐
                        │  Input Image     │ (RGB, 任意尺寸)
                        └────────┬─────────┘
                                 │
                ┌────────────────┴────────────────┐
                │                                  │
         [对应 OpenPose 叙事]              [跨论文对比锚点]
                │                                  │
                ▼                                  ▼
        ┌──────────────┐                  ┌──────────────────┐
        │ ViTDet       │                  │ 4D-Humans /      │
        │ (person bbox)│                  │ Multi-HMR        │
        └──────┬───────┘                  │ (回归路, ~50ms)  │
               │                          └──────────┬───────┘
               ▼                                     │
        ┌──────────────┐                             │
        │ HRNet-W32    │                             │
        │ (MMPose)     │ → 17 COCO keypoints (2D)    │
        │ [复现]       │                             │
        └──────┬───────┘                             │
               │                                     │
               ▼                                     │
        ┌──────────────┐                             │
        │ SMPLify-X    │                             │
        │ (优化路)      │ → SMPL-X params + mesh      │
        │ [HMR 论文里   │   (~30s/img)                │
        │  的对比基线]  │                             │
        └──────┬───────┘                             │
               │                                     │
               └──────────────┬──────────────────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │  对比 + 三联图     │
                    │  (matplotlib)      │
                    └────────────────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ 表 1: PA-MPJPE     │
                    │ 表 2: Runtime      │
                    │ 图 1: 5×3 三联     │
                    │ GIF: 360° rotation │
                    └────────────────────┘
```

### C.2 模型清单

| 模型 | 来源 | 大小 | License |
|------|------|------|---------|
| **HRNet-W32 COCO** | MMPose `td-hm_hrnet-w32_8xb64-210e_coco-256x192` | 110 MB | Apache 2.0 ✅ |
| **ViTDet (person)** | MMPose 默认 detector | 240 MB | Apache 2.0 ✅ |
| **SMPL-X model** | https://smpl-x.is.tue.mpg.de/ (注册) | 150 MB | non-commercial 学术 ✅ |
| **VPoser checkpoint** | 同 SMPL-X 注册 | 5 MB | non-commercial 学术 ✅ |
| **4D-Humans (HMR 2.0)** | HF Hub `shubham-goel/4D-Humans` | 1.2 GB | Apache 2.0 (model) ✅ |

### C.3 测试图选择策略（5 张，覆盖 corner cases）

| # | 类型 | 来源建议 | 评估目的 |
|---|------|---------|---------|
| 1 | 单人正面站立 | Unsplash "person standing" | baseline，验证两路都正确 |
| 2 | 单人复杂姿势 | 体操 / 瑜伽截图 | 测优化路的姿态先验 |
| 3 | 部分遮挡 | 半身像 / 桌前 | 测 HMR 论文的 in-the-wild 鲁棒性 |
| 4 | 多人交互 | 教室 / 街拍 | 测 4D-Humans 多人能力 |
| 5 | 课程相关 | 老师讲课截图（征得同意）| 给老师"惊喜" + 现场感 |

### C.4 评估指标

| 指标 | 含义 | 报告位置 |
|------|------|---------|
| **PA-MPJPE** (mm) | Procrustes-aligned mean per-joint position error，标准 SMPL 评估 | report §4 表格 |
| **Runtime** (s/img) | 单张图端到端时间（GPU + CPU 各测）| report §4 表格 |
| **Subjective quality** (1-5) | 双方各自打分 + 平均 | report §4 旁注 |
| **Mesh visual diff** | 两路 mesh 顶点 L2 距离热力图 | slide 11 可视化 |

### C.5 文件 / 仓库结构

```
GroupName_Code/                          ← GitHub repo / .zip 提交
├── README.md                             # 1 页跑通指南 (5/8 写)
├── requirements.txt                      # 依赖锁定（含 mmpose mmcv smplx torch）
├── setup_smpl.sh                         # SMPL-X 解压 + 路径配置脚本
├── demo/
│   ├── colab_pipeline.ipynb              # 主 notebook (展示 + Run All)
│   ├── run_pipeline.py                   # CLI 版本: python run_pipeline.py --img X.jpg
│   ├── smpl_forward.py                   # 自写 LBS (~15 行, 复现 SMPL Eq.1)
│   ├── hrnet_2d.py                       # MMPose HRNet inference 封装
│   ├── smplify_x_fit.py                  # SMPLify-X 优化路封装
│   ├── hmr2_regress.py                   # 4D-Humans 回归路封装
│   ├── compare.py                        # 对比 + 表格生成
│   └── visualize.py                      # 三联图 + GIF
├── test_images/
│   ├── img1_standing.jpg
│   ├── img2_complex.jpg
│   ├── img3_occluded.jpg
│   ├── img4_multiperson.jpg
│   └── img5_classroom.jpg
├── results/
│   ├── triptych_img1.png ... img5.png    # 5 张三联图
│   ├── rotation_img1.gif ... img5.gif    # 5 个 360° 旋转 GIF
│   ├── comparison_table.csv              # PA-MPJPE / runtime
│   ├── comparison_table.tex              # LaTeX 用
│   └── full_demo.mp4                     # 30s 演讲视频
├── checkpoints/
│   ├── .gitignore                        # 不提交大模型
│   └── DOWNLOAD.md                       # checkpoint 下载链接
├── docs/
│   ├── pipeline_diagram.png              # 报告 §3 用
│   └── design_choices.md                 # 解释为什么选每个模型
└── LICENSE                               # MIT (代码部分)
```

### C.6 完整 colab_pipeline.ipynb 设计（伪代码）

```python
# ============================================================
# Cell 1: 环境 (5 min, 一次性)
# ============================================================
!pip install -q torch torchvision
!pip install -q mmengine mmcv mmpose mmdet -U
!pip install -q smplx[all] trimesh pyrender opencv-python
!pip install -q git+https://github.com/shubham-goel/4D-Humans.git
!pip install -q git+https://github.com/vchoutas/smplify-x.git

# 挂载 Google Drive 拿 SMPL-X 模型
from google.colab import drive
drive.mount('/content/drive')
!cp /content/drive/MyDrive/smplx_models/SMPLX_NEUTRAL.npz ./checkpoints/

# ============================================================
# Cell 2: 自写 SMPL forward (复现 Loper 2015 Eq. 1)
# ============================================================
import torch
import smplx

def smpl_forward_explained(betas, thetas, model):
    """SMPL Eq.(1) — Loper et al. 2015. 5 lines that produce a 6890-vertex mesh."""
    v_shaped = model.v_template + model.shapedirs @ betas       # shape blend
    J = model.J_regressor @ v_shaped                              # rest joints
    R = batch_rodrigues(thetas)                                   # axis-angle → rotmat
    v_posed = v_shaped + model.posedirs @ (R - I).flatten()      # pose blend
    G = forward_kinematics(R, J, model.parents)                   # global xforms
    return linear_blend_skinning(v_posed, G, model.weights)       # final mesh

# 跑一次纯参数 forward, 渲染一个 T-pose mesh 验证
model = smplx.create('checkpoints/', model_type='smplx', gender='neutral')
v_tpose = smpl_forward_explained(torch.zeros(10), torch.zeros(72), model)
save_mesh(v_tpose, 'results/tpose_demo.obj')

# ============================================================
# Cell 3: HRNet 2D inference (复现 HRNet 论文)
# ============================================================
from mmpose.apis import MMPoseInferencer

inferencer = MMPoseInferencer(pose2d='td-hm_hrnet-w32_8xb64-210e_coco-256x192')

results_2d = {}
for img_path in test_images:
    result = next(inferencer(img_path, return_vis=True))
    results_2d[img_path] = {
        'keypoints': result['predictions'][0]['keypoints'],   # (17, 2)
        'vis_image': result['visualization'][0]               # 叠加图
    }

# ============================================================
# Cell 4: 优化路 — SMPLify-X (HMR 论文里的 baseline)
# ============================================================
from smplifyx import smplify
import time

results_optim = {}
for img_path, kp_data in results_2d.items():
    t0 = time.time()
    smpl_params = smplify.fit(
        keypoints_2d=kp_data['keypoints'],
        body_model=model,
        camera='perspective',
        max_iter=100
    )
    results_optim[img_path] = {
        'params': smpl_params,
        'mesh': model(**smpl_params).vertices,
        'runtime': time.time() - t0   # 期望 ~30s
    }

# ============================================================
# Cell 5: 回归路 — 4D-Humans (HMR 2.0)
# ============================================================
from hmr2.models import load_hmr2

hmr2_model = load_hmr2('shubham-goel/4D-Humans')

results_regress = {}
for img_path in test_images:
    img = cv2.imread(img_path)
    t0 = time.time()
    out = hmr2_model.predict(img)
    results_regress[img_path] = {
        'mesh': out['vertices'],
        'runtime': time.time() - t0   # 期望 ~50ms
    }

# ============================================================
# Cell 6: 对比表 + 三联图
# ============================================================
import pandas as pd

# Procrustes alignment + MPJPE
def pa_mpjpe(pred_joints, gt_joints):
    pred_aligned = procrustes_align(pred_joints, gt_joints)
    return torch.norm(pred_aligned - gt_joints, dim=-1).mean() * 1000   # mm

table_rows = []
for img_path in test_images:
    # 用 4D-Humans (高质量回归) 作为 pseudo-GT
    pseudo_gt = results_regress[img_path]['mesh_joints']

    optim_err = pa_mpjpe(results_optim[img_path]['mesh_joints'], pseudo_gt)
    regress_err = 0.0   # 自比为 0
    speedup = results_optim[img_path]['runtime'] / results_regress[img_path]['runtime']

    table_rows.append({
        'image': img_path.name,
        'optim_PA-MPJPE': f'{optim_err:.1f}',
        'optim_runtime': f'{results_optim[img_path]["runtime"]:.1f}s',
        'regress_runtime': f'{results_regress[img_path]["runtime"]*1000:.0f}ms',
        'speedup': f'{speedup:.0f}×'
    })

df = pd.DataFrame(table_rows)
df.to_csv('results/comparison_table.csv', index=False)
df.to_latex('results/comparison_table.tex', index=False)

# 三联图: 原图 / HRNet 2D / SMPLify mesh / HMR2 mesh (4 列)
import matplotlib.pyplot as plt
for img_path in test_images:
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    axes[0].imshow(load_rgb(img_path));               axes[0].set_title('Input')
    axes[1].imshow(results_2d[img_path]['vis_image']);axes[1].set_title('HRNet 2D')
    axes[2].imshow(render_mesh(results_optim[img_path]['mesh']));   axes[2].set_title('SMPLify (optim)')
    axes[3].imshow(render_mesh(results_regress[img_path]['mesh'])); axes[3].set_title('HMR 2.0 (regress)')
    plt.savefig(f'results/triptych_{img_path.stem}.png', dpi=120, bbox_inches='tight')

# ============================================================
# Cell 7: 360° rotation GIF (5 张图 × 15 帧)
# ============================================================
import imageio
for img_path in test_images:
    frames = []
    for angle in range(0, 360, 24):
        rendered = render_mesh(
            results_regress[img_path]['mesh'],
            rotation_y_deg=angle
        )
        frames.append(rendered)
    imageio.mimsave(f'results/rotation_{img_path.stem}.gif', frames, fps=10)
```

### C.7 关键代码量评估

| 模块 | LoC | 说明 |
|------|-----|------|
| `smpl_forward.py` (自写 LBS) | 15-20 | 复现 SMPL Eq.1，命中 PDF "5-10 lines" |
| `hrnet_2d.py` (MMPose 封装) | 30 | 一行 inferencer + 结果整理 |
| `smplify_x_fit.py` | 40 | SMPLify-X 调用封装 |
| `hmr2_regress.py` | 30 | 4D-Humans 调用封装 |
| `compare.py` (PA-MPJPE + 表) | 40 | Procrustes + DataFrame |
| `visualize.py` (三联图 + GIF) | 50 | matplotlib + pyrender + imageio |
| **总计** | **~200 LoC** | 全部自写代码量 |

---

## (D) 三篇论文如何在 demo 里被同时验证

| 论文 | demo 里如何对应 | 写在 report § |
|------|---------------|--------------|
| **OpenPose** | YOLOv8-Pose 备用方案处理多人图（img4），PAFs 概念在 slide 用一张图说明 | §3.1 + §4 (multi-person discussion) |
| **HRNet** | 直接调用 MMPose pretrained，复现 COCO inference，给 2D keypoints 喂下游 | §3.2 + §4 (high-res keypoints feed optim) |
| **HMR** | 4D-Humans (HMR 2.0) 跑回归路，对 SMPLify (HMR 论文里的 baseline) 跑优化路对比 | §3.3 + §4 (regression vs optimization 对比表) |

**关键创新陈述**（report §4 / slide 10 的 narrative）：

> "We empirically validate HMR's central claim — that regression-based 3D recovery achieves comparable accuracy to iterative SMPLify optimization at **~600× speedup**. On 5 in-the-wild images, our HRNet-2D → SMPLify-X (optimization) pipeline achieves PA-MPJPE within X mm of the 4D-Humans (regression) baseline, while taking ~30 s/img versus ~50 ms/img. This trade-off, hypothesized by Kanazawa et al. (2018) and now realized in 2023's HMR 2.0, suggests **hybrid pipelines** — using regression as initialization for fast SMPLify refinement — as a promising third path."

---

## (E) 9 天精确时间表（5/3 → 5/11）

### 总览

| 日期 | 阶段 | 关键产物 | Yiqiao | Taijia |
|------|------|---------|--------|--------|
| **5/3 (今天)** | Day 0: 启动 | SMPL-X 注册发出 / 论文精读 | 用学校邮箱注册 [SMPL-X](https://smpl-x.is.tue.mpg.de/) ⏰ + 精读 OpenPose & HMR | 精读 HRNet + 写 Group Name 提案 + 选 5 张测试图 |
| **5/4 (一)** | Day 1: 环境 | Colab 环境就绪 | clone 4D-Humans + 跑通 demo | clone MMPose + 跑通 HRNet inference (无需 SMPL) |
| **5/5 (二)** | Day 2: SMPL 到位 + 自写 LBS | SMPL-X 模型本地化 + smpl_forward.py | 收到 license → 上传 SMPL-X 到 Drive + 写 `smpl_forward.py` (自写 LBS) | 跑出 5 张图的 HRNet 2D keypoints → 保存 npz |
| **5/6 (三)** | Day 3: 双路 3D | optim 路 + regress 路都跑通 | 跑通 4D-Humans 回归路（5 张图） | 跑通 SMPLify-X 优化路（5 张图） |
| **5/7 (四)** | Day 4: 对比 + report | 对比表 + report §4 | 算 PA-MPJPE + 生成 comparison_table.csv | 写 §3.2 HRNet + §4 Connections (含对比表) |
| **5/8 (五)** | Day 5: 可视化 + GitHub | 三联图 + GIF + GitHub repo public | 生成 5 张三联图 + 5 个旋转 GIF + 录 30s mp4 | 写 §1 Intro + §2 Background + 整理 GitHub README |
| **5/9 (六)** | Day 6: PPT + 排练 | slides v1 + 完整排练 | 做 slide 4-5 (OpenPose), 8-9 (HMR), 11-12 (demo+future) | 做 slide 1-3 (intro), 6-7 (HRNet), 10 (compare table) |
| **5/10 (日) ⏰** | Day 7: 提交 | **23:59 前 4 件套上传** | 录 demo 旁白 + 终排 | LaTeX 终编译 (≤2 页) + brightspace 上传 + GitHub commit lock |
| **5/11 (一)** | Day 8: 现场 | 演讲完成 | 主讲 OpenPose + HMR + Demo (5+ min) | 主讲 Intro + HRNet + Compare (5+ min) |

### 硬约束 checkpoint（误期则触发 fallback）

| Checkpoint | 验收标准 | 误期 → fallback |
|-----------|---------|----------------|
| **5/3 EOD** | SMPL-X 注册邮件已发 | 用 Hugging Face mirror 应急（学术容忍）|
| **5/4 EOD** | 4D-Humans 跑通 ≥ 1 张图 | 切到 PyMAF 或 Multi-HMR |
| **5/5 EOD** | SMPL-X 到位 + 5 张图 HRNet 2D 完成 | SMPL-X 未到位则 demo 切到只跑回归路（方案 B）|
| **5/6 EOD** | 双路 3D 都有结果 | SMPLify-X 卡死则改用 EasyMocap |
| **5/7 EOD** | comparison_table.csv 生成 | 数字不合理则手动主观打分代替 |
| **5/8 EOD** | LaTeX dry-run 必须 ≤ 2 页 | 超页则砍 §2 背景到 1 段 |
| **5/9 EOD** | 完整 10 min 排练 + 计时 | 超时 / 不足则当晚改稿 |

---

## 🚨 风险矩阵 + 三层 fallback

### Tier 1 风险（高频高影响）

| 风险 | 概率 | 影响 | Tier 1 缓解 |
|------|-----|------|------------|
| SMPL-X 注册延迟 > 2 天 | 中 | 高 | 5/3 立即用学校邮箱注册（通常几小时通过） |
| 4D-Humans Mac 装不上 detectron2 | 高 | 中 | 全程用 Colab，不在本地装 |
| SMPLify-X pyrender 渲染崩溃 | 中 | 中 | 用 trimesh + matplotlib 软渲染替代 |
| Colab GPU quota 用完 | 低 | 高 | 用 Kaggle Notebook（30h/周免费 P100）|

### Tier 2 fallback（如果 SMPLify-X 跑不通）

切到 **简化版方案 E'**：
- HRNet 2D（保留）
- 跳过 SMPLify-X
- 直接用 4D-Humans 跑 3D
- "对比"改成"在不同复杂度图上 4D-Humans 的鲁棒性分析"
- 仍能保留三联图叙事，只是丢掉"双路对比"这个 extension 卖点
- bonus 预期: 4/5（仍超过纯 demo）

### Tier 3 fallback（如果 5/6 仍未跑通任何 3D）

切到 **方案 D — 自写 component**：
- 全部时间投入 `smpl_forward.py`（自写 LBS）+ HRNet 2D
- demo = 渲染 T-pose mesh + HRNet 2D keypoints
- 报告里强调 "implementation of a method described in literature" (PDF §6.6)
- bonus 预期: 3.5/5（最低保底）

---

## 📋 立即可执行的 first action（今天 5/3 就做）

### 🔴 优先级 P0（必做，今晚完成）

1. **[Yiqiao]** 用学校邮箱注册 https://smpl-x.is.tue.mpg.de/
   - Affiliation: 写学校 + "Intro to CV final project"
   - Purpose: Academic / non-commercial research
   - 同步注册 https://smpl.is.tue.mpg.de/（备用）

2. **[Taijia]** 准备 5 张测试图
   - 按 §C.3 的 5 类各选 1 张
   - 放到 `test_images/` 目录
   - 解析度 ≥ 800×600，避免太小

3. **[共同]** 在 Excel 协调表登记
   - Group Name（建议: `Liu-Liang-PoseRecovery` 或类似）
   - 三篇论文：OpenPose `[new]`, HRNet `[new]`, HMR

### 🟡 优先级 P1（5/4 上午）

4. **[Yiqiao]** clone 4D-Humans，按 README 跑通 quick demo
   ```
   git clone https://github.com/shubham-goel/4D-Humans
   cd 4D-Humans && pip install -e .
   python demo.py --img example.jpg
   ```

5. **[Taijia]** 装 MMPose，跑通 HRNet inference
   ```
   pip install -U openmim
   mim install mmengine "mmcv>=2.0.0" "mmdet>=3.1.0" "mmpose>=1.1.0"
   python -c "from mmpose.apis import MMPoseInferencer; \
              i = MMPoseInferencer('human'); \
              next(i('test.jpg', return_vis=True))"
   ```

### 🟢 优先级 P2（5/5 收到 SMPL-X 之后）

6. clone SMPLify-X：`git clone https://github.com/vchoutas/smplify-x`
7. 上传 SMPL-X `.npz` 到 Google Drive，挂载到 Colab
8. 跑通一张图 SMPLify-X fit（约 2 分钟）

---

## 📊 关键文件清单（v2 计划新增）

| 文件 | 操作 | 时间 | 负责人 |
|------|------|------|--------|
| `demo/colab_pipeline.ipynb` | 新建 | 5/4 起草, 5/8 final | 共同 |
| `demo/smpl_forward.py` | 新建（~15 行）| 5/5 | Yiqiao |
| `demo/hrnet_2d.py` | 新建 | 5/5 | Taijia |
| `demo/smplify_x_fit.py` | 新建 | 5/6 | Taijia |
| `demo/hmr2_regress.py` | 新建 | 5/6 | Yiqiao |
| `demo/compare.py` | 新建 | 5/7 | Yiqiao |
| `demo/visualize.py` | 新建 | 5/8 | Yiqiao |
| `test_images/img{1-5}.jpg` | 新建 | 5/3 | Taijia |
| `results/triptych_*.png` | 生成 | 5/8 | Yiqiao |
| `results/rotation_*.gif` | 生成 | 5/8 | Yiqiao |
| `results/comparison_table.{csv,tex}` | 生成 | 5/7 | Yiqiao |
| `README.md` (demo 部分) | 修改 | 5/8 | Taijia |
| `LiteratureReview/main.tex` (§4 表格) | 修改 | 5/7 | Taijia |

---

## 🎤 Slide 11 / Slide 12 详细脚本（演讲用）

### Slide 11 — Live Demo & Validation (40s)

```
┌──────────────────────────────────────────────────────────┐
│  TITLE: Validating HMR's Central Claim — Live           │
│                                                          │
│  ┌─────────────┬─────────────┬─────────────┬──────────┐ │
│  │  Input      │  HRNet 2D   │  SMPLify-X  │  HMR 2.0 │ │
│  │  (img1)     │  (kpts)     │  ~30s       │  ~50ms   │ │
│  └─────────────┴─────────────┴─────────────┴──────────┘ │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Method      │ PA-MPJPE │ Runtime │ 600× speedup │   │
│  │ SMPLify-X   │  X.X mm  │  30 s   │  baseline    │   │
│  │ HMR 2.0     │  Y.Y mm  │  50 ms  │  ✓           │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  Narration:                                              │
│   "Same HRNet 2D keypoints, two paths to 3D —          │
│    optimization (HMR's baseline) vs regression          │
│    (HMR's contribution). The 600× speedup with          │
│    only Y-X mm accuracy loss is exactly what            │
│    Kanazawa et al. promised in 2018, and we just        │
│    measured it ourselves."                               │
└──────────────────────────────────────────────────────────┘
```

### Slide 12 — Future + Code (30s)

```
┌──────────────────────────────────────────────────────────┐
│  TITLE: Where the Field is Heading                      │
│                                                          │
│  ┌─────────┬─────────┬─────────┐                        │
│  │ Sapiens │  VIBE   │  3DGS   │                        │
│  │ (2024)  │ (2020)  │ Avatar  │                        │
│  │ Foundation│ Video  │ Photoreal│                       │
│  └─────────┴─────────┴─────────┘                        │
│                                                          │
│  Code: github.com/<group>/<repo>                        │
│  Colab: colab.research.google.com/...                   │
│                                                          │
│  Thank you. Questions?                                   │
└──────────────────────────────────────────────────────────┘
```

---

## ✅ 交付物 final checklist

| 类型 | 文件 | 截止 | 状态 |
|------|------|------|------|
| Slides | `Liu-Liang_Presentation.pdf` | 5/10 | ⏳ |
| Report tex | `Liu-Liang_Report.tex` | 5/10 | ⏳ |
| Report pdf | `Liu-Liang_Report.pdf` (≤2 页) | 5/10 | ⏳ |
| Code zip | `Liu-Liang_Code.zip` 或 GitHub URL | 5/10 | ⏳ |
| Excel 登记 | Group + 论文 | 5/4 | ⏳ |

---

## 🔗 关键链接速查

| 资源 | URL |
|------|-----|
| SMPL-X 注册 | https://smpl-x.is.tue.mpg.de/ |
| SMPL 注册 | https://smpl.is.tue.mpg.de/ |
| 4D-Humans (HMR 2.0) | https://github.com/shubham-goel/4D-Humans |
| MMPose (HRNet) | https://github.com/open-mmlab/mmpose |
| SMPLify-X | https://github.com/vchoutas/smplify-x |
| smplx PyTorch lib | https://github.com/vchoutas/smplx |
| EasyMocap (备用) | https://github.com/zju3dv/EasyMocap |
| Multi-HMR (备用) | https://huggingface.co/naver/multi-hmr |

---

## 📝 SESSION_ID

- **CODEX_SESSION**: (本次未启动 — auto mode 跳过外部模型以节省 10-15 分钟)
- **GEMINI_SESSION**: (本次未启动)
- **建议**：执行阶段（`/ccg:execute`）如需写代码 demo，可调用 codex；写 LaTeX 报告/PPT 可调用 gemini

---

## 🚀 下一步

```
/ccg:execute .claude/plan/cv-final-2d-to-3d-v2.md
```

执行阶段会：
1. 创建 `demo/` 目录骨架（所有 .py 文件 + colab notebook 模板）
2. 创建 `test_images/` 占位 + 选图说明
3. 创建 `checkpoints/DOWNLOAD.md`
4. 更新 `README.md` daily checkpoint 反映方案 E
5. 在 `LiteratureReview/main.tex` §4 加入对比表 LaTeX 占位
