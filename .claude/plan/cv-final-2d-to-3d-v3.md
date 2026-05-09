# 📋 CV Final Project v3：方案 E（修复版）

> **方案 E (修复后)**：HRNet 2D (MMPose) + 4D-Humans 3D (HMR 2.0) 主线，SMPLify-X 优化路对比作为 **stretch goal**
>
> **核心定位**：用一个**跨论文 pipeline 串 3 篇论文叙事 + 一个 runtime/agreement 实验** 直接呼应 HMR 论文 §1 关于 regression-vs-optimization 的论述
>
> **v2→v3 关键修复**: 评估方法学（去循环逻辑）+ 演讲交接（一次切换）+ 现场 Live → 预录 mp4 + Multi env + License 合规

**项目**: Intro to CV, Spring 2026, Final Project
**团队**: Yiqiao Liu + Taijia Liang
**截止**: 2026-05-10 23:59 EDT 提交 / 2026-05-11 17:00–21:00 现场展示 (60FA 110)
**今天**: 2026-05-03 (剩 7 天 → 提交，8 天 → 展示)
**v2 审查记录**: `.claude/plan/cv-final-2d-to-3d-v2.md` + 主 conversation 综合 review 报告

---

## 🟢 实施进度（2026-05-03 EOD）

### Mac 本地（已落地，零 token 等待 Colab）
- ✅ Worktree venv: `/.../funny-shannon-ed770e/.venv` (torch 2.11 + MPS 可用)
- ✅ SMPL/SMPL-X/VPoser 模型已就绪 (`/Users/taijialiang/Documents/Intro to CV/CVFinal/model/`)
- ✅ symlink 标准布局（`bash demo/scripts/setup_smpl_paths.sh`）
- ✅ Mac smoke test 通过（SMPL-X T-pose forward + matplotlib 渲染）

### Demo 项目骨架（已落地）
```
demo/
├── README.md                            ✅ 跑通指南 + 验证矩阵
├── .gitignore                           ✅
├── checkpoints/{.gitignore, DOWNLOAD.md} ✅ License 合规
├── docs/pipeline_diagram.png            ✅ 报告 §3 用
├── envs/{requirements_local_mac.txt, env_hrnet_hmr2.yml, env_smplifyx.yml} ✅
├── notebooks/                           ✅ 4 个 notebook
│   ├── 00_mac_smoke_test.ipynb          (本地验证, < 10s)
│   ├── 01_main_pipeline.ipynb           (★ Colab 主 demo)
│   ├── 02_smplifyx_stretch.ipynb        (stretch goal)
│   └── 03_smpl_explain.ipynb            (SMPL Eq.(1) 教学)
├── results/                             ✅ 已生成 4 个产物
│   ├── smplx_tpose.obj  (715 KB)
│   ├── tpose_smplx.obj  (715 KB)
│   ├── smplx_4poses.png (70 KB) ── 报告/PPT 可直接用
│   ├── tpose_rotation.gif (71 KB) ── slide 11 备用 GIF
│   └── smplx_tpose_mpl.png (15 KB)
├── scripts/{setup_smpl_paths.sh, run_smoke_test.sh} ✅
├── src/                                 ✅ 6 个模块（~520 LoC 实际可执行代码）
│   ├── __init__.py            (MODEL_ROOT 路径常量)
│   ├── smpl_forward.py        (SMPL Eq.(1) 注释版包装)
│   ├── visualize.py           (matplotlib 3D + quad-plot + GIF + multi-person guard)
│   ├── hrnet_2d.py            (MMPose 包装)
│   ├── hmr2_demo_wrapper.py   (4D-Humans 包装, Colab only)
│   ├── compare.py             (Procrustes + agreement table)
│   └── smplifyx_cli.py        (BODY_25 转换 + CLI 调用)
└── test_images/README.md                ✅ 5 张 CC0 图选取规范
```

### 已经"提前完成"的 v3 任务
| v3 原排期 | 内容 | 实际完成 |
|----------|------|---------|
| 5/3 P0 | 注册 SMPL/SMPL-X | ✅ 模型已下载（节省 1-2 天 license 等待） |
| 5/4 P1 | clone + 装环境 | 🟡 Mac venv 已装 / Colab 待 5/4 |
| 5/5 主路 | 自写 LBS smpl_forward | ✅ 已写 + 验证可跑 |
| 5/6 visualize | quad-plot + GIF 函数 | ✅ 已写 + 已测 |
| 5/7 compare | runtime + agreement 表 | ✅ 已写（待 Colab 真数据填入） |

### 5/3 EOD 剩余 P0
- [ ] Yiqiao: 精读 OpenPose + HMR
- [ ] Taijia: 精读 HRNet + 选 5 张 CC0 图（按 `demo/test_images/README.md`）
- [ ] 共同: Group Name `Liu-Liang-PoseRecovery` 登记 Excel

### 5/4 起新关键路径（基于已有骨架）
- 上传 `model/smpl/SMPL_NEUTRAL.pkl` + `model/smplx/SMPLX_NEUTRAL.npz` 到 Google Drive
- Colab 跑 `notebooks/01_main_pipeline.ipynb` Cell 1 → 验证 4D-Humans 安装 + ckpt 下载
- 跑 Cell 2-5 → 出 5 张 quad-plot + GIF + runtime 表

---

## 🔁 Round-2 Review 集成（2026-05-03）

**gemini PASS (94/100)** — 全部 v2 critical 已修复
**codex NEEDS_IMPROVEMENT (67/100)** — 6 项技术细节 inline patched，详见下方 ★ 标记：

| codex round-2 发现 | v3 inline patch 位置 |
|------------------|--------------------|
| `pip install -e .` 应改 `[all]` | §C.6 Cell 1 ★ |
| `fetch_demo_data.sh` 不存在 | §C.6 Cell 1 改用 `download_models()` ★ |
| Checkpoint placeholder 路径 | §C.6 Cell 3 改用 `DEFAULT_CHECKPOINT` ★ |
| SMPL 文件名 `SMPL_NEUTRAL.pkl` 错误 | §C.2 + §C.6 改 `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` ★ |
| `model.faces` scope NameError | §C.8 重写为函数封装 ★ |
| HRNet "AP 复现" 过度声明 | §C.4 改 "mini sanity-check subset" ★ |
| Empty detection guard 缺失 | §C.6 Cell 3 加 if-len check ★ |
| T4 OOM (batch_size=8) | §C.6 Cell 3 改 batch_size=1 ★ |
| 多人图 verts[0] 选错人 | §C.6 Cell 4 加 `pick_center_person()` ★ |
| pyrender Colab headless crash | §C.6 Cell 1 加 osmesa + PYOPENGL_PLATFORM ★ |
| Slide ownership 表矛盾 | §F + §E 修正"谁讲谁做" ★ |
| Slide 11 narration 优化 | §F 用 gemini 重写 ★ |
| §4 OpenPose 段优化 | §D 用 gemini 重写 ★ |

---

## 🔁 v2 → v3 修复 diff 速览

| Critical 问题 | v2 状态 | v3 修复 |
|--------------|--------|---------|
| C1 PA-MPJPE 循环逻辑 | 用 4D-Humans 当 GT 算 "accuracy" | 改为 **agreement metric**，accuracy 引用论文 H3.6M 数字 |
| C2 SMPLify-X API 错 | 假装 `smplifyx.fit()` 存在 | 降级为 stretch goal + CLI 包装 + 5/5 硬截止 |
| C3 4D-Humans API 错 | `load_hmr2().predict()` | 严格按官方 demo.py 重写 |
| C4 SMPL/SMPL-X 模型缺 | 只列 SMPL-X | **同时注册 SMPL + SMPL-X** + 路径分别说明 |
| C5 演讲交接 ping-pong | T→Y→T→Y→T→Y 6 次 | **只切 1 次**：T 前 5 min / Y 后 5 min |
| C6 LaTeX 必超页 | figure1 + 5 章节硬塞 | 砍 figure1 + §2 压 1 段 + 5/8 dry-run 硬卡 |
| C7 现场 Live demo | "Run All" 在投影上跑 | **预录 30s mp4 嵌入 PPT** |
| C8 License + 灰色 mirror | "HF mirror SMPL-X" + Apache 标错 | MIT 修正 + 仅官方注册 + checkpoint 不进 zip |
| M1-7 (技术细节) | 见上一轮报告 | 全部修复（详见下方） |

---

## 🎯 与 PDF 要求逐条对照（v3 合规自检）

### PDF §4.3 Working Model bonus 5 条 — v3 命中策略

| PDF 要求 | v3 命中方案 |
|---------|------------|
| Reproduction of a key result | HRNet 在 COCO 单图 inference + AP/OKS 数字与论文 Table 6 对比（**改用 keypoint AP，不是 PA-MPJPE**） |
| Simple method implementation | `smpl_forward.py` 调用 `smplx.lbs.lbs(...)` 加注释逐行对应 SMPL Eq.(1) 渲染 T-pose mesh |
| Visualization | 5 张图 × **quad-plot**（原图 / HRNet 2D / HMR2 mesh 正面 / HMR2 mesh 侧面） + 5 个 360° GIF |
| Runnable + meaningful output | Colab notebook (`env_hrnet_hmr2.ipynb`) 顺序 Run All ≤ 8 分钟 |
| **Validates / extends papers' findings** | runtime 表 + agreement metric (**不称 accuracy**) + 论文 H3.6M 数字 → 在 §4 论述 HMR 600× speedup claim |

### PDF §6.6 working model 5 条 — 全部对应

| 要求 | v3 |
|------|----|
| Reproduction | ✅ HRNet COCO AP inference + 4D-Humans 论文报道 PA-MPJPE 引用 |
| Method implementation | ✅ smpl_forward.py 调用 smplx.lbs |
| Visualization | ✅ quad-plot + GIF |
| Runnable | ✅ Colab + 命令行 `run_pipeline.py` |
| Discusses validation/extension | ✅ §4 narrative + agreement table |

### PDF §7 提交 4 件套

| 文件 | 截止 |
|------|------|
| `<GroupName>_Presentation.pdf` | 5/10 23:59 EDT |
| `<GroupName>_Report.tex` | 5/10 23:59 |
| `<GroupName>_Report.pdf` (≤2 页严格) | 5/10 23:59 |
| `<GroupName>_Code.zip` 或 GitHub URL（**不含 checkpoints**）| 5/10 23:59 |

> ⚠️ **5/4 Excel 登记**: Group Name 建议 `Liu-Liang-PoseRecovery`（更具描述性，避免重复）

---

## (A) 整体故事线（v2 不变）

> **RQ**: How has the field progressed from 2D keypoints to 3D parametric body recovery from a single RGB image — and what design choices in representation, supervision, and architecture enabled this leap?

三段式: **OpenPose (multi-person 2D) → HRNet (high-res 2D) → HMR (single-image 3D)**

---

## (B) 论文分工 + 引用（v2 已落地）

参见 v1/v2 §B 与已有改动。已完成：
- ✅ `LiteratureReview/main.tex` (article class, 编译 2 页)
- ✅ `LiteratureReview/refs.bib` (11 BibTeX 条目)
- ✅ `README.md` Plan and Responsibility

**v3 新增 main.tex 编辑**（5/7 落地）：
- §1 → Taijia 写 (~5 句, 含 RQ + roadmap)
- §2 Background → **压到 1 段**（仅 SMPL + 一句 hourglass 谱系）
- §3.1 OpenPose → Yiqiao 已有初稿
- §3.2 HRNet → Taijia 写
- §3.3 HMR → Yiqiao 写
- §4 Comparison → **加入 quad-plot ref + agreement table + OpenPose 脱节解释段**
- §5 Future → Yiqiao 写

---

## (C) 方案 E v3 — 完整可执行框架

### C.1 数据流（修复版）

```
                         ┌──────────────────┐
                         │  Input Image     │
                         └────────┬─────────┘
                                  │
         ┌─────────────── 主路 (env_hrnet_hmr2) ───────────┐
         │                                                  │
         ▼                                                  ▼
   ┌─────────────┐                              ┌──────────────────┐
   │ MMPose      │                              │ 4D-Humans demo.py│
   │ HRNet-W48   │ → 17 COCO kpts (npz)         │ ViTDet+HMR2-ViT-L│
   │ + ViTDet    │                              │ → SMPL params    │
   │ [复现 AP]   │                              │ + 6890 mesh      │
   └─────────────┘                              └──────────────────┘
                                                          │
                                                          ▼
                                                  ┌─────────────┐
                                                  │ render mesh │
                                                  │ pyrender    │
                                                  └─────────────┘
                                                          │
         ┌─────────── stretch (env_smplifyx, 5/5 EOD 硬截止) ──┐
         │                                                       │
         ▼                                                       ▼
   ┌──────────────┐                                ┌─────────────┐
   │ SMPLify-X    │                                │ same render │
   │ CLI:         │ → SMPL-X params + 10475 mesh   │ + agreement │
   │ python -m    │                                │ joint-MPJPE │
   │ smplifyx.main│                                └─────────────┘
   └──────────────┘
                                  │
                                  ▼
                         ┌────────────────────┐
                         │  Reports / Outputs │
                         ├────────────────────┤
                         │ runtime_table.csv  │
                         │ agreement_table.csv│ (only if stretch done)
                         │ quad_plot_*.png    │
                         │ rotation_*.gif     │
                         │ tpose_demo.obj     │
                         └────────────────────┘
```

### C.2 模型 + 注册 + License（修复版）

| 资产 | 来源 | 大小 | License | 提交合规 |
|------|------|------|---------|---------|
| HRNet-W48 + ViTDet | MMPose `td-hm_hrnet-w48_8xb32-210e_coco-256x192` | ~250 MB | **Apache 2.0** ✅ | 不打包，下载脚本 |
| 4D-Humans (HMR 2.0) ckpt | https://github.com/shubham-goel/4D-Humans (`hmr2.utils.download_models()` + `DEFAULT_CHECKPOINT`) | 1.2 GB | **MIT (代码)**, MPI license (权重) ⚠️ | 不打包，运行时下载 |
| SMPL `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` | https://smpl.is.tue.mpg.de/ 注册 | 40 MB | **non-commercial 学术** | 不打包；**保留官方文件名**，4D-Humans `data/` 目录 expects 此精确名 |
| SMPL-X `SMPLX_NEUTRAL.npz` | https://smpl-x.is.tue.mpg.de/ 注册 (only if stretch) | 150 MB | **non-commercial 学术** | 不打包 |
| VPoser checkpoint | 同 SMPL-X 注册 (only if stretch) | 5 MB | non-commercial | 不打包 |

> ⚠️ **5/3 必做**：Yiqiao 同时注册 **SMPL + SMPL-X**（两个独立站点，都用学校邮箱）

### C.3 测试图选择（修复版 — license 安全）

| # | 类型 | 来源 | 评估目的 |
|---|------|------|---------|
| 1 | 单人正面站立 | Unsplash (CC0) | baseline |
| 2 | 单人复杂姿势 | Unsplash sport (CC0) | 姿态先验测试 |
| 3 | 部分遮挡 | Unsplash portrait (CC0) | in-the-wild 鲁棒 |
| 4 | 多人交互 | Unsplash street (CC0) | 多人能力 |
| 5 | 自拍合影 | 自己/室友（征同意）| 自带感染力 |

**不再用**：教室截图（涉及隐私 + license）、体操比赛截图（版权风险）

### C.4 评估方法（**修复版 — 去循环逻辑**）

#### 评估 1: HRNet 复现指标（命中"reproduction"）

| 指标 | 来源 | 报告位置 |
|------|------|---------|
| **COCO mini sanity-check (OKS/AP on 10-50 images)** | MMPose val 子集 | report §4 表格 + slide 6（**明确标注 "subset, not full COCO val"**）|
| 引用对比目标 | HRNet 论文 Table 6: W32 = 74.4 AP, W48 = 75.5 AP (full COCO val) | **不声称本地复现 paper 数字**（小子集方差太大）|

#### 评估 2: HMR2 / SMPLify-X **runtime + agreement**（**不称 accuracy**）

| 指标 | 含义 | 评估方法 |
|------|------|---------|
| **Runtime (s/img)** | 端到端时间 (Colab T4) | 直接 `time.time()` 测，HMR2 期望 ~50ms, SMPLify-X 期望 ~30s |
| **Joint agreement (mm)** | HMR2 vs SMPLify-X 在共同 14 关节 (J24 子集) Procrustes 对齐后 L2 | 仅 stretch 完成时报告，**明确标注 "agreement, not accuracy"** |
| **Reported PA-MPJPE** (引用) | HMR (2018) ≈ 58 mm / HMR2 (2023) ≈ 44 mm / SMPLify ≈ 82 mm (3DPW) | 直接引用论文，写在 §4 narrative |

> ⚠️ **不再用 mesh L2 distance**（SMPL 6890 vs SMPL-X 10475 拓扑不同）
> ⚠️ **不再写 `regress_err = 0.0`**（无意义）

#### Narrative 修正（report §4 / slide 11 用）

> "HMR's 2018 contribution was replacing SMPLify's 30-second optimization with a 50-millisecond forward pass. **We empirically measure both runtimes on identical 2D inputs from HRNet and confirm the ~600× speedup claim**. For accuracy, we cite the original papers' Human3.6M / 3DPW PA-MPJPE numbers (HMR ≈ 58 mm, SMPLify ≈ 82 mm, HMR 2.0 ≈ 44 mm). Joint agreement between our local SMPLify-X and HMR2 outputs is reported as a complementary qualitative signal — **not a substitute for ground-truth-based accuracy evaluation**."

### C.5 文件结构（修复版）

```
GroupName_Code/                         ← GitHub repo 提交（不含 checkpoints/）
├── README.md                            # 1 页跑通指南 (5/8 写)
├── LICENSE                              # MIT
├── envs/
│   ├── env_hrnet_hmr2.yml               # 主环境 (PyTorch 2.x + mmcv 2.x + 4D-Humans)
│   └── env_smplifyx.yml                 # stretch 环境 (PyTorch 1.x + smplifyx)
├── notebooks/
│   ├── 01_main_pipeline.ipynb           # 主 demo: HRNet + 4D-Humans (env_hrnet_hmr2)
│   ├── 02_smplifyx_stretch.ipynb        # 优化路 stretch (env_smplifyx)
│   └── 03_smpl_forward_explain.ipynb    # 自写 LBS T-pose 渲染
├── src/
│   ├── hrnet_2d.py                      # MMPose 封装 (~30 LoC)
│   ├── hmr2_demo_wrapper.py             # 4D-Humans 官方 demo.py 包装 (~50 LoC)
│   ├── smplifyx_cli.py                  # subprocess 调 smplifyx CLI (~40 LoC)
│   ├── smpl_forward.py                  # smplx.lbs 注释版 (~25 LoC)
│   ├── compare.py                       # runtime + Procrustes joint MPJPE (~50 LoC)
│   └── visualize.py                     # quad-plot + 360° GIF (~60 LoC)
├── test_images/                         # 5 张 CC0 图
├── results/                             # 生成的所有图表
│   ├── quadplot_img{1-5}.png
│   ├── rotation_img{1-5}.gif
│   ├── runtime_table.csv / .tex
│   ├── agreement_table.csv / .tex      # 仅 stretch 完成时存在
│   └── full_demo.mp4                    # 30s 演讲嵌入视频
├── checkpoints/
│   ├── .gitignore                       # 强制忽略所有大模型
│   └── DOWNLOAD.md                      # 注册 + 下载链接
├── docs/
│   ├── pipeline_diagram.png
│   └── design_choices.md
└── scripts/
    ├── download_4dhumans.sh             # 4D-Humans 官方下载脚本
    └── setup_smpl_paths.sh              # SMPL/SMPL-X 路径软链
```

### C.6 主 notebook 伪代码（修复版 — API 严谨）

```python
# ============================================================
# 01_main_pipeline.ipynb (env_hrnet_hmr2)
# ============================================================

# --- Cell 1: 环境（pin 版本 + headless 渲染）---
!apt-get install -q -y libosmesa6-dev freeglut3-dev libglfw3-dev   # pyrender headless 必需 (gemini 提示)
import os; os.environ['PYOPENGL_PLATFORM'] = 'osmesa'              # ★ 必须在 import pyrender 之前设置

!pip install -q --upgrade pip
!pip install -q torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
!pip install -q -U openmim
!mim install -q "mmengine>=0.7.0" "mmcv>=2.0.0,<2.2.0" "mmdet>=3.1.0" "mmpose>=1.1.0"

# 4D-Humans 官方安装（按 README）
!git clone https://github.com/shubham-goel/4D-Humans.git
%cd 4D-Humans
!pip install -e .[all]    # ★ 必须 [all], 包含 detectron2 / pyrender / pytorch3d 等

# SMPL .pkl 从 Google Drive (用户已上传)
# ★ 4D-Humans/data/ 期望精确文件名 basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
!mkdir -p data
!cp /content/drive/MyDrive/smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl ./data/

# 下载 4D-Humans 自带 ckpt (替代不存在的 fetch_demo_data.sh)
from hmr2.utils.download_util import cache_url, download_models, CACHE_DIR_4DHUMANS
download_models(CACHE_DIR_4DHUMANS)

# --- Cell 2: HRNet 2D (MMPose 高层 API) ---
from mmpose.apis import MMPoseInferencer
import numpy as np, time, json

inferencer = MMPoseInferencer(
    pose2d='td-hm_hrnet-w48_8xb32-210e_coco-256x192',
    det_model='rtmdet-m', device='cuda'
)

results_2d = {}
for img_path in test_images:
    t0 = time.time()
    result_gen = inferencer(img_path, return_vis=True, vis_out_dir='results/2d_vis/')
    result = next(result_gen)
    results_2d[img_path.name] = {
        'kpts': np.array(result['predictions'][0][0]['keypoints']),  # (17, 2)
        'scores': np.array(result['predictions'][0][0]['keypoint_scores']),
        'runtime_s': time.time() - t0,
    }
np.savez('results/hrnet_kpts.npz', **{k: v['kpts'] for k, v in results_2d.items()})

# 释放 HRNet 显存（关键，T4 才装得下下一个）
del inferencer; import torch; torch.cuda.empty_cache()

# --- Cell 3: 4D-Humans (按官方 demo.py 风格) ---
import torch
from hmr2.models import HMR2, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hmr2.utils.renderer import Renderer, cam_crop_to_full
from detectron2.config import LazyConfig
from detectron2.engine import DefaultPredictor_Lazy

# ★ 用 DEFAULT_CHECKPOINT 而非 placeholder 路径
model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
model = model.cuda().eval()

# 加载 ViTDet detector
detectron2_cfg = LazyConfig.load('./hmr2/config/cascade_mask_rcnn_vitdet_h_75ep.py')
detector = DefaultPredictor_Lazy(detectron2_cfg)

results_3d = {}
for img_path in test_images:
    img_cv2 = cv2.imread(str(img_path))
    det_out = detector(img_cv2)
    boxes = det_out['instances'].pred_boxes.tensor.cpu().numpy()
    boxes = boxes[det_out['instances'].pred_classes.cpu().numpy() == 0]  # person only
    if len(boxes) == 0:
        results_3d[img_path.name] = {'verts': np.zeros((0, 6890, 3)), 'cam_t': np.zeros((0, 3)), 'runtime_s': 0.0}
        continue   # ★ codex 提示：empty detection guard

    dataset = ViTDetDataset(model_cfg, img_cv2, boxes)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)  # ★ T4 OOM 防御 (codex 提示)

    t0 = time.time()
    all_verts, all_cam_t = [], []
    for batch in dataloader:
        batch = recursive_to(batch, 'cuda')
        with torch.no_grad():
            out = model(batch)
        all_verts.append(out['pred_vertices'].cpu().numpy())
        all_cam_t.append(out['pred_cam_t'].cpu().numpy())

    results_3d[img_path.name] = {
        'verts': np.concatenate(all_verts, 0),    # (N人, 6890, 3) SMPL
        'cam_t': np.concatenate(all_cam_t, 0),
        'runtime_s': time.time() - t0,
    }

np.savez_compressed('results/hmr2_meshes.npz', **{k: v['verts'] for k, v in results_3d.items()})

# --- Cell 4: 渲染 quad-plot (Input | HRNet 2D | HMR2 front | HMR2 side) ---
import matplotlib.pyplot as plt
from src.visualize import render_smpl_view

def pick_center_person(verts_all, cam_t_all, img_shape):
    """★ gemini 提示：多人图选 bbox 中心最近者，避免选到背景人."""
    if len(verts_all) == 0: return None
    if len(verts_all) == 1: return 0
    H, W = img_shape[:2]; center = np.array([W/2, H/2])
    # 用 cam_t 的 (x,y) 投影距离中心最近为主体
    dists = [np.linalg.norm(cam_t[:2] - center / max(W, H)) for cam_t in cam_t_all]
    return int(np.argmin(dists))

for img_path in test_images:
    img = cv2.imread(str(img_path))[:, :, ::-1]
    vis_2d = cv2.imread(f'results/2d_vis/{img_path.name}')[:, :, ::-1]
    verts_all = results_3d[img_path.name]['verts']
    cam_t_all = results_3d[img_path.name]['cam_t']
    idx = pick_center_person(verts_all, cam_t_all, img.shape)
    if idx is None:
        continue   # 该图无人，跳过 quad-plot
    verts = verts_all[idx]

    front = render_smpl_view(verts, view='front')
    side  = render_smpl_view(verts, view='side')

    fig, ax = plt.subplots(2, 2, figsize=(12, 12))   # 2x2 grid (gemini 建议)
    ax[0,0].imshow(img);     ax[0,0].set_title('Input');         ax[0,0].axis('off')
    ax[0,1].imshow(vis_2d);  ax[0,1].set_title('HRNet-W48 2D');  ax[0,1].axis('off')
    ax[1,0].imshow(front);   ax[1,0].set_title('HMR 2.0 (front)'); ax[1,0].axis('off')
    ax[1,1].imshow(side);    ax[1,1].set_title('HMR 2.0 (side)');  ax[1,1].axis('off')
    plt.savefig(f'results/quadplot_{img_path.stem}.png', dpi=120, bbox_inches='tight')
    plt.close()

# --- Cell 5: 360° rotation GIF ---
import imageio
for img_path in test_images:
    verts = results_3d[img_path.name]['verts'][0]
    frames = [render_smpl_view(verts, view='rotate', angle=a) for a in range(0, 360, 24)]
    imageio.mimsave(f'results/rotation_{img_path.stem}.gif', frames, fps=10)

# --- Cell 6: runtime 表 ---
import pandas as pd
df = pd.DataFrame([
    {'image': name, 'HRNet_2D_ms': results_2d[name]['runtime_s']*1000,
     'HMR2_3D_ms': results_3d[name]['runtime_s']*1000}
    for name in (p.name for p in test_images)
])
df.loc['mean'] = df.mean(numeric_only=True)
df.to_csv('results/runtime_table.csv')
df.to_latex('results/runtime_table.tex', float_format='%.1f')
```

### C.7 Stretch notebook（5/5 EOD 硬截止前完成 = 加分；否则跳过）

```python
# 02_smplifyx_stretch.ipynb (env_smplifyx, 独立环境)
# ⚠️ 必须用单独 conda env: PyTorch 1.x + smplifyx tested
# ⚠️ Skip if 5/5 EOD 仍未跑通

import subprocess
import json

# 把 HRNet 17 keypoints 写成 OpenPose 25 格式（SMPLify-X 要求）
def coco17_to_openpose25(kpts_17):
    """COCO 17 → OpenPose BODY_25 mapping (with confidence)."""
    op25 = np.zeros((25, 3))
    mapping = {0:0, 16:15, ...}  # 详细 mapping 见 src/smplifyx_cli.py
    for op_idx, coco_idx in mapping.items():
        op25[op_idx] = [*kpts_17[coco_idx], 1.0]
    return op25

# 写入 SMPLify-X 期望的目录结构
for img_path in test_images:
    op_kpts = coco17_to_openpose25(results_2d[img_path.name]['kpts'])
    json_data = {'people': [{'pose_keypoints_2d': op_kpts.flatten().tolist()}]}
    with open(f'smplifyx_data/keypoints/{img_path.stem}_keypoints.json', 'w') as f:
        json.dump(json_data, f)

# 调 SMPLify-X CLI（官方接口）
ret = subprocess.run([
    'python', 'smplify-x/smplifyx/main.py',
    '--config', 'smplify-x/cfg_files/fit_smplx.yaml',
    '--data_folder', 'smplifyx_data/',
    '--output_folder', 'results/smplifyx_out/',
    '--visualize', 'False',
    '--model_folder', '/path/to/SMPLX_NEUTRAL.npz',
    '--vposer_ckpt', '/path/to/vposer_v1_0',
], check=True)

# 读取 SMPLify-X 输出 (per-image .pkl)
import pickle
results_optim = {}
for img_path in test_images:
    pkl_path = f'results/smplifyx_out/results/{img_path.stem}/000.pkl'
    if not os.path.exists(pkl_path):
        continue   # 优化失败该图，跳过
    with open(pkl_path, 'rb') as f:
        results_optim[img_path.name] = pickle.load(f)

# Agreement metric: J24 子集 Procrustes-aligned MPJPE (只在 stretch 成功时报)
from src.compare import procrustes_align, joint_subset_mapping

agreement_rows = []
for img_path in test_images:
    if img_path.name not in results_optim:
        continue
    j_hmr2 = extract_joints(results_3d[img_path.name]['verts'][0], 'SMPL')[J24_COMMON]
    j_optim = extract_joints(results_optim[img_path.name]['vertices'], 'SMPL-X')[J24_COMMON_X]
    j_optim_aligned = procrustes_align(j_optim, j_hmr2)
    err = np.linalg.norm(j_optim_aligned - j_hmr2, axis=-1).mean() * 1000  # mm
    agreement_rows.append({'image': img_path.name, 'agreement_mm': err})

pd.DataFrame(agreement_rows).to_csv('results/agreement_table.csv', index=False)
```

### C.8 自写 SMPL forward 解释（5-15 行，命中 "method implementation"）

```python
# src/smpl_forward.py — Reproducing SMPL Eq.(1) Loper et al. 2015
# 真实实现委托给 smplx.lbs.lbs，注释逐行对应论文公式
import smplx
import smplx.lbs as lbs
import torch

def smpl_forward_eq1(body_pose_72, betas_10, smpl_model_path):
    """SMPL Eq.(1): M(beta, theta) = W(T_P(beta, theta), J(beta), theta, W)."""
    model = smplx.create(smpl_model_path, model_type='smpl', gender='neutral')
    betas = torch.tensor(betas_10).float().unsqueeze(0)         # (1, 10) shape coeffs
    pose  = torch.tensor(body_pose_72).float().unsqueeze(0)     # (1, 72) axis-angle pose
    out = model(betas=betas, body_pose=pose[:, 3:], global_orient=pose[:, :3])
    # out.vertices: (1, 6890, 3)  ← T(beta,theta) + B_S(beta) + B_P(theta), then LBS
    # out.joints:   (1, 24, 3)    ← J_reg @ shaped_template
    return out.vertices.squeeze(), out.joints.squeeze()

# Demo: 渲染 T-pose（修复 model.faces scope bug — codex 提示）
def render_tpose_demo(smpl_model_path: str, out_obj: str):
    """T-pose mesh demo with faces accessible at the right scope."""
    import trimesh
    model = smplx.create(smpl_model_path, model_type='smpl', gender='neutral')
    out = model(
        betas=torch.zeros(1, 10),
        body_pose=torch.zeros(1, 69),       # 23 joint × 3 axis-angle
        global_orient=torch.zeros(1, 3),
    )
    mesh = trimesh.Trimesh(
        vertices=out.vertices.squeeze().detach().numpy(),
        faces=model.faces                   # ★ 函数 scope 内访问，无 NameError
    )
    mesh.export(out_obj)

render_tpose_demo('checkpoints/smpl/', 'results/tpose_demo.obj')
```

> 这部分**真实可运行 ≤ 15 行业务代码**，命中 PDF §6.6 "5-10 lines is sufficient" + "implementation of a method described in literature"

### C.9 关键代码量评估（修复后）

| 模块 | LoC | 备注 |
|------|-----|------|
| `hrnet_2d.py` | 30 | MMPose 高层 API 调用 |
| `hmr2_demo_wrapper.py` | 50 | 严格按 4D-Humans demo.py 改 |
| `smplifyx_cli.py` (stretch) | 40 | subprocess + COCO→OpenPose 映射 |
| `smpl_forward.py` | 25 | smplx.lbs 注释版 |
| `compare.py` | 50 | Procrustes + DataFrame |
| `visualize.py` | 60 | matplotlib + pyrender + imageio |
| **主路总计** | **165 LoC** | （不含 stretch） |

---

## (D) 三篇论文映射 + OpenPose 脱节解释（v3 新增）

| 论文 | demo 角色 | report § 引用方式 |
|------|----------|------------------|
| **OpenPose** | demo 不直接用 | §3.1 详细方法 + §4 一段对比说明 |
| **HRNet** | 主路 2D 提取 + 复现 COCO AP | §3.2 详细方法 + §4 表中数字 |
| **HMR** | 主路 3D 回归（用 4D-Humans / HMR 2.0 现代实现）| §3.3 详细方法 + §4 runtime 表 + Future 提 HMR 2.0 |

### §4 OpenPose 脱节解释段（必写）

> "Although OpenPose pioneered bottom-up multi-person 2D estimation, our demonstration employs a top-down ViTDet+HRNet pipeline. This architectural choice is deliberate: it guarantees high-fidelity 2D keypoints to **strictly isolate the 3D lifting step**. By factoring out 2D detection errors, we enable a clean, apples-to-apples comparison between optimization-based (SMPLify-X) and regression-based (HMR 2.0) 3D recovery. Nevertheless, OpenPose's bottom-up philosophy remains foundational, heavily influencing modern multi-person 3D frameworks like ROMP and BEV." [gemini polished]

---

## (E) 9 天精确时间表（v3 修正版）

### 总览（critical 修正：演讲交接 + 双注册 + stretch 截止）

| 日期 | 阶段 | 关键产物 | Yiqiao | Taijia |
|------|------|---------|--------|--------|
| **5/3 (今天)** | Day 0 | **双注册 + 测试图 + Excel 登记** | 注册 [SMPL](https://smpl.is.tue.mpg.de/) **+** [SMPL-X](https://smpl-x.is.tue.mpg.de/) ⏰ + 精读 OpenPose & HMR | 选 5 张 CC0 测试图 + 精读 HRNet + 写 Group 名 (`Liu-Liang-PoseRecovery`) |
| **5/4 (一)** | Day 1 | env + Excel 登记完成 | clone 4D-Humans + `bash fetch_demo_data.sh` + 跑通 demo 1 张图 (Colab T4) | 装 MMPose + 跑通 HRNet inference + Excel 登记 |
| **5/5 (二) ⏰** | Day 2 | **SMPL/SMPL-X 到位 + 主路通 + stretch 决定** | 收 SMPL/SMPL-X license → 上传 Drive；写 `smpl_forward.py`；跑 4D-Humans 5 张图 | 跑 HRNet 5 张图 → npz；尝试 env_smplifyx 装。**5/5 EOD 决定 stretch 是否走** |
| **5/6 (三)** | Day 3 | 完整 quad-plot + GIF | 写 `visualize.py` + 生成 5 张 quadplot + 5 个 GIF + runtime 表 | （主路成功）写 §3.2 HRNet 终稿；（stretch 成功）跑 SMPLify-X 5 张图 |
| **5/7 (四)** | Day 4 | 报告主体 + agreement 表 | 写 §3.3 HMR 终稿 + §5 Future + (stretch 成功) `compare.py` 出 agreement 表 | 写 §1 Intro + §2 Background (压 1 段) + §4 Comparison (含 OpenPose 脱节段 + 表) |
| **5/8 (五) ⏰** | Day 5 | **LaTeX dry-run 必须 ≤ 2 页** + GitHub repo public + mp4 录制 | 录 30s `full_demo.mp4` (5 张图轮播 + 旋转 GIF) + 上传 GitHub | LaTeX 编译 → 验证 ≤ 2 页（超则砍 §2 / §5 文字）+ 写 README.md 跑通指南 |
| **5/9 (六)** | Day 6 | PPT v1 + **完整排练计时** | 做 slides 8-13（HMR + Comparison + Demo + Future, 自讲 5 min 部分）| 做 slides 1-7（Intro + Background + OpenPose + HRNet, 自讲 5 min 部分） |
| **5/10 (日) ⏰** | Day 7 | **23:59 前 4 件套提交** | 录终稿 mp4 + Yiqiao 自排 ≥ 2 次 | LaTeX final + brightspace 上传 + GitHub commit lock + Taijia 自排 ≥ 2 次 |
| **5/11 (一)** | Day 8 | 现场演讲 60FA 110, 17:00–21:00 | **后 5 min**: HMR + Comparison + Demo + Future | **前 5 min**: Intro + Background + OpenPose + HRNet |

### 硬约束 checkpoint（误期 → 触发对应 fallback）

| Checkpoint | 验收标准 | 误期 → fallback |
|-----------|---------|----------------|
| **5/3 EOD** | SMPL 和 SMPL-X 注册邮件都已发 | 学校邮箱通常几小时通过；最坏延迟到 5/5 |
| **5/4 EOD** | 4D-Humans 跑通 ≥ 1 张图 + HRNet 跑通 1 张图 + Excel 登记 | 4D-Humans 装不上 → Tier 1 fallback |
| **5/5 EOD ⏰** | 主路 5 张图全跑通 + **stretch 是否能做的二元决定** | stretch 跑不通 → 直接砍掉，不再尝试 |
| **5/6 EOD** | quad-plot 5 张 + GIF 5 个 + runtime 表 | 渲染失败 → 改用 trimesh 软渲染 |
| **5/7 EOD** | report §1-§5 全部写完 (含 stretch agreement 段，如果有) | 内容超 → 砍 §2 + §5 文字 |
| **5/8 EOD ⏰** | LaTeX 编译 ≤ 2 页（**绝对硬卡**）+ mp4 + GitHub | 超 2 页 → 必须当晚改稿 |
| **5/9 EOD** | 完整排练 1 次 + 计时 (10 min ± 30s) | 超时 → 当晚改稿 |
| **5/10 23:59** | 4 件套上传 brightspace 完成 | 不允许误期 |

---

## (F) PPT 设计（v3 修复 — 1 次切换 + 2×2 grid + 预录 mp4）

### Slide 总览（13 张, 10 min, 1 次切换）

| Slide | 时长 | 内容 | **主讲（v3 修正）** |
|-------|------|------|------------------|
| 1 | 30s | Title + 钩子图（一张震撼的 3D 人体 mesh）| **Taijia (前 5 min 开始)** |
| 2 | 60s | Motivation + Research Question（"2D→3D leap"）| Taijia |
| 3 | 30s | 三篇论文 roadmap 图 | Taijia |
| 4-5 | 100s | OpenPose: PAFs 直觉（**视为黑盒，重点讲 multi-person 思想**）| Taijia |
| 6-7 | 100s | HRNet: 多分辨率融合（**视为黑盒，重点讲 high-res 思想**）+ COCO mini 子集 AP | Taijia |
| **──** | **── 麦克风切换 (5 sec) ──** | | **Taijia → Yiqiao** |
| 8-9 | 130s | HMR: SMPL 模型 + 对抗训练（**讲细致，是 demo 的核心**）| Yiqiao (后 5 min 开始) |
| 10 | 60s | 三方对比表 + OpenPose 脱节解释段口播 | Yiqiao |
| 11 | 60s | **Demo (预录 mp4 30s, 2×2 grid)** + runtime 表 + 600× speedup 数字 | Yiqiao |
| 12 | 30s | Future: Sapiens / VIBE / 3DGS Avatar (各一句解释 *why*)| Yiqiao |
| 13 | — | Q&A | 双方共答 |

> ★ **PPT 制作分工修正**（codex 发现 §E 5/9 行有矛盾）：
> - **Taijia 制作** slides 1-7（自己讲的部分）
> - **Yiqiao 制作** slides 8-13（自己讲的部分）
> - "谁讲谁做"原则，避免理解偏差

### Slide 11 详细布局（**v3 修复 — 2×2 + GIF + 预录 mp4**）

```
┌───────────────────────────────────────────────────────────────┐
│  Validating HMR's 600× Speedup Claim — Pre-recorded Demo     │
│                                                               │
│  ┌───────────────────────┬───────────────────────────────┐   │
│  │   Input Image         │   HRNet-W48 2D Keypoints      │   │
│  │   (img1)              │   (overlay)                   │   │
│  └───────────────────────┴───────────────────────────────┘   │
│  ┌───────────────────────┬───────────────────────────────┐   │
│  │   HMR 2.0 (Front Mesh)│   HMR 2.0 (360° View) 🎞       │   │
│  │   static render       │   rotating loop               │   │
│  └───────────────────────┴───────────────────────────────┘   │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Method        │ Runtime/img │ PA-MPJPE (cited)       │  │
│  │  HMR2 (regress)│   50 ms     │   ~44 mm (3DPW, paper) │  │
│  │  SMPLify (opt) │   30 s      │   ~82 mm (3DPW, paper) │  │
│  │  Speedup       │   600×      │                        │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
│  Narration (40s) [gemini polished]:                          │
│   "Here we see the full pipeline in action. By feeding the   │
│    same HRNet 2D keypoints into both optimization and        │
│    regression pathways, we empirically verified Kanazawa's   │
│    core 2018 claim: achieving a 600× speedup — dropping      │
│    from 30 seconds per image to just 50 milliseconds —       │
│    without sacrificing visual coherence. For rigorous        │
│    accuracy validation, we rely on the Human3.6M and 3DPW    │
│    benchmark results reported in the literature."            │
└───────────────────────────────────────────────────────────────┘
```

### Slide 12 详细布局（v3 修复 — *why* 加上）

```
┌───────────────────────────────────────────────────────────────┐
│  Where the Field is Heading                                  │
│                                                               │
│  ┌─────────────┬─────────────┬─────────────┐                 │
│  │  Sapiens    │   VIBE      │  3DGS-Avatar│                 │
│  │  (ECCV'24)  │  (CVPR'20)  │  (2024)     │                 │
│  │ Foundation  │ Video temp. │ Photorealism│                 │
│  └─────────────┴─────────────┴─────────────┘                 │
│                                                               │
│  Why these are the future:                                   │
│   • Sapiens scales the 2D backbone with massive pretraining  │
│   • VIBE adds temporal priors that single-image HMR ignores  │
│   • 3DGS bypasses the SMPL mesh entirely for photo-real      │
│                                                               │
│  Code: github.com/<group>/<repo>                             │
│  Colab: colab.research.google.com/.../01_main_pipeline.ipynb │
│                                                               │
│  Thank you. Questions?                                       │
└───────────────────────────────────────────────────────────────┘
```

---

## 🚨 风险矩阵 + 三层 fallback（v3 修复）

### Tier 1（高频高影响）

| 风险 | 概率 | 影响 | 缓解 |
|------|-----|------|------|
| SMPL/SMPL-X 注册延迟 | 中 | 高 | **5/3 立即用学校邮箱注册，两站同时** |
| 4D-Humans 安装失败 (detectron2) | 中 | 高 | **全程 Colab，不在本地装**；Colab 已有官方 detectron2 |
| Colab T4 OOM | 中 | 中 | **顺序加载**: HRNet → del → 4D-Humans → del；中间存 npz |
| pyrender macOS 渲染崩溃 | 高 (本地) | 中 | Colab Linux 环境 + osmesa；本地 fallback 用 trimesh 软渲染 |
| Colab GPU quota 用完 | 低 | 高 | Kaggle Notebook (30h/周 P100) |

### Tier 1 Fallback（v3 修正：替换 v2 不切实际的 fallback）

> v2 fallback "PyMAF / Multi-HMR / EasyMocap" 同样依赖复杂 → **不再使用**

**v3 Tier 1 fallback** = **保底主路**：
- HRNet 2D ✅
- 4D-Humans 官方 demo ✅
- quad-plot + GIF + runtime 表 ✅
- 引用论文报道的 PA-MPJPE 数字到 §4 ✅
- **不做 SMPLify-X 优化路对比**（少一个 deliverable，但仍命中所有 PDF 要求）

### Tier 2 Fallback（5/5 主路也未跑通时）

切回 **方案 D — 自写 SMPL component**：
- 全力做 `smpl_forward.py` (smplx.lbs 注释版) → T-pose 渲染 + 手动改 pose 渲染示例
- HRNet 2D 可视化（只跑 1-2 张图，不依赖 4D-Humans）
- demo 改为 "我们实现并解释了 SMPL 模型的核心 LBS 公式 (Loper 2015 Eq.1)"
- bonus 预期: 3.5-4/5（仍命中 §6.6 "5-10 lines implementation"）

### Tier 3 Fallback（极端情况，5/7 仍未跑通任何 demo）

- 提交 report + slides，**no working model**
- bonus 0%，但 presentation (20) + report (20) 仍可拿
- 风险事先告知老师，争取理解

### 新增 license 风险

| 风险 | 缓解 |
|------|------|
| SMPL-X "HF mirror" 灰色路径 | **绝对禁止**，只用官方注册 |
| Checkpoint 被打包进 zip | `.gitignore` 强制忽略 + 提交前 `find . -name "*.pkl" -o -name "*.ckpt"` 检查 |
| 测试图版权 | 全部用 Unsplash CC0 + 自拍 |

---

## 📋 立即可执行 first action（5/3 今晚必做）

### 🔴 P0（今晚 3 小时内必完成）

1. **[Yiqiao]** 用学校邮箱（.edu）**同时**注册：
   - https://smpl.is.tue.mpg.de/
   - https://smpl-x.is.tue.mpg.de/
   - 申请理由统一写 "Intro to CV final project, non-commercial academic research, group of 2"

2. **[Taijia]** 准备 5 张 CC0 测试图：
   - 来源：Unsplash + 自拍合影
   - 命名：`img1_standing.jpg ... img5_selfie.jpg`
   - 放到 `test_images/`
   - 解析度 ≥ 800×800，**统一裁剪到 4:3 或 1:1 长宽比**（避免 quadplot 不齐）

3. **[共同]** 在 Excel 协调表登记：
   - Group Name: `Liu-Liang-PoseRecovery`
   - 三篇论文：OpenPose `[new]`, HRNet `[new]`, HMR

### 🟡 P1（5/4 上午）

4. **[Yiqiao]** Colab 上 clone 4D-Humans + `bash fetch_demo_data.sh` + 跑通 1 张图
5. **[Taijia]** Colab 上装 MMPose + 跑通 HRNet 1 张图
6. **[共同]** 在 GitHub 创建 repo `Liu-Liang-PoseRecovery`，初始化 LICENSE (MIT) + .gitignore (含 `checkpoints/`)

### 🟢 P2（5/5 SMPL/SMPL-X 到位之后）

7. 上传 SMPL/SMPL-X .pkl/.npz 到 Google Drive（私有，仅团队访问）
8. （可选 stretch）`git clone smplify-x` + 装独立 env

---

## ✅ 交付物 final checklist（v3）

| 类型 | 文件 | 截止 | 备注 |
|------|------|------|------|
| Slides | `Liu-Liang-PoseRecovery_Presentation.pdf` | 5/10 23:59 | 13 张, 10 min, 1 次切换 |
| Report tex | `Liu-Liang-PoseRecovery_Report.tex` | 5/10 23:59 | article class |
| Report pdf | `Liu-Liang-PoseRecovery_Report.pdf` (≤2 页严格) | 5/10 23:59 | 5/8 dry-run 必过 |
| Code | `Liu-Liang-PoseRecovery_Code.zip` 或 GitHub URL | 5/10 23:59 | **不含 checkpoints/** |
| Excel 登记 | Group + 3 论文 | 5/4 | OpenPose / HRNet 标 [new] |

---

## 🔗 关键链接速查（v3）

| 资源 | URL |
|------|-----|
| SMPL 注册 | https://smpl.is.tue.mpg.de/ |
| SMPL-X 注册 | https://smpl-x.is.tue.mpg.de/ |
| 4D-Humans (HMR 2.0) | https://github.com/shubham-goel/4D-Humans |
| MMPose | https://github.com/open-mmlab/mmpose |
| SMPLify-X (stretch) | https://github.com/vchoutas/smplify-x |
| smplx PyTorch lib | https://github.com/vchoutas/smplx |
| Unsplash (CC0 图) | https://unsplash.com/license |

---

## 📝 SESSION_ID（供 /ccg:execute resume 用）

- **CODEX_SESSION**: `019dec2c-7056-7812-89e8-54b6772aa58d` (v2 review)
- **GEMINI_SESSION**: `e1f1cb98-2a7c-4853-be62-e326abc0931e` (v2 review)
- 新一轮 v3 review 会启动新 session，写入文件末尾

---

## 🚀 下一步

```
/ccg:execute .claude/plan/cv-final-2d-to-3d-v3.md
```

执行阶段会按 v3 创建：
1. `notebooks/01_main_pipeline.ipynb` 骨架（API-faithful）
2. `notebooks/02_smplifyx_stretch.ipynb` 骨架（独立环境标记）
3. `notebooks/03_smpl_forward_explain.ipynb`（自写 LBS 解释）
4. `src/*.py` 6 个文件骨架
5. `envs/env_hrnet_hmr2.yml` + `envs/env_smplifyx.yml`
6. `checkpoints/.gitignore` + `checkpoints/DOWNLOAD.md`
7. `scripts/download_4dhumans.sh` + `setup_smpl_paths.sh`
8. 更新 `LiteratureReview/main.tex`：§4 加 OpenPose 脱节解释段、Compact comparison table 占位
9. 更新 `README.md` daily checkpoint 反映 v3 时间表
