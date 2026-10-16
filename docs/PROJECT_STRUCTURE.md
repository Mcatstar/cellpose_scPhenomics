# 项目结构详解

本文档逐目录说明项目结构、每个目录的用途、文件的生命周期，以及数据在不同目录之间的流转。

---

## 顶层目录

```
cellpose_scPhenomics/
├── LICENSE
├── Makefile                    <- 便捷命令 (make lint / format / test)
├── pyproject.toml              <- 依赖与打包配置 (uv)
├── pyrightconfig.json          <- 类型检查配置, VS Code / Pylance 同样读取
├── README.md
├── data/                       <- 所有数据 (不提交)
├── models/                     <- 模型权重与微调产物 (不提交)
├── deepcontact/                <- mito-ER 接触位点量化代码 (第三方重构)
├── empanada/                   <- 内嵌的 empanada-dl 上游库
├── empanada_core/              <- 剥离 GUI 后的 empanada 核心逻辑
├── src/                        <- 项目主代码
├── docs/                       <- 文档
├── notebooks/                  <- Jupyter 探索 (可选)
├── references/                 <- 参考文献
├── reports/                    <- 分析输出 (CSV 与图表)
└── tests/                      <- 单元测试
```

---

## data/

所有数据按阶段分层。每层只读上一层，不反向依赖。

```
data/
├── raw/                        原始数据 (最原始，永不修改)
├── images/                     原始 EM 图像，按组子目录组织
│   ├── C/                      Control
│   ├── M/                      Model
│   └── T/                      Treatment
├── external/                   人工标注
│   ├── <group>-<stem>_mito_labels.tif
│   └── <group>-<stem>_ld_labels.tif
├── interim/                    dataset.py 裁掉黑边后的中间产物
│   ├── <group>-<stem>_img.tif
│   ├── <group>-<stem>_mito_masks.tif
│   ├── <group>-<stem>_ld_masks.tif
│   ├── metadata.csv            <- 尺寸、放大倍率、nm/px
│   └── train_test_split.csv
└── processed/                  建模数据集
    ├── train/
    │   ├── mito/               *_img.tif + *_mito_masks.tif
    │   └── ld/                 *_img.tif + *_ld_masks.tif
    ├── test/
    │   ├── mito/
    │   └── ld/
    ├── infer/                  仅 *_img.tif
    ├── predictions/            推理输出
    │   ├── <name>_mito_masks_pred.tif
    │   └── <name>_ld_masks_pred.tif
    ├── finetune/               Empanada 微调数据
    │   └── <source>/{images,masks}/
    └── finetune_eval/          Empanada 验证数据
        └── <source>/{images,masks}/
```

### 各阶段职责

| 目录 | 谁生成 | 谁读取 | 生命周期 |
|---|---|---|---|
| `data/images/` | 实验人员 | `dataset.py` | 永久 |
| `data/external/` | 标注人员 | `dataset.py` | 永久 |
| `data/interim/` | `dataset.py` | `dataset.py` | 可重新生成 |
| `data/processed/train/` | `dataset.py` | `EmpanadaFinetune prepare` | 可重新生成 |
| `data/processed/infer/` | `dataset.py` | `EmpanadaPredict` | 可重新生成 |
| `data/processed/predictions/` | `EmpanadaPredict` | `postproc.py` | 可重新生成 |
| `data/processed/finetune/` | `EmpanadaFinetune prepare` | `EmpanadaFinetune finetune` | 可重新生成 |
| `data/processed/finetune_eval/` | `EmpanadaFinetune prepare --split test` | `EmpanadaFinetune finetune --eval-dir` | 可重新生成 |

### metadata.csv

`data/interim/metadata.csv` 是整个流程的关键中间产物。字段见 README 的「数据约定」。任何阶段需要 `nm_per_px` 都从这里查询，不要硬编码。

---

## models/

```
models/
├── MitoNet_v1.pth              预训练线粒体模型 (TorchScript)
├── DropNet_base_v1.pth         预训练脂滴模型
├── MitoNet_v1_ft.pth           微调产物
├── MitoNet_v1_ft.yaml          微调配置
└── DropNet_base_v1_ft.pth
```

预训练模型从 Zenodo 下载，放在 `models/` 下。微调产物也放在这里。

`~/.empanada/configs/` 是模型的注册目录，`get_configs()` 会扫描这里。微调后加 `--register` 会把配置复制到那里。

---

## deepcontact/

从 DeepContact 重构的接触量化逻辑。原项目基于 TensorFlow + Amira GUI，这里只保留了纯 numpy/scipy 的部分。

```
deepcontact/
├── __init__.py
└── contact.py                  距离变换 + 形态学 + cKDTree 的向量化实现
```

**注意**：本项目实际使用的接触量化逻辑在 `src/postproc.py`，`deepcontact/` 只作参考。两者的核心算法一致，但 `postproc.py` 完全独立，不 import `deepcontact/`。

---

## empanada/

内嵌的 `empanada-dl` 上游库。用于：

- 模型定义（`PanopticDeepLab`、`BiFPN`、量化模型）
- 数据集类（`SingleClassInstanceDataset`、`PanopticDataset`）
- 推理引擎（`PanopticDeepLabEngine`、`PanopticDeepLabEngine3d`）
- 评估指标（IoU、PQ、F1）

```
empanada/
├── inference/
│   ├── engines.py              Engine 定义
│   ├── postprocess.py          距离变换、panoptic 解码
│   ├── tracker.py              3D 跟踪
│   ├── matcher.py              Hungarian 匹配
│   └── rle.py                  Run-length encoding
├── data/
│   ├── single_class_instance_dataset.py
│   ├── panoptic_dataset.py
│   └── utils/
│       ├── transforms.py
│       └── target_creation.py
├── models/
│   ├── panoptic_deeplab.py
│   ├── panoptic_bifpn.py
│   └── quantization/
├── evaluation/
├── losses.py
├── metrics.py
├── consensus.py
├── config_loaders.py
└── zarr_utils.py
```

**不要修改这个目录**。它是内嵌的上游代码，更新时整体替换。

---

## empanada_core/

从 `empanada-napari` 剥离 GUI 后的核心逻辑。本项目的建模脚本通过它调用 Empanada。

```
empanada_core/
├── inference.py                Engine2d / Engine3d
├── multigpu.py                 MultiGPUEngine3d (DDP)
├── pipeline.py                 引擎构建、ROI 裁剪、批量切片、正交三轴驱动
├── train.py                    从零训练
├── finetune.py                 微调循环 + 配置组装
├── patches.py                  补丁抓取、大图分块与合并
├── tiles.py                    分块推理的拼接
├── labels.py                   形态学 / 合并 / 分水岭拆分
├── label_stats.py              小目标过滤、标签计数、xlsx 导出
├── metrics.py                  像素与实例 IoU / Dice / Hungarian 匹配
├── model_io.py                 .empanada 模型包导出 / 导入 / 归档
├── utils.py                    Preprocessor、权重下载、模型注册
├── configs/                    4 个内置模型配置
│   ├── MitoNet_v1.yaml
│   ├── MitoNet_v1_mini.yaml
│   ├── DropNet_base_v1.yaml
│   └── NucleoNet_base_v2.yaml
└── training/                   训练 / 微调 YAML 模板
    ├── train_config.yaml
    └── finetune_config.yaml
```

详见 `empanada_core/README.md`。

---

## src/

项目主代码。按数据流分层。

```
src/
├── config.py                   路径与日志配置
├── dataset.py                  裁黑边 + OCR 读标尺 + 组装 train/test/infer
├── features.py                 (已清空, 逻辑并入 dataset.py)
├── postproc.py                 形态学与接触指标
├── visualize.py                分组统计与可视化
├── modeling/
│   ├── CellposeTrain.py        Cellpose 微调
│   ├── CellposePredict.py      Cellpose 推理
│   ├── EmpanadaFinetune.py     Empanada 微调 (prepare / finetune)
│   └── EmpanadaPredict.py      Empanada 2D 推理
└── utils/                      padding / roi_cut / tiling
```

### 模块职责

| 模块 | 输入 | 输出 | 依赖 |
|---|---|---|---|
| `config.py` | 无 | 路径常量 | 无 |
| `dataset.py` | `data/images/` | `data/interim/`、`data/processed/` | `config` |
| `postproc.py` | `data/processed/predictions/` | `reports/*.csv` | `config` |
| `visualize.py` | `reports/*.csv` | `reports/figures/*.png` | `config` |
| `modeling/EmpanadaPredict.py` | `data/processed/infer/` | `data/processed/predictions/` | `empanada_core` |
| `modeling/EmpanadaFinetune.py` | `data/processed/{train,test}/` | `models/` | `empanada_core` |

### 入口命令

| 脚本 | 命令 |
|---|---|
| 数据准备 | `uv run python -m src.dataset` |
| Empanada 推理 | `uv run python -m src.modeling.EmpanadaPredict --tile-size 1024` |
| Cellpose 推理 | `uv run python -m src.modeling.CellposePredict` |
| 后处理 | `uv run python -m src.postproc` |
| 可视化 | `uv run python -m src.visualize` |
| Empanada 微调 | `uv run python -m src.modeling.EmpanadaFinetune finetune` |

---

## docs/

```
docs/
├── DevelopsRules.md            Python 开发强制约束
├── FINETUNE.md                 Empanada 微调教程
├── PROJECT_STRUCTURE.md        本文件
└── docs/                       mkdocs 输出
```

---

## reports/

分析输出。CSV 和图表可以直接作为论文素材。

```
reports/
├── global_metrics.csv          每张图的全局汇总
├── mito_morphology.csv         每个线粒体实例的形态 + PDM/CM
├── ld_morphology.csv           每个脂滴实例的形态 + 大小分类
├── contact_pairs.csv           每对 (mito, ld) 的接触
├── contact_width_bins.csv      每张图 × 宽度 bin
├── contact_patterns.csv        每个 mito 的接触模式
├── group_summary.csv           按组汇总
├── stat_tests_between_groups.csv
├── stat_tests_pdm_vs_cm.csv
└── figures/
    ├── global_metrics_boxplots.png
    ├── morphology_boxplots.png
    ├── contact_width_spectrum.png
    ├── pdm_vs_cm.png
    ├── ld_size_distribution.png
    ├── pdm_vs_ld_count.png
    ├── contact_length_per_pair.png
    ├── contact_ratio_per_instance.png
    ├── pdm_cm_contact_metrics.png
    └── contact_absolute_vs_relative.png
```

`reports/*.csv` 和 `reports/figures/*.png` 可以提交到 Git。`reports/thr_*/` 等阈值敏感性分析的中间产物建议不提交。

---

## 数据流转全景

```
data/images/C/1-2.5-1.tif
        │
        ▼ dataset.py
data/interim/C-1-2.5-1_img.tif
data/interim/metadata.csv  (记录 nm_per_px = 4.386)
        │
        ├──→ data/processed/train/mito/C-1-2.5-1_img.tif  ──→ EmpanadaFinetune prepare
        │                                                    ──→ models/MitoNet_v1_ft.pth
        │
        ├──→ data/processed/test/mito/  ──→ EmpanadaFinetune prepare --split test
        │                                 ──→ data/processed/finetune_eval/
        │
        └──→ data/processed/infer/C-1-2.5-1_img.tif
                    │
                    ▼ EmpanadaPredict
             data/processed/predictions/
                 C-1-2.5-1_mito_masks_pred.tif
                 C-1-2.5-1_ld_masks_pred.tif
                    │
                    ▼ postproc (查 metadata 拿 4.386)
             reports/
                 mito_morphology.csv  (含 C-1-2.5-1 所有 mito 行)
                 ld_morphology.csv
                 contact_pairs.csv
                 ...
                    │
                    ▼ visualize
             reports/figures/*.png
```

---

## 不提交到 Git 的目录

以下目录内容由流程生成，不提交：

- `data/raw/`
- `data/images/`
- `data/external/`
- `data/interim/`
- `data/processed/`
- `models/*.pth`

以下内容可以提交：

- `reports/*.csv`（论文素材）
- `reports/figures/*.png`（论文素材）
- `models/*.yaml`（模型配置，不含权重）
- 所有 `src/`、`docs/`、`tests/` 内容

`.gitignore` 已配置好。

---

## 相关文档

- 项目入口：`README.md`
- 微调教程：`docs/FINETUNE.md`
- 开发规范：`docs/DevelopsRules.md`
- empanada_core API：`empanada_core/README.md`