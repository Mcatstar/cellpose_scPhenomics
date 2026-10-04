# cellpose_scPhenomics

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Cellpose-based single-cell phenomics pipeline for microscopy imaging.

## 项目结构

```
├── LICENSE
├── Makefile                    <- 便捷命令 (make lint / format / test)
├── pyproject.toml              <- 依赖与打包配置 (uv)
├── pyrightconfig.json          <- 类型检查配置, VS Code / Pylance 同样读取
├── README.md
│
├── data
│   ├── raw                     <- 原始数据
│   ├── images                  <- EM 图像 (C / M / T 三组)
│   ├── interim                 <- 中间数据 (裁剪、归一化)
│   ├── external                <- 第三方数据; 也是 Empanada 推理输出目录
│   └── processed               <- 建模数据集
│       ├── train / test        <- Cellpose: *_img.tif + *_mito_masks.tif
│       ├── infer               <- 仅推理样本
│       ├── predictions         <- Cellpose 推理输出
│       └── finetune            <- Empanada 微调数据集 (根目录, 下设 <source>/)
│
├── models                      <- 权重: MitoNet_v1.pth / DropNet_base_v1.pth / 微调产物
│
├── deepcontact                 <- mito-ER 接触位点量化 (第三方代码重构, 无 GUI)
│   ├── contact.py              <- 距离变换 + 形态学 + cKDTree 的向量化实现
│   └── __init__.py
│
├── empanada                    <- 内嵌的 empanada-dl 上游库
│   ├── inference               <- engines / postprocess / tracker / matcher / rle ...
│   ├── data                    <- SingleClassInstanceDataset / PanopticDataset ...
│   ├── models                  <- PanopticDeepLab / BiFPN / 量化模型
│   ├── evaluation              <- 实例与语义指标
│   └── losses.py / metrics.py / consensus.py / config_loaders.py / zarr_utils.py
│
├── empanada_core               <- 由 empanada-napari 剥离 GUI 后的核心逻辑
│   ├── inference.py            <- Engine2d / Engine3d
│   ├── multigpu.py             <- MultiGPUEngine3d (DDP)
│   ├── pipeline.py             <- 引擎构建、ROI 裁剪、批量切片、正交三轴驱动
│   ├── train.py / finetune.py  <- 训练与微调循环 + 配置组装
│   ├── patches.py / tiles.py   <- 补丁抓取、大图分块与合并
│   ├── labels.py               <- 形态学 / 合并 / 分水岭拆分
│   ├── label_stats.py          <- 小目标过滤、标签计数、xlsx 导出
│   ├── metrics.py              <- 像素与实例 IoU / Dice / Hungarian 匹配
│   ├── model_io.py             <- .empanada 模型包导出 / 导入 / 归档
│   ├── utils.py                <- Preprocessor、权重下载、模型注册
│   ├── configs/                <- 4 个内置模型配置
│   └── training/               <- 训练 / 微调 YAML 模板
│
├── src
│   ├── config.py               <- 路径与日志配置
│   ├── dataset.py              <- 数据整理
│   ├── features.py             <- 划分 train/test、归档 infer
│   ├── postproc.py             <- 形态学与接触指标
│   ├── visualize.py
│   ├── modeling
│   │   ├── CellposeTrain.py    <- Cellpose 微调
│   │   ├── CellposePredict.py  <- Cellpose 推理
│   │   ├── EmpanadaFinetune.py <- Empanada 微调 (prepare / finetune)
│   │   └── EmpanadaPredict.py  <- Empanada 2D 推理
│   └── utils                   <- padding / roi_cut / tiling
│
├── docs
│   ├── DevelopsRules.md        <- Python 开发强制约束
│   └── docs/                   <- mkdocs
│
├── notebooks / references / reports
└── tests
```

## 建模脚本

| 脚本 | 作用 | 命令 |
| --- | --- | --- |
| `CellposeTrain` | Cellpose 微调 (mito + lipid) | `uv run python -m src.modeling.CellposeTrain` |
| `CellposePredict` | Cellpose 推理 | `uv run python -m src.modeling.CellposePredict` |
| `EmpanadaFinetune` | 准备 empanada 数据集并微调 | `uv run python -m src.modeling.EmpanadaFinetune prepare` / `finetune` |
| `EmpanadaPredict` | Empanada 2D 推理 (mito + lipid) | `uv run python -m src.modeling.EmpanadaPredict --tile-size 1024` |

`EmpanadaFinetune` 的 `--check-only` 只做数据集校验、配置组装与 epoch 计算, **不加载任何模型**,
适合先在跑不动模型的机器上确认环境。完整教程见文末的「Empanada 模型微调教程」。

## 环境与依赖

```bash
uv sync
```

除 Cellpose 之外, Empanada 链路还需要 `zarr` (3D 跟踪与共识)、`dask` (体积数据集与补丁抓取)、
`cztile` (分块推理), 三者都已声明在 `pyproject.toml` 里。

---

# 分析流程与指标

把三篇文献的指标体系组合成一个完整的分析流程，需要分三个层次：**分割层 → 形态层 → 互作层**。下面给出完整的 Python 分析脚本，输入是 Mito/LD mask，输出是一份包含全部指标的 CSV 表格。

---

## 指标整合框架

| 层次                      | 指标来源           | 具体指标                                                |
| ------------------------- | ------------------ | ------------------------------------------------------- |
| **A. 全局计数**     | 共有               | Mito/LD 总数、总面积、平均面积、总周长                  |
| **B. 形态参数**     | 人肝活检           | 面积、周长、圆度、长宽比、圆度、实度                    |
| **C. 伸长因子**     | DeepContact        | perimeter²/(4π×area)                                 |
| **D. 大小分类**     | T2DM-MASLD         | LD: <2 μm vs >2 μm                                    |
| **E. 接触宽度分布** | DeepContact        | 0–100 nm，10 nm bin，接触长度/比例                     |
| **F. 接触比例**     | DeepContact + T2DM | 接触长度/Mito周长、接触长度/LD周长、接触Mito数/总Mito数 |
| **G. PDM/CM分类**   | T2DM + 人肝活检    | 按是否接触LD分类，分别统计形态                          |
| **H. 接触模式**     | T2DM               | 单Mito-单LD vs 多Mito-多LD成簇                          |

---

## 输出文件说明

| 文件                          | 内容                               | 对应指标来源    |
| ----------------------------- | ---------------------------------- | --------------- |
| `global_metrics.csv`        | 每张图的全局汇总                   | 三篇共用        |
| `mito_morphology.csv`       | 每个线粒体的形态参数 + PDM/CM 标签 | 人肝活检 + T2DM |
| `ld_morphology.csv`         | 每个脂滴的形态参数 + 大小分类      | T2DM            |
| `contact_events.csv`        | 每个接触事件的宽度 bin 和接触长度  | DeepContact     |
| `pdm_cm_classification.csv` | 每个线粒体的 PDM/CM 分类           | T2DM + 人肝活检 |
| `contact_patterns.csv`      | 每个接触事件的模式分类             | T2DM            |

---

## 关键参数说明

| 参数                     | 默认值 | 说明                                       |
| ------------------------ | ------ | ------------------------------------------ |
| `pixel_size_nm`        | 10.0   | EM 图像分辨率，1 像素 = 10 nm              |
| `max_distance_nm`      | 100.0  | 接触宽度分析的最大距离                     |
| `bin_width_nm`         | 10.0   | 宽度分 bin 的步长                          |
| `contact_threshold_nm` | 30.0   | 定义"接触"的距离阈值（LD-Mito 常用 30 nm） |

如果你的图像分辨率不是 10 nm/pixel，**必须改 `pixel_size_nm`**，否则所有长度和面积单位都会错。

---

## 与现有工作流的衔接

```
→  *_mito_masks(_pred).tif
                    →  *_ld_masks(_pred).tif
                              ↓
postproc.py  →  global_metrics.csv
                    →  mito_morphology.csv
                    →  ld_morphology.csv
                    →  contact_events.csv
                    →  pdm_cm_classification.csv
                    →  contact_patterns.csv
                              ↓
统计脚本（t 检验、ANOVA、Cohen's d、MIC）
```

---

# Empanada 模型微调

本教程基于 `empanada_core`（由 empanada-napari 插件剥离 GUI 后的核心逻辑）。**全程不需要安装 napari / magicgui / qtpy。**

## 1. 什么时候需要微调

`MitoNet_v1` / `DropNet_base_v1` 是通用模型。当你的数据域与训练集差异较大（不同物种、不同成像参数、不同像素尺寸、制样方式不同）而出现漏检、粘连或边界糊在一起时，用几十张自己标注的图做微调，通常比从零训练划算得多。

微调只更新权重、不改网络结构，产物是一个 TorchScript 模型，可直接交给 `Engine2d` / `Engine3d` 推理。

## 2. 前置条件

```bash
uv sync          # 或 pip install.
```

基础模型权重本项目已经放在 `models/`：

| 文件                           | 用途                   |
| ------------------------------ | ---------------------- |
| `models/MitoNet_v1.pth`      | 线粒体（单类实例分割） |
| `models/DropNet_base_v1.pth` | 脂滴（单类实例分割）   |

模型配置文件在 `empanada_core/configs/`。`get_configs()` 会同时扫描**包内 `configs/`** 和 **`~/.empanada/configs/`**，两处的同名项以 `~/.empanada` 为准。

> 微调用的配置模板是 `empanada_core/training/finetune_config.yaml`，`build_finetune_config()` 会自动读取，一般不需要改。

## 3. 数据目录结构

`empanada` 的数据集类要求这样的层级——注意 `images` 和 `masks` 必须**同级且一一对应**：

```
<train_dir>/                     <- 传给 build_finetune_config 的 train_dir
└── <source_dataset>/            <- 每个子目录被当作一个"来源数据集"
    ├── images/
    │   ├── 001.tiff
    │   └── 002.tiff
    └── masks/
        ├── 001.tiff             <- 与 images 同名最保险
        └── 002.tiff
```

`<source_dataset>` 这一层不是可有可无的装饰：`_BaseDataset` 会把每个子目录视为一个来源，并按 `weight_gamma` 在来源之间做加权采样（权重 ∝ `(1/图片数)^gamma`，MitoNet_v1 用 0.7）。如果你把所有图平铺在 `train_dir` 下，数据集会读不到任何样本。

### 掩码格式

掩码必须是**整数标号图**，不是 0/255 二值图：

| 数据集类                       | 掩码含义                                            | 适用                               |
| ------------------------------ | --------------------------------------------------- | ---------------------------------- |
| `SingleClassInstanceDataset` | 任意正整数 = 一个实例；语义分割由`mask > 0` 得到  | 单类，MitoNet / DropNet 都是这一类 |
| `PanopticDataset`            | `label = class_id × label_divisor + instance_id` | 多类，要求`len(labels) > 1`      |

选哪个由**模型配置的 `FINETUNE.dataset_class` 决定**，不需要你手动指定。MitoNet_v1 是 `SingleClassInstanceDataset`，所以标注时只要保证每个线粒体一个独立的整数 ID 即可。

### 图像格式

图像由 `cv2.imread(path, 0)` 读取，**结果一定是 8 位灰度**。本项目原始数据是 uint16（例如 `data/images/C/1-2.5-1.tif` 是 2563×3296 uint16，0–65520），OpenCV 会隐式压到 0–255。为了让缩放规则可控、可复现，建议在准备数据时自己先转成 uint8：

```python
import numpy as np
import tifffile

img = tifffile.imread("data/images/C/1-2.5-1.tif")     # uint16
img8 = (img / 256).astype(np.uint8)                    # 明确的 >>8 缩放
# 或按百分位拉伸，视你的数据动态范围而定
```

### 标注规范

微调前先用 `get_model_info()` 打印该模型要求的标注规范，照着做就行：

```python
from empanada_core.finetune import get_model_info

get_model_info("MitoNet_v1")
```

它会输出补丁边长需要被 `padding_factor` 整除（MitoNet_v1 是 16）、label divisor（单类模型为 `None`，即不使用）、以及每个类的类型（instance / semantic）和起始标号。

## 4. 准备训练样本

有三条路，按省事程度排序。

### 路线 A：一行命令转换（推荐）

如果数据已经是 `*_img.tif` + `*_mito_masks.tif` 成对放在 `data/processed/train/`，直接用
[`EmpanadaFinetune.py`](src/modeling/EmpanadaFinetune.py) 的 `prepare`：

```bash
uv run python -m src.modeling.EmpanadaFinetune prepare --target mito
```

它会显式把图转成 uint8、掩码转成 int32，写到：

```
data/processed/finetune/mito/images/*.tiff
data/processed/finetune/mito/masks/*.tiff
```

所以 `train_dir` 传 **`data/processed/finetune`**（`<target>` 的上级，不是 `<target>` 自己）。
掩码是 0/1 二值图时会打警告（那样整张图会被当成一个实例）。

### 路线 B：整图当样本，手工摆放

按第 3 节的目录结构自己摆好即可。训练时数据增强会做随机裁剪（`RandomCrop`，尺寸由 `patch_size` 决定），所以**不需要**你事先切图。图片数量建议 ≥ 20 张。

### 路线 C：从大图切补丁

图非常大、或者只想在感兴趣区域标注时，用 `empanada_core.patches` 切成固定大小的补丁：

```python
import numpy as np
from empanada_core.patches import pick_patches, patch_suffices, store_dataset

image = ...            # 2D 或 3D numpy 数组
mask = ...             # 与 image 同形状的整数标号图

# label 不为 None 时返回 (补丁, 掩码补丁, 位置列表)
patches, label_patches, locs = pick_patches(
    image, patch_size=256, num_patches=40, points=[], label=mask
)

store_dataset(
    patches, label_patches,
    save_dir="data/processed/finetune",
    dataset_name="mito",
    metadata={"prefix": "mito", "suffices": patch_suffices(locs, 0)},
)
```

落盘结果是：

```
data/processed/finetune/mito/mito/images/*.tiff    # uint8 灰度
data/processed/finetune/mito/mito/masks/*.tiff     # int32 标号图
```

那么 `train_dir` 就是 `data/processed/finetune/mito`（**`dataset_name` 那一层**，不是它的上级）。

- `points=[]` 表示随机取位置；也可以传入 `(n, 2)` 的坐标数组指定补丁中心（数据坐标）。
- `metadata` 里的 `suffices` 让保存时能按位置裁掉补丁的 padding 边；不传 `metadata` 时前缀会退化成 `"unknown"`、后缀变成随机串，能跑但不便于追溯。

## 5. 三步 API

```python
from empanada_core.finetune import build_finetune_config, finetune_and_register

cfg = build_finetune_config(
    model_name="MitoNet_v1_ft",                  # 微调后模型的名字
    train_dir="data/processed/finetune",         # 第 4 节路线 A 的输出目录
    eval_dir=None,                               # 有验证集就填路径
    model_dir="models",                          # 产物输出目录
    finetune_model="MitoNet_v1",                 # 以哪个已注册模型为起点
    finetune_layer="all",                        # 解冻哪些 encoder 层
    iterations=1000,                             # 训练迭代数（见第 7 节）
    patch_size=256,                              # 随机裁剪边长
    custom_config="default config",              # 或换成你自己的 yaml 路径
)

# 用本地权重，跳过 zenodo 下载
cfg["MODEL"]["model"] = "models/MitoNet_v1.pth"

yaml_path = finetune_and_register(cfg)
print(yaml_path)        # models/MitoNet_v1_ft.yaml
```

三个函数的职责：

| 函数                           | 作用                                                                                           |
| ------------------------------ | ---------------------------------------------------------------------------------------------- |
| `build_finetune_config(...)` | 读`finetune_config.yaml` + 基础模型配置，算出 epochs、save_freq，填好指标标签，返回配置 dict |
| `run_finetuning(cfg)`        | 跑训练循环，把权重和配置写到`<model_dir>/`，返回 yaml 路径                                   |
| `finetune_and_register(cfg)` | 上面一步 + 调用`add_new_model()` 注册到 `~/.empanada/configs/`                             |

只想跑不想注册，就用 `run_finetuning(cfg)`。

> 不覆盖 `cfg["MODEL"]["model"]` 的话，`load_model_to_device()` 会按配置里的 URL 从 zenodo 下载权重到 `~/.empanada/`（约 220 MB）。本地已有权重时直接覆盖成路径更快。

## 6. 配置项详解

`build_finetune_config()` 接受 9 个参数，其余全部来自两个 yaml。`empanada_core/training/finetune_config.yaml` 里的常用项：

| 键                                        | 默认                  | 说明                                                                            |
| ----------------------------------------- | --------------------- | ------------------------------------------------------------------------------- |
| `TRAIN.lr_schedule`                     | `OneCycleLR`        | 学习率调度器类名，取自`torch.optim.lr_scheduler`                              |
| `TRAIN.schedule_params.max_lr`          | `0.003`             | 峰值学习率。微调建议再小一个量级                                                |
| `TRAIN.schedule_params.steps_per_epoch` | `-1`                | 会被自动改成`len(train_loader)`                                               |
| `TRAIN.optimizer`                       | `AdamW`             | 优化器类名，取自`torch.optim`                                                 |
| `TRAIN.optimizer_params.weight_decay`   | `0.1`               | BatchNorm 和 bias 会被自动排除在 weight decay 之外                              |
| `TRAIN.batch_size`                      | `16`                | 同时参与 epochs 计算                                                            |
| `TRAIN.workers`                         | `4`                 | DataLoader 进程数，macOS 上会被强制改成 0                                       |
| `TRAIN.amp`                             | `True`              | 自动混合精度（`GradScaler` + `autocast`）                                   |
| `TRAIN.augmentations`                   | 见文件                | albumentations 列表；`height`/`width` 为 `null` 的会被填成 `patch_size` |
| `TRAIN.metrics`                         | `IoU`               | 训练期指标                                                                      |
| `TRAIN.additional_train_dirs`           | `null`              | 额外训练目录列表，会与主目录合并                                                |
| `EVAL.eval_dir`                         | `null`              | 验证集目录，不做验证就留空                                                      |
| `EVAL.metrics`                          | `IoU`/`PQ`/`F1` | 验证期指标                                                                      |
| `FINETUNE.*`                            | 继承自模型 yaml       | `dataset_class`、`criterion`、`engine` 等，**不要手改**             |

`finetune_layer` 可选 `none` / `stage4` / `stage3` / `stage2` / `stage1` / `all`。实现上先冻结**整个 encoder**，再按该值解冻从对应 stage 起的所有 encoder 参数；`none` 表示只训练 decoder，`all` 表示全部解冻。数据量小就从 `none` 或 `stage4` 起步，能明显降低过拟合风险。

## 7. epochs 是怎么算出来的（重要）

训练轮数不是直接给的，而是由 `iterations` 反推：

```
n_imgs   = len(glob(<train_dir>/**/images/*))
epochs   = int(iterations // (n_imgs // batch_size))
save_freq       = epochs // 5
epochs_per_eval = epochs // 5
```

例如 40 张图、`batch_size=16`：`n_imgs // batch_size = 2`，`iterations=1000` → `epochs=500`。

算出来的轮数写回配置时有两条分支：如果 `schedule_params` 里已经有 `epochs` 键（`train_config.yaml` 和 `finetune_config.yaml` 两个模板都是这样），就写进 `TRAIN.schedule_params.epochs`；只有模板里没有这个键时才会写 `TRAIN.epochs`。所以读完配置想取轮数，要这样取：

```python
params = cfg["TRAIN"]["schedule_params"]
epochs = params["epochs"] if "epochs" in params else cfg["TRAIN"]["epochs"]
```

**注意一个上游遗留的坑：`epochs` 小于 5 时 `save_freq` 会等于 0**，随后 `(epoch + 1) % save_freq` 直接抛 `ZeroDivisionError`。`train.py` 那边写的是 `max(1, epochs // 5)`，`finetune.py` 漏了这个保护。所以你至少要保证：

```
iterations >= 5 × (n_imgs // batch_size)
```

反例：400 张图（`400 // 16 = 25`）配 `iterations=100` → `epochs=4` → 崩溃。改成 `iterations >= 125` 即可。

## 8. 产物与后续推理

微调结束后产生三个文件：

| 路径                                      | 内容                                                                                               |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `<model_dir>/<model_name>.pth`          | TorchScript 权重（`torch.jit.save`）                                                             |
| `<model_dir>/<model_name>.yaml`         | 模型配置：`norms`、`padding_factor`、`thing_list`、`labels`、`class_names`、`FINETUNE` |
| `~/.empanada/configs/<model_name>.yaml` | 注册副本，供`get_configs()` 发现                                                                 |

注册之后就能像内置模型一样直接用：

```python
from empanada.config_loaders import load_config
from empanada_core.inference import Engine2d

cfg = load_config("models/MitoNet_v1_ft.yaml")

engine = Engine2d(cfg, use_gpu=True)
pan_seg = engine.infer(image_2d)        # image_2d: 2D numpy 数组
```

如果图很大、需要分块推理，改用 `empanada_core.pipeline.create_engine_2d(cfg, "MitoNet_v1_ft", tile_size=512)`；需要 3D 正交平面 + 跟踪共识，用 `Engine3d` 或 `create_engine_3d`。

想看微调前后的差别，可以用 `empanada_core.metrics.compute_instance_metrics(gt, pred, iou_threshold=0.5)` 算 PQ / F1 / 平均 IoU，它内部用 Hungarian 匹配，不需要 napari。

## 9. 完整流程

推荐直接用 [`src/modeling/EmpanadaFinetune.py`](src/modeling/EmpanadaFinetune.py) —— 数据集准备、配置组装、epoch 守卫和注册都包好了：

```bash
# 1. 把 data/processed/train/*_img.tif + *_mito_masks.tif 转成 empanada 数据集结构
#    输出 data/processed/finetune/mito/{images,masks}/*.tiff (图像显式转 uint8, 掩码 int32)
uv run python -m src.modeling.EmpanadaFinetune prepare --target mito

# 2. 校验: 数据集结构、模型配置、epochs 全部检查一遍, 不加载模型
uv run python -m src.modeling.EmpanadaFinetune finetune --iterations 2000 --check-only

# 3. 确认无误后真正训练 (这一步才会加载模型)
uv run python -m src.modeling.EmpanadaFinetune finetune --iterations 2000

# 4. 训练并注册到 ~/.empanada/configs/, 之后可以按名字像内置模型一样引用
uv run python -m src.modeling.EmpanadaFinetune finetune --iterations 2000 --register
```

脂滴把 `--target` 换成 `lipid` 即可（对应 `DropNet_base_v1` + `_ld_masks.tif`）。微调后的模型用
[`EmpanadaPredict.py`](src/modeling/EmpanadaPredict.py) 推理，或把它的 `config_name` 指向新模型名。

### 等价的手写脚本

需要更细的控制（换 `eval_dir`、自定义 `custom_config`、批量跑多组参数）时，也可以直接调
`empanada_core` 的 API：

```python
"""微调 MitoNet_v1 的最小脚本。"""

from pathlib import Path

from empanada_core.finetune import build_finetune_config, finetune_and_register, get_model_info

# 0. 先看标注要求
get_model_info("MitoNet_v1")

# 1. 组装配置
cfg = build_finetune_config(
    model_name="MitoNet_v1_ft",
    train_dir="data/processed/finetune",
    eval_dir=None,
    model_dir="models",
    finetune_model="MitoNet_v1",
    finetune_layer="all",
    iterations=2000,
    patch_size=256,
    custom_config="default config",
)

# 2. 用本地权重, 跳过 zenodo 下载
local_weights = Path("models/MitoNet_v1.pth")
assert local_weights.is_file(), f"找不到基础权重: {local_weights}"
cfg["MODEL"]["model"] = str(local_weights)

# 3. 避开 save_freq=0 的坑 (第 7 节)
params = cfg["TRAIN"]["schedule_params"]
epochs = params["epochs"] if "epochs" in params else cfg["TRAIN"]["epochs"]
assert epochs >= 5, f"epochs={epochs} 太小, 会得到 save_freq={cfg['TRAIN']['save_freq']}"

# 4. 训练并注册
yaml_path = finetune_and_register(cfg)
print("微调完成:", yaml_path)
```

`train` 有命令行入口（`finetune` 的入口在 `src/modeling` 里，见上）：

```bash
uv run python -m empanada_core.train <config.yaml>
```

## 10. 常见坑速查

| 现象                                                            | 原因与处理                                                                                           |
| --------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `Found 0 image subdirectories with 0 images`                  | `train_dir` 层级不对。要传 `<train_dir>/<source_dataset>/images` 中 `<train_dir>` 那一层       |
| `ZeroDivisionError: integer division or modulo by zero`       | `epochs < 5` 导致 `save_freq == 0`。提高 `iterations`（见第 7 节）                             |
| `Need 16 images for batch size 16, got N`                     | 图片数少于`batch_size`。加图或调小 `TRAIN.batch_size`                                            |
| 训练图与掩码对不上                                              | `_BaseDataset` 用 `images/` 和 `masks/` 两次独立 `glob` 的**下标**配对，文件名必须一致 |
| 掩码全被当成一个实例                                            | 掩码是 0/255 二值图。要存成每个实例一个整数 ID 的标号图                                              |
| `Must be more than 1 label class!`                            | 单类模型误用了`PanopticDataset`。检查模型配置的 `FINETUNE.dataset_class`                         |
| 微调后模型重名                                                  | `add_new_model()` 发现同名会自动改名成 `<name>New`，在 `~/.empanada/configs/` 里确认实际文件名 |
| `OSError: [Errno 2] No such file or directory: '<model_dir>'` | `model_dir` 的父目录不存在时 `os.mkdir()` 不会递归创建。先手动建好目录                           |
| macOS 上卡住                                                    | 包导入时已强制`spawn` 启动方式；`num_workers` 会被设成 0，属预期行为                             |
| 显存不足                                                        | 调小`TRAIN.batch_size`、`patch_size`，或关掉 `TRAIN.amp=False` 试试                            |

## 11. 相关模块索引

| 模块                          | 你会用到的接口                                                                               |
| ----------------------------- | -------------------------------------------------------------------------------------------- |
| `src.modeling.EmpanadaFinetune` | `EmpanadaFinetuner` 基类 + `MitoFinetuner` / `LipidFinetuner`（`prepare` / `check` / `run`） |
| `src.modeling.EmpanadaPredict`  | `EmpanadaPredictor` 基类 + `MitoPredictor` / `LipidPredictor`、`decode_instances`       |
| `empanada_core.finetune`    | `get_model_info`、`build_finetune_config`、`run_finetuning`、`finetune_and_register` |
| `empanada_core.train`       | `build_train_config`、`run_training`、`train_and_register`（从零训练/多类训练）        |
| `empanada_core.patches`     | `pick_patches`、`store_dataset`、`patch_suffices`、`flipbook_suffices`               |
| `empanada_core.inference`   | `Engine2d`、`Engine3d`                                                                   |
| `empanada_core.pipeline`    | `create_engine_2d`、`create_engine_3d`、ROI 裁剪、批量切片                               |
| `empanada_core.metrics`     | `compute_pixel_metrics`、`compute_instance_metrics`                                      |
| `empanada_core.labels`      | `morph_labels`、`merge_labels`、`split_labels`、`delete_labels`（修标注）            |
| `empanada_core.label_stats` | `apply_label_filter`、`count_labels_*`（过滤小目标、统计）                               |
| `empanada_core.model_io`    | `export_model`、`import_model`、`archive_model`                                        |

更详细的模块对照表与"与上游的语义差异"见 [`empanada_core/README.md`](empanada_core/README.md)。
