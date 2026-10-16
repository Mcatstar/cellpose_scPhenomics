# cellpose_scPhenomics

基于 Cellpose / Empanada 的电镜线粒体-脂滴互作单细胞表型组学流程。

输入 C / M / T 三组 EM 图像，输出线粒体（mito）与脂滴（ld）的分割掩码、形态学参数、接触位点量化、分组统计与可视化图表。

---

## 环境与依赖

```bash
uv sync
```

主要依赖：

| 类别 | 包 |
|---|---|
| 分割 | cellpose, empanada-dl, torch |
| 图像 | tifffile, opencv-python, scikit-image |
| 数值 | numpy, scipy, pandas |
| 绘图 | matplotlib, seaborn |
| OCR | rapidocr（读标尺） |
| 几何 | cztile（分块推理）, zarr, dask（3D 跟踪与共识） |

命令行入口统一使用 `uv run python -m <module>`。

---

## 目录结构

```
├── data
│   ├── raw                    原始数据
│   ├── images                 原始 EM 图像（C / M / T 三组）
│   ├── external               人工标注的 *_mito_labels.tif / *_ld_labels.tif
│   ├── interim                dataset.py 裁掉黑边后的图与掩码、metadata.csv
│   └── processed
│       ├── train/mito         训练集（图像 + 掩码）
│       ├── train/ld
│       ├── test/mito, test/ld 验证集
│       ├── infer              仅含图像
│       ├── predictions        推理输出（*_mito_masks_pred.tif / *_ld_masks_pred.tif）
│       ├── finetune           Empanada 微调数据（prepare --split train）
│       └── finetune_eval      Empanada 验证数据（prepare --split test）
│
├── models                     预训练权重与微调产物
│   ├── MitoNet_v1.pth
│   ├── DropNet_base_v1.pth
│   └── <model_name>.pth / .yaml
│
├── empanada                   内嵌的 empanada-dl 上游库
├── empanada_core              剥离 GUI 后的核心逻辑
│   ├── inference.py           Engine2d / Engine3d
│   ├── finetune.py            微调配置组装与训练循环
│   ├── patches.py             大图切补丁与数据集装配
│   ├── pipeline.py            引擎构建、ROI 裁剪、分块推理
│   ├── labels.py              形态学与标签操作
│   ├── metrics.py             像素与实例 IoU / Dice
│   └── configs/               4 个内置模型配置
│
├── src
│   ├── config.py              路径与日志
│   ├── dataset.py             裁黑边 + OCR 读标尺 + 组装 train/test/infer
│   ├── postproc.py            形态学与接触指标
│   ├── visualize.py           分组统计与可视化
│   └── modeling
│       ├── CellposeTrain.py
│       ├── CellposePredict.py
│       ├── EmpanadaFinetune.py
│       └── EmpanadaPredict.py
│
└── reports                    分析输出（CSV 与图表）
    ├── global_metrics.csv
    ├── mito_morphology.csv
    ├── ld_morphology.csv
    ├── contact_pairs.csv
    ├── contact_width_bins.csv
    ├── contact_patterns.csv
    ├── group_summary.csv
    ├── stat_tests_between_groups.csv
    ├── stat_tests_pdm_vs_cm.csv
    └── figures/
```

---

## 完整工作流

四个阶段，串联运行：

```
原始 EM 图
    │
    ▼
1. dataset.py          裁黑边 + OCR 读标尺 → interim/ + metadata.csv
    │
    ▼
2. EmpanadaPredict.py  2D 推理 → predictions/*_masks_pred.tif
    │
    ▼
3. postproc.py         形态学 + 接触分析 → 6 个 CSV
    │
    ▼
4. visualize.py        分组统计 + 可视化 → figures/*.png
```

### 1. 数据准备

```bash
uv run python -m src.dataset
```

功能：

- 遍历 `data/images/<group>/*.tif`
- 自动裁掉底部黑条（时间戳、比例尺信息栏）
- 用 RapidOCR 识别比例尺文本，检测刻度线，计算每张图的 `nm_per_px`
- 输出到 `data/interim/`，同时写入 `metadata.csv`（含分辨率）和 `train_test_split.csv`
- 复制到 `data/processed/{train,test,infer}/`

关键输出 `metadata.csv`：

| 列 | 说明 |
|---|---|
| `origin_path`, `origin_name` | 原始路径与文件名 |
| `img_name` | 裁切后的图名（如 `C-1-2.5-1_img.tif`） |
| `origin_shape`, `shape` | 原始与裁切后尺寸 |
| `Magnification` | 放大倍率 |
| `nm_per_px` | 每像素物理尺寸（nm） |
| `dtype` | 原始位深 |

### 2. 推理

```bash
uv run python -m src.modeling.EmpanadaPredict --tile-size 1024
```

功能：

- 加载 `MitoNet_v1` 与 `DropNet_base_v1`
- 对 `data/processed/infer/*_img.tif` 逐张推理
- 输出连续实例标签图（0=背景，1~n 为实例）到 `data/processed/predictions/`

关键参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--tile-size` | 0 | 0 表示整图推理；大图设 512 或 1024 降低显存 |
| `--confidence-thr` | 配置值 | 覆盖模型的置信度阈值 |
| `--gpu` | True | 是否使用 GPU |

输出命名与 dataset 阶段对齐：`<name>_mito_masks_pred.tif` / `<name>_ld_masks_pred.tif`。

### 3. 后处理与量化

```bash
uv run python -m src.postproc --contact-threshold-nm 30
```

功能：

- 从 `metadata.csv` 按图名读取每张图的 `nm_per_px`
- 对每张图的 mito / ld mask 计算形态参数
- 用 `scipy.ndimage.distance_transform_edt` 向量化计算 Mito-LD 接触
- 按宽度 bin（0–100 nm，步长 10 nm）统计接触长度
- 将线粒体分类为 PDM（接触脂滴）与 CM（不接触）

关键参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--contact-threshold-nm` | 30.0 | 接触判定距离阈值 |
| `--max-distance-nm` | 100.0 | 宽度 bin 分析的最大距离 |
| `--bin-width-nm` | 10.0 | 宽度 bin 步长 |
| `--metadata-path` | `data/interim/metadata.csv` | 分辨率来源 |

### 4. 统计与可视化

```bash
uv run python -m src.visualize
```

功能：

- 从图名前缀推断分组（C / M / T）
- 输出分组汇总表、组间 Mann-Whitney U 检验、PDM vs CM 检验
- 生成 10 张主要图表

---

## 核心指标

按三篇文献的指标体系组合，分三个层次。

### 形态学指标

每个实例（mito 或 ld）一行：

| 指标 | 定义 | 来源 |
|---|---|---|
| `area_um2` | 面积（μm²） | 人肝活检 |
| `perimeter_um` | 周长（μm） | 人肝活检 |
| `circularity` | 4π·A / P² | 人肝活检 |
| `elongation` | P² / (4π·A) | DeepContact |
| `aspect_ratio` | 长轴 / 短轴 | 人肝活检 |
| `roundness` | 4A / (π·长轴²) | 人肝活检 |
| `solidity` | 面积 / 凸包面积 | 人肝活检 |
| `equivalent_diameter_um` | 等效直径 | T2DM-MASLD |

### 接触指标

每对（mito, ld）一行：

| 指标 | 定义 | 来源 |
|---|---|---|
| `min_dist_nm` | 双方边界的最小距离 | DeepContact |
| `is_contact` | 距离 ≤ 阈值 | DeepContact |
| `contact_length_um` | 该对的总接触长度 | DeepContact |
| `bin_0_10nm_um` 至 `bin_90_100nm_um` | 各宽度区间的接触长度 | DeepContact |

### PDM / CM 分类

| 类别 | 定义 | 功能倾向 |
|---|---|---|
| PDM（Peridroplet Mitochondria） | 边界与脂滴距离 ≤ 30 nm | 促脂滴扩张（合成代谢） |
| CM（Cytoplasmic Mitochondria） | 无脂滴接触 | 促脂滴消耗（分解代谢） |

### 单实例接触比例

截图文献强调的两个核心指标：

| 文件 | 字段 | 含义 |
|---|---|---|
| `mito_morphology.csv` | `contact_length_um` | 该线粒体与所有 LD 的接触长度之和 |
| `mito_morphology.csv` | `contact_ratio` | 接触长度 / 该线粒体周长 |
| `ld_morphology.csv` | `contact_length_um` | 该脂滴与所有 Mito 的接触长度之和 |
| `ld_morphology.csv` | `contact_ratio` | 接触长度 / 该脂滴周长 |

### 脂滴大小分类

| 类别 | 阈值 |
|---|---|
| small | 等效直径 < 2 μm |
| large | 等效直径 ≥ 2 μm |

### 接触模式

| 模式 | 定义 |
|---|---|
| `single_mito_single_ld` | 一个线粒体只接触一个脂滴 |
| `multi_contact` | 一个线粒体接触 ≥ 2 个脂滴 |

---

## 输出文件

### CSV

| 文件 | 粒度 | 内容 |
|---|---|---|
| `global_metrics.csv` | 一行一图 | 计数、面积/周长汇总、接触比、PDM/CM 比 |
| `mito_morphology.csv` | 一行一 mito | 形态 + PDM/CM + 单实例接触长度与比例 |
| `ld_morphology.csv` | 一行一 ld | 形态 + small/large + 单实例接触长度与比例 |
| `contact_pairs.csv` | 一行一 (mito, ld) | 最小距离、接触长度、各宽度 bin |
| `contact_width_bins.csv` | 一行一 (图, bin) | 各宽度 bin 的总接触长度 |
| `contact_patterns.csv` | 一行一 mito | 接触 LD 数量、模式分类 |
| `group_summary.csv` | 一行一 (指标, 组) | mean/std/sem/median/Q1/Q3/n |
| `stat_tests_between_groups.csv` | 一行一 (指标, 组对) | Mann-Whitney U p 值 + Cohen's d |
| `stat_tests_pdm_vs_cm.csv` | 一行一形态指标 | PDM vs CM 的差异检验 |

### 图表

| 图 | 内容 |
|---|---|
| `global_metrics_boxplots.png` | 计数类指标的组间对比（6 个子图） |
| `morphology_boxplots.png` | 形态类指标的组间对比（6 个子图） |
| `contact_width_spectrum.png` | 0–100 nm 接触宽度频谱（C/M/T 三组叠加） |
| `pdm_vs_cm.png` | PDM vs CM 的 6 个形态指标小提琴图 |
| `ld_size_distribution.png` | LD 面积直方图 + 等效直径箱线（含 2 μm 阈值线） |
| `pdm_vs_ld_count.png` | PDM 计数 vs LD 计数散点（含 Pearson r） |
| `contact_length_per_pair.png` | 每对接触的绝对长度分布 |
| `contact_ratio_per_instance.png` | 单实例的相对接触比例（mito 与 ld 分开） |
| `pdm_cm_contact_metrics.png` | PDM vs CM 的接触长度与接触比例 |
| `contact_absolute_vs_relative.png` | 每张图绝对接触长度 vs 相对比例 |

---

## 常见任务

### 快速跑一遍全流程

```bash
uv run python -m src.dataset
uv run python -m src.modeling.EmpanadaPredict --tile-size 1024
uv run python -m src.postproc
uv run python -m src.visualize
```

### 只跑一张图检查

```bash
uv run python -m src.postproc --pred-dir data/processed/predictions_sample
```

### 调整接触阈值

接触阈值对 PDM 分类影响很大。文献常用 30 nm，但 mask 级分析受分割边界内缩影响，可以试 40–50 nm：

```bash
uv run python -m src.postproc --contact-threshold-nm 50 --output-dir reports/thr50
```

对比多份 `global_metrics.csv` 的 `n_pdm` 和 `contact_ratio_mito`，选择落在生物学合理范围的阈值。

### 微调模型

数据域与预训练集差异较大时，用几十张自己标注的图做微调：

```bash
# 准备数据（train 和 test 分别跑一次）
uv run python -m src.modeling.EmpanadaFinetune prepare --target mito --split train
uv run python -m src.modeling.EmpanadaFinetune prepare --target mito --split test

# 校验配置，不加载模型
uv run python -m src.modeling.EmpanadaFinetune finetune --iterations 2000 --check-only

# 训练并注册
uv run python -m src.modeling.EmpanadaFinetune finetune --iterations 2000 --register
```

脂滴把 `--target` 换成 `lipid`。

### 交互式查看分割结果

推理输出的 mask 是连续整数标签图，可直接用 Fiji/ImageJ 打开。接触区可视化参考 `visualize.py` 的配色方案：

- LD：淡橙 (250, 220, 180)
- PDM：淡蓝 (170, 200, 235)
- 接触区：醒目红 (215, 85, 85)

---

## 注意事项

### 分辨率

`metadata.csv` 里的 `nm_per_px` 是逐图不同的。`postproc.py` 会自动按图名查询，不要用固定的 `--pixel-size-nm` 覆盖。

如果某张图不在 metadata 里，会打 warning 并用兜底值。跑完检查日志，确认没有这类警告。

### 单位换算

所有长度输出 μm，面积输出 μm²。像素到物理单位的换算在 `morphology_table` 和 `contact_table` 内部完成，输出的 CSV 全部是物理单位。

### 接触分析

`postproc.py` 用 `distance_transform_edt` 向量化计算，一张 2464×3296 的图约 1–3 秒。参考 DeepContact 的双重 for 循环实现会慢几百倍，不要抄它的代码。

### 统计检验

- `stat_tests_between_groups.csv` 做了 69 次两两检验，多重比较校正（如 Benjamini-Hochberg FDR）需要在后续分析里自行加上
- Cohen's d 效应量参考：0.2 小 / 0.5 中 / 0.8 大
- 组内样本量很小（每组 2–10 张图）时，p 值和效应量都可能不稳定

### 训练集与验证集

- 按**图像**随机划分 train/test 会高估模型性能，因为同批次、同切片的图高度相似
- 严格评估应按**动物/批次**划分，或做留一交叉验证
- `eval_dir` 用于训练过程中监控过拟合，不作为最终性能报告

### scikit-image 版本

`major_axis_length` / `minor_axis_length` 在 scikit-image 0.26 已改名。代码已兼容新旧名称，若装旧版本也能跑。

### empanada_core

`empanada_core` 是内嵌的 empanada-napari 剥离 GUI 后的核心逻辑，不需要安装 napari / magicgui / qtpy。相关 API 说明见 `empanada_core/README.md`。

---

## 许可

见 `LICENSE`。