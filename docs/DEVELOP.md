# 开发者文档

面向维护者与二次开发者，说明项目的架构、模块职责、接口约定、关键决策与常见陷阱。

---

## 1. 项目背景与目标

### 1.1 生物学问题

量化电镜图像中线粒体（mito）与脂滴（ld）的形态学特征和膜接触位点，研究 C（对照）、M（模型）、T（治疗）三组之间的系统差异。

### 1.2 技术挑战

- EM 图像分辨率高（2563×3296 起），单张处理需秒级
- 线粒体和脂滴需要实例级分割（不是语义分割）
- 膜接触是纳米级现象，需要按物理距离而非像素距离判定
- 每张图的 `nm_per_px` 不同（0.6–11 nm/px），所有指标必须逐图换算
- 三组样本量小（每组 2–10 张图），统计检验需要谨慎
- 人工标注代价高，需要预训练模型 + 微调

### 1.3 设计原则

1. **单位一致**：所有输出 CSV 使用物理单位（μm、μm²），不使用像素
2. **数据流单向**：raw → interim → processed → predictions → reports，每层只读上一层，不反向依赖
3. **函数无状态**：`postproc.py` 和 `visualize.py` 的每个函数独立，不共享全局状态
4. **向量化优先**：距离变换、形态学、分 bin 统计全部用 scipy/skimage，禁止双重 for 循环
5. **可复现**：`metadata.csv` 记录每张图的分辨率，所有下游依赖它
6. **分层解耦**：分割、量化、统计、可视化四层分离，任意一层可以替换

---

## 2. 架构

### 2.1 数据流

```
data/images/<group>/*.tif
        │
        │  src/dataset.py
        │  (裁黑边、OCR 读标尺、8/2 划分)
        ▼
data/interim/
    <group>-<stem>_img.tif
    <group>-<stem>_mito_masks.tif
    <group>-<stem>_ld_masks.tif
    metadata.csv                 ← 关键中间产物
    train_test_split.csv
        │
        │  src/dataset.py (copy)
        ▼
data/processed/
    train/mito/*_img.tif + *_mito_masks.tif
    train/ld/*_img.tif + *_ld_masks.tif
    test/mito, test/ld
    infer/*_img.tif
        │
        │  src/modeling/EmpanadaPredict.py
        ▼
data/processed/predictions/
    <name>_mito_masks_pred.tif   ← 连续实例标签图
    <name>_ld_masks_pred.tif
        │
        │  src/postproc.py
        │  (查 metadata 拿 nm_per_px，逐图换算)
        ▼
reports/
    global_metrics.csv
    mito_morphology.csv
    ld_morphology.csv
    contact_pairs.csv
    contact_width_bins.csv
    contact_patterns.csv
        │
        │  src/visualize.py
        ▼
reports/figures/*.png
reports/group_summary.csv
reports/stat_tests_*.csv
```

### 2.2 分层职责

| 层 | 模块 | 职责 | 不做什么 |
|---|---|---|---|
| 数据层 | `src/dataset.py` | 读原始图、算分辨率、划分数据集 | 不做分割 |
| 分割层 | `src/modeling/EmpanadaPredict.py` | 加载模型、推理、输出标签图 | 不做量化 |
| 量化层 | `src/postproc.py` | 形态学、接触分析、PDM/CM 分类 | 不做统计 |
| 统计层 | `src/visualize.py` | 分组汇总、检验、可视化 | 不读原图 |
| 支撑层 | `empanada_core/` | 引擎、微调、标签操作 | 无 GUI |

### 2.3 依赖方向

```
src/visualize.py  →  reports/*.csv
src/postproc.py   →  data/processed/predictions/
src/modeling/EmpanadaPredict.py  →  empanada_core/  →  empanada/
src/dataset.py    →  data/images/
```

严格单向：上层不 import 下层。`postproc.py` 不 import `visualize.py`，`visualize.py` 不 import `postproc.py`。

---

## 3. 模块详解

### 3.1 `src/config.py`

定义项目根路径与各层目录常量。

| 变量 | 路径 |
|---|---|
| `PROJ_ROOT` | 项目根 |
| `DATA_DIR` | `data/` |
| `RAW_DATA_DIR` | `data/raw/` |
| `IMAGES_DATA_DIR` | `data/images/` |
| `INTERIM_DATA_DIR` | `data/interim/` |
| `PROCESSED_DATA_DIR` | `data/processed/` |
| `EXTERNAL_DATA_DIR` | `data/external/` |
| `MODELS_DIR` | `models/` |
| `PRED_DATA_DIR` | `data/processed/predictions/` |
| `REPORTS_DIR` | `reports/` |
| `FIGURES_DIR` | `reports/figures/` |

所有模块从这里导入路径，禁止硬编码。

### 3.2 `src/dataset.py`

**输入**：`data/images/<group>/*.tif`

**输出**：`data/interim/` 里的裁切图、mask、`metadata.csv`、`train_test_split.csv`

**核心步骤**：

1. 用 `tifffile.TiffFile` 读图与页面元数据（描述、尺寸、位深）
2. 从页面描述里正则提取 `Magnification`
3. 用行中值检测黑条高度：`row_medians > 5` 的最大索引之外是黑条
4. 裁掉黑条，得到 `effective_img`
5. 用 RapidOCR 检测模式识别黑条里的比例尺文本与刻度线
6. 从所有文字 bbox 生成 mask，从二值图扣掉，剩下的白色竖线才是真刻度线
7. 取最左和最右刻度线端点，算 `pixel_distance`
8. `nm_per_px = physical_nm / pixel_distance`
9. 按 8:2 划分 train/test，写 `train_test_split.csv`
10. 复制到 `data/processed/{train,test,infer}/`

**关键函数**：

| 函数 | 作用 |
|---|---|
| `to_uint8(image)` | 任意位深线性映射到 uint8 |
| `main()` | CLI 入口 |

**扩展点**：如果要支持其他显微镜的比例尺格式，修改正则与刻度线检测逻辑。

### 3.3 `src/modeling/EmpanadaPredict.py`

**输入**：`data/processed/infer/*_img.tif`

**输出**：`data/processed/predictions/<name>_{mito,ld}_masks_pred.tif`

**核心逻辑**：

```python
class EmpanadaPredictor:
    def __init__(self, gpu, tile_size, confidence_thr, inference_scale):
        cfg = load_config(configs[self.config_name])
        engine_params = cfg["FINETUNE"]["engine_params"]
        # 关键：用 Engine2d 封装预处理与后处理
        self.engine = Engine2d(cfg, tile_size=tile_size, use_gpu=gpu)

    def predict(self, image):
        pan_seg = self.engine.infer(image)
        # 关键：panoptic 编码 → 连续实例标签
        return decode_instances(pan_seg, self.thing_list, self.label_divisor)
```

**关键函数**：

| 函数 | 作用 |
|---|---|
| `decode_instances(pan_seg, thing_list, label_divisor)` | panoptic 编码转为 1, 2, 3, ... |
| `EmpanadaPredictor.predict(image)` | 单图推理 |

**关键设计**：

- 用 `Engine2d` 而非手动 `PanopticDeepLabEngine.__call__`，因为 `Engine2d` 内部已经处理了归一化、padding、分块
- 分辨率从 `metadata.csv` 按图名查询，不在推理时处理
- `decode_instances` 处理多类别场景，即便当前只有一个 thing_list

### 3.4 `src/postproc.py`

**输入**：`data/processed/predictions/` 里的成对 mask

**输出**：`reports/` 根目录下 6 个 CSV

**核心逻辑**：

```python
def contact_table(mito_mask, ld_mask, pixel_size_nm, ...):
    # 1. 提取 LD 边界
    ld_boundary = _extract_boundary(ld_bin)
    # 2. 距离变换（向量化）
    ld_dist_nm = distance_transform_edt(~ld_boundary) * pixel_size_nm
    # 3. 查每个 mito 边界像素到 LD 的距离
    _, indices = distance_transform_edt(ld_rl == 0, return_indices=True)
    nearest_ld = ld_rl[indices[0], indices[1]]
    ys, xs = np.where(mito_boundary)
    dist_v = ld_dist_nm[ys, xs]
    # 4. 按 (mito, ld) 分组统计
    df = pd.DataFrame({"mito": mito_v, "ld": ld_v, "dist": dist_v})
    for (m, l), g in df.groupby(["mito", "ld"]):
        ...
```

**关键函数**：

| 函数 | 作用 |
|---|---|
| `load_metadata(path)` | 返回 `{stem: nm_per_px}` |
| `_extract_boundary(binary)` | 单像素边界（mask XOR erosion） |
| `morphology_table(mask, pixel_size_nm, kind, image_name)` | 每实例的形态参数 |
| `contact_table(...)` | 每对接触的 min_dist、接触长度、宽度 bin |
| `classify_pdm_cm(mito_mask, pair_df, name)` | PDM/CM 分类 |
| `attach_contact_to_morphology(...)` | 把接触长度 join 回形态表，算 `contact_ratio` |
| `per_mito_summary(pair_df)` | 每个 mito 的接触 LD 数、总接触长度 |
| `detect_contact_patterns(per_mito_df)` | 单接触 vs 多接触模式 |
| `build_global_row(...)` | 每张图的全局汇总 |

**关键设计**：

- **单位换算在函数内完成**：所有入参和出参都是物理单位
- **逐图查分辨率**：`stem_to_nm.get(name)` 从 metadata 拿，不依赖全局参数
- **`attach_contact_to_morphology` 单独抽出**：这是给 `mito_morphology.csv` 和 `ld_morphology.csv` 补 `contact_length_um` 和 `contact_ratio` 两列的关键逻辑，是截图文献核心指标的落地点
- **PANOPTIC 编码不需要额外解码**：EmpanadaPredict 已输出连续实例标签

### 3.5 `src/visualize.py`

**输入**：`reports/*.csv`

**输出**：`reports/figures/*.png`、`reports/group_summary.csv`、`reports/stat_tests_*.csv`

**核心函数**：

| 函数 | 作用 |
|---|---|
| `infer_group(image)` | 从图名前缀推断 C/M/T |
| `load_all(report_dir)` | 读所有 CSV 并加 group 列 |
| `group_summary(df, metrics)` | 按组计算 mean/std/sem/median |
| `test_between_groups(df, metrics)` | 两两 Mann-Whitney U + Cohen's d |
| `test_pdm_vs_cm(mito_df)` | PDM vs CM 形态差异 |
| `plot_group_boxplots(...)` | 计数与形态指标的组间对比 |
| `plot_contact_width_spectrum(bins_df, ...)` | 0–100 nm 接触宽度频谱 |
| `plot_pdm_cm_comparison(mito_df, ...)` | PDM vs CM 形态小提琴图 |
| `plot_ld_size_distribution(ld_df, ...)` | LD 面积分布 |
| `plot_pdm_ld_correlation(global_df, ...)` | PDM 计数 vs LD 计数 |
| `plot_contact_length_per_pair(pairs_df, ...)` | 每对接触的绝对长度 |
| `plot_contact_ratio_per_instance(mito_df, ld_df, ...)` | 单实例的相对接触比例 |
| `plot_pdm_cm_contact_metrics(mito_df, ...)` | PDM vs CM 接触指标 |
| `plot_contact_absolute_vs_relative(global_df, ...)` | 绝对长度 vs 相对比例 |

**关键设计**：

- **分组从图名推断**：`infer_group("C-1-2.5-1") == "C"`，不依赖外部映射表。如果命名规则变化，只改这一个函数
- **每个 plot 函数独立**：任一绘图失败不影响其他图
- **seaborn 调用规范**：`sns.boxplot` 和 `sns.violinplot` 必须同时有 `hue` 和 `legend=False`（新版 seaborn 要求），`hue` 的列名必须和 `x` 一致

### 3.6 `src/modeling/EmpanadaFinetune.py`

**功能**：微调 Empanada 模型，两个子命令

| 子命令 | 作用 |
|---|---|
| `prepare` | 把 `data/processed/{train,test}/<source>/` 转为 empanada 数据集格式 |
| `finetune` | 组装配置并训练 |

**核心类**：

```python
class EmpanadaFinetuner:
    source = "mito"
    base_model = "MitoNet_v1"
    weights_file = "MitoNet_v1.pth"
    mask_suffix = "_mito_masks.tif"

    def prepare(self, split): ...
    def check(self): ...
    def launch(self, register): ...
```

**关键设计**：

- **子类只覆盖类属性**：`MitoFinetuner` 和 `LipidFinetuner` 不重写逻辑
- **`--check-only` 不加载模型**：适合在没有 GPU 的机器上先校验配置
- **epochs 守卫**：`check()` 里检测 `epochs < 5` 直接报错，避免 `save_freq == 0` 导致 `ZeroDivisionError`

### 3.7 `empanada_core`

从 `empanada-napari` 剥离 GUI 后的核心逻辑。本项目的建模脚本通过它调用 Empanada。

| 模块 | 作用 |
|---|---|
| `inference.py` | `Engine2d` / `Engine3d`，封装预处理、分块、后处理 |
| `finetune.py` | `build_finetune_config` / `run_finetuning` / `finetune_and_register` |
| `patches.py` | 大图切补丁、装配数据集 |
| `pipeline.py` | 引擎构建、ROI 裁剪、批量切片 |
| `labels.py` | 形态学操作、标签合并/拆分 |
| `label_stats.py` | 小目标过滤、标签计数 |
| `metrics.py` | 像素与实例 IoU / Dice / Hungarian 匹配 |
| `model_io.py` | `.empanada` 模型包导出/导入/归档 |

详见 `empanada_core/README.md`。

---

## 4. 数据约定

### 4.1 掩码格式

**连续整数标签图**（不是 0/255 二值图）：

| 值 | 含义 |
|---|---|
| 0 | 背景 |
| 1, 2, 3, ... | 每个实例的唯一 ID |

EmpanadaPredict 输出已经满足这个格式。Cellpose 输出也是。

**微调时**：掩码要求一样，`SingleClassInstanceDataset` 用 `mask > 0` 得到语义，用正整数区分实例。

### 4.2 文件命名

| 阶段 | 命名 | 示例 |
|---|---|---|
| 原始 | `<stem>.tif` | `1-2.5-1.tif` |
| interim | `<group>-<stem>_img.tif` | `C-1-2.5-1_img.tif` |
| interim masks | `<group>-<stem>_{mito,ld}_masks.tif` | `C-1-2.5-1_mito_masks.tif` |
| 推理输出 | `<group>-<stem>_{mito,ld}_masks_pred.tif` | `C-1-2.5-1_mito_masks_pred.tif` |

`<group>` 是 C / M / T 之一，供 `infer_group` 解析。

### 4.3 metadata.csv 字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `origin_path` | str | 相对原始目录的路径 |
| `origin_name` | str | 原始文件名 |
| `img_name` | str | 裁切后的图名，如 `C-1-2.5-1_img.tif` |
| `origin_shape` | tuple | 裁切前形状 |
| `shape` | tuple | 裁切后形状 |
| `width_px` | int | 图像宽度（像素） |
| `height_px` | int | 裁切后高度 |
| `Magnification` | float | 放大倍率（如 2.5） |
| `nm_per_px` | float | 每像素物理尺寸 |
| `dtype` | str | 原始位深 |

**禁止修改字段名**：`postproc.py` 的 `load_metadata` 硬编码 `img_name` 和 `nm_per_px`。

---

## 5. 关键决策

### 5.1 为什么用 Empanada 而不是 Cellpose

| 维度 | Cellpose 3 | Empanada |
|---|---|---|
| EM 原生支持 | 需微调 | MitoNet / DropNet 直接可用 |
| 3D 支持 | 逐切片 | 原生 3D |
| 预训练模型 | 通用光镜 | EM 专用 |
| 微调数据量 | 数百张 | 数十张 |

Empanada 更适合 EM 数据。Cellpose 保留作为对照与备选。

### 5.2 为什么接触分析不用 DeepContact 的代码

DeepContact 的 `calContactDist_range_10pix_min` 用双重 for 循环遍历 1024×1024，一张图几分钟。我们用 `scipy.ndimage.distance_transform_edt` 向量化实现，一张图 1–3 秒。**算法思想抄，代码重写**。

### 5.3 为什么用物理单位而非像素

不同放大倍率的 `nm_per_px` 差 5 倍以上。同一批数据混用像素单位会导致跨图比较失真。所有输出 CSV 使用 μm 和 μm²。

### 5.4 为什么不用 `--pixel-size-nm` 全局参数

`metadata.csv` 每张图的 `nm_per_px` 不同，全局参数会让低倍率图的面积被高估。代码内部按图名查询。

### 5.5 为什么 `attach_contact_to_morphology` 单独抽出

截图文献强调的两个核心指标（单个线粒体接触比、单个脂滴接触比）需要在**实例级**形态表里。抽成独立函数避免逻辑重复，也方便单元测试。

### 5.6 为什么 PDM/CM 阈值默认 30 nm

三篇文献都用 30 nm。但 mask 级分析受分割边界内缩影响，可调至 40–50 nm。参数已暴露为 `--contact-threshold-nm`。

### 5.7 为什么 `visualize.py` 不合并到 `postproc.py`

- 分层解耦，量化脚本不依赖 matplotlib
- `postproc.py` 可在无图形环境（远程服务器）跑
- 可视化失败不影响量化结果

---

## 6. 开发规范

### 6.1 代码风格

- Python 3.10+
- 类型标注：所有公开函数的参数和返回值
- 禁止裸 `except:`，至少 `except Exception:`
- 路径一律用 `pathlib.Path`，不用字符串拼接
- 日志用 `loguru.logger`，禁止 `print`

### 6.2 命名

| 类型 | 规范 | 示例 |
|---|---|---|
| 模块 | 小写下划线 | `postproc.py` |
| 类 | PascalCase | `EmpanadaPredictor` |
| 函数 | 小写下划线 | `load_metadata` |
| 私有函数 | 前缀下划线 | `_extract_boundary` |
| 常量 | 大写下划线 | `DEFAULT_CONTACT_THRESHOLD_NM` |

### 6.3 依赖管理

用 `uv`。新增依赖：

```bash
uv add <package>
uv add --dev <package>  # 开发依赖
```

禁止手动编辑 `pyproject.toml`。

### 6.4 类型检查

`pyrightconfig.json` 已配置。提交前跑：

```bash
uv run pyright src/
```

Pylance 的严格类型警告（如 `reportArgumentType`）尽量修，实在无法修的加 `# type: ignore` 或 `# pyright: ignore`。

### 6.5 数据文件

- **禁止提交** `data/` 和 `models/` 下的内容
- **可以提交** `reports/*.csv` 和 `reports/figures/*.png`（论文素材）
- 大文件用 Git LFS 或外部存储

### 6.6 提交信息

```
<type>: <subject>

<body>
```

`type` 取值：`feat` / `fix` / `docs` / `refactor` / `test` / `chore`。

---

## 7. 常见陷阱

### 7.1 seaborn 新版 API

`palette` 必须同时指定 `hue`，且 `hue` 的列名要和 `x` 一致：

```python
# 正确
sns.boxplot(data=df, x="group", y=metric, hue="group",
            palette=GROUP_COLORS, legend=False)

# 错误：hue 未指定
sns.boxplot(data=df, x="group", y=metric, palette=GROUP_COLORS)

# 错误：hue 和 x 列名不一致
sns.boxplot(data=df, x="class", y=metric, hue="group",
            palette={"CM": ..., "PDM": ...})  # ValueError
```

### 7.2 matplotlib 中文字体

默认字体 Arial 不含中文字形。要么改英文标签，要么在 `_setup_plot_style` 里指定：

```python
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
```

### 7.3 scikit-image 0.26 API 变更

`major_axis_length` 改名为 `axis_major_length`。代码用 `hasattr` 兼容：

```python
if hasattr(prop, "axis_major_length"):
    major_px = float(prop.axis_major_length)
else:
    major_px = float(prop.major_axis_length)
```

### 7.4 metadata 缺失时的兜底

`FALLBACK_NM_PER_PX = 0.6154` 只在 metadata 找不到某图时用。跑完检查日志：

```
[WARNING] <name> 不在 metadata，使用兜底 0.6154 nm/px
```

如果有这种 warning，说明该图的物理单位可能是错的，需要修 dataset.py 或 metadata.csv。

### 7.5 接触阈值与 PDM 比例

30 nm 在 mask 级分析里偏严。如果发现：

- `n_pdm == 0` 或 PDM 比例 < 1%
- `contact_width_bins.csv` 里 0–10 nm 全是 0

先试 `--contact-threshold-nm 50`，对比 PDM 数和接触比例。如果 50 nm 才合理，说明分割边界内缩了约 20 nm。

### 7.6 微调 epochs 计算

```
epochs = iterations // (n_imgs // batch_size)
```

`epochs < 5` 会导致 `save_freq == 0`，训练时抛 `ZeroDivisionError`。保证：

```
iterations >= 5 * (n_imgs // batch_size)
```

### 7.7 train_dir 层级

Empanada 要求 `<train_dir>/<source_dataset>/images/`。用 `prepare` 时：

- 路线 A（`EmpanadaFinetune prepare`）：`train_dir = data/processed/finetune`
- 路线 C（`patches.store_dataset`）：`train_dir = data/processed/finetune/mito`

两者不能混。混了会报 `Found 0 image subdirectories`。

### 7.8 Empanada 的 `engine()` vs `engine.infer()`

- `PanopticDeepLabEngine.infer(tensor)` 返回 dict（`sem_logits`、`ctr_hmp`、`offsets`），未后处理
- `PanopticDeepLabEngine.__call__(tensor)` 返回完整的 pan_seg 标签图

用 `Engine2d` 时内部已处理，用户调用 `engine.infer(image)` 返回的就是 pan_seg。

### 7.9 Panoptic 编码

`pan_seg` 的值是 `class_id * label_divisor + instance_id`。MitoNet/DropNet 单类场景下 `class_id=1`、`label_divisor=1000`：

- 背景 = 0
- 第 1 个实例 = 1001
- 第 2 个实例 = 1002

`decode_instances` 把它重编码为 1, 2, 3, ...。

### 7.10 距离变换的 padding 影响

`distance_transform_edt` 对边界像素的处理是假设图像外无穷远。如果 mask 触到图像边缘，边缘像素的距离会被高估。当前流程不做特殊处理，实际影响很小。

---

## 8. 扩展指南

### 8.1 支持新的细胞器

1. 在 `empanada_core/configs/` 添加模型配置 YAML
2. 在 `EmpanadaPredict.py` 添加对应的 Predictor 子类
3. 在 `postproc.py` 里复用 `morphology_table`，接触分析按需扩展
4. 在 `visualize.py` 里添加对应的可视化函数

### 8.2 支持 3D 数据

- 推理：用 `empanada_core.inference.Engine3d`（正交平面 + 跟踪共识）
- 量化：`skimage.measure.regionprops` 有 3D 版本
- 接触：`distance_transform_edt` 支持 3D
- 输出：CSV 字段需要加 `centroid_z`、`bbox_3d`

### 8.3 支持新的统计方法

`visualize.py` 的 `test_between_groups` 用 Mann-Whitney U，可替换为：

- t 检验：`scipy.stats.ttest_ind`
- 混合效应模型：`statsmodels`，按图加随机效应
- 多重比较校正：`statsmodels.stats.multitest.multipletests`

### 8.4 支持新的分组规则

如果图名命名规则变化，只改 `infer_group` 一个函数：

```python
def infer_group(image: str) -> str:
    # 旧：从 "C-1-2.5-1" 取 "C"
    # 新：从 "exp_C_rep1" 取 "C"
    return image.split("_")[1]
```

### 8.5 添加新的接触指标

`contact_table` 返回的 `pair_df` 是核心数据结构。新增指标（如接触曲率、接触数量密度）：

1. 在 `contact_table` 内部或之后加一列
2. 在 `build_global_row` 里聚合
3. 在 `visualize.py` 里加绘图函数

---

## 9. 调试与测试

### 9.1 单图调试

跑单张图不走全流程：

```python
from pathlib import Path
from src.modeling.EmpanadaPredict import MitoPredictor, LipidPredictor
from tifffile import imread

img = imread("data/processed/infer/C-1-2.5-1_img.tif")
mito = MitoPredictor(gpu=False)
mask = mito.predict(img)
print(mask.shape, mask.dtype, mask.max())
```

### 9.2 量化调试

跳过推理，直接对已有 mask 跑量化：

```python
from src.postproc import contact_table, morphology_table
from tifffile import imread

mito = imread("data/processed/predictions/C-1-2.5-1_mito_masks_pred.tif")
ld = imread("data/processed/predictions/C-1-2.5-1_ld_masks_pred.tif")
pair_df, bin_df = contact_table(mito, ld, pixel_size_nm=4.386, image_name="C-1-2.5-1")
print(pair_df.head())
```

### 9.3 阈值敏感性分析

跑不同接触阈值，对比 PDM 比例：

```bash
for thr in 20 30 40 50 60; do
    uv run python -m src.postproc --contact-threshold-nm $thr \
        --output-dir reports/thr_$thr
done
```

然后对比 5 份 `global_metrics.csv` 的 `pdm_ratio`，选择落在生物学合理范围（5%–30%）的阈值。

### 9.4 日志排查

跑完 `postproc.py` 后检查：

1. 有没有 `不在 metadata` 的 warning
2. `global_metrics.csv` 里的 `mito_mean_area_um2` 量级（应在 0.1–2 μm²）
3. `n_pdm` 是否在 `(0, n_mito)` 之间
4. `contact_width_bins.csv` 里 `bin_0_10nm_um` 是否有非零值

### 9.5 单元测试

关键函数建议加测试：

| 函数 | 测试内容 |
|---|---|
| `decode_instances` | 输入 panoptic 编码，输出连续标签 |
| `attach_contact_to_morphology` | 验证 contact_ratio 计算正确 |
| `load_metadata` | 验证键名匹配 |
| `infer_group` | 验证 C/M/T 解析 |

---

## 10. 性能

### 10.1 推理

| 配置 | 单图耗时 |
|---|---|
| MitoNet_v1, CPU, 2464×3296 | 约 4 分钟 |
| MitoNet_v1, GPU, 2464×3296 | 约 10 秒 |
| MitoNet_v1_mini, GPU | 约 5 秒 |
| DropNet_base_v1, GPU | 约 8 秒 |

建议在 GPU 上跑。无 GPU 时用 mini 模型。

### 10.2 量化

| 步骤 | 单图耗时 |
|---|---|
| 形态学（`regionprops`） | 约 0.5 秒 |
| 接触分析（距离变换） | 约 1–3 秒 |
| 可视化（单张 PNG） | 约 3–5 秒 |

57 张图全流程约 5 分钟。

### 10.3 优化建议

- 大图用 `--tile-size 1024` 降低显存峰值
- 批处理时两个模型共用一份 GPU 上下文
- 可视化可以只对特定图跑，不必全量

---

## 11. 依赖版本

| 包 | 版本 | 说明 |
|---|---|---|
| Python | 3.10+ | 类型标注、`str.removesuffix` |
| numpy | 1.22+ | `empanada-dl` 要求 |
| scipy | 1.7+ | `distance_transform_edt` |
| scikit-image | 0.20+ | `regionprops` |
| pandas | 2.0+ | `groupby`、`map` |
| matplotlib | 3.7+ | 绘图 |
| seaborn | 0.13+ | 新版 palette API |
| tifffile | 2023+ | TIFF 读写 |
| torch | 2.0+ | Empanada 依赖 |
| empanada-dl | 0.1.7 | 上游 |
| rapidocr | 1.3+ | 读标尺 |
| zarr / dask / cztile | 最新 | 3D 与分块 |

版本冲突时优先满足 `empanada-dl`。

---

## 12. 参考

| 主题 | 文件 |
|---|---|
| 项目总览 | `README.md` |
| Empanada 微调教程 | `README.md` 的「Empanada 模型微调」章节 |
| empanada_core API | `empanada_core/README.md` |
| 开发规范 | `docs/DevelopsRules.md` |
| 类型检查配置 | `pyrightconfig.json` |
| 相关文献 | 见 README 的「相关文献」 |

---
