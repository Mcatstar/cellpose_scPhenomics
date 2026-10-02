# cellpose_scPhenomics

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Cellpose-based single-cell phenomics pipeline for microscopy imaging.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── images
│   ├── processed      <- The final, canonical data sets for modeling.
│   │   ├── train
│   │   └── test
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         src and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling              
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models        
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

---

# cellpose_scPhenomics

Cllpose-based single-cell phenomics pipeline for microscopy imaging.

把三篇文献的指标体系组合成一个完整的分析流程，需要分三个层次：**分割层 → 形态层 → 互作层**。下面给出完整的 Python 分析脚本，输入是 Mito/LD mask，输出是一份包含全部指标的 CSV 表格。

---

## 指标整合框架

| 层次 | 指标来源 | 具体指标 |
|---|---|---|
| **A. 全局计数** | 三篇共有 | Mito/LD 总数、总面积、平均面积、总周长 |
| **B. 形态参数** | 人肝活检 | 面积、周长、圆度、长宽比、圆度、实度 |
| **C. 伸长因子** | DeepContact | perimeter²/(4π×area) |
| **D. 大小分类** | T2DM-MASLD | LD: <2 μm vs >2 μm |
| **E. 接触宽度分布** | DeepContact | 0–100 nm，10 nm bin，接触长度/比例 |
| **F. 接触比例** | DeepContact + T2DM | 接触长度/Mito周长、接触长度/LD周长、接触Mito数/总Mito数 |
| **G. PDM/CM分类** | T2DM + 人肝活检 | 按是否接触LD分类，分别统计形态 |
| **H. 接触模式** | T2DM | 单Mito-单LD vs 多Mito-多LD成簇 |
| **I. 相关性** | 人肝活检 | PDM计数 vs LD计数、形态与接触的相关 |
| **J. 效应量** | 人肝活检 | Cohen's d（PDM vs CM） |

---

## 输出文件说明

| 文件 | 内容 | 对应指标来源 |
|---|---|---|
| `global_metrics.csv` | 每张图的全局汇总 | 三篇共用 |
| `mito_morphology.csv` | 每个线粒体的形态参数 + PDM/CM 标签 | 人肝活检 + T2DM |
| `ld_morphology.csv` | 每个脂滴的形态参数 + 大小分类 | T2DM |
| `contact_events.csv` | 每个接触事件的宽度 bin 和接触长度 | DeepContact |
| `pdm_cm_classification.csv` | 每个线粒体的 PDM/CM 分类 | T2DM + 人肝活检 |
| `contact_patterns.csv` | 每个接触事件的模式分类 | T2DM |

---

## 关键参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `pixel_size_nm` | 10.0 | EM 图像分辨率，1 像素 = 10 nm |
| `max_distance_nm` | 100.0 | 接触宽度分析的最大距离 |
| `bin_width_nm` | 10.0 | 宽度分 bin 的步长 |
| `contact_threshold_nm` | 30.0 | 定义"接触"的距离阈值（LD-Mito 常用 30 nm） |

如果你的图像分辨率不是 10 nm/pixel，**必须改 `pixel_size_nm`**，否则所有长度和面积单位都会错。

---

## 与现有工作流的衔接

```
EmpanadaPredict.py  →  *_mito_masks_pred.tif
                    →  *_ld_masks_pred.tif
                              ↓
analyze_mito_ld.py  →  global_metrics.csv
                    →  mito_morphology.csv
                    →  ld_morphology.csv
                    →  contact_events.csv
                    →  pdm_cm_classification.csv
                    →  contact_patterns.csv
                              ↓
统计脚本（t 检验、ANOVA、Cohen's d、MIC）
```