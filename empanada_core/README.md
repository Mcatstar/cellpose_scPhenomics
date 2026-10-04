# empanada_core

`empanada-napari` 插件的**非 GUI 核心逻辑**。

本包由 [https://github.com/volume-em/empanada-napari](https://github.com/volume-em/empanada-napari) 的 `empanada_napari` 改造而来,
删除了全部 napari / magicgui / qtpy 依赖与 GUI 线程装饰器, 只保留可复用的计算逻辑,
以便在不安装任何 GUI 库的环境下调用。

## 模块对照表

| 模块               | 原文件                                                    | 保留内容                                                                                                                                       | 删除内容                                              |
| ------------------ | --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| `inference.py`   | `inference.py`                                          | `Engine2d`、`Engine3d`、`instance_relabel`、`stack_postprocessing`、`tracker_consensus`                                              | 2 个`@thread_worker` 装饰器                         |
| `multigpu.py`    | `multigpu.py`                                           | `MultiGPUEngine3d`、`main_worker` (DDP)                                                                                                    | 未使用的`thread_worker` 导入                        |
| `pipeline.py`    | `_slice_inference.py`、`_volume_inference.py`         | `create_engine_2d`、`create_engine_3d`、ROI 裁剪、批量切片、正交三轴驱动、store url 命名                                                   | 全部 widget 类、`viewer` 取层、结果展示、layer 创建 |
| `utils.py`       | `utils.py`                                              | `Preprocessor`、`load_model_to_device`、`get_configs`、`add_new_model`、`abspath`、`valid_url_or_file`、下载重试                   | `enable_layer_rename_refresh` (viewer 事件耦合)     |
| `train.py`       | `train.py`、`_train.py`                               | 训练循环 +`build_train_config`、`run_training`、`train_and_register`                                                                     | `@magicgui` 表单、worker 接线                       |
| `finetune.py`    | `finetune.py`、`_finetune.py`                         | 微调循环 +`build_finetune_config`、`run_finetuning`、`finetune_and_register`、`get_model_info`                                         | 同上                                                  |
| `patches.py`     | `_pick_patches.py`                                      | `pick_patches`、`_pick_patches` / `_pick_paired_patches` / `_pick_flipbooks` / `_pick_paired_flipbooks`、`store_dataset`、后缀生成 | 点选坐标转换、视图显示、worker                        |
| `tiles.py`       | `_create_tiles.py`                                      | `chop_up_2d_im_into_patches`、`put_patches_back_together` (逐字保留)                                                                       | `create_tiles` / `merge_tiles` 包装               |
| `export.py`      | `_export_batch_segs.py`                                 | `export_batch_segs`、`_get_impaths_from_dask`                                                                                              | `viewer.dims` 断言、层取值                          |
| `labels.py`      | `_merge_split_widget.py`                                | `morph_labels`、`delete_labels`、`merge_labels`、`split_labels` 及全部辅助函数                                                         | widget 闭包、`viewer.dims`、跳转/查找标签           |
| `label_stats.py` | `_filter_small_labels.py`、`_label_counter_widget.py` | 小目标/边界过滤、`count_labels` 及 xlsx 导出、`parse_*`                                                                                    | 当前视图取层 (`_get_current_image`)                 |
| `metrics.py`     | `_accuracy_metrics.py`                                  | `compute_pixel_metrics`、`compute_instance_metrics`、`compute_instance_iou_dice` (逐字保留)                                              | 冒泡 widget                                           |
| `model_io.py`    | `_export_import_models.py`                              | `export_model`、`import_model`、`archive_model`                                                                                          | `viewer` 参数、模块级 `get_configs()` 副作用      |

已整体删除的纯 GUI 文件: `__init__.py` (插件入口)、`napari.yaml`、
`_open_docs.py`、`_register_model.py`、`_visualize_patches_from_points.py`。

## 与上游的语义差异

除去除 GUI 之外, 以下行为与原插件不同, 调用时需要知道:

1. **`n_contact` 之类的统计口径** — 本包未改动上游算法, 但 `pipeline.create_engine_3d`
   忠实保留了原 `get_engine` 的控制流: `multigpu=True` 时**第一次**调用只会构建
   `Engine3d`, 只有在配置未变、引擎已存在的后续调用中才会构建 `MultiGPUEngine3d`。
   这是上游既有的行为, 未做修正。
2. **dask 标签数组** — `labels.py` 的四个函数现在只接受稠密 numpy 数组。原 widget 会对
   dask 后端图层直接打印"不支持"并返回; 该守卫已随 GUI 一并删除。
3. **坐标参数** — `patches.pick_patches` 等函数接受的 `points` 必须已经是**数据坐标**。
   原 widget 用 `image_layer.world_to_data` / `layer.scale` 做的换算属于 napari 图层状态,
   没有移植。
4. **`put_patches_back_together`** — 与原实现一致, 要求 `save_directory` 必须已存在
   (被删掉的 GUI 包装层原本负责 `makedirs`)。

## 可选依赖

`label_stats.save_label_lists` 与 `label_stats.create_xlsx_from_label_queue_list`
需要 `openpyxl`, 该导入已下移到函数内部, 因此未安装 openpyxl 时本包仍可正常 import。

## 使用示例

```python
from empanada.config_loaders import load_config
from empanada_core.inference import Engine2d

config = load_config('empanada_core/configs/MitoNet_v1.yaml')
engine = Engine2d(config, use_gpu=True)
pan_seg = engine.infer(image)          # image: 2D numpy 数组
```

高层的引擎构建、ROI 裁剪与批量切片见 `empanada_core.pipeline`:

```python
from empanada_core.pipeline import create_engine_2d

engine = create_engine_2d(config, 'MitoNet_v1', confidence_thr=0.5, tile_size=512)
```

## 模型微调

微调的完整教程 (数据目录结构、标注规范、epochs 计算、常见坑) 见项目根目录的
[README.md 的「Empanada 模型微调教程」章节](../README.md#empanada-模型微调教程)。

最小用法:

```python
from empanada_core.finetune import build_finetune_config, finetune_and_register, get_model_info

get_model_info('MitoNet_v1')              # 打印该模型要求的标注规范

cfg = build_finetune_config(
    model_name='MitoNet_v1_ft',
    train_dir='data/processed/finetune/mito',
    eval_dir=None,
    model_dir='models',
    finetune_model='MitoNet_v1',
    finetune_layer='all',
    iterations=2000,
    patch_size=256,
    custom_config='default config',
)
cfg['MODEL']['model'] = 'models/MitoNet_v1.pth'   # 用本地权重, 跳过下载

finetune_and_register(cfg)                # 训练 + 注册到 ~/.empanada/configs/
```

从零训练 (多类 / 自建架构) 用 `empanada_core.train` 的
`build_train_config`、`run_training`、`train_and_register`。

