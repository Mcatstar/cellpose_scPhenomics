"""empanada_core —— empanada-napari 的非 GUI 核心逻辑。

本包由 https://github.com/volume-em/empanada-napari 的 empanada_napari 改造而来,
移除了全部 napari / magicgui / qtpy 依赖与 GUI 线程装饰器, 只保留可复用的计算逻辑。

原插件的 GUI 部件 (_slice_inference / _volume_inference / _merge_split_widget 等)
已删除, 其中与界面无关的算法被提取到下列模块:

    inference   Engine2d / Engine3d, 2D 与 3D 推理引擎
    multigpu    MultiGPUEngine3d, 多卡 3D 推理
    pipeline    由 GUI 部件提取的推理驱动层: 引擎构建、ROI 裁剪、批量切片、正交三轴
    utils       Preprocessor、模型下载与注册、配置发现
    train       训练: 配置组装、训练循环、TorchScript 导出
    finetune    微调: 配置组装、微调循环、模型信息打印
    patches     训练/微调补丁与 3D flipbook 的抓取与落盘
    tiles       大图分块、合并
    export      批量分割结果导出
    labels      标签形态学、合并、分水岭拆分
    label_stats 小目标/边界标签过滤、标签计数与 xlsx 导出
    metrics     像素级与实例级 IoU / Dice / Hungarian 匹配指标
    model_io    .empanada 模型包导出、导入、归档

子模块不会被本 __init__ 预先导入, 按需 from empanada_core.inference import ... 即可,
这样只依赖轻量库的场景不必加载 torch。
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

import platform

import torch
import torch.multiprocessing as mp

# Fix macOS (Darwin) child processes (e.g. the 3D inference matcher process,
# or DataLoader/training workers) forking a process that already has Cocoa /
# CoreFoundation loaded. A bare fork() in that state is unsafe and can silently
# hang the child. The 'spawn' start method avoids this by using fork+exec.
#
# This must run as early as possible (before anything implicitly creates a
# multiprocessing object, e.g. via a joblib/dask backend), since Python's
# multiprocessing context can only be set once per process. `force=True`
# guarantees 'spawn' wins even if a default context was already set.
if platform.system() == "Darwin":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

if torch.backends.quantized.engine in (None or "none"):
    if "qnnpack" in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = "qnnpack"
