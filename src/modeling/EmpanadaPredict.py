"""EmpanadaPredict.py —— 基于 empanada 的 2D 推理脚本。

底层用 ``empanada_core.inference.Engine2d``: 缩放、padding、分块拼接与 panoptic
解码都由它处理, 不需要自己写预处理。

输出是**连续实例标号图** (1, 2, 3, ...) 的 uint16 TIFF, 可直接喂给
``src/postproc.py`` / ``src/features.py``。

用法::

    uv run python -m src.modeling.EmpanadaPredict --help
    uv run python -m src.modeling.EmpanadaPredict --tile-size 1024 --gpu False

模型配置通过 ``get_configs()`` 解析, 会依次查 ``empanada_core/configs/`` 与
``~/.empanada/configs/``, 后者同名时优先。基础模型权重优先用 ``models/`` 下的
本地文件, 缺失时才按配置里的地址下载。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import typer
from loguru import logger
from numpy.typing import NDArray
from tifffile import imread, imwrite
from tqdm import tqdm

from empanada.config_loaders import load_config
from empanada_core.inference import Engine2d
from empanada_core.utils import get_configs
from src.config import EXTERNAL_DATA_DIR, INTERIM_DATA_DIR, MODELS_DIR

app = typer.Typer()


def decode_instances(
    pan_seg: NDArray,
    thing_list: list[int],
    label_divisor: int,
) -> NDArray[np.int32]:
    """把 panoptic 标号图重编码为连续实例标号 (1, 2, 3, ...)。

    panoptic 的编码是 ``label = class_id * label_divisor + instance_id``, 例如
    label_divisor=1000, class_id=1 时第 2 个实例是 1002。这里按类把原标号线性
    映射到连续区间, 并保留类之间的先后顺序。
    """
    decoded = np.zeros(pan_seg.shape, dtype=np.int32)
    next_id = 1

    for class_id in thing_list:
        in_class = (pan_seg // label_divisor) == class_id
        values = np.unique(pan_seg[in_class])
        if values.size == 0:
            continue

        # 类内原标号 -> [next_id, next_id + values.size) 的查表
        lookup = np.zeros(int(values.max()) + 1, dtype=np.int32)
        lookup[values] = np.arange(next_id, next_id + values.size, dtype=np.int32)
        decoded[in_class] = lookup[pan_seg[in_class]]
        next_id += values.size

    return decoded


class EmpanadaPredictor:
    """加载 empanada 模型做 2D 推理, 输出连续实例标号图。"""

    name: str = "empanada"
    config_name: str = "MitoNet_v1"
    weights_file: str = "MitoNet_v1.pth"

    def __init__(
        self,
        gpu: bool = True,
        tile_size: int = 0,
        confidence_thr: float | None = None,
        inference_scale: int = 1,
    ) -> None:
        configs = get_configs()
        if self.config_name not in configs:
            logger.error(f"找不到模型配置 {self.config_name}, 可用: {sorted(configs)}")
            raise typer.Exit(code=1)

        cfg = load_config(configs[self.config_name])

        # 推理超参沿用模型自带的 engine_params, 只让命令行覆盖置信度
        engine_params = cfg["FINETUNE"]["engine_params"]
        self.label_divisor = int(engine_params["label_divisor"])
        self.thing_list = list(cfg["thing_list"])
        self.confidence_thr = (
            float(engine_params["confidence_thr"])
            if confidence_thr is None
            else float(confidence_thr)
        )

        # 优先用仓库内的本地权重, 缺失时回退到配置里的下载地址
        local_weights = MODELS_DIR / self.weights_file
        if local_weights.is_file():
            cfg["model"] = str(local_weights)
            logger.info(f"[{self.name}] 使用本地权重 {local_weights}")
        else:
            logger.warning(
                f"[{self.name}] 本地权重 {local_weights} 不存在, "
                f"将按配置下载 {cfg['model']}"
            )

        self.engine = Engine2d(
            cfg,
            inference_scale=inference_scale,
            label_divisor=self.label_divisor,
            nms_threshold=float(engine_params["nms_threshold"]),
            nms_kernel=int(engine_params["nms_kernel"]),
            confidence_thr=self.confidence_thr,
            tile_size=tile_size,
            use_gpu=gpu,
        )
        logger.info(
            f"[{self.name}] engine ready: tile_size={tile_size}, "
            f"confidence_thr={self.confidence_thr}, label_divisor={self.label_divisor}"
        )

    def predict(self, image: NDArray) -> NDArray[np.uint16]:
        """对单张 2D 图推理, 返回 uint16 的连续实例标号图。"""
        pan_seg = self.engine.infer(image)
        return decode_instances(pan_seg, self.thing_list, self.label_divisor).astype(
            np.uint16
        )


class MitoPredictor(EmpanadaPredictor):
    name = "mito"
    config_name = "MitoNet_v1"
    weights_file = "MitoNet_v1.pth"


class LipidPredictor(EmpanadaPredictor):
    name = "lipid"
    config_name = "DropNet_base_v1"
    weights_file = "DropNet_base_v1.pth"


@app.command()
def main(
    infer_path: Path = INTERIM_DATA_DIR,
    output_path: Path = EXTERNAL_DATA_DIR,
    pattern: str = "*_img.tif",
    tile_size: int = 0,
    confidence_thr: float | None = None,
    inference_scale: int = 1,
    gpu: bool = True,
) -> None:
    """对 infer_path 下所有匹配 pattern 的图跑 mito + lipid 两套模型。

    tile_size=0 表示整图一次推理; 图很大 (例如 2563x3296) 时设成 1024 或 512
    可以显著降低显存/内存峰值。
    """
    output_path.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(infer_path.glob(pattern))
    logger.info(f"Number of images found: {len(img_paths)}")
    if not img_paths:
        logger.error(f"{infer_path} 下没有匹配 {pattern} 的图, 退出")
        raise typer.Exit(code=1)

    mito_predictor = MitoPredictor(
        gpu=gpu,
        tile_size=tile_size,
        confidence_thr=confidence_thr,
        inference_scale=inference_scale,
    )
    lipid_predictor = LipidPredictor(
        gpu=gpu,
        tile_size=tile_size,
        confidence_thr=confidence_thr,
        inference_scale=inference_scale,
    )

    for img_path in tqdm(img_paths, desc="Inference"):
        img = imread(img_path)

        masks_mito = mito_predictor.predict(img)
        masks_lipid = lipid_predictor.predict(img)

        stem = img_path.stem.replace("_img", "")
        imwrite(output_path / f"{stem}_mito_masks_pred.tif", masks_mito)
        imwrite(output_path / f"{stem}_ld_masks_pred.tif", masks_lipid)
        logger.info(f"saved masks for {img_path.name}")

    logger.success("Inference complete.")


if __name__ == "__main__":
    app()
