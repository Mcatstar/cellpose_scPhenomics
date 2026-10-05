"""EmpanadaFinetune.py —— 基于 empanada 微调线粒体 / 脂滴模型。

两个子命令:

``prepare``
    把 ``src/dataset.py`` 生成的 ``data/processed/<split>/<source>/`` 下成对数据,
    转成 empanada 数据集要求的 ``<dest>/<source>/{images,masks}/*.tiff``。
    ``--split train`` 落到 ``--train-dir``, ``--split test`` 落到 ``--eval-dir``。

``finetune``
    组装配置并微调。加 ``--check-only`` 只做校验与计划打印, **不会加载任何模型**,
    适合在跑不动模型的机器上先确认配置无误。

完整的数据规范、epochs 计算与常见坑见项目根目录 README 的
「Empanada 模型微调教程」章节。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import typer
from loguru import logger
from numpy.typing import NDArray
from tifffile import imread

from empanada_core.export import export_batch_segs
from empanada_core.finetune import (
    build_finetune_config,
    finetune_and_register,
    get_model_info,
    run_finetuning,
)
from empanada_core.utils import get_configs
from src.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@dataclass(frozen=True, slots=True)
class FinetunePlan:
    """不加载模型就能算出来的微调计划, 用于 --check-only 打印。"""

    model_name: str
    n_datasets: int
    n_images: int
    batch_size: int
    epochs: int
    save_freq: int
    dataset_class: str
    base_weights: str


def count_dataset_images(train_dir: Path) -> tuple[int, int]:
    """返回 empanada 数据集里的 (来源数据集个数, 图片总数)。

    empanada 要求 ``<train_dir>/<source>/images/*``, 每个子目录算一个来源。
    """
    if not train_dir.is_dir():
        return 0, 0

    sources = [p for p in train_dir.iterdir() if (p / "images").is_dir()]
    n_images = sum(len(list((p / "images").glob("*"))) for p in sources)
    return len(sources), n_images


def to_uint8(image: NDArray) -> NDArray[np.uint8]:
    """把任意位深的图线性映射到 uint8。

    empanada 的数据集用 ``cv2.imread(path, 0)`` 读图, 结果一定是 8 位; 先自己按
    确定的规则转换, 可以保证缩放可复现, 不依赖 OpenCV 的隐式行为。
    """
    if image.dtype == np.uint8:
        return image

    if np.issubdtype(image.dtype, np.integer):
        peak = float(np.iinfo(image.dtype).max)
        return np.rint(image.astype(np.float32) * (255.0 / peak)).astype(np.uint8)

    top = float(image.max())
    scale = 255.0 if top <= 1.0 else 255.0 / top
    return np.clip(np.rint(image.astype(np.float32) * scale), 0.0, 255.0).astype(
        np.uint8
    )


class EmpanadaFinetuner:
    """用 empanada 微调一个已注册的模型。

    子类只需覆盖下面几个类属性, 就能换成别的目标。
    """

    source: str = "mito"
    base_model: str = "MitoNet_v1"
    weights_file: str = "MitoNet_v1.pth"
    image_pattern: str = "*_img.tif"
    mask_suffix: str = "_mito_masks.tif"

    def __init__(
        self,
        train_dir: Path,
        model_dir: Path,
        model_name: str | None = None,
        finetune_layer: str = "all",
        iterations: int = 2000,
        patch_size: int = 512,
        eval_dir: Path | None = None,
        custom_config: str = "default config",
        batch_size: int | None = None,
    ) -> None:
        self.train_dir = Path(train_dir)
        self.model_dir = Path(model_dir)
        self.model_name = model_name or f"{self.base_model}_ft"
        self.finetune_layer = finetune_layer
        self.iterations = iterations
        self.patch_size = patch_size
        self.eval_dir = None if eval_dir is None else Path(eval_dir)
        self.custom_config = custom_config
        self.batch_size = batch_size

        # check() 组装好的配置, run() 直接复用
        self.config: dict[str, Any] | None = None

    def prepare(self, split: str = "train") -> int:
        """把 ``data/processed/<split>/<source>/`` 下的 img/mask 转成 empanada 结构。

        源目录由 ``src/dataset.py`` 生成, 形如
        ``data/processed/train/mito/<name>_img.tif`` + ``<name>_mito_masks.tif``。
        结果写到 ``<dest>/<source>/{images,masks}/*.tiff``, 其中 ``split="train"``
        用 ``self.train_dir``、``split="test"`` 用 ``self.eval_dir``。

        返回成功写入的样本对数。这一步不接触模型。
        """
        raw_dir = PROCESSED_DATA_DIR / split / self.source
        dest = self.train_dir if split == "train" else self.eval_dir
        if dest is None:
            logger.error(f"split={split} 需要 eval_dir 作为输出目录")
            raise typer.Exit(code=1)
        if not raw_dir.is_dir():
            logger.error(f"{raw_dir} 不存在, 先跑 src/dataset.py 生成 processed 数据")
            raise typer.Exit(code=1)

        img_paths = sorted(raw_dir.glob(self.image_pattern))
        logger.info(f"在 {raw_dir} 找到 {len(img_paths)} 张 {self.image_pattern}")
        if not img_paths:
            logger.error("没有找到任何图, 退出")
            raise typer.Exit(code=1)

        written = 0
        for img_path in img_paths:
            base = img_path.stem.removesuffix("_img")
            mask_path = raw_dir / f"{base}{self.mask_suffix}"
            if not mask_path.is_file():
                logger.warning(f"缺少对应的掩码, 跳过: {mask_path.name}")
                continue

            image = to_uint8(imread(img_path))
            mask = imread(mask_path)
            if mask.shape != image.shape:
                logger.warning(
                    f"图与掩码尺寸不一致, 跳过: {img_path.name} {image.shape} vs "
                    f"{mask_path.name} {mask.shape}"
                )
                continue

            # 二值图会被整张当成一个实例
            n_positive = np.unique(mask[mask > 0]).size
            if n_positive <= 1:
                logger.warning(
                    f"{mask_path.name} 只有 {n_positive} 个非零标号, 看起来是二值图; "
                    f"empanada 需要每个实例一个独立整数 ID"
                )

            export_batch_segs(
                image=image,
                mask=mask,
                image_name=base,
                export_type="2D images",
                dataset_name=self.source,
                save_dir=str(dest),
                grayscale=True,
            )
            written += 1

        if not written:
            logger.error("没有任何样本被写入, 退出")
            raise typer.Exit(code=1)

        logger.success(f"写入 {written} 对样本到 {dest / self.source}")
        return written

    def check(self) -> FinetunePlan:
        """组装配置并做训练前的全部校验; **不加载任何模型**。"""
        if self.base_model not in get_configs():
            logger.error(
                f"基础模型 {self.base_model} 不在 get_configs() 里, 可用: "
                f"{sorted(get_configs())}"
            )
            raise typer.Exit(code=1)

        n_datasets, n_images = count_dataset_images(self.train_dir)
        if n_images == 0:
            logger.error(
                f"{self.train_dir} 下没有 <source>/images/*, 先跑 prepare 子命令, "
                f"或参考 README 的微调教程"
            )
            raise typer.Exit(code=1)
        logger.info(f"数据集: {n_datasets} 个来源, {n_images} 张图")

        get_model_info(self.base_model)

        config = build_finetune_config(
            model_name=self.model_name,
            train_dir=str(self.train_dir),
            eval_dir=None if self.eval_dir is None else str(self.eval_dir),
            model_dir=str(self.model_dir),
            finetune_model=self.base_model,
            finetune_layer=self.finetune_layer,
            iterations=self.iterations,
            patch_size=self.patch_size,
            custom_config=self.custom_config,
            batch_size=self.batch_size,
        )

        # 有本地权重就用本地, 避免训练时从 zenodo 下载
        local_weights = MODELS_DIR / self.weights_file
        if local_weights.is_file():
            config["MODEL"]["model"] = str(local_weights)

        schedule = config["TRAIN"]["schedule_params"]
        epochs = (
            int(schedule["epochs"])
            if "epochs" in schedule
            else int(config["TRAIN"]["epochs"])
        )
        plan = FinetunePlan(
            model_name=self.model_name,
            n_datasets=n_datasets,
            n_images=n_images,
            batch_size=int(config["TRAIN"]["batch_size"]),
            epochs=epochs,
            save_freq=int(config["TRAIN"]["save_freq"]),
            dataset_class=str(config["FINETUNE"]["dataset_class"]),
            base_weights=str(config["MODEL"]["model"]),
        )

        # 校验通过后再落盘, 避免 check() 失败时留下半成品状态
        self.config = config

        logger.info(f"dataset_class = {plan.dataset_class}")
        logger.info(f"基础权重      = {plan.base_weights}")
        logger.info(
            f"batch_size {plan.batch_size} -> {plan.epochs} epochs, "
            f"每 {plan.save_freq} epoch 存一次"
        )

        if plan.epochs < 5:
            logger.error(
                f"epochs={plan.epochs} 太小, save_freq 会算成 0 并在训练时抛 "
                f"ZeroDivisionError。请把 iterations 提到至少 "
                f"{5 * (plan.n_images // plan.batch_size)}"
            )
            raise typer.Exit(code=1)

        return plan

    def launch(self, register: bool = False) -> Path:
        """开始训练并导出模型; **会加载模型**。未 check() 时自动先校验。"""
        if self.config is None:
            self.check()
        config = self.config
        assert config is not None, "check() 未能组装出配置"

        self.model_dir.mkdir(parents=True, exist_ok=True)

        yaml_path = (
            finetune_and_register(config) if register else run_finetuning(config)
        )
        logger.info(f"推理时把模型配置指向 {yaml_path} 即可")
        return Path(yaml_path)


class MitoFinetuner(EmpanadaFinetuner):
    source = "mito"
    base_model = "MitoNet_v1"
    weights_file = "MitoNet_v1.pth"
    mask_suffix = "_mito_masks.tif"


class LipidFinetuner(EmpanadaFinetuner):
    source = "ld"
    base_model = "DropNet_base_v1"
    weights_file = "DropNet_base_v1.pth"
    mask_suffix = "_ld_masks.tif"


FINETUNERS: dict[str, type[EmpanadaFinetuner]] = {
    "mito": MitoFinetuner,
    "lipid": LipidFinetuner,
}


def lookup(target: str) -> type[EmpanadaFinetuner]:
    """按名字取微调器类, 未知名直接退出并列出可选项。"""
    if target not in FINETUNERS:
        logger.error(f"未知的 target: {target}, 可选 {sorted(FINETUNERS)}")
        raise typer.Exit(code=1)
    return FINETUNERS[target]


@app.command()
def prepare(
    target: str = "mito",
    split: str = "train",
    train_dir: Path = PROCESSED_DATA_DIR / "finetune",
    eval_dir: Path = PROCESSED_DATA_DIR / "eval",
) -> None:
    """把 processed/<split>/<source>/ 的成对 img/mask 转成 empanada 数据集结构。

    split=train 落到 --train-dir, split=test 落到 --eval-dir, 两个目录都能直接
    交给 finetune 子命令。
    """
    if split not in ("train", "test"):
        logger.error(f"split 只能是 train 或 test, 收到 {split}")
        raise typer.Exit(code=1)

    finetuner = lookup(target)(
        train_dir=train_dir,
        model_dir=MODELS_DIR,
        eval_dir=eval_dir,
    )
    finetuner.prepare(split)

    flag = "--train-dir" if split == "train" else "--eval-dir"
    dest = train_dir if split == "train" else eval_dir
    logger.success(f"finetune: {flag} -> {dest}")


@app.command()
def finetune(
    target: str = "mito",
    train_dir: Path = PROCESSED_DATA_DIR / "finetune",
    model_dir: Path = MODELS_DIR,
    model_name: str = "",
    finetune_layer: str = "all",
    iterations: int = 2000,
    patch_size: int = 256,
    eval_dir: Path | None = None,
    custom_config: str = "default config",
    batch_size: int | None = None,
    register: bool = False,
    check_only: bool = False,
) -> None:
    """微调一个已注册的 empanada 模型; --check-only 不加载模型。

    设备由 empanada 自己选 (有 CUDA 就用 cuda:0, 否则 cpu), 这里没有开关。
    标注图不足模板默认的 16 张时, 用 --batch-size 调小。
    """
    finetuner = lookup(target)(
        train_dir=train_dir,
        model_dir=model_dir,
        model_name=model_name or None,
        finetune_layer=finetune_layer,
        iterations=iterations,
        patch_size=patch_size,
        eval_dir=eval_dir,
        custom_config=custom_config,
        batch_size=batch_size,
    )

    finetuner.check()

    if check_only:
        logger.success("校验通过 (未加载模型, 未开始训练)")
        return

    finetuner.launch(register=register)
    logger.success(f"微调完成: {finetuner.model_name}")


if __name__ == "__main__":
    app()
