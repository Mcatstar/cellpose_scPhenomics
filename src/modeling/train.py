from pathlib import Path

import numpy as np
import tifffile as tiff
from cellpose import io, models, train, core
from cellpose.models import CellposeModel
from loguru import logger
import typer

from src.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


class CellposeTrainer:
    """微调基类：加载单通道灰度图 + 对应掩码 → 微调预训练模型"""

    # 子类覆盖：掩码文件的后缀，用于从 *_img.tif 推回掩码路径
    mask_suffix: str = "_masks.tif"
    model_name: str = "cellpose_model"
    model_type: str = "cyto3"

    def __init__(self, train_path: Path, test_path: Path, model_path: Path,
                 gpu: bool = True):
        self.train_path = train_path
        self.test_path = test_path
        self.model_path = model_path
        self.gpu = gpu and core.use_gpu()

        self.train_images, self.train_labels = self._load(train_path)
        self.test_images, self.test_labels = self._load(test_path)
        self.model: CellposeModel = models.CellposeModel(
            gpu=self.gpu, model_type=self.model_type,
        )

    def _load(self, data_path: Path):
        """读取 *_img.tif 和对应的掩码文件（由 self.mask_suffix 指定）"""
        img_paths = sorted(data_path.glob("*_img.tif"))
        images, labels = [], []
        for img_path in img_paths:
            mask_path = img_path.with_name(
                img_path.name.replace("_img.tif", self.mask_suffix))
            if not mask_path.exists():
                raise FileNotFoundError(f"掩码未找到: {mask_path}")

            images.append(np.asarray(tiff.imread(img_path)))
            labels.append(np.asarray(tiff.imread(mask_path)).astype(np.uint16))

        logger.info(
            f"[{self.model_name}] {data_path.name}: {len(images)} samples")
        return images, labels

    def train(self, n_epochs: int = 300, learning_rate: float = 1e-5,
              batch_size: int = 8, min_train_masks: int = 1,
              normalize: bool = False):
        self.model_path.mkdir(parents=True, exist_ok=True)
        model_path = train.train_seg(
            self.model.net,
            train_data=self.train_images,
            train_labels=self.train_labels,
            test_data=self.test_images,
            test_labels=self.test_labels,
            channels=[0, 0],
            normalize=normalize,
            save_path=str(self.model_path),
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            min_train_masks=min_train_masks,
            model_name=self.model_name,
        )
        logger.success(f"{self.model_name} saved at {model_path}")
        return model_path


class MitoTrainer(CellposeTrainer):
    mask_suffix = "_mito_masks.tif"
    model_name = "mito_model"

    def __init__(self, train_path: Path = PROCESSED_DATA_DIR / "train",
                 test_path: Path = PROCESSED_DATA_DIR / "test",
                 model_path: Path = MODELS_DIR,
                 gpu: bool = True):
        super().__init__(train_path, test_path, model_path, gpu)


class LipidTrainer(CellposeTrainer):
    mask_suffix = "_ld_masks.tif"
    model_name = "lipid_model"

    def __init__(self, train_path: Path = PROCESSED_DATA_DIR / "train",
                 test_path: Path = PROCESSED_DATA_DIR / "test",
                 model_path: Path = MODELS_DIR,
                 gpu: bool = True):
        super().__init__(train_path, test_path, model_path, gpu)


@app.command()
def main(
    train_path: Path = PROCESSED_DATA_DIR / "train",
    test_path: Path = PROCESSED_DATA_DIR / "test",
    n_epochs: int = 5,          # 实际使用时改为 300
    learning_rate: float = 1e-5,
    batch_size: int = 2,        # 实际使用时改为 8
    normalize: bool = False,
    gpu: bool = True,
):
    io.logger_setup()

    logger.info("Training mito model ...")
    MitoTrainer(train_path=train_path, test_path=test_path, gpu=gpu).train(
        n_epochs=n_epochs, learning_rate=learning_rate,
        batch_size=batch_size, normalize=normalize,
    )

    logger.info("Training lipid model ...")
    LipidTrainer(train_path=train_path, test_path=test_path, gpu=gpu).train(
        n_epochs=n_epochs, learning_rate=learning_rate,
        batch_size=batch_size, normalize=normalize,
    )

    logger.success("Modeling training complete.")


if __name__ == "__main__":
    app()