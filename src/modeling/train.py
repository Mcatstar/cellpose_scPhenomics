from abc import ABC, abstractmethod
from pathlib import Path

from cellpose import io, models, train, core
from cellpose.models import CellposeModel
from loguru import logger
from tqdm import tqdm
import typer

from src.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


class CellposeTrainer:
    """训练基类：加载数据 → 初始化预训练模型 → 微调"""

    def __init__(self, train_path: Path, test_path: Path, model_path: Path,
                 model_type: str = "cyto3", gpu: bool = True):
        self.train_path = train_path
        self.test_path = test_path
        self.model_path = model_path
        self.gpu = gpu and core.use_gpu()

        output = io.load_train_test_data(
            train_path, test_path,
            image_filter="_img", mask_filter="_masks",
        )
        self.images, self.labels, _, self.test_images, self.test_labels, _ = output
        self.model: CellposeModel = models.CellposeModel(gpu=self.gpu, model_type=model_type)

    def train(self, n_epochs=300, learning_rate=1e-5, batch_size=8,
              min_train_masks=1, normalize=False):
        self.model_path.mkdir(parents=True, exist_ok=True)
        result = train.train_seg(
            self.model.net,
            train_data=self.images,
            train_labels=self.labels,
            test_data=self.test_images,
            test_labels=self.test_labels,
            channels=[0, 0],
            normalize=normalize,
            save_path=str(self.model_path),
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            min_train_masks=min_train_masks,
            model_name=self.__class__.__name__.lower(),
        )
        logger.success(f"model saved at {result[0]}")
        return result


class MitoTrainer(CellposeTrainer):
    def __init__(self, gpu=True):
        super().__init__(
            train_path=PROCESSED_DATA_DIR / "train" / "mito",
            test_path=PROCESSED_DATA_DIR / "test" / "mito",
            model_path=MODELS_DIR,
            gpu=gpu,
        )


class LipidTrainer(CellposeTrainer):
    def __init__(self, gpu=True):
        super().__init__(
            train_path=PROCESSED_DATA_DIR / "train" / "lipid",
            test_path=PROCESSED_DATA_DIR / "test" / "lipid",
            model_path=MODELS_DIR,
            gpu=gpu,
        )


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    train_path: Path = PROCESSED_DATA_DIR / "train",
    test_path: Path = PROCESSED_DATA_DIR / "test",
    model_path: Path = MODELS_DIR,
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Training some model...")
    MitoTrainer().train(n_epochs=300, learning_rate=1e-5, batch_size=8)
    LipidTrainer().train(n_epochs=300, learning_rate=1e-5, batch_size=8)
    logger.success("Modeling training complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
