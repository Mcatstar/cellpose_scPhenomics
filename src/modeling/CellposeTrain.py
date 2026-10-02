from pathlib import Path

import typer
from cellpose import io, models, train, core
from loguru import logger

from src.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


class CellposeTrainer:
    """微调基类：用 cellpose.io.load_train_test_data 加载数据 → 微调"""
    mask_suffix: str = "_masks"
    model_name: str = "cellpose_model"
    model_type: str = "cyto3"

    def __init__(self, train_path: Path, test_path: Path,
                 model_path: Path, gpu: bool = True):
        self.model_path = model_path
        self.gpu = gpu and core.use_gpu()

        (
            self.train_images, self.train_labels, _,
            self.test_images, self.test_labels, _,
        )  = io.load_train_test_data(
            str(train_path), str(test_path),
            image_filter="_img", mask_filter=self.mask_suffix,
        )

        logger.info(
            f"[{self.model_name}] train={len(self.train_images)}, test={len(self.test_images)}" # type: ignore
        )
        self.model = models.CellposeModel(
            gpu=self.gpu, model_type=self.model_type,
        )

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
    mask_suffix = "_mito_masks"
    model_name = "mito_model"


class LipidTrainer(CellposeTrainer):
    mask_suffix = "_ld_masks"
    model_name = "lipid_model"


@app.command()
def main(
    train_path: Path = PROCESSED_DATA_DIR / "train",
    test_path: Path = PROCESSED_DATA_DIR / "test",
    n_epochs: int = 5,          # 实际使用时改为 300
    learning_rate: float = 1e-5,
    batch_size: int = 2,        # 实际使用时改为 8
    normalize: bool = False,
    gpu: bool = False,
):
    io.logger_setup()

    logger.info("Training mito model ...")
    MitoTrainer(train_path, test_path, MODELS_DIR, gpu).train(
        n_epochs=n_epochs, learning_rate=learning_rate,
        batch_size=batch_size, normalize=normalize,
    )

    logger.info("Training lipid model ...")
    LipidTrainer(train_path, test_path, MODELS_DIR, gpu).train(
        n_epochs=n_epochs, learning_rate=learning_rate,
        batch_size=batch_size, normalize=normalize,
    )

    logger.success("Modeling training complete.")


if __name__ == "__main__":
    app()