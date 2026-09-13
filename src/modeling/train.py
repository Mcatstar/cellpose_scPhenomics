from pathlib import Path

from cellpose import io, models, train, core
from loguru import logger
from tqdm import tqdm
import typer

from src.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


class CellposeModelTrainer():
    def __init__(self, train_path: Path, test_path: Path, model_path: Path, gpu: bool = True, model_type: str = 'cyto'):
        self.train_path = train_path
        self.test_path = test_path
        self.model_path = model_path
        self.gpu = core.use_gpu()
        output = io.load_train_test_data(
            self.train_path, self.test_path, 
            image_filter="_img", 
            mask_filter="_masks"
        )
        self.images, self.labels, _, self.test_images, self.test_labels, _ = output
        self.model = models.CellposeModel(gpu=self.gpu, model_type='cyto')

    def train(self):
        # Implement training logic here
        logger.info(f"Training model with data from {self.train_path} and {self.test_path}")
        model_path, train_losses, test_losses = train.train_seg(
            self.model.net,                  # 从预训练模型初始化的网络
            train_data=self.images,
            train_labels=self.labels,
            test_data=self.test_images,
            test_labels=self.test_labels,
            channels=[0, 0],            # 根据你的图像通道调整，如 [0,0] 代表灰度
            save_path=str(self.model_path), # 保存路径
            n_epochs=300,               # 训练轮数，可按需调整
            learning_rate=1e-5,         # 微调时使用较小的学习率
            batch_size=8,               # 根据 GPU 显存调整
            min_train_masks=5           # 图像至少包含 5 个标记才用于训练
        )
        logger.success("Model training complete.")


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
    real_diameter = 18

    logger.success("Modeling training complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
