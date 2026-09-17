from pathlib import Path

import numpy as np
import tifffile as tiff
from cellpose import io, models, core
from cellpose.models import CellposeModel
from loguru import logger
from tqdm import tqdm
import typer

from src.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


class CellposePredictor:
    """推理基类：从训练输出目录加载模型，按 self.channel 抽通道做 eval。"""

    channel: int = 0
    model_path: Path = MODELS_DIR

    def __init__(self, model_path: Path | None = None, gpu: bool = True):
        if model_path is not None:
            self.model_path = model_path
        self.gpu = gpu and core.use_gpu()

        weight_files = self.model_path
        if not weight_files:
            raise FileNotFoundError(
                f"No model weights found in {self.model_path}")
        self.model_path = weight_files
        self.model: CellposeModel = models.CellposeModel(
            gpu=self.gpu, pretrained_model=str(self.model_path), # pyright: ignore[reportArgumentType]
        )
        logger.info(f"loaded model from {self.model_path}")

    def predict(self, image: np.ndarray, normalize: bool = False,
                diameter: float | None = None, flow_threshold: float = 0.4,
                cellprob_threshold: float = 0.0):
        """image 为多通道 (C, H, W)，内部按 self.channel 取单通道。"""
        if image.ndim == 3:
            img_ch = image[self.channel]
        else:
            img_ch = image

        masks, flows, styles = self.model.eval(
            img_ch,
            channels=[0, 0],
            normalize=normalize,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
        )
        return masks, flows, styles


class MitoPredictor(CellposePredictor):
    channel = 0

    def __init__(self, model_path: Path = MODELS_DIR / "models" / "mito_model",
                 gpu: bool = True):
        super().__init__(model_path=model_path, gpu=gpu)


class LipidPredictor(CellposePredictor):
    channel = 1

    def __init__(self, model_path: Path = MODELS_DIR / "models" / "lipid_model",
                 gpu: bool = True):
        super().__init__(model_path=model_path, gpu=gpu)


@app.command()
def main(
    test_path: Path = PROCESSED_DATA_DIR / "test",
    output_path: Path = PROCESSED_DATA_DIR / "predictions",
    normalize: bool = False,
    diameter: float | None = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    gpu: bool = True,
):
    io.logger_setup()
    output_path.mkdir(parents=True, exist_ok=True)

    mito_predictor = MitoPredictor(gpu=gpu)
    lipid_predictor = LipidPredictor(gpu=gpu)

    img_paths = sorted(test_path.glob("*_img.tif"))
    logger.info(f"Number of images found: {len(img_paths)}")

    for img_path in tqdm(img_paths, desc="Inference"):
        img = tiff.imread(img_path)  # (C, H, W)

        masks_mito, _, _ = mito_predictor.predict(
            img, normalize=normalize, diameter=None, # 建议手动测量目标的像素宽度
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
        )
        masks_lipid, _, _ = lipid_predictor.predict(
            img, normalize=normalize, diameter=None, # 建议手动测量目标的像素宽度
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
        )

        # 合并保存：第 0 层 mito，第 1 层 lipid
        out = np.stack([masks_mito, masks_lipid], axis=0).astype(np.uint16)
        out_path = output_path / img_path.name.replace(
            "_img.tif", "_masks_pred.tif")
        tiff.imwrite(out_path, out)
        logger.info(f"saved {out_path}")

    logger.success("Inference complete.")


if __name__ == "__main__":
    app()