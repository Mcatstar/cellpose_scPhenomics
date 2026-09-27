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


def find_weights(model_dir: Path) -> Path:
    """在训练输出目录（可能嵌套 models/ 子目录）里找到权重文件"""
    files = [p for p in model_dir.rglob("*") if p.is_file()]
    if not files:
        raise FileNotFoundError(f"未在 {model_dir} 下找到权重文件")
    return sorted(files)[-1]


class CellposePredictor:
    """从训练输出目录加载模型，对单通道灰度图做推理。"""

    name: str = "cellpose"

    def __init__(self, model_dir: Path, gpu: bool = True):
        self.model_path = find_weights(model_dir)
        self.gpu = gpu and core.use_gpu()
        self.model: CellposeModel = models.CellposeModel(
            gpu=self.gpu, pretrained_model=str(self.model_path),
        )
        logger.info(f"[{self.name}] loaded model from {self.model_path}")

    def predict(self, image: np.ndarray, normalize: bool = False,
                diameter: float | None = None,
                flow_threshold: float = 0.4,
                cellprob_threshold: float = 0.0):
        masks, flows, styles = self.model.eval(
            image,
            channels=[0, 0],
            normalize=normalize,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
        )
        return masks, flows, styles


class MitoPredictor(CellposePredictor):
    name = "mito"

    def __init__(self, model_dir: Path = MODELS_DIR / "models" / "mito_model",
                 gpu: bool = True):
        super().__init__(model_dir, gpu)


class LipidPredictor(CellposePredictor):
    name = "lipid"

    def __init__(self, model_dir: Path = MODELS_DIR / "models" / "lipid_model",
                 gpu: bool = True):
        super().__init__(model_dir, gpu)


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
        img = tiff.imread(img_path)          # (H, W) 单通道灰度图

        masks_mito, _, _ = mito_predictor.predict(
            img, normalize=normalize, diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
        )
        masks_lipid, _, _ = lipid_predictor.predict(
            img, normalize=normalize, diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
        )

        # 合并保存：第 0 层 mito，第 1 层 lipid（方便下游 postproc 处理）
        out = np.stack([masks_mito, masks_lipid], axis=0).astype(np.uint16)
        out_path = output_path / img_path.name.replace(
            "_img.tif", "_masks_pred.tif")
        tiff.imwrite(out_path, out)
        logger.info(f"saved {out_path}")

    logger.success("Inference complete.")


if __name__ == "__main__":
    app()