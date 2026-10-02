from pathlib import Path

import numpy as np
import typer
from cellpose import io, models, core
from cellpose.models import CellposeModel
from loguru import logger
from tqdm import tqdm

from src.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


class CellposePredictor:
    """加载微调模型做推理; 权重缺失时回退到 Cellpose 自带模型。"""

    name: str = "cellpose"

    def __init__(self, model_dir: Path, gpu: bool = True):
        self.gpu = gpu and core.use_gpu()
        self.model: CellposeModel = models.CellposeModel(
            gpu=self.gpu, pretrained_model=str(model_dir), # pyright: ignore[reportArgumentType]
        )
        logger.info(f"[{self.name}] loaded fine-tuned model from {str(model_dir)}")

    def predict(self, image: np.ndarray, normalize: bool = True,
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
    infer_path: Path = PROCESSED_DATA_DIR / "infer",
    output_path: Path = PROCESSED_DATA_DIR / "predictions",
    normalize: bool = True,
    diameter: float | None = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    gpu: bool = True,
):
    io.logger_setup()

    mito_predictor = MitoPredictor(gpu=gpu)
    lipid_predictor = LipidPredictor(gpu=gpu)

    img_paths = sorted(infer_path.glob("*_img.tif"))
    logger.info(f"Number of images found: {len(img_paths)}")

    for img_path in tqdm(img_paths, desc="Inference"):
        img = io.imread(img_path)

        masks_mito, _, _ = mito_predictor.predict(
            img, normalize=normalize, diameter=diameter, # pyright: ignore[reportArgumentType]
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
        )
        masks_lipid, _, _ = lipid_predictor.predict(
            img, normalize=normalize, diameter=diameter, # pyright: ignore[reportArgumentType]
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
        )

        io.imsave(output_path / f"{img_path.stem}_mito_masks_pred.tif",
                  np.asarray(masks_mito).astype(np.uint16))
        io.imsave(output_path / f"{img_path.stem}_ld_masks_pred.tif",
                  np.asarray(masks_lipid).astype(np.uint16))

        logger.info(f"saved masks_pred.tif successfully")

    logger.success("Inference complete.")


if __name__ == "__main__":
    app()