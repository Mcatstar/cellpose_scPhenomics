from pathlib import Path

from cellpose import models, core
from cellpose.models import CellposeModel
from loguru import logger
from tqdm import tqdm
import typer

from src.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


class CellposePredictor:
    """推理基类：加载微调模型 → eval"""

    def __init__(self, model_path: Path, gpu: bool = True):
        self.gpu = gpu and core.use_gpu()
        self.model: CellposeModel = models.CellposeModel(
            gpu=self.gpu, pretrained_model=str(model_path), # pyright: ignore[reportArgumentType]
        )
        logger.info(f"loaded model from {model_path}")

    def predict(self, images, batch_size=16, normalize=False, diameter=None,
                flow_threshold=0.4, cellprob_threshold=0.0):
        masks, flows, styles = self.model.eval(
            images,
            channels=[0, 0],
            normalize=normalize,
            batch_size=batch_size,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
        )
        return masks, flows, styles


class MitoPredictor(CellposePredictor):
    def __init__(self, gpu=True):
        super().__init__(MODELS_DIR / "mitotrainer", gpu)


class LipidPredictor(CellposePredictor):
    def __init__(self, gpu=True):
        super().__init__(MODELS_DIR / "lipidtrainer", gpu)


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Performing inference for model...")
    tmp_list: list[Path] = list(predictions_path.iterdir())
    img_list: list[Path] = []
    for tmp in tmp_list:
        if re.search('.tif', str(tmp)):
            img_list.append(tmp) # img_list have full path, not name, e.g. "/path/to/img.tif"
            
    logger.info("images found:\n"+"\n".join(map(lambda x: str(x.name), img_list)))
    logger.info(f"Number of images found: {len(img_list)}")
    # img_index = int(input('Select image: '))
    for img_index in tqdm(range(len(img_list)), total=len(img_list), desc="Dataset Generation"):
        
        masks_mito, _, _ = MitoPredictor().predict(imgs)
        masks_lipid, _, _ = LipidPredictor().predict(imgs)
    logger.success("Inference complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
